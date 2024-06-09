import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from Transition import Transition
from replay_memory import ReplayMemory
from kan import KAN


class RiiswaKANDQN:
    def __init__(
            self,
            width,
            grid,
            n_observations,
            n_actions,
            batch_size,
            gamma,
            lr,
            replay_memory_capacity,
            target_update_freq,
            episode_train_steps,
            device,
            warm_up_episodes,
    ):
        self.policy_net = KAN(
            width=[n_observations, width, n_actions],
            grid=grid,
            k=3,
            bias_trainable=False,
            sp_trainable=False,
            sb_trainable=False,
            # device=device,
        )
        self.target_net = KAN(
            width=[n_observations, width, n_actions],
            grid=grid,
            k=3,
            bias_trainable=False,
            sp_trainable=False,
            sb_trainable=False,
            # device=device,
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(replay_memory_capacity)
        self.steps_done = 0

        self.batch_size = batch_size
        self.gamma = gamma

        self.target_update_freq = target_update_freq
        self.episode_train_steps = episode_train_steps
        self.device = device
        self.warm_up_episodes = warm_up_episodes

    def select_action(self, state, env, episode_i):
        if episode_i < self.warm_up_episodes:
            return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
        with torch.no_grad():
            return self.policy_net(state).max(1).indices.view(1, 1)

    def store_transition(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def optimize_model(self):
        assert len(self.memory) >= self.batch_size, 'Not enough memory for sampling'

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if len(non_final_next_states_list) != 0:
            non_final_next_states = torch.cat(non_final_next_states_list)
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        else:
            expected_state_action_values = reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(
            self,
            env,
            num_episodes,
            device,
            seed,
            wandbrun=None,
    ):
        for i_episode in tqdm(range(num_episodes)):
            state, info = env.reset(seed=seed)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            done = False

            episode_reward = 0
            while not done:
                action = self.select_action(state, env, i_episode)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                self.store_transition(state, action, next_state, reward)

                state = next_state

                if done:
                    break

            loss = 0
            if len(self.memory) >= self.batch_size:
                for _ in range(self.episode_train_steps):
                    loss = self.optimize_model()

                if i_episode % 25 == 0 and i_episode < int(num_episodes * (1 / 2)):
                    all_transitions = self.memory.dump_memory()
                    all_transitions_T = Transition(*zip(*all_transitions))
                    all_states = torch.cat(all_transitions_T.state)
                    self.policy_net.update_grid_from_samples(all_states)
                    self.target_net.update_grid_from_samples(all_states)

                if i_episode % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            print("episode: {}, the episode reward is {}".format(i_episode, episode_reward))
            if wandbrun is not None:
                wandbrun.log({'loss': loss, 'reward': episode_reward})
