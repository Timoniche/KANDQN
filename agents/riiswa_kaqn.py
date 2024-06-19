import torch
import torch.nn as nn
# noinspection PyPackageRequirements
from kan import KAN
import numpy as np
from tqdm import tqdm


# noinspection DuplicatedCode
class ReplayBuffer:
    def __init__(self, capacity, observation_dim):
        self.capacity = capacity
        self.observations = torch.zeros(capacity, observation_dim)
        self.actions = torch.zeros(capacity, 1, dtype=torch.int64)
        self.next_observations = torch.zeros(capacity, observation_dim)
        self.rewards = torch.zeros(capacity, 1)
        self.terminations = torch.zeros(capacity, 1, dtype=torch.int)
        self.cursor = 0

    def add(self, observation, action, next_observation, reward, termination):
        index = self.cursor % self.capacity

        self.observations[index] = observation
        self.actions[index] = action
        self.next_observations[index] = next_observation
        self.rewards[index] = reward
        self.terminations[index] = termination

        self.cursor += 1

    def sample(self, batch_size):
        idx = np.random.permutation(np.arange(len(self)))[:batch_size]
        return (
            self.observations[idx],
            self.actions[idx],
            self.next_observations[idx],
            self.rewards[idx],
            self.terminations[idx],
        )

    def __len__(self):
        return min(self.cursor, self.capacity)


# noinspection DuplicatedCode
class RiiswaKAQN:
    # noinspection PyUnusedLocal
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
            warm_up_episodes,
            device,
    ):
        if isinstance(width, int):
            self.width = [width]
        else:
            self.width = [int(x) for x in width.split(',')]
        kan_layers = [n_observations] + self.width + [n_actions]
        self.q_network = KAN(
            width=kan_layers,
            grid=grid,
            k=3,
            bias_trainable=False,
            sp_trainable=False,
            sb_trainable=False,
        )
        self.target_network = KAN(
            width=kan_layers,
            grid=grid,
            k=3,
            bias_trainable=False,
            sp_trainable=False,
            sb_trainable=False,
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr)
        self.buffer = ReplayBuffer(replay_memory_capacity, n_observations)
        self.warm_up_episodes = warm_up_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.episode_train_steps = episode_train_steps

    def optimize_model(self):
        observations, actions, next_observations, rewards, terminations = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_q_values = self.q_network(next_observations)
            next_actions = next_q_values.argmax(dim=1)
            next_q_values_target = self.target_network(next_observations)
            target_max = next_q_values_target[range(len(next_q_values)), next_actions]
            td_target = rewards.flatten() + self.gamma * target_max * (
                    1 - terminations.flatten()
            )

        old_val = self.q_network(observations).gather(1, actions).squeeze()
        loss = nn.functional.mse_loss(td_target, old_val)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(
            self,
            env,
            num_episodes,
            seed,
            only_terminal_negative_reward,
            wandbrun=None,
    ):
        for episode in tqdm(range(num_episodes)):
            observation, info = env.reset(seed=seed)
            observation = torch.from_numpy(observation)
            finished = False
            episode_length = 0
            while not finished:
                if episode < self.warm_up_episodes:
                    action = env.action_space.sample()
                else:
                    action = (
                        self.q_network(observation.unsqueeze(0).double())
                        .argmax(axis=-1)
                        .squeeze()
                        .item()
                    )
                next_observation, reward, terminated, truncated, info = env.step(action)
                if only_terminal_negative_reward:
                    reward = -1 if terminated else 0
                next_observation = torch.from_numpy(next_observation)

                self.buffer.add(observation, action, next_observation, reward, terminated)

                observation = next_observation
                finished = terminated or truncated
                episode_length += 1

            loss = 0
            if len(self.buffer) >= self.batch_size:
                for _ in range(self.episode_train_steps):
                    loss = self.optimize_model()
                if (
                        episode % 25 == 0
                        and episode < int(num_episodes * (1 / 2))
                ):
                    self.q_network.update_grid_from_samples(self.buffer.observations[: len(self.buffer)])
                    self.target_network.update_grid_from_samples(
                        self.buffer.observations[: len(self.buffer)]
                    )

                if episode % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

            print("episode: {}, the episode reward is {}".format(episode, episode_length))
            if wandbrun is not None:
                wandbrun.log({'loss': loss, 'reward': episode_length})
