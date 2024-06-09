import torch
import torch.nn as nn
from kan import KAN
import numpy as np
from tqdm import tqdm


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


class Playground:
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
        self.q_network = KAN(
            width=[n_observations, width, n_actions],
            grid=grid,
            k=3,
            bias_trainable=False,
            sp_trainable=False,
            sb_trainable=False,
        )
        self.target_network = KAN(
            width=[n_observations, width, n_actions],
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

    def _train(
            self,
            net,
            target,
            data,
            optimizer,
            gamma=0.99,
    ):
        observations, actions, next_observations, rewards, terminations = data

        with torch.no_grad():
            next_q_values = net(next_observations)
            next_actions = next_q_values.argmax(dim=1)
            next_q_values_target = target(next_observations)
            target_max = next_q_values_target[range(len(next_q_values)), next_actions]
            td_target = rewards.flatten() + gamma * target_max * (
                    1 - terminations.flatten()
            )

        old_val = net(observations).gather(1, actions).squeeze()
        loss = nn.functional.mse_loss(td_target, old_val)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(
            self,
            env,
            num_episodes,
            device,
            seed,
            wandbrun=None,
    ):
        for episode in tqdm(range(num_episodes)):
            observation, info = env.reset()
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
                # if config.env_id == "CartPole-v1":
                #     reward = -1 if terminated else 0
                next_observation = torch.from_numpy(next_observation)

                self.buffer.add(observation, action, next_observation, reward, terminated)

                observation = next_observation
                finished = terminated or truncated
                episode_length += 1

            if len(self.buffer) >= self.batch_size:
                for _ in range(self.episode_train_steps):
                    loss = self._train(
                        self.q_network,
                        self.target_network,
                        self.buffer.sample(self.batch_size),
                        self.optimizer,
                        self.gamma,
                    )
                print("episode: {}, the episode reward is {}".format(episode, episode_length))
                # writer.add_scalar("episode_length", episode_length, episode)
                # writer.add_scalar("loss", loss, episode)
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
