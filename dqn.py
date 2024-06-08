import math
import random

import torch
import torch.nn as nn
import torch.optim as optim

from Transition import Transition
from qnet import QNet
from replay_memory import ReplayMemory


class EpsilonExploration:
    def __init__(
            self,
            eps_start,
            eps_end,
            eps_decay,
    ):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def exploit(self):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        return sample > eps_threshold


class DQN:
    def __init__(
            self,
            n_observations,
            n_actions,
            batch_size,
            gamma,
            epsilon_exploration: EpsilonExploration,
            tau,
            lr,
            replay_memory_capacity,
            device,
    ):
        self.policy_net = QNet(n_observations, n_actions).to(device)
        self.target_net = QNet(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(replay_memory_capacity)
        self.steps_done = 0

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_exploration = epsilon_exploration
        self.tau = tau

        self.device = device

    def select_action(self, state, env):
        if self.epsilon_exploration.exploit():
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)

    def store_transition(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (policy_net_state_dict[key] * self.tau +
                                          target_net_state_dict[key] * (1 - self.tau))
        self.target_net.load_state_dict(target_net_state_dict)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()
