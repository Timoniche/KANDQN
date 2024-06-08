from statistics import mean

import torch
from tqdm import tqdm

from dqn import DQN
from itertools import count


def train(
        dqn: DQN,
        env,
        num_episodes,
        device,
        wandbrun=None,
):
    for i_episode in tqdm(range(num_episodes)):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_losses = []
        for t in count():
            action = dqn.select_action(state, env)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            dqn.store_transition(state, action, next_state, reward)

            state = next_state

            step_loss = dqn.optimize_model()
            episode_losses.append(step_loss)

            dqn.update_target_network()

            if done:
                reward = t + 1
                episode_mean_loss = mean(episode_losses)
                print("episode: {}, the episode reward is {}".format(i_episode, reward))
                if wandbrun is not None:
                    wandbrun.log({'loss': episode_mean_loss, 'reward': reward})
                break
