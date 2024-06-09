from statistics import mean

import torch
from tqdm import tqdm

from itertools import count


def train(
        agent,
        env,
        num_episodes,
        device,
        seed,
        wandbrun=None,
):
    for i_episode in tqdm(range(num_episodes)):
        state, info = env.reset(seed=seed)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_losses = []
        for t in count():
            action = agent.select_action(state, env)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            agent.store_transition(state, action, next_state, reward)

            step_loss = agent.optimize_model()
            episode_losses.append(step_loss)

            agent.update_target_network()

            state = next_state

            if done:
                reward = t + 1
                episode_mean_loss = mean(episode_losses)
                print("episode: {}, the episode reward is {}".format(i_episode, reward))
                if wandbrun is not None:
                    wandbrun.log({'loss': episode_mean_loss, 'reward': reward})
                break