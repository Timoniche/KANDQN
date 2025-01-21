import os
import pickle

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
import random

import gymnasium as gym

from yaml import CLoader

from agents.init_agent import init_agent


def _fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.mps.deterministic = True


def prepare_env(seed):
    env = gym.make("CartPole-v1")
    state, _ = env.reset(seed=seed)
    # noinspection PyUnresolvedReferences
    n_actions = env.action_space.n
    n_observations = len(state)
    env.action_space.seed(seed=seed)

    return env, n_actions, n_observations


@click.command()
@click.option("--config_file", default="configs/efficient_kaqn/efficient_kaqn_8.yaml", help="Path to config YAML file")
@click.option("--wandb_enabled", default=False, help="Send metrics to wandb")
def main(
        config_file,
        wandb_enabled,
):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=CLoader)

    training_args = config['training_args']
    run_name = training_args['run_name']
    print(f'run name: {run_name}')

    if 'use_cuda' in training_args and training_args['use_cuda']:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = "cpu"

    raw_seeds = training_args['seed']
    if isinstance(raw_seeds, int):
        seeds = [raw_seeds]
    else:
        seeds = [int(x) for x in raw_seeds.split(',')]
    rewardss = []
    for seed in seeds:
        _fix_seed(seed)
        env, n_actions, n_observations = prepare_env(seed)

        wandbrun = None
        if wandb_enabled:
            wandb_key = os.getenv('WANDB_KEY')
            wandb.login(key=wandb_key)

            wandbrun = wandb.init(
                project=config['common']['project'],
                config=config,
                name=config['training_args']['run_name'],
            )

        agent = init_agent(
            training_args=training_args,
            n_actions=n_actions,
            n_observations=n_observations,
            device=device,
        )

        rewards = agent.train(
            num_episodes=training_args['num_episodes'],
            env=env,
            seed=seed,
            only_terminal_negative_reward=training_args['only_terminal_negative_reward'],
            wandbrun=wandbrun,
        )
        rewardss.append(rewards)
    avg_rewards = []
    for i in range(len(rewardss[0])):
        fst_avg = 0.0
        for j in range(len(rewardss)):
            fst_avg += rewardss[j][i]
        fst_avg /= len(rewardss)
        avg_rewards.append(fst_avg)
    with open(f'{run_name}.pkl', 'wb') as f:
        pickle.dump(avg_rewards, f)
    plt.plot(range(len(avg_rewards)), avg_rewards)
    plt.show()


if __name__ == '__main__':
    main()
