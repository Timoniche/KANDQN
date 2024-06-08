import os

import click
import numpy as np
import torch
import wandb
import yaml
import random

import gymnasium as gym

from yaml import CLoader

from dqn import DQN, EpsilonExploration
from trainer import train


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
@click.option("--config_file", default="configs/config.yaml", help="Path to config YAML file")
def main(config_file):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=CLoader)

    training_args = config['training_args']

    seed = training_args['seed']
    _fix_seed(seed)
    env, n_actions, n_observations = prepare_env(seed)

    wandb_enabled = False
    wandbrun = None

    if wandb_enabled:
        wandb_key = os.getenv('WANDB_KEY')
        wandb.login(key=wandb_key)

        wandbrun = wandb.init(
            project=config['common']['project'],
            config=config,
            name=config['training_args']['run_name'],
        )

    epsilon_exploration = EpsilonExploration(
        eps_start=training_args['eps_start'],
        eps_end=training_args['eps_end'],
        eps_decay=training_args['eps_decay'],
    )

    dqn = DQN(
        n_actions=n_actions,
        n_observations=n_observations,
        batch_size=training_args['batch_size'],
        gamma=training_args['gamma'],
        epsilon_exploration=epsilon_exploration,
        tau=training_args['tau'],
        lr=training_args['lr'],
        replay_memory_capacity=training_args['replay_memory_capacity'],
        device=device,
    )

    train(
        dqn,
        num_episodes=training_args['num_episodes'],
        env=env,
        device=device,
        seed=seed,
        wandbrun=wandbrun,
    )


if __name__ == '__main__':
    main()
