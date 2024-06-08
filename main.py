import os

import click
import torch
import wandb
import yaml

import gymnasium as gym

from yaml import CLoader

from dqn import DQN, EpsilonExploration
from trainer import train


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

    env = gym.make("CartPole-v1")
    # noinspection PyUnresolvedReferences
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

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
        wandbrun=wandbrun,
    )


if __name__ == '__main__':
    main()
