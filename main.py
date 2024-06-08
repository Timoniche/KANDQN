import os

import wandb
import yaml

import gymnasium as gym

from yaml import CLoader

from dqn import DQN
from trainer import train


def main():
    wandb_key = os.getenv('WANDB_KEY')
    wandb.login(key=wandb_key)

    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=CLoader)

    run = wandb.init(
        project=config['common']['project'],
        config=config,
        name=config['training_args']['run_name'],
    )

    env = gym.make("CartPole-v1")
    # noinspection PyUnresolvedReferences
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    dqn = DQN(
        n_actions=n_actions,
        n_observations=n_observations,
    )

    train(
        dqn,
        env=env,
        wandbrun=run,
    )


if __name__ == '__main__':
    main()
