import pickle

import matplotlib.pyplot as plt
import numpy as np


def count_statistics(rewards, name):
    max_reward = max(rewards)
    median = np.median(rewards)

    print(f'----{name} max_reward: {max_reward}')
    print(f'----{name} median_reward: {median}')

    return max_reward, median

def main():
    memory = 1
    print(f'memory: {memory}')
    fkaqn_rewards = f'runs_dump/fast-kaqn-width_16_memory_{memory}.pkl'
    dqn_rewards = f'runs_dump/dqn-width_32_memory_{memory}.pkl'
    with open(fkaqn_rewards, 'rb') as file:
        fkaqn = pickle.load(file)
    with open(dqn_rewards, 'rb') as file:
        dqn = pickle.load(file)

    count_statistics(fkaqn, 'fkaqn')
    count_statistics(dqn, 'dqn')

    plt.plot(range(len(fkaqn)), fkaqn, color='red', label='FastKAN, hidden=16, 938 params')
    plt.plot(range(len(dqn)), dqn, color='blue', label='DQN, hidden=32, 1282 params')
    plt.legend()
    plt.title('FastKAN vs DQN')
    plt.show()


if __name__ == '__main__':
    main()
