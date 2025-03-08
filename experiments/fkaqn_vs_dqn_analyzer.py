import pickle

import matplotlib.pyplot as plt
import numpy as np


def count_statistics(rewards, name):
    max_reward = max(rewards)
    median = np.median(rewards)

    print(f'----{name} max_reward: {max_reward}')
    print(f'----{name} median_reward: {median}')

    return max_reward, median


def plot_medians(memories, fkaqn_meds, dqn_meds):
    x = range(len(memories))
    plt.plot(x, fkaqn_meds, color='red', label='FastKAN Medians, hidden=16, 938 params')
    plt.plot(x, dqn_meds, color='blue', label='DQN Medians, hidden=32, 1282 params')
    plt.xticks(x, memories)
    plt.xlabel('Replay Memory Size')
    plt.ylabel('Median Reward')
    plt.legend()
    plt.title('Medians FastKAN vs DQN')
    plt.show()


def plot_maxs(memories, fkaqn_maxs, dqn_maxs):
    x = range(len(memories))
    plt.plot(x, fkaqn_maxs, color='red', label='FastKAN Max Reward, hidden=16, 938 params')
    plt.plot(x, dqn_maxs, color='blue', label='DQN Max Reward, hidden=32, 1282 params')
    plt.xticks(x, memories)
    plt.xlabel('Replay Memory Size')
    plt.ylabel('Max Reward')
    plt.legend()
    plt.title('Max Rewards FastKAN vs DQN')
    plt.show()


def main():
    memories = [
        1,
        128,
        500,
        1000,
        # 2000,
        # 5000,
        # 10000,
        # 20000,
        # 50000,
        # 100000,
    ]
    fkaqn_maxs = []
    fkaqn_meds = []
    dqn_maxs = []
    dqn_meds = []
    for memory in memories:
        print(f'memory: {memory}')
        fkaqn_rewards = f'runs_dump/fast-kaqn-width_16_memory_{memory}.pkl'
        dqn_rewards = f'runs_dump/dqn-width_32_memory_{memory}.pkl'
        with open(fkaqn_rewards, 'rb') as file:
            fkaqn = pickle.load(file)
        with open(dqn_rewards, 'rb') as file:
            dqn = pickle.load(file)

        fkaqn_max, fkaqn_med = count_statistics(fkaqn, 'fkaqn')
        dqn_max, dqn_med = count_statistics(dqn, 'dqn')

        fkaqn_maxs.append(fkaqn_max)
        fkaqn_meds.append(fkaqn_med)
        dqn_maxs.append(dqn_max)
        dqn_meds.append(dqn_med)

    plot_medians(memories, fkaqn_meds, dqn_meds)
    plot_maxs(memories, fkaqn_maxs, dqn_maxs)


if __name__ == '__main__':
    main()
