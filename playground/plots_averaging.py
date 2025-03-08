import pickle

import matplotlib.pyplot as plt


def main():
    # fkaqn_rewards = './../fast-kaqn-width_8-no-memory-10-seeds.pkl'
    fkaqn_rewards = './../runs_dump/fast-kaqn-width_8-no-memory-10-seeds.pkl'
    dqn_rewards = './../dqn-width_32-no_memory-10-seeds.pkl'
    with open(fkaqn_rewards, 'rb') as file:
        fkaqn = pickle.load(file)
    with open(dqn_rewards, 'rb') as file:
        dqn = pickle.load(file)
    plt.plot(range(len(fkaqn)), fkaqn, color='red', label='FastKAN, hidden=8, 482 params')
    plt.plot(range(len(dqn)), dqn, color='blue', label='DQN, hidden=32, 1282 params')
    plt.legend()
    plt.title('FastKAN vs DQN')
    plt.show()


if __name__ == '__main__':
    main()
