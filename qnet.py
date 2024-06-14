import torch.nn as nn
import torch.nn.functional as F
from kan import KAN


class QNet(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(QNet, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def main():
    net = QNet(n_observations=4, n_actions=2)
    params = sum(p.numel() for p in net.parameters())
    print('QNet: ', params)

    # QNet:  17410
    policy_net = KAN(
        width=[4, 8, 2],
        grid=5,
        k=3,
    )
    params_kan = sum(p.numel() for p in policy_net.parameters())
    # KAN:  1066
    print('KAN: ', params_kan)


if __name__ == '__main__':
    main()
