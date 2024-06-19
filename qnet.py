import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):

    def __init__(
            self,
            n_observations,
            hidden_dim,
            n_actions,
    ):
        super(QNet, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
