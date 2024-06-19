# noinspection PyPackageRequirements
from kan import KAN

from qnet import QNet


def params(model):
    return sum(p.numel() for p in model.parameters())


def kan_params(
        width: list,
        grid,
        k,
):
    kan_net = KAN(
        width=width,
        grid=grid,
        k=k,
    )

    return params(kan_net)


def main():
    n_observations = 4
    n_actions = 2

    qnet = QNet(n_observations=n_observations, n_actions=n_actions)
    print('QNet params: ', params(qnet))

    widths = [
        [n_observations, 8, n_actions],
        [n_observations, 8, 5, n_actions],
        [n_observations, 8, 16, 8, n_actions],
        [n_observations, 128, n_actions],
        [n_observations, 128, 128, n_actions],
    ]
    for width in widths:
        params_kan = kan_params(
            width=width,
            grid=5,
            k=3,
        )
        print(f'KAN width: {width}, params: ', params_kan)


if __name__ == '__main__':
    main()
