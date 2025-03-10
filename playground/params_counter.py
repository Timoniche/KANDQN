# noinspection PyPackageRequirements
from kan import KAN

from efficient_kan import KAN as EKAN
from fastkan import FastKAN
from kaenet import KAEImpl
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


def fast_kan_params(
        kan_layers,
):
    fast_kan = FastKAN(
        layers_hidden=kan_layers,
    )

    return params(fast_kan)


def efficient_kan_params(
        kan_layers,
):
    efficient_kan = EKAN(
        layers_hidden=kan_layers,
    )

    return params(efficient_kan)


def kae_params(
        n_observations,
        width,
        n_actions,
):
    kae = KAEImpl(
        input_dim=n_observations,
        latent_dim=width,
        output_dim=n_actions,
    )

    return params(kae)


def main():
    n_observations = 4
    n_actions = 2

    hidden_dims = [32, 64, 128]
    for hidden_dim in hidden_dims:
        qnet = QNet(
            n_observations=n_observations,
            hidden_dim=hidden_dim,
            n_actions=n_actions
        )
        print(f'QNet hidden: {hidden_dim}, params: ', params(qnet))
    print()

    widths = [
        [n_observations, 8, n_actions],
        [n_observations, 9, n_actions],
        [n_observations, 16, n_actions],
        [n_observations, 8, 5, n_actions],
        [n_observations, 8, 16, 8, n_actions],
        [n_observations, 32, n_actions],
        [n_observations, 64, n_actions],
        [n_observations, 128, n_actions],
        [n_observations, 32, 32, n_actions],
    ]
    for width in widths:
        params_kan = kan_params(
            width=width,
            grid=5,
            k=3,
        )
        print(f'KAN width: {width}, params: ', params_kan)
    print()
    for width in widths:
        params_fast_kan = fast_kan_params(width)
        print(f'FAST KAN width: {width}, params: ', params_fast_kan)
    print()
    for width in widths:
        params_ekan = efficient_kan_params(width)
        print(f'EFFICIENT KAN width: {width}, params: ', params_ekan)
    print()
    kae_width = 32
    params_kae = kae_params(n_observations, kae_width, n_actions)
    print(f'KAE width: {kae_width}, params: ', params_kae)


if __name__ == '__main__':
    main()
