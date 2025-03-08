import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class KAELayer(nn.Module):
    def __init__(self, input_dim, out_dim, order=3, addbias=True):
        super(KAELayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order + 1
        self.addbias = addbias
        self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order + 1) * 0.01)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x):
        shape = x.shape
        outshape = shape[0:-1] + (self.out_dim,)
        x = torch.reshape(x, (-1, self.input_dim))

        x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1)

        y = torch.zeros((x.shape[0], self.out_dim), device=x.device)

        for i in range(self.order):
            curcoeffs = self.coeffs[:, :, i]
            xi = (x_expanded ** i)
            term = xi * curcoeffs
            y += term.sum(dim=-1)

        if self.addbias:
            y += self.bias

        y = torch.reshape(y, outshape)
        return y

    def regularization_loss(self):
        pass


class KAEImpl(nn.Module):
    def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            output_dim: int,
    ):
        super(KAEImpl, self).__init__()

        encoder_layers = []

        encoder_layers.append(
            KAELayer(input_dim, latent_dim)
        )
        encoder_layers.append(nn.ReLU(True))

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []

        decoder_layers.append(
            KAELayer(latent_dim, output_dim)
        )
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

        self.latent_dim = latent_dim

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def forward(self, x) -> torch.Tensor:
        x = self.preprocess(x)
        z = self.encoder(x)
        return self.decoder(z)

    def training_loss(self, batch) -> torch.Tensor:
        x = self.preprocess(batch)
        recon_x = self.forward(x)
        loss = F.mse_loss(recon_x, x)
        return loss

    def testing_loss(self, batch) -> torch.Tensor:
        x = self.preprocess(batch)
        with torch.no_grad():
            recon_x = self.forward(x)
            loss = F.mse_loss(recon_x, x)
        return loss

    def encode(self, x) -> torch.Tensor:
        x = self.preprocess(x)
        z = self.encoder(x)
        return z

    def reduce(self, x: np.ndarray) -> np.ndarray:
        z = self.encode(torch.from_numpy(x).float())
        z = z.detach().numpy()
        return z
