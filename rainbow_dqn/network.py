import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float32


class NoisyLinear(nn.Module):
    weight_epsilon: Float32[torch.Tensor, "out_features in_features"]
    bias_epsilon: Float32[torch.Tensor, " out_features"]

    def __init__(
        self, in_features: int, out_features: int, sigma_0: float = 0.5
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_0 * bound)
        nn.init.constant_(self.bias_sigma, self.sigma_0 * bound)

    def _scaled_noise(self, size: int) -> Float32[torch.Tensor, " size"]:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    @torch.no_grad()
    def reset_noise(self) -> None:
        eps_in = self._scaled_noise(self.in_features)
        eps_out = self._scaled_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(
        self, x: Float32[torch.Tensor, "*batch in_features"]
    ) -> Float32[torch.Tensor, "*batch out_features"]:
        if not self.training:
            return F.linear(x, self.weight_mu, self.bias_mu)
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)


class QNetwork(nn.Module):
    def __init__(
        self, n_features: int, d_ff: int, n_atoms: int, sigma_0: float = 0.5
    ) -> None:
        super().__init__()
        self.fc1 = NoisyLinear(n_features, d_ff, sigma_0)
        self.fc2 = NoisyLinear(d_ff, d_ff, sigma_0)
        self.fc3 = NoisyLinear(d_ff, n_atoms, sigma_0)

    def reset_noise(self) -> None:
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

    def forward(
        self, x: Float32[torch.Tensor, "*batch n_features"]
    ) -> Float32[torch.Tensor, "*batch n_atoms"]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
