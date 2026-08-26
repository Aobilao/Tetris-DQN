import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float32


class QNetwork(nn.Module):
    def __init__(self, n_features: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_features, d_ff)
        self.fc2 = nn.Linear(d_ff, d_ff)
        self.fc3 = nn.Linear(d_ff, 1)

    def forward(
        self, x: Float32[torch.Tensor, "*batch n_features"]
    ) -> Float32[torch.Tensor, "*batch 1"]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
