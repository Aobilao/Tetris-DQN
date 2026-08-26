import numpy as np
import torch
from jaxtyping import Float32, Float64


class RunningNorm:
    mean: Float64[np.ndarray, " n_features"]
    var: Float64[np.ndarray, " n_features"]

    def __init__(self, n_features: int, eps: float = 1e-4) -> None:
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.var = np.ones(n_features, dtype=np.float64)
        self.count = eps

    def update(self, x: Float32[np.ndarray, "*batch n_features"]) -> None:
        if x.size == 0:
            return
        x = x.reshape(-1, x.shape[-1])
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (
            m_a + m_b + delta**2 * self.count * batch_count / total_count
        ) / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(
        self, x: Float32[torch.Tensor, "*batch n_features"]
    ) -> Float32[torch.Tensor, "*batch n_features"]:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(np.sqrt(self.var), dtype=x.dtype, device=x.device)
        return (x - mean) / (std + 1e-6)
