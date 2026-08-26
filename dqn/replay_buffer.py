import numpy as np
from jaxtyping import Bool, Float32


class ReplayBuffer:
    afterstates: Float32[np.ndarray, "capacity n_features"]
    next_states: Float32[np.ndarray, "capacity n_actions n_features"]
    next_masks: Bool[np.ndarray, "capacity n_actions"]
    rewards: Float32[np.ndarray, " capacity"]
    terminated: Bool[np.ndarray, " capacity"]
    truncated: Bool[np.ndarray, " capacity"]

    def __init__(
        self, capacity: int, n_features: int, n_actions: int, rng: np.random.Generator
    ) -> None:
        self.capacity = capacity
        self.size = 0
        self.idx = 0

        self.afterstates = np.zeros((capacity, n_features), dtype=np.float32)
        self.next_states = np.zeros((capacity, n_actions, n_features), dtype=np.float32)
        self.next_masks = np.zeros((capacity, n_actions), dtype=np.bool_)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool_)
        self.truncated = np.zeros(capacity, dtype=np.bool_)

        self.rng = rng

    def push(
        self,
        afterstate: Float32[np.ndarray, " n_features"],
        next_state: Float32[np.ndarray, "n_actions n_features"],
        next_mask: Bool[np.ndarray, " n_actions"],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.afterstates[self.idx] = afterstate
        self.next_states[self.idx] = next_state
        self.next_masks[self.idx] = next_mask
        self.rewards[self.idx] = reward
        self.terminated[self.idx] = terminated
        self.truncated[self.idx] = truncated

        self.size = min(self.size + 1, self.capacity)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[
        Float32[np.ndarray, "batch n_features"],
        Float32[np.ndarray, "batch n_actions n_features"],
        Bool[np.ndarray, "batch n_actions"],
        Float32[np.ndarray, " batch"],
        Bool[np.ndarray, " batch"],
        Bool[np.ndarray, " batch"],
    ]:
        idxs = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.afterstates[idxs],
            self.next_states[idxs],
            self.next_masks[idxs],
            self.rewards[idxs],
            self.terminated[idxs],
            self.truncated[idxs],
        )

    def __len__(self) -> int:
        return self.size
