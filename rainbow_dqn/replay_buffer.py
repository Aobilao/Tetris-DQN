import numpy as np
from jaxtyping import Bool, Float32, Float64, Int64


class SumTree:
    sums: Float64[np.ndarray, " nodes"]
    mins: Float64[np.ndarray, " nodes"]

    def __init__(self, capacity: int) -> None:
        self.n = 2
        while self.n < capacity:
            self.n *= 2
        self.sums = np.zeros(2 * self.n, dtype=np.float64)
        self.mins = np.full(2 * self.n, np.inf, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.sums[1])

    @property
    def min(self) -> float:
        return float(self.mins[1])

    def update(
        self,
        idxs: Int64[np.ndarray, " batch"],
        priorities: Float64[np.ndarray, " batch"],
    ) -> None:
        k = idxs + self.n
        self.sums[k] = priorities
        self.mins[k] = priorities
        while k[0] > 1:
            k = k // 2
            self.sums[k] = self.sums[2 * k] + self.sums[2 * k + 1]
            self.mins[k] = np.minimum(self.mins[2 * k], self.mins[2 * k + 1])

    def find(
        self, prefixes: Float64[np.ndarray, " batch"]
    ) -> Int64[np.ndarray, " batch"]:
        prefixes = prefixes.copy()
        k = np.ones(prefixes.shape[0], dtype=np.int64)
        while k[0] < self.n:
            left = 2 * k
            right = prefixes > self.sums[left]
            prefixes -= np.where(right, self.sums[left], 0.0)
            k = left + right
        return k - self.n


class ReplayBuffer:
    afterstates: Float32[np.ndarray, "capacity n_features"]
    next_states: Float32[np.ndarray, "capacity n_actions n_features"]
    next_masks: Bool[np.ndarray, "capacity n_actions"]
    rewards: Float32[np.ndarray, " capacity"]
    discounts: Float32[np.ndarray, " capacity"]
    terminated: Bool[np.ndarray, " capacity"]

    def __init__(
        self,
        capacity: int,
        n_features: int,
        n_actions: int,
        rng: np.random.Generator,
        alpha: float = 0.5,
        priority_eps: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.size = 0
        self.idx = 0

        self.afterstates = np.zeros((capacity, n_features), dtype=np.float32)
        self.next_states = np.zeros((capacity, n_actions, n_features), dtype=np.float32)
        self.next_masks = np.zeros((capacity, n_actions), dtype=np.bool_)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool_)

        self.rng = rng
        self.alpha = alpha
        self.priority_eps = priority_eps
        self.max_priority = 1.0
        self.tree = SumTree(capacity)

    def push(
        self,
        afterstate: Float32[np.ndarray, " n_features"],
        next_state: Float32[np.ndarray, "n_actions n_features"],
        next_mask: Bool[np.ndarray, " n_actions"],
        reward: float,
        discount: float,
        terminated: bool,
    ) -> None:
        self.afterstates[self.idx] = afterstate
        self.next_states[self.idx] = next_state
        self.next_masks[self.idx] = next_mask
        self.rewards[self.idx] = reward
        self.discounts[self.idx] = discount
        self.terminated[self.idx] = terminated

        self.tree.update(
            np.array([self.idx]), np.array([self.max_priority**self.alpha])
        )

        self.size = min(self.size + 1, self.capacity)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> tuple[
        Float32[np.ndarray, "batch n_features"],
        Float32[np.ndarray, "batch n_actions n_features"],
        Bool[np.ndarray, "batch n_actions"],
        Float32[np.ndarray, " batch"],
        Float32[np.ndarray, " batch"],
        Bool[np.ndarray, " batch"],
        Int64[np.ndarray, " batch"],
        Float32[np.ndarray, " batch"],
    ]:
        segment = self.tree.total / batch_size
        prefixes = (np.arange(batch_size) + self.rng.random(batch_size)) * segment
        idxs = np.minimum(self.tree.find(prefixes), self.size - 1)

        priorities = self.tree.sums[idxs + self.tree.n]
        weights = (self.tree.min / priorities) ** beta

        return (
            self.afterstates[idxs],
            self.next_states[idxs],
            self.next_masks[idxs],
            self.rewards[idxs],
            self.discounts[idxs],
            self.terminated[idxs],
            idxs,
            weights.astype(np.float32),
        )

    def update_priorities(
        self,
        idxs: Int64[np.ndarray, " batch"],
        errors: Float32[np.ndarray, " batch"],
    ) -> None:
        priorities = np.abs(errors).astype(np.float64) + self.priority_eps
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self.tree.update(idxs, priorities**self.alpha)

    def __len__(self) -> int:
        return self.size
