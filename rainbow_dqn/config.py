from dataclasses import dataclass


@dataclass
class RDQNConfig:
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_grad_norm: float = 10.0

    gamma: float = 0.99
    n_step: int = 3

    capacity: int = 200000
    learning_starts: int = 5000

    alpha: float = 0.5
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_anneal_steps: int = 1000000
    priority_eps: float = 1e-6

    max_steps: int = 1000000
    train_frequency: int = 1
    target_update_period: int = 1000
    checkpoint_period: int = 10000
    snapshot_period: int = 100000

    eval_period: int = 10000
    eval_episodes: int = 10

    d_ff: int = 64

    n_atoms: int = 131
    v_min: float = -6.0
    v_max: float = 30.0

    noisy_sigma: float = 0.5

    use_topout_mask: bool = False

    seed: int = 42
