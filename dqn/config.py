from dataclasses import dataclass


@dataclass
class DQNConfig:
    learning_rate: float = 3e-4
    batch_size: int = 256
    max_grad_norm: float = 10.0

    gamma: float = 0.99

    capacity: int = 200000
    learning_starts: int = 5000

    max_steps: int = 1000000
    train_frequency: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 300000
    target_update_period: int = 1000

    checkpoint_period: int = 10000
    snapshot_period: int = 100000
    eval_period: int = 10000
    eval_episodes: int = 10

    d_ff: int = 64

    use_topout_mask: bool = False

    seed: int = 42
