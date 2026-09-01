from dataclasses import replace

import torch
import gymnasium as gym

from dqn.agent import DQNAgent


def save_agent(
    agent: DQNAgent,
    path: str,
    step: int = 0,
    env: gym.Env | None = None,
    wandb_run_id: str | None = None,
    best_reward: float = float("-inf"),
) -> None:
    torch.save(
        {
            "config": agent.config,
            "step": step,
            "online_state_dict": agent.online.state_dict(),
            "target_state_dict": agent.target.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "rng": agent.rng,
            "env_rng": env.unwrapped.np_random if env is not None else None,
            "buffer": agent.buffer,
            "normalizer": agent.normalizer,
            "epsilon": agent.epsilon,
            "wandb_run_id": wandb_run_id,
            "best_reward": best_reward,
        },
        path,
    )


def load_agent(
    path: str,
    n_features: int,
    n_actions: int,
    device: torch.device = torch.device("cpu"),
    env: gym.Env | None = None,
) -> tuple[DQNAgent, int, str | None, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    agent = DQNAgent(n_features, n_actions, ckpt["config"], device)
    agent.online.load_state_dict(ckpt["online_state_dict"])
    agent.target.load_state_dict(ckpt["target_state_dict"])
    agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    agent.rng = ckpt["rng"]
    agent.buffer = ckpt["buffer"]
    agent.normalizer = ckpt["normalizer"]
    agent.epsilon = ckpt["epsilon"]
    if env is not None and ckpt["env_rng"] is not None:
        env.unwrapped.np_random = ckpt["env_rng"]
    return (
        agent,
        int(ckpt["step"]),
        ckpt.get("wandb_run_id"),
        ckpt.get("best_reward", float("-inf")),
    )


def load_policy(
    path: str,
    n_features: int,
    n_actions: int,
    device: torch.device = torch.device("cpu"),
) -> DQNAgent:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    agent = DQNAgent(n_features, n_actions, replace(ckpt["config"], capacity=1), device)
    agent.online.load_state_dict(ckpt["online_state_dict"])
    agent.normalizer = ckpt["normalizer"]
    return agent
