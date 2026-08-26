from dataclasses import replace

import gymnasium as gym
import torch

from rainbow_dqn.agent import RDQNAgent


def save_agent(
    agent: RDQNAgent,
    path: str,
    step: int = 0,
    env: gym.Env | None = None,
    wandb_run_id: str | None = None,
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
            "beta": agent.beta,
            "wandb_run_id": wandb_run_id,
        },
        path,
    )


def load_agent(
    path: str,
    n_features: int,
    n_actions: int,
    device: torch.device = torch.device("cpu"),
    env: gym.Env | None = None,
) -> tuple[RDQNAgent, int, str | None]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    agent = RDQNAgent(n_features, n_actions, ckpt["config"], device)
    agent.online.load_state_dict(ckpt["online_state_dict"])
    agent.target.load_state_dict(ckpt["target_state_dict"])
    agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    agent.rng = ckpt["rng"]
    agent.buffer = ckpt["buffer"]
    agent.normalizer = ckpt["normalizer"]
    agent.beta = ckpt["beta"]
    if env is not None and ckpt.get("env_rng") is not None:
        env.unwrapped.np_random = ckpt["env_rng"]
    return agent, int(ckpt["step"]), ckpt.get("wandb_run_id")


def load_policy(
    path: str,
    n_features: int,
    n_actions: int,
    device: torch.device = torch.device("cpu"),
) -> RDQNAgent:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    agent = RDQNAgent(
        n_features, n_actions, replace(ckpt["config"], capacity=1), device
    )
    agent.online.load_state_dict(ckpt["online_state_dict"])
    agent.normalizer = ckpt["normalizer"]
    agent.eval()
    return agent
