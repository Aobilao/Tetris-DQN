import argparse
import time
from typing import Any

import gymnasium as gym
from gymnasium import spaces

import tetris_rl  # noqa: F401 (registers envs)
from dqn.agent import DQNAgent
from dqn.checkpoint import load_policy

LOAD_PATH = "agent.pt"


def play(
    agent: DQNAgent, env: gym.Env, render: bool = False, seed: int | None = None
) -> tuple[float, dict[str, Any]]:
    total_reward = 0.0
    steps = 0
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    while not terminated and not truncated and steps < 20000:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += float(reward)
        if render:
            print("\033[2J\033[H", env.render(), sep="")
            time.sleep(0.03)

    return total_reward, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = gym.make("tetris_rl/Tetris-v0", render_mode="ansi")
    obs_space = env.observation_space
    assert isinstance(obs_space, spaces.Dict)
    feat_space = obs_space["afterstate_features"]
    assert isinstance(feat_space, spaces.Box)
    action_space = env.action_space
    assert isinstance(action_space, spaces.Discrete)

    agent = load_policy(LOAD_PATH, feat_space.shape[1], int(action_space.n))
    agent.eval()
    play(agent, env, render=args.render)


if __name__ == "__main__":
    main()
