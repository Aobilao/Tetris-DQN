import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

import tetris_rl  # noqa: F401 (registers envs)
from rainbow_dqn.agent import RDQNAgent
from rainbow_dqn.checkpoint import load_agent, save_agent
from rainbow_dqn.config import RDQNConfig
from rainbow_dqn.play import play

SAVE_PATH = "rainbow_agent.pt"


def snapshot_path(step: int) -> str:
    path = Path(SAVE_PATH)
    return str(path.with_name(f"{path.stem}_step{step}{path.suffix}"))


def evaluate(
    agent: RDQNAgent, eval_env: gym.Env, episodes: int
) -> tuple[float, dict[str, float]]:
    agent.eval()
    total_reward = 0.0
    info_sums: dict[str, float] = {}
    for _ in range(episodes):
        reward, info = play(agent, eval_env)
        total_reward += reward
        for key, value in info.items():
            info_sums[key] = info_sums.get(key, 0.0) + value
    agent.train()
    avg_reward = total_reward / episodes
    avg_info = {key: value / episodes for key, value in info_sums.items()}
    return avg_reward, avg_info


def beta_update(agent: RDQNAgent, config: RDQNConfig, step: int) -> None:
    beta = min(
        config.beta_start
        + (config.beta_end - config.beta_start) * step / config.beta_anneal_steps,
        config.beta_end,
    )
    agent.beta = beta


def train(
    agent: RDQNAgent,
    env: gym.Env,
    eval_env: gym.Env,
    config: RDQNConfig,
    start_step: int = 0,
) -> None:
    update_steps = 0
    loss = 0.0
    obs, _ = env.reset() if start_step else env.reset(seed=config.seed)
    done = False

    for step in range(start_step, config.max_steps):
        if done:
            obs, _ = env.reset()
            done = False
        obs, _, terminated, truncated, _ = agent.step(obs, env)
        done = terminated or truncated
        beta_update(agent, config, step + 1)
        if (
            len(agent.buffer) >= config.learning_starts
            and (step + 1) % config.train_frequency == 0
        ):
            loss = agent.update()
            update_steps += 1
            if update_steps % config.target_update_period == 0:
                agent.sync_target()
            if (step + 1) % 10000 == 0:
                print(f"Step {step + 1}: loss = {loss}")

        if (step + 1) % config.eval_period == 0:
            avg_reward, avg_info = evaluate(agent, eval_env, config.eval_episodes)
            print(f"Eval at step {step + 1}: avg_reward = {avg_reward}, {avg_info}")

        if (step + 1) % config.checkpoint_period == 0:
            save_agent(agent, SAVE_PATH, step + 1, env)
            print(f"Checkpointed at step {step + 1} -> {SAVE_PATH}")
        if (step + 1) % config.snapshot_period == 0:
            path = snapshot_path(step + 1)
            save_agent(agent, path, step + 1, env)
            print(f"Snapshotted at step {step + 1} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topout-mask", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    env = gym.make("tetris_rl/Tetris-v0")
    eval_env = gym.make("tetris_rl/Tetris-v0")
    obs_space = env.observation_space
    assert isinstance(obs_space, spaces.Dict)
    feat_space = obs_space["afterstate_features"]
    assert isinstance(feat_space, spaces.Box)
    action_space = env.action_space
    assert isinstance(action_space, spaces.Discrete)

    n_features, n_actions = feat_space.shape[1], int(action_space.n)
    if args.resume:
        agent, start_step = load_agent(SAVE_PATH, n_features, n_actions, env=env)
        config = agent.config
        print(f"Resumed from {SAVE_PATH} at step {start_step}")
    else:
        config = RDQNConfig(use_topout_mask=args.topout_mask)
        agent = RDQNAgent(n_features, n_actions, config)
        start_step = 0

    agent.train()
    train(agent, env, eval_env, config, start_step)
    save_agent(agent, SAVE_PATH, config.max_steps, env)


if __name__ == "__main__":
    main()
