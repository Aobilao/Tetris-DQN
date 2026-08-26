from __future__ import annotations

from typing import Any, TypedDict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from jaxtyping import Bool, Float32, UInt8

from .engine import TetrisEngine
from .features import FEATURE_SET_SIZES, placement_features

LINE_REWARD = (0.0, 1.0, 3.0, 5.0, 8.0)


class Observation(TypedDict):
    afterstate_features: Float32[np.ndarray, "n_actions n_features"]
    action_mask: Bool[np.ndarray, " n_actions"]
    afterstate_topout: Bool[np.ndarray, " n_actions"]


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        width: int = 10,
        height: int = 20,
        queue_size: int = 5,
        obs_type: str = "thiery",
        randomizer: str = "bag",
        step_reward: float = 0.01,
        topout_penalty: float = 5.0,
        hole_penalty: float = 0.5,
        line_reward: tuple[float, ...] = LINE_REWARD,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if obs_type not in FEATURE_SET_SIZES:
            raise ValueError(
                f"unknown obs_type {obs_type!r}; "
                f"expected one of {tuple(FEATURE_SET_SIZES)}"
            )

        self.engine = TetrisEngine(width, height, queue_size, randomizer)
        self.obs_type = obs_type
        self.n_features = FEATURE_SET_SIZES[obs_type]
        self.step_reward = step_reward
        self.topout_penalty = topout_penalty
        self.hole_penalty = hole_penalty
        self.line_reward = line_reward
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.engine.n_actions)
        self.observation_space = self._make_obs_space()

    def _make_obs_space(self) -> spaces.Dict:
        n = self.engine.n_actions
        return spaces.Dict(
            {
                "afterstate_features": spaces.Box(
                    0.0, np.inf, (n, self.n_features), np.float32
                ),
                "action_mask": spaces.MultiBinary(n),
                "afterstate_topout": spaces.MultiBinary(n),
            }
        )

    def _obs(self) -> Observation:
        e = self.engine
        n = e.n_actions
        feats = np.zeros((n, self.n_features), dtype=np.float32)
        topout = np.zeros(n, dtype=bool)

        if e.game_over:
            return {
                "afterstate_features": feats,
                "action_mask": np.zeros(n, dtype=bool),
                "afterstate_topout": topout,
            }

        mask = e.unique_actions()
        for a in np.flatnonzero(mask):
            res = e.simulate(*e.decode(int(a)))
            if res.topped_out:
                topout[a] = True
                continue
            feats[a] = placement_features(res)[: self.n_features]

        return {
            "afterstate_features": feats,
            "action_mask": mask,
            "afterstate_topout": topout,
        }

    def _info(self) -> dict[str, Any]:
        e = self.engine
        return {
            "lines_cleared": e.lines_cleared,
            "pieces_placed": e.pieces_placed,
            "holes": e.holes(),
            "max_height": int(e.heights.max()),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self.engine.reset(self.np_random)
        return self._obs(), self._info()

    def step(
        self, action: int
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        mask = self.engine.unique_actions()
        if not (0 <= action < mask.size) or not mask[action]:
            raise ValueError(
                f"action {action} is not in the current action_mask; "
                "sample only from obs['action_mask']"
            )

        rotation, col = self.engine.decode(action)
        holes_before = self.engine.holes()
        result = self.engine.place(rotation, col)
        new_holes = max(0, self.engine.holes() - holes_before)

        reward = self.line_reward[result.lines_cleared] + self.step_reward
        reward -= self.hole_penalty * new_holes
        if result.topped_out:
            reward -= self.topout_penalty

        return self._obs(), reward, result.topped_out, False, self._info()

    def render(self) -> str | UInt8[np.ndarray, "img_height img_width 3"] | None:
        if self.render_mode == "ansi":
            return self.engine.render_ansi()
        if self.render_mode == "rgb_array":
            return self._rgb()
        return None

    def _rgb(self, scale: int = 20) -> UInt8[np.ndarray, "img_height img_width 3"]:
        palette = np.array(
            [
                [18, 18, 22],
                [0, 240, 240],
                [240, 240, 0],
                [160, 0, 240],
                [0, 240, 0],
                [240, 0, 0],
                [0, 0, 240],
                [240, 160, 0],
            ],
            dtype=np.uint8,
        )
        img = palette[self.engine.board]
        return np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))


def register_envs() -> None:
    from gymnasium.envs.registration import register, registry

    if "tetris_rl/Tetris-v0" not in registry:
        register(
            id="tetris_rl/Tetris-v0",
            entry_point="tetris_rl.env:TetrisEnv",
            max_episode_steps=20_000,
        )
    if "tetris_rl/TetrisSmall-v0" not in registry:
        register(
            id="tetris_rl/TetrisSmall-v0",
            entry_point="tetris_rl.env:TetrisEnv",
            max_episode_steps=5_000,
            kwargs={"width": 6, "height": 10},
        )
