from __future__ import annotations

from .engine import PlacementResult, TetrisEngine, hole_mask
from .env import Observation, TetrisEnv, register_envs
from .features import (
    BCTS_FEATURE_NAMES,
    DELLACHERIE_FEATURE_NAMES,
    DELLACHERIE_WEIGHTS,
    FEATURE_NAMES,
    FEATURE_SET_SIZES,
    N_DELLACHERIE_FEATURES,
    N_FEATURES,
    N_THIERY_FEATURES,
    THIERY_FEATURE_NAMES,
    board_features,
    column_heights,
    placement_features,
)
from .pieces import N_PIECES, N_ROTATIONS, PIECE_NAMES, shape

register_envs()

__all__ = [
    "BCTS_FEATURE_NAMES",
    "DELLACHERIE_FEATURE_NAMES",
    "DELLACHERIE_WEIGHTS",
    "FEATURE_NAMES",
    "FEATURE_SET_SIZES",
    "N_DELLACHERIE_FEATURES",
    "N_FEATURES",
    "N_PIECES",
    "N_ROTATIONS",
    "N_THIERY_FEATURES",
    "PIECE_NAMES",
    "THIERY_FEATURE_NAMES",
    "Observation",
    "PlacementResult",
    "TetrisEngine",
    "TetrisEnv",
    "board_features",
    "column_heights",
    "hole_mask",
    "placement_features",
    "register_envs",
    "shape",
]
