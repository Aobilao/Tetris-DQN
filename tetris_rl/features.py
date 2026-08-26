from __future__ import annotations

import numpy as np
from jaxtyping import Float32, Float64, Int32, UInt8

from .engine import PlacementResult
from .engine import hole_mask as _hole_mask

PLACEMENT_FEATURE_NAMES = ("landing_height", "eroded_cells")
BOARD_FEATURE_NAMES = (
    "row_transitions",
    "column_transitions",
    "holes",
    "cumulative_wells",
    "hole_depth",
    "rows_with_holes",
    "pattern_diversity",
)
FEATURE_NAMES = PLACEMENT_FEATURE_NAMES + BOARD_FEATURE_NAMES
N_BOARD_FEATURES = len(BOARD_FEATURE_NAMES)
N_FEATURES = len(FEATURE_NAMES)

DELLACHERIE_FEATURE_NAMES = FEATURE_NAMES[:6]
BCTS_FEATURE_NAMES = FEATURE_NAMES[:8]
THIERY_FEATURE_NAMES = FEATURE_NAMES

N_DELLACHERIE_FEATURES = len(DELLACHERIE_FEATURE_NAMES)
N_THIERY_FEATURES = len(THIERY_FEATURE_NAMES)

FEATURE_SET_SIZES = {
    "dellacherie": N_DELLACHERIE_FEATURES,
    "thiery": N_THIERY_FEATURES,
}

DELLACHERIE_WEIGHTS: Float64[np.ndarray, " n_features"] = np.array(
    [
        -4.500158825,
        3.418126810,
        -3.217888287,
        -9.348695305,
        -7.899265427,
        -3.385597225,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)


def column_heights(
    board: UInt8[np.ndarray, "height width"],
) -> Int32[np.ndarray, " width"]:
    h = board.shape[0]
    occ = board > 0
    return np.where(occ.any(axis=0), h - occ.argmax(axis=0), 0).astype(np.int32)


def board_features(
    board: UInt8[np.ndarray, "height width"],
    heights: Int32[np.ndarray, " width"] | None = None,
) -> Float32[np.ndarray, " n_board_features"]:
    h, w = board.shape
    occ = board > 0

    padded_rows = np.empty((h, w + 2), dtype=bool)
    padded_rows[:, 0] = True
    padded_rows[:, -1] = True
    padded_rows[:, 1:-1] = occ
    row_transitions = int((padded_rows[:, 1:] != padded_rows[:, :-1]).sum())

    padded_cols = np.empty((h + 1, w), dtype=bool)
    padded_cols[:-1] = occ
    padded_cols[-1] = True
    column_transitions = int((padded_cols[1:] != padded_cols[:-1]).sum())

    holes_mask = _hole_mask(board)
    holes = int(holes_mask.sum())
    rows_with_holes = int(holes_mask.any(axis=1).sum())
    hole_depth = int((occ & (np.cumsum(holes_mask[::-1], axis=0)[::-1] > 0)).sum())

    left = np.ones((h, w), dtype=bool)
    right = np.ones((h, w), dtype=bool)
    left[:, 1:] = occ[:, :-1]
    right[:, :-1] = occ[:, 1:]
    well_cell = (~occ) & left & right
    counts = np.cumsum(well_cell, axis=0)
    resets = np.maximum.accumulate(np.where(well_cell, 0, counts), axis=0)
    cumulative_wells = int((counts - resets).sum())

    if heights is None:
        heights = column_heights(board)
    diffs = np.diff(heights)
    pattern_diversity = int(np.unique(diffs[np.abs(diffs) < 3]).size)

    return np.array(
        [
            row_transitions,
            column_transitions,
            holes,
            cumulative_wells,
            hole_depth,
            rows_with_holes,
            pattern_diversity,
        ],
        dtype=np.float32,
    )


def placement_features(result: PlacementResult) -> Float32[np.ndarray, " n_features"]:
    out = np.empty(N_FEATURES, dtype=np.float32)
    out[0] = result.landing_height
    out[1] = result.eroded_cells
    out[2:] = board_features(result.board, result.heights)
    return out
