from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Int8

PIECE_NAMES = ("I", "O", "T", "S", "Z", "J", "L")
N_PIECES = len(PIECE_NAMES)
N_ROTATIONS = 4

_SPAWN = {
    "I": [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
    "O": [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    "T": [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
    "S": [[0, 1, 1], [1, 1, 0], [0, 0, 0]],
    "Z": [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
    "J": [[1, 0, 0], [1, 1, 1], [0, 0, 0]],
    "L": [[0, 0, 1], [1, 1, 1], [0, 0, 0]],
}


@dataclass(frozen=True)
class Shape:
    piece: int
    rotation: int
    cells: tuple[tuple[int, int], ...]
    width: int
    height: int
    bottom: tuple[int, ...]


def _rot_cw(mat: Int8[np.ndarray, "n n"]) -> Int8[np.ndarray, "n n"]:
    return np.rot90(mat, k=-1)


def _build() -> tuple[list[list[Shape]], Bool[np.ndarray, "n_pieces n_rotations"]]:
    table: list[list[Shape]] = []
    for p, name in enumerate(PIECE_NAMES):
        mat = np.array(_SPAWN[name], dtype=np.int8)
        rots: list[Shape] = []
        for r in range(N_ROTATIONS):
            occupied = np.argwhere(mat > 0)
            occupied -= occupied.min(axis=0)
            cells = tuple(sorted((int(a), int(b)) for a, b in occupied))
            h = max(dr for dr, _ in cells) + 1
            w = max(dc for _, dc in cells) + 1
            bottom = []
            for c in range(w):
                col_cells = [dr for dr, dc in cells if dc == c]
                bottom.append(max(col_cells))
            rots.append(
                Shape(
                    piece=p,
                    rotation=r,
                    cells=cells,
                    width=w,
                    height=h,
                    bottom=tuple(bottom),
                )
            )
            mat = _rot_cw(mat)
        table.append(rots)

    unique = np.ones((N_PIECES, N_ROTATIONS), dtype=bool)
    for p in range(N_PIECES):
        seen: set = set()
        for r in range(N_ROTATIONS):
            k = table[p][r].cells
            if k in seen:
                unique[p, r] = False
            seen.add(k)
    return table, unique


SHAPES, UNIQUE_ROTATION = _build()


def shape(piece: int, rotation: int) -> Shape:
    return SHAPES[piece][rotation]
