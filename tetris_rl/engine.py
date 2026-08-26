from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Int32, UInt8

from .pieces import N_PIECES, N_ROTATIONS, UNIQUE_ROTATION, shape


def hole_mask(
    board: UInt8[np.ndarray, "height width"],
) -> Bool[np.ndarray, "height width"]:
    occ = board > 0
    covered = np.cumsum(occ, axis=0) > 0
    return covered & ~occ


@dataclass
class PlacementResult:
    board: UInt8[np.ndarray, "height width"]
    heights: Int32[np.ndarray, " width"]
    lines_cleared: int
    landing_height: float
    eroded_cells: int
    topped_out: bool
    valid: bool = True


class TetrisEngine:
    board: UInt8[np.ndarray, "height width"]
    heights: Int32[np.ndarray, " width"]

    def __init__(
        self,
        width: int = 10,
        height: int = 20,
        queue_size: int = 5,
        randomizer: str = "bag",
    ) -> None:
        if randomizer not in ("bag", "uniform"):
            raise ValueError("randomizer must be 'bag' or 'uniform'")
        self.width = width
        self.height = height
        self.queue_size = queue_size
        self.randomizer = randomizer
        self.n_actions = N_ROTATIONS * width

        self.board = np.zeros((height, width), dtype=np.uint8)
        self.heights = np.zeros(width, dtype=np.int32)
        self.queue: list[int] = []
        self._bag: list[int] = []
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.board[:] = 0
        self.heights[:] = 0
        self._bag.clear()
        self.queue = [self._draw() for _ in range(self.queue_size + 1)]
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False

    def _draw(self) -> int:
        assert self._rng is not None, "call reset() before drawing pieces"
        if self.randomizer == "uniform":
            return int(self._rng.integers(N_PIECES))
        if not self._bag:
            self._bag = [int(i) for i in self._rng.permutation(N_PIECES)]
        return self._bag.pop()

    @property
    def current_piece(self) -> int:
        return self.queue[0]

    @property
    def preview(self) -> list[int]:
        return self.queue[1 : self.queue_size + 1]

    def decode(self, action: int) -> tuple[int, int]:
        rotation, col = divmod(action, self.width)
        return rotation, col

    def valid_actions(self, piece: int | None = None) -> Bool[np.ndarray, " n_actions"]:
        piece = self.current_piece if piece is None else piece
        mask = np.zeros(self.n_actions, dtype=bool)
        for r in range(N_ROTATIONS):
            w = shape(piece, r).width
            if w <= self.width:
                mask[r * self.width : r * self.width + (self.width - w + 1)] = True
        return mask

    def unique_actions(
        self, piece: int | None = None
    ) -> Bool[np.ndarray, " n_actions"]:
        piece = self.current_piece if piece is None else piece
        mask = self.valid_actions(piece)
        for r in range(N_ROTATIONS):
            if not UNIQUE_ROTATION[piece, r]:
                mask[r * self.width : (r + 1) * self.width] = False
        return mask

    def _landing_row(self, piece: int, rotation: int, col: int) -> int:
        s = shape(piece, rotation)
        row = self.height
        for j, b in enumerate(s.bottom):
            row = min(row, self.height - int(self.heights[col + j]) - 1 - b)
        return row

    def simulate(self, rotation: int, col: int) -> PlacementResult:
        return self._apply(self.board, self.heights, self.current_piece, rotation, col)

    def _apply(
        self,
        board: UInt8[np.ndarray, "height width"],
        heights: Int32[np.ndarray, " width"],
        piece: int,
        rotation: int,
        col: int,
    ) -> PlacementResult:
        s = shape(piece, rotation)
        if col < 0 or col + s.width > self.width:
            return PlacementResult(
                board,
                heights,
                lines_cleared=0,
                landing_height=0.0,
                eroded_cells=0,
                topped_out=False,
                valid=False,
            )

        row = self._landing_row(piece, rotation, col)

        if row < 0:
            return PlacementResult(
                board.copy(),
                heights.copy(),
                lines_cleared=0,
                landing_height=0.0,
                eroded_cells=0,
                topped_out=True,
            )

        new_board = board.copy()
        rows = np.fromiter((row + dr for dr, _ in s.cells), dtype=np.intp, count=4)
        cols = np.fromiter((col + dc for _, dc in s.cells), dtype=np.intp, count=4)
        new_board[rows, cols] = piece + 1

        bottom_row = row + s.height - 1
        landing_height = (self.height - 1 - bottom_row) + (s.height - 1) / 2.0

        touched = np.unique(rows)
        full = touched[new_board[touched].all(axis=1)]
        n_full = int(full.size)

        eroded = 0
        if n_full:
            eroded = n_full * int(np.isin(rows, full).sum())
            keep = np.ones(self.height, dtype=bool)
            keep[full] = False
            kept = new_board[keep]
            new_board = np.vstack(
                [np.zeros((n_full, self.width), dtype=np.uint8), kept]
            )

        new_heights = self._compute_heights(new_board)
        return PlacementResult(
            board=new_board,
            heights=new_heights,
            lines_cleared=n_full,
            landing_height=landing_height,
            eroded_cells=eroded,
            topped_out=False,
        )

    def _compute_heights(
        self, board: UInt8[np.ndarray, "height width"]
    ) -> Int32[np.ndarray, " width"]:
        occupied = board > 0
        any_occ = occupied.any(axis=0)
        first_filled = occupied.argmax(axis=0)
        return np.where(any_occ, self.height - first_filled, 0).astype(np.int32)

    def place(self, rotation: int, col: int) -> PlacementResult:
        if self.game_over:
            raise RuntimeError("place() on a finished game; call reset() first")

        result = self.simulate(rotation, col)
        if not result.valid:
            return result

        self.board = result.board
        self.heights = result.heights
        self.lines_cleared += result.lines_cleared

        if result.topped_out:
            self.game_over = True
        else:
            self.pieces_placed += 1
            self.queue.pop(0)
            self.queue.append(self._draw())
        return result

    def holes(self) -> int:
        return int(hole_mask(self.board).sum())

    def render_ansi(self) -> str:
        glyphs = " IOTSZJL"
        lines = [
            "|" + "".join(glyphs[v] if v else " " for v in row) + "|"
            for row in self.board
        ]
        lines.append("+" + "-" * self.width + "+")
        lines.append(
            f" piece={'IOTSZJL'[self.current_piece]}"
            f" next={''.join('IOTSZJL'[p] for p in self.preview)}"
            f" lines={self.lines_cleared} pieces={self.pieces_placed}"
        )
        return "\n".join(lines)
