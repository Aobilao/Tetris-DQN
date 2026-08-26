from __future__ import annotations

import numpy as np

from ..engine import TetrisEngine
from ..features import DELLACHERIE_WEIGHTS, N_FEATURES, placement_features


def greedy_action(engine: TetrisEngine, weights: np.ndarray) -> int:
    best_score, best_action = -np.inf, -1
    for a in np.flatnonzero(engine.unique_actions()):
        rotation, col = engine.decode(int(a))
        result = engine.simulate(rotation, col)
        if result.topped_out:
            continue
        score = float(weights @ placement_features(result))
        if score > best_score:
            best_score, best_action = score, int(a)
    if best_action < 0:
        best_action = int(np.flatnonzero(engine.valid_actions())[0])
    return best_action


def rollout(
    weights: np.ndarray,
    rng: np.random.Generator,
    max_pieces: int = 2000,
    width: int = 10,
    height: int = 20,
) -> tuple[int, int]:
    engine = TetrisEngine(width=width, height=height)
    engine.reset(rng)
    for _ in range(max_pieces):
        action = greedy_action(engine, weights)
        rotation, col = engine.decode(action)
        if engine.place(rotation, col).topped_out:
            break
    return engine.lines_cleared, engine.pieces_placed


def cem(
    iterations: int = 20,
    population: int = 40,
    elite_frac: float = 0.25,
    rollouts_per_candidate: int = 3,
    max_pieces: int = 400,
    seed: int = 0,
    noise_decay: float = 0.92,
    verbose: bool = True,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = np.zeros(N_FEATURES)
    std = np.ones(N_FEATURES) * 2.0
    n_elite = max(2, int(population * elite_frac))

    for it in range(iterations):
        candidates = rng.normal(mean, std, size=(population, N_FEATURES))
        scores = np.empty(population)
        for i, w in enumerate(candidates):
            scores[i] = np.mean(
                [rollout(w, rng, max_pieces)[0] for _ in range(rollouts_per_candidate)]
            )
        elite = candidates[np.argsort(scores)[-n_elite:]]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 0.05
        std *= noise_decay
        if verbose:
            print(
                f"  iter {it:2d}  best={scores.max():7.1f}  "
                f"mean={scores.mean():7.1f}  |w|={np.linalg.norm(mean):.2f}",
                flush=True,
            )
    return mean


__all__ = ["DELLACHERIE_WEIGHTS", "cem", "greedy_action", "rollout"]
