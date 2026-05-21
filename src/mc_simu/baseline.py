"""Baseline predictors — ignorance floors that any Elo-based model should beat.

Implements Phase 3 §3.8. Same callable interface as `predict_match` so it plugs
into `build_group_samplers` / `build_ko_advance_table` via `Predictor` protocol.

Two baselines:
- 40/40/20 — fixed home/draw/away split anchored to international football
  base rates (~40% home win, ~25-30% draw, ~30-35% away).
- 33/33/33 — pure ignorance triple-equal split.

The goal grid is intentionally crude: uniform over a small (0..3) × (0..3) window
because tournament MC only samples scorelines for the GROUP STAGE (need exact GF/GA
for tiebreakers); knockout decisions use only p_home/p_draw/p_away marginals.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from mc_simu._common import banner  # noqa: E402,F401 — UTF-8 stdout side-effect
from mc_simu.single_game import MatchContext, MatchPrediction  # noqa: E402


_MAX_GOALS = 8  # match single_game.goal_distribution shape


def _uniform_goal_grid(p_home: float, p_draw: float, p_away: float) -> np.ndarray:
    """Build a 9x9 grid whose triangular sums match (p_home, p_draw, p_away).

    Uniform within each region:
      diagonal (9 cells)         — total mass = p_draw, each cell = p_draw / 9
      lower triangle (36 cells)  — total mass = p_home, each cell = p_home / 36
      upper triangle (36 cells)  — total mass = p_away, each cell = p_away / 36

    Sums to 1.0 exactly when input probs do.
    """
    grid = np.zeros((_MAX_GOALS + 1, _MAX_GOALS + 1), dtype=np.float64)
    n = _MAX_GOALS + 1
    n_diag = n
    n_tri = n * (n - 1) // 2  # 36

    cell_draw = p_draw / n_diag
    cell_home = p_home / n_tri
    cell_away = p_away / n_tri

    for i in range(n):
        for j in range(n):
            if i == j:
                grid[i, j] = cell_draw
            elif i > j:  # home goals > away goals → home win
                grid[i, j] = cell_home
            else:
                grid[i, j] = cell_away

    return grid


def predict_match_40_40_20(
    elo_a: float,        # noqa: ARG001 — interface compat
    elo_b: float,        # noqa: ARG001
    ctx: MatchContext,   # noqa: ARG001
) -> MatchPrediction:
    """Ignorance baseline aligned to int'l football base rates.

    40/20/40 chosen over 40/25/35 for round-number "obvious ignorance" framing;
    any reasonable choice in (38-42, 20-30, 30-40) gives the same baseline-vs-Elo
    head-to-head verdict in Phase 4 LOTO-CV.
    """
    grid = _uniform_goal_grid(0.40, 0.20, 0.40)
    return MatchPrediction(p_home=0.40, p_draw=0.20, p_away=0.40, goal_grid=grid)


def predict_match_uniform(
    elo_a: float,        # noqa: ARG001
    elo_b: float,        # noqa: ARG001
    ctx: MatchContext,   # noqa: ARG001
) -> MatchPrediction:
    """Pure-ignorance triple-equal baseline (33.33/33.33/33.33)."""
    third = 1.0 / 3.0
    grid = _uniform_goal_grid(third, third, third)
    return MatchPrediction(p_home=third, p_draw=third, p_away=third, goal_grid=grid)
