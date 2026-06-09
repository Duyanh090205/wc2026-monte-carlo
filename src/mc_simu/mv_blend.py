"""Blend Elo ratings with Transfermarkt squad market values.

Insertion point: between `load_ratings_for_source()` (returns `dict[team, elo]`)
and the predictor (`predict_match` consumes Elo numerically). The blend output
is on the Elo scale — predict_match doesn't know/care that MV was mixed in.

Formula:
    z_elo[t]  = (elo[t] - mean(elo)) / std(elo)
    z_mv[t]   = (log(mv[t]) - mean(log(mv))) / std(log(mv))
    z_blend   = (1 - alpha) * z_elo + alpha * z_mv
    blended_elo[t] = z_blend * std(elo) + mean(elo)

Why log(MV): MV is heavy right-skewed (max/min ≈ 80x). Log makes "tier difference"
roughly constant — France→England gap (linear €170m) becomes comparable to
Egypt→Jordan gap (linear €120m, but huge proportionally). Z-score then makes
the two signals comparable in standardized units.

Why z-score (not min-max): preserves spread shape; alpha weights signals in
"standardized strength" not in arbitrary scales. Mapping back to Elo scale keeps
predict_match's Elo math unchanged.

Sam direction 2026-05-27 + memory [[mc-player-market-value-feature]]: hardcoded
alpha=0.7 v1 (MV-heavy per user intuition that PM/Kalshi price by MV).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MV_CSV = PROJECT_ROOT / "data" / "mc_simu" / "transfermarkt_mv_wc2026.csv"
DEFAULT_ALPHA = 0.7


def load_mv_table(csv_path: Path | None = None) -> dict[str, float]:
    """Load Transfermarkt MV CSV → {team: total_mv_eur}. Skips zero-MV rows."""
    csv_path = csv_path or DEFAULT_MV_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MV CSV missing: {csv_path}. Run `python src/mc_simu/scrape_transfermarkt_mv.py` first."
        )
    out: dict[str, float] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mv = float(row["total_mv_eur"])
            if mv > 0:
                out[row["team"]] = mv
    return out


def blend_elo_with_mv(
    elo: dict[str, float],
    mv: dict[str, float],
    alpha: float = DEFAULT_ALPHA,
) -> tuple[dict[str, float], list[str]]:
    """Blend Elo with log(MV) in z-score space, return blended dict on Elo scale.

    Returns:
        (blended_ratings, teams_without_mv)
        Teams without MV keep their original Elo unchanged (no MV signal available).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    teams_with_mv = [t for t in elo if t in mv]
    teams_without_mv = [t for t in elo if t not in mv]

    if len(teams_with_mv) < 2:
        return dict(elo), teams_without_mv  # can't compute stats with <2 samples

    elos = np.array([elo[t] for t in teams_with_mv], dtype=float)
    log_mvs = np.log(np.array([mv[t] for t in teams_with_mv], dtype=float))

    elo_mean, elo_std = elos.mean(), elos.std()
    lmv_mean, lmv_std = log_mvs.mean(), log_mvs.std()

    if elo_std == 0 or lmv_std == 0:
        return dict(elo), teams_without_mv  # degenerate, no blend possible

    z_elo = (elos - elo_mean) / elo_std
    z_mv = (log_mvs - lmv_mean) / lmv_std
    z_blend = (1.0 - alpha) * z_elo + alpha * z_mv
    blended = z_blend * elo_std + elo_mean

    out = {t: float(blended[i]) for i, t in enumerate(teams_with_mv)}
    for t in teams_without_mv:
        out[t] = float(elo[t])
    return out, teams_without_mv


def blend_summary(
    elo: dict[str, float],
    mv: dict[str, float],
    blended: dict[str, float],
    top_n: int = 10,
) -> str:
    """Pretty-print blend movements for diagnostic logs."""
    lines = [f"{'Team':<22} {'Elo':>7} {'MV(€bn)':>9} {'Blended':>8} {'Delta':>8}"]
    teams_sorted = sorted(elo.keys(), key=lambda t: -blended.get(t, elo[t]))[:top_n]
    for t in teams_sorted:
        e = elo[t]
        mv_b = mv.get(t, 0) / 1e9
        b = blended.get(t, e)
        d = b - e
        lines.append(f"{t:<22} {e:>7.0f} {mv_b:>9.2f} {b:>8.0f} {d:>+8.1f}")
    return "\n".join(lines)


__all__ = ["load_mv_table", "blend_elo_with_mv", "blend_summary",
           "DEFAULT_ALPHA", "DEFAULT_MV_CSV"]
