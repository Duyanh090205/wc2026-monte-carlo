"""Phase 1 §2.2 validation gate WITH MV blend — Brier check for MV-blended Elo.

Caveat: uses 2026-current Transfermarkt MV for historical 2018-2024 matches.
Biased (e.g., Brazil 2018 squad had different MV profile than 2026). Documenting
this as a *coarse* sanity check — not the final word on historical predictivity.

To run a fair check, would need historical Transfermarkt MV per year (per-snapshot
scraping of archived pages) — out of scope for v1.

Blend formula (per team, applied uniformly across all historical matches):
    mv_elo[t]   = elo_mean_48 + z_log_mv[t] * elo_std_48     # 2026 WC cohort stats
    blended[t,d] = (1-alpha) * live_elo[t,d] + alpha * mv_elo[t]

Teams not in 2026 MV CSV → blended = live_elo (no shift).

HARD-STOP rule (per MODEL_SPEC §2.2): best_brier > 0.21 → kill MV blend feature.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mc_simu._common import banner  # noqa: E402
from mc_simu.elo import build_rating_history, get_rating_as_of  # noqa: E402
from mc_simu.mv_blend import load_mv_table  # noqa: E402
from mc_simu.run_phase3_baselines import load_ratings_for_source  # noqa: E402
from mc_simu.single_game import ModelParams, predict_match  # noqa: E402
from mc_simu.validate_phase1 import (  # noqa: E402
    DATA_DIR, RESULTS_CSV, TOURNAMENT_META, VAL_TOURNAMENTS,
    MINI_TUNE_ALPHAS, MINI_TUNE_DIAGS,
    _brier_mean, _make_context, _outcome_vec,
)
import json  # noqa: E402


def build_mv_elo_table(alpha: float) -> dict[str, float]:
    """Per-team MV-derived Elo using 2026 WC cohort z-score stats.

    Returns {team: mv_elo} for teams in MV CSV. Teams not here will fall back
    to raw live Elo (alpha effectively 0 for them).

    Note: also returns elo_mean_48 for additive-shift computation.
    """
    mv = load_mv_table()
    # Pull 2026 Elo for the 48 cohort to get mean/std (snapshot-style)
    with open(DATA_DIR / "wc2026_groups.json", encoding="utf-8") as f:
        g = json.load(f)
    teams_48 = [t for grp in g["groups"].values() for t in grp]
    ratings, _ = load_ratings_for_source("eloratings", pd.Timestamp.now(), DATA_DIR, teams_48)
    elos = np.array([ratings[t] for t in teams_48 if t in ratings])
    elo_mean = float(elos.mean())
    elo_std = float(elos.std())

    mv_teams = [t for t in teams_48 if t in mv]
    log_mvs = np.log(np.array([mv[t] for t in mv_teams]))
    lmv_mean, lmv_std = float(log_mvs.mean()), float(log_mvs.std())

    mv_elo: dict[str, float] = {}
    for t in mv_teams:
        z_mv = (np.log(mv[t]) - lmv_mean) / lmv_std
        mv_elo[t] = elo_mean + float(z_mv) * elo_std
    return mv_elo, elo_mean, alpha


def blend_live_elo(live_elo: float, team: str, mv_elo: dict[str, float], alpha: float) -> float:
    """Apply additive MV shift: blended = (1-α)·live + α·mv_elo[t]. No shift if team missing."""
    if team not in mv_elo:
        return live_elo
    return (1.0 - alpha) * live_elo + alpha * mv_elo[team]


def load_val_matches() -> pd.DataFrame:
    """Replicate validate_phase1.py's match selection."""
    df = pd.read_csv(RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    keep = pd.DataFrame()
    for tname, year, label in VAL_TOURNAMENTS:
        sub = df[(df.tournament == tname) & (df.year == year)].copy()
        sub["label"] = label
        keep = pd.concat([keep, sub], ignore_index=True)
    return keep


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mv-alpha", type=float, default=0.5,
                   help="MV blend weight (0=Elo only, 0.5=balanced, 0.7=MV-heavy)")
    args = p.parse_args()

    banner(f"Phase 1 Brier validation WITH MV blend (alpha={args.mv_alpha})")
    print("Caveat: 2026 MV applied to 2018-2024 matches — historically biased.")
    print()

    print("Loading match history + building Elo timeline ...")
    df = pd.read_csv(RESULTS_CSV)
    history = build_rating_history(df)
    print(f"  teams in history: {len(history)}")

    val_matches = load_val_matches()
    print(f"  validation matches: {len(val_matches)} (across {len(VAL_TOURNAMENTS)} tournaments)")

    mv_elo, elo_mean_48, _ = build_mv_elo_table(args.mv_alpha)
    print(f"  MV-derived Elo built for {len(mv_elo)} teams (2026 WC cohort)")
    print(f"  elo_mean_48={elo_mean_48:.0f}")

    with open(TOURNAMENT_META) as f:
        meta = json.load(f)

    cache = []
    blended_team_count = 0
    for row in val_matches.itertuples(index=False):
        rd = row._asdict()
        date = pd.Timestamp(rd["date"])
        elo_h_raw = get_rating_as_of(history, rd["home_team"], date)
        elo_a_raw = get_rating_as_of(history, rd["away_team"], date)
        elo_h = blend_live_elo(elo_h_raw, rd["home_team"], mv_elo, args.mv_alpha)
        elo_a = blend_live_elo(elo_a_raw, rd["away_team"], mv_elo, args.mv_alpha)
        if rd["home_team"] in mv_elo: blended_team_count += 1
        if rd["away_team"] in mv_elo: blended_team_count += 1
        ctx = _make_context(rd, rd["label"], meta)
        actual = _outcome_vec(rd["home_score"], rd["away_score"])
        cache.append({
            "elo_h": elo_h, "elo_a": elo_a, "ctx": ctx, "actual": actual,
            "label": rd["label"],
        })
    pct_blended = blended_team_count / (2 * len(cache)) * 100
    print(f"  team-slots blended (in MV cohort): {pct_blended:.1f}%")
    print()

    per_combo = {}
    for a in MINI_TUNE_ALPHAS:
        for d in MINI_TUNE_DIAGS:
            params = ModelParams(alpha=a, beta=0.13, diagonal_inflation=d, blend_match_weight=1.0)
            briers = [_brier_mean(predict_match(e["elo_h"], e["elo_a"], e["ctx"], params), e["actual"]) for e in cache]
            per_combo[(a, d)] = float(np.mean(briers))

    best_combo = min(per_combo.keys(), key=lambda k: per_combo[k])
    best_brier = per_combo[best_combo]
    print(f"Per-combo Brier grid (lower=better):")
    print(f"          diag=0.10  diag=0.20  diag=0.30")
    for a in MINI_TUNE_ALPHAS:
        row = f"  α={a:.2f}"
        for d in MINI_TUNE_DIAGS:
            row += f"    {per_combo[(a, d)]:.4f}"
        print(row)
    print()

    # Per-tournament
    print(f"Per-tournament Brier (at best combo α={best_combo[0]}, diag={best_combo[1]}):")
    params = ModelParams(alpha=best_combo[0], beta=0.13, diagonal_inflation=best_combo[1], blend_match_weight=1.0)
    for tname, year, label in VAL_TOURNAMENTS:
        sub_cache = [e for e in cache if e["label"] == label]
        if not sub_cache: continue
        briers = [_brier_mean(predict_match(e["elo_h"], e["elo_a"], e["ctx"], params), e["actual"]) for e in sub_cache]
        print(f"  {label:<28} {np.mean(briers):.4f}  (n={len(briers)})")
    print()

    banner(f"VERDICT: best_brier={best_brier:.4f}")
    if best_brier <= 0.19:
        print("  PASS [tier 1: best_brier <= 0.19]"); exit_code = 0
    elif best_brier <= 0.21:
        print("  PASS-WARN [tier 2: 0.19 < best_brier <= 0.21]"); exit_code = 1
    else:
        print("  HARD STOP [tier 3: best_brier > 0.21] — kill MV blend feature"); exit_code = 2
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
