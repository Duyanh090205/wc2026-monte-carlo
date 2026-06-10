"""Apples-to-apples: ELO / +MV / +MV+star / +MV+UEFA-disc, JSD vs live market.

Everything held identical except the one intended variable:
  - same Elo (eloratings, point-in-time 2026-06-09), same D=1400, diag=0.20,
    same HFA params, same n / seed, same WC2026 bundle.
  - SAME MV source for all MV configs: dcaribou per-player WC2026 squads
    (top-23 by citizenship) — so the UEFA discount is finally comparable
    (it needs per-player club data the official aggregate scrape lacks).
  - star = curated 2026 star counts (point-in-time), +15 Elo per star.
  - discount = UEFA-club valuations x0.70 on the SAME dcaribou squads.

Metric = JSD / L1 to the PM+Kalshi market consensus (the goal: track market).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mc_simu.single_game as sg  # noqa: E402
from mc_simu._common import banner  # noqa: E402
from mc_simu.experiment_uefa_discount import build_uefa_mv  # noqa: E402
from mc_simu.mv_blend import blend_elo_with_mv  # noqa: E402
from mc_simu.run_phase3_baselines import load_ratings_for_source  # noqa: E402
from mc_simu.single_game import ModelParams  # noqa: E402
from mc_simu.star_presence import apply_star_bonus, load_star_counts  # noqa: E402
from mc_simu.tournaments.wc2026 import load_wc2026_bundle, run_monte_carlo  # noqa: E402
from mc_simu.tune_to_market import jsd, normalize  # noqa: E402
from mc_simu.wc2026_vs_multi import load_prices_endpoint  # noqa: E402

DATA = PROJECT_ROOT / "data" / "mc_simu"
DEFAULT_API = "https://seal-app-yatxw.ondigitalocean.app/api"


def _teams48():
    return [t for g in json.load(open(DATA / "wc2026_groups.json"))["groups"].values() for t in g]


def champ(ratings_base, mv_dict, star_counts, n, seed):
    """mv_dict None -> ELO-only. star_counts None -> no star."""
    teams48 = _teams48()
    r = dict(ratings_base)
    if mv_dict is not None:
        elo_sub = {t: ratings_base[t] for t in teams48 if t in ratings_base}
        mv_pos = {t: v for t, v in mv_dict.items() if v and v > 0}
        blended, _ = blend_elo_with_mv(elo_sub, mv_pos, alpha=0.5)
        r.update(blended)
    if star_counts is not None:
        r = apply_star_bonus(r, bonus_X=15.0, star_counts=star_counts)
    sg.ELO_GOALS_DENOMINATOR = 1400.0
    bundle = load_wc2026_bundle(r, params=ModelParams(diagonal_inflation=0.20))
    res = run_monte_carlo(bundle, n_iterations=n, seed=seed, progress=False)
    return normalize({t: s["mc_fair_prob"] for t, s in res["champion"].items()})


def main() -> int:
    n, seed = 30000, 42
    teams48 = _teams48()
    banner("Build dcaribou WC2026 squad MV (raw + UEFA-disc) — common source for all configs")
    mv = build_uefa_mv()
    mv_raw = {t: d["raw"] for t, d in mv.items()}
    mv_disc = {t: d["disc"] for t, d in mv.items()}

    ratings, _ = load_ratings_for_source("eloratings", pd.Timestamp("2026-06-09"), DATA, teams48)
    star_counts = load_star_counts()

    banner(f"Run 4 configs (n={n}, identical Elo/D=1400/diag=0.20/HFA/seed)")
    configs = {
        "ELO": champ(ratings, None, None, n, seed),
        "ELO+MV": champ(ratings, mv_raw, None, n, seed),
        "ELO+MV+star": champ(ratings, mv_raw, star_counts, n, seed),
        "ELO+MV+disc": champ(ratings, mv_disc, None, n, seed),
    }

    prices = load_prices_endpoint(DEFAULT_API)
    cons = {}
    for t, row in prices.items():
        vals = [row[k] for k in ("Polymarket", "Kalshi") if k in row]
        if vals:
            cons[t] = float(np.median(vals))
    market = normalize({t: cons[t] for t in cons if t in teams48})
    teams = [t for t in teams48 if t in market]

    banner("Apples-to-apples — JSD / L1 vs PM+Kalshi consensus (lower = closer to market)")
    print(f"{'Config':<14}{'JSD':>10}{'L1 (pp)':>10}   {'Spain':>7}{'France':>7}{'Brazil':>7}{'Portugal':>9}")
    rank = []
    for name, p in configs.items():
        pn = normalize({t: p.get(t, 0.0) for t in teams})
        mn = normalize({t: market[t] for t in teams})
        j = jsd(pn, mn, teams)
        l1 = sum(abs(pn[t] - mn[t]) for t in teams) * 100
        rank.append((name, j, l1))
        print(f"{name:<14}{j:>10.4f}{l1:>10.1f}   "
              f"{pn.get('Spain',0)*100:>6.1f}%{pn.get('France',0)*100:>6.1f}%"
              f"{pn.get('Brazil',0)*100:>6.1f}%{pn.get('Portugal',0)*100:>8.1f}%")
    print(f"\n  market consensus:                  "
          f"{market.get('Spain',0)/sum(market[t] for t in teams)*100:>6.1f}%"
          f"{market.get('France',0)/sum(market[t] for t in teams)*100:>6.1f}%"
          f"{market.get('Brazil',0)/sum(market[t] for t in teams)*100:>6.1f}%"
          f"{market.get('Portugal',0)/sum(market[t] for t in teams)*100:>8.1f}%")
    best = min(rank, key=lambda r: r[1])
    print(f"\n  -> Closest to market (lowest JSD): {best[0]} ({best[1]:.4f})")
    return 0


__all__ = ["champ", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
