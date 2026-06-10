"""LOTO-CV tune of D (Elo->goals) and diag (draw inflation) on tournament finals.

Leave-one-tournament-out cross-validation over 11 editions (WC 2002-2022 +
Euro 2004-2024). For each held-out tournament, pick (D, diag) minimizing mean
3-way Brier on the OTHER 10, then score the held-out tournament -> out-of-sample
Brier. Answers: is the production (D=1400, diag=0.20) robust out-of-sample, or
overfit to the original 4-tournament Phase-1 subset?

Held FIXED: HFA alpha=0.27/beta=0.09, MV/star OFF. Elo = eloratings as-of
(production source), latest rating BEFORE each match (updates within tournament).
HFA fires for host games via real MatchContext. Reuses validate_phase1 scoring.

Usage: PYTHONPATH=src python -m mc_simu.loto_cv_d_diag
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
from mc_simu.run_phase3_baselines import NAME_ALIASES_BY_SOURCE  # noqa: E402
from mc_simu.single_game import ModelParams, predict_match  # noqa: E402
from mc_simu.validate_phase1 import (  # noqa: E402
    RESULTS_CSV, TOURNAMENT_META, _brier_mean, _make_context, _outcome_vec,
)

DATA = PROJECT_ROOT / "data" / "mc_simu"
HFA_A, HFA_B = 0.27, 0.09
D_GRID = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800]
DIAG_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]
# (Kaggle tournament name, year, display label)
EDITIONS = (
    [("FIFA World Cup", y, f"WC{y}") for y in (2002, 2006, 2010, 2014, 2018, 2022)]
    + [("UEFA Euro", y, f"Euro{lab}") for y, lab in
       [(2004, 2004), (2008, 2008), (2012, 2012), (2021, 2020), (2024, 2024)]]
)


def _eloratings_asof():
    df = pd.read_csv(DATA / "elo_history_eloratings.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    by_team = {t: (g["date"].to_numpy(), g["rating_after"].to_numpy())
               for t, g in df.groupby("team")}
    alias = NAME_ALIASES_BY_SOURCE["eloratings"]

    def asof(team, date):
        t = alias.get(team, team)
        if t not in by_team:
            return None
        dates, rats = by_team[t]
        i = np.searchsorted(dates, np.datetime64(date)) - 1
        return float(rats[i]) if i >= 0 else None
    return asof


def main() -> int:
    df = pd.read_csv(RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    tmeta = json.load(open(TOURNAMENT_META))
    asof = _eloratings_asof()

    cache, labels, fb = [], [], 0
    for tname, year, lab in EDITIONS:
        sub = df[(df["tournament"] == tname) & (df["year"] == year)]
        sub = sub[sub["home_score"].notna() & sub["away_score"].notna()]
        for row in sub.itertuples(index=False):
            r = row._asdict()
            eh, ea = asof(r["home_team"], r["date"]), asof(r["away_team"], r["date"])
            if eh is None or ea is None:
                eh, ea = eh or 1500.0, ea or 1500.0
                fb += 1
            meta_lab = {"FIFA World Cup": f"FIFA World Cup {year}",
                        "UEFA Euro": f"UEFA Euro {2020 if year == 2021 else year}"}[tname]
            ctx = _make_context(r, meta_lab, tmeta)
            cache.append({"eh": eh, "ea": ea, "ctx": ctx,
                          "actual": _outcome_vec(r["home_score"], r["away_score"])})
            labels.append(lab)
    labels = np.array(labels)
    n = len(cache)
    banner(f"Loaded {n} finals matches across {len(set(labels))} editions "
           f"(eloratings fallback on {fb} team-lookups)")

    # per-match mean-Brier for every (D, diag) cell
    cells = [(D, d) for D in D_GRID for d in DIAG_GRID]
    B = {}
    for D, d in cells:
        sg.ELO_GOALS_DENOMINATOR = float(D)
        p = ModelParams(alpha=HFA_A, beta=HFA_B, diagonal_inflation=d, blend_match_weight=1.0)
        B[(D, d)] = np.array([_brier_mean(predict_match(e["eh"], e["ea"], e["ctx"], p), e["actual"])
                              for e in cache])

    # pooled optimum + current
    pooled = {c: B[c].mean() for c in cells}
    best_pool = min(pooled, key=pooled.get)
    banner("Pooled (in-sample, all 11 editions)")
    print(f"  current (D=1400, diag=0.20):  Brier = {pooled[(1400,0.20)]:.4f}")
    print(f"  pooled-optimal D={best_pool[0]}, diag={best_pool[1]}:  Brier = {pooled[best_pool]:.4f}")

    # LOTO-CV
    banner("LOTO-CV — leave-one-tournament-out (out-of-sample)")
    print(f"{'Held-out':<9}{'n':>4}{'pick D':>8}{'pick diag':>10}{'OOS Brier':>11}{'@1400/.20':>11}")
    oos_best, oos_cur, picks_D, picks_d = [], [], [], []
    for lab in sorted(set(labels)):
        out = labels == lab
        inn = ~out
        train = {c: B[c][inn].mean() for c in cells}
        pick = min(train, key=train.get)
        oos = B[pick][out].mean()
        cur = B[(1400, 0.20)][out].mean()
        oos_best.append((oos, out.sum())); oos_cur.append((cur, out.sum()))
        picks_D.append(pick[0]); picks_d.append(pick[1])
        print(f"{lab:<9}{int(out.sum()):>4}{pick[0]:>8}{pick[1]:>10}{oos:>11.4f}{cur:>11.4f}")

    wm = lambda lst: sum(v * w for v, w in lst) / sum(w for _, w in lst)
    print("-" * 53)
    print(f"{'POOLED OOS':<23}{wm(oos_best):>22.4f}{wm(oos_cur):>11.4f}")
    print(f"\n  Stability of LOTO pick:  D {sorted(set(picks_D))} (mode {max(set(picks_D),key=picks_D.count)})"
          f"   diag {sorted(set(picks_d))} (mode {max(set(picks_d),key=picks_d.count)})")
    print(f"  OOS Brier:  LOTO-tuned {wm(oos_best):.4f}  vs  fixed 1400/0.20 {wm(oos_cur):.4f}  "
          f"(Δ {wm(oos_best)-wm(oos_cur):+.4f})")
    return 0


__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
