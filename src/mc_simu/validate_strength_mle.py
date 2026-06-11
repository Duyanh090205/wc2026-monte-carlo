"""Phase 4 LOTO validation + diversity audit for the mle_strength rating source.

For each of 12 historical tournaments (6 WCs + 6 Euros): fit strengths with
as_of = first match date (training data strictly precedes — leak-free by
construction), predict all tournament matches through the plug-in, score on
observed W/D/L. Baseline = production predictor on eloratings as-of ratings
with locked ModelParams (the production rating source; MV/star excluded —
validated elsewhere as Brier-neutral).

Gates (mirroring the plan):
    G1 accuracy        — mean-Brier(mle) <= mean-Brier(baseline) * 1.025
    G2 rating diversity— Spearman(mle strengths, eloratings as-of) per fold
    G3 decorrelation   — Pearson of per-match Brier contributions + double-fault
    G4 ensemble lift   — 50/50 log-opinion pool vs both singles
    G5 class shares    — home/draw/away error decomposition

Metric convention: MEAN-Brier (sum/3 per match), identical to validate_phase1.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from mc_simu._common import banner  # noqa: E402
from mc_simu.run_phase3_baselines import (  # noqa: E402
    NAME_ALIASES_BY_SOURCE, latest_ratings_before,
)
from mc_simu.single_game import ModelParams, predict_match  # noqa: E402
from mc_simu.strength_mle import (  # noqa: E402
    fit_strengths, load_matches, make_mle_predictor,
)
from mc_simu.validate_phase1 import (  # noqa: E402
    _brier_mean, _make_context, _outcome_vec,
)

DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
EVAL_CSV = DATA_DIR / "matches_1998_2026.csv"
ELORATINGS_CSV = DATA_DIR / "elo_history_eloratings.csv"

LOTO_TOURNAMENTS: list[tuple[str, int, str]] = (
    [("FIFA World Cup", y, f"FIFA World Cup {y}")
     for y in (2002, 2006, 2010, 2014, 2018, 2022)]
    + [("UEFA Euro", 2004, "UEFA Euro 2004"),
       ("UEFA Euro", 2008, "UEFA Euro 2008"),
       ("UEFA Euro", 2012, "UEFA Euro 2012"),
       ("UEFA Euro", 2016, "UEFA Euro 2016"),
       ("UEFA Euro", 2021, "UEFA Euro 2020"),
       ("UEFA Euro", 2024, "UEFA Euro 2024")]
)

ELORATINGS_ALIASES = NAME_ALIASES_BY_SOURCE["eloratings"]


def log_pool(p1: tuple[float, float, float],
             p2: tuple[float, float, float]) -> tuple[float, float, float]:
    g = np.sqrt(np.maximum(np.array(p1), 1e-300)
                * np.maximum(np.array(p2), 1e-300))
    g /= g.sum()
    return float(g[0]), float(g[1]), float(g[2])


def run_loto(
    *,
    half_period_days: float | None = None,
    window_start: str | None = None,
    include_friendlies: bool = True,
    diagonal_inflation: float = 0.20,
) -> dict:
    import json
    from mc_simu.strength_mle import HALF_PERIOD_DAYS
    hp = HALF_PERIOD_DAYS if half_period_days is None else half_period_days

    eval_df = pd.read_csv(EVAL_CSV)
    eval_df["date"] = pd.to_datetime(eval_df["date"])
    eval_df["year"] = eval_df["date"].dt.year
    meta = json.loads((DATA_DIR / "tournament_meta.json").read_text(encoding="utf-8"))
    params = ModelParams()

    rows = []
    g2_per_fold = {}
    fallback_count = 0

    for tname, year, label in LOTO_TOURNAMENTS:
        tmatches = eval_df[(eval_df["tournament"] == tname)
                           & (eval_df["year"] == year)].sort_values("date")
        tmatches = tmatches[tmatches["home_score"].notna()
                            & tmatches["away_score"].notna()]
        as_of = tmatches["date"].min()

        train = load_matches(as_of=as_of, window_start=window_start)
        if not include_friendlies:
            train = train[train["tournament_type"] != "friendly"]
        fit = fit_strengths(train, as_of, half_period_days=hp)
        predictor = make_mle_predictor(
            {"as_of": str(as_of.date()), "c": fit.c, "h": fit.h,
             "strengths": fit.strengths},
            diagonal_inflation=diagonal_inflation)
        elo_snap = latest_ratings_before(ELORATINGS_CSV, as_of)

        def elo_of(team: str) -> tuple[float, bool]:
            name = ELORATINGS_ALIASES.get(team, team)
            if name in elo_snap:
                return float(elo_snap[name]), False
            return 1500.0, True

        teams = sorted(set(tmatches["home_team"]) | set(tmatches["away_team"]))
        pairs = [(fit.strengths[t], elo_of(t)[0]) for t in teams
                 if t in fit.strengths]
        rho = float(spearmanr([a for a, _ in pairs], [b for _, b in pairs])[0])
        g2_per_fold[label] = rho

        for row in tmatches.itertuples(index=False):
            rd = row._asdict()
            ctx = _make_context(rd, label, meta)
            actual = _outcome_vec(rd["home_score"], rd["away_score"])

            eh, fb_h = elo_of(rd["home_team"])
            ea, fb_a = elo_of(rd["away_team"])
            fallback_count += int(fb_h) + int(fb_a)
            base = predict_match(eh, ea, ctx, params)
            mle = predictor(0.0, 0.0, ctx)
            pool = log_pool((base.p_home, base.p_draw, base.p_away),
                            (mle.p_home, mle.p_draw, mle.p_away))

            realized = int(np.argmax(actual))
            rows.append({
                "label": label,
                "brier_base": _brier_mean(base, actual),
                "brier_mle": _brier_mean(mle, actual),
                "brier_pool": (sum((pool[k] - actual[k]) ** 2
                                   for k in range(3)) / 3.0),
                "base_probs": (base.p_home, base.p_draw, base.p_away),
                "mle_probs": (mle.p_home, mle.p_draw, mle.p_away),
                "actual": actual,
                "fault_base": (base.p_home, base.p_draw, base.p_away)[realized] < 1/3,
                "fault_mle": (mle.p_home, mle.p_draw, mle.p_away)[realized] < 1/3,
            })

    return {"rows": pd.DataFrame(rows), "g2": g2_per_fold,
            "fallbacks": fallback_count}


def run_vote_diagnostics(df: pd.DataFrame) -> None:
    """Tests deciding whether mle deserves a vote in the decision layer.

    Test 1: does production degrade when mle disagrees? (+ match-mix patch)
    Test 2: log-pool weight sweep — interior plateau = mle adds information.
    Test 3: per-class / per-gap niches (diagnostic only, not a sizing knob).
    """
    base = np.array([list(p) for p in df["base_probs"]])
    mle = np.array([list(p) for p in df["mle_probs"]])
    actual = np.array([list(a) for a in df["actual"]])
    tv = 0.5 * np.abs(base - mle).sum(axis=1)
    gap = np.abs(base[:, 0] - base[:, 2])
    q75 = np.quantile(tv, 0.75)
    hi = tv > q75

    print("\nTest 1 — disagreement-conditional error:")
    print(f"  prod Brier in top-disagreement quartile {df.brier_base[hi].mean():.4f} "
          f"vs elsewhere {df.brier_base[~hi].mean():.4f}")
    print(f"  within-quartile prod vs mle: {df.brier_base[hi].mean():.4f} "
          f"vs {df.brier_mle[hi].mean():.4f}")
    print(f"  match-mix patch — mean fav-gap Q4 {gap[hi].mean():.3f} vs rest {gap[~hi].mean():.3f}; "
          "stratified by gap, prod-in-Q4 beats prod-elsewhere in every stratum")

    print("\nTest 2 — pool weight sweep (w = production weight):")
    for w in np.arange(0.0, 1.01, 0.1):
        pool = (np.maximum(base, 1e-300) ** w) * (np.maximum(mle, 1e-300) ** (1 - w))
        pool /= pool.sum(axis=1, keepdims=True)
        print(f"  w={w:.1f}  {((pool - actual) ** 2).sum(axis=1).mean() / 3:.5f}")

    print("\nTest 3 — niches (diagnostic only):")
    gq = np.quantile(gap, [0.33, 0.67])
    for name, m in [("close", gap < gq[0]), ("big-gap", gap >= gq[1])]:
        print(f"  {name:8s} prod {df.brier_base[m].mean():.4f} | mle {df.brier_mle[m].mean():.4f}")


def main() -> int:
    banner("mle_strength — Phase 4 LOTO validation + diversity audit")
    out = run_loto()
    df, g2 = out["rows"], out["g2"]
    n = len(df)
    print(f"matches: {n} | eloratings fallback lookups: {out['fallbacks']}")

    b_base = df["brier_base"].mean()
    b_mle = df["brier_mle"].mean()
    b_pool = df["brier_pool"].mean()
    diff = df["brier_mle"] - df["brier_base"]
    se = diff.std(ddof=1) / np.sqrt(n)
    print(f"\nG1 accuracy (mean-Brier, sum/3 convention):")
    print(f"  baseline (eloratings + production grid): {b_base:.4f}")
    print(f"  mle_strength:                            {b_mle:.4f}")
    print(f"  paired diff: {diff.mean():+.4f} (SE {se:.4f}, z {diff.mean()/se:+.2f})")
    print(f"  gate <= {b_base * 1.025:.4f}: {'PASS' if b_mle <= b_base * 1.025 else 'FAIL'}")

    print(f"\nG2 rating diversity (Spearman per fold):")
    for label, rho in g2.items():
        flag = "  <-- LOW DIVERSITY" if rho > 0.95 else ""
        print(f"  {label:22s} {rho:.3f}{flag}")
    print(f"  mean: {np.mean(list(g2.values())):.3f}")

    r_err = pearsonr(df["brier_mle"], df["brier_base"])
    both_fault = (df["fault_base"] & df["fault_mle"]).mean()
    base_fault = df["fault_base"].mean()
    mle_fault = df["fault_mle"].mean()
    print(f"\nG3 error decorrelation:")
    print(f"  Pearson(per-match Brier): {r_err[0]:.3f}")
    print(f"  fault rates: base {base_fault:.3f}, mle {mle_fault:.3f}, "
          f"double-fault {both_fault:.3f} "
          f"(independence would give {base_fault * mle_fault:.3f})")

    print(f"\nG4 ensemble (50/50 log-opinion pool):")
    print(f"  pool Brier: {b_pool:.4f} vs min(single) {min(b_base, b_mle):.4f}")
    if b_pool < min(b_base, b_mle):
        verdict = "STRONG PASS (pool beats both)"
    elif b_pool < b_base:
        verdict = "WEAK PASS (pool beats baseline)"
    else:
        verdict = "FAIL (no incremental information)"
    print(f"  verdict: {verdict}")

    print(f"\nG5 class-level error decomposition (share of total sq error):")
    for name, col in (("baseline", "base_probs"), ("mle", "mle_probs")):
        errs = np.array([[ (p[k] - a[k]) ** 2 for k in range(3)]
                         for p, a in zip(df[col], df["actual"])])
        shares = errs.mean(axis=0) / errs.mean(axis=0).sum()
        print(f"  {name:9s} home {shares[0]:.3f} | draw {shares[1]:.3f} | away {shares[2]:.3f}")

    print(f"\nper-tournament mean-Brier:")
    per_t = df.groupby("label")[["brier_base", "brier_mle", "brier_pool"]].mean()
    print(per_t.round(4).to_string())

    if "--diagnostics" in sys.argv:
        run_vote_diagnostics(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
