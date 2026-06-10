"""Does adding squad market value improve historical match prediction vs Elo-only?

The clean, direct validation of the MV blend that we previously could NOT run
(no historical MV existed). Now that build_historical_mv.py produces point-in-time
squad MV per WC edition, we test it the disciplined way — vs realized match
outcomes, NOT vs market (per CLAUDE.md "never tune against market").

Method (per-match Brier, the same single_game model both ways, only the input
ratings differ):
  for each WC edition 2010/14/18/22:
    elo[t]      = latest eloratings rating before kickoff
    blended[t]  = z-score blend(Elo, log squad-MV), alpha (default 0.5) — same as
                  the WC2026 production model
    for each match (neutral ctx, HFA cancels):
      p_elo  = predict_match(elo[A], elo[B])
      p_mv   = predict_match(blended[A], blended[B])
    score 3-way Brier of each vs actual W/D/L (regulation goals).
  Lower mean Brier = better predictor. Paired diff + win-rate reported.

Usage:
    PYTHONPATH=src python -m mc_simu.validate_mv_history
    PYTHONPATH=src python -m mc_simu.validate_mv_history --alpha 0.5 0.3 0.7
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
from mc_simu.build_historical_mv import SEASON_TO_WC  # noqa: E402
from mc_simu.mv_blend import blend_elo_with_mv  # noqa: E402
from mc_simu.single_game import MatchContext, ModelParams, predict_match  # noqa: E402

DATA = PROJECT_ROOT / "data" / "mc_simu"


def _neutral_ctx(a: str, b: str) -> MatchContext:
    """Neutral venue: no host -> HFA bonus 0 for both, so prediction is pure rating."""
    return MatchContext(
        is_neutral=True, tournament_type="world_cup",
        home_country=a, away_country=b, venue_country="",
        venue_confederation="UNKNOWN", home_confederation="UNKNOWN",
        away_confederation="UNKNOWN", host_countries=[], host_confederation=None,
        attendance_pct=1.0,
    )


def _elo_before(elo: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, float]:
    sub = elo[elo["date"] < cutoff].sort_values("date")
    return sub.groupby("team").tail(1).set_index("team")["rating_after"].to_dict()


def _brier(p: tuple[float, float, float], outcome: int) -> float:
    a = [0.0, 0.0, 0.0]
    a[outcome] = 1.0
    return sum((p[k] - a[k]) ** 2 for k in range(3))


def run(alpha: float, params: ModelParams) -> dict:
    elo = pd.read_csv(DATA / "elo_history_eloratings.csv")
    elo["date"] = pd.to_datetime(elo["date"])
    mv_all = pd.read_csv(DATA / "historical_squad_mv.csv")
    games = pd.read_csv(DATA / "cache" / "tmdata" / "games.csv.gz", low_memory=False)
    games["date"] = pd.to_datetime(games["date"])

    per_ed: list[dict] = []
    diffs: list[float] = []
    skipped = 0
    for season, wc_id in sorted(SEASON_TO_WC.items()):
        ed_games = games[(games["competition_id"] == "FIWC") & (games["season"] == season)]
        if ed_games.empty:
            continue
        wc = f"WC{wc_id.split('-')[1]}"
        mv_ed = mv_all[mv_all["edition"] == wc]
        if mv_ed.empty:
            continue
        cutoff = ed_games["date"].min()
        elo_d = _elo_before(elo, cutoff)
        mv_d = {r["team"]: float(r["squad_mv_eur"]) for _, r in mv_ed.iterrows()
                if r["squad_mv_eur"] > 0}
        # blend over this edition's field (same z-score blend as production)
        field = {t: elo_d[t] for t in mv_d if t in elo_d}
        blended, _ = blend_elo_with_mv(field, mv_d, alpha=alpha)

        b_elo, b_mv, n = 0.0, 0.0, 0
        for _, g in ed_games.iterrows():
            a, b = g["home_club_name"], g["away_club_name"]
            if a not in field or b not in field:
                skipped += 1
                continue
            gh, ga = g["home_club_goals"], g["away_club_goals"]
            if pd.isna(gh) or pd.isna(ga):
                skipped += 1
                continue
            outcome = 0 if gh > ga else (1 if gh == ga else 2)
            ctx = _neutral_ctx(a, b)
            pe = predict_match(field[a], field[b], ctx, params)
            pm = predict_match(blended[a], blended[b], ctx, params)
            be = _brier((pe.p_home, pe.p_draw, pe.p_away), outcome)
            bm = _brier((pm.p_home, pm.p_draw, pm.p_away), outcome)
            b_elo += be
            b_mv += bm
            diffs.append(bm - be)
            n += 1
        per_ed.append({"edition": wc, "n": n,
                       "brier_elo": b_elo / n, "brier_mv": b_mv / n,
                       "delta": (b_mv - b_elo) / n})
    d = np.array(diffs)
    return {"per_ed": per_ed, "n_total": len(d), "skipped": skipped,
            "brier_elo": float(np.mean([p["brier_elo"] * p["n"] for p in per_ed]) / max(1, len(d))) if per_ed else None,
            "mean_delta": float(d.mean()), "sd_delta": float(d.std(ddof=1)),
            "se_delta": float(d.std(ddof=1) / np.sqrt(len(d))),
            "pct_mv_better": float((d < 0).mean() * 100)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, nargs="+", default=[0.5],
                   help="MV blend weight(s) to test (production = 0.5)")
    p.add_argument("--D", type=float, default=1400.0)
    p.add_argument("--diag", type=float, default=0.20)
    args = p.parse_args(argv)

    import mc_simu.single_game as sg
    sg.ELO_GOALS_DENOMINATOR = args.D
    params = ModelParams(diagonal_inflation=args.diag)

    for alpha in args.alpha:
        res = run(alpha, params)
        banner(f"MV validation — alpha={alpha}, D={args.D:.0f}, diag={args.diag}")
        print(f"{'Edition':<10} {'n':>4} {'Brier ELO':>11} {'Brier ELO+MV':>13} {'Δ (mv-elo)':>12}")
        tot_elo = tot_mv = tot_n = 0
        for e in res["per_ed"]:
            mark = "  MV better" if e["delta"] < 0 else "  ELO better"
            print(f"{e['edition']:<10} {e['n']:>4} {e['brier_elo']:>11.4f} "
                  f"{e['brier_mv']:>13.4f} {e['delta']:>+12.4f}{mark}")
            tot_elo += e["brier_elo"] * e["n"]; tot_mv += e["brier_mv"] * e["n"]; tot_n += e["n"]
        print("-" * 56)
        print(f"{'POOLED':<10} {tot_n:>4} {tot_elo/tot_n:>11.4f} {tot_mv/tot_n:>13.4f} "
              f"{(tot_mv-tot_elo)/tot_n:>+12.4f}")
        print(f"\n  matches used: {res['n_total']} (skipped {res['skipped']})")
        print(f"  mean ΔBrier (MV−ELO) = {res['mean_delta']:+.5f}  ± {res['se_delta']:.5f} SE  "
              f"(− = MV better)")
        z = res["mean_delta"] / res["se_delta"] if res["se_delta"] else 0.0
        print(f"  paired z ≈ {z:+.2f}  |  MV beats ELO on {res['pct_mv_better']:.0f}% of matches")
        verdict = ("MV IMPROVES" if res["mean_delta"] < 0 and abs(z) > 1.96 else
                   "MV HURTS" if res["mean_delta"] > 0 and abs(z) > 1.96 else
                   "NO SIGNIFICANT DIFFERENCE")
        print(f"  => {verdict} (per-match Brier, n={res['n_total']})")
    return 0


__all__ = ["run", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
