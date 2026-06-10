"""Daily conditional rerun — production model (ELO+MV+star) vs LIVE market.

Each day of WC2026: lock real results played so far (conditioning harness),
re-sim the remainder with the STATIC production model, pull the live market,
and APPEND the model-vs-market snapshot to a time-series log. This log is the
raw material for the edge layer (characterize the model's stable bias vs market
-> flag days the divergence is anomalous). Model never re-trains; only the
bracket is conditioned on facts.

    PYTHONPATH=src python -m mc_simu.run_daily
    PYTHONPATH=src python -m mc_simu.run_daily --played-csv data/mc_simu/wc2026_played.csv --date 2026-06-20

Output (append-only, gitignored): data/mc_simu/daily_log.csv
    date, team, model_pct, pm_pct, kalshi_pct, consensus_pct, abs_pp, rel_pct
Pre-tournament (no/empty played-csv) -> unconditioned full sim.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mc_simu.single_game as sg  # noqa: E402
from mc_simu._common import banner  # noqa: E402
from mc_simu.mv_blend import blend_elo_with_mv, load_mv_table  # noqa: E402
from mc_simu.run_phase3_baselines import load_ratings_for_source  # noqa: E402
from mc_simu.single_game import ModelParams  # noqa: E402
from mc_simu.star_presence import apply_star_bonus, load_star_counts  # noqa: E402
from mc_simu.tournaments.wc2026 import (  # noqa: E402
    load_played_results, load_wc2026_bundle, run_monte_carlo,
)
from mc_simu.tune_to_market import jsd, normalize  # noqa: E402
from mc_simu.wc2026_vs_multi import load_prices_endpoint  # noqa: E402

DATA = PROJECT_ROOT / "data" / "mc_simu"
DEFAULT_API = "https://seal-app-yatxw.ondigitalocean.app/api"
LOG_COLS = ["date", "team", "model_pct", "pm_pct", "kalshi_pct",
            "consensus_pct", "abs_pp", "rel_pct"]


def push_supabase(rows: list[dict]) -> bool:
    """Upsert the day's rows into Supabase table `daily_log` (on date,team).

    Needs SUPABASE_URL + SUPABASE_SERVICE_KEY in the environment; silently
    skipped when absent (local runs keep the CSV log only).
    """
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return False
    import requests
    payload = [{c: (None if r[c] == "" else r[c]) for c in LOG_COLS} for r in rows]
    resp = requests.post(
        f"{url.rstrip('/')}/rest/v1/daily_log?on_conflict=date,team",
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "Prefer": "resolution=merge-duplicates,return=minimal"},
        json=payload, timeout=30,
    )
    resp.raise_for_status()
    return True


def _teams48():
    return [t for g in json.load(open(DATA / "wc2026_groups.json"))["groups"].values() for t in g]


def production_ratings() -> dict[str, float]:
    """ELO(eloratings) + MV blend alpha=0.5 + star X=15 — the locked production config."""
    teams48 = _teams48()
    ratings, _ = load_ratings_for_source("eloratings", pd.Timestamp.now(), DATA, teams48)
    elo_sub = {t: ratings[t] for t in teams48 if t in ratings}
    blended, _ = blend_elo_with_mv(elo_sub, load_mv_table(), alpha=0.5)
    ratings.update(blended)
    return apply_star_bonus(ratings, bonus_X=15.0, star_counts=load_star_counts())


def model_champion_probs(played, n: int, seed: int) -> dict[str, float]:
    sg.ELO_GOALS_DENOMINATOR = 1400.0
    bundle = load_wc2026_bundle(production_ratings(), params=ModelParams(diagonal_inflation=0.20))
    res = run_monte_carlo(bundle, n_iterations=n, seed=seed, progress=False, played=played)
    return normalize({t: s["mc_fair_prob"] for t, s in res["champion"].items()})


def live_market(api: str) -> dict[str, dict[str, float]]:
    """{team: {'pm':, 'kalshi':, 'consensus': median}} from FairLine /prices."""
    prices = load_prices_endpoint(api)
    out = {}
    for t, row in prices.items():
        pm, kal = row.get("Polymarket"), row.get("Kalshi")
        vals = [v for v in (pm, kal) if v is not None]
        if vals:
            out[t] = {"pm": pm, "kalshi": kal, "consensus": float(np.median(vals))}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default=_dt.date.today().isoformat())
    p.add_argument("--played-csv", type=Path, default=DATA / "wc2026_played.csv")
    p.add_argument("--log-csv", type=Path, default=DATA / "daily_log.csv")
    p.add_argument("--api-url", default=DEFAULT_API)
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    played = load_played_results(args.played_csv)
    n_locked = len(played.group_scores) + len(played.ko_winners)
    banner(f"Daily rerun {args.date} — conditioned on {n_locked} played matches "
           f"({'pre-tournament' if n_locked == 0 else args.played_csv.name})")

    model = model_champion_probs(played, args.n, args.seed)
    market = live_market(args.api_url)
    # log every WC2026 team that the market quotes — eliminated teams show model 0%
    # (clean time series: a knocked-out favorite reads 19%->0%, not "vanished").
    common = [t for t in _teams48() if t in market]
    mkt_cons = normalize({t: market[t]["consensus"] for t in common})
    mdl = normalize({t: model.get(t, 0.0) for t in common})

    rows = []
    for t in sorted(common, key=lambda x: -mdl[x]):
        m, c = mdl[t], mkt_cons[t]
        rows.append({
            "date": args.date, "team": t,
            "model_pct": round(m * 100, 3),
            "pm_pct": round(market[t]["pm"] * 100, 3) if market[t]["pm"] else "",
            "kalshi_pct": round(market[t]["kalshi"] * 100, 3) if market[t]["kalshi"] else "",
            "consensus_pct": round(c * 100, 3),
            "abs_pp": round((m - c) * 100, 3),
            "rel_pct": round((c - m) / m * 100, 1) if m > 0 else "",
        })

    args.log_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not args.log_csv.exists()
    with args.log_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"Appended {len(rows)} rows to {args.log_csv}")
    try:
        if push_supabase(rows):
            print(f"Upserted {len(rows)} rows to Supabase daily_log")
    except Exception as e:
        print(f"Supabase push FAILED (CSV log intact): {e}")

    # daily summary
    j = jsd(mdl, mkt_cons, common)
    l1 = sum(abs(mdl[t] - mkt_cons[t]) for t in common) * 100
    banner(f"Snapshot {args.date} — JSD={j:.4f}  L1={l1:.1f}pp vs market consensus")
    print(f"{'Team':<16}{'model%':>8}{'mkt%':>8}{'abs pp':>9}{'rel %':>8}")
    by_abs = sorted(rows, key=lambda r: -abs(r["abs_pp"]))
    for r in by_abs[:8]:
        rel = r["rel_pct"]
        rel_s = f"{rel:>+7.0f}%" if isinstance(rel, (int, float)) else f"{'—':>8}"
        print(f"{r['team']:<16}{r['model_pct']:>7.2f}%{r['consensus_pct']:>7.2f}%"
              f"{r['abs_pp']:>+8.2f}{rel_s}")
    return 0


__all__ = ["production_ratings", "model_champion_probs", "live_market", "push_supabase", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
