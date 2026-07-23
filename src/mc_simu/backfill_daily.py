"""Backfill one lost daily_log day (champion series) after the fact.

run_daily.py snapshots the LIVE market, so a day the cron lost (2026-06-29 died
behind the pre-pipefail tee) cannot be re-run as-is. This rebuilds that day:

    - model side: the same production + mle sims, conditioned on an as-of played
      state (build it first with `fetch_played --until <cutoff> --csv <tmp>`).
    - market side: venue history endpoints (backfill_market fetchers), taking
      end-of-day mids for --market-day (default: the day before --date — what
      the ~08:30 UTC live fetch would have seen, no matches run overnight).

Rows go through run_daily.snapshot_rows + push_supabase so schema and
normalization semantics match live rows exactly. Polymarket history carries no
book -> pm_bid/pm_ask stay blank; Kalshi bid/ask come from candlestick closes.

    PYTHONPATH=src python -m mc_simu.fetch_played --until 2026-06-29T06:00:00Z --csv asof.csv
    PYTHONPATH=src python -m mc_simu.backfill_daily --date 2026-06-29 --played-csv asof.csv --dry-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from mc_simu._common import banner  # noqa: E402
from mc_simu.backfill_market import fetch_kalshi_history, fetch_poly_history  # noqa: E402
from mc_simu.run_daily import (  # noqa: E402
    _teams48, mle_champion_probs, model_champion_probs, push_supabase,
    snapshot_rows,
)
from mc_simu.tournaments.wc2026 import load_played_results  # noqa: E402
from mc_simu.tune_to_market import normalize  # noqa: E402


def historical_market(day: str) -> dict[str, dict[str, float]]:
    """live_market()-shaped dict from venue history for one UTC day."""
    d = _dt.date.fromisoformat(day)
    start_ts = int(_dt.datetime.combine(d - _dt.timedelta(days=1), _dt.time(),
                                        tzinfo=_dt.timezone.utc).timestamp())
    end_ts = int(_dt.datetime.combine(d + _dt.timedelta(days=2), _dt.time(),
                                      tzinfo=_dt.timezone.utc).timestamp())
    poly = fetch_poly_history(start_ts, end_ts).get(day, {})
    kalshi = fetch_kalshi_history(start_ts, end_ts).get(day, {})
    out: dict[str, dict[str, float]] = {}
    for t in set(poly) | set(kalshi):
        pm = poly.get(t)
        kal = (kalshi.get(t) or {}).get("mid")
        vals = [v for v in (pm, kal) if v is not None]
        if not vals:
            continue
        out[t] = {"pm": pm, "kalshi": kal, "consensus": float(np.median(vals)),
                  "pm_bid": None, "pm_ask": None,
                  "kalshi_bid": (kalshi.get(t) or {}).get("bid"),
                  "kalshi_ask": (kalshi.get(t) or {}).get("ask")}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True, help="daily_log date to backfill")
    p.add_argument("--played-csv", type=Path, required=True,
                   help="as-of played state (from fetch_played --until)")
    p.add_argument("--market-day", default=None,
                   help="UTC day whose end-of-day prices stand in for the "
                        "morning-of --date live fetch (default: date - 1)")
    p.add_argument("--n", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    mkt_day = args.market_day or (
        _dt.date.fromisoformat(args.date) - _dt.timedelta(days=1)).isoformat()
    banner(f"Backfill daily_log {args.date} (market close {mkt_day}, "
           f"state {args.played_csv.name})")

    market = historical_market(mkt_day)
    if not market:
        print("no historical market rows for that day — aborting, nothing pushed")
        return 1
    print(f"market: {len(market)} teams quoted on {mkt_day}")

    played = load_played_results(args.played_csv)
    print(f"conditioning on {len(played.group_scores)} group + "
          f"{len(played.ko_winners)} KO results")
    model = model_champion_probs(played, args.n, args.seed)
    try:
        mle = mle_champion_probs(played, args.n, args.seed)
    except Exception as e:
        print(f"mle_strength run FAILED (mle/pool columns blank): {e}")
        mle = None

    common = [t for t in _teams48() if t in market]
    mkt_cons = normalize({t: market[t]["consensus"] for t in common})
    mdl = normalize({t: model.get(t, 0.0) for t in common})
    mle_n = normalize({t: mle.get(t, 0.0) for t in common}) if mle is not None else None
    pool = (normalize({t: (mdl[t] * mle_n[t]) ** 0.5 for t in common})
            if mle_n is not None else None)
    rows = snapshot_rows(args.date, common, mdl, mkt_cons, market, mle_n, pool)

    print(f"{len(rows)} rows for {args.date}; largest model probabilities:")
    for r in rows[:6]:
        print(f"  {r['team']:<16}model={r['model_pct']:>7.2f}%  "
              f"consensus={r['consensus_pct']:>6.2f}%")
    if args.dry_run:
        print("dry run — not pushing")
        return 0
    if push_supabase(rows):
        print(f"Upserted {len(rows)} rows to Supabase daily_log")
        return 0
    print("SUPABASE_URL/SUPABASE_SERVICE_KEY not set — nothing pushed")
    return 1


__all__ = ["historical_market", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
