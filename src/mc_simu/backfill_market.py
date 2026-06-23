"""Backfill daily champion-market odds from Polymarket + Kalshi history.

The live daily tracker (run_daily.py) only ever captured a current snapshot, so
the model-vs-market series starts the day it first ran. Both venues, however,
expose per-day price history that reaches back to before kickoff:

    - Polymarket CLOB  /prices-history?market=<token>&fidelity=1440  (mid only)
    - Kalshi public    /series/<s>/markets/<ticker>/candlesticks      (bid/ask/last)

This pulls both back to --start (default the mc_simu tracking start, 2026-06-10),
maps every market to the wc2026_groups.json canonical name, and writes BOTH the
raw implied price (per-source overround intact, sums >100%) and the devigged
probability (per-source normalized to sum=100%, comparable to the model). Raw is
kept so the overround/liquidity is recoverable; devig can't be undone from raw
alone is false (it's just a divide) — but raw can't be rebuilt from devig, so we
store raw and derive devig.

Polymarket history carries no book, so pm bid/ask stay blank (mid only); Kalshi
candlesticks carry per-day bid/ask close.

    PYTHONPATH=src python -m mc_simu.backfill_market
    PYTHONPATH=src python -m mc_simu.backfill_market --start 2026-06-10 --out data/mc_simu/market_history.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from mc_simu._common import banner  # noqa: E402
from mc_simu.tune_to_market import _norm  # noqa: E402

DATA = PROJECT_ROOT / "data" / "mc_simu"

POLY_EVENT_SLUG = "world-cup-winner"
POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_CLOB = "https://clob.polymarket.com"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_EVENT = "KXMENWORLDCUP-26"
KALSHI_SERIES = "KXMENWORLDCUP"

OUT_COLS = ["date", "team",
            "pm_raw_pct", "pm_devig_pct",
            "kalshi_raw_pct", "kalshi_devig_pct",
            "kalshi_bid_pct", "kalshi_ask_pct",
            "consensus_pct"]


def _teams48() -> list[str]:
    g = json.load(open(DATA / "wc2026_groups.json"))["groups"]
    return [t for grp in g.values() for t in grp]


def _day(ts: int) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).date().isoformat()


def fetch_poly_history(start_ts: int, end_ts: int, timeout: int = 30,
                       pause: float = 0.15) -> dict[str, dict[str, float]]:
    """{date: {team: mid}} — last daily point per (date, team). Mid only."""
    import requests
    ev = requests.get(f"{POLY_GAMMA}/events/slug/{POLY_EVENT_SLUG}", timeout=timeout)
    ev.raise_for_status()
    markets = ev.json().get("markets", [])
    out: dict[str, dict[str, float]] = {}
    n = 0
    for m in markets:
        team = _norm(m.get("groupItemTitle") or "")
        toks = m.get("clobTokenIds")
        if not team or not toks:
            continue
        try:
            yes_tok = json.loads(toks)[0]
        except (json.JSONDecodeError, IndexError):
            continue
        r = requests.get(f"{POLY_CLOB}/prices-history",
                         params={"market": yes_tok, "startTs": start_ts,
                                 "endTs": end_ts, "fidelity": 1440},
                         timeout=timeout)
        if r.status_code != 200:
            continue
        for p in r.json().get("history", []):
            t, px = p.get("t"), p.get("p")
            if t is None or px is None or not (0 < float(px) < 1):
                continue
            # fidelity=1440 stamps ~00:00 UTC = that day's open = prior day's close;
            # shift back a full day so date D reflects end-of-D market (the D+1 00:00
            # point), aligning with results known by end of D. See backfill_group_winner.
            out.setdefault(_day(t - 86400), {})[team] = float(px)  # last point wins
        n += 1
        time.sleep(pause)
    print(f"  Polymarket: {n} markets, {len(out)} days")
    return out


def fetch_kalshi_history(start_ts: int, end_ts: int, timeout: int = 30,
                         pause: float = 0.15) -> dict[str, dict[str, dict]]:
    """{date: {team: {mid, bid, ask}}} from daily candlestick closes."""
    import requests
    mk = requests.get(f"{KALSHI_API}/markets",
                      params={"event_ticker": KALSHI_EVENT, "limit": 200},
                      timeout=timeout)
    mk.raise_for_status()
    out: dict[str, dict[str, dict]] = {}
    n = 0
    for m in mk.json().get("markets", []):
        team = _norm((m.get("yes_sub_title") or "").strip())
        ticker = m.get("ticker")
        if not team or team == "Poland" or not ticker:
            continue
        r = requests.get(
            f"{KALSHI_API}/series/{KALSHI_SERIES}/markets/{ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1440},
            timeout=timeout)
        if r.status_code != 200:
            continue
        for c in r.json().get("candlesticks", []):
            ts = c.get("end_period_ts")
            if ts is None:
                continue
            bid = _close(c.get("yes_bid"))
            ask = _close(c.get("yes_ask"))
            last = _close(c.get("price"))
            if bid and ask:
                mid = (bid + ask) / 2
            elif last and 0 < last < 1:
                mid = last
            elif ask:
                mid = ask
            else:
                continue
            if 0 < mid < 1:
                # end_period_ts stamps ~04:00 UTC = that day's open ≈ prior match
                # day's close; shift back a day so date D carries end-of-D market,
                # consistent with the Polymarket shift above.
                out.setdefault(_day(ts - 86400), {})[team] = {"mid": mid, "bid": bid, "ask": ask}
        n += 1
        time.sleep(pause)
    print(f"  Kalshi: {n} markets, {len(out)} days")
    return out


def _close(block) -> float | None:
    if not isinstance(block, dict):
        return None
    v = block.get("close_dollars")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _devig(raw: dict[str, float]) -> dict[str, float]:
    s = sum(raw.values())
    return {t: v / s for t, v in raw.items()} if s > 0 else {}


def build_rows(poly: dict, kalshi: dict, teams: list[str]) -> list[dict]:
    tset = set(teams)
    dates = sorted(set(poly) | set(kalshi))
    rows = []
    for d in dates:
        pm_raw = {t: v for t, v in poly.get(d, {}).items() if t in tset}
        kal_day = {t: q for t, q in kalshi.get(d, {}).items() if t in tset}
        kal_raw = {t: q["mid"] for t, q in kal_day.items()}
        pm_dv, kal_dv = _devig(pm_raw), _devig(kal_raw)
        cons_raw = {}
        for t in tset:
            vals = [x for x in (pm_dv.get(t), kal_dv.get(t)) if x is not None]
            if vals:
                cons_raw[t] = float(np.median(vals))
        cons = _devig(cons_raw)
        for t in sorted(tset, key=lambda x: -cons.get(x, 0.0)):
            if t not in pm_raw and t not in kal_raw:
                continue
            q = kal_day.get(t, {})
            rows.append({
                "date": d, "team": t,
                "pm_raw_pct": _pct(pm_raw.get(t)),
                "pm_devig_pct": _pct(pm_dv.get(t)),
                "kalshi_raw_pct": _pct(kal_raw.get(t)),
                "kalshi_devig_pct": _pct(kal_dv.get(t)),
                "kalshi_bid_pct": _pct(q.get("bid")),
                "kalshi_ask_pct": _pct(q.get("ask")),
                "consensus_pct": _pct(cons.get(t)),
            })
    return rows


def _pct(v) -> str | float:
    return round(v * 100, 3) if v else ""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2026-06-10", help="first UTC date to backfill")
    p.add_argument("--end", default=_dt.date.today().isoformat(), help="last UTC date (inclusive)")
    p.add_argument("--out", type=Path, default=DATA / "market_history.csv")
    args = p.parse_args(argv)

    start_ts = int(_dt.datetime.fromisoformat(args.start)
                   .replace(tzinfo=_dt.timezone.utc).timestamp())
    end_ts = int((_dt.datetime.fromisoformat(args.end) + _dt.timedelta(days=1))
                 .replace(tzinfo=_dt.timezone.utc).timestamp())
    banner(f"Backfill market history {args.start} -> {args.end}")

    poly = fetch_poly_history(start_ts, end_ts)
    kalshi = fetch_kalshi_history(start_ts, end_ts)
    rows = [r for r in build_rows(poly, kalshi, _teams48()) if args.start <= r["date"] <= args.end]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(rows)

    days = sorted({r["date"] for r in rows})
    banner(f"Wrote {len(rows)} rows ({len(days)} days {days[0]}..{days[-1]}) to {args.out}"
           if days else "No rows written")
    return 0


__all__ = ["fetch_poly_history", "fetch_kalshi_history", "build_rows", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
