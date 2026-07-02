"""Backfill per-day KNOCKOUT-REACH series: model vs Polymarket, back to --start.

Same shape as backfill_group_winner but for the four knockout-reach rounds.
For each round in (r16, qf, sf, final), two lines per team:

    - model: P(team enters that round) = P(wins the feeder tie), re-simulated
      on the as-of played state (git history of wc2026_played.csv — exact for
      committed days, forward-filled between commits).
    - Polymarket: world-cup-nation-to-reach-<round> markets, devigged by
      scaling the round's quotes to its slot count (16/8/4/2).

Kalshi has no per-round markets and FairLine is snapshot-only — both absent.
Git states carry the KO rows too, so post-group-stage days condition on real
knockout results, matching what the daily cron logged.

    PYTHONPATH=src python -m mc_simu.backfill_reach
    PYTHONPATH=src python -m mc_simu.backfill_reach --no-model   # Poly only
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mc_simu.single_game as sg  # noqa: E402
from mc_simu._common import banner  # noqa: E402
from mc_simu.backfill_group_winner import (  # noqa: E402
    _daterange, _day, _git_played_states,
)
from mc_simu.run_daily import REACH_SLOTS, REACH_SLUGS  # noqa: E402
from mc_simu.single_game import ModelParams  # noqa: E402
from mc_simu.tournaments.wc2026 import REACH_ROUNDS  # noqa: E402
from mc_simu.tune_to_market import _norm  # noqa: E402

DATA = PROJECT_ROOT / "data" / "mc_simu"
POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_CLOB = "https://clob.polymarket.com"

OUT_COLS = ["date", "round", "team", "model_pct", "model_state_matches",
            "pm_raw_pct", "pm_devig_pct"]


# ── Model side ────────────────────────────────────────────────────────────────


def model_reach(played_csv: Path, n: int, seed: int) -> dict[str, dict[str, float]]:
    """{round: {team: P(reach)}} on a given played state."""
    from mc_simu.run_daily import production_ratings
    from mc_simu.tournaments.wc2026 import (
        load_played_results, load_wc2026_bundle, run_monte_carlo,
    )
    sg.ELO_GOALS_DENOMINATOR = 1400.0
    played = load_played_results(played_csv)
    bundle = load_wc2026_bundle(production_ratings(),
                                params=ModelParams(diagonal_inflation=0.20))
    res = run_monte_carlo(bundle, n_iterations=n, seed=seed, progress=False, played=played)
    return {rnd: {t: s["mc_fair_prob"] for t, s in teams.items()}
            for rnd, teams in res["reach"].items()}


def _n_matches(txt: str) -> int:
    return sum(1 for ln in txt.splitlines() if ln.startswith(("group,", "ko,")))


def model_series(dates: list[str], n: int, seed: int,
                 states: list[tuple[str, str]]) -> dict[str, dict]:
    """{date: {'matches': int, 'reach': {round: {team: prob}}}} forward-filled."""
    empty = "stage,match_id,home_team,away_team,home_goals,away_goals,winner\n"
    cache: dict[str, dict] = {}

    def sim(txt: str) -> dict:
        key = hashlib.sha256(txt.encode()).hexdigest()
        if key not in cache:
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                             encoding="utf-8") as f:
                f.write(txt)
                tmp = Path(f.name)
            try:
                cache[key] = model_reach(tmp, n, seed)
            finally:
                tmp.unlink(missing_ok=True)
        return cache[key]

    out = {}
    for d in dates:
        usable = [(cd, txt) for cd, txt in states if cd <= d]
        txt = usable[-1][1] if usable else empty
        out[d] = {"matches": _n_matches(txt), "reach": sim(txt)}
        print(f"  model {d}: state={_n_matches(txt)} matches"
              + (" (carried fwd)" if usable and usable[-1][0] != d else ""))
    return out


# ── Polymarket side ───────────────────────────────────────────────────────────


def fetch_poly_reach(timeout: int = 30,
                     pause: float = 0.12) -> dict[str, dict[str, dict[str, float]]]:
    """{date: {round: {team: raw_price}}} from world-cup-nation-to-reach-*.

    interval=max instead of an explicit startTs/endTs window: the CLOB rejects
    windows past ~2 weeks ("interval is too long"); max returns the market's
    whole daily history and the caller date-filters. Dates outside [start, end]
    are dropped in main().
    """
    import requests
    out: dict[str, dict[str, dict[str, float]]] = {}
    for rnd, slug in REACH_SLUGS.items():
        ev = requests.get(f"{POLY_GAMMA}/events/slug/{slug}", timeout=timeout)
        if ev.status_code != 200:
            print(f"  Poly reach {rnd}: {ev.status_code} — skipped")
            continue
        n_pts = 0
        for m in ev.json().get("markets", []):
            team = _norm(m.get("groupItemTitle") or "")
            toks = m.get("clobTokenIds")
            if not team or not toks:
                continue
            try:
                yes_tok = json.loads(toks)[0]
            except (json.JSONDecodeError, IndexError):
                continue
            r = requests.get(f"{POLY_CLOB}/prices-history",
                             params={"market": yes_tok, "interval": "max",
                                     "fidelity": 1440}, timeout=timeout)
            if r.status_code != 200:
                print(f"  Poly reach {rnd}/{team}: prices-history {r.status_code}")
                continue
            for p in r.json().get("history", []):
                t, px = p.get("t"), p.get("p")
                if t is None or px is None or not (0 < float(px) <= 1):
                    continue
                # Same day-shift as backfill_group_winner: the fidelity=1440
                # point at D 00:00 is the close of D-1.
                out.setdefault(_day(t - 86400), {}).setdefault(rnd, {})[team] = float(px)
                n_pts += 1
            time.sleep(pause)
        print(f"  Poly reach {rnd}: {n_pts} points")
    return out


def _devig_slots(raw: dict[str, float], slots: int) -> dict[str, float]:
    """Scale a round's quotes so they sum to the slot count; {} when quotes
    cover less than half the slots (resolved or partially-listed event)."""
    s = sum(raw.values())
    if s < slots * 0.5:
        return {}
    return {t: min(1.0, v * slots / s) for t, v in raw.items()}


def _pct(v) -> str | float:
    return round(v * 100, 3) if v else ""


# ── Assemble ──────────────────────────────────────────────────────────────────


def build_rows(model: dict, poly: dict, teams48: list[str]) -> list[dict]:
    rows = []
    for d in sorted(set(model) | set(poly)):
        for rnd in REACH_ROUNDS:
            mdl = model.get(d, {}).get("reach", {}).get(rnd, {})
            matches = model.get(d, {}).get("matches", "")
            pm_raw = {t: v for t, v in poly.get(d, {}).get(rnd, {}).items()
                      if t in teams48}
            pm_dv = _devig_slots(pm_raw, REACH_SLOTS[rnd])
            order = sorted(teams48, key=lambda t: -(mdl.get(t) or pm_dv.get(t) or 0))
            for t in order:
                if t not in mdl and t not in pm_raw:
                    continue
                rows.append({
                    "date": d, "round": rnd, "team": t,
                    # sim ran (mdl non-empty) -> explicit 0.0 for an eliminated
                    # team, vs "" = no model that day (see backfill_group_winner).
                    "model_pct": round(mdl.get(t, 0.0) * 100, 3) if mdl else "",
                    "model_state_matches": matches if mdl else "",
                    "pm_raw_pct": _pct(pm_raw.get(t)),
                    "pm_devig_pct": _pct(pm_dv.get(t)),
                })
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2026-06-10")
    p.add_argument("--end", default=_dt.date.today().isoformat())
    p.add_argument("--out", type=Path, default=DATA / "reach_log.csv")
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-model", action="store_true", help="Poly only (skip sim runs)")
    args = p.parse_args(argv)

    teams48 = [t for g in json.load(open(DATA / "wc2026_groups.json"))["groups"].values()
               for t in g]
    dates = _daterange(args.start, args.end)
    banner(f"Backfill reach {args.start} -> {args.end}")

    print("Polymarket reach history:")
    poly = fetch_poly_reach()

    model = {}
    if not args.no_model:
        print("Model reach per day (git as-of states):")
        model = model_series(dates, args.n, args.seed, _git_played_states())

    rows = [r for r in build_rows(model, poly, teams48)
            if args.start <= r["date"] <= args.end]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(rows)
    days = sorted({r["date"] for r in rows})
    banner(f"Wrote {len(rows)} rows ({len(days)} days {days[0]}..{days[-1]}) to {args.out}"
           if days else "No rows written")
    return 0


__all__ = ["model_reach", "model_series", "fetch_poly_reach", "_devig_slots",
           "build_rows", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
