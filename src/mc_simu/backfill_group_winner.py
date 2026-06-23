"""Backfill per-day GROUP-WINNER series: model vs Polymarket, back to --start.

The daily tracker logs champion only; group-winner is computed by the sim then
discarded. This reconstructs the group-winner time series for the one window it
is live (group stage, ~06-10..06-27 WC2026) and writes it to a dedicated log the
dashboard's Group-winner tab reads. Two lines per team:

    - model: P(team wins its group), re-simulated on the as-of played state.
    - Polymarket: world-cup-group-{a..l}-winner markets, devigged within group.

Kalshi has no group-winner market; FairLine is snapshot-only — both absent here.

As-of played state per date comes from git history of wc2026_played.csv (exact
for committed days) forward-filled to each date; `model_state_matches` records
how many group results the model line actually saw (frozen value flags a carried-
forward day). With FOOTBALL_DATA_TOKEN set, --from-api rebuilds the per-day state
from real match dates (utcDate) instead, closing any post-last-commit gap.

    PYTHONPATH=src python -m mc_simu.backfill_group_winner
    PYTHONPATH=src python -m mc_simu.backfill_group_winner --from-api   # needs token
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mc_simu.single_game as sg  # noqa: E402
from mc_simu._common import banner  # noqa: E402
from mc_simu.single_game import ModelParams  # noqa: E402
from mc_simu.tune_to_market import _norm  # noqa: E402

DATA = PROJECT_ROOT / "data" / "mc_simu"
PLAYED = DATA / "wc2026_played.csv"
POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_CLOB = "https://clob.polymarket.com"
GROUP_LETTERS = "ABCDEFGHIJKL"

OUT_COLS = ["date", "group", "team", "model_pct", "model_state_matches",
            "pm_raw_pct", "pm_devig_pct"]


def _day(ts: int) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).date().isoformat()


def _daterange(start: str, end: str) -> list[str]:
    s = _dt.date.fromisoformat(start)
    e = _dt.date.fromisoformat(end)
    return [(s + _dt.timedelta(days=i)).isoformat() for i in range((e - s).days + 1)]


# ── Model side ────────────────────────────────────────────────────────────────


def model_group_winner(played_csv: Path, n: int, seed: int) -> dict[str, dict[str, float]]:
    """{group: {team: P(win group)}} on a given played state."""
    from mc_simu.run_daily import production_ratings
    from mc_simu.tournaments.wc2026 import (
        load_played_results, load_wc2026_bundle, run_monte_carlo,
    )
    sg.ELO_GOALS_DENOMINATOR = 1400.0
    played = load_played_results(played_csv)
    bundle = load_wc2026_bundle(production_ratings(),
                                params=ModelParams(diagonal_inflation=0.20))
    res = run_monte_carlo(bundle, n_iterations=n, seed=seed, progress=False, played=played)
    return {g: {t: s["mc_fair_prob"] for t, s in teams.items()}
            for g, teams in res["group_winners"].items()}


def _csv_text(rows: list[dict]) -> str:
    head = "stage,match_id,home_team,away_team,home_goals,away_goals,winner\n"
    body = "".join(f"group,,{r['home_team']},{r['away_team']},"
                   f"{r['home_goals']},{r['away_goals']},\n" for r in rows)
    return head + body


def _api_played_states(token: str) -> list[tuple[str, str]]:
    """[(real_date, cumulative_csv)] from football-data.org utcDate.

    Reuses fetch_played's name-mapping and same-group fixture guard, so the
    reconstructed states match what the daily cron would have conditioned on.
    Each date's CSV holds every group result with utcDate <= that date.
    """
    from mc_simu.fetch_played import (
        canonical_team, fetch_finished, group_fixture_issue, load_group_membership,
        load_tournament_window, plausible_score, _norm,
    )
    membership = load_group_membership()
    by_norm = {_norm(t): t for teams in membership.values() for t in teams}
    group_of = {t: g for g, teams in membership.items() for t in teams}
    window = load_tournament_window()
    dated: list[tuple[str, dict]] = []
    for m in fetch_finished(token):
        if str(m.get("stage")) != "GROUP_STAGE":
            continue
        home = canonical_team(m.get("homeTeam") or {}, by_norm)
        away = canonical_team(m.get("awayTeam") or {}, by_norm)
        ft = (m.get("score") or {}).get("fullTime") or {}
        hg, ag = ft.get("home"), ft.get("away")
        if home is None or away is None or hg is None or ag is None:
            continue
        if not plausible_score(hg, ag):
            continue
        if group_fixture_issue(home, away, str(m.get("utcDate", "")), group_of, window):
            continue
        dated.append((str(m.get("utcDate", ""))[:10],
                      {"home_team": home, "away_team": away,
                       "home_goals": int(hg), "away_goals": int(ag)}))
    dates = sorted({d for d, _ in dated})
    return [(d, _csv_text([r for dd, r in dated if dd <= d])) for d in dates]


def _git_played_states() -> list[tuple[str, str]]:
    """[(commit_date, csv_text)] for wc2026_played.csv, oldest first."""
    rel = "data/mc_simu/wc2026_played.csv"
    log = subprocess.run(["git", "log", "--format=%H %ad", "--date=short", "--", rel],
                         cwd=PROJECT_ROOT, capture_output=True, text=True).stdout
    out = []
    for line in log.splitlines():
        h, d = line.split(maxsplit=1)
        txt = subprocess.run(["git", "show", f"{h}:{rel}"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True).stdout
        out.append((d.strip(), txt))
    out.reverse()
    # keep latest commit per date (intraday revisions collapse to the final state)
    by_date: dict[str, str] = {}
    for d, txt in out:
        by_date[d] = txt
    return sorted(by_date.items())


def _n_group(txt: str) -> int:
    return sum(1 for ln in txt.splitlines() if ln.startswith("group,"))


def model_series(dates: list[str], n: int, seed: int,
                 states: list[tuple[str, str]]) -> dict[str, dict]:
    """{date: {'matches': int, 'gw': {group: {team: prob}}}} forward-filled.

    `states` is [(as_of_date, played_csv)] oldest-first (git commits or real match
    dates). Each distinct state is simulated once (cached by content hash); dates
    before the first state use an empty played state.
    """
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
                cache[key] = model_group_winner(tmp, n, seed)
            finally:
                tmp.unlink(missing_ok=True)
        return cache[key]

    out = {}
    for d in dates:
        usable = [(cd, txt) for cd, txt in states if cd <= d]
        txt = usable[-1][1] if usable else empty
        out[d] = {"matches": _n_group(txt), "gw": sim(txt)}
        print(f"  model {d}: state={_n_group(txt)} group matches"
              + (" (carried fwd)" if usable and usable[-1][0] != d else ""))
    return out


# ── Polymarket side ───────────────────────────────────────────────────────────


def fetch_poly_group_winner(start_ts: int, end_ts: int, timeout: int = 30,
                            pause: float = 0.12) -> dict[str, dict[str, dict[str, float]]]:
    """{date: {group: {team: raw_price}}} from world-cup-group-{a..l}-winner."""
    import requests
    out: dict[str, dict[str, dict[str, float]]] = {}
    for letter in GROUP_LETTERS:
        slug = f"world-cup-group-{letter.lower()}-winner"
        ev = requests.get(f"{POLY_GAMMA}/events/slug/{slug}", timeout=timeout)
        if ev.status_code != 200:
            print(f"  Poly group {letter}: {ev.status_code} — skipped")
            continue
        for m in ev.json().get("markets", []):
            team = _norm(m.get("groupItemTitle") or "")
            toks = m.get("clobTokenIds")
            if not team or team == "Other" or not toks:
                continue
            try:
                yes_tok = json.loads(toks)[0]
            except (json.JSONDecodeError, IndexError):
                continue
            r = requests.get(f"{POLY_CLOB}/prices-history",
                             params={"market": yes_tok, "startTs": start_ts,
                                     "endTs": end_ts, "fidelity": 1440}, timeout=timeout)
            if r.status_code != 200:
                continue
            for p in r.json().get("history", []):
                t, px = p.get("t"), p.get("p")
                if t is None or px is None or not (0 < float(px) <= 1):
                    continue
                # fidelity=1440 stamps each point a few seconds past 00:00 UTC =
                # the OPEN of that day = the CLOSE of the prior day. Shift back a
                # full day so date D carries end-of-D market (the D+1 00:00 point),
                # aligning with the model which conditions on matches utcDate <= D.
                out.setdefault(_day(t - 86400), {}).setdefault(letter, {})[team] = float(px)
            time.sleep(pause)
        print(f"  Poly group {letter}: done")
    return out


def _devig(raw: dict[str, float]) -> dict[str, float]:
    s = sum(raw.values())
    return {t: v / s for t, v in raw.items()} if s > 0 else {}


def _pct(v) -> str | float:
    return round(v * 100, 3) if v else ""


# ── Assemble ──────────────────────────────────────────────────────────────────


def build_rows(model: dict, poly: dict, groups: dict[str, list[str]]) -> list[dict]:
    rows = []
    for d in sorted(set(model) | set(poly)):
        for g in GROUP_LETTERS:
            teams = groups[g]
            mdl = model.get(d, {}).get("gw", {}).get(g, {})
            matches = model.get(d, {}).get("matches", "")
            pm_raw = {t: v for t, v in poly.get(d, {}).get(g, {}).items() if t in teams}
            # Once a group resolves, the winner's market closes and prices-history
            # stops returning its points; only a residual losing market may stay
            # open. Devigging that lone tiny price gives a bogus ~100%. Drop Poly
            # for the cell when the survivors no longer sum near 1 (winner's mass gone).
            if sum(pm_raw.values()) < 0.5:
                pm_raw = {}
            pm_dv = _devig(pm_raw)
            order = sorted(teams, key=lambda t: -(mdl.get(t) or pm_dv.get(t) or 0))
            for t in order:
                if t not in mdl and t not in pm_raw:
                    continue
                rows.append({
                    "date": d, "group": g, "team": t,
                    # sim ran (mdl non-empty) -> explicit 0.0 for a team with no
                    # group-winning iterations (eliminated), vs "" = no model that
                    # day. Keeps the model-vs-market gap visible on the chart.
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
    p.add_argument("--out", type=Path, default=DATA / "group_winner_log.csv")
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-model", action="store_true", help="Poly only (skip sim runs)")
    p.add_argument("--from-api", action="store_true",
                   help="rebuild per-day model state from football-data.org match "
                        "dates (needs FOOTBALL_DATA_TOKEN) instead of git history")
    args = p.parse_args(argv)

    start_ts = int(_dt.datetime.fromisoformat(args.start)
                   .replace(tzinfo=_dt.timezone.utc).timestamp())
    end_ts = int((_dt.datetime.fromisoformat(args.end) + _dt.timedelta(days=1))
                 .replace(tzinfo=_dt.timezone.utc).timestamp())
    groups = json.load(open(DATA / "wc2026_groups.json"))["groups"]
    dates = _daterange(args.start, args.end)
    banner(f"Backfill group-winner {args.start} -> {args.end}")

    print("Polymarket group-winner history:")
    poly = fetch_poly_group_winner(start_ts, end_ts)

    model = {}
    if not args.no_model:
        if args.from_api:
            import os
            token = os.environ.get("FOOTBALL_DATA_TOKEN")
            if not token:
                print("--from-api needs FOOTBALL_DATA_TOKEN — aborting model line")
                return 2
            print("Model group-winner per day (football-data.org real match dates):")
            states = _api_played_states(token)
        else:
            print("Model group-winner per day (git as-of states):")
            states = _git_played_states()
        model = model_series(dates, args.n, args.seed, states)

    rows = [r for r in build_rows(model, poly, groups) if args.start <= r["date"] <= args.end]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(rows)
    days = sorted({r["date"] for r in rows})
    banner(f"Wrote {len(rows)} rows ({len(days)} days {days[0]}..{days[-1]}) to {args.out}"
           if days else "No rows written")
    return 0


__all__ = ["model_group_winner", "model_series", "fetch_poly_group_winner",
           "build_rows", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
