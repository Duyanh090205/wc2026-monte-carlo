"""Auto-feed wc2026_played.csv from football-data.org finished results.

Removes the manual step of the daily tracker loop: pull final scoreboards for
WC2026 (competition code WC), normalize team names to the model's, recover
FIFA KO match ids (73-104) by resolving the bracket from the group scores, and
upsert into data/mc_simu/wc2026_played.csv for run_daily to condition on.

Conditioning stays facts-only (scoreboards) — this never reads market state.
Degrades on any failure: missing token, network error, or an unmapped team
just leaves the affected matches unlocked (the sim re-rolls them) and keeps
the existing CSV. Unresolvable KO pairs are skipped with a warning, never
guessed.

    PYTHONPATH=src python -m mc_simu.fetch_played
    PYTHONPATH=src python -m mc_simu.fetch_played --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mc_simu._common import banner  # noqa: E402
from mc_simu.standings import (  # noqa: E402
    GroupMatch, compute_stats, rank_best_thirds, rank_group,
)
from mc_simu.tournaments.wc2026 import (  # noqa: E402
    ELO_HISTORY_NAME_ALIASES, GROUP_LETTERS, LATER_ROUNDS, R32_BRACKET,
)

DATA = PROJECT_ROOT / "data" / "mc_simu"
API_URL = "https://api.football-data.org/v4/competitions/WC/matches"
CSV_COLS = ["stage", "match_id", "home_team", "away_team",
            "home_goals", "away_goals", "duration", "pen_home", "pen_away",
            "winner"]

# football-data.org name -> wc2026_groups.json name. Keys are pre-normalized
# (lowercase, diacritics stripped) — extend here when the warning log surfaces
# a new unmapped spelling.
TEAM_ALIASES: dict[str, str] = {
    "czech republic": "Czechia",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "korea, south": "South Korea",
    "turkiye": "Turkey",
    "cote d'ivoire": "Ivory Coast",
    "cote divoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "cape verde islands": "Cape Verde",
    "congo dr": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "congo, democratic republic of the": "DR Congo",
    "usa": "United States",
    "united states of america": "United States",
    "ir iran": "Iran",
    "iran, islamic republic of": "Iran",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
}


def _norm(name: str) -> str:
    stripped = "".join(c for c in unicodedata.normalize("NFKD", name)
                       if not unicodedata.combining(c))
    return stripped.casefold().strip()


def load_group_membership() -> dict[str, list[str]]:
    with (DATA / "wc2026_groups.json").open() as f:
        return {g: list(t) for g, t in json.load(f)["groups"].items()}


def load_tournament_window(margin_days: int = 1) -> tuple:
    """(start, end) dates bounding the whole tournament, from wc2026_groups.json.

    Used to reject results from a wrong competition/season. Deliberately NOT
    per-fixture: wc2026_fixtures.csv carries only coarse bucketed matchday dates
    (off the real schedule by up to ~10 days), so a tight per-match date check
    would drop legitimate played matches. The same-group pairing check is the
    real fixture guard; this only catches gross wrong-tournament dates.
    """
    import datetime as _dt
    with (DATA / "wc2026_groups.json").open() as f:
        meta = json.load(f)
    start = _dt.date.fromisoformat(meta["kickoff_date"]) - _dt.timedelta(days=margin_days)
    end = _dt.date.fromisoformat(meta["final_date"]) + _dt.timedelta(days=margin_days)
    return start, end


def plausible_score(hg, ag) -> bool:
    return 0 <= int(hg) <= 15 and 0 <= int(ag) <= 15


def group_fixture_issue(home: str, away: str, api_date: str,
                        group_of: dict[str, str],
                        window: tuple) -> str | None:
    """Reason string when a reported group result cannot be a real fixture.

    Two same-group teams meet exactly once, so the pairing check confirms a real
    group fixture; the date only needs to fall inside the tournament window to
    rule out a mislabeled wrong-competition result.
    """
    if group_of.get(home) is None or group_of.get(home) != group_of.get(away):
        return "not a scheduled group fixture (teams not in the same group)"
    if len(api_date) >= 10:
        from datetime import date
        d = date.fromisoformat(api_date[:10])
        start, end = window
        if not (start <= d <= end):
            return f"dated {api_date[:10]} outside tournament window {start}..{end}"
    return None


def canonical_team(api_team: dict, by_norm: dict[str, str]) -> str | None:
    for cand in (api_team.get("name"), api_team.get("shortName")):
        if not cand:
            continue
        n = _norm(str(cand))
        if n in by_norm:
            return by_norm[n]
        if n in TEAM_ALIASES:
            return TEAM_ALIASES[n]
    return None


def fetch_finished(token: str) -> list[dict]:
    import requests
    resp = requests.get(API_URL, headers={"X-Auth-Token": token}, timeout=30)
    resp.raise_for_status()
    return [m for m in resp.json().get("matches", [])
            if m.get("status") == "FINISHED"]


def ko_match_goals(score: dict) -> tuple[int, int]:
    """Final match goals — extra time included, shootout goals excluded.

    football-data.org v4 folds penalty-shootout goals into fullTime for
    PENALTY_SHOOTOUT matches; recover the real score from regularTime +
    extraTime when present, else back the shootout out of fullTime.
    """
    ft = score.get("fullTime") or {}
    hg, ag = int(ft["home"]), int(ft["away"])
    if score.get("duration") != "PENALTY_SHOOTOUT":
        return hg, ag
    reg, ext = score.get("regularTime") or {}, score.get("extraTime") or {}
    parts = (reg.get("home"), reg.get("away"), ext.get("home"), ext.get("away"))
    if all(v is not None for v in parts):
        return int(reg["home"]) + int(ext["home"]), int(reg["away"]) + int(ext["away"])
    pens = score.get("penalties") or {}
    ph, pa = pens.get("home"), pens.get("away")
    if ph is not None and pa is not None and hg - int(ph) >= 0 and ag - int(pa) >= 0:
        return hg - int(ph), ag - int(pa)
    return hg, ag


def ko_winner(match: dict, home: str, away: str) -> str | None:
    score = match.get("score") or {}
    w = score.get("winner")
    if w == "HOME_TEAM":
        return home
    if w == "AWAY_TEAM":
        return away
    pens = score.get("penalties") or {}
    ph, pa = pens.get("home"), pens.get("away")
    if ph is not None and pa is not None and ph != pa:
        return home if ph > pa else away
    return None


def resolve_r32_pairings(
    group_scores: dict[frozenset, dict[str, int]],
    membership: dict[str, list[str]],
    r32_table: dict[str, dict[str, str]],
    tiebreaker: dict[str, float],
) -> dict[int, frozenset] | None:
    """Deterministic R32 pairings from full group results; None until all 72 played."""
    matches_by_group: dict[str, list[GroupMatch]] = {}
    for g in GROUP_LETTERS:
        teams = membership[g]
        ms: list[GroupMatch] = []
        for i in range(4):
            for j in range(i + 1, 4):
                real = group_scores.get(frozenset((teams[i], teams[j])))
                if real is None:
                    return None
                ms.append(GroupMatch(home_team=teams[i], away_team=teams[j],
                                     home_goals=real[teams[i]], away_goals=real[teams[j]]))
        matches_by_group[g] = ms

    winners: dict[str, str] = {}
    runners_up: dict[str, str] = {}
    thirds: list[tuple[str, str]] = []
    all_stats: dict = {}
    for g in GROUP_LETTERS:
        stats = compute_stats(matches_by_group[g])
        ranked = rank_group(matches_by_group[g], final_tiebreaker=tiebreaker, stats=stats)
        winners[g], runners_up[g] = ranked[0], ranked[1]
        thirds.append((ranked[2], g))
        all_stats.update(stats)

    top8 = rank_best_thirds(thirds, all_stats, tiebreaker)[:8]
    seeding = r32_table["".join(sorted(g for _t, g in top8))]
    third_by_group = {g: t for t, g in top8}

    def resolve(src: tuple[str, str]) -> str:
        kind, key = src
        if kind == "W":
            return winners[key]
        if kind == "RU":
            return runners_up[key]
        return third_by_group[seeding[f"1{key}"][1]]

    return {mid: frozenset((resolve(left), resolve(right)))
            for mid, left, right in R32_BRACKET}


def find_ko_match_id(
    pair: frozenset,
    r32_pairs: dict[int, frozenset] | None,
    ko_winners: dict[int, str],
) -> int | None:
    if r32_pairs:
        for mid, p in r32_pairs.items():
            if p == pair and mid not in ko_winners:
                return mid
    for mid, left, right in LATER_ROUNDS:
        if mid in ko_winners or left not in ko_winners or right not in ko_winners:
            continue
        if frozenset((ko_winners[left], ko_winners[right])) == pair:
            return mid
    return None


def load_existing_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_key(row: dict) -> tuple:
    if str(row["stage"]).strip().lower() == "group":
        return ("group", frozenset((row["home_team"], row["away_team"])))
    return ("ko", int(float(row["match_id"])))


def merge_rows(existing: list[dict], fetched: list[dict]) -> tuple[list[dict], int]:
    merged: dict[tuple, dict] = {row_key(r): r for r in existing}
    n_changed = 0
    for row in fetched:
        key = row_key(row)
        old = merged.get(key)
        if old is not None and {k: str(old.get(k, "")) for k in CSV_COLS} == row:
            continue
        if old is not None:
            print(f"fetch_played: overwriting {key}: {old} -> {row}")
        merged[key] = row
        n_changed += 1
    group = sorted((r for k, r in merged.items() if k[0] == "group"),
                   key=lambda r: (r["home_team"], r["away_team"]))
    ko = sorted((r for k, r in merged.items() if k[0] == "ko"),
                key=lambda r: int(float(r["match_id"])))
    return group + ko, n_changed


def production_tiebreaker(membership: dict[str, list[str]]) -> dict[str, float]:
    from mc_simu.run_daily import production_ratings
    ratings = production_ratings()
    out: dict[str, float] = {}
    for teams in membership.values():
        for t in teams:
            out[t] = ratings.get(t, ratings.get(ELO_HISTORY_NAME_ALIASES.get(t, t), 1500.0))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DATA / "wc2026_played.csv")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    token = os.environ.get("FOOTBALL_DATA_TOKEN")
    if not token:
        print("fetch_played: FOOTBALL_DATA_TOKEN not set -- skipped (CSV unchanged)")
        return 0
    try:
        finished = fetch_finished(token)
    except Exception as e:
        print(f"fetch_played: WARNING fetch failed ({e}) -- keeping existing CSV")
        return 0

    banner(f"fetch_played: {len(finished)} finished matches from football-data.org")
    membership = load_group_membership()
    by_norm = {_norm(t): t for teams in membership.values() for t in teams}
    group_of = {t: g for g, teams in membership.items() for t in teams}
    window = load_tournament_window()

    existing = load_existing_rows(args.csv)
    group_scores: dict[frozenset, dict[str, int]] = {}
    for r in existing:
        if str(r["stage"]).strip().lower() == "group":
            group_scores[frozenset((r["home_team"], r["away_team"]))] = {
                r["home_team"]: int(r["home_goals"]), r["away_team"]: int(r["away_goals"])}

    fetched: list[dict] = []
    unmapped: set[str] = set()
    ko_api: list[tuple[frozenset, dict, str, str]] = []
    for m in sorted(finished, key=lambda m: str(m.get("utcDate", ""))):
        stage = str(m.get("stage", ""))
        if stage in ("THIRD_PLACE", "PLAY_OFFS"):
            continue
        home = canonical_team(m.get("homeTeam") or {}, by_norm)
        away = canonical_team(m.get("awayTeam") or {}, by_norm)
        if home is None or away is None:
            raw_h = (m.get("homeTeam") or {}).get("name")
            raw_a = (m.get("awayTeam") or {}).get("name")
            unmapped.update(n for n, t in ((raw_h, home), (raw_a, away)) if t is None and n)
            continue
        ft = (m.get("score") or {}).get("fullTime") or {}
        hg, ag = ft.get("home"), ft.get("away")
        if hg is None or ag is None:
            print(f"fetch_played: WARNING no fullTime score for {home} vs {away} -- skipped")
            continue
        if not plausible_score(hg, ag):
            print(f"fetch_played: WARNING implausible score {hg}-{ag} for "
                  f"{home} vs {away} -- skipped")
            continue
        if stage == "GROUP_STAGE":
            issue = group_fixture_issue(home, away, str(m.get("utcDate", "")),
                                        group_of, window)
            if issue:
                print(f"fetch_played: WARNING {home} vs {away}: {issue} -- skipped")
                continue
            group_scores[frozenset((home, away))] = {home: int(hg), away: int(ag)}
            fetched.append({"stage": "group", "match_id": "", "home_team": home,
                            "away_team": away, "home_goals": str(int(hg)),
                            "away_goals": str(int(ag)), "duration": "",
                            "pen_home": "", "pen_away": "", "winner": ""})
        else:
            ko_api.append((frozenset((home, away)), m, home, away))

    r32_pairs = None
    if ko_api:
        with (DATA / "r32_seeding_table.json").open() as f:
            r32_table = json.load(f)
        r32_pairs = resolve_r32_pairings(
            group_scores, membership, r32_table, production_tiebreaker(membership))
        if r32_pairs is None:
            print("fetch_played: WARNING KO results present but group stage incomplete "
                  "-- KO matches skipped")

    ko_locked: dict[int, str] = {}
    for pair, m, home, away in ko_api:
        mid = find_ko_match_id(pair, r32_pairs, ko_locked)
        if mid is None:
            print(f"fetch_played: WARNING cannot place KO pair {set(pair)} in bracket "
                  "-- skipped (add manually if the bracket resolution disagrees)")
            continue
        w = ko_winner(m, home, away)
        if w is None:
            print(f"fetch_played: WARNING no winner for KO match {mid} {set(pair)} -- skipped")
            continue
        score = m.get("score") or {}
        hg, ag = ko_match_goals(score)
        pens = score.get("penalties") or {}
        ph, pa = pens.get("home"), pens.get("away")
        ko_locked[mid] = w
        fetched.append({"stage": "ko", "match_id": str(mid), "home_team": home,
                        "away_team": away, "home_goals": str(hg),
                        "away_goals": str(ag),
                        "duration": str(score.get("duration") or "REGULAR"),
                        "pen_home": "" if ph is None else str(int(ph)),
                        "pen_away": "" if pa is None else str(int(pa)),
                        "winner": w})

    rows, n_changed = merge_rows(existing, fetched)
    n_group = sum(1 for r in rows if r["stage"] == "group")
    n_ko = len(rows) - n_group
    if unmapped:
        print(f"fetch_played: WARNING unmapped team names {sorted(unmapped)} "
              "-- add to TEAM_ALIASES")
    print(f"fetch_played: locked {n_group}/72 group + {n_ko} KO; "
          f"{n_changed} new/changed rows")

    if n_changed == 0:
        print("fetch_played: CSV unchanged")
        return 0
    if args.dry_run:
        print("fetch_played: dry run -- not writing")
        return 0
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"fetch_played: wrote {args.csv}")
    return 0


__all__ = [
    "TEAM_ALIASES",
    "canonical_team",
    "fetch_finished",
    "find_ko_match_id",
    "group_fixture_issue",
    "ko_winner",
    "load_existing_rows",
    "ko_match_goals",
    "load_tournament_window",
    "merge_rows",
    "plausible_score",
    "resolve_r32_pairings",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
