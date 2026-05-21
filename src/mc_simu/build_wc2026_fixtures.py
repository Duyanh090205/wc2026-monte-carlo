"""Build WC2026 fixtures CSV (104 matches) for Phase 0 Check 6 structural verification.

Strategy (pragmatic for Phase 0):
  - Group matches (72): enumerated round-robin from data/mc_simu/wc2026_groups.json.
    12 groups × 4 teams × C(4,2) = 12 × 6 = 72 matches. Order: official FIFA round
    pattern (matchday 1, 2, 3 with specific pairings).
  - Knockout matches (32): placeholder rows with stage labels and TBD teams.
    R32 (16) → R16 (8) → QF (4) → SF (2) → 3rd-place (1) + Final (1) = 32.

Venue + kickoff_utc populated from a known venues table for opener and final;
TBD for intermediate matches. Phase 3 will enrich as needed for HFA calculation.

Output: data/mc_simu/wc2026_fixtures.csv (105 rows incl header).
Schema : match_id, stage, group, date, kickoff_utc, venue, venue_country,
         venue_confederation, home_team, away_team, attendance_pct.

Per spec §1.4 deliverable + §1.3 Check 6.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mc_simu._common import banner, hard_stop  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
GROUPS_JSON = DATA_DIR / "wc2026_groups.json"
OUTPUT_CSV = DATA_DIR / "wc2026_fixtures.csv"

# Per-tournament metadata. WC2026 is co-hosted by USA/Mexico/Canada (CONCACAF).
# All 16 venues are in CONCACAF region.
HOST_CONFEDERATION = "CONCACAF"

# Known venues (city → venue, country). Per FIFA WC2026 venue list.
WC2026_VENUES = {
    # USA (11 venues)
    "Atlanta":      ("Mercedes-Benz Stadium", "USA"),
    "Boston":       ("Gillette Stadium", "USA"),
    "Dallas":       ("AT&T Stadium", "USA"),
    "Houston":      ("NRG Stadium", "USA"),
    "Kansas City":  ("Arrowhead Stadium", "USA"),
    "Los Angeles":  ("SoFi Stadium", "USA"),
    "Miami":        ("Hard Rock Stadium", "USA"),
    "New York":     ("MetLife Stadium", "USA"),
    "Philadelphia": ("Lincoln Financial Field", "USA"),
    "San Francisco":("Levi's Stadium", "USA"),
    "Seattle":      ("Lumen Field", "USA"),
    # Mexico (3 venues)
    "Mexico City":  ("Estadio Azteca", "Mexico"),
    "Guadalajara":  ("Estadio Akron", "Mexico"),
    "Monterrey":    ("Estadio BBVA", "Mexico"),
    # Canada (2 venues)
    "Toronto":      ("BMO Field", "Canada"),
    "Vancouver":    ("BC Place", "Canada"),
}

# Known fixed-point fixtures (opening + final).
KNOWN_FIXTURES = {
    "OPENER":  {"date": "2026-06-11", "venue": "Estadio Azteca",
                "venue_country": "Mexico", "kickoff_utc": "20:00"},
    "FINAL":   {"date": "2026-07-19", "venue": "MetLife Stadium",
                "venue_country": "USA", "kickoff_utc": "15:00"},
    "THIRD":   {"date": "2026-07-18", "venue": "Hard Rock Stadium",
                "venue_country": "USA", "kickoff_utc": "15:00"},
}

# Round-robin matchday pattern for a 4-team group (1,2,3,4):
# Day 1: 1 vs 2, 3 vs 4
# Day 2: 1 vs 3, 4 vs 2
# Day 3: 4 vs 1, 2 vs 3
MATCHDAY_PATTERN = [
    [(0, 1), (2, 3)],
    [(0, 2), (3, 1)],
    [(3, 0), (1, 2)],
]


def build_group_matches(groups: dict[str, list[str]]) -> list[dict]:
    """Enumerate 72 group matches from groups dict."""
    fixtures: list[dict] = []
    match_id = 1
    for group_letter, teams in groups.items():
        if len(teams) != 4:
            hard_stop(f"build_wc2026_fixtures group {group_letter}",
                      f"{len(teams)} teams", "exactly 4")
        for matchday_idx, pairings in enumerate(MATCHDAY_PATTERN, start=1):
            for home_idx, away_idx in pairings:
                fixtures.append({
                    "match_id": match_id,
                    "stage": "group",
                    "group": group_letter,
                    "date": f"2026-06-{10 + matchday_idx * 4:02d}",  # approx matchday-spread
                    "kickoff_utc": "TBD",
                    "venue": "TBD",
                    "venue_country": "TBD",
                    "venue_confederation": HOST_CONFEDERATION,
                    "home_team": teams[home_idx],
                    "away_team": teams[away_idx],
                    "attendance_pct": 1.0,
                })
                match_id += 1
    return fixtures


def build_knockout_placeholders(start_id: int) -> list[dict]:
    """Generate 32 placeholder rows for knockout bracket."""
    fixtures: list[dict] = []
    stages = [
        ("r32", 16, "2026-06-30"),
        ("r16", 8,  "2026-07-04"),
        ("qf",  4,  "2026-07-09"),
        ("sf",  2,  "2026-07-14"),
        ("third", 1, KNOWN_FIXTURES["THIRD"]["date"]),
        ("final", 1, KNOWN_FIXTURES["FINAL"]["date"]),
    ]
    match_id = start_id
    for stage, n_matches, default_date in stages:
        for slot in range(1, n_matches + 1):
            row = {
                "match_id": match_id,
                "stage": stage,
                "group": "",
                "date": default_date,
                "kickoff_utc": "TBD",
                "venue": "TBD",
                "venue_country": "TBD",
                "venue_confederation": HOST_CONFEDERATION,
                "home_team": "TBD",
                "away_team": "TBD",
                "attendance_pct": 1.0,
            }
            # Enrich known fixed-point matches
            if stage == "final":
                row.update({
                    "date": KNOWN_FIXTURES["FINAL"]["date"],
                    "kickoff_utc": KNOWN_FIXTURES["FINAL"]["kickoff_utc"],
                    "venue": KNOWN_FIXTURES["FINAL"]["venue"],
                    "venue_country": KNOWN_FIXTURES["FINAL"]["venue_country"],
                })
            elif stage == "third":
                row.update({
                    "date": KNOWN_FIXTURES["THIRD"]["date"],
                    "kickoff_utc": KNOWN_FIXTURES["THIRD"]["kickoff_utc"],
                    "venue": KNOWN_FIXTURES["THIRD"]["venue"],
                    "venue_country": KNOWN_FIXTURES["THIRD"]["venue_country"],
                })
            fixtures.append(row)
            match_id += 1
    return fixtures


def enrich_opener(group_fixtures: list[dict]) -> None:
    """Enrich the first match (Mexico vs first opponent in Group A) with opener venue."""
    if not group_fixtures:
        return
    first = group_fixtures[0]
    first.update({
        "date": KNOWN_FIXTURES["OPENER"]["date"],
        "kickoff_utc": KNOWN_FIXTURES["OPENER"]["kickoff_utc"],
        "venue": KNOWN_FIXTURES["OPENER"]["venue"],
        "venue_country": KNOWN_FIXTURES["OPENER"]["venue_country"],
    })


def main() -> int:
    banner("MC Simu — WC2026 Fixtures Builder")

    if not GROUPS_JSON.exists():
        hard_stop("build_wc2026_fixtures", f"{GROUPS_JSON} missing",
                  "wc2026_groups.json must exist")

    groups_data = json.loads(GROUPS_JSON.read_text(encoding="utf-8"))
    groups = groups_data["groups"]
    print(f"Loaded {len(groups)} groups from {GROUPS_JSON.name}")
    print(f"  Host nations: {', '.join(groups_data['host_nations'])}")
    print(f"  All {groups_data['teams_confirmed']} teams confirmed "
          f"({groups_data['teams_tbd']} TBD)")

    group_fixtures = build_group_matches(groups)
    enrich_opener(group_fixtures)
    knockout_fixtures = build_knockout_placeholders(start_id=len(group_fixtures) + 1)
    all_fixtures = group_fixtures + knockout_fixtures

    if len(all_fixtures) != 104:
        hard_stop("build_wc2026_fixtures total count",
                  f"{len(all_fixtures)}", "exactly 104 (72 group + 32 KO)")

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["match_id", "stage", "group", "date", "kickoff_utc",
                  "venue", "venue_country", "venue_confederation",
                  "home_team", "away_team", "attendance_pct"]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_fixtures:
            writer.writerow(row)

    # Verify per-group count
    group_count: dict[str, int] = {}
    for row in group_fixtures:
        group_count[row["group"]] = group_count.get(row["group"], 0) + 1
    for letter, count in sorted(group_count.items()):
        if count != 6:
            hard_stop(f"build_wc2026_fixtures group {letter}",
                      f"{count} matches", "exactly 6 per group")

    n_known_venue = sum(1 for r in all_fixtures if r["venue"] != "TBD")
    print()
    print(f"Wrote {OUTPUT_CSV} ({len(all_fixtures)} rows)")
    print(f"  group matches: {len(group_fixtures)} (12 × 6 = 72)")
    print(f"  KO matches:    {len(knockout_fixtures)} (32)")
    print(f"  known venues:  {n_known_venue}/{len(all_fixtures)} (opener + 3rd + final)")

    banner("WC2026 fixtures build complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
