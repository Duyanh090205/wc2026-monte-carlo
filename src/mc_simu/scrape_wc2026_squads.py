"""Scrape provisional WC2026 squads (48 teams x ~25 players) from Wikipedia.

Source: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads (single page with all teams).

Output: data/mc_simu/wc2026_squads_provisional.json.
Schema : {team_name: [{player, club, position}, ...], ...}

Final squads due 2026-06-02. Phase 2 will re-run with official squads when available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import requests  # noqa: E402
from bs4 import BeautifulSoup, Tag  # noqa: E402

from mc_simu._common import banner, hard_stop  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_JSON = DATA_DIR / "wc2026_squads_provisional.json"

SOURCE_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
CACHE_FILE = "wc2026_squads.html"

# 48 teams confirmed (from data/mc_simu/wc2026_groups.json).
EXPECTED_TEAMS = {
    "Mexico", "South Africa", "South Korea", "Czechia",
    "Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina",
    "Brazil", "Morocco", "Haiti", "Scotland",
    "United States", "Paraguay", "Australia", "Turkey",
    "Germany", "Curacao", "Ivory Coast", "Ecuador",
    "Netherlands", "Japan", "Tunisia", "Sweden",
    "Belgium", "Egypt", "Iran", "New Zealand",
    "Spain", "Cape Verde", "Saudi Arabia", "Uruguay",
    "France", "Senegal", "Norway", "Iraq",
    "Argentina", "Algeria", "Austria", "Jordan",
    "Portugal", "DR Congo", "Uzbekistan", "Colombia",
    "England", "Croatia", "Ghana", "Panama",
}

# Wikipedia section heading aliases vs canonical name in wc2026_groups.json.
HEADING_ALIASES = {
    "Czech Republic": "Czechia",
    "United States": "United States",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Curaçao": "Curacao",
    "DR Congo": "DR Congo",
    "Republic of Korea": "South Korea",
}


def _fetch_html(force: bool = False) -> str:
    cache_path = CACHE_DIR / CACHE_FILE
    if cache_path.exists() and not force:
        return cache_path.read_text(encoding="utf-8")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {SOURCE_URL} ...")
    resp = requests.get(SOURCE_URL, headers={"User-Agent": "FairLine-MC-Simu/0.1"}, timeout=60)
    if resp.status_code != 200:
        hard_stop("scrape_wc2026_squads", f"HTTP {resp.status_code}", "HTTP 200")
    cache_path.write_text(resp.text, encoding="utf-8")
    return resp.text


def _next_squad_table(heading_div: Tag) -> Tag | None:
    """Find the next <table class="...wikitable..."> sibling after a heading div."""
    for sib in heading_div.find_all_next():
        # Stop at the next h3 heading div — no table found before it
        if sib.name == "div" and "mw-heading3" in (sib.get("class") or []):
            return None
        if sib.name == "table" and "wikitable" in (sib.get("class") or []):
            return sib
    return None


def _parse_squad_rows(table: Tag) -> list[dict]:
    """Parse <tr class='nat-fs-player'> rows → list of {player, club, position} dicts."""
    out: list[dict] = []
    for row in table.find_all("tr", class_="nat-fs-player"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 7:
            continue

        # Position — second cell, look for <a> link with abbreviation
        pos_cell = cells[1]
        pos_link = pos_cell.find("a")
        position = (pos_link.get_text(strip=True) if pos_link
                    else pos_cell.get_text(strip=True))

        # Player — third cell (th), use <a> link text if present
        player_cell = cells[2]
        player_link = player_cell.find("a")
        player = (player_link.get_text(strip=True) if player_link
                  else player_cell.get_text(strip=True))

        # Club — last cell, find last <a> link (skip flag icon link)
        club_cell = cells[-1]
        club_links = club_cell.find_all("a")
        # Filter out flag links (alt text contains "Football Association" or similar)
        meaningful_links = [
            a for a in club_links
            if a.get("title") and "Football" not in a.get("title", "")
            and "Federation" not in a.get("title", "")
            and "Association" not in a.get("title", "")
        ]
        if meaningful_links:
            club = meaningful_links[-1].get_text(strip=True)
        elif club_links:
            club = club_links[-1].get_text(strip=True)
        else:
            club = club_cell.get_text(strip=True)

        if player:
            out.append({"player": player, "club": club, "position": position})
    return out


def scrape_squads(force_fetch: bool = False) -> dict[str, list[dict]]:
    """Scrape all team squads from the WC2026 squads Wikipedia page."""
    html = _fetch_html(force=force_fetch)
    soup = BeautifulSoup(html, "html.parser")

    squads: dict[str, list[dict]] = {}

    # Each team section has a div.mw-heading.mw-heading3 containing h3 with the team name.
    for heading_div in soup.find_all("div", class_="mw-heading3"):
        h3 = heading_div.find("h3")
        if not h3 or not h3.get("id"):
            continue
        team_id = h3["id"]
        # Skip group headers (Group_A, Group_B, ...)
        if team_id.startswith("Group_"):
            continue
        team_text = h3.get_text(strip=True)
        team_name = HEADING_ALIASES.get(team_text, team_text)

        # Filter to expected teams (skip stray sections like "See also")
        if team_name not in EXPECTED_TEAMS:
            continue

        table = _next_squad_table(heading_div)
        if table is None:
            print(f"  WARN: no squad table found for {team_name}")
            continue

        roster = _parse_squad_rows(table)
        if not roster:
            print(f"  WARN: 0 players parsed for {team_name}")
            continue

        squads[team_name] = roster
        print(f"  {team_name:30s} -> {len(roster)} players")

    return squads


def write_json(squads: dict[str, list[dict]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(squads, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape WC2026 provisional squads")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args(argv)

    banner("MC Simu — WC2026 Squads Scraper")
    squads = scrape_squads(force_fetch=args.force)
    write_json(squads)

    n_teams = len(squads)
    n_players = sum(len(s) for s in squads.values())
    missing = EXPECTED_TEAMS - set(squads.keys())

    print()
    print(f"Wrote {OUTPUT_JSON}")
    print(f"  teams:   {n_teams}/{len(EXPECTED_TEAMS)}")
    print(f"  players: {n_players}")
    if missing:
        print(f"  WARN: missing teams: {sorted(missing)}")

    # Per spec §1.2 fallback note: provisional partial squads are acceptable before
    # the June 2, 2026 official deadline. Check 5 evaluates over teams we do have.
    if n_teams < 10:
        hard_stop("scrape_wc2026_squads team count", f"{n_teams} teams",
                  ">= 10 (something is structurally broken)")

    banner(f"WC2026 squads scrape complete ({n_teams}/{len(EXPECTED_TEAMS)} teams, {n_players} players)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
