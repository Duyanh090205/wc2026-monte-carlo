"""Scrape WC2026 team-aggregate market values from Transfermarkt.

Source: https://www.transfermarkt.com/world-cup/teilnehmer/pokalwettbewerb/FIWC
Output: data/mc_simu/transfermarkt_mv_wc2026.csv
        columns: team, total_mv_eur, squad_size, avg_age, scraped_date
Cache : data/mc_simu/cache/transfermarkt/wc2026_teams.html (committed; predict path is offline per CLAUDE.md).

Phase-1 scope: team-aggregate MV only (1 page, ~50 teams). Phase-2 upgrade to
per-squad scrape (48 extra page fetches) only if outlier teams (Argentina/Egypt
star-driven) diverge from PM after team-aggregate MV blend.

Transfermarkt MV string format: "€1.36bn" / "€845.50m" / "€42.50m" → cents-aware
parse to a float number of EUR.

Sam direction 2026-05-27 + memory [[mc-player-market-value-feature]]: MV as 2nd
strength signal alongside Elo to bring mc_simu closer to PM/Kalshi distribution.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import requests  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

from mc_simu._common import banner, hard_stop  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
CACHE_DIR = DATA_DIR / "cache" / "transfermarkt"
OUTPUT_CSV = DATA_DIR / "transfermarkt_mv_wc2026.csv"

SOURCE_URL = "https://www.transfermarkt.com/world-cup/teilnehmer/pokalwettbewerb/FIWC"
CACHE_FILE = "wc2026_teams.html"

# Transfermarkt team-name aliases vs canonical name in wc2026_groups.json.
# Built from observed mismatches; extend if scrape WARNs about unmapped teams.
NAME_ALIASES = {
    "USA": "United States",
    "Korea, South": "South Korea",
    "Republic of Korea": "South Korea",
    "Cape Verde Islands": "Cape Verde",
    "Czech Republic": "Czechia",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Curaçao": "Curacao",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}

# 48 expected teams (same as scrape_wc2026_squads.EXPECTED_TEAMS).
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


def _fetch_html(force: bool = False) -> str:
    cache_path = CACHE_DIR / CACHE_FILE
    if cache_path.exists() and not force:
        print(f"  using cached {cache_path}")
        return cache_path.read_text(encoding="utf-8")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {SOURCE_URL} ...")
    # Realistic UA — Transfermarkt blocks default python-requests UA.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(SOURCE_URL, headers=headers, timeout=60)
    if resp.status_code != 200:
        hard_stop(
            "scrape_transfermarkt_mv",
            f"HTTP {resp.status_code} — Transfermarkt may have blocked the bot. "
            f"Fallback: open {SOURCE_URL} in browser, Save Page As .html into {cache_path}",
            "HTTP 200",
        )
    cache_path.write_text(resp.text, encoding="utf-8")
    return resp.text


_MV_RE = re.compile(r"€\s*([0-9]+(?:\.[0-9]+)?)\s*(bn|m|k)?", re.IGNORECASE)


def parse_mv_to_eur(s: str) -> float:
    """Parse Transfermarkt money string like '€1.36bn', '€845.50m', '€42.50m' to EUR float."""
    m = _MV_RE.search(s.replace(",", ""))
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    mult = {"bn": 1_000_000_000.0, "m": 1_000_000.0, "k": 1_000.0, "": 1.0}[unit]
    return val * mult


def _normalize_team(raw: str) -> str:
    return NAME_ALIASES.get(raw.strip(), raw.strip())


def parse_table(html: str) -> list[dict]:
    """Extract per-team rows from Transfermarkt teilnehmer table.

    Observed column layout (8 cells, verified 2026-05-27):
        [0] wappen icon | [1] Club (team) | [2] Squad size | [3] avg age |
        [4] WC particip. | [5] Foreigners% | [6] Total Market Value | [7] avg Market Value
    Total MV at cells[6] (i.e. cells[-2]) NOT cells[-1] — avg MV is the last column.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="items")
    if table is None:
        hard_stop("scrape_transfermarkt_mv", "no <table class='items'> found", "found")
    rows = []
    tbody = table.find("tbody")
    for tr in tbody.find_all("tr", recursive=False):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 8:
            continue
        team_cell = cells[1]
        team_link = team_cell.find("a")
        if not team_link:
            continue
        raw_team = team_link.get("title") or team_link.get_text(strip=True)
        team = _normalize_team(raw_team)

        squad_size = int(cells[2].get_text(strip=True) or 0)
        avg_age_txt = cells[3].get_text(strip=True).replace(",", ".")
        try:
            avg_age = float(avg_age_txt) if avg_age_txt else 0.0
        except ValueError:
            avg_age = 0.0
        total_mv_txt = cells[6].get_text(strip=True)
        total_mv_eur = parse_mv_to_eur(total_mv_txt)
        rows.append({
            "team": team,
            "total_mv_eur": total_mv_eur,
            "squad_size": squad_size,
            "avg_age": avg_age,
            "_raw_mv": total_mv_txt,
        })
    return rows


def write_csv(rows: list[dict], scraped_date: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team", "total_mv_eur", "squad_size", "avg_age", "scraped_date"])
        for r in sorted(rows, key=lambda x: -x["total_mv_eur"]):
            w.writerow([r["team"], r["total_mv_eur"], r["squad_size"], r["avg_age"], scraped_date])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape WC2026 team market values from Transfermarkt")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args(argv)

    banner("STEP 1 — Fetch HTML")
    html = _fetch_html(force=args.force)
    print(f"  HTML length: {len(html):,} chars")

    banner("STEP 2 — Parse table")
    rows = parse_table(html)
    print(f"  Parsed {len(rows)} team rows")
    if not rows:
        hard_stop("scrape_transfermarkt_mv", "0 rows parsed", ">0")

    banner("STEP 3 — Coverage check vs 48 WC2026 teams")
    scraped_teams = {r["team"] for r in rows}
    missing = EXPECTED_TEAMS - scraped_teams
    extra = scraped_teams - EXPECTED_TEAMS
    print(f"  Covered: {len(scraped_teams & EXPECTED_TEAMS)}/48")
    if missing:
        print(f"  MISSING ({len(missing)}): {sorted(missing)}")
    if extra:
        print(f"  EXTRA (not in WC2026): {sorted(extra)}")

    banner("STEP 4 — Write CSV")
    write_csv(rows, scraped_date=date.today().isoformat())
    print(f"  wrote {OUTPUT_CSV}")

    banner("Top 10 by MV")
    for r in sorted(rows, key=lambda x: -x["total_mv_eur"])[:10]:
        mv_b = r["total_mv_eur"] / 1e9
        print(f"  {r['team']:<26} €{mv_b:>5.2f}bn  squad={r['squad_size']:>3d}  avg_age={r['avg_age']:>4.1f}  raw={r['_raw_mv']}")

    return 0


__all__ = ["scrape_transfermarkt_mv", "parse_mv_to_eur", "parse_table", "EXPECTED_TEAMS"]


def scrape_transfermarkt_mv(force_fetch: bool = False) -> list[dict]:
    """Public API: scrape and return list of {team, total_mv_eur, squad_size, avg_age} dicts."""
    html = _fetch_html(force=force_fetch)
    return parse_table(html)


if __name__ == "__main__":
    sys.exit(main())
