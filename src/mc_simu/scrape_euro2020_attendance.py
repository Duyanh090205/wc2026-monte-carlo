"""Scrape per-match attendance for UEFA Euro 2020 from Wikipedia.

Output: data/mc_simu/euro2020_attendance.csv (51 matches).
Schema : match_id, date, home, away, venue, capacity, actual_attendance, attendance_pct.

Strategy:
  1. Download Wikipedia UEFA Euro 2020 main page once (cache HTML).
  2. Parse <div class="footballbox"> elements — one per match.
  3. Cross-reference venue name against hardcoded 11-venue capacity lookup.
  4. Compute attendance_pct = actual / capacity.

If main page lookup yields <51 matches, fall back to per-group pages.

Per spec §1.4 deliverable.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import requests  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

from mc_simu._common import banner, hard_stop  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_CSV = DATA_DIR / "euro2020_attendance.csv"

MAIN_URL = "https://en.wikipedia.org/wiki/UEFA_Euro_2020"
GROUP_URLS = {
    "A": "https://en.wikipedia.org/wiki/UEFA_Euro_2020_Group_A",
    "B": "https://en.wikipedia.org/wiki/UEFA_Euro_2020_Group_B",
    "C": "https://en.wikipedia.org/wiki/UEFA_Euro_2020_Group_C",
    "D": "https://en.wikipedia.org/wiki/UEFA_Euro_2020_Group_D",
    "E": "https://en.wikipedia.org/wiki/UEFA_Euro_2020_Group_E",
    "F": "https://en.wikipedia.org/wiki/UEFA_Euro_2020_Group_F",
}
KNOCKOUT_URL = "https://en.wikipedia.org/wiki/UEFA_Euro_2020_knockout_stage"

# 11 venues used in Euro 2020 — capacity from Wikipedia venue pages (2021 figures).
EURO2020_VENUE_CAPACITY = {
    "Wembley Stadium": 90_000,
    "Krestovsky Stadium": 68_134,
    "Saint Petersburg Stadium": 68_134,        # alternate name
    "Allianz Arena": 75_024,
    "Football Arena Munich": 75_024,           # branded name during tournament
    "Stadio Olimpico": 70_634,
    "Olimpico in Rome": 70_634,                # alternate phrasing
    "Puskás Aréna": 67_215,
    "Puskas Arena": 67_215,                    # ASCII variant
    "National Arena": 55_634,
    "Arena Națională": 55_634,
    "Bucharest Arena": 55_634,                 # alternate
    "Hampden Park": 51_866,
    "Parken Stadium": 38_065,
    "Estadio La Cartuja": 60_000,
    "Estadio de La Cartuja": 60_000,
    "La Cartuja": 60_000,
    "Johan Cruyff Arena": 54_990,
    "Cruijff Arena": 54_990,                   # alternate spelling
    "Cruyff Arena": 54_990,
    "Olympic Stadium": 68_700,                 # Baku
    "Baku Olympic Stadium": 68_700,
}


def _cache_fetch(url: str, cache_name: str, *, force: bool = False) -> str:
    """Download URL and cache HTML; return body text."""
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists() and not force:
        return cache_path.read_text(encoding="utf-8")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url} ...")
    resp = requests.get(url, headers={"User-Agent": "FairLine-MC-Simu/0.1"}, timeout=60)
    if resp.status_code != 200:
        hard_stop(
            f"scrape_euro2020_attendance({cache_name})",
            f"HTTP {resp.status_code}",
            "HTTP 200",
        )
    cache_path.write_text(resp.text, encoding="utf-8")
    return resp.text


def _normalize_team(name: str) -> str:
    """Strip whitespace and footnote marks from a team name."""
    return re.sub(r"\s+", " ", name).strip().rstrip("*").strip()


def _parse_date(text: str) -> date | None:
    """Best-effort parse of common Wikipedia date formats found in match infoboxes."""
    text = text.strip()
    # Common formats: "11 June 2021", "11 June 2021 (2021-06-11)", "2021-06-11"
    for fmt in ("%d %B %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:30].strip(), fmt).date()
        except ValueError:
            continue
    iso_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if iso_match:
        try:
            return datetime.strptime(iso_match.group(1), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def parse_footballboxes(html: str) -> list[dict]:
    """Extract match records from any Wikipedia page containing <div class='footballbox'>."""
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict] = []

    for box in soup.find_all("div", class_="footballbox"):
        text = box.get_text(" ", strip=True)

        # Attendance — required (skip box if absent).
        att_match = re.search(r"Attendance:\s*([\d,]+)", text)
        if not att_match:
            continue
        attendance = int(att_match.group(1).replace(",", ""))

        # Date — date appears in itemprop="startDate" or as plain text.
        date_val: date | None = None
        date_meta = box.find(attrs={"itemprop": "startDate"})
        if date_meta and date_meta.get("content"):
            date_val = _parse_date(date_meta["content"])
        if date_val is None:
            # Fallback: scan a <span class="bday-time-date"> or just sniff text
            bday = box.find("span", class_="bday")
            if bday:
                date_val = _parse_date(bday.get_text())

        # Teams — first two <a> inside the score table heading rows usually.
        team_links = box.find_all("a", attrs={"itemprop": "name"})
        if len(team_links) < 2:
            # Fallback: first two team-like links inside fhome/faway tds
            home_td = box.find("th", class_="fhome") or box.find("td", class_="fhome")
            away_td = box.find("th", class_="faway") or box.find("td", class_="faway")
            home = _normalize_team(home_td.get_text()) if home_td else None
            away = _normalize_team(away_td.get_text()) if away_td else None
        else:
            home = _normalize_team(team_links[0].get_text())
            away = _normalize_team(team_links[1].get_text())

        # Venue — itemprop="location" wraps a place link.
        venue = None
        loc = box.find(attrs={"itemprop": "location"})
        if loc:
            place_link = loc.find("a")
            if place_link:
                venue = _normalize_team(place_link.get_text())

        records.append({
            "date": date_val.isoformat() if date_val else "",
            "home": home or "",
            "away": away or "",
            "venue": venue or "",
            "actual_attendance": attendance,
        })

    return records


def venue_capacity(venue: str) -> int | None:
    """Lookup capacity by venue name (handles known aliases)."""
    return EURO2020_VENUE_CAPACITY.get(venue.strip())


def scrape_euro2020(force_fetch: bool = False) -> list[dict]:
    """Scrape all 51 matches. Returns list of dicts with full schema."""
    matches: list[dict] = []

    # Primary: main UEFA Euro 2020 page (often consolidates all knockout boxes).
    html = _cache_fetch(MAIN_URL, "euro2020_main.html", force=force_fetch)
    matches.extend(parse_footballboxes(html))
    print(f"  main page: {len(matches)} match boxes")

    # Augment from per-group pages — main page may omit some group matches.
    for group_letter, url in GROUP_URLS.items():
        html = _cache_fetch(url, f"euro2020_group_{group_letter}.html", force=force_fetch)
        group_matches = parse_footballboxes(html)
        print(f"  Group {group_letter}: +{len(group_matches)} match boxes")
        matches.extend(group_matches)

    # Knockout stage page (may overlap with main; dedupe later).
    html = _cache_fetch(KNOCKOUT_URL, "euro2020_knockout.html", force=force_fetch)
    knockout_matches = parse_footballboxes(html)
    print(f"  knockout: +{len(knockout_matches)} match boxes")
    matches.extend(knockout_matches)

    # Dedupe by (date, home, away)
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    for m in matches:
        key = (m["date"], m["home"], m["away"])
        if key in seen or not all(key):
            continue
        seen.add(key)
        unique.append(m)
    print(f"  deduped: {len(unique)} unique matches")

    # Enrich with capacity + attendance_pct
    enriched: list[dict] = []
    unknown_venues: list[str] = []
    for i, m in enumerate(unique, start=1):
        cap = venue_capacity(m["venue"])
        if cap is None:
            unknown_venues.append(m["venue"])
            pct = None
        else:
            pct = round(m["actual_attendance"] / cap, 4) if cap else None
        enriched.append({
            "match_id": i,
            "date": m["date"],
            "home": m["home"],
            "away": m["away"],
            "venue": m["venue"],
            "capacity": cap if cap is not None else "",
            "actual_attendance": m["actual_attendance"],
            "attendance_pct": pct if pct is not None else "",
        })

    if unknown_venues:
        print(f"  WARN: {len(set(unknown_venues))} unknown venues: {sorted(set(unknown_venues))}")

    return enriched


def write_csv(records: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["match_id", "date", "home", "away", "venue",
                  "capacity", "actual_attendance", "attendance_pct"]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape Euro 2020 per-match attendance")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args(argv)

    banner("MC Simu — Euro 2020 Attendance Scraper")
    records = scrape_euro2020(force_fetch=args.force)
    write_csv(records)

    n = len(records)
    n_with_cap = sum(1 for r in records if r["capacity"] != "")
    n_low_att = sum(1 for r in records if isinstance(r["attendance_pct"], float)
                    and r["attendance_pct"] < 0.50)
    print()
    print(f"Wrote {OUTPUT_CSV} ({n} matches)")
    print(f"  with capacity:    {n_with_cap}/{n}")
    print(f"  low attendance:   {n_low_att}/{n} (<50%)")

    if n < 50:
        hard_stop("scrape_euro2020_attendance count", f"{n} matches", ">= 50 (expected 51)")

    banner(f"Euro 2020 scrape complete ({n} matches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
