"""eloratings.net per-team match history scraper (Phase 1 §2.4).

Endpoints discovered 2026-05-20 by reverse-engineering scripts/ratings.js:
- https://eloratings.net/en.teams.tsv         — team code → English name(s) (~330 entries)
- https://eloratings.net/en.tournaments.tsv   — tournament code → English name (~656 codes)
- https://eloratings.net/World.tsv            — current global rankings (~243 active teams)
- https://eloratings.net/{Name_with_underscores}.tsv — per-team match history

Per-team TSV row format (16 cols, observed from Spain.tsv):
    year | month | day | team1_code | team2_code | score1 | score2 |
    tournament_code | neutral_host_code | rating_change_team1 |
    rating_after_team1 | rating_after_team2 |
    rank_change_team1 | rank_change_team2 | rank_after_team1 | rank_after_team2

Output schema (drop-in compat with elo.history_to_long_df()):
    team, date, rating_after, opponent, tournament_type, K, mov, gs_self, gs_opp

Cache: data/mc_simu/cache/eloratings/{slug}.tsv — idempotent (re-fetch only with --force).
"""
from __future__ import annotations

import argparse
import sys
import time
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from mc_simu._common import banner  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
CACHE_DIR = DATA_DIR / "cache" / "eloratings"
OUTPUT_CSV = DATA_DIR / "elo_history_eloratings.csv"

BASE_URL = "https://eloratings.net"
USER_AGENT = "Mozilla/5.0 (research-scraper; contact: nguyenanhkt9205@gmail.com)"
THROTTLE_SEC = 0.5  # polite throttling between requests

# Tournament code → K-factor bucket. Maps eloratings.net 656 codes to our 6 buckets.
# Conservative defaults: end-in-q → qualifier, fallback → other_tournament.
TOURNAMENT_TYPE_OVERRIDES: dict[str, str] = {
    # Top-level championships
    "WC": "world_cup_final",
    "EC": "continental_final",
    "CA": "continental_final",  # Copa America
    "AR": "continental_final",  # African Cup of Nations
    "AC": "continental_final",  # Asian Cup
    "CGC": "continental_final",  # CONCACAF Gold Cup
    "OC": "continental_final",  # Oceania Nations Cup
    "CC": "continental_final",  # Confederations Cup
    "NL": "nations_league",
    "NLF": "nations_league",
    "F": "friendly",
    "OG": "other_tournament",
}


def classify_tournament(code: str) -> str:
    """Map eloratings.net tournament code → our K_FACTORS bucket key."""
    if code in TOURNAMENT_TYPE_OVERRIDES:
        return TOURNAMENT_TYPE_OVERRIDES[code]
    # Generic rules
    if code.endswith("q") or code.endswith("Q"):
        return "qualifier"
    if code.startswith("F"):
        return "friendly"
    # Default catch-all for the long tail of regional tournaments
    return "other_tournament"


def _http_get(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.content


def _cached_fetch(slug: str, *, force: bool = False) -> str:
    """Fetch a TSV with on-disk cache. Returns text body."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{slug}.tsv"
    if cache_path.exists() and not force:
        return cache_path.read_text(encoding="utf-8")
    body = _http_get(f"{BASE_URL}/{slug}.tsv")
    cache_path.write_bytes(body)
    time.sleep(THROTTLE_SEC)
    return body.decode("utf-8")


def load_team_codes() -> dict[str, list[str]]:
    """Return {code: [primary_name, alias1, alias2, ...]} from en.teams.tsv."""
    body = _cached_fetch("en.teams")
    out: dict[str, list[str]] = {}
    for line in body.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        code, *names = parts
        out[code] = names
    return out


def load_world_ranking() -> pd.DataFrame:
    """Return World.tsv as DataFrame with at least: rank, code, current_rating."""
    body = _cached_fetch("World")
    rows = []
    for line in body.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        rows.append({
            "rank": int(parts[0]),
            "code": parts[2],
            "current_rating": int(parts[3]),
        })
    return pd.DataFrame(rows)


def team_url_slug(name: str) -> str:
    """eloratings.net URL convention: spaces → underscores, diacritics stripped.

    Confirmed 2026-05-20: Curaçao → Curacao.tsv (HTTP 200), Cura%C3%A7ao.tsv → 404.
    NFD decomposes letters into base + combining mark; we drop the combining marks.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return ascii_only.replace(" ", "_")


def parse_team_tsv(body: str, *, team_code: str, team_name: str,
                   code_to_name: dict[str, list[str]]) -> pd.DataFrame:
    """Parse one team's .tsv into the elo.history_to_long_df schema."""
    rows = []
    for raw in body.strip().splitlines():
        parts = raw.split("\t")
        if len(parts) < 12:
            continue  # malformed/short rows
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        # eloratings.net uses month=0 / day=0 to mean "approximate / unknown".
        # ~547/99k rows affected (mostly pre-1980 friendlies). Coerce to first
        # of month/year so parse_dates downstream succeeds. Without this, ONE
        # bad row silently disables parse_dates for the entire column.
        if month == 0:
            month = 1
        if day == 0:
            day = 1
        team1_code, team2_code = parts[3], parts[4]
        score1, score2 = int(parts[5]), int(parts[6])
        tourn_code = parts[7]
        # parts[8] = neutral host code (may be empty)
        rating_after_t1 = int(parts[10])
        rating_after_t2 = int(parts[11])

        # Determine "self" vs "opponent" from team_code
        if team1_code == team_code:
            opp_code = team2_code
            rating_after = rating_after_t1
            gs_self, gs_opp = score1, score2
        elif team2_code == team_code:
            opp_code = team1_code
            rating_after = rating_after_t2
            gs_self, gs_opp = score2, score1
        else:
            # Some rows in the team's own .tsv may include matches where the
            # team appears with a code variant — skip defensively.
            continue

        opp_name = (code_to_name.get(opp_code) or [opp_code])[0]
        tt = classify_tournament(tourn_code)
        gd = abs(gs_self - gs_opp)
        mov = 1.0 if gd <= 1 else 1.5 if gd == 2 else 1.75 if gd == 3 else 1.75 + (gd - 3) / 8.0
        # K-factor: piggyback off elo.K_FACTORS (kept consistent with self-compute)
        from mc_simu.elo import K_FACTORS
        K = K_FACTORS.get(tt, 20)

        rows.append({
            "team": team_name,
            "date": f"{year:04d}-{month:02d}-{day:02d}",
            "rating_after": rating_after,
            "opponent": opp_name,
            "tournament_type": tt,
            "K": K,
            "mov": mov,
            "gs_self": gs_self,
            "gs_opp": gs_opp,
        })
    return pd.DataFrame(rows)


def scrape_team(team_code: str, team_name: str,
                code_to_name: dict[str, list[str]],
                *, force: bool = False) -> pd.DataFrame:
    """Fetch + parse one team. Returns history DataFrame."""
    slug = team_url_slug(team_name)
    body = _cached_fetch(slug, force=force)
    return parse_team_tsv(body, team_code=team_code, team_name=team_name,
                          code_to_name=code_to_name)


def scrape_all(*, force: bool = False, limit: int | None = None) -> pd.DataFrame:
    """Scrape ratings for all teams in World.tsv (243 active nations).

    Writes data/mc_simu/elo_history_eloratings.csv (long format).
    """
    banner("eloratings.net full scrape")
    code_to_name = load_team_codes()
    world = load_world_ranking()
    if limit:
        world = world.head(limit)
    print(f"  Targets: {len(world)} teams from World.tsv")

    all_parts: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    for i, row in enumerate(world.itertuples(index=False), 1):
        code = row.code
        names = code_to_name.get(code)
        if not names:
            failures.append((code, "no name in en.teams.tsv"))
            continue
        primary = names[0]
        try:
            df = scrape_team(code, primary, code_to_name, force=force)
        except requests.HTTPError as e:
            failures.append((code, f"HTTP {e.response.status_code} for slug {team_url_slug(primary)}"))
            continue
        if df.empty:
            failures.append((code, f"empty parse for {primary}"))
            continue
        all_parts.append(df)
        if i % 10 == 0:
            print(f"  [{i}/{len(world)}] {primary}: {len(df)} matches")

    if not all_parts:
        print("ERROR: no teams scraped successfully")
        return pd.DataFrame()
    big = pd.concat(all_parts, ignore_index=True)
    big.sort_values(["team", "date"], inplace=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    big.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Wrote {OUTPUT_CSV} — {len(big)} rows, {big['team'].nunique()} teams")
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for code, reason in failures[:20]:
            print(f"    {code}: {reason}")
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more")
    return big


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="Bypass cache; re-fetch all")
    ap.add_argument("--limit", type=int, default=None, help="Only scrape top-N teams (testing)")
    ap.add_argument("--team", type=str, default=None,
                    help="Test single team by primary name (e.g., 'Spain')")
    args = ap.parse_args()

    if args.team:
        code_to_name = load_team_codes()
        # Reverse lookup name → code
        target_code = None
        for code, names in code_to_name.items():
            if args.team in names:
                target_code = code
                break
        if target_code is None:
            print(f"ERROR: '{args.team}' not found in en.teams.tsv")
            return 1
        banner(f"eloratings.net — single team: {args.team} ({target_code})")
        df = scrape_team(target_code, args.team, code_to_name, force=args.force)
        print(f"  {len(df)} matches scraped")
        print(df.head())
        print("  ...")
        print(df.tail())
        return 0

    scrape_all(force=args.force, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
