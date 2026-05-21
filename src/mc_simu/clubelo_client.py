"""ClubElo HTTP client + JSON cache.

API: http://api.clubelo.com/{ClubName} (free, no auth).
Response: CSV with columns Rank, Club, Country, Level, Elo, From, To.

Cache: data/mc_simu/clubelo_cache.json (idempotent — Elo doesn't change retroactively
for past dates, so caching the latest available rating is safe within a session).

Pulled forward from Phase 2 §2.1 because Phase 0 Check 5 (ClubElo coverage) needs it.
Phase 2 will extend with z-normalize + blend logic.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import requests  # noqa: E402

from mc_simu._common import banner  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
CACHE_PATH = DATA_DIR / "clubelo_cache.json"

API_BASE = "http://api.clubelo.com"

_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if CACHE_PATH.exists():
        _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    else:
        _cache = {
            "_meta": {
                "fetched_at": "",
                "source": API_BASE,
                "note": "Cache populated lazily by get_clubelo() calls.",
            }
        }
    return _cache


def _save_cache() -> None:
    cache = _load_cache()
    cache["_meta"]["fetched_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# Manual aliases — ClubElo uses short native names. Extend as Check 5 reveals misses.
NAME_ALIASES = {
    # Germany
    "Bayern Munich": "Bayern",
    "FC Bayern Munich": "Bayern",
    "Borussia Dortmund": "Dortmund",
    "Borussia Mönchengladbach": "Gladbach",
    "Bayer Leverkusen": "Leverkusen",
    "RB Leipzig": "RBLeipzig",
    "Eintracht Frankfurt": "Frankfurt",
    # France
    "Paris Saint-Germain": "ParisSG",
    # England — ClubElo uses short forms
    "Manchester United": "ManUnited",
    "Manchester City": "ManCity",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle",
    "West Ham United": "WestHam",
    "Brighton & Hove Albion": "Brighton",
    "Crystal Palace": "CrystalPalace",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Forest",
    "Aston Villa": "AstonVilla",
    "Sheffield United": "SheffieldUnited",
    "Leicester City": "Leicester",
    # Spain
    "Atletico Madrid": "Atletico",
    "Atlético Madrid": "Atletico",
    "Real Sociedad": "Sociedad",
    "Real Betis": "Betis",
    "Athletic Bilbao": "Bilbao",
    "Athletic Club": "Bilbao",
    # Italy
    "Inter Milan": "Inter",
    "AC Milan": "Milan",
    # Czech Republic — ClubElo uses Czech-language form
    "Slavia Prague": "SlaviaPraha",
    "Sparta Prague": "SpartaPraha",
    "Viktoria Plzeň": "ViktoriaPlzen",
    "Viktoria Plzen": "ViktoriaPlzen",
    "RB Salzburg": "Salzburg",
    "Red Bull Salzburg": "Salzburg",
    "Baník Ostrava": "BanikOstrava",
    "Banik Ostrava": "BanikOstrava",
    "Sigma Olomouc": "SigmaOlomouc",
    # Turkey
    "Fenerbahçe": "Fenerbahce",
    "Galatasaray": "Galatasaray",
    "Beşiktaş": "Besiktas",
    "Trabzonspor": "Trabzonspor",
    # Netherlands
    "Feyenoord": "Feyenoord",
    "PSV Eindhoven": "PSV",
    "AZ": "Alkmaar",
    "AZ Alkmaar": "Alkmaar",
    # Belgium
    "Club Brugge": "Brugge",
    "Royal Antwerp": "Antwerp",
    # Portugal
    "Sporting CP": "Sporting",
    "Sporting Lisbon": "Sporting",
    # Poland
    "Górnik Zabrze": "GornikZabrze",
    "Gornik Zabrze": "GornikZabrze",
    "Legia Warsaw": "Legia",
    "Lech Poznań": "Lech",
}


def _strip_diacritics(s: str) -> str:
    """Best-effort ASCII fold for names like 'Fenerbahçe' → 'Fenerbahce'."""
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _strip_prefix(name: str) -> str:
    for p in ("FC ", "AFC ", "AC ", "CF ", "SC ", "VfL ", "VfB ", "TSG ", "Club "):
        if name.startswith(p):
            return name[len(p):]
    return name


def _candidate_url_names(name: str) -> list[str]:
    """Return ordered list of name forms to try against the ClubElo API."""
    if name in NAME_ALIASES:
        return [NAME_ALIASES[name]]
    # Try alias on ASCII-folded form too (handles "Fenerbahçe" → "Fenerbahce" alias)
    ascii_name = _strip_diacritics(name)
    if ascii_name != name and ascii_name in NAME_ALIASES:
        return [NAME_ALIASES[ascii_name]]
    base = _strip_prefix(name).strip()
    base_ascii = _strip_diacritics(base)
    candidates: list[str] = []
    # Concatenated full (both original and ASCII)
    candidates.append("".join(base.split()))
    if base_ascii != base:
        candidates.append("".join(base_ascii.split()))
    # Last word only ("Borussia Dortmund" -> "Dortmund")
    parts = base.split()
    if len(parts) > 1:
        candidates.append(parts[-1])
        # First word only ("Real Madrid" -> "Real" — last resort, often wrong but cheap)
        candidates.append(parts[0])
    # Original with spaces (URL-encoded by requests)
    candidates.append(base)
    # Dedup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _fetch_club_history(club_name: str) -> list[dict] | None:
    """Fetch raw CSV from api.clubelo.com and parse to list of dicts.

    Tries multiple URL name forms (per ClubElo's short-name convention).
    Returns None if no form yields data.
    """
    for url_name in _candidate_url_names(club_name):
        url = f"{API_BASE}/{url_name}"
        try:
            resp = requests.get(url, timeout=15)
        except requests.RequestException:
            continue
        if resp.status_code != 200 or not resp.text.strip():
            continue
        reader = csv.DictReader(io.StringIO(resp.text))
        rows = list(reader)
        if rows and rows[0].get("Elo"):
            return rows
    return None


def get_clubelo(club_name: str, as_of: date | None = None) -> float | None:
    """Return latest ClubElo for `club_name`, or None if not found.

    `as_of` accepted for API compatibility with Phase 2 (which will do history-aware
    lookup). Phase 0 only needs existence, so the latest cached rating is returned.

    Uses on-disk cache; first call per club triggers an HTTP fetch.
    """
    # `as_of` is reserved for Phase 2 history-aware lookup; validate type now.
    if as_of is not None and not isinstance(as_of, date):
        raise TypeError(f"as_of must be a date, got {type(as_of).__name__}")
    cache = _load_cache()
    key = club_name.strip()
    if not key:
        return None

    entry = cache.get(key)
    if entry is None:
        rows = _fetch_club_history(key)
        if rows is None:
            # Cache the miss too so we don't re-fetch every call.
            cache[key] = {"latest_elo": None, "as_of": "", "note": "not found"}
            _save_cache()
            return None
        latest_row = rows[-1]
        try:
            elo = float(latest_row["Elo"])
        except (ValueError, KeyError):
            elo = None
        entry = {
            "latest_elo": elo,
            "as_of": latest_row.get("To") or latest_row.get("From") or "",
            "rank": latest_row.get("Rank"),
            "country": latest_row.get("Country"),
            "level": latest_row.get("Level"),
        }
        cache[key] = entry
        _save_cache()

    elo = entry.get("latest_elo")
    if elo is None:
        return None

    # If as_of is provided AND latest cache entry is newer, the cache value is still
    # acceptable per our note above (Elo doesn't change retroactively for past dates).
    # Phase 2 will extend with history-aware lookup; Phase 0 needs only existence.
    return float(elo) if elo is not None else None


def main() -> int:
    """CLI: quick smoke test fetching 3 well-known clubs."""
    banner("MC Simu — ClubElo Client smoke test")
    for club in ["Real Madrid", "Barcelona", "Bayern Munich"]:
        elo = get_clubelo(club)
        print(f"  {club:25s} -> {elo}")
    cache = _load_cache()
    n_clubs = sum(1 for k in cache if not k.startswith("_"))
    print(f"\nCache file: {CACHE_PATH}")
    print(f"  entries: {n_clubs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
