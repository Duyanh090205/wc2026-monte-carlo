"""Unified Elo source loader — bridges self-compute and eloratings.net.

Plug-in interface per Sam's framework (mc_simu.md §2.4 + Phase 3 §3.5):

    history = load_elo_history(source='self')        # build from matches_1998_2026.csv
    history = load_elo_history(source='eloratings')  # load from elo_history_eloratings.csv

Both return `dict[team_name, DataFrame]` consumable by `elo.get_rating_as_of()`.

# Team-name canonicalization

The data files inside this project use **inconsistent team-name conventions**
across files (audit_phase1_wiring.py 2026-05-20 confirmed empirically):

  matches_1998_2026.csv  →  "Czech Republic", "South Korea", "China PR",
                            "Republic of Ireland", "Curaçao", "Turkey"
  wc2026_fixtures.csv    →  "Czechia",        "South Korea", (n/a),
                            (n/a),                "Curacao",  "Turkey"
  elo_history_elor*.csv  →  "Czechia",        "Korea",       "China",
                            "Ireland",            "Curaçao",  "Turkey"

To make callers indifferent to source AND robust to caller's name choice, the
loader returns a history dict that has **alias keys** — every known synonym
points to the same underlying DataFrame. Callers can pass any of
{"Czech Republic", "Czechia"} and get the right result.

Canonical convention internally = matches_1998_2026.csv (the larger dataset,
the file that build_rating_history() produces directly).

Fixes for audit findings 2026-05-20:
  BUG 1 — Always reads eloratings CSV with parse_dates=["date"]. Scraper also
          coerces day=0/month=0 so all rows parse cleanly.
  BUG 2 — Alias keys in history dict resolve cross-source naming.
  BUG 5 — Subsumed by BUG 2 fix.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from mc_simu.elo import build_rating_history  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
MATCHES_CSV = DATA_DIR / "matches_1998_2026.csv"
ELORATINGS_CSV = DATA_DIR / "elo_history_eloratings.csv"

# alias → canonical (matches_1998_2026.csv convention).
# Canonical name itself is OMITTED from this map (it's an identity mapping).
# Aliases are added as additional keys to history dict on load.
TEAM_ALIASES: dict[str, str] = {
    # Czechia / Czech Republic — fixtures + eloratings use "Czechia"
    "Czechia": "Czech Republic",
    # Curacao / Curaçao — fixtures uses ASCII, matches+eloratings use diacritic
    "Curacao": "Curaçao",
    # Korea — eloratings uses short forms
    "Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    # Other ISO-vs-FIFA name differences (eloratings convention → Kaggle)
    "Ireland": "Republic of Ireland",
    "China": "China PR",
    "USA": "United States",
    "Cape Verde Islands": "Cape Verde",
    "Cote d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
}


Source = Literal["self", "eloratings"]


def canonical_team_name(name: str) -> str:
    """Return matches_1998_2026.csv canonical name for any known alias.
    Unknown names pass through unchanged.
    """
    return TEAM_ALIASES.get(name, name)


def _add_alias_keys(history: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """For each TEAM_ALIASES entry, add the alias key pointing to the same
    DataFrame as the canonical key (if canonical exists). Mutates and returns.
    """
    for alias, canonical in TEAM_ALIASES.items():
        if canonical in history and alias not in history:
            history[alias] = history[canonical]
        # Also handle reverse direction — if only the alias key is present
        # (e.g., eloratings source has "Czechia" but not yet renamed),
        # add canonical pointing to same data.
        elif alias in history and canonical not in history:
            history[canonical] = history[alias]
    return history


def _normalize_dataframe_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite team + opponent columns to canonical names. Returns NEW DataFrame."""
    out = df.copy()
    out["team"] = out["team"].replace(TEAM_ALIASES)
    out["opponent"] = out["opponent"].replace(TEAM_ALIASES)
    return out


def load_elo_history(source: Source = "self") -> dict[str, pd.DataFrame]:
    """Load per-team Elo history from the specified source.

    Returns dict[team_name → DataFrame(date, rating_after, opponent,
    tournament_type, K, mov, gs_self, gs_opp)] usable by get_rating_as_of().

    Both canonical names AND known aliases are keys in the returned dict (e.g.,
    history["Czech Republic"] and history["Czechia"] return the same DataFrame).

    Args:
        source: 'self' → build from matches_1998_2026.csv via build_rating_history
                'eloratings' → load elo_history_eloratings.csv

    Raises FileNotFoundError if input missing, ValueError on unknown source.
    """
    if source == "self":
        if not MATCHES_CSV.exists():
            raise FileNotFoundError(
                f"{MATCHES_CSV} missing — run download_match_history.py + data_audit.py"
            )
        matches = pd.read_csv(MATCHES_CSV)
        history = build_rating_history(matches)
        return _add_alias_keys(history)

    if source == "eloratings":
        if not ELORATINGS_CSV.exists():
            raise FileNotFoundError(
                f"{ELORATINGS_CSV} missing — run eloratings_client.py"
            )
        # parse_dates is critical: without it, get_rating_as_of's
        # `df["date"] < pd.Timestamp(...)` raises TypeError (audit BUG 1).
        df = pd.read_csv(ELORATINGS_CSV, parse_dates=["date"])
        df = _normalize_dataframe_names(df)
        history = {team: g.reset_index(drop=True) for team, g in df.groupby("team")}
        return _add_alias_keys(history)

    raise ValueError(f"Unknown source: {source!r} (must be 'self' or 'eloratings')")


__all__ = [
    "TEAM_ALIASES",
    "canonical_team_name",
    "load_elo_history",
]
