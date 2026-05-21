"""Tests for src/mc_simu/elo_source.py — unified Elo loader.

Verifies the fixes for audit findings BUG 1 + BUG 2 + BUG 5 land correctly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import mc_simu.elo_source as es
from mc_simu.elo import get_rating_as_of


def test_canonical_team_name_known_aliases():
    assert es.canonical_team_name("Czechia") == "Czech Republic"
    assert es.canonical_team_name("Curacao") == "Curaçao"
    assert es.canonical_team_name("Ireland") == "Republic of Ireland"
    assert es.canonical_team_name("Korea") == "South Korea"
    assert es.canonical_team_name("China") == "China PR"


def test_canonical_team_name_passes_through_unknown():
    assert es.canonical_team_name("Spain") == "Spain"  # canonical itself
    assert es.canonical_team_name("Atlantis") == "Atlantis"  # unknown


# ── Loader interface ───────────────────────────────────────────────────────────


def test_load_self_returns_dict_of_dataframes(mc_simu_data_dir: Path):
    if not (mc_simu_data_dir / "matches_1998_2026.csv").exists():
        pytest.skip("matches_1998_2026.csv missing")
    history = es.load_elo_history("self")
    assert isinstance(history, dict) and history
    sample = next(iter(history.values()))
    assert isinstance(sample, pd.DataFrame)
    assert "rating_after" in sample.columns


def test_load_eloratings_returns_dict_of_dataframes(mc_simu_data_dir: Path):
    if not (mc_simu_data_dir / "elo_history_eloratings.csv").exists():
        pytest.skip("elo_history_eloratings.csv missing")
    history = es.load_elo_history("eloratings")
    assert isinstance(history, dict) and history
    sample = next(iter(history.values()))
    assert isinstance(sample, pd.DataFrame)
    assert "rating_after" in sample.columns


def test_load_rejects_unknown_source():
    with pytest.raises(ValueError, match="Unknown source"):
        es.load_elo_history("polymarket")  # type: ignore[arg-type]


# ── BUG 1 regression: dates are parsed (datetime64) not strings ────────────────


def test_eloratings_dates_are_timestamps_not_strings(mc_simu_data_dir: Path):
    """Audit BUG 1 — without parse_dates, get_rating_as_of crashes silently.
    Loader must guarantee datetime dtype regardless of caller awareness.
    """
    if not (mc_simu_data_dir / "elo_history_eloratings.csv").exists():
        pytest.skip("elo_history_eloratings.csv missing")
    history = es.load_elo_history("eloratings")
    spain = history.get("Spain")
    assert spain is not None, "Spain should be in eloratings history"
    assert pd.api.types.is_datetime64_any_dtype(spain["date"]), (
        f"date column must be datetime64, got {spain['date'].dtype}"
    )


def test_eloratings_get_rating_as_of_does_not_raise(mc_simu_data_dir: Path):
    """End-to-end BUG 1 regression — loader output works with get_rating_as_of."""
    if not (mc_simu_data_dir / "elo_history_eloratings.csv").exists():
        pytest.skip("elo_history_eloratings.csv missing")
    history = es.load_elo_history("eloratings")
    # Pre-WC2022 lookup for Argentina should succeed without TypeError
    rating = get_rating_as_of(history, "Argentina", pd.Timestamp("2022-11-20"))
    assert isinstance(rating, float)
    assert 1700 < rating < 2300, f"Argentina pre-WC2022 rating {rating} outside plausible band"


# ── BUG 2 regression: Kaggle names resolve in eloratings-loaded history ────────


def test_bug2_czech_republic_resolves(mc_simu_data_dir: Path):
    """Kaggle 'Czech Republic' must resolve via 'Czechia' alias."""
    if not (mc_simu_data_dir / "elo_history_eloratings.csv").exists():
        pytest.skip("elo_history_eloratings.csv missing")
    history = es.load_elo_history("eloratings")
    rating = get_rating_as_of(history, "Czech Republic", pd.Timestamp("2022-12-01"))
    assert rating != 1500.0, "Czech Republic should resolve via 'Czechia' alias, not fall back to 1500"
    assert 1500 < rating < 2100, f"plausibility band, got {rating}"


def test_bug2_republic_of_ireland_resolves(mc_simu_data_dir: Path):
    if not (mc_simu_data_dir / "elo_history_eloratings.csv").exists():
        pytest.skip("elo_history_eloratings.csv missing")
    history = es.load_elo_history("eloratings")
    rating = get_rating_as_of(history, "Republic of Ireland", pd.Timestamp("2016-06-01"))
    assert rating != 1500.0, "'Republic of Ireland' should resolve via 'Ireland' alias"
    assert 1500 < rating < 2000


def test_bug2_china_pr_resolves(mc_simu_data_dir: Path):
    if not (mc_simu_data_dir / "elo_history_eloratings.csv").exists():
        pytest.skip("elo_history_eloratings.csv missing")
    history = es.load_elo_history("eloratings")
    rating = get_rating_as_of(history, "China PR", pd.Timestamp("2022-12-01"))
    assert rating != 1500.0, "'China PR' should resolve via 'China' alias"


# ── BUG 5 regression: LOTO-CV team coverage ────────────────────────────────────


def test_bug5_loto_team_coverage(mc_simu_data_dir: Path):
    """All Kaggle team names that appear in the 12 LOTO-CV tournaments (per spec
    §0.3: WC 2002-2022 + Euro 2004-2024) must be resolvable in eloratings history.

    Scope tightened 2026-05-20 after Phase 0 deep-audit-bug fix expanded
    matches_1998_2026.csv to full 1872+ range — pre-2002 WC/Euros bring in
    defunct teams (Yugoslavia, Czechoslovakia, German DR) which are out of
    LOTO-CV scope and don't need eloratings entries.
    """
    if not (mc_simu_data_dir / "elo_history_eloratings.csv").exists():
        pytest.skip("elo_history_eloratings.csv missing")
    matches_csv = mc_simu_data_dir / "matches_1998_2026.csv"
    if not matches_csv.exists():
        pytest.skip("matches_1998_2026.csv missing")
    matches = pd.read_csv(matches_csv, parse_dates=["date"])
    # Restrict to the 12 LOTO-CV tournaments per spec §0.3
    loto_years_wc = {2002, 2006, 2010, 2014, 2018, 2022}
    loto_years_euro = {2004, 2008, 2012, 2016, 2021, 2024}  # Euro 2020 played 2021
    loto = matches[
        ((matches["tournament"] == "FIFA World Cup") & matches["date"].dt.year.isin(loto_years_wc)) |
        ((matches["tournament"] == "UEFA Euro") & matches["date"].dt.year.isin(loto_years_euro))
    ]
    loto_teams = set(loto["home_team"]) | set(loto["away_team"])
    history = es.load_elo_history("eloratings")
    missing = sorted(loto_teams - set(history.keys()))
    assert not missing, f"LOTO teams not in eloratings history: {missing}"


def test_bug5_wc2026_team_coverage(mc_simu_data_dir: Path):
    """WC2026 fixture team names must all resolve in eloratings-source."""
    if not (mc_simu_data_dir / "elo_history_eloratings.csv").exists():
        pytest.skip("elo_history_eloratings.csv missing")
    fixt_csv = mc_simu_data_dir / "wc2026_fixtures.csv"
    if not fixt_csv.exists():
        pytest.skip("wc2026_fixtures.csv missing")
    fixt = pd.read_csv(fixt_csv)
    teams = set(fixt["home_team"].dropna()) | set(fixt["away_team"].dropna())
    # Filter placeholder values (TBD/Winner X/Runner-up Y/3rd-place X)
    teams = {t for t in teams if t and not any(t.startswith(p) for p in
             ("Winner", "Runner", "3rd", "TBD"))}
    history = es.load_elo_history("eloratings")
    missing = sorted(teams - set(history.keys()))
    assert not missing, f"WC2026 fixture teams not in eloratings history: {missing}"


# ── Alias key access ───────────────────────────────────────────────────────────


def test_eloratings_history_resolves_both_canonical_and_alias_keys(mc_simu_data_dir: Path):
    """Caller using either name convention gets the same DataFrame."""
    if not (mc_simu_data_dir / "elo_history_eloratings.csv").exists():
        pytest.skip("elo_history_eloratings.csv missing")
    history = es.load_elo_history("eloratings")
    a = history.get("Czech Republic")
    b = history.get("Czechia")
    assert a is not None and b is not None
    assert a is b, "Alias key must point to same DataFrame object as canonical"


def test_self_history_resolves_both_canonical_and_alias_keys(mc_simu_data_dir: Path):
    if not (mc_simu_data_dir / "matches_1998_2026.csv").exists():
        pytest.skip("matches_1998_2026.csv missing")
    history = es.load_elo_history("self")
    a = history.get("Czech Republic")  # canonical (matches CSV native name)
    b = history.get("Czechia")          # alias (fixtures/eloratings name)
    assert a is not None
    assert b is not None and a is b
