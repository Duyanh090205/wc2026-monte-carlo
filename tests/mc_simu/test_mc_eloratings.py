"""Tests for src/mc_simu/eloratings_client.py — Phase 1 §2.4 scraper."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import mc_simu.eloratings_client as ec


def test_team_url_slug_basic():
    assert ec.team_url_slug("Spain") == "Spain"
    assert ec.team_url_slug("United States") == "United_States"


def test_team_url_slug_strips_diacritics():
    """Confirmed 2026-05-20: Curaçao → Curacao.tsv (HTTP 200)."""
    assert ec.team_url_slug("Curaçao") == "Curacao"
    assert ec.team_url_slug("Türkiye") == "Turkiye"


def test_classify_tournament_known_codes():
    assert ec.classify_tournament("WC") == "world_cup_final"
    assert ec.classify_tournament("EC") == "continental_final"
    assert ec.classify_tournament("CA") == "continental_final"
    assert ec.classify_tournament("NL") == "nations_league"
    assert ec.classify_tournament("F") == "friendly"


def test_classify_tournament_qualifier_suffix():
    assert ec.classify_tournament("WCq") == "qualifier"
    assert ec.classify_tournament("ECq") == "qualifier"
    assert ec.classify_tournament("ACq") == "qualifier"


def test_classify_tournament_unknown_falls_back():
    assert ec.classify_tournament("ZZZ") == "other_tournament"


def test_parse_team_tsv_spain_cached(mc_simu_data_dir: Path):
    """Spain.tsv cached during scrape run — verify parser output."""
    cache = mc_simu_data_dir / "cache" / "eloratings" / "Spain.tsv"
    if not cache.exists():
        pytest.skip("Spain.tsv cache missing — run eloratings_client.py first")
    code_to_name = ec.load_team_codes()
    df = ec.parse_team_tsv(cache.read_text(encoding="utf-8"),
                           team_code="ES", team_name="Spain",
                           code_to_name=code_to_name)
    assert len(df) >= 700, f"Expected ~780 matches, got {len(df)}"
    assert df["team"].unique().tolist() == ["Spain"]
    assert set(df.columns) == {"team", "date", "rating_after", "opponent",
                               "tournament_type", "K", "mov", "gs_self", "gs_opp"}
    # First match: Spain 1-0 Denmark, 1920-08-28, rating after = 2026
    first = df.iloc[0]
    assert first["date"] == "1920-08-28"
    assert first["opponent"] == "Denmark"
    assert first["rating_after"] == 2026
    assert first["gs_self"] == 1
    assert first["gs_opp"] == 0


def test_elo_history_eloratings_csv_present(mc_simu_data_dir: Path):
    """Phase 1 §2.4 final deliverable."""
    csv = mc_simu_data_dir / "elo_history_eloratings.csv"
    if not csv.exists():
        pytest.skip("elo_history_eloratings.csv missing — run eloratings_client.py")
    df = pd.read_csv(csv)
    assert len(df) >= 90_000, f"Expected ~99k matches, got {len(df)}"
    assert df["team"].nunique() >= 240, f"Expected ~244 teams, got {df['team'].nunique()}"
    # Schema compat with elo.history_to_long_df
    assert set(df.columns) >= {"team", "date", "rating_after", "opponent",
                                "tournament_type", "K", "mov", "gs_self", "gs_opp"}


def test_load_team_codes_present(mc_simu_data_dir: Path):
    """en.teams.tsv must be in cache for scraper to work."""
    code_to_name = ec.load_team_codes()
    assert "ES" in code_to_name and "Spain" in code_to_name["ES"]
    assert "BR" in code_to_name and "Brazil" in code_to_name["BR"]
    assert len(code_to_name) >= 300, f"Expected ~330 codes, got {len(code_to_name)}"
