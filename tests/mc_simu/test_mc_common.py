"""Tests for src/mc_simu/_common.py — hard_stop, banner, infer_tournament_type."""

from __future__ import annotations

import pytest

from mc_simu._common import infer_tournament_type


class TestInferTournamentType:
    """Heuristic must route Kaggle `tournament` strings to the 6 K_FACTORS buckets.

    Bug history: pre-hotfix the heuristic only caught 'fifa world cup' and 'uefa euro'
    as majors. AFCON / Copa America / Gold Cup / AFC Asian Cup / Confederations Cup /
    OFC Nations Cup all fell into 'friendly' (K=20) instead of 'continental_final' (K=50);
    their qualifiers fell into 'friendly' instead of 'qualifier' (K=40). ~6000 matches
    mis-K'd. Tests below cover the regression-prone strings.
    """

    @pytest.mark.parametrize("tournament,expected", [
        # K=60: FIFA WC finals only
        ("FIFA World Cup", "world_cup_final"),
        # K=50: continental finals (all confederations) + Confederations Cup
        ("UEFA Euro", "continental_final"),
        ("Copa América", "continental_final"),
        ("Copa America", "continental_final"),
        ("African Cup of Nations", "continental_final"),
        ("Africa Cup of Nations", "continental_final"),
        ("AFC Asian Cup", "continental_final"),
        ("Gold Cup", "continental_final"),
        ("Oceania Nations Cup", "continental_final"),
        ("Confederations Cup", "continental_final"),
        ("FIFA Confederations Cup", "continental_final"),
        # K=40: qualifiers (all confederations) — qualifier wins over continental
        ("FIFA World Cup qualification", "qualifier"),
        ("UEFA Euro qualification", "qualifier"),
        ("African Cup of Nations qualification", "qualifier"),
        ("AFC Asian Cup qualification", "qualifier"),
        ("CFU Caribbean Cup qualification", "qualifier"),
        # K=40: Nations Leagues
        ("UEFA Nations League", "nations_league"),
        ("CONCACAF Nations League", "nations_league"),
        # K=30: other competitive (sub-confed + invitationals + Olympic + CHAN + CONIFA)
        ("Kirin Cup", "other_tournament"),
        ("King's Cup", "other_tournament"),
        ("China Cup", "other_tournament"),
        ("CECAFA Cup", "other_tournament"),
        ("COSAFA Cup", "other_tournament"),
        ("AFF Championship", "other_tournament"),
        ("Gulf Cup", "other_tournament"),
        ("SAFF Cup", "other_tournament"),
        ("EAFF Championship", "other_tournament"),
        ("CFU Caribbean Cup", "other_tournament"),
        ("UNCAF Cup", "other_tournament"),
        ("WAFF Championship", "other_tournament"),
        ("Arab Cup", "other_tournament"),
        ("Cyprus International Tournament", "other_tournament"),
        ("Olympic Games", "other_tournament"),
        ("African Nations Championship", "other_tournament"),
        ("CONIFA World Football Cup", "other_tournament"),
        ("Island Games", "other_tournament"),
        ("AFC Challenge Cup", "other_tournament"),
        # K=20: bilateral friendlies + unknown fallback
        ("Friendly", "friendly"),
        ("Test Tournament XYZ", "friendly"),  # unknown → fallback
        ("", "friendly"),                     # empty → fallback
    ])
    def test_classification(self, tournament: str, expected: str) -> None:
        assert infer_tournament_type(tournament) == expected

    def test_k_factors_keys_match_categories(self) -> None:
        """K_FACTORS in elo.py must have exactly the 6 keys returned by infer_tournament_type.

        Catches drift if either side adds/renames a bucket. (elo.py is imported lazily —
        skip if not yet built.)
        """
        try:
            from mc_simu.elo import K_FACTORS
        except ImportError:
            pytest.skip("elo.py not yet built")

        expected_keys = {
            "world_cup_final", "continental_final", "qualifier",
            "nations_league", "other_tournament", "friendly",
        }
        assert set(K_FACTORS.keys()) == expected_keys
