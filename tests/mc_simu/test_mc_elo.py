"""Tests for src/mc_simu/elo.py — Phase 1 §2.1 Steps 1.1–1.3."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from mc_simu.elo import (
    K_FACTORS,
    build_rating_history,
    elo_expected,
    elo_update,
    get_rating_as_of,
    latest_ratings,
    mov_multiplier,
)


# ── K_FACTORS sanity ──────────────────────────────────────────────────────────


def test_k_factors_exact_six_buckets() -> None:
    assert set(K_FACTORS.keys()) == {
        "world_cup_final", "continental_final", "qualifier",
        "nations_league", "other_tournament", "friendly",
    }
    assert K_FACTORS["world_cup_final"] == 60
    assert K_FACTORS["continental_final"] == 50
    assert K_FACTORS["qualifier"] == 40
    assert K_FACTORS["nations_league"] == 40
    assert K_FACTORS["other_tournament"] == 30
    assert K_FACTORS["friendly"] == 20


# ── elo_expected ──────────────────────────────────────────────────────────────


class TestEloExpected:
    def test_equal_ratings_neutral_is_half(self) -> None:
        assert elo_expected(1500, 1500, is_neutral=True) == pytest.approx(0.5, abs=1e-12)

    def test_equal_ratings_with_hfa_about_064(self) -> None:
        """+100 Elo HFA at parity ≈ 0.640 (textbook value)."""
        assert elo_expected(1500, 1500, hfa_elo=100, is_neutral=False) == pytest.approx(0.640, abs=0.001)

    def test_plus_400_neutral_about_091(self) -> None:
        """+400 Elo diff (no HFA) → ~0.909 (10:1 odds)."""
        assert elo_expected(1900, 1500, is_neutral=True) == pytest.approx(0.909, abs=0.01)

    def test_symmetry_under_swap_neutral(self) -> None:
        """elo_expected(a, b) + elo_expected(b, a) == 1 when neutral."""
        e_ab = elo_expected(1700, 1500, is_neutral=True)
        e_ba = elo_expected(1500, 1700, is_neutral=True)
        assert e_ab + e_ba == pytest.approx(1.0, abs=1e-12)


# ── mov_multiplier ────────────────────────────────────────────────────────────


class TestMovMultiplier:
    @pytest.mark.parametrize("gd,expected", [
        (0, 1.0), (1, 1.0), (-1, 1.0),
        (2, 1.5), (-2, 1.5),
        (3, 1.75),
        (4, 1.875),    # 1.75 + 1/8
        (5, 2.0),      # 1.75 + 2/8
        (6, 2.125),
    ])
    def test_known_values(self, gd: int, expected: float) -> None:
        assert mov_multiplier(gd) == pytest.approx(expected, abs=1e-12)


# ── elo_update ────────────────────────────────────────────────────────────────


class TestEloUpdate:
    def test_zero_sum_within_match(self) -> None:
        """Δ_home + Δ_away = 0 (Elo is zero-sum per match)."""
        rh, ra = 1700.0, 1500.0
        e_h = elo_expected(rh, ra, hfa_elo=0, is_neutral=True)
        e_a = 1.0 - e_h
        K, mov = 40, 1.5
        # home win 3-1
        new_rh = elo_update(rh, e_h, 1.0, K, mov)
        new_ra = elo_update(ra, e_a, 0.0, K, mov)
        assert (new_rh - rh) + (new_ra - ra) == pytest.approx(0.0, abs=1e-12)

    def test_monotonic_in_K(self) -> None:
        """Bigger K → bigger update magnitude when actual > expected."""
        small = elo_update(1500, 0.5, 1.0, K=20, mov_mult=1.0)
        big   = elo_update(1500, 0.5, 1.0, K=60, mov_mult=1.0)
        assert big - 1500 > small - 1500 > 0

    def test_no_change_when_actual_matches_expected(self) -> None:
        """If expected == actual, rating is unchanged."""
        new = elo_update(1500, 0.5, 0.5, K=40, mov_mult=1.5)
        assert new == pytest.approx(1500, abs=1e-12)


# ── build_rating_history + get_rating_as_of ───────────────────────────────────


def _synthetic_matches() -> pd.DataFrame:
    """5 teams playing a small round-robin of bilateral friendlies."""
    rows = []
    teams = ["A", "B", "C", "D", "E"]
    date = pd.Timestamp("2020-01-01")
    for i, h in enumerate(teams):
        for j, a in enumerate(teams):
            if i >= j:
                continue
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "home_team": h, "away_team": a,
                "home_score": 1 + i, "away_score": j,
                "tournament": "Friendly", "neutral": "FALSE",
            })
            date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


class TestBuildRatingHistory:
    def test_returns_per_team_dataframes(self) -> None:
        df = _synthetic_matches()
        hist = build_rating_history(df)
        assert set(hist.keys()) == {"A", "B", "C", "D", "E"}
        for team_df in hist.values():
            assert "rating_after" in team_df.columns
            assert "tournament_type" in team_df.columns
            assert (team_df["tournament_type"] == "friendly").all()

    def test_total_rating_preserved_by_zero_sum(self) -> None:
        """Elo zero-sum per match → sum of final ratings == n × initial_rating."""
        df = _synthetic_matches()
        hist = build_rating_history(df, initial_rating=1500.0)
        latest = latest_ratings(hist)
        assert sum(latest.values()) == pytest.approx(5 * 1500.0, abs=1e-9)

    def test_drops_nan_score_rows(self) -> None:
        df = _synthetic_matches()
        # Inject a NaN-score row (future fixture)
        df = pd.concat([df, pd.DataFrame([{
            "date": "2030-01-01",
            "home_team": "A", "away_team": "B",
            "home_score": math.nan, "away_score": math.nan,
            "tournament": "FIFA World Cup", "neutral": "TRUE",
        }])], ignore_index=True)
        hist = build_rating_history(df)
        # No 2030 entry should appear in any team's history
        for team_df in hist.values():
            assert (pd.to_datetime(team_df["date"]) < pd.Timestamp("2030-01-01")).all()

    def test_drops_future_dated_rows(self) -> None:
        df = _synthetic_matches()
        hist = build_rating_history(df, cutoff_date=pd.Timestamp("2020-01-15"))
        # Only matches dated ≤ 2020-01-15 in any team's series
        for team_df in hist.values():
            assert (team_df["date"] <= pd.Timestamp("2020-01-15")).all()


class TestGetRatingAsOf:
    def test_unseen_team_returns_initial(self) -> None:
        hist: dict[str, pd.DataFrame] = {}
        assert get_rating_as_of(hist, "Nowhere", pd.Timestamp("2025-01-01")) == 1500.0

    def test_strict_less_than(self) -> None:
        """Query at match-date returns rating BEFORE that match (no lookahead)."""
        df = _synthetic_matches()
        hist = build_rating_history(df)
        # First match in synthetic series is 2020-01-01 A vs B → rating at 2020-01-01 must be 1500 (no prior matches)
        assert get_rating_as_of(hist, "A", pd.Timestamp("2020-01-01")) == 1500.0
        # By 2020-01-08 (day after first match), A has played → rating ≠ 1500
        r_after_first = get_rating_as_of(hist, "A", pd.Timestamp("2020-01-08"))
        assert r_after_first != 1500.0
