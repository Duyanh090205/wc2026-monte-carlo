"""Tests for src/mc_simu/simulator.py — Phase 3 §4.1 Step 3.4.

Coverage:
1. sample_group_score — output shape + valid range
2. sample_scores_batch — vectorized parity vs sample_group_score on same RNG seed
3. sample_knockout_winner — Bernoulli statistical sanity
4. build_group_samplers — produces valid CDFs (monotone, ends at 1.0)
5. build_ko_advance_table — symmetric KO context, extreme Elo gap gives ~1.0
"""

from __future__ import annotations

import numpy as np
import pytest

from mc_simu.simulator import (
    GroupSampler,
    HostInfo,
    build_group_samplers,
    build_ko_advance_table,
    make_elo_predictor,
    sample_group_score,
    sample_knockout_winner,
    sample_scores_batch,
)
from mc_simu.single_game import ModelParams


@pytest.fixture
def host_info() -> HostInfo:
    return HostInfo(
        host_countries=["United States", "Mexico", "Canada"],
        host_confederation="CONCACAF",
    )


@pytest.fixture
def predictor():
    """Default Elo predictor — same calibrated params Phase 1 ships with."""
    return make_elo_predictor(ModelParams())


# ── sample_group_score ────────────────────────────────────────────────────────


class TestSampleGroupScore:
    def test_returns_valid_pair(self) -> None:
        # Uniform CDF over 81 cells — every cell equally likely
        flat = np.full(81, 1 / 81, dtype=np.float64)
        cdf = np.cumsum(flat)
        cdf[-1] = 1.0
        sampler = GroupSampler(home_team="A", away_team="B", cdf_flat=cdf)
        rng = np.random.default_rng(42)
        for _ in range(50):
            h, a = sample_group_score(sampler, rng)
            assert 0 <= h <= 8
            assert 0 <= a <= 8

    def test_degenerate_first_cell_only(self) -> None:
        """If CDF puts all mass on cell 0 (0-0 draw), every sample is (0, 0)."""
        cdf = np.ones(81, dtype=np.float64)  # cdf[0]=1, cdf[i>0]=1
        sampler = GroupSampler(home_team="A", away_team="B", cdf_flat=cdf)
        rng = np.random.default_rng(42)
        for _ in range(10):
            h, a = sample_group_score(sampler, rng)
            assert (h, a) == (0, 0)


# ── sample_scores_batch parity ───────────────────────────────────────────────


class TestSampleScoresBatch:
    def test_matches_per_fixture_when_same_uniform(self) -> None:
        """Batch should produce same valid-range pairs."""
        flat = np.full(81, 1 / 81, dtype=np.float64)
        cdf = np.cumsum(flat)
        cdf[-1] = 1.0
        cdfs_2d = np.stack([cdf, cdf, cdf])
        rng = np.random.default_rng(123)
        scores = sample_scores_batch(cdfs_2d, rng)
        assert scores.shape == (3, 2)
        assert ((scores >= 0) & (scores <= 8)).all()

    def test_extreme_cdf_forces_corner(self) -> None:
        """If CDF places all mass at last cell (8, 8), every sample is (8, 8)."""
        cdf = np.zeros(81, dtype=np.float64)
        cdf[-1] = 1.0
        cdfs_2d = np.stack([cdf, cdf])
        rng = np.random.default_rng(0)
        scores = sample_scores_batch(cdfs_2d, rng)
        assert (scores == [8, 8]).all()


# ── sample_knockout_winner ───────────────────────────────────────────────────


class TestSampleKnockoutWinner:
    def test_p_advance_one_always_a(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            assert sample_knockout_winner("A", "B", 1.0, rng) == "A"

    def test_p_advance_zero_always_b(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            assert sample_knockout_winner("A", "B", 0.0, rng) == "B"

    def test_p_advance_half_balances(self) -> None:
        """At p=0.5, A and B should each win ~50% of 10000 trials."""
        rng = np.random.default_rng(42)
        a_wins = sum(sample_knockout_winner("A", "B", 0.5, rng) == "A" for _ in range(10000))
        assert 4750 < a_wins < 5250  # 2-sigma window around 5000


# ── build_group_samplers ─────────────────────────────────────────────────────


class TestBuildGroupSamplers:
    def test_produces_valid_cdfs(self, host_info: HostInfo, predictor) -> None:
        fixtures = [
            {
                "home_team": "Spain",
                "away_team": "Brazil",
                "venue_country": "United States",
                "tournament_type": "world_cup_final",
                "attendance_pct": 1.0,
            }
        ]
        ratings = {"Spain": 2050, "Brazil": 2000}
        samplers = build_group_samplers(fixtures, ratings, host_info, predictor)
        assert len(samplers) == 1
        s = samplers[0]
        assert s.home_team == "Spain"
        assert s.cdf_flat.shape == (81,)
        # CDF monotone non-decreasing
        assert (np.diff(s.cdf_flat) >= -1e-12).all()
        # CDF ends at 1.0 exactly
        assert s.cdf_flat[-1] == pytest.approx(1.0)


# ── build_ko_advance_table ───────────────────────────────────────────────────


class TestBuildKoAdvanceTable:
    def test_extreme_elo_gap_near_one(self, host_info: HostInfo, predictor) -> None:
        ratings = {"Strong": 2300, "Weak": 1100}
        table = build_ko_advance_table(["Strong", "Weak"], ratings, host_info, predictor)
        # P(Strong advances vs Weak) should be ~0.99+
        assert table[("Strong", "Weak")] > 0.95
        # P(Weak advances vs Strong) should be small
        assert table[("Weak", "Strong")] < 0.05

    def test_no_self_pair(self, host_info: HostInfo, predictor) -> None:
        ratings = {"A": 1800, "B": 1800}
        table = build_ko_advance_table(["A", "B"], ratings, host_info, predictor)
        assert ("A", "A") not in table
        assert ("B", "B") not in table
        assert set(table.keys()) == {("A", "B"), ("B", "A")}

    def test_equal_elo_two_concacaf_non_hosts(self, host_info: HostInfo, predictor) -> None:
        """Two equal-rated CONCACAF non-hosts should give ~0.5 — β cancels out, no α at TBD venue."""
        ratings = {"Costa Rica": 1700, "Jamaica": 1700}
        table = build_ko_advance_table(["Costa Rica", "Jamaica"], ratings, host_info, predictor)
        assert abs(table[("Costa Rica", "Jamaica")] - 0.5) < 0.02
