"""Tests for src/mc_simu/baseline.py — Phase 3 §3.8.

Verifies:
1. Baseline predictors return the documented W/D/L probabilities
2. Goal grids sum to 1 and partition correctly across home/draw/away triangles
3. Baseline implements the Predictor protocol (plugs into build_group_samplers)
4. WC2026 sim with baseline produces uniform-ish champion distribution
   (top team should have substantially LOWER probability than Elo case)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mc_simu.baseline import (
    _uniform_goal_grid,
    predict_match_40_40_20,
    predict_match_uniform,
)
from mc_simu.simulator import HostInfo, build_group_samplers, build_ko_advance_table
from mc_simu.single_game import MatchContext


@pytest.fixture
def trivial_ctx() -> MatchContext:
    return MatchContext(
        is_neutral=True,
        tournament_type="world_cup_final",
        home_country="A",
        away_country="B",
        venue_country="",
        venue_confederation="UNKNOWN",
        home_confederation="UNKNOWN",
        away_confederation="UNKNOWN",
        host_countries=[],
        host_confederation=None,
    )


@pytest.fixture
def host_info() -> HostInfo:
    return HostInfo(host_countries=[], host_confederation=None)


# ── 40/40/20 marginals ───────────────────────────────────────────────────────


class TestPredictMatch404020:
    def test_probabilities_match_documented_split(self, trivial_ctx: MatchContext) -> None:
        pred = predict_match_40_40_20(1500, 1500, trivial_ctx)
        assert pred.p_home == 0.40
        assert pred.p_draw == 0.20
        assert pred.p_away == 0.40

    def test_ignores_elo_inputs(self, trivial_ctx: MatchContext) -> None:
        """Even at +1000 Elo gap, baseline returns 40/20/40 — that's the point."""
        p1 = predict_match_40_40_20(2500, 1500, trivial_ctx)
        p2 = predict_match_40_40_20(1500, 2500, trivial_ctx)
        assert (p1.p_home, p1.p_draw, p1.p_away) == (0.40, 0.20, 0.40)
        assert (p2.p_home, p2.p_draw, p2.p_away) == (0.40, 0.20, 0.40)

    def test_goal_grid_sums_to_one(self, trivial_ctx: MatchContext) -> None:
        pred = predict_match_40_40_20(1500, 1500, trivial_ctx)
        assert pred.goal_grid.sum() == pytest.approx(1.0, abs=1e-12)

    def test_goal_grid_triangle_masses(self, trivial_ctx: MatchContext) -> None:
        pred = predict_match_40_40_20(1500, 1500, trivial_ctx)
        grid = pred.goal_grid
        home_mass = float(np.sum(np.tril(grid, k=-1)))
        draw_mass = float(np.sum(np.diag(grid)))
        away_mass = float(np.sum(np.triu(grid, k=1)))
        assert home_mass == pytest.approx(0.40, abs=1e-12)
        assert draw_mass == pytest.approx(0.20, abs=1e-12)
        assert away_mass == pytest.approx(0.40, abs=1e-12)


# ── Uniform 1/3 1/3 1/3 ──────────────────────────────────────────────────────


class TestPredictMatchUniform:
    def test_thirds(self, trivial_ctx: MatchContext) -> None:
        pred = predict_match_uniform(1500, 1500, trivial_ctx)
        assert pred.p_home == pytest.approx(1 / 3, abs=1e-12)
        assert pred.p_draw == pytest.approx(1 / 3, abs=1e-12)
        assert pred.p_away == pytest.approx(1 / 3, abs=1e-12)
        assert pred.goal_grid.sum() == pytest.approx(1.0, abs=1e-12)


# ── Uniform grid helper ──────────────────────────────────────────────────────


class TestUniformGoalGrid:
    def test_arbitrary_marginals_preserved(self) -> None:
        grid = _uniform_goal_grid(0.5, 0.3, 0.2)
        assert grid.sum() == pytest.approx(1.0, abs=1e-12)
        assert float(np.sum(np.tril(grid, k=-1))) == pytest.approx(0.5, abs=1e-12)
        assert float(np.sum(np.diag(grid))) == pytest.approx(0.3, abs=1e-12)
        assert float(np.sum(np.triu(grid, k=1))) == pytest.approx(0.2, abs=1e-12)


# ── Predictor protocol — plugs into samplers ─────────────────────────────────


class TestBaselinePluggability:
    def test_build_group_samplers_accepts_baseline(self, host_info: HostInfo) -> None:
        fixtures = [
            {
                "home_team": "A", "away_team": "B",
                "venue_country": "", "tournament_type": "world_cup_final",
                "attendance_pct": 1.0,
            },
        ]
        ratings = {"A": 1500, "B": 1500}
        samplers = build_group_samplers(fixtures, ratings, host_info, predict_match_40_40_20)
        assert len(samplers) == 1
        # CDF should encode 40/20/40 marginals
        cdf = samplers[0].cdf_flat
        assert cdf[-1] == pytest.approx(1.0)

    def test_build_ko_advance_table_accepts_baseline(self, host_info: HostInfo) -> None:
        ratings = {"A": 1500, "B": 1500}
        table = build_ko_advance_table(["A", "B"], ratings, host_info, predict_match_40_40_20)
        # p_advance = p_home + 0.5 * p_draw = 0.4 + 0.1 = 0.5 (symmetric since equal)
        assert table[("A", "B")] == pytest.approx(0.5, abs=1e-12)
        assert table[("B", "A")] == pytest.approx(0.5, abs=1e-12)


# ── End-to-end: baseline on WC2026 sim should be near-uniform champions ─────


class TestBaselineOnWC2026:
    def test_baseline_champions_more_uniform_than_elo(
        self, mc_simu_data_dir: Path
    ) -> None:
        """When all matches are 40/20/40, no team has Elo advantage → champion
        distribution should be much closer to 1/48 = 2.08% per team, with the
        top team well below the Elo case's Spain 19.85%.
        """
        from mc_simu.tournaments.wc2026 import load_wc2026_bundle, run_monte_carlo

        elo_df = pd.read_csv(mc_simu_data_dir / "elo_history.csv")
        elo_df["date"] = pd.to_datetime(elo_df["date"])
        elo_df = elo_df.sort_values("date")
        ratings = elo_df.groupby("team").tail(1).set_index("team")["rating_after"].to_dict()
        bundle = load_wc2026_bundle(ratings)

        result = run_monte_carlo(
            bundle,
            n_iterations=500,
            seed=42,
            progress=False,
            predictor=predict_match_40_40_20,
        )
        # Sum still = 1
        total = sum(s["mc_fair_prob"] for s in result["champion"].values())
        assert abs(total - 1.0) < 1e-9
        # Top team should be much lower than Spain 19.85% in Elo case
        top_prob = max(s["mc_fair_prob"] for s in result["champion"].values())
        assert top_prob < 0.10, f"baseline top {top_prob:.3f} should be <10% (Elo had ~20%)"
        # Distribution should NOT degenerate — at least 30 teams should have non-zero prob
        nonzero = sum(1 for s in result["champion"].values() if s["mc_fair_prob"] > 0)
        assert nonzero >= 30, f"only {nonzero} teams have non-zero champion prob"
