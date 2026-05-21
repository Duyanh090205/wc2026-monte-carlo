"""Tests for src/mc_simu/single_game.py — §2.1 Steps 1.4–1.6."""

from __future__ import annotations

import numpy as np
import pytest

from mc_simu.single_game import (
    MatchContext,
    ModelParams,
    elo_to_lambda,
    goal_distribution,
    hfa_log_goals,
    predict_match,
)


# ── ModelParams defaults ──────────────────────────────────────────────────────


def test_model_params_default_blend_is_one_per_phase_2_skip() -> None:
    """§0.8 override: blend_match_weight must default to 1.0 (match-Elo only)."""
    p = ModelParams()
    assert p.blend_match_weight == 1.0


def test_model_params_defaults_match_audit_calibration() -> None:
    """Phase 1 deep audit (post-ChatGPT cross-verify, finding #1 corrected):
    α=0.27 (avg of 538 0.26 + Bilalić 0.28 log-goals, both unit-converted from additive)
    β=0.09 (538 SPI additive 0.13 goals → 0.09 log-goals at baseline λ=1.35)
    diagonal_inflation=0.20 (Phase 1 empirical optimum for int'l tournaments)
    """
    p = ModelParams()
    assert p.alpha == 0.27
    assert p.beta == 0.09
    assert p.diagonal_inflation == 0.20


# ── elo_to_lambda ─────────────────────────────────────────────────────────────


class TestEloToLambda:
    def test_equal_elo_no_hfa_sums_to_avg(self) -> None:
        """λ_team + λ_opp ≈ league_avg_goals when ratings equal and HFA zero."""
        lh = elo_to_lambda(1500, 1500, hfa_log_goals=0.0)
        la = elo_to_lambda(1500, 1500, hfa_log_goals=0.0)
        assert lh + la == pytest.approx(2.7, abs=1e-9)

    def test_hfa_multiplicative_exp(self) -> None:
        """λ · exp(α) for α=0.40 → ~1.49× baseline."""
        base = elo_to_lambda(1500, 1500, hfa_log_goals=0.0)
        boosted = elo_to_lambda(1500, 1500, hfa_log_goals=0.40)
        assert boosted / base == pytest.approx(np.exp(0.40), abs=1e-9)

    def test_positive_elo_diff_increases_lambda(self) -> None:
        weak = elo_to_lambda(1500, 1700)   # team weaker
        strong = elo_to_lambda(1700, 1500) # team stronger
        assert strong > 1.35 > weak

    def test_d_denominator_calibration(self) -> None:
        """REGRESSION: ELO_GOALS_DENOMINATOR was 400 (broken — caused 10× goal
        ratio at +400 Elo), then 800 (over-calibrated to binary win prob), now
        1400 (Brier-minimized on validation set). At +400 Elo, λ_strong / λ_weak
        should be ~3.7 (= 10^(800/1400)), giving a realistic strong-vs-weak total
        goals around 3.0 instead of the 4.7 we had at D=800.
        """
        l_strong = elo_to_lambda(1900, 1500)
        l_weak   = elo_to_lambda(1500, 1900)
        ratio = l_strong / l_weak
        assert 3.5 < ratio < 4.0, f"At ΔElo=400, ratio expected ~3.73, got {ratio:.2f}"


# ── goal_distribution ─────────────────────────────────────────────────────────


class TestGoalDistribution:
    def test_grid_sums_to_one(self) -> None:
        g = goal_distribution(1.5, 1.2, diagonal_inflation=0.09)
        assert g.sum() == pytest.approx(1.0, abs=1e-12)

    def test_inflation_zero_is_normalized_truncated_poisson(self) -> None:
        """With diagonal_inflation=0, grid is pure Poisson product (then normalized)."""
        from scipy.stats import poisson
        g = goal_distribution(1.5, 1.2, diagonal_inflation=0.0, max_goals=8)
        i = np.arange(9)
        pmf_h = poisson.pmf(i, 1.5)
        pmf_a = poisson.pmf(i, 1.2)
        expected = np.outer(pmf_h, pmf_a)
        expected /= expected.sum()
        assert np.allclose(g, expected, atol=1e-12)

    def test_symmetry_under_lambda_swap(self) -> None:
        """goal_distribution(λa, λb) == goal_distribution(λb, λa).T (lambda swap = transpose)."""
        g_ab = goal_distribution(1.5, 1.2, diagonal_inflation=0.09)
        g_ba = goal_distribution(1.2, 1.5, diagonal_inflation=0.09)
        assert np.allclose(g_ab, g_ba.T, atol=1e-12)

    def test_inflation_increases_diagonal_mass(self) -> None:
        g0 = goal_distribution(1.5, 1.5, diagonal_inflation=0.0)
        g_inflated = goal_distribution(1.5, 1.5, diagonal_inflation=0.20)
        assert g_inflated.diagonal().sum() > g0.diagonal().sum()


# ── hfa_log_goals ─────────────────────────────────────────────────────────────


def _ctx_for(venue: str, host_countries: list[str], host_confed: str | None,
             home: str, away: str, home_confed: str, away_confed: str,
             venue_confed: str, attendance: float = 1.0, neutral: bool = False) -> MatchContext:
    return MatchContext(
        is_neutral=neutral, tournament_type="continental_final",
        home_country=home, away_country=away,
        venue_country=venue, venue_confederation=venue_confed,
        home_confederation=home_confed, away_confederation=away_confed,
        host_countries=host_countries, host_confederation=host_confed,
        attendance_pct=attendance,
    )


class TestHfaLogGoals:
    def test_own_country_gets_alpha_only(self) -> None:
        """Germany hosting Euro 2024 at home stadium → α, no β (is_host)."""
        ctx = _ctx_for(venue="Germany", host_countries=["Germany"], host_confed="UEFA",
                       home="Germany", away="Scotland",
                       home_confed="UEFA", away_confed="UEFA", venue_confed="UEFA")
        params = ModelParams()
        bonus = hfa_log_goals("Germany", "UEFA", ctx, params)
        assert bonus == pytest.approx(params.alpha, abs=1e-9)

    def test_same_confed_non_host_gets_beta_only(self) -> None:
        """Scotland at Euro 2024 (Germany host, UEFA confed) → β only."""
        ctx = _ctx_for(venue="Germany", host_countries=["Germany"], host_confed="UEFA",
                       home="Germany", away="Scotland",
                       home_confed="UEFA", away_confed="UEFA", venue_confed="UEFA")
        params = ModelParams()
        bonus = hfa_log_goals("Scotland", "UEFA", ctx, params)
        assert bonus == pytest.approx(params.beta, abs=1e-9)

    def test_different_confed_gets_zero(self) -> None:
        """Argentina at Euro 2024 → 0 (CONMEBOL ≠ UEFA, not host)."""
        ctx = _ctx_for(venue="Germany", host_countries=["Germany"], host_confed="UEFA",
                       home="Germany", away="Argentina",
                       home_confed="UEFA", away_confed="CONMEBOL", venue_confed="UEFA")
        bonus = hfa_log_goals("Argentina", "CONMEBOL", ctx, ModelParams())
        assert bonus == pytest.approx(0.0, abs=1e-12)

    def test_covid_scaling_under_25pct(self) -> None:
        """Post-audit calibration: scaler 0.55 (was 0.67) per Bilalić 2021 ~41% HA reduction empty."""
        ctx = _ctx_for(venue="Germany", host_countries=["Germany"], host_confed="UEFA",
                       home="Germany", away="Scotland",
                       home_confed="UEFA", away_confed="UEFA", venue_confed="UEFA",
                       attendance=0.15)
        bonus = hfa_log_goals("Germany", "UEFA", ctx, ModelParams())
        assert bonus == pytest.approx(0.27 * 0.55, abs=1e-9)

    def test_covid_scaling_25_to_50(self) -> None:
        """Post-audit calibration: scaler 0.65 (was 0.80) per Euro 2020 ~50% reduction at 33% cap."""
        ctx = _ctx_for(venue="Germany", host_countries=["Germany"], host_confed="UEFA",
                       home="Germany", away="Scotland",
                       home_confed="UEFA", away_confed="UEFA", venue_confed="UEFA",
                       attendance=0.40)
        bonus = hfa_log_goals("Germany", "UEFA", ctx, ModelParams())
        assert bonus == pytest.approx(0.27 * 0.65, abs=1e-9)

    def test_covid_scaling_50_to_75_new_bin(self) -> None:
        """NEW bin (post-audit): 50-75% capacity gets 0.85 scaling (was no scaling)."""
        ctx = _ctx_for(venue="Germany", host_countries=["Germany"], host_confed="UEFA",
                       home="Germany", away="Scotland",
                       home_confed="UEFA", away_confed="UEFA", venue_confed="UEFA",
                       attendance=0.60)
        bonus = hfa_log_goals("Germany", "UEFA", ctx, ModelParams())
        assert bonus == pytest.approx(0.27 * 0.85, abs=1e-9)

    def test_unknown_confederation_safe_fallback(self) -> None:
        """A team with UNKNOWN confed (e.g., CONIFA Catalonia) never gets β."""
        ctx = _ctx_for(venue="Germany", host_countries=["Germany"], host_confed="UEFA",
                       home="Germany", away="Catalonia",
                       home_confed="UEFA", away_confed="UNKNOWN", venue_confed="UEFA")
        bonus = hfa_log_goals("Catalonia", "UNKNOWN", ctx, ModelParams())
        assert bonus == 0.0

    def test_host_confed_none_qualifier_or_friendly(self) -> None:
        """For non-tournament matches (host_confederation=None), only own-country α applies."""
        ctx = _ctx_for(venue="France", host_countries=[], host_confed=None,
                       home="France", away="Spain",
                       home_confed="UEFA", away_confed="UEFA", venue_confed="UEFA")
        # France at home → α only; Spain (UEFA but no host) → 0
        assert hfa_log_goals("France", "UEFA", ctx, ModelParams()) == pytest.approx(0.27, abs=1e-9)
        assert hfa_log_goals("Spain",  "UEFA", ctx, ModelParams()) == 0.0


# ── predict_match invariants ──────────────────────────────────────────────────


class TestPredictMatch:
    def test_probabilities_sum_to_one(self) -> None:
        ctx = _ctx_for(venue="Neutral", host_countries=[], host_confed=None,
                       home="A", away="B",
                       home_confed="UEFA", away_confed="UEFA", venue_confed="UEFA",
                       neutral=True)
        pred = predict_match(1700, 1500, ctx, ModelParams())
        assert pred.p_home + pred.p_draw + pred.p_away == pytest.approx(1.0, abs=1e-9)

    def test_symmetric_inputs_neutral_yield_symmetric_outputs(self) -> None:
        """Equal Elo + truly neutral (no HFA either side) → P_home ≈ P_away."""
        ctx = _ctx_for(venue="Neutral", host_countries=[], host_confed=None,
                       home="A", away="B",
                       home_confed="UEFA", away_confed="UEFA", venue_confed="UEFA",
                       neutral=True)
        pred = predict_match(1500, 1500, ctx, ModelParams())
        assert pred.p_home == pytest.approx(pred.p_away, abs=1e-9)

    def test_higher_elo_team_more_likely_to_win(self) -> None:
        ctx = _ctx_for(venue="Neutral", host_countries=[], host_confed=None,
                       home="Strong", away="Weak",
                       home_confed="UEFA", away_confed="UEFA", venue_confed="UEFA",
                       neutral=True)
        pred = predict_match(1900, 1500, ctx, ModelParams())
        assert pred.p_home > pred.p_away
        assert pred.p_home > 0.6   # +400 Elo should be very dominant

    def test_grid_attached_to_prediction(self) -> None:
        ctx = _ctx_for(venue="X", host_countries=[], host_confed=None,
                       home="A", away="B",
                       home_confed="UEFA", away_confed="UEFA", venue_confed="UEFA",
                       neutral=True)
        pred = predict_match(1500, 1500, ctx, ModelParams())
        assert pred.goal_grid.shape == (9, 9)
        assert pred.goal_grid.sum() == pytest.approx(1.0, abs=1e-9)

    def test_extreme_elo_gap_does_not_return_zero_distribution(self) -> None:
        """REGRESSION (audit bug #2): at extreme ΔElo, Poisson PMFs underflow → grid
        sums to literal 0. Without the underflow guard, predict_match returned
        (0,0,0). Now: degenerate (1,0,0) when home stronger, (0,0,1) when away.

        Trigger threshold scales with ELO_GOALS_DENOMINATOR. At D=1400, need
        ΔElo > ~5000 to push λ above ~200 where truncated Poisson(0..8) underflows.
        """
        ctx = _ctx_for(venue="X", host_countries=[], host_confed=None,
                       home="Strong", away="Weak",
                       home_confed="UEFA", away_confed="UEFA", venue_confed="UEFA",
                       neutral=True)
        # ΔElo=6000 — guarantees underflow regardless of D in {800, 1400, 2000}
        pred = predict_match(7500, 1500, ctx, ModelParams())
        assert pred.p_home + pred.p_draw + pred.p_away == pytest.approx(1.0, abs=1e-9)
        assert pred.p_home == pytest.approx(1.0, abs=1e-9)
        assert pred.p_away == pytest.approx(0.0, abs=1e-9)

        # Mirror case — extreme away advantage
        pred_away = predict_match(1500, 7500, ctx, ModelParams())
        assert pred_away.p_home + pred_away.p_draw + pred_away.p_away == pytest.approx(1.0, abs=1e-9)
        assert pred_away.p_away == pytest.approx(1.0, abs=1e-9)
