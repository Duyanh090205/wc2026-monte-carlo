"""Phase 2 fitting-engine gates (T2.x) for the mle_strength rating source."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from mc_simu.strength_mle import (
    _nll_and_grad,
    fit_strengths,
    load_matches,
    match_weights,
)

AS_OF = pd.Timestamp("2026-06-11")


def simulate_matches(rng: np.random.Generator, r: np.ndarray, c: float, h: float,
                     n_matches: int) -> pd.DataFrame:
    n_teams = len(r)
    idx_h = rng.integers(0, n_teams, n_matches)
    idx_a = rng.integers(0, n_teams, n_matches)
    redraw = idx_h == idx_a
    idx_a[redraw] = (idx_h[redraw] + 1) % n_teams
    neutral = rng.random(n_matches) < 0.3
    a = (~neutral).astype(float)
    lam_h = np.exp(c + r[idx_h] + h * a - r[idx_a])
    lam_a = np.exp(c + r[idx_a] - r[idx_h] - h * a)
    dates = AS_OF - pd.to_timedelta(rng.integers(1, 1500, n_matches), unit="D")
    return pd.DataFrame({
        "date": dates,
        "home_team": [f"T{k:03d}" for k in idx_h],
        "away_team": [f"T{k:03d}" for k in idx_a],
        "home_score": rng.poisson(lam_h),
        "away_score": rng.poisson(lam_a),
        "neutral_bool": neutral,
        "importance": 1.0,
    })


@pytest.fixture(scope="module")
def truth() -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 0.4, 50)
    r -= r.mean()
    return r, 0.1, 0.25


@pytest.fixture(scope="module")
def real_matches() -> pd.DataFrame:
    return load_matches(as_of=AS_OF)


@pytest.fixture(scope="module")
def real_fit(real_matches: pd.DataFrame):
    return fit_strengths(real_matches, AS_OF)


class TestT21SyntheticRecovery:
    def test_recovery_20k(self, truth) -> None:
        r_true, c_true, h_true = truth
        rng = np.random.default_rng(11)
        df = simulate_matches(rng, r_true, c_true, h_true, 20_000)
        fit = fit_strengths(df, AS_OF)
        r_hat = np.array([fit.strengths[f"T{k:03d}"] for k in range(len(r_true))])
        assert fit.converged
        assert np.corrcoef(r_hat, r_true)[0, 1] > 0.995
        assert abs(fit.c - c_true) < 0.02
        assert abs(fit.h - h_true) < 0.02

    def test_degradation_3k(self, truth) -> None:
        r_true, c_true, h_true = truth
        rng = np.random.default_rng(13)
        df = simulate_matches(rng, r_true, c_true, h_true, 3_000)
        fit = fit_strengths(df, AS_OF)
        r_hat = np.array([fit.strengths[f"T{k:03d}"] for k in range(len(r_true))])
        assert fit.converged
        assert np.corrcoef(r_hat, r_true)[0, 1] > 0.97
        assert abs(fit.c - c_true) < 0.06
        assert abs(fit.h - h_true) < 0.06


class TestT22Gradient:
    def test_analytic_matches_numeric(self, truth) -> None:
        r_true, c_true, h_true = truth
        rng = np.random.default_rng(17)
        df = simulate_matches(rng, r_true, c_true, h_true, 400)
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        index = {t: k for k, t in enumerate(teams)}
        args = (
            df["home_team"].map(index).to_numpy(np.int64),
            df["away_team"].map(index).to_numpy(np.int64),
            df["home_score"].to_numpy(np.float64),
            df["away_score"].to_numpy(np.float64),
            (~df["neutral_bool"]).to_numpy(np.float64),
            match_weights(df, AS_OF),
            len(teams),
        )
        for seed in range(5):
            x = np.random.default_rng(seed).normal(0, 0.3, len(teams) + 1)
            err = check_grad(lambda v: _nll_and_grad(v, *args)[0],
                             lambda v: _nll_and_grad(v, *args)[1], x)
            grad_norm = np.linalg.norm(_nll_and_grad(x, *args)[1])
            assert err / grad_norm < 1e-5


class TestT23Identification:
    def test_shift_invariance_of_likelihood(self, truth) -> None:
        r_true, c_true, h_true = truth
        rng = np.random.default_rng(19)
        df = simulate_matches(rng, r_true, c_true, h_true, 500)

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        index = {t: k for k, t in enumerate(teams)}
        hidx = df["home_team"].map(index).to_numpy(np.int64)
        aidx = df["away_team"].map(index).to_numpy(np.int64)
        gh = df["home_score"].to_numpy(np.float64)
        ga = df["away_score"].to_numpy(np.float64)
        a = (~df["neutral_bool"]).to_numpy(np.float64)
        r_full = r_true[[int(t[1:]) for t in teams]]

        def loglik_at(r_shift: np.ndarray) -> float:
            lam_h = np.exp(c_true + r_shift[hidx] + h_true * a - r_shift[aidx])
            lam_a = np.exp(c_true + r_shift[aidx] - r_shift[hidx] - h_true * a)
            return float((gh * np.log(lam_h) - lam_h
                          + ga * np.log(lam_a) - lam_a).sum())
        assert loglik_at(r_full) == pytest.approx(loglik_at(r_full + 3.7), rel=1e-12)

    def test_post_fit_sum_zero(self, truth) -> None:
        r_true, c_true, h_true = truth
        rng = np.random.default_rng(23)
        df = simulate_matches(rng, r_true, c_true, h_true, 2_000)
        fit = fit_strengths(df, AS_OF)
        assert abs(sum(fit.strengths.values())) < 1e-10


class TestT24Convergence:
    def test_multistart_same_optimum(self, truth) -> None:
        r_true, c_true, h_true = truth
        rng = np.random.default_rng(29)
        df = simulate_matches(rng, r_true, c_true, h_true, 2_000)
        base = fit_strengths(df, AS_OF)
        assert base.converged
        from mc_simu.strength_mle import FitResult
        for seed in range(3):
            jitter = np.random.default_rng(seed)
            warm = FitResult(
                strengths={t: float(jitter.normal(0, 0.5))
                           for t in base.strengths},
                c=float(jitter.normal(0, 0.3)), h=float(jitter.normal(0, 0.3)),
                loglik=0.0, n_matches=0, n_teams=0, as_of="",
                half_period_days=1095.0, converged=True, optimizer_message="")
            refit = fit_strengths(df, AS_OF, warm_start=warm)
            assert refit.converged
            assert refit.loglik == pytest.approx(base.loglik, abs=1e-6 * abs(base.loglik))


class TestT25RealDataSanity:
    def test_home_effect_positive(self, real_fit) -> None:
        assert real_fit.h > 0

    def test_score_equation_total_goals_identity(self, real_fit, real_matches) -> None:
        # At the optimum dL/dc = 0 forces weighted predicted total goals to
        # equal weighted actual total goals exactly (Poisson score equation).
        # 2*exp(c) alone would undershoot by the cosh(strength gap) factor.
        w = match_weights(real_matches, AS_OF)
        r = real_matches["home_team"].map(real_fit.strengths).to_numpy()
        s = real_matches["away_team"].map(real_fit.strengths).to_numpy()
        a = (~real_matches["neutral_bool"]).to_numpy(np.float64)
        lam_h = np.exp(real_fit.c + r + real_fit.h * a - s)
        lam_a = np.exp(real_fit.c + s - r - real_fit.h * a)
        predicted = float((w * (lam_h + lam_a)).sum())
        actual = float((w * (real_matches["home_score"]
                             + real_matches["away_score"])).sum())
        assert predicted == pytest.approx(actual, rel=1e-3)

    def test_top10_recognizable(self, real_fit) -> None:
        top10 = sorted(real_fit.strengths, key=real_fit.strengths.get,
                       reverse=True)[:10]
        plausible = {
            "Spain", "Argentina", "France", "Brazil", "England", "Portugal",
            "Netherlands", "Germany", "Italy", "Belgium", "Croatia", "Morocco",
            "Colombia", "Uruguay", "Japan", "South Korea", "Mexico",
            "United States", "Switzerland", "Denmark", "Norway", "Austria",
        }
        assert len(set(top10) & plausible) >= 7


class TestT27Determinism:
    def test_bitwise_identical_refit(self, truth) -> None:
        r_true, c_true, h_true = truth
        rng = np.random.default_rng(31)
        df = simulate_matches(rng, r_true, c_true, h_true, 2_000)
        f1 = fit_strengths(df, AS_OF)
        f2 = fit_strengths(df, AS_OF)
        assert f1.strengths == f2.strengths
        assert f1.c == f2.c and f1.h == f2.h


class TestT28Runtime:
    def test_full_real_fit_under_60s(self, real_matches) -> None:
        import time
        start = time.perf_counter()
        fit = fit_strengths(real_matches, AS_OF)
        elapsed = time.perf_counter() - start
        assert fit.converged
        assert elapsed < 60.0
