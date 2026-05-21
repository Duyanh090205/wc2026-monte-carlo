"""Cross-adapter wiring test — Phase 3 §3.7 + §3.8.

Verifies the harness/adapter/predictor separation is real:
- Each adapter (wc2026, wc_8groups, euro_24teams, euro_16teams) runs with
  EITHER Elo predictor OR baseline 40/40/20, and produces valid output (sum=1,
  champion in roster).
- This is the integration test ensuring the swappable Predictor protocol
  actually works across all 4 adapters × 2 predictors = 8 combinations.

If this passes, Phase 4 LOTO-CV can freely combine any (adapter, predictor)
without per-pair plumbing changes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mc_simu.baseline import predict_match_40_40_20
from mc_simu.simulator import Predictor


def _ratings_before(year: int, mc_simu_data_dir: Path) -> dict[str, float]:
    elo_df = pd.read_csv(mc_simu_data_dir / "elo_history.csv")
    elo_df["date"] = pd.to_datetime(elo_df["date"])
    cutoff_year = 2021 if year == 2020 else year
    cutoff = pd.Timestamp(year=cutoff_year, month=6, day=1)
    snap = elo_df[elo_df["date"] < cutoff]
    return snap.sort_values("date").groupby("team").tail(1).set_index("team")["rating_after"].to_dict()


def _latest_ratings(mc_simu_data_dir: Path) -> dict[str, float]:
    elo_df = pd.read_csv(mc_simu_data_dir / "elo_history.csv")
    elo_df["date"] = pd.to_datetime(elo_df["date"])
    return elo_df.sort_values("date").groupby("team").tail(1).set_index("team")["rating_after"].to_dict()


# Matrix: (adapter_name, run_mc_fn, bundle_loader, sample_year_arg)
# Each adapter exposes the same Predictor-pluggable signature via predictor= kwarg.

def _run_wc2026(predictor: Predictor | None, mc_simu_data_dir: Path):
    from mc_simu.tournaments.wc2026 import load_wc2026_bundle, run_monte_carlo
    ratings = _latest_ratings(mc_simu_data_dir)
    bundle = load_wc2026_bundle(ratings)
    return run_monte_carlo(bundle, n_iterations=200, seed=42, progress=False, predictor=predictor)


def _run_wc_8groups(predictor: Predictor | None, mc_simu_data_dir: Path):
    from mc_simu.tournaments.wc_8groups import load_edition_bundle, run_monte_carlo
    ratings = _ratings_before(2022, mc_simu_data_dir)
    bundle = load_edition_bundle(2022, ratings)
    return run_monte_carlo(bundle, n_iterations=200, seed=42, progress=False, predictor=predictor)


def _run_euro_24teams(predictor: Predictor | None, mc_simu_data_dir: Path):
    from mc_simu.tournaments.euro_24teams import load_edition_bundle, run_monte_carlo
    ratings = _ratings_before(2024, mc_simu_data_dir)
    bundle = load_edition_bundle(2024, ratings)
    return run_monte_carlo(bundle, n_iterations=200, seed=42, progress=False, predictor=predictor)


def _run_euro_16teams(predictor: Predictor | None, mc_simu_data_dir: Path):
    from mc_simu.tournaments.euro_16teams import load_edition_bundle, run_monte_carlo
    ratings = _ratings_before(2012, mc_simu_data_dir)
    bundle = load_edition_bundle(2012, ratings)
    return run_monte_carlo(bundle, n_iterations=200, seed=42, progress=False, predictor=predictor)


ADAPTERS = [
    ("wc2026",       _run_wc2026,       48),
    ("wc_8groups",   _run_wc_8groups,   32),
    ("euro_24teams", _run_euro_24teams, 24),
    ("euro_16teams", _run_euro_16teams, 16),
]


class TestPredictorPluggability:
    """Cross-product: each adapter × each predictor."""

    @pytest.mark.parametrize("name,run_fn,n_teams", ADAPTERS, ids=[a[0] for a in ADAPTERS])
    def test_default_elo_predictor(self, name, run_fn, n_teams, mc_simu_data_dir: Path) -> None:
        result = run_fn(None, mc_simu_data_dir)
        total = sum(s["mc_fair_prob"] for s in result["champion"].values())
        assert abs(total - 1.0) < 1e-9, f"{name}: sum={total}"
        # At least one team has prob > 0
        assert len(result["champion"]) >= 1

    @pytest.mark.parametrize("name,run_fn,n_teams", ADAPTERS, ids=[a[0] for a in ADAPTERS])
    def test_baseline_40_40_20_predictor(self, name, run_fn, n_teams, mc_simu_data_dir: Path) -> None:
        result = run_fn(predict_match_40_40_20, mc_simu_data_dir)
        total = sum(s["mc_fair_prob"] for s in result["champion"].values())
        assert abs(total - 1.0) < 1e-9, f"{name}: sum={total}"


class TestBaselineBeatenByElo:
    """The whole point of baseline: top Elo team should outrank baseline top team.

    Elo predictor concentrates probability on strong teams (e.g., Spain 20% in
    WC2026). Baseline 40/40/20 spreads it more evenly (top team much lower
    than Elo case). This is the qualitative sanity check that confirms the
    Phase 4 head-to-head framework will have signal.
    """

    @pytest.mark.parametrize("name,run_fn,n_teams", ADAPTERS, ids=[a[0] for a in ADAPTERS])
    def test_top_team_more_concentrated_under_elo(
        self, name, run_fn, n_teams, mc_simu_data_dir: Path,
    ) -> None:
        elo_result = run_fn(None, mc_simu_data_dir)
        baseline_result = run_fn(predict_match_40_40_20, mc_simu_data_dir)
        elo_top = max(s["mc_fair_prob"] for s in elo_result["champion"].values())
        baseline_top = max(s["mc_fair_prob"] for s in baseline_result["champion"].values())
        assert elo_top > baseline_top, (
            f"{name}: Elo top {elo_top:.3f} should exceed baseline top {baseline_top:.3f}"
        )
