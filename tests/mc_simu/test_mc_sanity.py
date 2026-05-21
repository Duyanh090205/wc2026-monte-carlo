"""Tests for src/mc_simu/eloratings_snapshot.py — Phase 1 sanity-check helper."""

from __future__ import annotations

import pandas as pd
import pytest

from mc_simu.eloratings_snapshot import (
    CODE_TO_TEAM,
    load_fallback_fixture,
    sanity_check_history,
)


def _mk_history(latest_ratings: dict[str, float]) -> dict[str, pd.DataFrame]:
    """Build a minimal history dict where each team has a single row whose
    rating_after equals the target value. Sufficient for sanity_check_history."""
    return {
        team: pd.DataFrame([{
            "date": pd.Timestamp("2026-05-17"), "rating_after": rating,
            "opponent": "X", "tournament_type": "friendly",
            "K": 20, "mov": 1.0, "gs_self": 0, "gs_opp": 0,
        }])
        for team, rating in latest_ratings.items()
    }


# ── Code → team mapping coverage ──────────────────────────────────────────────


def test_code_to_team_contains_top20_codes() -> None:
    """Top-20 codes seen 2026-05-17 must all map."""
    top20_codes = {"ES","AR","FR","EN","BR","PT","CO","NL","EC","HR",
                   "DE","NO","JP","TR","UY","CH","SN","DK","BE","MX"}
    missing = top20_codes - set(CODE_TO_TEAM.keys())
    assert not missing, f"Top-20 codes missing from CODE_TO_TEAM: {missing}"


# ── Layer A: absolute bounds [1000, 2500] ─────────────────────────────────────


class TestLayerAAbsoluteBounds:
    def test_in_range_passes(self) -> None:
        # Weak teams legitimately below 1000 are OK (floor is 500)
        hist = _mk_history({"Brazil": 2000, "England": 1900, "Brunei": 900})
        r = sanity_check_history(hist, top20_reference=None)
        assert r["layer_a"] == "PASS"

    def test_below_floor_hard_stops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Below 500 indicates numerical runaway, not legitimate weakness."""
        monkeypatch.delenv("MC_SIMU_OVERRIDE_HARD_STOP", raising=False)
        hist = _mk_history({"Brazil": 450})
        with pytest.raises(SystemExit) as exc:
            sanity_check_history(hist, top20_reference=None)
        assert exc.value.code == 2

    def test_layer_a_error_message_mentions_current_bounds(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """REGRESSION (audit bug #1): Layer A HARD STOP message must report the
        actual bounds [500, 2500], not the stale plan-original [1000, 2500]."""
        monkeypatch.delenv("MC_SIMU_OVERRIDE_HARD_STOP", raising=False)
        hist = _mk_history({"Brazil": 450})
        with pytest.raises(SystemExit):
            sanity_check_history(hist, top20_reference=None)
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "[500, 2500]" in out
        assert "[1000, 2500]" not in out

    def test_above_ceiling_hard_stops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MC_SIMU_OVERRIDE_HARD_STOP", raising=False)
        hist = _mk_history({"Brazil": 2600})
        with pytest.raises(SystemExit) as exc:
            sanity_check_history(hist, top20_reference=None)
        assert exc.value.code == 2


# ── Layer B: top-20 reference comparison ──────────────────────────────────────


class TestLayerBReferenceComparison:
    def test_no_reference_returns_skip(self) -> None:
        hist = _mk_history({"Brazil": 2000})
        r = sanity_check_history(hist, top20_reference=None)
        assert r["layer_b"].startswith("SKIP")

    def test_within_tolerance_passes(self) -> None:
        hist = _mk_history({"Brazil": 2000, "Spain": 2150, "England": 2010})
        ref = {"Brazil": 1980, "Spain": 2165, "England": 2020}   # all within 100
        r = sanity_check_history(hist, top20_reference=ref, tolerance=100, max_failures=5)
        assert r["layer_a"] == "PASS"
        assert "0 of" in r["layer_b"]

    def test_excess_divergence_hard_stops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """6 teams diverging by >100 with max_failures=5 → HARD STOP."""
        monkeypatch.delenv("MC_SIMU_OVERRIDE_HARD_STOP", raising=False)
        hist = _mk_history({
            "Brazil": 1500, "Spain": 1500, "France": 1500,
            "England": 1500, "Argentina": 1500, "Portugal": 1500,
        })
        ref = {
            "Brazil": 2000, "Spain": 2000, "France": 2000,
            "England": 2000, "Argentina": 2000, "Portugal": 2000,
        }
        with pytest.raises(SystemExit) as exc:
            sanity_check_history(hist, top20_reference=ref, tolerance=100, max_failures=5)
        assert exc.value.code == 2


# ── Golden fixture sanity ─────────────────────────────────────────────────────


def test_fallback_fixture_loadable() -> None:
    fixture = load_fallback_fixture()
    assert fixture is not None
    assert len(fixture) == 20
    # Spot-check: top 3 should be Spain/Argentina/France
    sorted_teams = sorted(fixture.items(), key=lambda x: -x[1])
    top3 = {t for t, _ in sorted_teams[:3]}
    assert top3 == {"Spain", "Argentina", "France"}
