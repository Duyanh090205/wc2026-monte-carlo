"""Tests for backfill_reach — devig scaling + row assembly (offline, synthetic)."""

from __future__ import annotations

from mc_simu.backfill_reach import _devig_slots, build_rows


class TestDevigSlots:
    def test_scales_to_slot_count(self) -> None:
        raw = {"A": 0.9, "B": 0.6, "C": 0.5, "D": 0.4}  # sum 2.4, slots 2
        dv = _devig_slots(raw, 2)
        assert abs(sum(dv.values()) - 2.0) < 1e-9
        assert dv["A"] == 0.9 * 2 / 2.4

    def test_caps_at_one(self) -> None:
        dv = _devig_slots({"A": 0.99, "B": 0.03}, 2)
        assert dv["A"] == 1.0

    def test_resolved_event_dropped(self) -> None:
        assert _devig_slots({"A": 0.4}, 8) == {}

    def test_empty(self) -> None:
        assert _devig_slots({}, 4) == {}


class TestBuildRows:
    TEAMS = ["France", "Brazil", "Haiti"]

    def test_model_and_market_merge(self) -> None:
        model = {"2026-07-01": {"matches": 82, "reach": {
            "final": {"France": 0.4, "Brazil": 0.3}}}}
        poly = {"2026-07-01": {"final": {"France": 0.5, "Brazil": 0.4,
                                         "Haiti": 0.6, "Brasilia": 0.9}}}
        rows = build_rows(model, poly, self.TEAMS)
        by_team = {r["team"]: r for r in rows if r["round"] == "final"}
        assert "Brasilia" not in by_team          # not a WC2026 team
        assert by_team["France"]["model_pct"] == 40.0
        assert by_team["France"]["model_state_matches"] == 82
        assert by_team["France"]["pm_raw_pct"] == 50.0
        # devig: sum 1.5 -> scale by 2/1.5
        assert by_team["France"]["pm_devig_pct"] == round(0.5 * 2 / 1.5 * 100, 3)
        # Haiti: market-only -> model shows explicit 0 (sim ran that day)
        assert by_team["Haiti"]["model_pct"] == 0.0

    def test_no_model_day_blank(self) -> None:
        poly = {"2026-07-01": {"sf": {"France": 0.8, "Brazil": 0.7, "Haiti": 0.9}}}
        rows = build_rows({}, poly, self.TEAMS)
        assert all(r["model_pct"] == "" for r in rows)
        assert all(r["model_state_matches"] == "" for r in rows)

    def test_rounds_missing_from_both_sides_skipped(self) -> None:
        model = {"2026-07-01": {"matches": 82, "reach": {"qf": {"France": 0.9}}}}
        rows = build_rows(model, {}, self.TEAMS)
        assert {r["round"] for r in rows} == {"qf"}
