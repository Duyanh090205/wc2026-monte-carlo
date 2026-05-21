"""Smoke test for src/mc_simu/validate_phase1.py — gate threshold semantics."""

from __future__ import annotations

import pytest

from mc_simu.validate_phase1 import gate_verdict


class TestGateVerdict:
    """3-tier gate boundaries: ≤0.19 PASS, ≤0.21 WARN, >0.21 HARD STOP."""

    @pytest.mark.parametrize("brier,expected_verdict,expected_code", [
        (0.10, "PASS", 0),
        (0.19, "PASS", 0),         # boundary inclusive
        (0.1900001, "PASS-WARN", 1),
        (0.20, "PASS-WARN", 1),
        (0.21, "PASS-WARN", 1),    # boundary inclusive
        (0.2100001, "HARD-STOP-PIVOT", 2),
        (0.30, "HARD-STOP-PIVOT", 2),
    ])
    def test_boundaries(self, brier: float, expected_verdict: str, expected_code: int) -> None:
        verdict, code = gate_verdict(brier)
        assert verdict == expected_verdict
        assert code == expected_code
