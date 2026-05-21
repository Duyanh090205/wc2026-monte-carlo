"""Tests for src/mc_simu/preflight.py — spec §1.1 checks.

DB-dependent checks (0.1, 0.2, 0.3, 0.5) are exercised via mocking; only the
file-based check (0.4 entity_map.json) hits the real filesystem.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mc_simu import preflight


# ── Check 0.4: entity_map.json ────────────────────────────────────────────────


def test_entity_map_exists_in_repo(project_root: Path) -> None:
    """The repo ships data/clean/entity_map.json — preflight 0.4 must find it."""
    assert preflight.ENTITY_MAP_PATH == project_root / "data" / "clean" / "entity_map.json"
    label, ok, detail = preflight.check_entity_map()
    assert ok is True
    assert label == "0.4 entity_map.json"
    assert "OK" in detail


def test_entity_map_missing_triggers_hard_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If entity_map.json absent, check_entity_map() must HARD STOP (sys.exit 2)."""
    fake_path = tmp_path / "entity_map.json"  # does not exist
    monkeypatch.setattr(preflight, "ENTITY_MAP_PATH", fake_path)
    # Make sure override env isn't accidentally set
    monkeypatch.delenv("MC_SIMU_OVERRIDE_HARD_STOP", raising=False)

    with pytest.raises(SystemExit) as exc:
        preflight.check_entity_map()
    assert exc.value.code == 2


def test_entity_map_hard_stop_overridable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MC_SIMU_OVERRIDE_HARD_STOP=1 bypasses the hard stop (spec §0.7)."""
    fake_path = tmp_path / "entity_map.json"
    monkeypatch.setattr(preflight, "ENTITY_MAP_PATH", fake_path)
    monkeypatch.setenv("MC_SIMU_OVERRIDE_HARD_STOP", "1")

    # Should NOT raise — hard_stop returns instead of exiting.
    # check_entity_map() will still try to stat the file after the override,
    # which raises FileNotFoundError. That's acceptable — override skips the
    # ordered exit but doesn't fabricate a passing return value.
    with pytest.raises(FileNotFoundError):
        preflight.check_entity_map()


# ── run_preflight() orchestration ─────────────────────────────────────────────


def test_run_preflight_file_only_runs_just_check_04(capsys: pytest.CaptureFixture[str]) -> None:
    """Default (no --with-db) → 1 PASS (0.4), 4 NOT RUN."""
    results = preflight.run_preflight(with_db=False)
    assert len(results) == 5
    statuses = [s for _, s, _ in results]
    assert statuses.count("PASS") == 1
    assert statuses.count("NOT RUN") == 4

    out = capsys.readouterr().out
    assert "Preflight: 1 PASS, 4 NOT RUN" in out
    assert "Use --with-db flag" in out


def test_run_preflight_with_db_all_5_run(monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    """--with-db → all 5 checks run when DB is mocked to PASS."""
    monkeypatch.setattr(preflight, "check_db_connectivity",
                        lambda: ("0.1 DB connectivity", True, "OK (mocked)"))
    monkeypatch.setattr(preflight, "check_required_tables",
                        lambda: ("0.2 Required tables", True, "OK (5/5 mocked)"))
    monkeypatch.setattr(preflight, "check_recent_fair_odds",
                        lambda: ("0.3 Recent fair_odds writes", True, "OK (42 rows mocked)"))
    monkeypatch.setattr(preflight, "check_fair_odds_schema",
                        lambda: ("0.5 fair_odds schema", True, "OK (mocked)"))

    results = preflight.run_preflight(with_db=True)
    statuses = [s for _, s, _ in results]
    assert statuses.count("PASS") == 5
    assert statuses.count("NOT RUN") == 0

    out = capsys.readouterr().out
    assert "Preflight: 5 PASS, 0 NOT RUN" in out


# ── CLI entry point ───────────────────────────────────────────────────────────


def test_main_no_args_returns_0() -> None:
    rc = preflight.main([])
    assert rc == 0


def test_main_with_db_flag_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag --with-db is forwarded to run_preflight(with_db=True)."""
    seen = {}

    def fake_run(*, with_db: bool):
        seen["with_db"] = with_db
        return []

    monkeypatch.setattr(preflight, "run_preflight", fake_run)
    rc = preflight.main(["--with-db"])
    assert rc == 0
    assert seen["with_db"] is True
