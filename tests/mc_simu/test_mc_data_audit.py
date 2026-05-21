"""Tests for src/mc_simu/data_audit.py — Checks 1-3 (Day 1 scope)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mc_simu import data_audit


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_minimal_df() -> pd.DataFrame:
    """Construct a tiny DataFrame that satisfies Check 1's expectations.

    One row per (tournament, year) in TOURNAMENT_EXPECTED, repeated to reach
    expected count. Plus an 1872 row to anchor date range minimum.
    """
    rows = []
    rows.append({
        "date": "1872-11-30", "home_team": "Scotland", "away_team": "England",
        "home_score": 0, "away_score": 0, "tournament": "Friendly",
        "neutral": "FALSE", "country": "Scotland", "city": "Glasgow",
    })
    rows.append({
        "date": "2026-06-15", "home_team": "USA", "away_team": "Mexico",
        "home_score": 2, "away_score": 1, "tournament": "Friendly",
        "neutral": "FALSE", "country": "USA", "city": "Los Angeles",
    })
    # Add minimum 6 categories for Check 2 (post-filter to 1998+).
    # The "Friendly" row from the 1872 + 2026 entries above covers the 'friendly' bucket
    # (the 1872 row is dropped by the 1998+ filter, but the 2026 row stays).
    for tname, cat in [
        ("FIFA World Cup", "world_cup_final"),
        ("UEFA Euro", "continental_final"),
        ("FIFA World Cup qualification", "qualifier"),
        ("UEFA Nations League", "nations_league"),
        ("Kirin Cup", "other_tournament"),
    ]:
        rows.append({
            "date": "2020-01-15", "home_team": "A", "away_team": "B",
            "home_score": 1, "away_score": 0, "tournament": tname,
            "neutral": "TRUE", "country": "Neutral", "city": "X",
        })
    return pd.DataFrame(rows)


# ── infer_tournament_type bridge (it's tested in _common via preflight tests,
# but data_audit relies on it heavily so re-exercise here) ────────────────────


def test_filter_adds_tournament_type_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """build_filtered_csv derives tournament_type + attendance_pct columns.

    Date filter was removed 2026-05-20 (Phase 0 deep-audit-bug fix) — full range
    preserved so build_rating_history sees 1872+ history needed for early-LOTO Elo.
    """
    monkeypatch.setattr(data_audit, "FILTERED_CSV", tmp_path / "filtered.csv")
    df_raw = _make_minimal_df()
    out = data_audit.build_filtered_csv(df_raw)

    assert "tournament_type" in out.columns
    assert "attendance_pct" in out.columns
    # No date filter anymore — all rows preserved
    assert len(out) == len(df_raw)
    # Filtered file persisted
    assert (tmp_path / "filtered.csv").exists()
    persisted = pd.read_csv(tmp_path / "filtered.csv")
    assert len(persisted) == len(out)


# ── Check 1 ───────────────────────────────────────────────────────────────────


def test_check_1_passes_on_real_results_csv(mc_simu_data_dir: Path) -> None:
    """Real downloaded results.csv satisfies Check 1 (12 tournaments, 100% coverage)."""
    csv = mc_simu_data_dir / "results.csv"
    if not csv.exists():
        pytest.skip("results.csv not downloaded — run download_match_history.py")
    df = pd.read_csv(csv)
    r = data_audit.check_1_history_completeness(df)
    assert r.status == "PASS"
    assert "49," in r.detail or "100%" in r.detail  # rough sanity


def test_check_1_hard_stops_on_missing_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MC_SIMU_OVERRIDE_HARD_STOP", raising=False)
    df = pd.DataFrame({"date": ["2000-01-01"], "home_team": ["X"]})  # most cols missing
    with pytest.raises(SystemExit) as exc:
        data_audit.check_1_history_completeness(df)
    assert exc.value.code == 2


def test_check_1_hard_stops_on_short_date_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """If min date > 1998-01-01, Check 1 must HARD STOP."""
    monkeypatch.delenv("MC_SIMU_OVERRIDE_HARD_STOP", raising=False)
    df = pd.DataFrame({
        "date": ["2000-01-01", "2027-01-01"],
        "home_team": ["A", "B"], "away_team": ["B", "A"],
        "home_score": [1, 2], "away_score": [0, 1],
        "tournament": ["Friendly", "Friendly"],
        "neutral": ["FALSE", "FALSE"], "country": ["X", "Y"],
    })
    with pytest.raises(SystemExit) as exc:
        data_audit.check_1_history_completeness(df)
    assert exc.value.code == 2


# ── Check 2 ───────────────────────────────────────────────────────────────────


def test_check_2_passes_when_all_6_categories_present(monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: Path) -> None:
    monkeypatch.setattr(data_audit, "FILTERED_CSV", tmp_path / "f.csv")
    df = data_audit.build_filtered_csv(_make_minimal_df())
    r = data_audit.check_2_tournament_types(df)
    assert r.status == "PASS"
    assert "6/6" in r.detail


def test_check_2_hard_stops_when_category_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If tournament_type column has only friendlies, Check 2 must HARD STOP."""
    monkeypatch.delenv("MC_SIMU_OVERRIDE_HARD_STOP", raising=False)
    df = pd.DataFrame({"tournament_type": ["friendly"] * 10})
    with pytest.raises(SystemExit) as exc:
        data_audit.check_2_tournament_types(df)
    assert exc.value.code == 2


# ── Check 3 ───────────────────────────────────────────────────────────────────


def test_check_3_passes_on_real_filtered_data(mc_simu_data_dir: Path) -> None:
    csv = mc_simu_data_dir / "matches_1998_2026.csv"
    if not csv.exists():
        pytest.skip("matches_1998_2026.csv not built — run data_audit.py")
    df = pd.read_csv(csv)
    r = data_audit.check_3_neutral_flag(df)
    assert r.status == "PASS"
    assert "5/5 hosts validated" in r.detail


def test_check_3_hard_stops_when_host_match_misflagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a host's home match is flagged neutral, Check 3 must HARD STOP."""
    monkeypatch.delenv("MC_SIMU_OVERRIDE_HARD_STOP", raising=False)
    df = pd.DataFrame({
        "date": ["2014-06-12"], "tournament": ["FIFA World Cup"],
        "home_team": ["Brazil"], "away_team": ["Croatia"],
        "neutral": ["TRUE"],   # bug — host match should NOT be neutral
        "home_score": [3], "away_score": [1], "country": ["Brazil"],
    })
    with pytest.raises(SystemExit) as exc:
        data_audit.check_3_neutral_flag(df)
    assert exc.value.code == 2


# ── Orchestrator + report writer ──────────────────────────────────────────────


def test_audit_report_written_with_expected_sections(mc_simu_data_dir: Path,
                                                      project_root: Path) -> None:
    """After run_data_audit, audit_report.md has all expected section headers."""
    if not (mc_simu_data_dir / "results.csv").exists():
        pytest.skip("results.csv missing — run download_match_history.py")
    data_audit.run_data_audit()
    report = (project_root / "mc_simu" / "phase0_audit_report.md").read_text(encoding="utf-8")
    assert "# MC Simu Phase 0 Audit Report" in report
    assert "## Preflight checks (§1.1)" in report
    assert "## Data quality checks (§1.3)" in report
    assert "## Hard-stop status:" in report


# ── Check 4 ───────────────────────────────────────────────────────────────────


def test_check_4_passes_with_real_data(mc_simu_data_dir: Path,
                                        monkeypatch: pytest.MonkeyPatch,
                                        tmp_path: Path) -> None:
    """Check 4 merges Euro 2020 attendance into the filtered matches CSV."""
    matches_csv = mc_simu_data_dir / "matches_1998_2026.csv"
    att_csv = mc_simu_data_dir / "euro2020_attendance.csv"
    if not (matches_csv.exists() and att_csv.exists()):
        pytest.skip("Phase 0 inputs missing — run audit + Euro 2020 scraper first")

    # Use tmp path to avoid clobbering the real filtered CSV.
    monkeypatch.setattr(data_audit, "FILTERED_CSV", tmp_path / "filtered.csv")
    df_matches = pd.read_csv(matches_csv)
    df_att = pd.read_csv(att_csv)
    r = data_audit.check_4_euro2020_attendance(df_matches, df_att)
    assert r.status == "PASS"
    assert "51/51" in r.detail


def test_check_4_hard_stops_on_low_merge_coverage(monkeypatch: pytest.MonkeyPatch,
                                                   tmp_path: Path) -> None:
    """If <90% of Euro 2020 matches merge, Check 4 HARD STOPs."""
    monkeypatch.delenv("MC_SIMU_OVERRIDE_HARD_STOP", raising=False)
    monkeypatch.setattr(data_audit, "FILTERED_CSV", tmp_path / "f.csv")
    # 50 Euro 2020 matches in match df, but attendance df has only 5 (10% coverage)
    e2020_dates = pd.date_range("2021-06-11", periods=50, freq="D").strftime("%Y-%m-%d")
    df_matches = pd.DataFrame({
        "date": e2020_dates,
        "tournament": ["UEFA Euro"] * 50,
        "home_team": [f"T{i}" for i in range(50)],
        "away_team": [f"U{i}" for i in range(50)],
        "neutral": ["TRUE"] * 50,
        "attendance_pct": [pd.NA] * 50,
    })
    df_att = pd.DataFrame({
        "date": e2020_dates[:5],
        "home": [f"T{i}" for i in range(5)],
        "away": [f"U{i}" for i in range(5)],
        "attendance_pct": [0.2] * 5,
    })
    with pytest.raises(SystemExit) as exc:
        data_audit.check_4_euro2020_attendance(df_matches, df_att)
    assert exc.value.code == 2


# ── Check 5 ───────────────────────────────────────────────────────────────────


def test_check_5_warn_when_coverage_below_85pct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check 5 returns WARN (not HARD STOP) when any team < 85% coverage."""
    from mc_simu import clubelo_client

    def fake_get_clubelo(club, as_of=None):
        # Only "Real Madrid" resolves; rest miss.
        return 1900.0 if club == "Real Madrid" else None

    monkeypatch.setattr(clubelo_client, "get_clubelo", fake_get_clubelo)
    # Also patch in data_audit's import (the bound name inside Check 5 fn)
    monkeypatch.setattr("mc_simu.clubelo_client.get_clubelo", fake_get_clubelo)

    squads = {
        "TeamA": [{"club": "Real Madrid"}, {"club": "Unknown FC"}],  # 50%
        "TeamB": [{"club": "Real Madrid"}, {"club": "Real Madrid"}],  # 100%
    }
    r = data_audit.check_5_clubelo_coverage(squads)
    assert r.status == "WARN"
    assert "TeamA" in r.detail


def test_check_5_pass_when_all_teams_above_85pct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mc_simu.clubelo_client.get_clubelo", lambda c, as_of=None: 1900.0)
    squads = {"TeamA": [{"club": "X"}, {"club": "Y"}, {"club": "Z"}]}
    r = data_audit.check_5_clubelo_coverage(squads)
    assert r.status == "PASS"


# ── Check 6 ───────────────────────────────────────────────────────────────────


def test_check_6_passes_on_real_artifacts(mc_simu_data_dir: Path) -> None:
    fixtures_csv = mc_simu_data_dir / "wc2026_fixtures.csv"
    r32_json = mc_simu_data_dir / "r32_seeding_table.json"
    if not (fixtures_csv.exists() and r32_json.exists()):
        pytest.skip("Run build_wc2026_fixtures + build_r32_seeding first")
    import json
    df = pd.read_csv(fixtures_csv)
    r32 = json.loads(r32_json.read_text(encoding="utf-8"))
    r = data_audit.check_6_wc2026_structure(df, r32)
    assert r.status == "PASS"
    assert "495 R32" in r.detail


def test_check_6_hard_stops_on_wrong_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MC_SIMU_OVERRIDE_HARD_STOP", raising=False)
    bad_fixtures = pd.DataFrame({
        "match_id": range(50),
        "stage": ["group"] * 50,
        "group": ["A"] * 50,
    })
    with pytest.raises(SystemExit) as exc:
        data_audit.check_6_wc2026_structure(bad_fixtures, {})
    assert exc.value.code == 2
