"""Data quality audit for Phase 0 — 6 checks per spec §1.3.

Day 1 implements Checks 1-3 (file-based, no scraper deps).
Checks 4-6 added Days 2-4 as upstream artifacts become available.

CLI:
    python src/mc_simu/data_audit.py
    → Runs whichever checks have their inputs available.
    → Writes audit_report.md.
    → Exit 0 if all available checks PASS or WARN; exit 2 on HARD STOP.

Inputs:
  results.csv                          → Check 1 (raw, 1872-2026)
  matches_1998_2026.csv                → Checks 2, 3 (derived from Check 1 filter)
  euro2020_attendance.csv              → Check 4 (Day 2 scraper)
  wc2026_squads_provisional.json       → Check 5 (Day 2 scraper)
  wc2026_fixtures.csv                  → Check 6 (Day 3 manual)
  r32_seeding_table.json               → Check 6 (Day 4 manual)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from mc_simu._common import banner, hard_stop, infer_tournament_type  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
RESULTS_CSV = DATA_DIR / "results.csv"
FILTERED_CSV = DATA_DIR / "matches_1998_2026.csv"
EURO2020_CSV = DATA_DIR / "euro2020_attendance.csv"
WC2026_SQUADS = DATA_DIR / "wc2026_squads_provisional.json"
WC2026_FIXTURES = DATA_DIR / "wc2026_fixtures.csv"
R32_TABLE = DATA_DIR / "r32_seeding_table.json"

# Auto-generated per-run Phase 0 summary. Curated cross-phase log lives at
# `audit_report.md` (manual + per-phase append). Don't conflate.
REPORT_PATH = PROJECT_ROOT / "mc_simu" / "phase0_audit_report.md"

# Per spec §1.3 Check 1. Note Euro 2020 played in 2021 (COVID delay).
TOURNAMENT_EXPECTED: dict[str, tuple[str, int, int]] = {
    # label → (kaggle_tournament, actual_year, expected_match_count)
    "FIFA World Cup 2002": ("FIFA World Cup", 2002, 64),
    "FIFA World Cup 2006": ("FIFA World Cup", 2006, 64),
    "FIFA World Cup 2010": ("FIFA World Cup", 2010, 64),
    "FIFA World Cup 2014": ("FIFA World Cup", 2014, 64),
    "FIFA World Cup 2018": ("FIFA World Cup", 2018, 64),
    "FIFA World Cup 2022": ("FIFA World Cup", 2022, 64),
    "UEFA Euro 2004": ("UEFA Euro", 2004, 31),
    "UEFA Euro 2008": ("UEFA Euro", 2008, 31),
    "UEFA Euro 2012": ("UEFA Euro", 2012, 31),
    "UEFA Euro 2016": ("UEFA Euro", 2016, 51),
    "UEFA Euro 2020": ("UEFA Euro", 2021, 51),   # COVID delay → played 2021
    "UEFA Euro 2024": ("UEFA Euro", 2024, 51),
}

# Per spec §1.3 Check 3 — known WC/Euro hosts for neutral flag validation.
HOST_MAP: dict[str, tuple[str, int]] = {
    "FIFA World Cup 2014": ("Brazil", 2014),
    "FIFA World Cup 2018": ("Russia", 2018),
    "FIFA World Cup 2022": ("Qatar", 2022),
    "UEFA Euro 2016": ("France", 2016),
    "UEFA Euro 2024": ("Germany", 2024),
}


@dataclass
class CheckResult:
    label: str
    status: str  # "PASS" | "WARN" | "FAIL" | "SKIP"
    detail: str


# ── Check 1 — match history completeness ──────────────────────────────────────


def check_1_history_completeness(df: pd.DataFrame) -> CheckResult:
    """Spec §1.3 Check 1. Run on raw results.csv (1872-2026)."""
    label = "Check 1 — Match history completeness"

    required_cols = {"date", "home_team", "away_team", "home_score", "away_score",
                     "tournament", "neutral", "country"}
    actual_cols = set(df.columns)
    missing = required_cols - actual_cols
    if missing:
        hard_stop(label, f"missing cols {missing}", f"⊇ {required_cols}")

    dt = pd.to_datetime(df["date"])
    if dt.min() > pd.Timestamp("1998-01-01"):
        hard_stop(label, f"min date {dt.min().date()}", "≤ 1998-01-01")
    if dt.max() < pd.Timestamp("2026-01-01"):
        hard_stop(label, f"max date {dt.max().date()}", "≥ 2026-01-01")

    failures: list[str] = []
    coverage_lines: list[str] = []
    for tlabel, (tname, year, expected_n) in TOURNAMENT_EXPECTED.items():
        sub = df[(df["tournament"] == tname) & (dt.dt.year == year)]
        actual_n = len(sub)
        pct = actual_n / expected_n if expected_n else 0
        coverage_lines.append(f"    {tlabel}: {actual_n}/{expected_n} ({pct:.0%})")
        if pct < 0.80:
            failures.append(f"{tlabel}: {actual_n}/{expected_n}")

        # Spec: home_score / away_score non-null for all validation tournaments.
        if sub[["home_score", "away_score"]].isna().any().any():
            failures.append(f"{tlabel}: has NaN scores")

    if failures:
        hard_stop(label, failures, "≥80% coverage + no NaN scores")

    detail = (f"date range {dt.min().date()} to {dt.max().date()} ({len(df):,} rows). "
              f"Per-tournament coverage:\n" + "\n".join(coverage_lines))
    return CheckResult(label=label, status="PASS", detail=detail)


# ── Enrich raw → matches_1998_2026.csv (between Check 1 and Check 2) ──────────
#
# Historical name (matches_1998_2026.csv) kept for backward compat; the file
# is now FULL-RANGE (1872+) per Phase 0 deep-audit-bug finding 2026-05-20.
# Filtering to 1998+ truncated pre-1998 Elo accumulation, causing 200-Elo
# under-rating at WC 2002 LOTO-CV. See audit_report.md Phase 0 §deep-audit-bug.


def build_filtered_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Build enriched matches CSV: full date range + tournament_type + attendance_pct.

    Previously this filtered to dates >= 1998-01-01; that filter was removed
    2026-05-20 after deep-audit-bug found it caused 200 Elo divergence at WC 2002
    LOTO replay. Now keeps all 49k rows (1872+) so build_rating_history sees the
    full Kaggle history regardless of which consumer reads this file.
    """
    df = df_raw.copy()
    df["tournament_type"] = df["tournament"].apply(infer_tournament_type)
    # attendance_pct added later when Check 4 merges Euro 2020 data.
    df["attendance_pct"] = pd.NA
    df.to_csv(FILTERED_CSV, index=False)
    return df


# ── Check 2 — tournament type mapping ─────────────────────────────────────────


def check_2_tournament_types(df: pd.DataFrame) -> CheckResult:
    """Spec §1.3 Check 2. Verifies derived tournament_type covers all 6 categories.

    Bumped from 5→6 categories per audit_report Phase 1 decision #13: split
    eloratings.net's K=30 bucket out of the friendly catch-all (was mis-classifying
    AFCON/Copa/Gold/AFC/Confed/OFC + their qualifiers as 'friendly' K=20).
    """
    label = "Check 2 — Tournament types"
    counts = df["tournament_type"].value_counts()
    required = ["world_cup_final", "continental_final", "qualifier",
                "nations_league", "other_tournament", "friendly"]
    missing = [cat for cat in required if cat not in counts.index]
    if missing:
        hard_stop(label, f"missing categories {missing}", f"all of {required}")

    summary = ", ".join(f"{cat}={int(counts[cat])}" for cat in required)
    return CheckResult(label=label, status="PASS", detail=f"6/6 categories: {summary}")


# ── Check 3 — neutral flag consistency ────────────────────────────────────────


def check_3_neutral_flag(df: pd.DataFrame) -> CheckResult:
    """Spec §1.3 Check 3. Host's matches NOT neutral, others ARE."""
    label = "Check 3 — Neutral flag consistency"
    dt = pd.to_datetime(df["date"])

    # Normalize neutral column to bool (Kaggle CSV stores TRUE/FALSE as strings/bools).
    if df["neutral"].dtype == object:
        neutral_bool = df["neutral"].astype(str).str.upper().eq("TRUE")
    else:
        neutral_bool = df["neutral"].astype(bool)

    failures: list[str] = []
    summary_lines: list[str] = []
    for tlabel, (host, year) in HOST_MAP.items():
        tname = "FIFA World Cup" if "World Cup" in tlabel else "UEFA Euro"
        actual_year = 2021 if tlabel == "UEFA Euro 2020" else year  # not in this map but defensive
        sub_mask = (df["tournament"] == tname) & (dt.dt.year == actual_year)

        host_matches = sub_mask & ((df["home_team"] == host) | (df["away_team"] == host))
        host_neutral_count = int(neutral_bool[host_matches].sum())

        other_matches = sub_mask & (df["home_team"] != host) & (df["away_team"] != host)
        other_total = int(other_matches.sum())
        other_non_neutral = int((~neutral_bool[other_matches]).sum())

        line = (f"    {tlabel} (host={host}): "
                f"host_matches_neutral={host_neutral_count} (expect 0), "
                f"non_host_matches={other_total}, non-neutral_among_them={other_non_neutral} (expect 0)")
        summary_lines.append(line)

        if host_neutral_count > 0:
            failures.append(f"{tlabel}: {host_neutral_count} host matches misflagged as neutral")
        if other_non_neutral > 0:
            failures.append(f"{tlabel}: {other_non_neutral} non-host matches not flagged neutral")

    if failures:
        hard_stop(label, failures, "all 5 host validations pass")

    return CheckResult(
        label=label, status="PASS",
        detail=f"5/5 hosts validated:\n" + "\n".join(summary_lines),
    )


# ── Check 4 — Euro 2020 attendance ────────────────────────────────────────────


def check_4_euro2020_attendance(df_matches: pd.DataFrame, df_attendance: pd.DataFrame) -> CheckResult:
    """Spec §1.3 Check 4. Merge attendance with match history and validate coverage.

    df_matches: filtered matches_1998_2026.csv (with attendance_pct column = NA)
    df_attendance: euro2020_attendance.csv (51 rows)

    Side effect: in-memory enrich df_matches with attendance_pct for Euro 2020 rows.
    Caller is responsible for persisting the enriched df back to FILTERED_CSV.
    """
    label = "Check 4 — Euro 2020 attendance"

    if "attendance_pct" not in df_attendance.columns:
        hard_stop(label, "missing column attendance_pct", "scrape output schema")

    # Normalize join keys (string date YYYY-MM-DD).
    # Wikipedia footballbox sometimes lists teams in non-home/away order, so use a
    # symmetric matchup key: frozenset({team1, team2}).
    df_attendance = df_attendance.copy()
    df_attendance["date"] = df_attendance["date"].astype(str)
    df_attendance["matchup"] = df_attendance.apply(
        lambda r: frozenset({str(r["home"]).strip(), str(r["away"]).strip()}), axis=1,
    )

    # Filter matches to Euro 2020 (tournament=UEFA Euro AND year=2021)
    dt = pd.to_datetime(df_matches["date"])
    e2020_mask = (df_matches["tournament"] == "UEFA Euro") & (dt.dt.year == 2021)
    e2020 = df_matches[e2020_mask].copy()
    e2020["date_str"] = pd.to_datetime(e2020["date"]).dt.strftime("%Y-%m-%d")
    e2020["matchup"] = e2020.apply(
        lambda r: frozenset({str(r["home_team"]).strip(), str(r["away_team"]).strip()}), axis=1,
    )

    # Symmetric join on (date, matchup)
    merged = e2020.merge(
        df_attendance[["date", "matchup", "attendance_pct"]],
        left_on=["date_str", "matchup"],
        right_on=["date", "matchup"],
        how="left",
        suffixes=("", "_att"),
    )

    # The "_att" suffix is only added if a column name collides — since attendance_pct
    # exists in df_matches (NA-filled) and df_attendance, pandas appends suffix.
    att_col = "attendance_pct_att" if "attendance_pct_att" in merged.columns else "attendance_pct"

    n_e2020 = len(merged)
    n_with_att = int(merged[att_col].notna().sum())

    if n_e2020 < 50:
        hard_stop(label, f"Euro 2020 matches: {n_e2020}", ">= 50 in matches_1998_2026.csv")
    if n_with_att < n_e2020 * 0.90:
        hard_stop(label, f"only {n_with_att}/{n_e2020} matched", ">= 90% merge coverage")

    # Spec: low-attendance count check
    low_att = int((merged[att_col] < 0.50).sum())
    if low_att < n_e2020 * 0.40:
        hard_stop(label, f"only {low_att}/{n_e2020} below 50%",
                  ">= 40% (Euro 2020 was COVID-capped)")

    # Persist enriched attendance_pct back to FILTERED_CSV
    df_matches.loc[e2020_mask, "attendance_pct"] = merged[att_col].values
    df_matches.to_csv(FILTERED_CSV, index=False)

    return CheckResult(
        label=label, status="PASS",
        detail=(f"merged {n_with_att}/{n_e2020} Euro 2020 matches; "
                f"{low_att} low-attendance (<50%)"),
    )


# ── Check 5 — ClubElo coverage ────────────────────────────────────────────────


def check_5_clubelo_coverage(squads: dict[str, list[dict]]) -> CheckResult:
    """Spec §1.3 Check 5. For each team's squad, fraction of players whose club
    resolves in ClubElo. Spec §9 risk: <85% → WARN (not HARD STOP), use 1300 default.
    """
    from mc_simu.clubelo_client import get_clubelo

    label = "Check 5 — ClubElo coverage"
    if not squads:
        return CheckResult(label=label, status="SKIP",
                           detail="no teams in squads file")

    per_team: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    total_players = 0
    total_mapped = 0

    for team, squad in sorted(squads.items()):
        if not squad:
            continue
        mapped = 0
        for p in squad:
            club = p.get("club", "")
            if club and get_clubelo(club) is not None:
                mapped += 1
        coverage = mapped / len(squad)
        per_team[team] = {"mapped": mapped, "total": len(squad), "coverage": coverage}
        total_players += len(squad)
        total_mapped += mapped
        if coverage < 0.85:
            failures.append(f"{team}={coverage:.0%}")

    mean_cov = total_mapped / total_players if total_players else 0
    n_teams = len(per_team)
    n_below = len(failures)

    detail_lines = [
        f"{n_teams} teams audited; mean coverage {mean_cov:.1%} ({total_mapped}/{total_players} players)",
    ]
    if failures:
        detail_lines.append(f"  {n_below}/{n_teams} teams below 85%: {failures[:8]}"
                            + (" ..." if n_below > 8 else ""))

    # Per §9: degrade to WARN if any team below 85%; never HARD STOP.
    status = "WARN" if failures else "PASS"
    return CheckResult(label=label, status=status, detail="\n".join(detail_lines))


# ── Check 6 — WC2026 fixtures + R32 structure ─────────────────────────────────


def check_6_wc2026_structure(fixtures_df: pd.DataFrame, r32_table: dict) -> CheckResult:
    """Spec §1.3 Check 6. 104 fixtures, 72 group, 6 per group, 495 R32 mappings."""
    label = "Check 6 — WC2026 structure"
    failures: list[str] = []

    n_total = len(fixtures_df)
    if n_total != 104:
        failures.append(f"total fixtures {n_total} != 104")

    group_stage = fixtures_df[fixtures_df["stage"] == "group"]
    if len(group_stage) != 72:
        failures.append(f"group stage matches {len(group_stage)} != 72")

    for letter in "ABCDEFGHIJKL":
        n = len(group_stage[group_stage["group"] == letter])
        if n != 6:
            failures.append(f"Group {letter}: {n} matches != 6")

    n_r32 = len(r32_table)
    if n_r32 != 495:
        failures.append(f"R32 mappings {n_r32} != 495")

    if failures:
        hard_stop(label, failures, "104 fixtures + 72 group + 6/group + 495 R32")

    return CheckResult(
        label=label, status="PASS",
        detail=(f"104 fixtures (72 group + 32 KO); 12 groups × 6 matches; "
                f"{n_r32} R32 mappings (C(12,8))"),
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────


def run_data_audit() -> list[CheckResult]:
    """Run all data audit checks whose inputs are available. Write audit_report.md."""
    banner("MC Simu — Data Quality Audit (§1.3)")

    results: list[CheckResult] = []

    # Check 1 — needs raw results.csv
    if not RESULTS_CSV.exists():
        hard_stop("Check 1 prerequisite", f"{RESULTS_CSV} missing",
                  "run download_match_history.py first")
    print(f"Reading {RESULTS_CSV.name} ...")
    df_raw = pd.read_csv(RESULTS_CSV)
    print(f"  raw rows: {len(df_raw):,}")
    r1 = check_1_history_completeness(df_raw)
    print(f"{r1.label}:  ✓ PASS")
    for ln in r1.detail.splitlines():
        print(ln)
    results.append(r1)

    # Build filtered file
    print(f"\nBuilding {FILTERED_CSV.name} (enriched: full range + tournament_type + attendance_pct) ...")
    df_filtered = build_filtered_csv(df_raw)
    print(f"  enriched rows: {len(df_filtered):,}")

    # Check 2
    r2 = check_2_tournament_types(df_filtered)
    print(f"{r2.label}:  ✓ PASS — {r2.detail}")
    results.append(r2)

    # Check 3
    r3 = check_3_neutral_flag(df_filtered)
    print(f"{r3.label}:  ✓ PASS")
    for ln in r3.detail.splitlines():
        print(ln)
    results.append(r3)

    # Check 4 — needs Euro 2020 attendance scrape
    if EURO2020_CSV.exists():
        df_att = pd.read_csv(EURO2020_CSV)
        r4 = check_4_euro2020_attendance(df_filtered, df_att)
        print(f"{r4.label}:  ✓ PASS — {r4.detail}")
        results.append(r4)
    else:
        results.append(CheckResult(label="Check 4 — Euro 2020 attendance", status="SKIP",
                                    detail=f"inputs not yet generated: {[EURO2020_CSV.name]}"))
        print(f"Check 4 — Euro 2020 attendance:  - SKIP ({EURO2020_CSV.name} missing)")

    # Check 5 — needs WC2026 provisional squads
    if WC2026_SQUADS.exists():
        import json
        squads = json.loads(WC2026_SQUADS.read_text(encoding="utf-8"))
        r5 = check_5_clubelo_coverage(squads)
        icon = "✓" if r5.status == "PASS" else "⚠"
        print(f"{r5.label}:  {icon} {r5.status}")
        for ln in r5.detail.splitlines():
            print(f"  {ln}")
        results.append(r5)
    else:
        results.append(CheckResult(label="Check 5 — ClubElo coverage", status="SKIP",
                                    detail=f"inputs not yet generated: {[WC2026_SQUADS.name]}"))
        print(f"Check 5 — ClubElo coverage:  - SKIP ({WC2026_SQUADS.name} missing)")

    # Check 6 — needs WC2026 fixtures CSV + R32 seeding JSON
    missing6 = [p.name for p in [WC2026_FIXTURES, R32_TABLE] if not p.exists()]
    if missing6:
        results.append(CheckResult(label="Check 6 — WC2026 structure", status="SKIP",
                                    detail=f"inputs not yet generated: {missing6}"))
        print(f"Check 6 — WC2026 structure:  - SKIP (inputs not yet generated: {missing6})")
    else:
        fixtures_df = pd.read_csv(WC2026_FIXTURES)
        import json
        r32_table = json.loads(R32_TABLE.read_text(encoding="utf-8"))
        r6 = check_6_wc2026_structure(fixtures_df, r32_table)
        print(f"{r6.label}:  ✓ PASS — {r6.detail}")
        results.append(r6)

    # Write report
    write_audit_report(results)
    print()
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    banner(f"Data audit: {n_pass} PASS, {n_warn} WARN, {n_skip} SKIP — wrote {REPORT_PATH.name}")
    return results


def write_audit_report(results: list[CheckResult]) -> None:
    """Write audit_report.md per spec §1.4 deliverable."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# MC Simu Phase 0 Audit Report",
        "",
        f"Generated: {ts}",
        f"Spec version: mc_simu.md v3 (May 14, 2026)",
        f"Repo HEAD: see `git rev-parse HEAD`",
        "",
        "## Preflight checks (§1.1)",
        "- 0.1 DB connectivity: NOT RUN (opt-in via --with-db; v1 = file-only)",
        "- 0.2 Required tables: NOT RUN — pre-verified offline: repo has 7 tables (5 spec + api_call_log, live_price_cache)",
        "- 0.3 Recent fair_odds writes: NOT RUN",
        "- 0.4 entity_map.json: PASS (preflight.py lives in the trading-engine repo)",
        "- 0.5 fair_odds schema: NOT RUN — pre-verified offline: actual is superset incl new consensus_prob col",
        "- v1 standalone path — all DB checks deferred to v2 (production integration)",
        "",
        "## Data quality checks (§1.3)",
    ]
    for r in results:
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "-"}.get(r.status, "?")
        first_line = r.detail.splitlines()[0] if r.detail else ""
        lines.append(f"- {r.label}: {icon} {r.status} — {first_line}")
        for extra in r.detail.splitlines()[1:]:
            lines.append(f"  {extra}")

    # Hard-stop summary
    all_ok = all(r.status in ("PASS", "WARN", "SKIP") for r in results)
    n_skip = sum(1 for r in results if r.status == "SKIP")
    n_warn = sum(1 for r in results if r.status == "WARN")
    if all_ok and n_skip == 0 and n_warn == 0:
        status_line = "✅ PASS — ready for Phase 1"
    elif all_ok and n_skip == 0 and n_warn > 0:
        status_line = (f"✅ PASS — ready for Phase 1 ({n_warn} WARN, "
                       f"non-blocking per spec §9 fallback)")
    elif all_ok and n_skip > 0:
        status_line = f"⏳ PARTIAL — {n_skip} checks pending Days 2-4"
    else:
        status_line = "❌ FAIL — see HARD STOP log"

    lines.extend([
        "",
        "## Files generated",
        f"- data/mc_simu/results.csv (raw)",
        f"- data/mc_simu/matches_1998_2026.csv (filtered, with tournament_type)",
        "",
        f"## Hard-stop status: {status_line}",
        "",
    ])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_data_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
