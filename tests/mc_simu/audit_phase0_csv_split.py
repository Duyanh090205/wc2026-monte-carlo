"""Regression guard: RESULTS_CSV and matches_1998_2026.csv must produce
identical Elo via build_rating_history.

Originally written 2026-05-20 to PROVE a wiring bug: the filtered CSV
(formerly 1998+ only) lost 125 years of pre-1998 Elo accumulation, causing
200-Elo divergence at WC 2002 LOTO-CV. The fix (same date) removed the date
filter from data_audit.build_filtered_csv — matches_1998_2026.csv now contains
the FULL 1872+ range plus tournament_type + attendance_pct columns.

After fix: both CSVs produce IDENTICAL Elo for any team at any date.
This script is kept as a regression guard — if anyone reintroduces a date
filter or otherwise truncates the enriched CSV, the per-tournament diffs
will jump from +0 back to +200 at WC 2002, alerting reviewers.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from mc_simu.elo import build_rating_history, get_rating_as_of, latest_ratings  # noqa: E402

DATA = PROJECT_ROOT / "data" / "mc_simu"
RESULTS_CSV = DATA / "results.csv"
MATCHES_CSV = DATA / "matches_1998_2026.csv"


def banner(t: str) -> None:
    print("\n" + "=" * 78); print(t); print("=" * 78)


def build_both() -> tuple[dict, dict]:
    print("Building self-Elo from RESULTS_CSV (1872-2026, 49k rows) ...")
    raw = pd.read_csv(RESULTS_CSV)
    hist_raw = build_rating_history(raw)
    print(f"  teams: {len(hist_raw)}, latest match date: {raw['date'].max()}")

    print("Building self-Elo from MATCHES_CSV (1998-2026, 27k rows) ...")
    flt = pd.read_csv(MATCHES_CSV)
    hist_flt = build_rating_history(flt)
    print(f"  teams: {len(hist_flt)}, latest match date: {flt['date'].max()}")
    return hist_raw, hist_flt


def compare_latest(hist_raw: dict, hist_flt: dict) -> None:
    banner("AS-OF 2026-05-01: divergence on top-20 by raw-path Elo")
    latest_raw = latest_ratings(hist_raw)
    latest_flt = latest_ratings(hist_flt)
    common = sorted(latest_raw.keys() & latest_flt.keys(),
                    key=lambda t: -latest_raw[t])[:20]
    print(f"  {'team':28s} {'raw':>8s} {'filtered':>10s} {'diff':>8s}")
    diffs = []
    for t in common:
        r, f = latest_raw[t], latest_flt[t]
        d = r - f
        diffs.append(d)
        print(f"  {t:28s} {r:>8.0f} {f:>10.0f} {d:>+8.0f}")
    abs_diffs = [abs(d) for d in diffs]
    print(f"\n  Mean |diff| top-20: {sum(abs_diffs)/len(abs_diffs):.1f} Elo")
    print(f"  Max  |diff| top-20: {max(abs_diffs):.1f} Elo")
    print(f"  Median diff (signed): {sorted(diffs)[len(diffs)//2]:+.1f} Elo")


def compare_historical(hist_raw: dict, hist_flt: dict) -> None:
    """Phase 4 LOTO-CV will need as-of ratings at pre-tournament dates.
    Closer to 1998, the divergence grows (filtered path was just initialized).
    """
    banner("PRE-TOURNAMENT AS-OF ratings (Phase 4 LOTO-CV relevance)")
    test_cases = [
        ("WC 2002 (pre-kickoff)",  pd.Timestamp("2002-05-30")),
        ("WC 2006 (pre-kickoff)",  pd.Timestamp("2006-06-08")),
        ("Euro 2008 (pre-kickoff)", pd.Timestamp("2008-06-06")),
        ("WC 2018 (pre-kickoff)",  pd.Timestamp("2018-06-13")),
        ("WC 2022 (pre-kickoff)",  pd.Timestamp("2022-11-19")),
    ]
    sample_teams = ["Brazil", "Germany", "Argentina", "France", "Spain", "Italy"]
    for label, dt in test_cases:
        print(f"\n  {label} ({dt.date()}):")
        print(f"    {'team':14s} {'raw':>8s} {'filtered':>10s} {'diff':>8s}")
        for t in sample_teams:
            r = get_rating_as_of(hist_raw, t, dt)
            f = get_rating_as_of(hist_flt, t, dt)
            print(f"    {t:14s} {r:>8.0f} {f:>10.0f} {r - f:>+8.0f}")


if __name__ == "__main__":
    if not (RESULTS_CSV.exists() and MATCHES_CSV.exists()):
        print("ERROR: missing CSVs"); sys.exit(1)
    hist_raw, hist_flt = build_both()
    compare_latest(hist_raw, hist_flt)
    compare_historical(hist_raw, hist_flt)
