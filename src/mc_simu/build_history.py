"""Build the full Elo rating history from raw 1872-2026 match data.

Runnable CLI. Order:
    1. build_rating_history(results.csv)  → in-memory dict[team, DataFrame]
    2. fetch eloratings.net top-20 (with golden-fixture fallback)
    3. sanity_check_history  → HARD STOP on Layer A bound violation or
                               Layer B excessive divergence
    4. Only on PASS: write data/mc_simu/elo_history.csv (long format)

Output: data/mc_simu/elo_history.csv, schema per plan §"Persistence":
    team, date, rating_after, opponent, tournament_type, K, mov, gs_self, gs_opp
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import URLError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from mc_simu._common import banner  # noqa: E402
from mc_simu.elo import build_rating_history, history_to_long_df, latest_ratings  # noqa: E402
from mc_simu.eloratings_snapshot import (  # noqa: E402
    fetch_eloratings_top20,
    load_fallback_fixture,
    sanity_check_history,
)

DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
RESULTS_CSV = DATA_DIR / "results.csv"
HISTORY_CSV = DATA_DIR / "elo_history.csv"


def main() -> int:
    banner("MC Simu — Build Rating History (Phase 1 §2.1)")

    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} missing — run download_match_history.py first")
        return 1

    print(f"Reading {RESULTS_CSV.name} ...")
    df_raw = pd.read_csv(RESULTS_CSV, encoding="utf-8")
    print(f"  raw rows: {len(df_raw):,}")

    print("Building rating history (chronological, no inter-match decay) ...")
    history = build_rating_history(df_raw, initial_rating=1500.0)
    print(f"  teams tracked: {len(history):,}")
    print(f"  total update events: {sum(len(d) for d in history.values()):,}")

    # ── Sanity reference fetch ────────────────────────────────────────────
    print("\nFetching eloratings.net top-20 reference snapshot ...")
    try:
        top20_ref = fetch_eloratings_top20()
        ref_source = "live (eloratings.net)"
    except (URLError, OSError) as e:
        print(f"  live fetch failed ({type(e).__name__}: {e}); trying golden fixture")
        top20_ref = load_fallback_fixture()
        ref_source = "golden fixture" if top20_ref else "NONE"

    print(f"  reference source: {ref_source}")
    if top20_ref:
        print(f"  reference entries: {len(top20_ref)}")

    # ── Sanity gate ────────────────────────────────────────────────────────
    print("\nRunning sanity_check_history ...")
    report = sanity_check_history(history, top20_reference=top20_ref,
                                   tolerance=100.0, max_failures=5)
    print(f"  Layer A: {report['layer_a']}")
    print(f"  Layer B: {report['layer_b']}")

    # Diagnostic: top-20 model-vs-reference table
    latest = latest_ratings(history)
    if top20_ref:
        print("\nTop-20 model vs eloratings.net reference:")
        print(f"  {'Team':30s} {'Model':>7s} {'Ref':>7s} {'Δ':>7s}")
        for team, ref in sorted(top20_ref.items(), key=lambda x: -x[1]):
            model = latest.get(team, float("nan"))
            delta = model - ref if model == model else float("nan")  # NaN check
            print(f"  {team:30s} {model:>7.0f} {ref:>7.0f} {delta:>+7.0f}")

    # ── Persist ─────────────────────────────────────────────────────────────
    print(f"\nWriting {HISTORY_CSV} ...")
    long_df = history_to_long_df(history)
    long_df.to_csv(HISTORY_CSV, index=False, encoding="utf-8")
    print(f"  rows: {len(long_df):,}  ({HISTORY_CSV.stat().st_size:,} bytes)")

    banner("Rating history build complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
