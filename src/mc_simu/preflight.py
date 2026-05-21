"""Pre-flight checks for existing FairLine pipeline (spec §1.1).

Default invocation (v1 file-only):
    python src/mc_simu/preflight.py
    → Runs only Check 0.4 (entity_map.json file check).

Opt-in DB mode (v2, when DATABASE_URL is set):
    python src/mc_simu/preflight.py --with-db
    → Runs all 5 checks against the DB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: F401, E402 — side-effect: load .env
from mc_simu._common import banner, hard_stop  # noqa: E402


# Spec §1.1 expects 5 tables; current main has 7 (adds api_call_log, live_price_cache).
SPEC_TABLES = ["fair_odds", "pm_prices", "edge_signals", "paper_positions", "sportsbook_odds"]

# Spec §1.1 check 0.5: fair_odds must contain at least these columns.
FAIR_ODDS_EXPECTED_COLS = {
    "event", "market", "outcome", "fair_prob", "z_estimate", "n_sources", "computed_at",
}

ENTITY_MAP_PATH = PROJECT_ROOT / "data" / "clean" / "entity_map.json"


def check_entity_map() -> tuple[str, bool, str]:
    """Check 0.4 — entity_map.json exists. Returns (label, pass, detail)."""
    label = "0.4 entity_map.json"
    if not ENTITY_MAP_PATH.exists():
        hard_stop(label, "missing", f"{ENTITY_MAP_PATH} should exist (run normalize.py)")
    size = ENTITY_MAP_PATH.stat().st_size
    return label, True, f"OK ({size} bytes)"


def check_db_connectivity() -> tuple[str, bool, str]:
    """Check 0.1 — get_engine() returns a live Engine."""
    label = "0.1 DB connectivity"
    from db import get_engine

    eng = get_engine()
    if eng is None:
        hard_stop(label, "None", "live SQLAlchemy Engine (DATABASE_URL set)")
    return label, True, "OK (engine connected)"


def check_required_tables() -> tuple[str, bool, str]:
    """Check 0.2 — required tables exist."""
    label = "0.2 Required tables"
    from sqlalchemy import text

    from db import get_engine

    eng = get_engine()
    if eng is None:
        hard_stop(label, "no engine", "live engine for table introspection")
    assert eng is not None  # for type-checker
    missing: list[str] = []
    with eng.connect() as conn:
        for table in SPEC_TABLES:
            row = conn.execute(
                text("SELECT 1 FROM information_schema.tables WHERE table_name = :t"),
                {"t": table},
            ).fetchone()
            if row is None:
                missing.append(table)
    if missing:
        hard_stop(label, f"missing {missing}", f"all of {SPEC_TABLES}")
    return label, True, f"OK ({len(SPEC_TABLES)}/{len(SPEC_TABLES)} present)"


def check_recent_fair_odds() -> tuple[str, bool, str]:
    """Check 0.3 — fair_odds has writes in the last 7 days (Stream 1 alive)."""
    label = "0.3 Recent fair_odds writes"
    from sqlalchemy import text

    from db import get_engine

    eng = get_engine()
    if eng is None:
        hard_stop(label, "no engine", "live engine")
    assert eng is not None
    with eng.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM fair_odds WHERE computed_at > NOW() - INTERVAL '7 days'")
        ).scalar()
    if not count or count == 0:
        hard_stop(label, "0 rows in last 7d", ">0 — pipeline may be down")
    return label, True, f"OK ({count} rows in last 7d)"


def check_fair_odds_schema() -> tuple[str, bool, str]:
    """Check 0.5 — fair_odds columns include the spec-expected set."""
    label = "0.5 fair_odds schema"
    from sqlalchemy import text

    from db import get_engine

    eng = get_engine()
    if eng is None:
        hard_stop(label, "no engine", "live engine")
    assert eng is not None
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'fair_odds'"
            )
        ).fetchall()
    actual = {r[0] for r in rows}
    missing = FAIR_ODDS_EXPECTED_COLS - actual
    if missing:
        hard_stop(label, f"missing {missing}", f"⊇ {FAIR_ODDS_EXPECTED_COLS}")
    extra = sorted(actual - FAIR_ODDS_EXPECTED_COLS)
    return label, True, f"OK ({len(FAIR_ODDS_EXPECTED_COLS)} expected present; extras: {extra})"


def run_preflight(*, with_db: bool) -> list[tuple[str, str, str]]:
    """Run preflight checks. Returns list of (label, status, detail) tuples.

    status is one of: 'PASS', 'NOT RUN'. HARD STOP exits before returning if any
    check fails.
    """
    banner("MC Simu — Preflight Checks (§1.1" + (", DB mode" if with_db else ", file-only mode") + ")")

    results: list[tuple[str, str, str]] = []

    if with_db:
        for fn in (
            check_db_connectivity,
            check_required_tables,
            check_recent_fair_odds,
        ):
            label, _, detail = fn()
            print(f"Check {label:35s} ✓ {detail}")
            results.append((label, "PASS", detail))
    else:
        for label in ("0.1 DB connectivity", "0.2 Required tables", "0.3 Recent fair_odds writes"):
            print(f"Check {label:35s} NOT RUN (--with-db not set)")
            results.append((label, "NOT RUN", "v1 file-only mode"))

    # Check 0.4 — always runs (file-only)
    label, _, detail = check_entity_map()
    print(f"Check {label:35s} ✓ {detail}")
    results.append((label, "PASS", detail))

    if with_db:
        label, _, detail = check_fair_odds_schema()
        print(f"Check {label:35s} ✓ {detail}")
        results.append((label, "PASS", detail))
    else:
        label = "0.5 fair_odds schema"
        print(f"Check {label:35s} NOT RUN (--with-db not set)")
        results.append((label, "NOT RUN", "v1 file-only mode"))

    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_not_run = sum(1 for _, s, _ in results if s == "NOT RUN")
    banner(
        f"Preflight: {n_pass} PASS, {n_not_run} NOT RUN"
        + ("" if with_db else " (v1 file-only, DB checks for v2)")
    )
    if not with_db:
        print("Use --with-db flag to run DB checks (requires DATABASE_URL)")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MC Simu pre-flight audit (spec §1.1)")
    parser.add_argument(
        "--with-db",
        action="store_true",
        help="Run DB-dependent checks 0.1-0.3, 0.5 (requires DATABASE_URL)",
    )
    args = parser.parse_args(argv)
    run_preflight(with_db=args.with_db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
