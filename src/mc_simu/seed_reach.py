"""Seed the Supabase `reach_log` table from the backfilled CSV.

One-time upload of data/mc_simu/reach_log.csv (model vs Polymarket knockout
reach) so the deployed dashboard's Knockout rounds tab reads it from Supabase
instead of the local file. Idempotent: upserts on (date, round, team), so re-runs
overwrite rather than duplicate. Run the deploy/supabase_schema.sql migration
first (creates the table + anon-read policy).

    SUPABASE_URL=https://<project>.supabase.co \
    SUPABASE_SERVICE_KEY=<service_role key> \
    PYTHONPATH=src python -m mc_simu.seed_reach
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "data" / "mc_simu"


def main() -> int:
    import requests

    from mc_simu.seed_group_winner import _row, merge_existing

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("Set SUPABASE_URL + SUPABASE_SERVICE_KEY in the environment.")
        return 2

    csv_path = DATA / "reach_log.csv"
    if not csv_path.exists():
        print(f"{csv_path} missing — run `python -m mc_simu.backfill_reach` first.")
        return 2
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [_row(r) for r in csv.DictReader(f)]

    base = f"{url.rstrip('/')}/rest/v1/reach_log"
    auth = {"apikey": key, "Authorization": f"Bearer {key}"}
    rows = merge_existing(rows, base, auth, ("date", "round", "team"))
    dates = sorted({r["date"] for r in rows})
    print(f"Upserting {len(rows)} rows to reach_log ({dates[0]}..{dates[-1]}) ...")
    # Clean-replace the backfilled window first (see seed_group_winner).
    dele = requests.delete(
        f"{base}?date=gte.{dates[0]}&date=lte.{dates[-1]}",
        headers={**auth, "Prefer": "return=minimal"}, timeout=30)
    dele.raise_for_status()
    print(f"  cleared existing rows in {dates[0]}..{dates[-1]}")

    endpoint = f"{base}?on_conflict=date,round,team"
    headers = {**auth, "Content-Type": "application/json",
               "Prefer": "resolution=merge-duplicates,return=minimal"}
    for i in range(0, len(rows), 500):
        batch = rows[i:i + 500]
        resp = requests.post(endpoint, headers=headers, json=batch, timeout=30)
        resp.raise_for_status()
        print(f"  rows {i + 1}-{i + len(batch)} OK")
    print("Done. The dashboard Knockout rounds tab now reads from Supabase.")
    return 0


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
