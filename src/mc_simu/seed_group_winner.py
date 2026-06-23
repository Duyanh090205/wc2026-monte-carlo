"""Seed the Supabase `group_winner_log` table from the backfilled CSV.

One-time upload of data/mc_simu/group_winner_log.csv (model vs Polymarket group
winner) so the deployed dashboard's Group-winner tab reads it from Supabase
instead of the local file. Idempotent: upserts on (date, group, team), so re-runs
overwrite rather than duplicate. Run the deploy/supabase_schema.sql migration
first (creates the table + anon-read policy).

    SUPABASE_URL=https://<project>.supabase.co \
    SUPABASE_SERVICE_KEY=<service_role key> \
    PYTHONPATH=src python -m mc_simu.seed_group_winner
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "data" / "mc_simu"

NUM = {"model_pct", "model_state_matches", "pm_raw_pct", "pm_devig_pct"}
INT = {"model_state_matches"}


def _row(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        if v == "" or v is None:
            out[k] = None
        elif k in INT:
            out[k] = int(float(v))
        elif k in NUM:
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main() -> int:
    import requests

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("Set SUPABASE_URL + SUPABASE_SERVICE_KEY in the environment.")
        return 2

    csv_path = DATA / "group_winner_log.csv"
    if not csv_path.exists():
        print(f"{csv_path} missing — run `python -m mc_simu.backfill_group_winner` first.")
        return 2
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [_row(r) for r in csv.DictReader(f)]
    dates = sorted({r["date"] for r in rows})
    print(f"Upserting {len(rows)} rows to group_winner_log ({dates[0]}..{dates[-1]}) ...")

    base = f"{url.rstrip('/')}/rest/v1/group_winner_log"
    auth = {"apikey": key, "Authorization": f"Bearer {key}"}
    # Clean-replace the backfilled window first, so rows dropped by a fixed
    # regenerate (e.g. resolved-group Poly garbage) don't linger from a prior seed.
    dele = requests.delete(
        f"{base}?date=gte.{dates[0]}&date=lte.{dates[-1]}",
        headers={**auth, "Prefer": "return=minimal"}, timeout=30)
    dele.raise_for_status()
    print(f"  cleared existing rows in {dates[0]}..{dates[-1]}")

    endpoint = f"{base}?on_conflict=date,group,team"
    headers = {**auth, "Content-Type": "application/json",
               "Prefer": "resolution=merge-duplicates,return=minimal"}
    for i in range(0, len(rows), 500):
        batch = rows[i:i + 500]
        resp = requests.post(endpoint, headers=headers, json=batch, timeout=30)
        resp.raise_for_status()
        print(f"  rows {i + 1}-{i + len(batch)} OK")
    print("Done. The dashboard Group-winner tab now reads from Supabase.")
    return 0


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
