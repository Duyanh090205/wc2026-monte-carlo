"""WC2026 MC simulation distribution vs FairLine fair_prob (Stream 1).

Answers Sam's request (2026-05-22): for all 48 WC2026 teams, show where the
MC simulator's champion-probability output overstates vs understates compared
to FairLine's devigged sportsbook fair_prob.

Sources:
  --mc-csv     MC output (default: data/mc_simu/mc_wc2026_v1_100k_seed42.csv).
               Champion-market rows only (market_type=='champion').
  --fair-csv   Local fair_odds snapshot CSV (default:
               data/fair_odds/world_cup_2026/winner/latest.csv).
               This is a STALE FILE (last updated 2026-04-23 per file mtime).
  --use-db     Fetch fresh fair_odds from Supabase via db.latest_fair_odds
               instead of reading the local CSV. Requires DATABASE_URL.

Output:
  data/mc_simu/phase3_baselines/wc2026_vs_fair.csv  (gitignored)
  Markdown table to stdout: top 10 overstates + top 10 understates.

Sign convention: edge = mc_prob - fair_prob.
  POSITIVE → MC overstates the team's chance (model > market)
  NEGATIVE → MC understates the team's chance (model < market)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mc_simu._common import banner  # noqa: E402


# Team-name aliases: map fair_odds source spelling → MC adapter spelling.
# (Applied to the fair_odds side so it joins with MC keys.)
FAIR_TEAM_ALIASES: dict[str, str] = {
    "Korea Republic":         "South Korea",
    "Cabo Verde":             "Cape Verde",
    "Cote d'Ivoire":          "Ivory Coast",
    "Czech Republic":         "Czechia",
    "Bosnia & Herzegovina":   "Bosnia and Herzegovina",
    "Curaçao":                "Curacao",
    "USA":                    "United States",
}


def load_mc_champions(csv_path: Path) -> dict[str, float]:
    """Return {team: mc_champion_prob} from MC output CSV."""
    out: dict[str, float] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["market_type"] == "champion":
                out[row["team"]] = float(row["mc_fair_prob"])
    return out


def load_fair_local(csv_path: Path) -> tuple[dict[str, float], str]:
    """Return ({team: fair_prob}, source_timestamp) from local fair_odds CSV."""
    out: dict[str, float] = {}
    ts = "unknown"
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            team = row["outcome"]
            out[FAIR_TEAM_ALIASES.get(team, team)] = float(row["fair_prob"])
            ts = row.get("computed_at", ts)
    return out, ts


def _resolve_db_url(explicit: str | None) -> str:
    """Resolve DATABASE_URL: explicit arg → env var → .do/app.yaml fallback."""
    import os
    import re
    if explicit:
        return explicit
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    do_yaml = PROJECT_ROOT / ".do" / "app.yaml"
    if do_yaml.exists():
        m = re.search(r"DATABASE_URL.*?value:\s*(postgresql://\S+)", do_yaml.read_text(), re.DOTALL)
        if m:
            return m.group(1)
    raise SystemExit(
        "ERROR: no DATABASE_URL — set env var, pass --db-url, or check .do/app.yaml")


def load_fair_api(api_base: str, event: str = "world_cup_2026",
                  market: str = "winner") -> tuple[dict[str, float], str]:
    """Return ({team: fair_prob}, source_timestamp) from deployed FairLine API.

    Hits {api_base}/events/{event}/fair-odds?market={market}. Recommended over
    direct DB because the API service is inside DO's network and bypasses the
    Postgres trusted-sources allowlist that blocks local connections.
    """
    import requests
    url = f"{api_base.rstrip('/')}/events/{event}/fair-odds"
    resp = requests.get(url, params={"market": market}, timeout=30)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise SystemExit(f"ERROR: API returned 0 rows for {event}/{market}")
    out: dict[str, float] = {}
    for r in rows:
        team = r["outcome"]
        out[FAIR_TEAM_ALIASES.get(team, team)] = float(r["fair_prob"])
    return out, str(rows[0].get("computed_at", "live"))


def load_fair_db(event: str = "world_cup_2026", market: str = "winner",
                 db_url: str | None = None) -> tuple[dict[str, float], str]:
    """Return ({team: fair_prob}, source_timestamp) from live Supabase.

    Connects directly via DATABASE_URL — bypasses local `src/db.py` because
    main has updated `latest_fair_odds()` to use `get_read_engine()` which
    doesn't exist on this branch.
    """
    from sqlalchemy import create_engine, text

    url = _resolve_db_url(db_url)
    engine = create_engine(url, pool_pre_ping=True)
    q = text("""
        SELECT outcome, fair_prob, computed_at
        FROM fair_odds
        WHERE event = :event AND market = :market
          AND computed_at = (
              SELECT MAX(computed_at) FROM fair_odds
              WHERE event = :event AND market = :market
          )
        ORDER BY fair_prob DESC
    """)
    with engine.connect() as conn:
        rows = conn.execute(q, {"event": event, "market": market}).mappings().all()
    if not rows:
        raise SystemExit(f"ERROR: no fair_odds rows for event={event} market={market}")
    out: dict[str, float] = {}
    for r in rows:
        team = r["outcome"]
        out[FAIR_TEAM_ALIASES.get(team, team)] = float(r["fair_prob"])
    return out, str(rows[0]["computed_at"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="WC2026 MC champion-prob vs FairLine fair_prob comparison.")
    parser.add_argument(
        "--mc-csv", type=Path,
        default=PROJECT_ROOT / "data" / "mc_simu" / "mc_wc2026_v1_100k_seed42.csv",
        help="MC output CSV (must have market_type, team, mc_fair_prob columns).")
    parser.add_argument(
        "--fair-csv", type=Path,
        default=PROJECT_ROOT / "data" / "fair_odds" / "world_cup_2026" / "winner" / "latest.csv",
        help="Local fair_odds snapshot (used if --use-db not set).")
    parser.add_argument(
        "--use-db", action="store_true",
        help="Pull fresh fair_odds from Supabase directly (env DATABASE_URL or .do/app.yaml). "
             "May fail with timeout if your IP is not allowlisted on DO Postgres.")
    parser.add_argument(
        "--db-url", default=None,
        help="Explicit DATABASE_URL; overrides env var and .do/app.yaml.")
    parser.add_argument(
        "--use-api", action="store_true",
        help="Pull fresh fair_odds from deployed FairLine API (no DB allowlist needed).")
    parser.add_argument(
        "--api-url", default="https://seal-app-yatxw.ondigitalocean.app/api",
        help="FairLine API base URL.")
    parser.add_argument(
        "--out-csv", type=Path,
        default=PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines" / "wc2026_vs_fair.csv")
    args = parser.parse_args(argv)

    if not args.mc_csv.exists():
        raise SystemExit(f"ERROR: MC output not found: {args.mc_csv}\n"
                         f"Run `python -m mc_simu.run_mc_simu` first.")
    mc = load_mc_champions(args.mc_csv)

    if args.use_api:
        fair, fair_ts = load_fair_api(args.api_url)
        fair_source = f"FairLine API ({args.api_url})"
    elif args.use_db:
        fair, fair_ts = load_fair_db(db_url=args.db_url)
        fair_source = "Supabase live (direct DB)"
    else:
        if not args.fair_csv.exists():
            raise SystemExit(f"ERROR: fair_odds CSV not found: {args.fair_csv}")
        fair, fair_ts = load_fair_local(args.fair_csv)
        fair_source = f"local CSV ({args.fair_csv.name})"

    # Build rows — one per team in union(mc_keys, fair_keys)
    teams = set(mc.keys()) | set(fair.keys())
    rows: list[dict] = []
    for team in teams:
        mc_p = mc.get(team, 0.0)
        fair_p = fair.get(team, 0.0)
        edge = mc_p - fair_p
        rows.append({
            "team": team,
            "mc_prob":     round(mc_p,   5),
            "fair_prob":   round(fair_p, 5),
            "edge":        round(edge,   5),
            "verdict":     "OVERSTATE" if edge > 0.01 else "UNDERSTATE" if edge < -0.01 else "MATCH",
            "mc_in_csv":   team in mc,
            "fair_in_csv": team in fair,
        })

    rows.sort(key=lambda r: -abs(r["edge"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {args.out_csv}")

    banner(f"WC2026 — MC champion-prob vs fair_prob ({fair_source}, ts={fair_ts})")
    print(f"MC rows: {len(mc)} | Fair rows: {len(fair)} | Union: {len(teams)}")
    print(f"Sum(mc) = {sum(mc.values()):.4f}   Sum(fair) = {sum(fair.values()):.4f}")
    print()

    overstates = [r for r in rows if r["edge"] > 0 and r["fair_in_csv"]][:10]
    understates = [r for r in rows if r["edge"] < 0 and r["fair_in_csv"]][:10]

    banner("Top 10 OVERSTATES (MC > fair) — teams MC thinks are MORE likely than market")
    print(f"{'Team':<22} {'MC':>8} {'Fair':>8} {'Edge':>9}")
    for r in overstates:
        print(f"{r['team']:<22} {r['mc_prob']*100:>7.2f}% {r['fair_prob']*100:>7.2f}% "
              f"{r['edge']*100:>+8.2f}pp")

    print()
    banner("Top 10 UNDERSTATES (MC < fair) — teams MC thinks are LESS likely than market")
    print(f"{'Team':<22} {'MC':>8} {'Fair':>8} {'Edge':>9}")
    for r in understates:
        print(f"{r['team']:<22} {r['mc_prob']*100:>7.2f}% {r['fair_prob']*100:>7.2f}% "
              f"{r['edge']*100:>+8.2f}pp")

    missing_in_fair = [r["team"] for r in rows if r["mc_in_csv"] and not r["fair_in_csv"]]
    missing_in_mc = [r["team"] for r in rows if r["fair_in_csv"] and not r["mc_in_csv"]]
    if missing_in_fair:
        print()
        banner(f"Teams in MC but NOT in fair_odds ({len(missing_in_fair)})")
        print(", ".join(sorted(missing_in_fair)))
    if missing_in_mc:
        print()
        banner(f"Teams in fair_odds but NOT in MC ({len(missing_in_mc)}) — check team-name aliases")
        print(", ".join(sorted(missing_in_mc)))

    return 0


__all__ = ["load_mc_champions", "load_fair_local", "load_fair_db", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
