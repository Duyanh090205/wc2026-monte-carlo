"""WC2026 model predictions vs Polymarket prices — preliminary edge table.

Phase 3 §3.10 supplement. Pulls live Polymarket WC2026 mid prices from
data/live_prices/prices.json (FairLine project snapshot) and joins with all
3 model predictions to show side-by-side edges per team.

Output:
- data/mc_simu/phase3_baselines/wc2026_vs_polymarket.csv
- Markdown table to stdout

CAVEAT: Polymarket mids sum to ~0.66 (not 1.0) because only ~20 most-liquid
team markets are listed. Untraded weak teams effectively assumed near-zero.
Edges shown here are RAW mid-vs-model differences, NOT trade-sizing signals
(no liquidity / fee adjustment). For trade decisions consult Phase 4 calibrated
output via paper_trader.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mc_simu._common import banner  # noqa: E402


MODELS: list[tuple[str, str]] = [
    ("self",       "self_elo"),
    ("eloratings", "eloratings"),
    ("baseline",   "baseline"),
]

# Polymarket uses inconsistent capitalization for some teams ("Usa" not "USA").
# Map Polymarket spelling → our adapter canonical spelling.
POLYMARKET_TEAM_ALIASES: dict[str, str] = {
    "Usa": "United States",
    "South Africa": "South Africa",  # explicit
    # Polymarket may also use Korea Republic etc.; add as encountered
}


def load_polymarket_wc2026(prices_json: Path) -> dict[str, dict[str, float]]:
    """Return {team: {'bid': , 'ask': , 'mid': }} for WC2026 market only."""
    data = json.loads(prices_json.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for entry in data.get("polymarket", {}).values():
        if entry.get("event") != "world_cup_2026":
            continue
        raw_team = entry["team"]
        team = POLYMARKET_TEAM_ALIASES.get(raw_team, raw_team)
        out[team] = {
            "bid": float(entry.get("bid", 0)),
            "ask": float(entry.get("ask", 0)),
            "mid": float(entry.get("mid", 0)),
        }
    return out


def load_model_probs(csv_path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["market_type"] == "champion":
                out[row["team"]] = float(row["mc_fair_prob"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WC2026 model vs Polymarket comparison.")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prices-json", type=Path,
                        default=PROJECT_ROOT / "data" / "live_prices" / "prices.json")
    parser.add_argument("--baselines-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines")
    parser.add_argument("--out-csv", type=Path,
                        default=PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines" / "wc2026_vs_polymarket.csv")
    args = parser.parse_args(argv)

    pm = load_polymarket_wc2026(args.prices_json)
    if not pm:
        print(f"WARNING: no WC2026 markets in {args.prices_json}", file=sys.stderr)
        return 1

    # Load all 3 model predictions
    model_probs: dict[str, dict[str, float]] = {}
    for tag, label in MODELS:
        csv_path = args.baselines_dir / f"mc_wc2026_{tag}_n{args.n}_seed{args.seed}.csv"
        if not csv_path.exists():
            print(f"WARNING: missing {csv_path}", file=sys.stderr)
            continue
        model_probs[label] = load_model_probs(csv_path)

    # Union of teams across PM + 3 models
    teams = set(pm.keys())
    for probs in model_probs.values():
        teams.update(probs.keys())

    # Build rows — one per team
    rows: list[dict] = []
    for team in teams:
        pm_entry = pm.get(team, {})
        row = {
            "team": team,
            "pm_mid": round(pm_entry.get("mid", 0.0), 4),
            "pm_bid": round(pm_entry.get("bid", 0.0), 4),
            "pm_ask": round(pm_entry.get("ask", 0.0), 4),
        }
        for _tag, label in MODELS:
            p = model_probs.get(label, {}).get(team, 0.0)
            row[f"{label}_prob"] = round(p, 4)
            row[f"{label}_edge_vs_pm"] = round(p - pm_entry.get("mid", 0.0), 4)
        rows.append(row)

    # Sort by average model prob descending
    rows.sort(key=lambda r: -sum(r[f"{label}_prob"] for _t, label in MODELS) / len(MODELS))

    # Write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {args.out_csv}")

    # Markdown table — top 15 by avg model prob
    banner("WC2026 — model predictions vs Polymarket mid (top 15 by avg)")
    print(f"{'Team':<22} {'PM mid':>8} {'self':>8} {'elor':>8} {'baseline':>10}  {'edge_self':>10} {'edge_elor':>10}")
    print("-" * 100)
    for r in rows[:15]:
        print(
            f"{r['team']:<22} "
            f"{r['pm_mid']*100:>7.2f}% "
            f"{r['self_elo_prob']*100:>7.2f}% "
            f"{r['eloratings_prob']*100:>7.2f}% "
            f"{r['baseline_prob']*100:>9.2f}%  "
            f"{r['self_elo_edge_vs_pm']*100:>+9.2f}% "
            f"{r['eloratings_edge_vs_pm']*100:>+9.2f}%"
        )

    # Notable edges (≥3pp) sorted by abs edge
    print()
    banner("Notable edges (|edge| ≥ 3pp) — eloratings model")
    edges = [(r["team"], r["pm_mid"], r["eloratings_prob"], r["eloratings_edge_vs_pm"])
             for r in rows if abs(r["eloratings_edge_vs_pm"]) >= 0.03]
    edges.sort(key=lambda x: -abs(x[3]))
    print(f"{'Team':<22} {'PM mid':>8} {'eloratings':>11} {'edge':>10}")
    for team, pm_m, p, edge in edges:
        sign = "BUY " if edge > 0 else "FADE"
        print(f"{team:<22} {pm_m*100:>7.2f}% {p*100:>10.2f}% {edge*100:>+9.2f}%  ({sign})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
