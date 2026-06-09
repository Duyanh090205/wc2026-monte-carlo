"""Distribution of all 48 WC2026 teams vs Fairline Model (devigged sportsbook consensus).

Per Sam's 2026-05-25 ask: "share distribution for all 48 teams vs our 'fair model'
... I want to see where your simulations overstate the probability and where they
understate. Then perhaps you can feed that information into your cursor/claude code
/antigravity to ask for their suggestion on how you can improve your single game model."

Source choice = Fairline Model (devigged Shin/Power consensus of sportsbooks). Full
48/48 coverage, no Polymarket "Team Z" bug, no Kalshi 14-missing bug.

Input: data/mc_simu/phase3_baselines/wc2026_vs_multi.csv (regen via wc2026_vs_multi.py).
Output to stdout: classified table + structural suggestions for the single-game model.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mc_simu._common import banner  # noqa: E402


THRESHOLD_PP = 0.005  # 0.5pp boundary for over/match/under classification


def _load_48_teams() -> list[str]:
    with (PROJECT_ROOT / "data" / "mc_simu" / "wc2026_groups.json").open() as f:
        data = json.load(f)
    return [t for g in data["groups"].values() for t in g]


def _load_comparison(csv_path: Path) -> list[dict]:
    out: list[dict] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mc = row.get("mc_prob") or ""
            fl = row.get("fairline_model") or ""
            if not mc or not fl:
                continue
            out.append({
                "team": row["team"],
                "mc_prob": float(mc),
                "fline": float(fl),
                "edge": float(mc) - float(fl),
            })
    return out


def main(argv: list[str] | None = None) -> int:
    csv_path = PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines" / "wc2026_vs_multi.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}\nRun: python -m mc_simu.wc2026_vs_multi")

    teams_48 = set(_load_48_teams())
    rows = [r for r in _load_comparison(csv_path) if r["team"] in teams_48]
    rows.sort(key=lambda r: -r["fline"])

    over = [r for r in rows if r["edge"] >= THRESHOLD_PP]
    under = [r for r in rows if r["edge"] <= -THRESHOLD_PP]
    match = [r for r in rows if abs(r["edge"]) < THRESHOLD_PP]

    banner(f"WC2026 — MC vs Fairline Model, all {len(rows)}/48 teams")
    print(f"{'Team':<24} {'MC':>8} {'FLine':>8} {'edge (pp)':>10} {'verdict':>12}")
    print("-" * 70)
    for r in rows:
        if r["edge"] >= THRESHOLD_PP:    v = "OVERSTATE"
        elif r["edge"] <= -THRESHOLD_PP: v = "UNDERSTATE"
        else:                            v = "MATCH"
        print(f"{r['team']:<24} {r['mc_prob']*100:>7.2f}% {r['fline']*100:>7.2f}% "
              f"{r['edge']*100:>+9.2f} {v:>12}")

    print()
    banner(f"OVERSTATE — MC > FLine by ≥0.5pp ({len(over)} teams) — MC thinks team is stronger than market")
    print(f"{'Team':<24} {'edge (pp)':>10}    confederation")
    for r in sorted(over, key=lambda r: -r["edge"]):
        print(f"  {r['team']:<22} {r['edge']*100:>+9.2f}")

    print()
    banner(f"UNDERSTATE — MC < FLine by ≥0.5pp ({len(under)} teams) — MC thinks team is weaker than market")
    print(f"{'Team':<24} {'edge (pp)':>10}    confederation")
    for r in sorted(under, key=lambda r: r["edge"]):
        print(f"  {r['team']:<22} {r['edge']*100:>+9.2f}")

    print()
    banner(f"MATCH — |edge| < 0.5pp ({len(match)} teams)")
    print(", ".join(r["team"] for r in match))

    # Mass-conservation check
    sum_over = sum(r["edge"] for r in over)
    sum_under = sum(r["edge"] for r in under)
    sum_match = sum(r["edge"] for r in match)
    print()
    banner("Mass conservation check (should sum to ~0 since both distributions normalize to 1)")
    print(f"  Σ overstate edges  = {sum_over*100:>+7.2f}pp")
    print(f"  Σ understate edges = {sum_under*100:>+7.2f}pp")
    print(f"  Σ match edges      = {sum_match*100:>+7.2f}pp")
    print(f"  total              = {(sum_over+sum_under+sum_match)*100:>+7.2f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
