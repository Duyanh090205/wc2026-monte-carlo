"""Phase 3 MC harness orchestrator. Default invocation runs WC2026 MC simulation.

Usage:
    python -m mc_simu.run_mc_simu                          # 100k iterations, seed=42
    python -m mc_simu.run_mc_simu --n 10000 --seed 7       # custom
    python -m mc_simu.run_mc_simu --output mc_v1.csv       # write CSV

Outputs:
    Console: top-10 champion probabilities + per-group winner table
    CSV (if --output): one row per (market, team, mc_fair_prob, mc_se, n_iterations)

This file is the Phase 3 endpoint. Phase 4 (LOTO-CV + θ tuning) will wrap this
in a calibration loop; the v1 CSV-only path (per §0.6) is plumbed here.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from mc_simu._common import banner  # noqa: E402
from mc_simu.tournaments.wc2026 import (  # noqa: E402
    GROUP_LETTERS,
    load_wc2026_bundle,
    run_monte_carlo,
)


def load_latest_ratings(elo_history_csv: Path) -> dict[str, float]:
    """Return {team: latest rating_after} from elo_history.csv."""
    elo_df = pd.read_csv(elo_history_csv)
    elo_df["date"] = pd.to_datetime(elo_df["date"])
    elo_df = elo_df.sort_values("date")
    return elo_df.groupby("team").tail(1).set_index("team")["rating_after"].to_dict()


def write_csv(result: dict, output_path: Path, n_iterations: int) -> None:
    """One row per (market_type, market_key, team, mc_fair_prob, mc_se, n_iterations).

    market_type ∈ {"champion", "group_winner"}.
    market_key  = "" for champion, group letter for group_winner.
    """
    rows: list[dict] = []
    for team, stats in result["champion"].items():
        rows.append({
            "market_type": "champion",
            "market_key": "",
            "team": team,
            "mc_fair_prob": stats["mc_fair_prob"],
            "mc_se": stats["mc_se"],
            "n_iterations": n_iterations,
        })
    for g in GROUP_LETTERS:
        for team, stats in result["group_winners"][g].items():
            rows.append({
                "market_type": "group_winner",
                "market_key": g,
                "team": team,
                "mc_fair_prob": stats["mc_fair_prob"],
                "mc_se": stats["mc_se"],
                "n_iterations": n_iterations,
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


def print_summary(result: dict, n_iterations: int) -> None:
    """Console table of top-10 champion probs + per-group top-2."""
    banner("CHAMPION — top 10")
    champ_sorted = sorted(
        result["champion"].items(), key=lambda kv: -kv[1]["mc_fair_prob"]
    )[:10]
    print(f"{'Team':<28} {'P(champion)':>12}   {'±SE':>8}")
    for team, stats in champ_sorted:
        print(f"{team:<28} {stats['mc_fair_prob']:>12.4f}   {stats['mc_se']:>8.4f}")

    banner("GROUP WINNERS")
    for g in GROUP_LETTERS:
        winners = result["group_winners"][g]
        sorted_w = sorted(winners.items(), key=lambda kv: -kv[1]["mc_fair_prob"])[:2]
        line = f"Group {g}: "
        line += " | ".join(f"{t} {s['mc_fair_prob']:.3f}" for t, s in sorted_w)
        print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WC2026 Monte Carlo tournament simulator.")
    parser.add_argument("--n", type=int, default=100_000, help="number of MC iterations")
    parser.add_argument("--seed", type=int, default=42, help="rng seed (Gate 3.C reproducibility)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional CSV output path (e.g., data/mc_simu/mc_fair_odds_v1.csv)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-iteration progress bar",
    )
    args = parser.parse_args(argv)

    elo_history = PROJECT_ROOT / "data" / "mc_simu" / "elo_history.csv"
    ratings = load_latest_ratings(elo_history)
    bundle = load_wc2026_bundle(ratings)

    banner(f"WC2026 MC SIMULATION — n={args.n:,} seed={args.seed}")
    t0 = time.time()
    result = run_monte_carlo(
        bundle,
        n_iterations=args.n,
        seed=args.seed,
        progress=not args.quiet,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({1000 * elapsed / args.n:.3f}ms / iter)")

    print_summary(result, args.n)

    if args.output is not None:
        write_csv(result, args.output, args.n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
