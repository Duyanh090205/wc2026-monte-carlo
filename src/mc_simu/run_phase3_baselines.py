"""Run Elo-baseline MC simulations across all Phase 3 supported tournaments.

Writes one CSV per edition to data/mc_simu/phase3_baselines/:
    mc_<adapter>_<year>_<source>_n<n>_seed<seed>.csv

Coverage (12 editions per Phase 3 §3.7 + §3.8 cleanup):
  wc2026         — WC 2026 (12 groups + best-thirds)
  wc_8groups     — WC 2002, 2006, 2010, 2014, 2018, 2022 (6 editions)
  euro_24teams   — Euro 2020, 2024 (2 editions; 2016 deferred per decision #38)
  euro_16teams   — Euro 2004, 2008, 2012 (3 editions)

Two Elo sources supported (per Phase 1 §2.4 + Sam's framework note 2026-05-20):
  --elo-source self        — data/mc_simu/elo_history.csv         (336 teams, self-built)
  --elo-source eloratings  — data/mc_simu/elo_history_eloratings.csv (244 teams, eloratings.net)

Source-specific name aliases handle the differences (e.g., eloratings uses
"Czechia" + "Ireland" + "China"; self-Elo uses "Czech Republic" + "Republic of
Ireland" + "China PR"). Missing teams default to Elo=1500 with a warning.

Each CSV has the same schema as run_mc_simu.py:
    market_type, market_key, team, mc_fair_prob, mc_se, n_iterations

These baselines are the diffing target for Phase 4 PARAM_GRID tuning + the
head-to-head comparison between Elo sources.
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


# Source → CSV path resolver. Per Phase 1 §2.4 addendum, both files share
# identical schema: team, date, rating_after, opponent, tournament_type, K,
# mov, gs_self, gs_opp.
ELO_SOURCE_FILES = {
    "self":       "elo_history.csv",
    "eloratings": "elo_history_eloratings.csv",
}

# Name aliases per source — adapter canonical names → source file canonical names.
# Adapters store names following Kaggle international_results convention (WC2026
# groups.json + historical group memberships). Each Elo source has its own
# preferred spelling; resolve here at load time, BEFORE missing-team fallback,
# so legitimately-present teams aren't spuriously defaulted to 1500.
#
# Format: adapter_name → source_canonical_name
NAME_ALIASES_BY_SOURCE: dict[str, dict[str, str]] = {
    "self": {
        # Self-Elo derives from Kaggle international_results.csv. Adapter uses
        # "Czechia" (modern FIFA) and "Curacao" (ASCII) but the Kaggle file
        # keeps historical "Czech Republic" + cedilla "Curaçao".
        "Czechia": "Czech Republic",
        "Curacao": "Curaçao",
        # Serbia and Montenegro (WC 2006) dissolved during the tournament;
        # Kaggle uses "Serbia" post-split.
        "Serbia and Montenegro": "Serbia",
    },
    "eloratings": {
        # eloratings.net uses modern names where Kaggle uses old ones.
        "Czech Republic":      "Czechia",
        "Republic of Ireland": "Ireland",
        "China PR":            "China",
        "Curacao":             "Curaçao",
        # Same Serbia and Montenegro fallback.
        "Serbia and Montenegro": "Serbia",
    },
}

# Fallback rating when a team is genuinely missing from the source (rare —
# only small-nation friendlies absent from eloratings). Same default as Phase
# 1 initial Elo for unseen FIFA members.
FALLBACK_RATING = 1500.0


def latest_ratings_before(
    elo_history_csv: Path,
    cutoff: pd.Timestamp,
) -> dict[str, float]:
    """Latest Elo per team STRICTLY before `cutoff` date."""
    elo_df = pd.read_csv(elo_history_csv)
    elo_df["date"] = pd.to_datetime(elo_df["date"])
    snap = elo_df[elo_df["date"] < cutoff]
    return snap.sort_values("date").groupby("team").tail(1).set_index("team")["rating_after"].to_dict()


def load_ratings_for_source(
    source: str,
    cutoff: pd.Timestamp,
    data_dir: Path,
    required_teams: list[str] | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Load ratings + apply source-specific aliasing + fallback for missing.

    Returns (ratings_dict_keyed_by_adapter_canonical_name, list_of_fallback_teams).

    For source='eloratings', applies ELORATINGS_NAME_ALIASES on top of the
    loaded dict so adapter-canonical names resolve (e.g., "Czech Republic"
    looks up "Czechia"'s eloratings entry).

    Teams in required_teams that are still missing after aliasing get the
    FALLBACK_RATING and are returned in the warnings list.
    """
    csv_path = data_dir / ELO_SOURCE_FILES[source]
    raw = latest_ratings_before(csv_path, cutoff)

    # Apply source-specific aliases BEFORE fallback — so legitimately-present
    # teams under a different spelling aren't spuriously defaulted to 1500.
    for adapter_name, source_name in NAME_ALIASES_BY_SOURCE.get(source, {}).items():
        if adapter_name not in raw and source_name in raw:
            raw[adapter_name] = raw[source_name]

    fallbacks: list[str] = []
    if required_teams:
        for team in required_teams:
            if team not in raw:
                raw[team] = FALLBACK_RATING
                fallbacks.append(team)
    return raw, fallbacks


def _collect_required_teams(adapter_name: str, year: int | None) -> list[str]:
    """Return the list of team names this edition needs ratings for."""
    if adapter_name == "wc2026":
        import json
        groups_path = PROJECT_ROOT / "data" / "mc_simu" / "wc2026_groups.json"
        with groups_path.open() as f:
            data = json.load(f)
        return [t for g in data["groups"].values() for t in g]
    if adapter_name == "wc_8groups":
        from mc_simu.tournaments.wc_8groups import EDITIONS as E
        return [t for g in E[year]["groups"].values() for t in g]
    if adapter_name == "euro_24teams":
        from mc_simu.tournaments.euro_24teams import EDITIONS as E
        return [t for g in E[year]["groups"].values() for t in g]
    if adapter_name == "euro_16teams":
        from mc_simu.tournaments.euro_16teams import EDITIONS as E
        return [t for g in E[year]["groups"].values() for t in g]
    raise ValueError(f"Unknown adapter: {adapter_name}")


def _cutoff(year: int, tournament_type: str) -> pd.Timestamp:
    """Approximate tournament start date for Elo snapshot cutoff."""
    if tournament_type == "wc" and year == 2022:
        return pd.Timestamp("2022-11-20")  # Qatar winter WC
    if tournament_type == "euro" and year == 2020:
        return pd.Timestamp("2021-06-11")  # COVID-delayed Euro 2020 played in 2021
    # Most WC/Euro start early-to-mid June
    return pd.Timestamp(year=year, month=6, day=1)


def _write_csv(result: dict, n_iter: int, out_path: Path, group_letters: list[str]) -> None:
    rows = []
    for team, stats in result["champion"].items():
        rows.append({
            "market_type": "champion",
            "market_key": "",
            "team": team,
            "mc_fair_prob": stats["mc_fair_prob"],
            "mc_se": stats["mc_se"],
            "n_iterations": n_iter,
        })
    for g in group_letters:
        for team, stats in result["group_winners"][g].items():
            rows.append({
                "market_type": "group_winner",
                "market_key": g,
                "team": team,
                "mc_fair_prob": stats["mc_fair_prob"],
                "mc_se": stats["mc_se"],
                "n_iterations": n_iter,
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one_edition(
    adapter_name: str,
    year: int | None,
    n_iter: int,
    seed: int,
    out_dir: Path,
    data_dir: Path,
    source: str,
    predictor_kind: str = "elo",
) -> tuple[str, float, list[str]]:
    """Run MC for one edition, write CSV, return (csv_name, elapsed_sec, fallbacks).

    Args:
        predictor_kind: "elo" (default — uses Elo source via predict_match) or
                        "baseline" (40/40/20 ignorance predictor — ignores Elo).
                        When "baseline", the `source` arg is recorded in CSV name
                        only for provenance; ratings are loaded but not consulted
                        for predictions.
    """
    required = _collect_required_teams(adapter_name, year)

    # Build the predictor callable
    if predictor_kind == "baseline":
        from mc_simu.baseline import predict_match_40_40_20
        predictor = predict_match_40_40_20
        # Provenance suffix in filename: 'baseline' is its own "model family"
        file_tag = "baseline"
    elif predictor_kind == "elo":
        predictor = None  # bundle default = make_elo_predictor
        file_tag = source
    else:
        raise ValueError(f"Unknown predictor_kind: {predictor_kind}")

    if adapter_name == "wc2026":
        from mc_simu.tournaments.wc2026 import (
            GROUP_LETTERS,
            load_wc2026_bundle,
            run_monte_carlo,
        )
        ratings, fallbacks = load_ratings_for_source(source, pd.Timestamp.now(), data_dir, required)
        bundle = load_wc2026_bundle(ratings)
        out_path = out_dir / f"mc_wc2026_{file_tag}_n{n_iter}_seed{seed}.csv"

    elif adapter_name == "wc_8groups":
        from mc_simu.tournaments.wc_8groups import (
            GROUP_LETTERS,
            load_edition_bundle,
            run_monte_carlo,
        )
        ratings, fallbacks = load_ratings_for_source(source, _cutoff(year, "wc"), data_dir, required)
        bundle = load_edition_bundle(year, ratings)
        out_path = out_dir / f"mc_wc{year}_{file_tag}_n{n_iter}_seed{seed}.csv"

    elif adapter_name == "euro_24teams":
        from mc_simu.tournaments.euro_24teams import (
            GROUP_LETTERS,
            load_edition_bundle,
            run_monte_carlo,
        )
        ratings, fallbacks = load_ratings_for_source(source, _cutoff(year, "euro"), data_dir, required)
        bundle = load_edition_bundle(year, ratings)
        out_path = out_dir / f"mc_euro{year}_{file_tag}_n{n_iter}_seed{seed}.csv"

    elif adapter_name == "euro_16teams":
        from mc_simu.tournaments.euro_16teams import (
            GROUP_LETTERS,
            load_edition_bundle,
            run_monte_carlo,
        )
        ratings, fallbacks = load_ratings_for_source(source, _cutoff(year, "euro"), data_dir, required)
        bundle = load_edition_bundle(year, ratings)
        out_path = out_dir / f"mc_euro{year}_{file_tag}_n{n_iter}_seed{seed}.csv"

    else:
        raise ValueError(f"Unknown adapter: {adapter_name}")

    t0 = time.time()
    result = run_monte_carlo(
        bundle, n_iterations=n_iter, seed=seed, progress=False, predictor=predictor,
    )
    elapsed = time.time() - t0
    _write_csv(result, n_iter, out_path, GROUP_LETTERS)
    # Baseline doesn't need fallback warnings (it ignores Elo)
    return out_path.name, elapsed, fallbacks if predictor_kind == "elo" else []


# All 13 (adapter, year) pairs Phase 3 v1 supports
EDITIONS_TO_RUN: list[tuple[str, int | None]] = [
    ("wc2026",       None),
    ("wc_8groups",   2002),
    ("wc_8groups",   2006),
    ("wc_8groups",   2010),
    ("wc_8groups",   2014),
    ("wc_8groups",   2018),
    ("wc_8groups",   2022),
    ("euro_24teams", 2020),
    ("euro_24teams", 2024),
    ("euro_16teams", 2004),
    ("euro_16teams", 2008),
    ("euro_16teams", 2012),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Elo-baseline MC for all Phase 3 editions.")
    parser.add_argument("--n", type=int, default=10_000,
                        help="iterations per edition (default 10k — quick baseline; "
                             "use 100k for Phase 4-grade reference output)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines")
    parser.add_argument(
        "--elo-source",
        choices=sorted(ELO_SOURCE_FILES.keys()),
        default="self",
        help="Which Elo history file to use as ratings source (default: self).",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both 'self' and 'eloratings' sources back-to-back (24 CSVs total).",
    )
    parser.add_argument(
        "--predictor",
        choices=["elo", "baseline"],
        default="elo",
        help="Which predictor: 'elo' (uses --elo-source) or 'baseline' (40/40/20).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run self-Elo + eloratings + baseline back-to-back (36 CSVs total).",
    )
    args = parser.parse_args(argv)

    data_dir = PROJECT_ROOT / "data" / "mc_simu"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build run plan: list of (source, predictor_kind) pairs
    if args.all_models:
        run_plan = [("self", "elo"), ("eloratings", "elo"), ("self", "baseline")]
    elif args.both:
        run_plan = [("self", "elo"), ("eloratings", "elo")]
    else:
        run_plan = [(args.elo_source, args.predictor)]

    overall_summary: list[tuple[str, str, str, str, float, list[str]]] = []
    grand_t0 = time.time()

    for source, predictor_kind in run_plan:
        model_label = f"baseline" if predictor_kind == "baseline" else source
        banner(f"Phase 3 Baselines — model={model_label} × {len(EDITIONS_TO_RUN)} editions × n={args.n:,} iter")
        for adapter, year in EDITIONS_TO_RUN:
            label = f"{adapter}/{year if year else 'wc2026'}"
            print(f"  Running {label} ...", end="", flush=True)
            try:
                csv_name, elapsed, fallbacks = run_one_edition(
                    adapter, year, args.n, args.seed, out_dir, data_dir,
                    source, predictor_kind=predictor_kind,
                )
                tag = f" [{len(fallbacks)} fallback]" if fallbacks else ""
                print(f" {elapsed:.1f}s{tag} -> {csv_name}")
                overall_summary.append((model_label, source, label, csv_name, elapsed, fallbacks))
            except Exception as e:
                print(f" FAILED -- {type(e).__name__}: {e}")
                overall_summary.append((model_label, source, label, "FAILED", 0.0, []))

    grand_elapsed = time.time() - grand_t0
    banner(f"DONE -- {len(overall_summary)} runs in {grand_elapsed:.1f}s")
    for model, source, label, csv_name, elapsed, fallbacks in overall_summary:
        marker = "OK  " if csv_name != "FAILED" else "FAIL"
        fb_str = f"  fallbacks={fallbacks}" if fallbacks else ""
        print(f"  [{marker}] {model:<10} {label:<28} {elapsed:>6.1f}s  {csv_name}{fb_str}")

    return 0 if all(s[3] != "FAILED" for s in overall_summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
