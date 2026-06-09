"""Diagnostic 1-D parameter sweeps for WC2026 MC sim — Phase 4 prep.

Sweeps four predict_match knobs one at a time (others held at default) to
identify which parameter drives the Argentina/Spain over- and England/France
under-statement vs market consensus.

Knobs (per-knob value list, default highlighted):
    diagonal_inflation : {0.10, 0.15, 0.20*, 0.25, 0.30}
    D (ELO denominator): {1000, 1200, 1400*, 1600, 1800}
    alpha (own-country): {0.15, 0.20, 0.27*, 0.35, 0.45}
    beta  (confed HFA) : {0.00, 0.05, 0.09*, 0.15}

Sim setup: WC2026, n=10000, seed=42, ratings source = eloratings.net.

Diagnostic evaluation (5 markets × 48 teams):
    Pulls Betfair, Other SB, Fairline Model, Kalshi, Polymarket live from
    /odds-comparison and /prices on the deployed FairLine API. For each
    config, computes:
      - 5-key-team edge table (Arg/Fra/Eng/Spa/Bra vs Polymarket)
      - aggregate L1 distance (Σ|MC − src| over the common-team set per source)
        — this is the multi-source robustness check.
    Sportsbooks only quote ~24 favorites (anchor ≥0.5pp filter); Kalshi/PM
    quote all 35–44 teams. L1 uses each source's natural team coverage.

Outputs:
    data/mc_simu/phase3_baselines/param_sweep.csv (gitignored)
    Per-knob detail table + multi-source L1 summary to stdout
    Range-power recommendation block

Diagnostic only — market-distance is NOT a tuning signal (per MODEL_SPEC §4.4
+ memory mc-phase4-distribution-eval-target). LOTO-CV on historical W/D/L
outcomes remains the Phase 4 tuning target. This script confirms which knobs
*could* move the model toward market consensus IF we chose to tune for it.
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

from mc_simu import single_game  # noqa: E402  — for ELO_GOALS_DENOMINATOR override
from mc_simu._common import banner  # noqa: E402
from mc_simu.run_phase3_baselines import load_ratings_for_source  # noqa: E402
from mc_simu.single_game import ModelParams  # noqa: E402
from mc_simu.tournaments.wc2026 import load_wc2026_bundle, run_monte_carlo  # noqa: E402


KEY_TEAMS = ["Argentina", "France", "England", "Spain", "Brazil"]

SOURCES = ["Betfair", "Other SB", "Fairline Model", "Kalshi", "Polymarket"]

DEFAULTS = {
    "diagonal_inflation": 0.20,
    "D":                  1400.0,
    "alpha":              0.27,
    "beta":               0.09,
}

SWEEP_GRID: dict[str, list[float]] = {
    "diagonal_inflation": [0.10, 0.15, 0.20, 0.25, 0.30],
    "D":                  [1000.0, 1200.0, 1400.0, 1600.0, 1800.0],
    "alpha":              [0.15, 0.20, 0.27, 0.35, 0.45],
    "beta":               [0.00, 0.05, 0.09, 0.15],
}

# Maps API team names → wc2026_groups.json canonical names (MC convention).
# Polymarket/Kalshi /prices endpoint produces some lossy names (Curaao, Usa,
# Turkiye, Bosnia Herzegovina, Korea Republic, Cabo Verde). Normalize here
# before joining on MC output.
API_TEAM_ALIASES: dict[str, str] = {
    "USA":              "United States",
    "Usa":              "United States",
    "Korea Republic":   "South Korea",
    "Cabo Verde":       "Cape Verde",
    "Cote d'Ivoire":    "Ivory Coast",
    "Curaao":           "Curacao",
    "Curaçao":          "Curacao",
    "Turkiye":          "Turkey",
    "Türkiye":          "Turkey",
    "Bosnia & Herzegovina":  "Bosnia and Herzegovina",
    "Bosnia Herzegovina":    "Bosnia and Herzegovina",
    "Czech Republic":   "Czechia",
    "Congo Dr":         "DR Congo",
    "Team Z":           None,   # placeholder team — drop
}


def _load_all_source_baselines(api_url: str, event: str = "world_cup_2026"
                                ) -> dict[str, dict[str, float]]:
    """Pull all 5 sources from /odds-comparison + /prices, return {src: {team: prob}}.

    /odds-comparison covers ~24 teams with ≥0.5pp anchor — has sportsbook devig +
    Fairline + Kalshi + Polymarket. /prices fills Kalshi (~36) + Polymarket
    (~44) longshots that /odds-comparison filters out. /fair-odds backfills
    Fairline Model to all 48 teams.

    All team names are normalized via API_TEAM_ALIASES → MC canonical names.
    """
    import requests
    out: dict[str, dict[str, float]] = {s: {} for s in SOURCES}

    # 1. /odds-comparison — top-anchor table (sportsbooks devigged)
    r = requests.get(f"{api_url.rstrip('/')}/events/{event}/odds-comparison", timeout=30)
    r.raise_for_status()
    for row in r.json().get("rows", []):
        raw = row["team"]
        team = API_TEAM_ALIASES.get(raw, raw)
        if team is None:
            continue
        for src in SOURCES:
            val = row.get(src)
            if val is not None and val > 0:
                out[src][team] = float(val)

    # 2. /prices — fills Kalshi/Polymarket longshots (all teams)
    r = requests.get(f"{api_url.rstrip('/')}/events/{event}/prices", timeout=30)
    r.raise_for_status()
    for row in r.json():
        raw = str(row.get("team", ""))
        team = API_TEAM_ALIASES.get(raw, raw)
        if team is None or not team:
            continue
        plat = str(row.get("platform", ""))
        src = "Kalshi" if "kalshi" in plat else ("Polymarket" if "polymarket" in plat else None)
        if src is None:
            continue
        mid = row.get("mid")
        if mid is None or not (0 < mid < 1):
            continue
        # /prices overrides /odds-comparison only when missing — odds-comparison
        # already runs power_devig; raw mid is fine for longshots where devig
        # error is negligible (probs < 0.5%).
        out[src].setdefault(team, float(mid))

    # 3. /fair-odds — backfill Fairline Model to all 48 teams
    r = requests.get(f"{api_url.rstrip('/')}/events/{event}/fair-odds",
                     params={"market": "winner"}, timeout=30)
    r.raise_for_status()
    for row in r.json():
        raw = row.get("outcome") or row.get("team") or ""
        team = API_TEAM_ALIASES.get(raw, raw)
        if team is None or not team:
            continue
        fp = row.get("fair_prob")
        if fp is not None and 0 < float(fp) < 1:
            out["Fairline Model"].setdefault(team, float(fp))

    return out


POLYMARKET_BASELINE = {
    "Argentina": 0.0835,
    "France":    0.1795,
    "England":   0.1125,
    "Spain":     0.1715,
    "Brazil":    0.0885,
}


def _load_polymarket_baseline(multi_csv: Path) -> dict[str, float]:
    """Polymarket baseline for the 5 key teams (5-team view)."""
    if not multi_csv.exists():
        return dict(POLYMARKET_BASELINE)
    pm: dict[str, float] = {}
    with multi_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["team"] in KEY_TEAMS and row.get("polymarket"):
                pm[row["team"]] = float(row["polymarket"])
    for t in KEY_TEAMS:
        pm.setdefault(t, POLYMARKET_BASELINE[t])
    return pm


def _required_wc2026_teams() -> list[str]:
    import json
    with (PROJECT_ROOT / "data" / "mc_simu" / "wc2026_groups.json").open() as f:
        data = json.load(f)
    return [t for g in data["groups"].values() for t in g]


def _run_single_config(
    ratings: dict[str, float],
    params: ModelParams,
    D: float,
    n_iter: int,
    seed: int,
) -> dict[str, float]:
    """Run one MC config; return {team: champion_prob} for KEY_TEAMS only."""
    # Override module-level ELO_GOALS_DENOMINATOR — predict_match reads it via
    # global lookup at call time, so set BEFORE build_sim_context runs.
    single_game.ELO_GOALS_DENOMINATOR = float(D)
    bundle = load_wc2026_bundle(ratings, params=params)
    result = run_monte_carlo(bundle, n_iterations=n_iter, seed=seed, progress=False)
    champ = result["champion"]
    return {t: champ.get(t, {"mc_fair_prob": 0.0})["mc_fair_prob"] for t in KEY_TEAMS}


def _build_params(knob: str, value: float) -> tuple[ModelParams, float]:
    """Return (ModelParams, D) for a sweep cell — all knobs at default except `knob`."""
    d = float(DEFAULTS["D"])
    p = ModelParams(
        alpha=float(DEFAULTS["alpha"]),
        beta=float(DEFAULTS["beta"]),
        diagonal_inflation=float(DEFAULTS["diagonal_inflation"]),
    )
    if knob == "diagonal_inflation":
        p.diagonal_inflation = float(value)
    elif knob == "alpha":
        p.alpha = float(value)
    elif knob == "beta":
        p.beta = float(value)
    elif knob == "D":
        d = float(value)
    else:
        raise ValueError(f"Unknown knob: {knob}")
    return p, d


def _print_knob_table(knob: str, rows: list[dict], pm_baseline: dict[str, float]) -> None:
    banner(f"Knob = {knob}  (others fixed at default)")
    parts_prob = " ".join(f"{t[:6]:>8}" for t in KEY_TEAMS)
    parts_edge = " ".join(f"{('e_'+t[:4]):>8}" for t in KEY_TEAMS)
    print(f"{'value':>8} | {parts_prob} || {parts_edge}")
    print("-" * (10 + len(parts_prob) + 4 + len(parts_edge)))
    for r in rows:
        if r["knob"] != knob:
            continue
        probs = " ".join(f"{r[f'p_{t}']*100:>7.2f}%" for t in KEY_TEAMS)
        edges = " ".join(f"{(r[f'p_{t}']-pm_baseline[t])*100:>+7.2f}" for t in KEY_TEAMS)
        marker = "*" if abs(r["value"] - DEFAULTS[knob]) < 1e-9 else " "
        print(f"{r['value']:>8.3f}{marker}| {probs} || {edges}")
    print()


def _compute_slopes(rows: list[dict], pm_baseline: dict[str, float]) -> dict[str, dict[str, float]]:
    """For each knob, compute TOTAL Δedge across the swept range for each key team.

    Use range-total Δedge (not per-unit slope) because per-unit slopes aren't
    comparable across knobs with vastly different scales — D's 800-unit range
    vs diagonal_inflation's 0.20-unit range would make D's per-unit slope look
    ~4000× smaller despite D being the dominant lever.

    delta[team] = edge(knob_at_max) - edge(knob_at_min)
    Sign convention: positive delta means INCREASING the knob INCREASES that
    team's MC-vs-PM gap (i.e., overstates more or understates less).
    """
    out: dict[str, dict[str, float]] = {}
    for knob in SWEEP_GRID:
        knob_rows = sorted([r for r in rows if r["knob"] == knob], key=lambda r: r["value"])
        deltas: dict[str, float] = {}
        for team in KEY_TEAMS:
            edge_lo = knob_rows[0][f"p_{team}"] - pm_baseline[team]
            edge_hi = knob_rows[-1][f"p_{team}"] - pm_baseline[team]
            deltas[team] = edge_hi - edge_lo
        out[knob] = deltas
    return out


def _print_recommendation(deltas: dict[str, dict[str, float]]) -> None:
    banner("PHASE 4 TUNE RECOMMENDATION")
    # Score each knob by total Δedge it can move across its swept range.
    # Argentina (overstate, edge > 0): want delta < 0 (knob move that drops Arg edge)
    # France    (understate, edge < 0): want delta > 0 (knob move that raises Fra edge toward 0)
    # Sum power = |Δedge_Arg| + |Δedge_Fra|; this is range-normalized so knobs
    # with different scales (D's 800-unit span vs diag's 0.20) are comparable.
    print(f"{'Knob':<22} {'Δedge_Arg':>11} {'Δedge_Fra':>11} {'sum |Δ|':>11} {'one-dir fix?':>14}")
    print("-" * 72)
    for knob, dmap in deltas.items():
        d_arg = dmap["Argentina"]
        d_fra = dmap["France"]
        # "agrees" = the SAME direction of knob move helps both biases.
        # Helps Arg: d_arg<0 (knob ↑ closes Arg overstate). Helps Fra: d_fra>0.
        # If signs differ across teams → one knob direction helps both ⇒ "yes".
        if abs(d_arg) < 1e-6 and abs(d_fra) < 1e-6:
            agree = "—"
        elif (d_arg < 0 and d_fra > 0) or (d_arg > 0 and d_fra < 0):
            agree = "yes"
        else:
            agree = "no"
        print(f"{knob:<22} {d_arg*100:>+10.2f}pp {d_fra*100:>+10.2f}pp "
              f"{(abs(d_arg)+abs(d_fra))*100:>+10.2f}pp {agree:>14}")
    print()

    # Rank knobs by total Δedge power across the swept range
    ranked = sorted(
        deltas.items(),
        key=lambda kv: -(abs(kv[1]["Argentina"]) + abs(kv[1]["France"])),
    )
    top_knob, top_deltas = ranked[0]
    arg_dir = "DECREASE" if top_deltas["Argentina"] > 0 else "INCREASE"
    fra_dir = "INCREASE" if top_deltas["France"] > 0 else "DECREASE"
    same_direction = arg_dir == fra_dir
    print(f"Primary tune candidate: **{top_knob}** "
          f"(range power {(abs(top_deltas['Argentina'])+abs(top_deltas['France']))*100:.2f}pp)")
    print(f"  - To reduce Argentina overstatement: {arg_dir} {top_knob}")
    print(f"  - To reduce France understatement:   {fra_dir} {top_knob}")
    print(f"  - Single-direction fix?  "
          f"{'YES — same move helps both' if same_direction else 'NO — knob trades one bias for the other'}")
    print()
    second_knob, second_deltas = ranked[1]
    third_knob, _ = ranked[2]
    print(f"Secondary candidates (next two by range power): {second_knob} "
          f"({(abs(second_deltas['Argentina'])+abs(second_deltas['France']))*100:.2f}pp), {third_knob}")
    print()
    print("Phase 4 LOTO-CV grid prioritization:")
    print(f"  1. Tighten grid spacing on **{top_knob}** (primary leverage).")
    if same_direction:
        print(f"     Bias suggests the corrective move is to {arg_dir} {top_knob} from current default {DEFAULTS[top_knob]}.")
    else:
        print(f"     {top_knob} alone can't close both gaps simultaneously — pair with {second_knob}.")
    print(f"  2. Keep **{second_knob}** in the grid for joint-effect search.")
    print(f"  3. Fallback if primary knob doesn't close the gap on LOTO-CV-validated runs:")
    print(f"     the residual bias is structural (Elo source / CONMEBOL strength inflation in")
    print(f"     eloratings.net) rather than a free-parameter knob. Re-evaluate Elo source choice")
    print(f"     (self-Elo vs eloratings.net) before further parameter tuning.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="1-D parameter sensitivity sweep on WC2026 MC sim.")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--elo-source", choices=["self", "eloratings"], default="eloratings",
        help="Elo ratings source. Default eloratings.net per Phase 3 finding.")
    parser.add_argument(
        "--out-csv", type=Path,
        default=PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines" / "param_sweep.csv")
    parser.add_argument(
        "--multi-csv", type=Path,
        default=PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines" / "wc2026_vs_multi.csv",
        help="If present, refresh Polymarket baseline for 5 key teams from this file.")
    args = parser.parse_args(argv)

    data_dir = PROJECT_ROOT / "data" / "mc_simu"
    required_teams = _required_wc2026_teams()
    ratings, fallbacks = load_ratings_for_source(
        args.elo_source, pd.Timestamp.now(), data_dir, required_teams,
    )
    if fallbacks:
        print(f"WARN: {len(fallbacks)} teams defaulted to 1500: {fallbacks}")

    pm_baseline = _load_polymarket_baseline(args.multi_csv)

    # Total configs across sweeps (count default cells deduped via key).
    # Run every cell explicitly — keeps logic + CSV simple.
    configs: list[tuple[str, float]] = []
    for knob, values in SWEEP_GRID.items():
        for v in values:
            configs.append((knob, v))

    banner(f"PARAM SWEEP — n={args.n:,}, seed={args.seed}, source={args.elo_source}, "
           f"{len(configs)} configs")
    print(f"Polymarket baseline: " + ", ".join(f"{t}={pm_baseline[t]*100:.2f}%" for t in KEY_TEAMS))
    print()

    rows: list[dict] = []
    t_start = time.time()
    for i, (knob, value) in enumerate(configs, 1):
        params, D = _build_params(knob, value)
        t0 = time.time()
        probs = _run_single_config(ratings, params, D, args.n, args.seed)
        elapsed = time.time() - t0
        row = {
            "knob":  knob,
            "value": value,
            "alpha":              params.alpha,
            "beta":               params.beta,
            "diagonal_inflation": params.diagonal_inflation,
            "D":                  D,
            **{f"p_{t}": probs[t] for t in KEY_TEAMS},
            **{f"edge_vs_pm_{t}": probs[t] - pm_baseline[t] for t in KEY_TEAMS},
            "elapsed_sec": round(elapsed, 1),
        }
        rows.append(row)
        edges_summary = " ".join(
            f"{t[:3]}:{(probs[t]-pm_baseline[t])*100:+.1f}" for t in KEY_TEAMS
        )
        print(f"  [{i:>2}/{len(configs)}] {knob}={value:<6} ({elapsed:5.1f}s)  {edges_summary}")
    total = time.time() - t_start
    print(f"\nTotal sweep time: {total:.1f}s ({total/60:.1f} min)\n")

    # Write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        # Round prob/edge columns for human-readability
        for r in rows:
            r_out = {k: (round(v, 5) if isinstance(v, float) and k != "value" else v)
                     for k, v in r.items()}
            writer.writerow(r_out)
    print(f"Wrote: {args.out_csv}\n")

    # Per-knob tables
    for knob in SWEEP_GRID:
        _print_knob_table(knob, rows, pm_baseline)

    # Slopes + recommendation
    slopes = _compute_slopes(rows, pm_baseline)
    _print_recommendation(slopes)

    return 0


__all__ = ["main", "KEY_TEAMS", "DEFAULTS", "SWEEP_GRID", "POLYMARKET_BASELINE"]


if __name__ == "__main__":
    raise SystemExit(main())
