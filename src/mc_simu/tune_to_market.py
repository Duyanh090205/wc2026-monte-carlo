"""Tune MC simulator toward market consensus — Phase 4 implementation.

Pipeline:
    1. Pull 3 sources directly (bypass FairLine scraper bugs):
       - Polymarket gamma API (groupItemTitle, not slug)
       - Kalshi public API (ask fallback when bid=0/last=None)
       - Fairline /fair-odds (devigged sportsbook consensus)
    2. Normalize each, build consensus = median per team
    3. Grid: D × diag (α, β fixed — sweep confirmed near-noise)
    4. Each cell: run MC, compute JSD vs consensus + per-bin calibration
    5. Pick θ* = arg min JSD

Output:
    data/mc_simu/phase3_baselines/tune_to_market.csv (gitignored)
    Per-cell scorecard + best θ* + calibration check at θ*
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mc_simu import single_game  # noqa: E402
from mc_simu._common import banner  # noqa: E402
from mc_simu.run_phase3_baselines import load_ratings_for_source  # noqa: E402
from mc_simu.single_game import ModelParams  # noqa: E402
from mc_simu.tournaments.wc2026 import load_wc2026_bundle, run_monte_carlo  # noqa: E402


# ── Team-name normalization ───────────────────────────────────────────────────

# Maps every API spelling → wc2026_groups.json canonical.
TEAM_ALIASES: dict[str, str] = {
    # Polymarket gamma
    "Bosnia-Herzegovina":   "Bosnia and Herzegovina",
    "Congo DR":             "DR Congo",
    "Curaçao":              "Curacao",
    "Turkiye":              "Turkey",
    "Türkiye":              "Turkey",
    "USA":                  "United States",
    "Usa":                  "United States",
    # Kalshi
    "Congo DR":             "DR Congo",
    # Fairline /fair-odds
    "Korea Republic":       "South Korea",
    "Cabo Verde":           "Cape Verde",
    "Czech Republic":       "Czechia",
    "Cote d'Ivoire":        "Ivory Coast",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    # Drop placeholders
    "Team Z":               "",
    "Other":                "",
}


def _norm(team: str) -> str:
    n = TEAM_ALIASES.get(team, team)
    return n


# ── Source pulls (direct, bypass FairLine scrapers) ───────────────────────────


def fetch_polymarket(timeout: int = 30) -> dict[str, float]:
    """Pull 48-team distribution from Polymarket gamma API. Uses groupItemTitle
    (NOT slug) so Panama is correctly named (FairLine scraper bug bypass).
    """
    import requests
    r = requests.get(
        "https://gamma-api.polymarket.com/events/slug/2026-fifa-world-cup-winner-595",
        timeout=timeout,
    )
    r.raise_for_status()
    out: dict[str, float] = {}
    for m in r.json().get("markets", []):
        if not m.get("active") or m.get("closed"):
            continue
        team = _norm(m.get("groupItemTitle") or "")
        if not team:
            continue
        px = m.get("lastTradePrice")
        if px is None or not (0 < float(px) < 1):
            continue
        out[team] = float(px)
    return out


def fetch_kalshi(timeout: int = 30) -> dict[str, float]:
    """Pull 48-team distribution from Kalshi public API.

    Price priority: (bid+ask)/2 → last → ask (for ask-only longshots, FairLine
    scraper drops these — we pick them up).
    """
    import requests
    r = requests.get(
        "https://api.elections.kalshi.com/trade-api/v2/markets",
        params={"event_ticker": "KXMENWORLDCUP-26", "limit": 100, "status": "open"},
        timeout=timeout,
    )
    r.raise_for_status()
    out: dict[str, float] = {}
    for m in r.json().get("markets", []):
        if m.get("status") not in ("open", "active"):
            continue
        team = _norm((m.get("yes_sub_title") or "").strip())
        if not team:
            continue
        bid = m.get("yes_bid_dollars")
        ask = m.get("yes_ask_dollars")
        last = m.get("last_price_dollars") or m.get("previous_price_dollars")
        bid_f = float(bid) if bid is not None else 0.0
        ask_f = float(ask) if ask is not None else 0.0
        last_f = float(last) if last is not None else None
        if bid_f > 0 and ask_f > 0:
            px = (bid_f + ask_f) / 2
        elif last_f is not None and 0 < last_f < 1:
            px = last_f
        elif ask_f > 0:
            px = ask_f
        else:
            continue
        if 0 < px < 1:
            out[team] = px
    # Drop dead market Poland (eliminated in qualifying)
    out.pop("Poland", None)
    return out


def fetch_fairline(timeout: int = 30,
                   api_base: str = "https://seal-app-yatxw.ondigitalocean.app/api"
                   ) -> dict[str, float]:
    """Pull devigged sportsbook consensus from FairLine /fair-odds."""
    import requests
    r = requests.get(
        f"{api_base}/events/world_cup_2026/fair-odds",
        params={"market": "winner"}, timeout=timeout,
    )
    r.raise_for_status()
    out: dict[str, float] = {}
    for row in r.json():
        team = _norm(row.get("outcome") or row.get("team") or "")
        if not team:
            continue
        fp = row.get("fair_prob")
        if fp is not None and 0 < float(fp) < 1:
            out[team] = float(fp)
    return out


# ── Consensus + math ──────────────────────────────────────────────────────────


def normalize(d: dict[str, float]) -> dict[str, float]:
    total = sum(d.values())
    if total <= 0:
        return d
    return {k: v / total for k, v in d.items()}


def build_consensus(sources: dict[str, dict[str, float]],
                    teams_48: list[str]) -> dict[str, float]:
    """Median per team across sources; teams missing in a source are skipped for that team's
    median (per-team median over available sources, ≥1)."""
    consensus: dict[str, float] = {}
    for team in teams_48:
        vals = [s[team] for s in sources.values() if team in s]
        if vals:
            consensus[team] = float(np.median(vals))
        else:
            consensus[team] = 0.0
    return normalize(consensus)


def jsd(p: dict[str, float], q: dict[str, float], teams: list[str],
        eps: float = 1e-6) -> float:
    """Jensen-Shannon divergence over an ordered team set. Base-2 (bits), bounded [0, 1]."""
    P = np.array([max(p.get(t, 0.0), eps) for t in teams])
    Q = np.array([max(q.get(t, 0.0), eps) for t in teams])
    P /= P.sum()
    Q /= Q.sum()
    M = 0.5 * (P + Q)
    def kl(a, b):
        return float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def per_bin_calibration(mc: dict[str, float], market: dict[str, float],
                        teams: list[str]) -> dict[str, dict]:
    """5-bin calibration check by predicted (MC) prob.
    Per bin: mean MC vs mean market — both should be similar for calibrated model.
    """
    bins = [(0.001, 0.005), (0.005, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.0)]
    labels = ["lottery", "longshot", "darkhorse", "contender", "favorite"]
    out: dict[str, dict] = {}
    for (lo, hi), lab in zip(bins, labels):
        members = [t for t in teams if lo <= mc.get(t, 0.0) < hi]
        if not members:
            out[lab] = {"n": 0, "mc_mean": None, "mkt_mean": None, "err_pp": None}
            continue
        mc_mean = float(np.mean([mc[t] for t in members]))
        mkt_mean = float(np.mean([market.get(t, 0.0) for t in members]))
        out[lab] = {
            "n": len(members),
            "mc_mean": mc_mean,
            "mkt_mean": mkt_mean,
            "err_pp": (mc_mean - mkt_mean) * 100,
        }
    return out


# ── MC run + grid ─────────────────────────────────────────────────────────────


def _required_teams() -> list[str]:
    import json
    with (PROJECT_ROOT / "data" / "mc_simu" / "wc2026_groups.json").open() as f:
        data = json.load(f)
    return [t for g in data["groups"].values() for t in g]


def run_mc(ratings: dict[str, float], params: ModelParams, D: float,
           n_iter: int, seed: int) -> dict[str, float]:
    single_game.ELO_GOALS_DENOMINATOR = float(D)
    bundle = load_wc2026_bundle(ratings, params=params)
    result = run_monte_carlo(bundle, n_iterations=n_iter, seed=seed, progress=False)
    return {t: s["mc_fair_prob"] for t, s in result["champion"].items()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tune MC toward market consensus.")
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--elo-source", choices=["self", "eloratings"], default="eloratings")
    p.add_argument("--D-grid", default="1300,1400,1500,1600,1700,1800")
    p.add_argument("--diag-grid", default="0.05,0.10,0.15,0.20,0.25")
    p.add_argument("--out-csv", type=Path,
                   default=PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines" / "tune_to_market.csv")
    p.add_argument("--mv-blend-alpha", type=float, default=0.0,
                   help="If >0, blend Elo with Transfermarkt MV in z-score space "
                        "(0=pure Elo, 0.7=MV-heavy per Sam direction, 1=pure MV)")
    p.add_argument("--star-bonus-X", type=float, default=0.0,
                   help="If >0, add bonus Elo per identified 'star' player per team "
                        "(addresses Messi/Ronaldo MV deflation; X=15 calibrated 2026-05-27)")
    args = p.parse_args(argv)

    teams_48 = _required_teams()

    banner("STEP 1 — Pull 3 sources directly")
    print("  Polymarket gamma API ...", end="", flush=True)
    pm_raw = fetch_polymarket()
    pm = normalize({_norm(k): v for k, v in pm_raw.items() if _norm(k) in teams_48})
    print(f" {len(pm)} teams")

    print("  Kalshi public API   ...", end="", flush=True)
    kal_raw = fetch_kalshi()
    kal = normalize({_norm(k): v for k, v in kal_raw.items() if _norm(k) in teams_48})
    print(f" {len(kal)} teams")

    print("  Fairline /fair-odds ...", end="", flush=True)
    fl_raw = fetch_fairline()
    fl = normalize({_norm(k): v for k, v in fl_raw.items() if _norm(k) in teams_48})
    print(f" {len(fl)} teams")

    banner("STEP 2 — Build consensus (median per team, PM+Kalshi only)")
    # Sam direction 2026-05-27: do NOT use Fairline for selection metric — Fairline is also
    # our team's model (devigged sportsbook), self-referential. Use PM (primary) + Kalshi
    # (fallback) only. Fairline still fetched above as sanity-check log, not in consensus.
    consensus = build_consensus({"PM": pm, "Kalshi": kal}, teams_48)
    print(f"  consensus teams with data: {sum(1 for v in consensus.values() if v > 0)}/48")
    print(f"  Top 5 by consensus: " + ", ".join(
        f"{t}={consensus[t]*100:.2f}%" for t in sorted(consensus, key=lambda x: -consensus[x])[:5]))
    fl_diff = sum(abs(fl.get(t, 0) - consensus.get(t, 0)) for t in teams_48)
    print(f"  [sanity] L1(Fairline, PM+Kalshi consensus) = {fl_diff*100:+.2f}pp (informational, not used for tuning)")

    banner("STEP 3 — Load Elo ratings + grid sweep")
    data_dir = PROJECT_ROOT / "data" / "mc_simu"
    ratings, fallbacks = load_ratings_for_source(
        args.elo_source, pd.Timestamp.now(), data_dir, teams_48,
    )
    if fallbacks:
        print(f"  WARN: {len(fallbacks)} fallback teams: {fallbacks}")

    if args.mv_blend_alpha > 0.0:
        from mc_simu.mv_blend import load_mv_table, blend_elo_with_mv, blend_summary
        mv = load_mv_table()
        elo_subset = {t: ratings[t] for t in teams_48 if t in ratings}
        blended, mv_missing = blend_elo_with_mv(elo_subset, mv, alpha=args.mv_blend_alpha)
        if mv_missing:
            print(f"  MV missing for: {mv_missing} (kept original Elo)")
        print(f"  Blended Elo with MV (alpha={args.mv_blend_alpha}, z-score on log(MV))")
        print(blend_summary(elo_subset, mv, blended, top_n=10))
        # Apply blended back to ratings dict (only overrides 48 WC teams; alias rows untouched)
        ratings.update(blended)

    if args.star_bonus_X > 0.0:
        from mc_simu.star_presence import load_star_counts, apply_star_bonus, summarize_stars
        star_counts = load_star_counts()
        print(f"  Applying star bonus X={args.star_bonus_X} Elo per star")
        print("  " + summarize_stars(star_counts).replace("\n", "\n  "))
        ratings = apply_star_bonus(ratings, bonus_X=args.star_bonus_X, star_counts=star_counts)

    D_grid = [float(x) for x in args.D_grid.split(",")]
    diag_grid = [float(x) for x in args.diag_grid.split(",")]
    print(f"  Grid: {len(D_grid)} D × {len(diag_grid)} diag = {len(D_grid)*len(diag_grid)} cells")
    print(f"  D ∈ {D_grid}")
    print(f"  diag ∈ {diag_grid}")
    print(f"  α=0.27, β=0.09 (fixed per sweep — near-noise)")

    rows: list[dict] = []
    t_start = time.time()
    n_cells = len(D_grid) * len(diag_grid)
    cell_i = 0
    for D in D_grid:
        for diag in diag_grid:
            cell_i += 1
            t0 = time.time()
            params = ModelParams(alpha=0.27, beta=0.09, diagonal_inflation=diag)
            mc_raw = run_mc(ratings, params, D, args.n, args.seed)
            mc = normalize({t: mc_raw.get(t, 0.0) for t in teams_48})
            elapsed = time.time() - t0

            jsd_consensus = jsd(mc, consensus, teams_48)
            jsd_pm = jsd(mc, pm, teams_48)
            jsd_kalshi = jsd(mc, kal, teams_48)
            jsd_fline = jsd(mc, fl, teams_48)
            l1_consensus = sum(abs(mc[t] - consensus[t]) for t in teams_48)
            cal = per_bin_calibration(mc, consensus, teams_48)
            max_bin_err = max(abs(b["err_pp"]) for b in cal.values() if b["err_pp"] is not None)

            rows.append({
                "D": D, "diag": diag,
                "jsd_consensus": round(jsd_consensus, 6),
                "jsd_pm": round(jsd_pm, 6),
                "jsd_kalshi": round(jsd_kalshi, 6),
                "jsd_fline": round(jsd_fline, 6),
                "l1_consensus": round(l1_consensus, 5),
                "max_bin_err_pp": round(max_bin_err, 3),
                **{f"bin_{lab}_err": round(b["err_pp"], 3) if b["err_pp"] is not None else None
                   for lab, b in cal.items()},
                **{f"mc_{t}": round(mc[t], 5) for t in teams_48},
                "elapsed_sec": round(elapsed, 1),
            })
            print(f"  [{cell_i:>2}/{n_cells}] D={D:.0f} diag={diag:.2f} "
                  f"JSD={jsd_consensus:.4f} L1={l1_consensus*100:+.2f}pp "
                  f"max_bin={max_bin_err:+.2f}pp ({elapsed:.1f}s)")

    total = time.time() - t_start
    print(f"\nGrid done in {total:.1f}s ({total/60:.1f} min)")

    # Write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {args.out_csv}\n")

    # ── Results ─────────────────────────────────────────────────────────────
    banner("STEP 4 — Grid scorecard (JSD vs consensus)")
    print(f"{'D':>6}   " + "  ".join(f"diag={d:.2f}" for d in diag_grid))
    for D in D_grid:
        row_str = f"{D:>6.0f}   "
        for diag in diag_grid:
            r = next(r for r in rows if r["D"] == D and r["diag"] == diag)
            row_str += f"  {r['jsd_consensus']:.4f}"
        print(row_str)

    # Best cell
    best = min(rows, key=lambda r: r["jsd_consensus"])
    banner(f"BEST θ*: D={best['D']:.0f}, diag={best['diag']:.2f}")
    print(f"  JSD(consensus) = {best['jsd_consensus']:.4f}")
    print(f"  JSD(PM)        = {best['jsd_pm']:.4f}")
    print(f"  JSD(Kalshi)    = {best['jsd_kalshi']:.4f}")
    print(f"  JSD(Fairline)  = {best['jsd_fline']:.4f}")
    print(f"  L1 vs consens  = {best['l1_consensus']*100:+.2f}pp")
    print(f"  max bin err    = {best['max_bin_err_pp']:+.2f}pp")
    print()
    print("  Per-bin calibration (45° line check):")
    for lab in ["lottery", "longshot", "darkhorse", "contender", "favorite"]:
        v = best.get(f"bin_{lab}_err")
        if v is None:
            print(f"    {lab:<10}  (no teams in bin)")
        else:
            mark = "OK" if abs(v) < 1.0 else "WARN"
            print(f"    {lab:<10}  err={v:+.2f}pp  [{mark}]")

    # Top vs bottom 5 teams at θ*
    print()
    print("  Distribution sample at θ* (top 8 + tail 3):")
    print(f"    {'Team':<24} {'MC':>8} {'Consensus':>10} {'edge':>8}")
    sorted_teams = sorted(teams_48, key=lambda t: -consensus[t])
    for t in sorted_teams[:8] + sorted_teams[-3:]:
        mc_v = best[f"mc_{t}"]
        ce = consensus[t]
        edge = (mc_v - ce) * 100
        print(f"    {t:<24} {mc_v*100:>7.2f}% {ce*100:>9.2f}% {edge:>+7.2f}pp")

    # Source-disagreement diagnostic
    banner("Single-source JSD (sanity check — should not differ wildly)")
    print(f"  JSD(PM)       = {best['jsd_pm']:.4f}")
    print(f"  JSD(Kalshi)   = {best['jsd_kalshi']:.4f}")
    print(f"  JSD(Fairline) = {best['jsd_fline']:.4f}")
    spread = max(best['jsd_pm'], best['jsd_kalshi'], best['jsd_fline']) - \
             min(best['jsd_pm'], best['jsd_kalshi'], best['jsd_fline'])
    if spread > 0.005:
        print(f"  WARN: spread = {spread:.4f} — θ* fits one source much better than others")
    else:
        print(f"  OK: spread = {spread:.4f} — θ* equally close to all 3 sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
