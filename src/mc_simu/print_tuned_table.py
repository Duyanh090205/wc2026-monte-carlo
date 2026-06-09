"""Print Top/Tail comparison table for tuned MC θ* vs all 5 market sources.

Reads:
    data/mc_simu/phase3_baselines/tune_to_market.csv  (per-cell MC distribution)
    data/mc_simu/phase3_baselines/wc2026_vs_multi.csv (market columns)
Picks best cell by jsd_consensus, joins on team, prints screenshot-format table.
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
from mc_simu.tune_to_market import fetch_polymarket, fetch_kalshi, fetch_fairline, _norm  # noqa: E402


def _pct(x):
    return f"{x*100:>7.2f}%" if x is not None and x != "" else "   —   "


def _edge(x):
    return f"{x*100:>+7.2f}pp" if x is not None and x != "" else "   —   "


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tune-csv", type=Path,
                   default=PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines" / "tune_to_market.csv")
    p.add_argument("--multi-csv", type=Path,
                   default=PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines" / "wc2026_vs_multi.csv")
    p.add_argument("--metric", choices=["jsd_consensus", "jsd_pm", "jsd_fline"],
                   default="jsd_consensus", help="Which JSD column to minimize")
    args = p.parse_args(argv)

    # Load 48-team canonical list
    with (PROJECT_ROOT / "data" / "mc_simu" / "wc2026_groups.json").open() as f:
        teams_48 = [t for g in json.load(f)["groups"].values() for t in g]

    # Load tuned grid; pick best row
    with args.tune_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    best = min(rows, key=lambda r: float(r[args.metric]))
    D = float(best["D"])
    diag = float(best["diag"])
    print(f"θ* selected by min {args.metric}: D={D:.0f}, diag={diag:.2f}")
    print(f"  JSD(consensus)={float(best['jsd_consensus']):.4f}, "
          f"JSD(PM)={float(best['jsd_pm']):.4f}, JSD(Fairline)={float(best['jsd_fline']):.4f}")
    print(f"  L1(consensus)={float(best['l1_consensus'])*100:+.2f}pp")
    print()

    mc_tuned = {t: float(best[f"mc_{t}"]) for t in teams_48}

    # PM/Kalshi/Fairline-fair-odds: pull directly from public APIs (bypass FairLine
    # scraper bugs that drop Panama / 14 Kalshi longshots).
    print("Pulling PM / Kalshi / Fairline directly from public APIs ...")
    pm = {_norm(k): v for k, v in fetch_polymarket().items()}
    kal = {_norm(k): v for k, v in fetch_kalshi().items()}
    fl = {_norm(k): v for k, v in fetch_fairline().items()}
    print(f"  PM={len(pm)} Kalshi={len(kal)} Fairline={len(fl)}")

    # Betfair + Other SB only available via FairLine /odds-comparison (devigged from raw
    # sportsbook lines). They genuinely only quote ~24 teams — sportsbook market reality.
    with args.multi_csv.open(encoding="utf-8") as f:
        multi_rows = {r["team"]: r for r in csv.DictReader(f)}

    rows_out: list[dict] = []
    for team in teams_48:
        m = multi_rows.get(team, {})
        def get_multi(k):
            v = m.get(k, "")
            return float(v) if v not in ("", None) else None
        rows_out.append({
            "team":     team,
            "mc":       mc_tuned[team],
            "fline":    fl.get(team) or get_multi("fairline_model"),
            "betfair":  get_multi("betfair"),
            "other_sb": get_multi("other_sb"),
            "kalshi":   kal.get(team),
            "polym":    pm.get(team),
        })
        rows_out[-1]["edge_fline"] = (
            rows_out[-1]["mc"] - rows_out[-1]["fline"] if rows_out[-1]["fline"] is not None else None
        )
        rows_out[-1]["edge_poly"] = (
            rows_out[-1]["mc"] - rows_out[-1]["polym"] if rows_out[-1]["polym"] is not None else None
        )

    # Split: Top = teams with Polymarket OR Kalshi non-trivial; Tail = otherwise
    # Match the screenshot's behavior: Top = teams with Polym AND Kalshi (or just Polym if no kalshi);
    # Tail = teams where Polym + Kalshi missing or both at noise floor (sportsbook columns blank).
    def in_top(r):
        # In the original screenshot, "Top" = teams that passed the /odds-comparison anchor cut
        # which roughly equals teams with Betfair + Other SB data filled.
        return r["betfair"] is not None and r["other_sb"] is not None

    top = sorted([r for r in rows_out if in_top(r)],
                 key=lambda r: -(r["polym"] or r["kalshi"] or 0))
    tail = sorted([r for r in rows_out if not in_top(r)], key=lambda r: -r["mc"])

    banner(f"WC2026 — TUNED MC (D={D:.0f}, diag={diag:.2f}) vs 5 markets")
    print(f"{'Team':<24} | {'MC':>7} | {'FLine':>7} | {'Betfair':>7} | {'OtherSB':>7} | "
          f"{'Kalshi':>7} | {'Polym':>7} | {'edgeFLine':>10} | {'edgePoly':>10} |")
    print("-" * 120)
    for r in top:
        print(f"{r['team']:<24} | "
              f"{_pct(r['mc'])} | "
              f"{_pct(r['fline'])} | "
              f"{_pct(r['betfair'])} | "
              f"{_pct(r['other_sb'])} | "
              f"{_pct(r['kalshi'])} | "
              f"{_pct(r['polym'])} | "
              f"{_edge(r['edge_fline'])} | "
              f"{_edge(r['edge_poly'])} |")

    print()
    banner("Tail (teams in MC but no Betfair/OtherSB sportsbook quote)")
    print(f"{'Team':<24} | {'MC':>7} | {'FLine':>7} | {'Kalshi':>7} | {'Polym':>7} | "
          f"{'edgeFLine':>10} | {'edgePoly':>10} |")
    print("-" * 95)
    for r in tail:
        print(f"{r['team']:<24} | "
              f"{_pct(r['mc'])} | "
              f"{_pct(r['fline'])} | "
              f"{_pct(r['kalshi'])} | "
              f"{_pct(r['polym'])} | "
              f"{_edge(r['edge_fline'])} | "
              f"{_edge(r['edge_poly'])} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
