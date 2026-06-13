"""WC2026 MC distribution vs ALL available models from FairLine API.

Extends `wc2026_vs_fair.py` from 1 source (Fairline Model) to all 5 active sources:
Betfair, Other SB, Fairline Model, Kalshi, Polymarket (Pinnacle currently has_data=0).

Hits two endpoints from the deployed FairLine API:
  /events/{event}/odds-comparison  → 24 top-liquid teams × 5 source columns
  /events/{event}/fair-odds        → 48 finalists × Fairline Model only (fills gaps
                                      for teams outside top-24)

Output:
  data/mc_simu/phase3_baselines/wc2026_vs_multi.csv  (gitignored)
  Markdown table to stdout — top overstates/understates per source.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mc_simu._common import banner  # noqa: E402
from mc_simu.wc2026_vs_fair import FAIR_TEAM_ALIASES, load_mc_champions  # noqa: E402


SOURCES = ["Betfair", "Other SB", "Fairline Model", "Kalshi", "Polymarket"]

# /prices endpoint uses lossy spellings for some teams that don't appear in
# FAIR_TEAM_ALIASES. Normalize to wc2026_groups.json canonical names here.
# Drop placeholder teams (None value) so they don't show up at all.
PRICES_TEAM_ALIASES: dict[str, str | None] = {
    "Usa":              "United States",
    "USA":              "United States",
    "Curaao":           "Curacao",
    "Curaçao":          "Curacao",
    "Turkiye":          "Turkey",
    "Türkiye":          "Turkey",
    "Bosnia Herzegovina":   "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina":   "Bosnia and Herzegovina",
    "Congo Dr":         "DR Congo",
    "Congo DR":         "DR Congo",
    "Korea Republic":   "South Korea",
    "Cabo Verde":       "Cape Verde",
    "Czech Republic":   "Czechia",
    "Cote d'Ivoire":    "Ivory Coast",
    "Team Z":           None,
}


def load_odds_comparison(api_base: str, event: str = "world_cup_2026"
                         ) -> tuple[dict[str, dict[str, float | None]], list[str]]:
    """Return ({team: {source: prob_or_None}}, active_sources) from /odds-comparison."""
    import requests
    url = f"{api_base.rstrip('/')}/events/{event}/odds-comparison"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    active = [s["id"] for s in payload.get("series", []) if s.get("has_data")]
    out: dict[str, dict[str, float | None]] = {}
    for row in payload.get("rows", []):
        team = FAIR_TEAM_ALIASES.get(row["team"], row["team"])
        out[team] = {src: row.get(src) for src in SOURCES}
    return out, active


def load_prices_endpoint(api_base: str, event: str = "world_cup_2026"
                         ) -> dict[str, dict[str, float]]:
    """Return {team: {Kalshi: mid, Polymarket: mid}} from /prices (covers longshots).

    /odds-comparison filters teams below 0.5pp anchor (~24 teams). /prices has
    full Kalshi (~36 teams) + Polymarket (~44 teams) coverage including longshots
    like Canada, Paraguay, Cape Verde. Use this to backfill the tail.
    """
    import requests
    url = f"{api_base.rstrip('/')}/events/{event}/prices"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    out: dict[str, dict[str, float]] = {}
    for row in resp.json():
        raw = str(row.get("team", ""))
        team = PRICES_TEAM_ALIASES.get(raw, FAIR_TEAM_ALIASES.get(raw, raw))
        if team is None or not team:
            continue
        plat = str(row.get("platform", ""))
        if "kalshi" in plat:
            src = "Kalshi"
        elif "polymarket" in plat:
            src = "Polymarket"
        else:
            continue
        mid = row.get("mid")
        if mid is None or not (0 < float(mid) < 1):
            continue
        out.setdefault(team, {}).setdefault(src, float(mid))
    return out


def load_prices_full(api_base: str, event: str = "world_cup_2026"
                     ) -> dict[str, dict[str, dict[str, float]]]:
    """Like load_prices_endpoint but keeps bid/ask: {team: {src: {mid,bid,ask}}}.

    Used by the daily tracker to log the executable spread, not just the mid —
    edge inside the spread is untradeable, so a future backtest needs both sides.
    bid/ask are best-effort (None when the platform doesn't quote a side).
    """
    import requests
    url = f"{api_base.rstrip('/')}/events/{event}/prices"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    out: dict[str, dict[str, dict[str, float]]] = {}
    for row in resp.json():
        raw = str(row.get("team", ""))
        team = PRICES_TEAM_ALIASES.get(raw, FAIR_TEAM_ALIASES.get(raw, raw))
        if team is None or not team:
            continue
        plat = str(row.get("platform", ""))
        src = "Kalshi" if "kalshi" in plat else ("Polymarket" if "polymarket" in plat else None)
        if src is None:
            continue
        mid = row.get("mid")
        if mid is None or not (0 < float(mid) < 1):
            continue

        def _side(v):
            return float(v) if v is not None and 0 < float(v) < 1 else None

        out.setdefault(team, {}).setdefault(
            src, {"mid": float(mid), "bid": _side(row.get("bid")),
                  "ask": _side(row.get("ask"))})
    return out


def load_fair_odds_endpoint(api_base: str, event: str = "world_cup_2026",
                            market: str = "winner") -> dict[str, float]:
    """Return {team: fair_prob} from /fair-odds (covers all 48 finalists, FLine only)."""
    import requests
    url = f"{api_base.rstrip('/')}/events/{event}/fair-odds"
    resp = requests.get(url, params={"market": market}, timeout=30)
    resp.raise_for_status()
    out: dict[str, float] = {}
    for r in resp.json():
        team = FAIR_TEAM_ALIASES.get(r["outcome"], r["outcome"])
        out[team] = float(r["fair_prob"])
    return out


def _pct(x: float | None) -> str:
    return f"{x*100:>7.2f}%" if x is not None else "   —   "


def _edge_pct(x: float | None) -> str:
    return f"{x*100:>+7.2f}pp" if x is not None else "   —   "


def _rel_pct(mc: float | None, mkt: float | None) -> str:
    """Sam's relative view (2026-06-09): (market - model)/model.

    +ve => market prices the team ABOVE our model (model cheap / under-values it);
    -ve => model prices it above market (model rich / over-values it). Reads the
    same disagreement as the abs-pp edge but normalized by model size, so a
    +0.5pp miss on a 0.5% longshot shows as +100%, not "negligible".
    """
    if mc is None or mkt is None or mc <= 0:
        return "   —    "
    return f"{(mkt - mc) / mc * 100:>+7.1f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="WC2026 MC champion-prob vs all FairLine API model sources.")
    parser.add_argument(
        "--mc-csv", type=Path,
        default=PROJECT_ROOT / "data" / "mc_simu" / "mc_wc2026_v1_100k_seed42.csv")
    parser.add_argument(
        "--api-url", default="https://seal-app-yatxw.ondigitalocean.app/api")
    parser.add_argument(
        "--out-csv", type=Path,
        default=PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines" / "wc2026_vs_multi.csv")
    parser.add_argument(
        "--sort-by", choices=["fline", "polymarket", "mc"], default="fline",
        help="Sort table by this source's probability descending.")
    parser.add_argument(
        "--format", choices=["full", "split"], default="full",
        help="full = single table; split = Top (PM/Kalshi-liquid) + Tail (FLine-only).")
    args = parser.parse_args(argv)

    if not args.mc_csv.exists():
        raise SystemExit(f"ERROR: MC output not found: {args.mc_csv}")
    mc = load_mc_champions(args.mc_csv)

    multi, active_sources = load_odds_comparison(args.api_url)
    fair48 = load_fair_odds_endpoint(args.api_url)
    prices = load_prices_endpoint(args.api_url)

    # Universe = MC finalists (48) ∪ multi (24) ∪ fair48 (54) ∪ prices (44). Default to MC list.
    teams = set(mc.keys()) | set(multi.keys()) | set(fair48.keys()) | set(prices.keys())

    rows: list[dict] = []
    for team in teams:
        mc_p = mc.get(team)
        multi_row = multi.get(team, {})
        # Use multi-source if available; fall back to /fair-odds for Fairline Model
        fline = multi_row.get("Fairline Model")
        if fline is None and team in fair48:
            fline = fair48[team]

        # Backfill Kalshi / Polymarket from /prices when /odds-comparison filtered
        # the team out (anchor <0.5pp). /odds-comparison applies power_devig;
        # raw mid from /prices is fine for longshots where devig delta < noise.
        prices_row = prices.get(team, {})
        kalshi_p = multi_row.get("Kalshi")
        if kalshi_p is None:
            kalshi_p = prices_row.get("Kalshi")
        polymarket_p = multi_row.get("Polymarket")
        if polymarket_p is None:
            polymarket_p = prices_row.get("Polymarket")

        row: dict = {
            "team":              team,
            "mc_prob":           round(mc_p, 5) if mc_p is not None else None,
            "betfair":           multi_row.get("Betfair"),
            "other_sb":          multi_row.get("Other SB"),
            "fairline_model":    fline,
            "kalshi":            kalshi_p,
            "polymarket":        polymarket_p,
        }
        # Compute edges (MC - source) where both are present
        for src_key, col in [("betfair", "edge_vs_betfair"),
                              ("other_sb", "edge_vs_other_sb"),
                              ("fairline_model", "edge_vs_fairline"),
                              ("kalshi", "edge_vs_kalshi"),
                              ("polymarket", "edge_vs_polymarket")]:
            val = row[src_key]
            if mc_p is not None and val is not None:
                row[col] = round(mc_p - val, 5)
            else:
                row[col] = None
            # Relative edge (Sam 2026-06-09): (market - model)/model, in %.
            rel_col = col.replace("edge_vs_", "rel_vs_")
            if mc_p is not None and mc_p > 0 and val is not None:
                row[rel_col] = round((val - mc_p) / mc_p * 100, 2)
            else:
                row[rel_col] = None
        # Round source values for CSV
        for k in ("betfair", "other_sb", "fairline_model", "kalshi", "polymarket"):
            if row[k] is not None:
                row[k] = round(row[k], 5)
        rows.append(row)

    # Sort by chosen source desc; treat None as 0
    sort_key = {"fline": "fairline_model", "polymarket": "polymarket", "mc": "mc_prob"}[args.sort_by]
    rows.sort(key=lambda r: -(r.get(sort_key) or 0))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {args.out_csv}")

    banner(f"WC2026 — MC vs {len(active_sources)} active sources: {', '.join(active_sources)}")
    print(f"MC rows: {len(mc)} | /odds-comparison: {len(multi)} teams × {len(active_sources)} sources | "
          f"/fair-odds backfill: {len(fair48)} teams")

    if args.format == "split":
        # Split criterion = original screenshot layout: Top = teams that made the
        # /odds-comparison anchor cut (≥0.5pp on Pinnacle/OtherSB anchor); Tail =
        # teams in MC that didn't make the cut. Tail teams now DO get Kalshi +
        # Polymarket columns (from /prices) — sportsbook columns stay blank
        # because Betfair/OtherSB simply don't quote longshots below ~0.5%.
        top = sorted(
            [r for r in rows if r["team"] in multi],
            key=lambda r: -(r.get("polymarket") or r.get("kalshi") or 0),
        )
        tail = sorted(
            [r for r in rows
             if r.get("mc_prob") is not None and r["team"] not in multi],
            key=lambda r: -(r.get("mc_prob") or 0),
        )
        banner("WC2026 — Top section (passes /odds-comparison ≥0.5pp anchor cut)")
        print(f"{'Team':<24} | {'MC':>7} | {'FLine':>7} | {'Betfair':>7} | {'OtherSB':>7} | "
              f"{'Kalshi':>7} | {'Polym':>7} | {'edgePoly':>10} | {'relPoly':>9} |")
        print("-" * 120)
        for r in top:
            print(f"{r['team']:<24} | "
                  f"{_pct(r['mc_prob'])} | "
                  f"{_pct(r['fairline_model'])} | "
                  f"{_pct(r['betfair'])} | "
                  f"{_pct(r['other_sb'])} | "
                  f"{_pct(r['kalshi'])} | "
                  f"{_pct(r['polymarket'])} | "
                  f"{_edge_pct(r['edge_vs_polymarket'])} | "
                  f"{_rel_pct(r['mc_prob'], r['polymarket'])} |")
        print()
        banner("Tail (longshots — Kalshi/Polym from /prices, sportsbooks don't quote)")
        print(f"{'Team':<24} | {'MC':>7} | {'FLine':>7} | {'Kalshi':>7} | {'Polym':>7} | "
              f"{'edgePoly':>10} | {'relPoly':>9} | {'relKalshi':>9} |")
        print("-" * 110)
        for r in tail:
            print(f"{r['team']:<24} | "
                  f"{_pct(r['mc_prob'])} | "
                  f"{_pct(r['fairline_model'])} | "
                  f"{_pct(r['kalshi'])} | "
                  f"{_pct(r['polymarket'])} | "
                  f"{_edge_pct(r['edge_vs_polymarket'])} | "
                  f"{_rel_pct(r['mc_prob'], r['polymarket'])} | "
                  f"{_rel_pct(r['mc_prob'], r['kalshi'])} |")
    else:
        banner(f"Full table (sorted by {sort_key})")
        print(f"{'Team':<24} {'MC':>8} {'Betfair':>8} {'OtherSB':>8} {'FLine':>8} {'Kalshi':>8} {'Polym':>8}  "
              f"{'edgeFLine':>10} {'edgePoly':>10}")
        print("-" * 120)
        for r in rows:
            if r["mc_prob"] is None and not any(r.get(s) for s in ["betfair", "fairline_model", "polymarket"]):
                continue
            print(f"{r['team']:<24} "
                  f"{_pct(r['mc_prob'])} "
                  f"{_pct(r['betfair'])} "
                  f"{_pct(r['other_sb'])} "
                  f"{_pct(r['fairline_model'])} "
                  f"{_pct(r['kalshi'])} "
                  f"{_pct(r['polymarket'])}  "
                  f"{_edge_pct(r['edge_vs_fairline'])} "
                  f"{_edge_pct(r['edge_vs_polymarket'])}")

    # Per-source summary stats — only over teams that have BOTH MC and that source
    print()
    banner("Per-source agreement with MC (over teams present in both)")
    print(f"{'Source':<18} {'n':>4} {'mean_edge':>11} {'mean|edge|':>11} {'max_overstate':>15} {'max_understate':>16}")
    for src, col in [("Betfair", "edge_vs_betfair"),
                     ("Other SB", "edge_vs_other_sb"),
                     ("Fairline Model", "edge_vs_fairline"),
                     ("Kalshi", "edge_vs_kalshi"),
                     ("Polymarket", "edge_vs_polymarket")]:
        edges = [r[col] for r in rows if r[col] is not None]
        if not edges:
            continue
        n = len(edges)
        mean_e = sum(edges) / n
        mean_abs = sum(abs(e) for e in edges) / n
        mx_over = max(edges)
        mx_under = min(edges)
        top_over = next(r['team'] for r in rows if r[col] == mx_over)
        top_under = next(r['team'] for r in rows if r[col] == mx_under)
        print(f"{src:<18} {n:>4} {mean_e*100:>+10.2f}pp {mean_abs*100:>+10.2f}pp "
              f"{mx_over*100:>+8.2f}pp ({top_over[:8]}) "
              f"{mx_under*100:>+8.2f}pp ({top_under[:8]})")

    # Relative-edge summary (Sam 2026-06-09): (market - model)/model. Use MEDIAN
    # — relative% explodes on near-zero-prob longshots, so mean is meaningless.
    print()
    banner("Per-source RELATIVE agreement — (market - model)/model, median over both")
    print(f"{'Source':<18} {'n':>4} {'median_rel':>12} {'median|rel|':>12}  (+ = model UNDER-prices)")
    for src, col in [("Betfair", "rel_vs_betfair"),
                     ("Other SB", "rel_vs_other_sb"),
                     ("Fairline Model", "rel_vs_fairline"),
                     ("Kalshi", "rel_vs_kalshi"),
                     ("Polymarket", "rel_vs_polymarket")]:
        rels = sorted(r[col] for r in rows if r.get(col) is not None)
        if not rels:
            continue
        n = len(rels)
        med = rels[n // 2]
        med_abs = sorted(abs(x) for x in rels)[n // 2]
        print(f"{src:<18} {n:>4} {med:>+11.1f}% {med_abs:>+11.1f}%")

    return 0


__all__ = ["load_odds_comparison", "load_fair_odds_endpoint", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
