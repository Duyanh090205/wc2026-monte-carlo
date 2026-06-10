"""Model vs MARKET vs ACTUAL on past World Cups (2018, 2022).

Now possible because oddschecker pre/early-tournament winner odds were recovered
from the Wayback Machine (the historical-outright-odds gap is closed for these
two editions). For each edition we put three things side by side on the SAME
tournament for the first time:
    model champion prob (ELO-only and ELO+MV)
    market champion prob (oddschecker, devigged)
    actual champion

Questions answered:
  - Does ELO+MV get CLOSER to market in the past (JSD), like it does for WC2026?
  - Did the model or the market give the actual champion a higher prob?

Usage: PYTHONPATH=src python -m mc_simu.validate_mv_market_history --plot
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import requests  # noqa: E402

from mc_simu._common import banner  # noqa: E402
from mc_simu.tune_to_market import jsd, normalize  # noqa: E402
from mc_simu.validate_mv_tournament import champion_probs  # noqa: E402

WAYBACK = {
    2018: "http://web.archive.org/web/20180617051841/https://www.oddschecker.com/football/world-cup/winner",
    2022: "http://web.archive.org/web/20221118194846/https://www.oddschecker.com/football/world-cup/winner",
}
ACTUAL = {2018: "France", 2022: "Argentina"}
CACHE = PROJECT_ROOT / "data" / "mc_simu" / "cache"

# oddschecker spelling -> adapter/eloratings canonical
MKT_ALIAS = {
    "USA": "United States", "South Korea": "Korea Republic", "Korea Republic": "Korea Republic",
    "IR Iran": "Iran", "Czech Republic": "Czechia", "Costa Rica": "Costa Rica",
}


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().split())


def parse_market(url: str) -> dict[str, float]:
    """Return {team: devigged market prob} from an oddschecker winner snapshot."""
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).text
    dec: dict[str, float] = {}
    for tr in re.findall(r"<tr\b[^>]*>", html):
        if "data-best-dig=" not in tr:
            continue
        bn = re.search(r'data-bname="([^"]+)"', tr)
        bd = re.search(r'data-best-dig="([\d.]+)"', tr)
        if bn and bd and float(bd.group(1)) > 1:
            dec[MKT_ALIAS.get(bn.group(1), bn.group(1))] = float(bd.group(1))
    tot = sum(1 / d for d in dec.values())
    return {t: (1 / d) / tot for t, d in dec.items()}


def _align(model: dict[str, float], market: dict[str, float]) -> dict[str, str]:
    """market team -> model team key (by normalized name)."""
    mnorm = {_norm(t): t for t in model}
    out = {}
    for t in market:
        k = _norm(t)
        if k in mnorm:
            out[t] = mnorm[k]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    fig_rows = []
    for year in (2018, 2022):
        market = parse_market(WAYBACK[year])
        elo = normalize(champion_probs(year, 0.0, args.n, 42))
        mv = normalize(champion_probs(year, 0.5, args.n, 42))
        amap = _align(elo, market)  # market-name -> model-name
        # restrict to teams present in both (model names)
        common_model = sorted({amap[t] for t in amap}, key=lambda t: -mv.get(t, 0))
        mkt_by_model = {amap[t]: market[t] for t in amap}
        mkt_n = normalize({t: mkt_by_model[t] for t in common_model})
        elo_n = normalize({t: elo.get(t, 0.0) for t in common_model})
        mv_n = normalize({t: mv.get(t, 0.0) for t in common_model})
        teams = common_model
        champ = ACTUAL[year]

        banner(f"WC{year} — model vs MARKET vs ACTUAL ({champ})  [{len(teams)} teams matched]")
        print(f"  JSD(model, market):  ELO={jsd(elo_n, mkt_n, teams):.4f}   "
              f"ELO+MV={jsd(mv_n, mkt_n, teams):.4f}   (lower = closer to market)")
        l1e = sum(abs(elo_n[t] - mkt_n[t]) for t in teams)
        l1m = sum(abs(mv_n[t] - mkt_n[t]) for t in teams)
        print(f"  L1 (model-market):   ELO={l1e*100:.1f}pp   ELO+MV={l1m*100:.1f}pp")

        def rk(d, t):
            return sorted(d, key=lambda x: -d[x]).index(t) + 1 if t in d else None
        print(f"\n  Prob given to actual champion ({champ}):")
        print(f"    market   {mkt_n.get(champ,0)*100:>5.1f}%  (rank #{rk(mkt_n,champ)})")
        print(f"    ELO      {elo_n.get(champ,0)*100:>5.1f}%  (rank #{rk(elo_n,champ)})")
        print(f"    ELO+MV   {mv_n.get(champ,0)*100:>5.1f}%  (rank #{rk(mv_n,champ)})")

        print(f"\n  {'Team':<16}{'market':>8}{'ELO':>8}{'ELO+MV':>8}{'MV−mkt':>8}")
        for t in teams[:10]:
            star = " *CHAMP" if t == champ else ""
            print(f"    {t:<14}{mkt_n[t]*100:>7.2f}%{elo_n[t]*100:>7.2f}%{mv_n[t]*100:>7.2f}%"
                  f"{(mv_n[t]-mkt_n[t])*100:>+7.2f}{star}")
        fig_rows.append((year, champ, teams, mkt_n, mv_n))

    if args.plot:
        _plot(fig_rows)
    return 0


def _plot(fig_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(fig_rows), figsize=(7 * len(fig_rows), 9))
    for ax, (year, champ, teams, mkt, mv) in zip(np_axes(axes), fig_rows):
        rows = sorted(teams, key=lambda t: mv[t] - mkt[t])
        diffs = [(mv[t] - mkt[t]) * 100 for t in rows]
        colours = ["#2a8a2a" if d > 0 else "#cc3b2f" for d in diffs]
        ys = range(len(rows))
        ax.barh(list(ys), diffs, color=colours, edgecolor="black", linewidth=0.4)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([f"{t} *" if t == champ else t for t in rows], fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"WC{year}: ELO+MV − market (pp)\n* = actual champion ({champ})")
        ax.set_xlabel("model − market (pp)")
    fig.tight_layout()
    out = PROJECT_ROOT / "mc_simu" / "figures" / "wc_past_model_vs_market.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nWrote figure: {out}")


def np_axes(axes):
    return axes if hasattr(axes, "__len__") else [axes]


__all__ = ["parse_market", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
