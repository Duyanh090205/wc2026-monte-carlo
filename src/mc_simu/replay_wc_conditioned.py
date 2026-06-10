"""Backtest the conditioning harness on a past World Cup — conditional champion-prob trajectory vs market.

Replay a past WC (2018 or 2022) round by round, locking the real results as they
happened, and watch each team's conditional champion prob evolve. Overlay the
archived betting market (oddschecker via Wayback) at each snapshot's TRUE date
(interpolated onto the round axis). Validates that conditional re-sim gives
calibrated updates and tracks the market where we can observe it. Model = ELO+MV
(eloratings as-of + historical squad MV, alpha=0.5), STATIC — only the bracket is
conditioned. Market coverage is whatever Wayback archived (sparse, irregular).

Usage: PYTHONPATH=src python -m mc_simu.replay_wc_conditioned --year 2018 --plot
"""

from __future__ import annotations

import argparse
import sys
import time
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mc_simu.single_game as sg  # noqa: E402
from mc_simu._common import banner  # noqa: E402
from mc_simu.mv_blend import blend_elo_with_mv  # noqa: E402
from mc_simu.run_phase3_baselines import _collect_required_teams, _cutoff, load_ratings_for_source  # noqa: E402
from mc_simu.tournaments.wc_8groups import load_edition_bundle, run_monte_carlo  # noqa: E402

DATA = PROJECT_ROOT / "data" / "mc_simu"
ODDS_URL = "https://www.oddschecker.com/football/world-cup/winner"


def _fs(*t):
    return frozenset(t)


YEAR_CFG = {
    2018: dict(
        champion="France",
        group_end="2018-06-28",
        stage_dates={"pre": "2018-06-14", "groups": "2018-06-28", "R16": "2018-07-03",
                     "QF": "2018-07-07", "SF": "2018-07-11"},
        rounds=[("R16", "2018-06-30", "2018-07-03"), ("QF", "2018-07-06", "2018-07-07"),
                ("SF", "2018-07-10", "2018-07-11"), ("Final", "2018-07-15", "2018-07-15")],
        pen_adv={_fs("Russia", "Spain"): "Russia", _fs("Croatia", "Denmark"): "Croatia",
                 _fs("Colombia", "England"): "England", _fs("Russia", "Croatia"): "Croatia"},
        watch=["France", "Croatia", "Belgium", "Brazil", "England", "Germany", "Spain", "Argentina", "Russia"],
        fig_watch=["France", "Croatia", "Belgium", "Brazil", "Germany", "Russia", "Sweden", "Japan"],
        underdogs=["Russia", "Sweden", "Japan"],  # host + QF/R16 surprises
        market_window=("20180601", "20180716"),
    ),
    2022: dict(
        champion="Argentina",
        group_end="2022-12-02",
        stage_dates={"pre": "2022-11-20", "groups": "2022-12-02", "R16": "2022-12-06",
                     "QF": "2022-12-10", "SF": "2022-12-14"},
        rounds=[("R16", "2022-12-03", "2022-12-06"), ("QF", "2022-12-09", "2022-12-10"),
                ("SF", "2022-12-13", "2022-12-14"), ("Final", "2022-12-18", "2022-12-18")],
        pen_adv={_fs("Croatia", "Japan"): "Croatia", _fs("Morocco", "Spain"): "Morocco",
                 _fs("Croatia", "Brazil"): "Croatia", _fs("Argentina", "Netherlands"): "Argentina",
                 _fs("Argentina", "France"): "Argentina"},
        watch=["Argentina", "France", "Croatia", "Morocco", "Brazil", "Japan", "Senegal", "Australia", "Spain"],
        # underdogs (Morocco/Japan/Senegal/Australia) alongside the favourites
        fig_watch=["Argentina", "France", "Brazil", "Morocco", "Croatia", "Japan", "Senegal", "Australia"],
        underdogs=["Morocco", "Japan", "Senegal", "Australia"],
        market_window=("20221101", "20221231"),
    ),
}


def _norm(s):
    return " ".join(unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().split())


def build_ratings(year, alpha=0.5):
    required = _collect_required_teams("wc_8groups", year)
    ratings, _ = load_ratings_for_source("eloratings", _cutoff(year, "wc"), DATA, required)
    mv_all = pd.read_csv(DATA / "historical_squad_mv.csv")
    mv_ed = mv_all[mv_all["edition"] == f"WC{year}"]
    norm2name = {_norm(t): t for t in required}
    mv_d = {norm2name[_norm(r["team"])]: float(r["squad_mv_eur"]) for _, r in mv_ed.iterrows()
            if _norm(r["team"]) in norm2name and r["squad_mv_eur"] > 0}
    field = {t: ratings[t] for t in required if t in ratings}
    blended, _ = blend_elo_with_mv(field, mv_d, alpha=alpha)
    ratings.update(blended)
    return ratings


def load_results(year):
    cfg = YEAR_CFG[year]
    df = pd.read_csv(DATA / "matches_1998_2026.csv")
    df["date"] = pd.to_datetime(df["date"])
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == year)]
    group_scores, ko_by_round = {}, {lab: {} for lab, _, _ in cfg["rounds"]}
    for r in wc.itertuples(index=False):
        h, a, hs, as_ = r.home_team, r.away_team, int(r.home_score), int(r.away_score)
        if r.date <= pd.Timestamp(cfg["group_end"]):
            group_scores[frozenset((h, a))] = {h: hs, a: as_}
            continue
        for lab, lo, hi in cfg["rounds"]:
            if pd.Timestamp(lo) <= r.date <= pd.Timestamp(hi):
                w = h if hs > as_ else a if as_ > hs else cfg["pen_adv"][frozenset((h, a))]
                ko_by_round[lab][frozenset((h, a))] = w
    return group_scores, ko_by_round


def discover_market(year):
    """All daily-collapsed Wayback captures (status 200) of the oddschecker winner page."""
    import requests
    lo, hi = YEAR_CFG[year]["market_window"]
    params = {"url": "oddschecker.com/football/world-cup/winner", "from": lo, "to": hi,
              "output": "json", "filter": "statuscode:200", "collapse": "timestamp:8"}
    for _ in range(3):
        try:
            rows = requests.get("http://web.archive.org/cdx/search/cdx", params=params, timeout=40).json()[1:]
            return [row[1] for row in rows]
        except Exception:
            time.sleep(3)
    print("  CDX discovery failed (network) — no market overlay")
    return []


def fetch_market_points(year, model_teams):
    """[(date, {model_team: devigged prob})] for every archived snapshot that parses."""
    from mc_simu.validate_mv_market_history import parse_market
    mnorm = {_norm(t): t for t in model_teams}
    pts = []
    for ts in discover_market(year):
        url = f"http://web.archive.org/web/{ts}/{ODDS_URL}"
        try:
            mk = parse_market(url)
        except Exception:
            continue
        aligned = {mnorm[_norm(t)]: p for t, p in mk.items() if _norm(t) in mnorm}
        if not aligned:
            continue
        tot = sum(aligned.values()) or 1.0
        date = pd.Timestamp(f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}")
        pts.append((date, {t: p / tot for t, p in aligned.items()}))
    return pts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2022, choices=sorted(YEAR_CFG))
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)
    cfg = YEAR_CFG[args.year]

    sg.ELO_GOALS_DENOMINATOR = 1400.0
    ratings = build_ratings(args.year)
    bundle = load_edition_bundle(args.year, ratings)  # default params = prod diag 0.20 / HFA
    gs, ko_round = load_results(args.year)

    stages = [("pre", {}, {}), ("groups", gs, {})]
    surv = {}
    for lab, _, _ in cfg["rounds"][:-1]:  # R16, QF, SF (Final trivial: champion known)
        surv = {**surv, lab: set(ko_round[lab].values())}  # teams that advanced past `lab`
        stages.append((lab, gs, dict(surv)))

    traj = {}
    for label, g, k in stages:
        res = run_monte_carlo(bundle, n_iterations=args.n, seed=42, progress=False,
                              group_scores=g, survivors=k)
        for t, s in res["champion"].items():
            traj.setdefault(t, {})[label] = s["mc_fair_prob"]

    stage_labels = [s[0] for s in stages]
    banner(f"WC{args.year} conditional champion-prob trajectory (ELO+MV, n={args.n})  *=champion {cfg['champion']}")
    print(f"{'Team':<13}" + "".join(f"{s:>9}" for s in stage_labels))
    for t in cfg["watch"]:
        row = "".join(f"{traj.get(t, {}).get(s, 0) * 100:>8.1f}%" for s in stage_labels)
        print(f"{t:<13}{row}{' *' if t == cfg['champion'] else ''}")

    if args.plot:
        market_pts = fetch_market_points(args.year, list(traj.keys()))
        print(f"  market snapshots used: {len(market_pts)} "
              f"({', '.join(d.strftime('%m-%d') for d, _ in market_pts)})")
        _plot(args.year, traj, cfg["fig_watch"], stage_labels, cfg["stage_dates"], market_pts,
              cfg["champion"], cfg["underdogs"])
    return 0


def _plot(year, traj, watch, stage_labels, stage_dates, market_points, champion, underdogs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sx = list(range(len(stage_labels)))
    sd = [pd.Timestamp(stage_dates[s]).value for s in stage_labels]

    def date_to_x(d):
        return float(np.interp(pd.Timestamp(d).value, sd, sx))

    cmap = plt.get_cmap("tab10")
    colors = {t: cmap(i % 10) for i, t in enumerate(watch)}

    # Collapse snapshots that land on ~the same round position (e.g. several
    # pre-tournament crawls all map to x=0) — keep the latest, so markers don't stack.
    by_x: dict[float, tuple] = {}
    for date, mkt in sorted(market_points, key=lambda dm: dm[0]):
        by_x[round(date_to_x(date), 1)] = (date, mkt)
    mkt_xy = sorted(by_x.items())

    def draw(ax):
        for t in watch:
            ys = [traj.get(t, {}).get(s, 0) * 100 for s in stage_labels]
            ax.plot(sx, ys, "-o", color=colors[t], lw=2.4 if t == champion else 1.5,
                    ms=5, label=t, zorder=3)
        for xx, (_date, mkt) in mkt_xy:
            for t in watch:
                if t in mkt:
                    ax.scatter([xx], [mkt[t] * 100], color=colors[t], marker="X",
                               s=110, edgecolor="black", linewidth=0.7, zorder=5)
        ax.set_xticks(sx); ax.set_xticklabels(stage_labels)
        ax.set_xlabel("results locked in  →   (✕ = market at its real date)")
        ax.grid(True, alpha=0.25)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6.5))
    draw(axL); draw(axR)
    axL.plot([], [], "X", color="grey", markeredgecolor="black", ms=11, label="market (oddschecker, archived)")
    axL.set_ylabel("champion probability (%)")
    axL.set_title("full range — favourites climb / get knocked out")
    axL.legend(fontsize=8, ncol=2)
    axR.set_ylim(0, 12)
    axR.set_title(f"zoom 0–12% — underdogs ({', '.join(underdogs)})")
    fig.suptitle(f"WC{year} — model conditional champion prob through the rounds, vs market  (champion: {champion})\n"
                 "lines = STATIC model (ELO+MV) re-conditioned each round;  ✕ = archived betting market at its real date",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = PROJECT_ROOT / "mc_simu" / "figures" / f"wc{year}_conditional_trajectory.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nWrote figure: {out}")


__all__ = ["build_ratings", "load_results", "fetch_market_points", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
