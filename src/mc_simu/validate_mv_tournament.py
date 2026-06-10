"""Tournament-level view: ELO vs ELO+MV champion probabilities vs actual winners.

Companion to validate_mv_history.py (which tests per-match Brier, n=253 — the
statistically strong test). THIS is the whole-tournament view the per-match
reliability curve explicitly is NOT: model champion prob vs realized champions,
for Elo-only and Elo+MV, over WC2010/14/18/22.

HARD CAVEAT: only 4 editions = 4 champions. Any champion-level calibration here
is ILLUSTRATIVE, not conclusive. Treat as a sanity view, not a verdict.

Runs the historical MC sim twice per edition (raw eloratings, and eloratings
z-blended with point-in-time squad MV from historical_squad_mv.csv), then:
  - reliability: bin teams by predicted champion prob, actual champion rate
  - per-edition rank + prob the model gave the ACTUAL champion (does MV help
    see the winner?)

Usage:
    PYTHONPATH=src python -m mc_simu.validate_mv_tournament --plot
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mc_simu._common import banner  # noqa: E402
from mc_simu.mv_blend import blend_elo_with_mv  # noqa: E402
from mc_simu.run_phase3_baselines import (  # noqa: E402
    _collect_required_teams, _cutoff, load_ratings_for_source,
)
from mc_simu.tournaments.wc_8groups import load_edition_bundle, run_monte_carlo  # noqa: E402

DATA = PROJECT_ROOT / "data" / "mc_simu"
ACTUAL = {2010: "Spain", 2014: "Germany", 2018: "France", 2022: "Argentina"}
EDITIONS = [2010, 2014, 2018, 2022]


def _norm(s: object) -> str:
    return " ".join(unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().split())


def champion_probs(year: int, alpha: float, n: int, seed: int) -> dict[str, float]:
    """Run the historical MC sim for `year`; alpha=0 -> Elo-only, >0 -> Elo+MV blend."""
    required = _collect_required_teams("wc_8groups", year)
    ratings, _ = load_ratings_for_source("eloratings", _cutoff(year, "wc"), DATA, required)
    if alpha > 0:
        mv_all = pd.read_csv(DATA / "historical_squad_mv.csv")
        mv_ed = mv_all[mv_all["edition"] == f"WC{year}"]
        norm2name = {_norm(t): t for t in required}
        mv_d = {norm2name[_norm(r["team"])]: float(r["squad_mv_eur"])
                for _, r in mv_ed.iterrows()
                if _norm(r["team"]) in norm2name and r["squad_mv_eur"] > 0}
        field = {t: ratings[t] for t in required if t in ratings}
        blended, _ = blend_elo_with_mv(field, mv_d, alpha=alpha)
        ratings.update(blended)
    bundle = load_edition_bundle(year, ratings)
    res = run_monte_carlo(bundle, n_iterations=n, seed=seed, progress=False)
    return {t: s["mc_fair_prob"] for t, s in res["champion"].items()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", type=Path, nargs="?",
                   const=PROJECT_ROOT / "mc_simu" / "audits" / "mv_tournament_calibration.png",
                   default=None)
    args = p.parse_args(argv)

    rows = []          # per (edition, team): pred_elo, pred_mv, is_champ
    champ_rank = []    # per edition: rank/prob of actual champion under each model
    for year in EDITIONS:
        pe = champion_probs(year, 0.0, args.n, args.seed)
        pm = champion_probs(year, args.alpha, args.n, args.seed)
        actual = ACTUAL[year]
        for t in pe:
            rows.append({"year": year, "team": t, "elo": pe[t],
                         "mv": pm.get(t, 0.0), "champ": int(t == actual)})
        def rk(d):
            s = sorted(d.items(), key=lambda kv: -kv[1])
            return next((i + 1 for i, (tt, _) in enumerate(s) if tt == actual), len(s) + 1)
        champ_rank.append({"year": year, "actual": actual,
                           "elo_rank": rk(pe), "elo_p": pe.get(actual, 0.0),
                           "mv_rank": rk(pm), "mv_p": pm.get(actual, 0.0)})

    df = pd.DataFrame(rows)
    cr = pd.DataFrame(champ_rank)

    banner(f"Per-edition: rank + prob of ACTUAL champion (alpha={args.alpha}, n={args.n})")
    print(f"{'Year':<6}{'Champion':<12}{'ELO rank':>9}{'ELO P':>8}{'ELO+MV rank':>13}{'ELO+MV P':>10}")
    for _, r in cr.iterrows():
        print(f"{r['year']:<6}{r['actual']:<12}{r['elo_rank']:>9}{r['elo_p']*100:>7.1f}%"
              f"{r['mv_rank']:>13}{r['mv_p']*100:>9.1f}%")
    print(f"\n  mean rank of actual champion:  ELO {cr['elo_rank'].mean():.2f}  vs  "
          f"ELO+MV {cr['mv_rank'].mean():.2f}   (lower = saw winner better)")
    print(f"  mean P(actual champion):       ELO {cr['elo_p'].mean()*100:.2f}%  vs  "
          f"ELO+MV {cr['mv_p'].mean()*100:.2f}%  (higher = better)")

    # reliability bins
    bins = [0, 0.01, 0.03, 0.06, 0.10, 1.0]
    labels = ["<1%", "1-3%", "3-6%", "6-10%", ">10%"]
    banner("Champion-prob reliability (pred bin -> actual champion rate) — n=4 champions, ILLUSTRATIVE")
    print(f"{'bin':<8}{'n':>4}{'ELO pred':>10}{'ELO actual':>12}{'MV pred':>10}{'MV actual':>12}")
    rel = []
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        for col, who in [("elo", "ELO"), ("mv", "ELO+MV")]:
            m = df[(df[col] >= lo) & (df[col] < hi)]
            rel.append({"bin": lab, "model": who, "n": len(m),
                        "pred": m[col].mean() if len(m) else np.nan,
                        "actual": m["champ"].mean() if len(m) else np.nan})
    reldf = pd.DataFrame(rel)
    for lab in labels:
        e = reldf[(reldf.bin == lab) & (reldf.model == "ELO")].iloc[0]
        v = reldf[(reldf.bin == lab) & (reldf.model == "ELO+MV")].iloc[0]
        print(f"{lab:<8}{e['n']:>4}{e['pred']*100:>9.1f}%{e['actual']*100:>11.1f}%"
              f"{v['pred']*100:>9.1f}%{v['actual']*100:>11.1f}%")

    if args.plot is not None:
        _figure(df, cr, reldf, labels, args.plot, args.alpha)
    return 0


def _figure(df, cr, reldf, labels, out, alpha):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Task 1 (tournament-level) — ELO vs ELO+MV champion prob vs actual, WC2010-2022\n"
                 f"alpha={alpha}  |  CAVEAT: only 4 champions — illustrative, not conclusive "
                 f"(per-match Brier n=253 is the strong test)", fontsize=11, fontweight="bold")

    # Panel 1 — reliability (binned)
    a = ax[0]
    a.plot([0, 0.4], [0, 0.4], "k--", lw=0.8, label="perfect")
    for who, c in [("ELO", "#1f77b4"), ("ELO+MV", "#ff7f0e")]:
        sub = reldf[(reldf.model == who) & (reldf.n > 0)].dropna()
        a.plot(sub["pred"], sub["actual"], "-o", color=c, label=who)
    a.set_xlabel("Model predicted champion prob (bin mean)")
    a.set_ylabel("Actual champion frequency in bin")
    a.set_title("(1) Champion-prob reliability\n(below diagonal = model OVER-values; n=4 champs)")
    a.legend()

    # Panel 2 — prob given to the actual champion, per edition
    b = ax[1]
    x = np.arange(len(cr))
    b.bar(x - 0.2, cr["elo_p"] * 100, 0.4, label="ELO", color="#1f77b4")
    b.bar(x + 0.2, cr["mv_p"] * 100, 0.4, label="ELO+MV", color="#ff7f0e")
    for i, r in cr.iterrows():
        b.annotate(f"#{int(r['elo_rank'])}", (i - 0.2, r["elo_p"] * 100), ha="center",
                   va="bottom", fontsize=7)
        b.annotate(f"#{int(r['mv_rank'])}", (i + 0.2, r["mv_p"] * 100), ha="center",
                   va="bottom", fontsize=7)
    b.set_xticks(x)
    b.set_xticklabels([f"{r['year']}\n{r['actual']}" for _, r in cr.iterrows()], fontsize=9)
    b.set_ylabel("Model prob given to ACTUAL champion (%)")
    b.set_title("(2) Did the model see the winner? (rank labelled)\nhigher bar / lower # = better")
    b.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nWrote figure: {out}")


__all__ = ["champion_probs", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
