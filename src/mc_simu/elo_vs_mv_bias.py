"""ELO-only vs ELO+MV: is the MV blend a systematic favorite/underdog distortion?

Answers Sam's question (2026-06-09): does our current model (Elo blended with
Transfermarkt market value) systematically over/under-value favorites vs
underdogs relative to Elo-only — following a rule — or is the shift random?

Reads three already-computed configs (same D=1400, diag=0.20, eloratings source,
n=10k seed=42 — so the deltas isolate the blend, not run noise):
    tune_baseline_alpha0.csv    Elo-only            (alpha=0.0, X=0)
    tune_blend_alpha50.csv      Elo+MV              (alpha=0.5, X=0)
    tune_final_mv50_star15.csv  Elo+MV+star (final) (alpha=0.5, X=15)
plus market columns (Polymarket / Kalshi / Fairline) from
    wc2026_final_mv_star_vs_market.csv

Everything is read from cached CSVs — no network. Outputs:
  - per-team table (Elo, +MV, final, market, deltas, abs-pp + relative-% edges)
  - per-tier means (favorite -> longshot) for the MV-induced delta
  - Spearman/Pearson of delta vs favorite-rank => systematic-vs-random verdict
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from mc_simu._common import banner  # noqa: E402

BASE = PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines"


def load_tune_row(csv_path: Path) -> dict[str, float]:
    """Single-cell tune CSV -> {team: champion_prob} from mc_<team> columns."""
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return {k[3:]: float(v) for k, v in row.items()
            if k.startswith("mc_") and v not in ("", None)}


def load_market(csv_path: Path) -> dict[str, dict]:
    """final-vs-market CSV -> {team: {final, pm, kalshi, fairline, stars, mv}}."""
    out: dict[str, dict] = {}
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            def fnum(key):
                v = r.get(key, "")
                return float(v) if v not in ("", None) else None
            out[r["team"]] = {
                "final": fnum("mc_prob"),
                "pm": fnum("polymarket"),
                "kalshi": fnum("kalshi"),
                "fairline": fnum("fairline"),
                "stars": fnum("stars"),
                "mv": fnum("mv_eur_total"),
            }
    return out


def consensus(m: dict) -> float | None:
    """Median of PM + Kalshi (Sam: not Fairline — self-referential)."""
    vals = [m[k] for k in ("pm", "kalshi") if m.get(k) is not None]
    return float(np.median(vals)) if vals else None


def spearman(x: list[float], y: list[float]) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def squad_value_gap() -> dict[str, float]:
    """Per-team (z_mv - z_elo) over the 48 finalists — the exact axis the MV
    blend moves teams along (see mv_blend.blend_elo_with_mv). +ve => team is
    MV-rich relative to its Elo (blend pushes it UP); -ve => Elo-rich, MV-poor
    (blend pulls it DOWN). Reuses the production loaders so names align.
    """
    import pandas as pd

    from mc_simu.mv_blend import load_mv_table
    from mc_simu.run_phase3_baselines import load_ratings_for_source
    from mc_simu.tune_to_market import _required_teams

    teams_48 = _required_teams()
    data_dir = PROJECT_ROOT / "data" / "mc_simu"
    ratings, _ = load_ratings_for_source(
        "eloratings", pd.Timestamp("2026-06-09"), data_dir, teams_48)
    mv = load_mv_table()

    twm = [t for t in teams_48 if t in mv and t in ratings]
    elos = np.array([ratings[t] for t in twm], dtype=float)
    lmv = np.log(np.array([mv[t] for t in twm], dtype=float))
    z_elo = (elos - elos.mean()) / elos.std()
    z_mv = (lmv - lmv.mean()) / lmv.std()
    return {t: float(z_mv[i] - z_elo[i]) for i, t in enumerate(twm)}


def make_figure(rows: list[dict], out_path: Path) -> None:
    """4-panel proof for Task 1 (MV blend = systematic squad-value reorder)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gap = squad_value_gap()
    ranked = [r for r in rows if r["cons"]]
    label = {"Spain", "Argentina", "France", "England", "Brazil",
             "Portugal", "Germany", "Norway", "Colombia", "Mexico"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Task 1 — ELO+MV vs ELO-only: systematic squad-value reorder, "
                 "not random (and not blanket favorite/underdog bias)",
                 fontsize=13, fontweight="bold")

    # Panel 1 — Δ_MV vs favorite-rank (market): messy => NOT favorite-driven
    ax = axes[0, 0]
    x1 = [r["cons"] * 100 for r in ranked]
    y1 = [r["d_mv"] * 100 for r in ranked]
    ax.scatter(x1, y1, s=28, c="#888", zorder=3)
    for r in ranked:
        if r["team"] in label:
            ax.annotate(r["team"], (r["cons"] * 100, r["d_mv"] * 100),
                        fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    sp = spearman([r["cons"] for r in ranked], y1)
    ax.set_title(f"(1) Shift vs favorite-rank — Spearman={sp:+.2f} (weak/noisy)\n"
                 "=> NOT explained by 'how favorite you are'")
    ax.set_xlabel("Market champion prob (%, log)")
    ax.set_ylabel("Δ prob from MV blend (pp)")

    # Panel 2 — Δ_MV vs squad-value gap: systematic mechanism. Point size ~ prob
    # (the dead tail sits at Δ~0 regardless of gap and only dilutes a linear fit;
    # rank-correlation + visual weighting show the real signal among movers).
    ax = axes[0, 1]
    pts = [(gap[r["team"]], r["d_mv"] * 100, r["team"], r["elo"]) for r in ranked if r["team"] in gap]
    gx = np.array([p[0] for p in pts]); gy = np.array([p[1] for p in pts])
    sizes = np.array([18 + p[3] * 100 * 9 for p in pts])
    ax.scatter(gx, gy, s=sizes, c="#1f77b4", alpha=0.7, zorder=3)
    movers = [p for p in pts if p[3] > 0.005]
    mx = np.array([p[0] for p in movers]); my = np.array([p[1] for p in movers])
    m, b = np.polyfit(mx, my, 1)
    xs = np.linspace(gx.min(), gx.max(), 50)
    ax.plot(xs, m * xs + b, "r--", lw=1.5)
    sp_all = spearman(list(gx), list(gy))
    sp_mov = spearman(list(mx), list(my))
    for gxx, gyy, t, _ in pts:
        if t in label:
            ax.annotate(t, (gxx, gyy), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_title(f"(2) Shift vs squad-value gap (z_MV - z_Elo) — Spearman={sp_all:+.2f} "
                 f"(movers {sp_mov:+.2f})\n=> SYSTEMATIC: blend moves each team toward its MV rank "
                 "(size ~ prob)")
    ax.set_xlabel("Squad-value gap  z_MV - z_Elo   (-ve: Elo-rich/MV-poor)")
    ax.set_ylabel("Δ prob from MV blend (pp)")

    # Panel 3 — mean Δ by favorite tier: redistribution + tail untouched
    ax = axes[1, 0]
    tiers = [("Favorites\n(top 6)", ranked[:6]), ("Contenders\n(7-14)", ranked[6:14]),
             ("Dark horses\n(15-26)", ranked[14:26]), ("Longshots\n(27+)", ranked[26:])]
    names = [t[0] for t in tiers]
    dmv = [np.mean([r["d_mv"] for r in g]) * 100 for _, g in tiers]
    dfin = [np.mean([r["d_final"] for r in g]) * 100 for _, g in tiers]
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.2, dmv, 0.4, label="Δ from MV", color="#1f77b4")
    ax.bar(xpos + 0.2, dfin, 0.4, label="Δ from MV+star (final)", color="#ff7f0e")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xpos); ax.set_xticklabels(names, fontsize=9)
    ax.set_title("(3) Mean shift by tier — MV reorders the top, star re-inflates it,\n"
                 "underdog tail ~0 (MV/prob both ~0 there)")
    ax.set_ylabel("Mean Δ champion prob (pp)")
    ax.legend(fontsize=9)

    # Panel 4 — per-team path ELO -> +MV -> final, top teams
    ax = axes[1, 1]
    top = sorted(rows, key=lambda r: -r["elo"])[:8]
    xs3 = [0, 1, 2]
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(top):
        ys = [r["elo"] * 100, r["mv"] * 100, r["final"] * 100]
        ax.plot(xs3, ys, "-o", color=cmap(i % 10), lw=1.6, ms=4, label=r["team"])
        ax.annotate(r["team"], (2, r["final"] * 100), fontsize=8,
                    xytext=(4, 0), textcoords="offset points")
    ax.set_xticks(xs3); ax.set_xticklabels(["ELO-only", "ELO+MV", "ELO+MV+star"])
    ax.set_title("(4) Per-team path — Spain/Argentina dip on MV then recover on star;\n"
                 "France/England/Germany rise on MV")
    ax.set_ylabel("Champion prob (%)")
    ax.set_xlim(-0.2, 2.6)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Wrote figure: {out_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--elo-csv", type=Path, default=BASE / "tune_baseline_alpha0.csv")
    p.add_argument("--mv-csv", type=Path, default=BASE / "tune_blend_alpha50.csv")
    p.add_argument("--final-csv", type=Path, default=BASE / "tune_final_mv50_star15.csv")
    p.add_argument("--market-csv", type=Path, default=BASE / "wc2026_final_mv_star_vs_market.csv")
    p.add_argument("--plot", type=Path, nargs="?",
                   const=PROJECT_ROOT / "mc_simu" / "audits" / "elo_vs_mv_bias.png",
                   default=None, help="write 4-panel proof figure to this path")
    args = p.parse_args(argv)

    for f in (args.elo_csv, args.mv_csv, args.final_csv, args.market_csv):
        if not f.exists():
            raise SystemExit(f"missing: {f}")

    p_elo = load_tune_row(args.elo_csv)
    p_mv = load_tune_row(args.mv_csv)
    p_final = load_tune_row(args.final_csv)
    mkt = load_market(args.market_csv)

    teams = sorted(set(p_elo) & set(p_mv) & set(p_final),
                   key=lambda t: -(consensus(mkt.get(t, {})) or 0))

    rows = []
    for t in teams:
        m = mkt.get(t, {})
        c = consensus(m)
        d_mv = p_mv[t] - p_elo[t]          # pure MV effect (vs Elo-only)
        d_final = p_final[t] - p_elo[t]    # MV + star effect
        # edges of the FINAL model vs consensus: absolute pp + relative %
        abs_pp = (p_final[t] - c) * 100 if c else None
        # Sam's relative framing: (market - model)/model
        rel_mkt_vs_model = (c - p_final[t]) / p_final[t] * 100 if (c and p_final[t] > 0) else None
        # trading-natural: model relative to market
        rel_model_vs_mkt = (p_final[t] / c - 1) * 100 if c else None
        rows.append({
            "team": t, "elo": p_elo[t], "mv": p_mv[t], "final": p_final[t],
            "cons": c, "d_mv": d_mv, "d_final": d_final,
            "abs_pp": abs_pp, "rel_mkt_vs_model": rel_mkt_vs_model,
            "rel_model_vs_mkt": rel_model_vs_mkt, "stars": m.get("stars"),
        })

    # ── per-team table ──────────────────────────────────────────────────────
    banner("Per-team: Elo-only -> +MV -> +MV+star (final), vs market consensus")
    print(f"{'Team':<22} {'Elo%':>6} {'+MV%':>6} {'Fin%':>6} {'Mkt%':>6} | "
          f"{'dMV(pp)':>8} {'dFin(pp)':>8} | {'abs pp':>7} {'rel%(mkt/mdl)':>13}")
    print("-" * 104)
    for r in rows:
        c_s = f"{r['cons']*100:>6.2f}" if r['cons'] else "   -  "
        abs_s = f"{r['abs_pp']:>+7.2f}" if r['abs_pp'] is not None else "   -   "
        rel_s = f"{r['rel_mkt_vs_model']:>+12.1f}%" if r['rel_mkt_vs_model'] is not None else "     -      "
        print(f"{r['team']:<22} {r['elo']*100:>6.2f} {r['mv']*100:>6.2f} "
              f"{r['final']*100:>6.2f} {c_s} | {r['d_mv']*100:>+8.2f} {r['d_final']*100:>+8.2f} | "
              f"{abs_s} {rel_s}")

    # ── per-tier means of the MV-induced delta ──────────────────────────────
    # tiers by market-consensus rank (external favorite axis)
    ranked = [r for r in rows if r["cons"]]
    n = len(ranked)
    tiers = [
        ("Favorites (top 6)", ranked[:6]),
        ("Contenders (7-14)", ranked[6:14]),
        ("Dark horses (15-26)", ranked[14:26]),
        ("Longshots (27+)", ranked[26:]),
    ]
    banner("MV-induced shift by favorite tier (mean over teams in tier)")
    print(f"{'Tier':<22} {'n':>3} {'mean dMV(pp)':>13} {'mean dFinal(pp)':>16} "
          f"{'mean abs-edge(pp)':>18} {'mean rel-edge%':>15}")
    for name, grp in tiers:
        if not grp:
            continue
        dmv = np.mean([r["d_mv"] for r in grp]) * 100
        dfin = np.mean([r["d_final"] for r in grp]) * 100
        abse = np.mean([r["abs_pp"] for r in grp if r["abs_pp"] is not None])
        rele = np.mean([r["rel_mkt_vs_model"] for r in grp if r["rel_mkt_vs_model"] is not None])
        print(f"{name:<22} {len(grp):>3} {dmv:>+13.3f} {dfin:>+16.3f} {abse:>+18.3f} {rele:>+14.1f}%")

    # ── systematic vs random ────────────────────────────────────────────────
    banner("Systematic vs random — is the MV shift correlated with favorite-rank?")
    dmv_v = [r["d_mv"] for r in ranked]
    dfin_v = [r["d_final"] for r in ranked]
    log_mkt = [np.log(r["cons"]) for r in ranked]
    print(f"  n teams with market = {n}")
    print(f"  Spearman(dMV, favorite-rank)        = {spearman([r['cons'] for r in ranked], dmv_v):+.3f}")
    print(f"  Pearson(dMV, log market-prob)       = {np.corrcoef(log_mkt, dmv_v)[0,1]:+.3f}")
    print(f"  Spearman(dFinal, favorite-rank)     = {spearman([r['cons'] for r in ranked], dfin_v):+.3f}")
    print(f"  Pearson(dFinal, log market-prob)    = {np.corrcoef(log_mkt, dfin_v)[0,1]:+.3f}")
    # signed totals: how much probability mass MV moves up among favorites
    fav_set = ranked[:14]
    dog_set = ranked[14:]
    print(f"  Net dMV mass, favorites(top14)      = {sum(r['d_mv'] for r in fav_set)*100:+.2f}pp")
    print(f"  Net dMV mass, underdogs(15+)        = {sum(r['d_mv'] for r in dog_set)*100:+.2f}pp")

    if args.plot is not None:
        make_figure(rows, args.plot)
    return 0


__all__ = ["load_tune_row", "load_market", "consensus", "spearman",
           "squad_value_gap", "make_figure", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
