"""Current FINAL model (Elo+MV+star) vs LIVE market — sorted diff bar chart.

Durable, re-runnable replacement for the old ad-hoc presentation diff figures
(which had no generator and went stale as market moved). The model side is
stable (Elo+MV+star champion probs from the final-config CSV); the market side
is pulled LIVE each run, so the figure is always current.

    PYTHONPATH=src python -m mc_simu.plot_model_vs_market            # live PM
    PYTHONPATH=src python -m mc_simu.plot_model_vs_market --source kalshi

Negative bar = model below market; positive = model above. Colour bins match
the legend used in the audit figures. Annotates relative % (Sam 2026-06-09)
since abs-pp hides how large the tail disagreement really is.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mc_simu._common import banner  # noqa: E402
from mc_simu.tune_to_market import _required_teams, normalize  # noqa: E402
from mc_simu.wc2026_vs_multi import load_prices_endpoint  # noqa: E402

DEFAULT_API = "https://seal-app-yatxw.ondigitalocean.app/api"


def load_final_model(csv_path: Path) -> dict[str, float]:
    """{team: mc_prob} from the final-config vs-market CSV (model side is stable)."""
    out: dict[str, float] = {}
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = r.get("mc_prob", "")
            if v not in ("", None):
                out[r["team"]] = float(v)
    return out


def _colour(diff_pp: float) -> str:
    if diff_pp > 2:    return "#1a6b1a"   # higher >2pp
    if diff_pp > 1:    return "#3ca33c"   # higher 1-2pp
    if diff_pp > 0:    return "#a6d8a6"   # higher 0-1pp
    if diff_pp > -1:   return "#f2c14e"   # lower 0-1pp
    if diff_pp > -2:   return "#e8862e"   # lower 1-2pp
    return "#cc3b2f"                       # lower >2pp


def _compare_figure(plt, model, mkt, common, out, source) -> None:
    """2 panels, same teams: LEFT = absolute pp, RIGHT = relative %, each sorted
    by its own metric. Both use '+ = model above market' so signs are consistent.
    Shows the reordering: absolute is dominated by favorites (big probs, big pp),
    relative is dominated by the longshot tail (tiny probs explode proportionally).
    """
    src = source.capitalize()
    data = []
    for t in common:
        m, k = model[t], mkt[t]
        abs_pp = (m - k) * 100
        rel = (m / k - 1) * 100 if k > 0 else 0.0  # model vs market, + = model above
        data.append((t, abs_pp, rel))
    abs_sorted = sorted(data, key=lambda r: r[1], reverse=True)
    rel_sorted = sorted(data, key=lambda r: r[2], reverse=True)

    def col(v):
        return "#2a8a2a" if v > 0 else "#cc3b2f"

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(15, 13))
    fig.suptitle(f"WC2026 final model vs LIVE {src} — ABSOLUTE vs RELATIVE "
                 "(+ = model above market)\nSame 46 teams, each panel sorted by its own "
                 "metric → the ranking flips completely", fontsize=12, fontweight="bold")

    ys = list(range(len(abs_sorted)))
    axa.barh(ys, [r[1] for r in abs_sorted], color=[col(r[1]) for r in abs_sorted],
             edgecolor="black", linewidth=0.4)
    axa.set_yticks(ys); axa.set_yticklabels([r[0] for r in abs_sorted], fontsize=7)
    axa.invert_yaxis(); axa.axvline(0, color="black", lw=0.8)
    av = [r[1] for r in abs_sorted]
    axa.set_xlim(min(av) - 1.4, max(av) + 1.4)
    for i, r in enumerate(abs_sorted):
        axa.annotate(f"{r[1]:+.2f}", (r[1], i), fontsize=6, va="center",
                     ha="left" if r[1] >= 0 else "right",
                     xytext=(2 if r[1] >= 0 else -2, 0), textcoords="offset points")
    axa.set_xlabel(f"model − {src}  (pp, absolute)")
    axa.set_title("ABSOLUTE (pp) — sorted by pp\nFAVORITES dominate the extremes")

    CAP_HI, CAP_LO = 250.0, -110.0
    draw = [max(CAP_LO, min(CAP_HI, r[2])) for r in rel_sorted]
    axb.barh(ys, draw, color=[col(r[2]) for r in rel_sorted],
             edgecolor="black", linewidth=0.4)
    axb.set_yticks(ys); axb.set_yticklabels([r[0] for r in rel_sorted], fontsize=7)
    axb.invert_yaxis(); axb.axvline(0, color="black", lw=0.8)
    axb.set_xlim(CAP_LO - 15, CAP_HI + 55)
    for i, r in enumerate(rel_sorted):
        d = draw[i]
        axb.annotate(f"{r[2]:+.0f}%", (d, i), fontsize=6, va="center",
                     ha="left" if d >= 0 else "right",
                     xytext=(2 if d >= 0 else -2, 0), textcoords="offset points")
    axb.set_xlabel(f"(model / {src} − 1)  (%, relative)   [bars clipped to ±, true % labelled]")
    axb.set_title("RELATIVE (%) — sorted by %\nLONGSHOT TAIL dominates the extremes")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out2 = out.with_name("wc2026_model_vs_market_abs_vs_rel.png")
    out2.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out2, dpi=130)
    print(f"Wrote figure: {out2}")


def _table_figure(plt, model, mkt, common, out, source) -> None:
    """Per-team numbers as a table figure: model %, market %, abs pp, rel %."""
    src = source.capitalize()
    rows = sorted(common, key=lambda t: -model[t])
    cell_text, cell_colours = [], []
    GP, GN, W = "#cdebcd", "#f4cccc", "#ffffff"
    for t in rows:
        m, k = model[t] * 100, mkt[t] * 100
        absd = m - k
        rel = (model[t] / mkt[t] - 1) * 100 if mkt[t] > 0 else 0.0
        cell_text.append([t, f"{m:.2f}", f"{k:.2f}", f"{absd:+.2f}", f"{rel:+.0f}%"])
        c = GP if absd > 0 else GN
        cell_colours.append([W, W, W, c, c])

    fig, ax = plt.subplots(figsize=(8.2, 0.28 * len(rows) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text,
                   colLabels=["Team", "Model %", f"{src} %", "Abs (pp)", "Rel %"],
                   cellColours=cell_colours, colColours=["#d9d9d9"] * 5,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.15)
    for (r, c), cell in tbl.get_celld().items():
        if c == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.03
        if r == 0:
            cell.set_text_props(fontweight="bold")
    tbl.auto_set_column_width([0, 1, 2, 3, 4])
    ax.set_title(f"WC2026 final model vs LIVE {src} — per-team numbers (sorted by model %)\n"
                 "Abs = model − market (pp)   |   Rel = model/market − 1 (%)   |   "
                 "green = model above market, red = below",
                 fontsize=10, fontweight="bold", pad=16)
    out2 = out.with_name("wc2026_model_vs_market_table.png")
    out2.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Wrote figure: {out2}")


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    p = argparse.ArgumentParser(description="Final model vs live market diff bar chart.")
    p.add_argument("--model-csv", type=Path,
                   default=PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines"
                   / "wc2026_final_mv_star_vs_market.csv")
    p.add_argument("--source", choices=["polymarket", "kalshi"], default="polymarket")
    p.add_argument("--api-url", default=DEFAULT_API)
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "mc_simu" / "figures" / "wc2026_model_vs_market.png")
    p.add_argument("--compare", action="store_true",
                   help="2-panel absolute-pp vs relative-%% (each sorted by its own metric)")
    p.add_argument("--table", action="store_true",
                   help="render per-team numbers (model%%, market%%, abs pp, rel %%) as a table figure")
    args = p.parse_args(argv)

    if not args.model_csv.exists():
        raise SystemExit(f"missing model CSV: {args.model_csv}")

    teams_48 = _required_teams()
    model = normalize({t: v for t, v in load_final_model(args.model_csv).items() if t in teams_48})

    banner(f"Fetch LIVE {args.source} via {args.api_url}/prices")
    src_key = "Polymarket" if args.source == "polymarket" else "Kalshi"
    prices = load_prices_endpoint(args.api_url)
    raw = {t: row[src_key] for t, row in prices.items() if src_key in row}
    mkt = normalize({t: v for t, v in raw.items() if t in teams_48})
    print(f"  {len(mkt)} teams from {args.source}")

    common = [t for t in model if t in mkt]

    if args.compare:
        _compare_figure(plt, model, mkt, common, args.out, args.source)
        return 0

    if args.table:
        _table_figure(plt, model, mkt, common, args.out, args.source)
        return 0

    rows = sorted(((t, (model[t] - mkt[t]) * 100,
                    (mkt[t] - model[t]) / model[t] * 100 if model[t] > 0 else None)
                   for t in common), key=lambda r: r[1], reverse=True)

    fig, ax = plt.subplots(figsize=(9, 12))
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows],
            color=[_colour(r[1]) for r in rows], edgecolor="black", linewidth=0.4)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    diffs = [r[1] for r in rows]
    ax.set_xlim(min(diffs) - 1.6, max(diffs) + 1.2)  # room for value labels
    for i, (_, dpp, rel) in enumerate(rows):
        rel_s = f"  ({rel:+.0f}%)" if rel is not None and abs(dpp) >= 0.3 else ""
        ax.annotate(f"{dpp:+.2f}{rel_s}",
                    (dpp, i), fontsize=6.5, va="center",
                    ha="left" if dpp >= 0 else "right",
                    xytext=(3 if dpp >= 0 else -3, 0), textcoords="offset points")
    src_label = args.source.capitalize()
    ax.set_xlabel(f"MC final model (Elo+MV+star) − {src_label}  (pp)")
    ax.set_title(f"WC2026 — current final model vs LIVE {src_label}\n"
                 "Negative = model below market | Positive = model above  "
                 "(rel % in parentheses)", fontsize=11, fontweight="bold")
    legend = [Patch(facecolor="#1a6b1a", label="higher >2pp"),
              Patch(facecolor="#3ca33c", label="higher 1-2pp"),
              Patch(facecolor="#a6d8a6", label="higher 0-1pp"),
              Patch(facecolor="#f2c14e", label="lower 0-1pp"),
              Patch(facecolor="#e8862e", label="lower 1-2pp"),
              Patch(facecolor="#cc3b2f", label="lower >2pp")]
    ax.legend(handles=legend, title="Model vs market", fontsize=8, loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Wrote figure: {args.out}")
    return 0


__all__ = ["load_final_model", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
