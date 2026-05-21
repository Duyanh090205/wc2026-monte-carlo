"""Score Phase 3 baseline CSVs against actual historical outcomes.

Phase 3 §3.10 — preliminary head-to-head before Phase 4 LOTO-CV tuning.

Compares 3 models (self-Elo, eloratings, baseline 40/40/20) across all 11
historical editions with known champions (WC 2002/06/10/14/18/22, Euro
2004/08/12/2020/24). Reports two metrics per (edition, model):

  log P(actual_champion)  — proper scoring rule (higher = better)
  rank(actual_champion)   — easier-to-interpret tournament-level metric

WC 2026 is intentionally excluded from scoring (no actual outcome). Its
predictions are summarized separately for the Polymarket sanity table.

Usage:
    python -m mc_simu.historical_replay
        → writes data/mc_simu/phase3_baselines/replay_scores.csv
                + Markdown report to stdout

Note: this is a PRELIMINARY scorer using untuned default params (α=0.27,
β=0.09, diag=0.20, D=1400 — Phase 1 calibration). Phase 4 LOTO-CV will
re-score after PARAM_GRID tuning; expect ±0.005-0.01 Brier improvement +
possibly reordering between close (model A, model B) pairs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mc_simu._common import banner  # noqa: E402


# Actual historical champions for each scoring edition.
# WC2026 deliberately omitted — no outcome yet (sanity-only via Polymarket).
ACTUAL_CHAMPIONS: dict[str, str] = {
    # WC 8-group editions
    "wc2002": "Brazil",
    "wc2006": "Italy",
    "wc2010": "Spain",
    "wc2014": "Germany",
    "wc2018": "France",
    "wc2022": "Argentina",
    # Euro 24-team editions (2016 deferred per Phase 3 decision #38)
    "euro2020": "Italy",       # played 2021 due to COVID
    "euro2024": "Spain",
    # Euro 16-team editions
    "euro2004": "Greece",      # famous upset
    "euro2008": "Spain",
    "euro2012": "Spain",
}

# Models to score (filename suffix → display name)
MODELS: list[tuple[str, str]] = [
    ("self",       "self-Elo"),
    ("eloratings", "eloratings"),
    ("baseline",   "baseline 40/40/20"),
]


# ── Loaders ──────────────────────────────────────────────────────────────────


def load_champion_probs(csv_path: Path) -> dict[str, float]:
    """Return {team: mc_fair_prob} for champion market in one CSV."""
    out: dict[str, float] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["market_type"] == "champion":
                out[row["team"]] = float(row["mc_fair_prob"])
    return out


def find_csv(baselines_dir: Path, event: str, model_tag: str, n: int, seed: int) -> Path | None:
    """Resolve canonical baseline CSV path; return None if missing."""
    p = baselines_dir / f"mc_{event}_{model_tag}_n{n}_seed{seed}.csv"
    return p if p.exists() else None


# ── Scoring ──────────────────────────────────────────────────────────────────


# Floor for log-prob to avoid -infinity when model assigned 0 to actual champion.
# 1e-5 ≈ "1 in 100,000 iter" — softer than 1/n for typical n=10000 to keep the
# diagnostic interpretable.
LOG_P_FLOOR = 1e-5


def score_edition(probs: dict[str, float], actual: str) -> tuple[float, int, float]:
    """Return (log_p_actual, rank_actual, prob_actual) for one (edition, model)."""
    p_actual = probs.get(actual, 0.0)
    log_p = math.log(max(p_actual, LOG_P_FLOOR))
    # Rank: 1 = highest-prob team. Teams not in dict get rank = len+1.
    sorted_teams = sorted(probs.items(), key=lambda kv: -kv[1])
    rank = next((i + 1 for i, (t, _) in enumerate(sorted_teams) if t == actual), len(sorted_teams) + 1)
    return log_p, rank, p_actual


# ── Report builders ──────────────────────────────────────────────────────────


def build_replay_table(baselines_dir: Path, n: int, seed: int) -> list[dict]:
    """One row per (edition, model). Includes log_p, rank, prob, missing CSV flag."""
    rows: list[dict] = []
    for event, actual in ACTUAL_CHAMPIONS.items():
        for model_tag, model_display in MODELS:
            csv_path = find_csv(baselines_dir, event, model_tag, n, seed)
            if csv_path is None:
                rows.append({
                    "edition": event, "actual_champion": actual,
                    "model": model_display,
                    "log_p_actual": None, "rank_actual": None, "prob_actual": None,
                    "csv": "MISSING",
                })
                continue
            probs = load_champion_probs(csv_path)
            log_p, rank, p_actual = score_edition(probs, actual)
            rows.append({
                "edition": event, "actual_champion": actual,
                "model": model_display,
                "log_p_actual": round(log_p, 4),
                "rank_actual": rank,
                "prob_actual": round(p_actual, 4),
                "csv": csv_path.name,
            })
    return rows


def summarize_by_model(rows: list[dict]) -> list[dict]:
    """Aggregate per-model: sum/mean log P, mean rank, # editions where #1, hit@3."""
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        if r["log_p_actual"] is None:
            continue
        by_model.setdefault(r["model"], []).append(r)

    summary = []
    for model, entries in by_model.items():
        n_ed = len(entries)
        sum_log_p = sum(e["log_p_actual"] for e in entries)
        mean_rank = sum(e["rank_actual"] for e in entries) / n_ed
        n_top1 = sum(1 for e in entries if e["rank_actual"] == 1)
        n_top3 = sum(1 for e in entries if e["rank_actual"] <= 3)
        n_top5 = sum(1 for e in entries if e["rank_actual"] <= 5)
        summary.append({
            "model": model,
            "n_editions": n_ed,
            "sum_log_p": round(sum_log_p, 3),
            "mean_log_p": round(sum_log_p / n_ed, 4),
            "mean_rank": round(mean_rank, 2),
            "n_top1": n_top1,
            "n_top3": n_top3,
            "n_top5": n_top5,
            "top3_rate": f"{n_top3}/{n_ed}",
        })
    summary.sort(key=lambda r: -r["sum_log_p"])  # best-scoring first
    return summary


def winner_per_edition(rows: list[dict]) -> list[dict]:
    """For each edition, the model that gave the highest prob to the actual champion."""
    by_ed: dict[str, list[dict]] = {}
    for r in rows:
        if r["log_p_actual"] is None:
            continue
        by_ed.setdefault(r["edition"], []).append(r)
    out = []
    for ed, entries in by_ed.items():
        actual = entries[0]["actual_champion"]
        best = max(entries, key=lambda e: e["log_p_actual"])
        out.append({
            "edition": ed,
            "actual_champion": actual,
            "winner_model": best["model"],
            "winner_log_p": best["log_p_actual"],
            "winner_prob": best["prob_actual"],
            "winner_rank": best["rank_actual"],
        })
    return out


# ── Markdown rendering ───────────────────────────────────────────────────────


def render_markdown(rows: list[dict], summary: list[dict], per_ed: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Historical replay — preliminary head-to-head (untuned)\n")
    lines.append("**WARNING — PRELIMINARY RESULTS WITH UNTUNED DEFAULT PARAMS.** Phase 1 calibrated values (α=0.27, β=0.09, diag=0.20, D=1400) used as-is. Phase 4 LOTO-CV will re-score after PARAM_GRID tuning; expect ±0.005-0.01 Brier improvement and possible reordering of close (model A, model B) pairs.\n")
    lines.append("Scoring rule: `log P(actual_champion)` per edition. Higher (less negative) = better. Floor = log(1e-5) ≈ -11.5 to handle teams the model assigned 0%.\n")

    # Summary table
    lines.append("## Aggregate scores (11 editions)\n")
    lines.append("| Model | Σ log P | Mean log P | Mean rank | #1 | Top-3 | Top-5 |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in summary:
        lines.append(
            f"| {s['model']} | {s['sum_log_p']:+.2f} | {s['mean_log_p']:+.3f} | "
            f"{s['mean_rank']:.2f} | {s['n_top1']}/{s['n_editions']} | "
            f"{s['n_top3']}/{s['n_editions']} | {s['n_top5']}/{s['n_editions']} |"
        )
    lines.append("")

    # Per-edition winner table
    lines.append("## Winning model per edition\n")
    lines.append("| Edition | Actual | Winner model | P(actual) | Rank |")
    lines.append("|---|---|---|---|---|")
    for e in per_ed:
        lines.append(
            f"| {e['edition']} | {e['actual_champion']} | {e['winner_model']} | "
            f"{e['winner_prob']:.3f} | #{e['winner_rank']} |"
        )
    lines.append("")

    # Per-edition detail (all models)
    lines.append("## Full replay (all models, all editions)\n")
    lines.append("| Edition | Actual | Model | P(actual) | Rank | log P |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        if r["log_p_actual"] is None:
            lines.append(f"| {r['edition']} | {r['actual_champion']} | {r['model']} | — | — | MISSING |")
        else:
            lines.append(
                f"| {r['edition']} | {r['actual_champion']} | {r['model']} | "
                f"{r['prob_actual']:.3f} | #{r['rank_actual']} | {r['log_p_actual']:+.2f} |"
            )
    return "\n".join(lines)


# ── Driver ──────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score Phase 3 baselines against actuals.")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baselines-dir", type=Path,
        default=PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines",
    )
    parser.add_argument(
        "--out-csv", type=Path,
        default=PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines" / "replay_scores.csv",
    )
    parser.add_argument(
        "--out-md", type=Path,
        default=PROJECT_ROOT / "data" / "mc_simu" / "phase3_baselines" / "replay_report.md",
    )
    args = parser.parse_args(argv)

    banner(f"Historical replay scoring — n={args.n:,} seed={args.seed}")

    rows = build_replay_table(args.baselines_dir, args.n, args.seed)
    summary = summarize_by_model(rows)
    per_ed = winner_per_edition(rows)

    # Write CSV (one row per (edition, model))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote per-row scores: {args.out_csv}")

    # Write Markdown
    md = render_markdown(rows, summary, per_ed)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"Wrote report: {args.out_md}\n")

    # Echo summary to console
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
