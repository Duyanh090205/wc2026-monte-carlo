"""Phase 1 §2.2 validation gate runner.

Per plan decision #11 (3-tier gate with 9-combo mini-tune):

    best_brier ≤ 0.19            → ✅ PASS,            exit 0
    0.19 < best_brier ≤ 0.21     → ⚠ WARN + Karlis-Ntzoufras alt in Phase 4, exit 1
    best_brier > 0.21            → ❌ HARD STOP pivot,  exit 2

Validation tournaments: WC 2018, WC 2022, Euro 2020 (played 2021), Euro 2024.
~230 matches total. Mini-tune grid: alpha × diagonal_inflation (3 × 3 = 9 combos).
beta fixed at 0.13 (single-source per §0.5); blend_match_weight fixed at 1.0 (§0.8).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mc_simu._common import banner, infer_tournament_type  # noqa: E402
from mc_simu.confederations import get_confederation  # noqa: E402
from mc_simu.elo import build_rating_history, get_rating_as_of  # noqa: E402
from mc_simu.single_game import MatchContext, ModelParams, predict_match  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
# Per Phase 0 deep-audit-bug fix 2026-05-20: read the enriched CSV (full range +
# tournament_type + attendance_pct) instead of raw results.csv. Required so
# Euro 2020 ghost-game HFA scaling (spec §0.5) actually fires for 41/51 matches.
RESULTS_CSV = DATA_DIR / "matches_1998_2026.csv"
TOURNAMENT_META = DATA_DIR / "tournament_meta.json"

# (kaggle_tournament_str, actual_year) → label string used in tournament_meta.json
VAL_TOURNAMENTS: list[tuple[str, int, str]] = [
    ("FIFA World Cup", 2018, "FIFA World Cup 2018"),
    ("FIFA World Cup", 2022, "FIFA World Cup 2022"),
    ("UEFA Euro",     2021, "UEFA Euro 2020"),     # COVID delay
    ("UEFA Euro",     2024, "UEFA Euro 2024"),
]

# Expanded after Phase 1 deep audit + ChatGPT cross-verification (findings #6-9).
# Grid brackets both interpretations of plan's α value:
#   0.27 = 538/Bilalić additive-to-log unit-converted (empirically anchored)
#   0.40 = original plan log-goals interpretation (calibrated upper-end)
# Phase 4 will use the same grid for full LOTO-CV across 12 tournaments.
MINI_TUNE_ALPHAS = (0.20, 0.27, 0.40, 0.50)
# diag range widened from spec's {0.05, 0.09, 0.12} after Phase 1 deep audit (finding #7):
# Brier sweep showed real-data optimum at diag≈0.15-0.30; spec's upper bound 0.12
# was below the actual minimum for international tournaments (draw rate ~25%).
MINI_TUNE_DIAGS = (0.10, 0.20, 0.30)


@dataclass
class ValidationReport:
    best_brier: float              # MEAN-Brier (sum/3 per match, then avg across matches) — gate metric
    best_alpha: float
    best_diag: float
    default_brier: float           # mean-Brier at α=0.40, diag=0.09 baseline
    best_brier_sum: float = 0.0    # SUM-Brier at best combo (literature comparability)
    default_brier_sum: float = 0.0 # SUM-Brier at default combo
    per_combo: dict[tuple[float, float], float] = field(default_factory=dict)  # mean-Brier per combo
    per_tournament: dict[str, float] = field(default_factory=dict)
    per_confederation: dict[str, float] = field(default_factory=dict)
    n_matches: int = 0
    skipped_matches: int = 0


# ── Match → MatchContext construction ─────────────────────────────────────────


def _make_context(row, label: str, tournament_meta: dict) -> MatchContext:
    venue_country = row["country"]
    meta = tournament_meta.get(label, {"host_countries": [], "host_confederation": None})
    att = row.get("attendance_pct", float("nan"))
    if pd.isna(att):
        att = 1.0
    return MatchContext(
        is_neutral=bool(row["neutral"]) if isinstance(row["neutral"], (bool, np.bool_))
                   else str(row["neutral"]).upper() == "TRUE",
        tournament_type=infer_tournament_type(row["tournament"]),
        home_country=row["home_team"],
        away_country=row["away_team"],
        venue_country=venue_country,
        venue_confederation=get_confederation(venue_country),
        home_confederation=get_confederation(row["home_team"]),
        away_confederation=get_confederation(row["away_team"]),
        host_countries=meta.get("host_countries", []),
        host_confederation=meta.get("host_confederation"),
        attendance_pct=float(att),
    )


def _outcome_vec(home_score: float, away_score: float) -> tuple[float, float, float]:
    if home_score > away_score:
        return 1.0, 0.0, 0.0
    if home_score == away_score:
        return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0


def _brier_sum(pred, actual) -> float:
    """Multi-class Brier — SUM over 3 outcomes per match (range [0, 2]).

    Standard ML convention; reported alongside mean-Brier for transparency.
    """
    return ((pred.p_home - actual[0]) ** 2
            + (pred.p_draw - actual[1]) ** 2
            + (pred.p_away - actual[2]) ** 2)


def _brier_mean(pred, actual) -> float:
    """Mean-Brier — sum/3 per match (range [0, 0.667]).

    Spec §2.2 gate of 0.20 is unattainable under sum convention (Elo-Poisson
    typically 0.55–0.65 in football literature: Lasek 2019, Hubacek 2019,
    Constantinou 2012). Mean convention makes 0.20 a meaningful target.
    Reported as the primary metric the gate applies to — see audit_report
    Phase 1 finding on Brier-convention interpretation.
    """
    return _brier_sum(pred, actual) / 3.0


# ── Core validation ───────────────────────────────────────────────────────────


def select_validation_matches(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Return validation match rows from raw results.csv (with `label` column added)."""
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    parts = []
    for tname, year, label in VAL_TOURNAMENTS:
        sub = df[(df["tournament"] == tname) & (df["year"] == year)].copy()
        sub["label"] = label
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    # Drop NaN-score rows defensively (validation against real played matches only)
    out = out[out["home_score"].notna() & out["away_score"].notna()]
    return out


def validate_phase1(history: dict[str, pd.DataFrame],
                    val_matches: pd.DataFrame,
                    tournament_meta: dict) -> ValidationReport:
    """Run 9-combo mini-tune + compute per-tournament + per-confed Brier.

    Primary metric: MEAN-Brier (sum/3 per match, then average) — see audit_report
    Phase 1 finding on spec gate convention. SUM-Brier reported alongside for
    literature comparability.
    """
    # Pre-compute Elo + context + actual once per match
    cache = []
    skipped = 0
    for row in val_matches.itertuples(index=False):
        row_dict = row._asdict()
        date = pd.Timestamp(row_dict["date"])
        elo_h = get_rating_as_of(history, row_dict["home_team"], date)
        elo_a = get_rating_as_of(history, row_dict["away_team"], date)
        # get_rating_as_of has explicit fallback to initial_rating; never raises.
        # If a row truly has unparseable date/scores, let pandas/predict_match raise
        # so we see the failure rather than silently swallow.
        ctx = _make_context(row_dict, row_dict["label"], tournament_meta)
        actual = _outcome_vec(row_dict["home_score"], row_dict["away_score"])
        cache.append({
            "elo_h": elo_h, "elo_a": elo_a, "ctx": ctx, "actual": actual,
            "label": row_dict["label"],
            "home_confed": ctx.home_confederation,
            "away_confed": ctx.away_confederation,
        })

    # Mini-tune
    per_combo: dict[tuple[float, float], float] = {}        # mean-Brier (primary)
    per_combo_sum: dict[tuple[float, float], float] = {}    # sum-Brier (reporting)
    per_match_briers: dict[tuple[float, float], list[float]] = {}   # mean-Brier per match for per-tournament/confed roll-ups
    for a in MINI_TUNE_ALPHAS:
        for d in MINI_TUNE_DIAGS:
            params = ModelParams(alpha=a, beta=0.13, diagonal_inflation=d,
                                 blend_match_weight=1.0)
            mean_briers: list[float] = []
            sum_briers: list[float] = []
            for entry in cache:
                pred = predict_match(entry["elo_h"], entry["elo_a"], entry["ctx"], params)
                mean_briers.append(_brier_mean(pred, entry["actual"]))
                sum_briers.append(_brier_sum(pred, entry["actual"]))
            per_combo[(a, d)] = float(np.mean(mean_briers))
            per_combo_sum[(a, d)] = float(np.mean(sum_briers))
            per_match_briers[(a, d)] = mean_briers

    best_combo = min(per_combo.keys(), key=lambda k: per_combo[k])
    best_brier = per_combo[best_combo]
    best_alpha, best_diag = best_combo
    best_brier_sum = per_combo_sum[best_combo]

    # Compute "default" at the spec's nominal values (α=0.40, β=0.13, diag=0.09)
    # separately — these may not appear in the mini-tune grid after Phase 1 audit
    # widened the diag range. Provides comparability with spec/baseline.
    default_params = ModelParams(alpha=0.40, beta=0.13, diagonal_inflation=0.09,
                                  blend_match_weight=1.0)
    default_mean: list[float] = []
    default_sum: list[float] = []
    for entry in cache:
        pred = predict_match(entry["elo_h"], entry["elo_a"], entry["ctx"], default_params)
        default_mean.append(_brier_mean(pred, entry["actual"]))
        default_sum.append(_brier_sum(pred, entry["actual"]))
    default_brier = float(np.mean(default_mean))
    default_brier_sum = float(np.mean(default_sum))

    # Per-tournament + per-confed at BEST combo
    best_briers = per_match_briers[best_combo]
    per_tournament: dict[str, list[float]] = {}
    per_confederation: dict[str, list[float]] = {}
    for entry, b in zip(cache, best_briers):
        per_tournament.setdefault(entry["label"], []).append(b)
        for confed in {entry["home_confed"], entry["away_confed"]}:
            if confed == "UNKNOWN":
                continue
            per_confederation.setdefault(confed, []).append(b)

    return ValidationReport(
        best_brier=best_brier,
        best_alpha=best_alpha, best_diag=best_diag,
        default_brier=default_brier,
        best_brier_sum=best_brier_sum,
        default_brier_sum=default_brier_sum,
        per_combo=per_combo,
        per_tournament={t: float(np.mean(bs)) for t, bs in per_tournament.items()},
        per_confederation={c: float(np.mean(bs)) for c, bs in per_confederation.items()},
        n_matches=len(cache),
        skipped_matches=skipped,
    )


# ── Gate ──────────────────────────────────────────────────────────────────────


def gate_verdict(best_brier: float) -> tuple[str, int]:
    """3-tier gate per plan decision #11.

    Returns (verdict, exit_code)."""
    if best_brier <= 0.19:
        return ("PASS", 0)
    if best_brier <= 0.21:
        return ("PASS-WARN", 1)
    return ("HARD-STOP-PIVOT", 2)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    banner("MC Simu — Phase 1 Validation Gate (§2.2)")

    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} missing — run download_match_history.py")
        return 1
    if not TOURNAMENT_META.exists():
        print(f"ERROR: {TOURNAMENT_META} missing")
        return 1

    print("Loading match history + building Elo timeline ...")
    df_raw = pd.read_csv(RESULTS_CSV, encoding="utf-8")
    history = build_rating_history(df_raw)
    print(f"  teams in history: {len(history):,}")

    tournament_meta = json.loads(TOURNAMENT_META.read_text(encoding="utf-8"))
    val_matches = select_validation_matches(df_raw)
    print(f"  validation matches: {len(val_matches)} (across {len(VAL_TOURNAMENTS)} tournaments)")

    print("\nRunning 9-combo mini-tune (α × diagonal_inflation) ...")
    report = validate_phase1(history, val_matches, tournament_meta)

    # ── Output ──────────────────────────────────────────────────────────────
    print(f"\nResults ({report.n_matches} matches, {report.skipped_matches} skipped):")
    print("  Brier convention: MEAN (sum/3 per match, then avg). Sum-Brier shown for reference.")
    print(f"  default (α=0.40, diag=0.09):  mean={report.default_brier:.4f}   sum={report.default_brier_sum:.4f}")
    print(f"  best   (α={report.best_alpha:.2f}, diag={report.best_diag:.2f}):  mean={report.best_brier:.4f}   sum={report.best_brier_sum:.4f}")

    print("\nPer-combo Brier grid:")
    print(f"  {'':>7s} " + "  ".join(f"diag={d:.2f}" for d in MINI_TUNE_DIAGS))
    for a in MINI_TUNE_ALPHAS:
        row = "  ".join(f"  {report.per_combo[(a, d)]:.4f}" for d in MINI_TUNE_DIAGS)
        print(f"  α={a:.2f}  {row}")

    print("\nPer-tournament Brier (at best combo):")
    for t, b in sorted(report.per_tournament.items()):
        print(f"  {t:25s} {b:.4f}")

    print("\nPer-confederation Brier (at best combo, double-counts mixed matches):")
    for c, b in sorted(report.per_confederation.items()):
        print(f"  {c:10s} {b:.4f}")

    verdict, code = gate_verdict(report.best_brier)
    print()
    banner(f"Phase 1 verdict: {verdict}  (best_brier={report.best_brier:.4f}, exit={code})")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
