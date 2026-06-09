"""Phase-1 empirical audit: does the underdog's W rate fall short of Elo expectation
more severely in later WC stages?

Pre-registered hypothesis (Sam meeting 2026-05-27):
    H1: actual_underdog_winrate < expected_underdog_winrate (from Elo),
        AND the shortfall grows monotonically from Group → R16 → QF → SF → Final.

Pre-registered decision rule (anti-data-mining):
    - Test ONE statistic per stage: mean(actual - expected) for underdog matches.
    - Signal = "shortfall grows" → flag for Phase-3 stage-decay implementation.
    - No signal → close scope, do NOT implement.
    - Sample size is tiny in late KO rounds (~5-10 matches/stage × 5 WCs).
      Pool R16+QF and SF+Final for power. Report both raw + pooled.
    - No multiple-testing correction needed: 1 hypothesis pre-registered.

Inputs: data/mc_simu/results.csv + data/mc_simu/elo_history.csv
Output: stdout table (no file artifact — reproducible from code).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "mc_simu"


# 32-team WC stage layout (2002-2022 inclusive). 64 matches / WC.
STAGE_COUNTS = [
    ("Group", 48),
    ("R16", 8),
    ("QF", 4),
    ("SF", 2),
    ("3rd", 1),
    ("Final", 1),
]


def assign_stages(year_matches: pd.DataFrame) -> pd.Series:
    """Assign stage labels by sorting matches by date and using fixed cumulative counts.

    Caveat: WC matches within a stage can span 4-7 days; assignment is by RANK not by
    absolute date gap. For 2002-2022 (32-team format) the count-per-stage is fixed, so
    sorting by date and assigning the first 48 to Group, next 8 to R16, etc. is exact.
    """
    sorted_idx = year_matches.sort_values("date").index
    labels = []
    for stage, n in STAGE_COUNTS:
        labels.extend([stage] * n)
    if len(labels) != len(sorted_idx):
        raise ValueError(
            f"WC year has {len(sorted_idx)} matches, expected {sum(n for _, n in STAGE_COUNTS)}"
        )
    out = pd.Series(index=sorted_idx, dtype=object)
    for idx, label in zip(sorted_idx, labels, strict=True):
        out[idx] = label
    return out


def build_elo_lookup(elo: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-team sorted (date, rating_after) for fast asof lookup."""
    by_team = {}
    for team, sub in elo.groupby("team"):
        by_team[team] = sub.sort_values("date")[["date", "rating_after"]].reset_index(drop=True)
    return by_team


def elo_before(lookup: dict, team: str, match_date: pd.Timestamp) -> float | None:
    """Return rating_after of team's most recent match strictly before match_date."""
    sub = lookup.get(team)
    if sub is None or sub.empty:
        return None
    pos = sub["date"].searchsorted(match_date, side="left") - 1
    if pos < 0:
        return None
    return float(sub.iloc[pos]["rating_after"])


def expected_winrate(elo_a: float, elo_b: float) -> float:
    """Standard Elo: P(A beats B). 2-way (treats draws as 0.5 contribution naturally)."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def actual_winrate(
    home_score: int,
    away_score: int,
    focal_is_home: bool,
    pk_winner: str | None = None,
    home_team: str | None = None,
) -> float:
    """Focal team's 2-way win share (W=1, D=0.5, L=0). Matches Elo's 2-way expected.

    KO matches that ended on penalty kicks are stored in results.csv with the
    post-ET regulation score (which is a draw). Without consulting shootouts.csv,
    the draw would be coded as 0.5/0.5, but in KO context one team actually
    advances. If pk_winner is supplied (from shootouts.csv) for a tied scoreline,
    use it as the true outcome instead.
    """
    if home_score == away_score:
        if pk_winner is None:
            return 0.5
        if focal_is_home:
            return 1.0 if pk_winner == home_team else 0.0
        return 0.0 if pk_winner == home_team else 1.0
    if focal_is_home:
        return 1.0 if home_score > away_score else 0.0
    return 1.0 if away_score > home_score else 0.0


def build_pk_lookup(shootouts: pd.DataFrame) -> dict:
    """Map (date, home_team, away_team) -> winner from shootouts.csv."""
    out = {}
    for _, row in shootouts.iterrows():
        out[(row["date"], row["home_team"], row["away_team"])] = row["winner"]
    return out


def main():
    results = pd.read_csv(DATA / "results.csv")
    elo = pd.read_csv(DATA / "elo_history.csv")
    shootouts = pd.read_csv(DATA / "shootouts.csv")
    results["date"] = pd.to_datetime(results["date"])
    elo["date"] = pd.to_datetime(elo["date"])
    shootouts["date"] = pd.to_datetime(shootouts["date"])
    pk_lookup = build_pk_lookup(shootouts)

    wc = results[results["tournament"] == "FIFA World Cup"].copy()
    wc["year"] = wc["date"].dt.year
    wc = wc[(wc["year"] >= 2002) & (wc["year"] <= 2022)].copy()
    print(f"[setup] WC matches 2002-2022: {len(wc)}")

    # Assign stages per year
    stage_col = pd.Series(index=wc.index, dtype=object)
    for year, sub in wc.groupby("year"):
        stage_col.loc[sub.index] = assign_stages(sub)
    wc["stage"] = stage_col

    # Drop 3rd-place playoff (not a real "stage decay" signal — both losers, low stakes)
    wc = wc[wc["stage"] != "3rd"].copy()
    print(f"[setup] After dropping 3rd-place playoff: {len(wc)}")

    # Lookup Elo before match
    elo_lookup = build_elo_lookup(elo)
    home_elo, away_elo = [], []
    for _, row in wc.iterrows():
        home_elo.append(elo_before(elo_lookup, row["home_team"], row["date"]))
        away_elo.append(elo_before(elo_lookup, row["away_team"], row["date"]))
    wc["elo_home"] = home_elo
    wc["elo_away"] = away_elo
    missing = wc["elo_home"].isna() | wc["elo_away"].isna()
    print(f"[setup] Matches with missing Elo (drop): {missing.sum()}")
    wc = wc[~missing].copy()
    print(f"[setup] Final n: {len(wc)}")

    # Identify underdog (lower Elo) per match
    wc["elo_underdog"] = wc[["elo_home", "elo_away"]].min(axis=1)
    wc["elo_favorite"] = wc[["elo_home", "elo_away"]].max(axis=1)
    wc["elo_gap"] = wc["elo_favorite"] - wc["elo_underdog"]
    wc["underdog_is_home"] = wc["elo_home"] < wc["elo_away"]

    # Skip matches where elo_home == elo_away (no underdog defined). Rare but possible.
    even = (wc["elo_home"] == wc["elo_away"])
    if even.any():
        print(f"[setup] Dropping {even.sum()} even-Elo matches")
        wc = wc[~even].copy()

    # Expected vs actual underdog winrate (2-way)
    wc["expected_W"] = wc.apply(
        lambda r: expected_winrate(r["elo_underdog"], r["elo_favorite"]), axis=1
    )
    wc["pk_winner"] = wc.apply(
        lambda r: pk_lookup.get((r["date"], r["home_team"], r["away_team"])), axis=1
    )
    wc["actual_W"] = wc.apply(
        lambda r: actual_winrate(
            r["home_score"], r["away_score"], r["underdog_is_home"],
            pk_winner=r["pk_winner"], home_team=r["home_team"],
        ),
        axis=1,
    )
    n_pk_resolved = wc["pk_winner"].notna().sum()
    print(f"[setup] KO matches resolved via PK lookup: {n_pk_resolved}")
    wc["gap"] = wc["actual_W"] - wc["expected_W"]  # >0 underdog overperforms, <0 underperforms

    # ── Pre-registered analysis: mean gap per stage ──
    print()
    print("=" * 80)
    print("PRE-REGISTERED TEST: mean(actual_W - expected_W) for underdog per stage")
    print("Negative = underdog underperforms vs Elo expectation")
    print("=" * 80)
    print(f"{'Stage':<10} {'n':>5} {'mean gap':>10} {'std':>8} {'SE':>8} "
          f"{'95% CI':>20} {'actual W%':>10} {'expected W%':>12}")
    rows = []
    for stage, _ in STAGE_COUNTS:
        if stage == "3rd":
            continue
        sub = wc[wc["stage"] == stage]
        if len(sub) == 0:
            continue
        n = len(sub)
        mean = sub["gap"].mean()
        std = sub["gap"].std()
        se = std / np.sqrt(n) if n > 1 else float("nan")
        ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
        rows.append({
            "stage": stage, "n": n, "mean_gap": mean,
            "actual_W_pct": sub["actual_W"].mean() * 100,
            "expected_W_pct": sub["expected_W"].mean() * 100,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
        })
        print(f"{stage:<10} {n:>5d} {mean:>+10.4f} {std:>8.4f} {se:>8.4f} "
              f"[{ci_lo:>+6.3f}, {ci_hi:>+6.3f}]  "
              f"{sub['actual_W'].mean()*100:>9.1f}% {sub['expected_W'].mean()*100:>11.1f}%")

    # ── Pooled view (power) ──
    print()
    print("POOLED for power:")
    pool_defs = [
        ("Group", ["Group"]),
        ("R16+QF", ["R16", "QF"]),
        ("SF+Final", ["SF", "Final"]),
    ]
    for label, stages in pool_defs:
        sub = wc[wc["stage"].isin(stages)]
        n = len(sub)
        if n == 0:
            continue
        mean = sub["gap"].mean()
        se = sub["gap"].std() / np.sqrt(n) if n > 1 else float("nan")
        ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
        print(f"  {label:<10} n={n:>4d}  mean gap = {mean:>+.4f}  "
              f"95% CI [{ci_lo:>+.3f}, {ci_hi:>+.3f}]  "
              f"actual {sub['actual_W'].mean()*100:.1f}%  expected {sub['expected_W'].mean()*100:.1f}%")

    # ── Decision rule output ──
    print()
    print("=" * 80)
    print("DECISION RULE (pre-registered):")
    print("  Signal IF (a) pooled-late gap < -0.05 (5pp underperformance), AND")
    print("            (b) 95% CI excludes 0 for late pool, AND")
    print("            (c) monotone decay across Group -> R16+QF -> SF+Final.")
    print("=" * 80)
    group_gap = wc[wc["stage"] == "Group"]["gap"].mean()
    early_ko_gap = wc[wc["stage"].isin(["R16", "QF"])]["gap"].mean()
    late_ko_gap = wc[wc["stage"].isin(["SF", "Final"])]["gap"].mean()
    monotone = (group_gap >= early_ko_gap >= late_ko_gap)
    late_sub = wc[wc["stage"].isin(["SF", "Final"])]
    late_n = len(late_sub)
    late_se = late_sub["gap"].std() / np.sqrt(late_n) if late_n > 1 else float("nan")
    late_ci_excludes_zero = (late_ko_gap + 1.96 * late_se) < 0
    print(f"  Group       gap = {group_gap:+.4f}")
    print(f"  R16+QF      gap = {early_ko_gap:+.4f}")
    print(f"  SF+Final    gap = {late_ko_gap:+.4f}   (n={late_n}, 95% upper bound = {late_ko_gap + 1.96 * late_se:+.4f})")
    print(f"  Monotone (Group >= R16+QF >= SF+Final)? {monotone}")
    print(f"  Late pool gap < -0.05? {late_ko_gap < -0.05}")
    print(f"  Late pool 95% CI excludes 0 (upper < 0)? {late_ci_excludes_zero}")
    signal = monotone and (late_ko_gap < -0.05) and late_ci_excludes_zero
    print(f"\n  >>> SIGNAL = {signal} <<<")
    if signal:
        print("  -> Implement stage-decay in Phase 3 (1-param linear shrinkage on underdog).")
    else:
        print("  -> NO signal per pre-registered rule. Default = close scope for point 2.")
        print("  -> Caveats: small n in late stages; absence of signal != absence of effect,")
        print("     but pre-registered threshold not met -> do not data-mine for alternative tests.")


if __name__ == "__main__":
    main()
