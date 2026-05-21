"""Cross-path audit: Phase 1 self-Elo source vs eloratings.net source.

Looks for integration-layer wiring bugs that will surface when Phase 3/4 plugs
the eloratings CSV into get_rating_as_of() or similar consumers.

Per /deep-audit-bug skill — Step 5 + 6. Empirical evidence collection, not
speculation.

Run: python tests/audit_phase1_wiring.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from mc_simu._common import infer_tournament_type  # noqa: E402
from mc_simu.elo import build_rating_history, get_rating_as_of  # noqa: E402
from mc_simu.eloratings_client import classify_tournament  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
MATCHES_CSV = DATA_DIR / "matches_1998_2026.csv"
ELORATINGS_CSV = DATA_DIR / "elo_history_eloratings.csv"


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# ── BUG 1 — Date type silently breaks get_rating_as_of ─────────────────────────
def audit_date_type_wiring() -> dict:
    """If a future caller loads elo_history_eloratings.csv WITHOUT parse_dates,
    df["date"] is string. get_rating_as_of compares pd.Timestamp < string → ???
    """
    banner("BUG 1 — Date type wiring")

    # Load eloratings WITHOUT parse_dates (the natural / default way)
    df_default = pd.read_csv(ELORATINGS_CSV)
    print(f"  Default read_csv: df['date'].dtype = {df_default['date'].dtype}")
    sample_val = df_default["date"].iloc[0]
    print(f"  First value: {sample_val!r} (type: {type(sample_val).__name__})")

    # Build history dict in the natural way (groupby team)
    history_default = {team: g for team, g in df_default.groupby("team")}

    # Call get_rating_as_of with a real Timestamp
    target_date = pd.Timestamp("2018-06-13")
    try:
        rating = get_rating_as_of(history_default, "Spain", target_date)
        print(f"  get_rating_as_of(Spain, 2018-06-13) = {rating}")
        # Now load WITH parse_dates and compare
        df_parsed = pd.read_csv(ELORATINGS_CSV, parse_dates=["date"])
        history_parsed = {team: g for team, g in df_parsed.groupby("team")}
        rating_parsed = get_rating_as_of(history_parsed, "Spain", target_date)
        print(f"  get_rating_as_of(Spain, 2018-06-13) [parse_dates]  = {rating_parsed}")
        if abs(rating - rating_parsed) > 0.01:
            return {
                "status": "FAIL",
                "max_diff": abs(rating - rating_parsed),
                "default_call": rating,
                "parsed_call": rating_parsed,
                "explanation": "Default read_csv yields string dates → wrong comparison vs Timestamp",
            }
        return {"status": "PASS", "note": "Default + parsed match"}
    except Exception as e:
        return {
            "status": "FAIL-EXCEPTION",
            "exception": f"{type(e).__name__}: {e}",
            "explanation": "Default read_csv (no parse_dates) raises in get_rating_as_of",
        }


# ── BUG 2 — Team name mismatch (Kaggle vs eloratings conventions) ──────────────
def audit_team_name_wiring() -> dict:
    """If a caller has Kaggle names ('Czech Republic') but history is from
    eloratings ('Czechia'), get_rating_as_of returns initial_rating silently.
    """
    banner("BUG 2 — Team name convention mismatch")

    df = pd.read_csv(ELORATINGS_CSV, parse_dates=["date"])
    elor_teams = set(df["team"].unique())
    matches = pd.read_csv(MATCHES_CSV)
    kaggle_teams = set(matches["home_team"]) | set(matches["away_team"])

    only_in_kaggle = kaggle_teams - elor_teams
    only_in_elor = elor_teams - kaggle_teams

    # Find suspect pairs (Kaggle name vs eloratings name for SAME country)
    suspects = []
    known_aliases = {
        "Czech Republic": "Czechia",
        "South Korea": "Korea",
        "North Korea": "Korea DPR",
        "United States": "USA",
        "Türkiye": "Turkey",
        "Cape Verde": "Cape Verde Islands",
        "Ivory Coast": "Cote d'Ivoire",
        "DR Congo": "Congo DR",
    }
    for kag, elo in known_aliases.items():
        if kag in kaggle_teams and elo in elor_teams:
            suspects.append((kag, elo))

    print(f"  Kaggle-only teams: {len(only_in_kaggle)} (sample: {sorted(only_in_kaggle)[:5]})")
    print(f"  eloratings-only teams: {len(only_in_elor)} (sample: {sorted(only_in_elor)[:5]})")
    print(f"  Known alias pairs (will silently fall back to 1500):")
    for kag, elo in suspects:
        print(f"    Kaggle '{kag}'  ↔  eloratings '{elo}'")

    # Concrete simulation: caller passes Kaggle name to eloratings-loaded history
    history_elor = {team: g for team, g in df.groupby("team")}
    target_date = pd.Timestamp("2022-12-01")  # before WC2022
    bug_triggered = []
    for kag, elo in suspects:
        rating_via_kaggle_name = get_rating_as_of(history_elor, kag, target_date)
        rating_via_elor_name = get_rating_as_of(history_elor, elo, target_date)
        if rating_via_kaggle_name == 1500.0 and rating_via_elor_name != 1500.0:
            bug_triggered.append({
                "kaggle_name": kag,
                "elor_name": elo,
                "wrong_lookup_rating": rating_via_kaggle_name,
                "correct_lookup_rating": rating_via_elor_name,
            })

    if bug_triggered:
        print(f"\n  ⚠ Triggered silent fallback bug on {len(bug_triggered)} teams:")
        for b in bug_triggered:
            print(f"    {b['kaggle_name']:25s} → {b['wrong_lookup_rating']} "
                  f"(should be ≈ {b['correct_lookup_rating']:.0f} via '{b['elor_name']}')")
        return {"status": "FAIL", "n_triggered": len(bug_triggered), "examples": bug_triggered[:5]}
    return {"status": "PASS", "note": "All suspect pairs resolved correctly"}


# ── BUG 3 — Tournament classification disagreement ─────────────────────────────
def audit_tournament_classification() -> dict:
    """For matches present in BOTH sources, do they get the same K-factor?

    Self-compute classifies via infer_tournament_type(long_kaggle_string).
    eloratings classifies via classify_tournament(short_code).
    Independent logic — likely diverges on some categories.
    """
    banner("BUG 3 — Tournament classification asymmetry")

    df_elor = pd.read_csv(ELORATINGS_CSV, parse_dates=["date"])
    df_matches = pd.read_csv(MATCHES_CSV, parse_dates=["date"])

    # Inner-join on (date, home, away) — but team-name conventions differ →
    # use date alone as a coarse proxy, count by classification distribution.
    print(f"  Self-compute (Kaggle) tournament_type distribution:")
    self_counts = df_matches["tournament_type"].value_counts() if "tournament_type" in df_matches.columns else None
    if self_counts is None:
        # derive on the fly
        self_counts = df_matches["tournament"].apply(infer_tournament_type).value_counts()
    print(self_counts.to_string())

    print(f"\n  eloratings tournament_type distribution:")
    elor_counts = df_elor["tournament_type"].value_counts()
    print(elor_counts.to_string())

    # Distributional comparison
    all_cats = set(self_counts.index) | set(elor_counts.index)
    diffs = []
    for cat in sorted(all_cats):
        s = self_counts.get(cat, 0)
        e = elor_counts.get(cat, 0)
        diffs.append({"category": cat, "self": s, "elor": e, "ratio_e_over_s": (e / s) if s else float("inf")})

    print(f"\n  Side-by-side counts:")
    print(f"  {'category':22s} {'self':>8s} {'elor':>8s} {'elor/self':>10s}")
    for d in diffs:
        ratio = f"{d['ratio_e_over_s']:.2f}" if d['ratio_e_over_s'] != float("inf") else "inf"
        print(f"  {d['category']:22s} {d['self']:>8d} {d['elor']:>8d} {ratio:>10s}")

    # Find the most asymmetric bucket
    finite = [d for d in diffs if d["ratio_e_over_s"] != float("inf") and d["self"] > 100]
    finite.sort(key=lambda d: abs(d["ratio_e_over_s"] - 1))
    worst = finite[-1] if finite else None
    if worst and abs(worst["ratio_e_over_s"] - 1) > 0.5:
        return {"status": "WARN", "worst_category": worst}
    return {"status": "PASS-WARN", "note": "Distributions broadly similar"}


# ── BUG 4 — Schema column compatibility ────────────────────────────────────────
def audit_schema_compat() -> dict:
    """Verify eloratings CSV schema is drop-in compat with history_to_long_df."""
    banner("BUG 4 — Schema compat")
    df_elor = pd.read_csv(ELORATINGS_CSV)

    # Build a tiny self history and flatten
    matches_tiny = pd.read_csv(MATCHES_CSV).head(50)
    from mc_simu.elo import history_to_long_df
    history_self = build_rating_history(matches_tiny)
    df_self_long = history_to_long_df(history_self)

    elor_cols = set(df_elor.columns)
    self_cols = set(df_self_long.columns)
    print(f"  eloratings cols: {sorted(elor_cols)}")
    print(f"  self_long cols:  {sorted(self_cols)}")
    missing_in_elor = self_cols - elor_cols
    extra_in_elor = elor_cols - self_cols
    if missing_in_elor:
        print(f"  ⚠ Missing in eloratings: {missing_in_elor}")
    if extra_in_elor:
        print(f"  ⚠ Extra in eloratings: {extra_in_elor}")
    return {"status": "PASS" if not missing_in_elor else "FAIL",
            "missing_in_elor": list(missing_in_elor),
            "extra_in_elor": list(extra_in_elor)}


# ── BUG 5 — Coverage gap (historical teams absent from eloratings World.tsv) ───
def audit_historical_team_coverage() -> dict:
    """Teams that played in 12 LOTO-CV tournaments but are now defunct
    (Yugoslavia, Serbia and Montenegro, Czechoslovakia) — eloratings.net may
    have them under different codes, or not at all.

    Phase 4 historical replay (§5.3) will need ratings for these teams as of
    pre-tournament dates. Silent fallback to 1500 would distort replay scores.
    """
    banner("BUG 5 — Historical team coverage for LOTO-CV replay")

    df_matches = pd.read_csv(MATCHES_CSV, parse_dates=["date"])
    df_elor = pd.read_csv(ELORATINGS_CSV, parse_dates=["date"])
    elor_teams = set(df_elor["team"].unique())

    # Find teams that played in 12 LOTO-CV tournaments per spec §0.3
    loto = df_matches[df_matches["tournament"].isin(["FIFA World Cup", "UEFA Euro"])]
    loto_teams = set(loto["home_team"]) | set(loto["away_team"])
    missing_from_elor = sorted(loto_teams - elor_teams)
    print(f"  Teams in 12-LOTO matches but NOT in eloratings dataset: {len(missing_from_elor)}")
    for t in missing_from_elor:
        # How many matches in LOTO does this team have?
        n = ((loto["home_team"] == t) | (loto["away_team"] == t)).sum()
        print(f"    {t:30s} ({n} LOTO matches)")
    return {"status": ("FAIL" if missing_from_elor else "PASS"),
            "missing_teams": missing_from_elor}


def main() -> int:
    if not ELORATINGS_CSV.exists():
        print(f"ERROR: {ELORATINGS_CSV} missing — run eloratings_client.py first")
        return 1
    if not MATCHES_CSV.exists():
        print(f"ERROR: {MATCHES_CSV} missing — run data_audit.py first")
        return 1

    results = {
        "BUG 1 — date type": audit_date_type_wiring(),
        "BUG 2 — team names": audit_team_name_wiring(),
        "BUG 3 — tournament class": audit_tournament_classification(),
        "BUG 4 — schema": audit_schema_compat(),
        "BUG 5 — coverage": audit_historical_team_coverage(),
    }

    banner("AUDIT SUMMARY")
    for name, r in results.items():
        print(f"  {r['status']:14s}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
