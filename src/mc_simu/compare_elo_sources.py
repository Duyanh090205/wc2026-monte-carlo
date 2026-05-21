"""Cross-check self-computed Elo vs eloratings.net scrape.

Phase 1 §2.4 sanity gate — verify the 2 Elo sources are consistent.

Decision rule (per mc_simu.md §2.4):
- Top-20 self-Elo nations: diff < 100 Elo points each → ✓ PASS
- Diff > 100 Elo on > 5/20 teams → ✗ HARD STOP (scrape bug, K-factor mismatch,
                                                 or tournament classification drift)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from mc_simu._common import banner  # noqa: E402
from mc_simu.elo import build_rating_history, latest_ratings  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
MATCHES_CSV = DATA_DIR / "matches_1998_2026.csv"
ELORATINGS_CSV = DATA_DIR / "elo_history_eloratings.csv"

# eloratings.net team name → Kaggle (matches_1998_2026.csv) name. Add as needed.
ELO_TO_KAGGLE_ALIAS: dict[str, str] = {
    "USA": "United States",
    "Czechia": "Czech Republic",
    "Korea": "South Korea",
    "Korea DPR": "North Korea",
    "Cape Verde": "Cape Verde Islands",
    "Turkey": "Türkiye",
}


def latest_eloratings() -> dict[str, float]:
    """Return {team_name (eloratings convention): latest rating_after}."""
    df = pd.read_csv(ELORATINGS_CSV, parse_dates=["date"])
    last = df.sort_values("date").groupby("team").tail(1)
    return dict(zip(last["team"], last["rating_after"].astype(float)))


def latest_self() -> dict[str, float]:
    """Return {team_name (Kaggle convention): latest rating from build_rating_history}."""
    matches = pd.read_csv(MATCHES_CSV)
    history = build_rating_history(matches)
    return latest_ratings(history)


def cross_check() -> int:
    banner("Phase 1 §2.4 — Cross-check self-Elo vs eloratings.net")

    elor = latest_eloratings()
    self_ = latest_self()

    rows = []
    for elor_name, elor_rating in elor.items():
        kaggle_name = ELO_TO_KAGGLE_ALIAS.get(elor_name, elor_name)
        self_rating = self_.get(kaggle_name)
        if self_rating is None:
            continue
        rows.append({
            "team": elor_name,
            "self_elo": self_rating,
            "elor_elo": elor_rating,
            "diff": self_rating - elor_rating,
        })

    if not rows:
        print("ERROR: no team-name overlap between sources — check aliases")
        return 2

    # Sort by eloratings rank (proxy: by elor_rating desc) and take top-20
    df = pd.DataFrame(rows).sort_values("elor_elo", ascending=False)
    top20 = df.head(20).reset_index(drop=True)

    print(f"\n  Top-20 by eloratings.net current rating:\n")
    print(top20.to_string(index=False, formatters={
        "self_elo": "{:.0f}".format, "elor_elo": "{:.0f}".format,
        "diff": "{:+.0f}".format,
    }))

    over_100 = top20[top20["diff"].abs() > 100]
    print(f"\n  Teams with |diff| > 100 Elo: {len(over_100)}/20")

    if len(over_100) > 5:
        print("  ❌ HARD STOP — too many large divergences. Investigate before Phase 4.")
        return 2
    elif len(over_100) > 0:
        print("  ⚠ WARN — some divergence in top-20. Document and proceed.")
        return 1
    else:
        print("  ✅ PASS — both sources agree within 100 Elo across top-20.")
        return 0


if __name__ == "__main__":
    sys.exit(cross_check())
