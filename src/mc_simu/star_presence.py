"""Star-presence indicator — count stars per team, add bonus to Elo rating.

Captures the "Messi / Ronaldo problem": aging legends have deflated Transfermarkt
MV (€10-20m) but remain elite players. MV blend alone systematically under-rates
their teams. Adding +X Elo per identified "star" lifts these teams back.

Star definition (3-way union, applied via hand-curation 2026-05-27):
    (a) current Transfermarkt MV >= €70m  → catches young rising stars (Yamal, Wirtz)
    (b) peak career MV >= €100m            → catches aging legends (Messi, Ronaldo)
    (c) Ballon d'Or top-30 in 2024/2025    → catches anyone else famous

Curated counts persisted in data/mc_simu/star_presence_wc2026.json (editable by Sam).

Empirical calibration (2026-05-27, n=15k MC + LOTO-CV Brier):
    X=15 → JSD vs PM 0.0239 (-33% vs MV-blend baseline), Brier 0.1946 (PASS-WARN)

Sam philosophy caveat ([[sam-quant-philosophy-model-market]]):
    Tuning to market = lose alpha. X=15 minimizes JSD AND Brier simultaneously
    — not pure market-fitting. But Sam should review the star list manually before
    final commit since it's hand-curated, not from a public-API ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STAR_JSON = PROJECT_ROOT / "data" / "mc_simu" / "star_presence_wc2026.json"
DEFAULT_BONUS_X = 15.0


def load_star_counts(json_path: Path | None = None) -> dict[str, int]:
    """Load {team: star_count} dict from JSON."""
    json_path = json_path or DEFAULT_STAR_JSON
    if not json_path.exists():
        raise FileNotFoundError(
            f"Star presence JSON missing: {json_path}. "
            f"This file is hand-curated — see module docstring for criteria."
        )
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {k: int(v) for k, v in data["stars_by_team"].items()}


def apply_star_bonus(
    ratings: dict[str, float],
    bonus_X: float = DEFAULT_BONUS_X,
    star_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    """Return new ratings dict with star bonus added: rating[t] += X * stars[t].

    Teams not in star_counts get 0 bonus (no shift). Original ratings dict not mutated.
    """
    if star_counts is None:
        star_counts = load_star_counts()
    out = dict(ratings)
    for team, count in star_counts.items():
        if team in out:
            out[team] = out[team] + bonus_X * count
    return out


def summarize_stars(star_counts: dict[str, int]) -> str:
    """Pretty summary of star distribution for logs."""
    from collections import Counter
    counts = Counter(star_counts.values())
    lines = [f"Total stars: {sum(star_counts.values())} across {len(star_counts)} teams"]
    for c in sorted(counts.keys(), reverse=True):
        lines.append(f"  {c} stars: {counts[c]} teams")
    return "\n".join(lines)


__all__ = [
    "load_star_counts", "apply_star_bonus", "summarize_stars",
    "DEFAULT_STAR_JSON", "DEFAULT_BONUS_X",
]
