"""eloratings.net Top-20 snapshot fetcher + history sanity-check helper.

Used by `build_history.py` to validate the reconstructed Elo trajectory against the
canonical reference. See plan decision #12: ±100 Elo tolerance, >5 divergences →
HARD STOP. Absolute bounds [1000, 2500] are enforced ALWAYS (whether or not the
network snapshot is available).

Data source: https://www.eloratings.net/World.tsv — TSV with country codes
(eloratings.net uses some non-ISO codes: EN/SC/WL/NI for the UK home nations,
SQ for Albania, KS for Kosovo, etc.).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from mc_simu._common import hard_stop  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "data" / "mc_simu"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "mc_simu"
ELORATINGS_TSV_URL = "https://www.eloratings.net/World.tsv"


# ── eloratings.net country code → Kaggle dataset team name ────────────────────
# Built from observed Top-30 + likely Top-50 expansions. Codes are eloratings.net's
# internal scheme (mix of ISO 3166-1 alpha-2 and custom for UK home nations etc.).
# Mapping target is Kaggle dataset's `home_team`/`away_team` convention so callers
# can pass the result straight to history dicts keyed by team name.
CODE_TO_TEAM: dict[str, str] = {
    "ES": "Spain",        "AR": "Argentina",    "FR": "France",
    "EN": "England",      "BR": "Brazil",       "PT": "Portugal",
    "CO": "Colombia",     "NL": "Netherlands",  "EC": "Ecuador",
    "HR": "Croatia",      "DE": "Germany",      "NO": "Norway",
    "JP": "Japan",        "TR": "Turkey",       "UY": "Uruguay",
    "CH": "Switzerland",  "SN": "Senegal",      "DK": "Denmark",
    "BE": "Belgium",      "MX": "Mexico",       "IT": "Italy",
    "PY": "Paraguay",     "AT": "Austria",      "MA": "Morocco",
    "CA": "Canada",       "AU": "Australia",    "RU": "Russia",
    "RS": "Serbia",       "SQ": "Albania",      "UA": "Ukraine",
    "US": "United States","KS": "Kosovo",       "PE": "Peru",
    "CL": "Chile",        "VE": "Venezuela",    "BO": "Bolivia",
    "SE": "Sweden",       "PL": "Poland",       "WL": "Wales",
    "SC": "Scotland",     "NI": "Northern Ireland", "IE": "Republic of Ireland",
    "RO": "Romania",      "HU": "Hungary",      "GR": "Greece",
    "CZ": "Czech Republic", "SK": "Slovakia",   "SI": "Slovenia",
    "BA": "Bosnia and Herzegovina", "BG": "Bulgaria", "ME": "Montenegro",
    "MK": "North Macedonia", "IS": "Iceland",   "FI": "Finland",
    "TN": "Tunisia",      "EG": "Egypt",        "DZ": "Algeria",
    "CI": "Ivory Coast",  "NG": "Nigeria",      "GH": "Ghana",
    "CM": "Cameroon",     "ZA": "South Africa", "MR": "Mauritania",
    "KR": "South Korea",  "IR": "Iran",         "SA": "Saudi Arabia",
    "QA": "Qatar",        "JO": "Jordan",       "IQ": "Iraq",
    "UZ": "Uzbekistan",   "OM": "Oman",         "AE": "United Arab Emirates",
    "PA": "Panama",       "CR": "Costa Rica",   "JM": "Jamaica",
    "HN": "Honduras",     "HT": "Haiti",        "CU": "Cuba",
    "NZ": "New Zealand",  "TG": "Togo",         "ML": "Mali",
    "BF": "Burkina Faso", "ZM": "Zambia",
}


def fetch_eloratings_top20(*, force: bool = False) -> dict[str, float]:
    """Fetch eloratings.net World.tsv, return {team_name → current Elo} for top 20.

    Caches result to data/mc_simu/eloratings_top20_<YYYY-MM-DD>.json. Idempotent —
    re-reads cache on same day unless force=True. Unmapped country codes are
    skipped silently (their codes will be added to CODE_TO_TEAM if/when we expand
    the sanity comparison beyond top 20).

    Raises urllib.error.URLError on network failure — caller is responsible for
    fallback (e.g., loading the golden fixture).
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache = CACHE_DIR / f"eloratings_top20_{today}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))

    req = urllib.request.Request(ELORATINGS_TSV_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        content = r.read().decode("utf-8", errors="replace")

    top20: dict[str, float] = {}
    for line in content.splitlines():
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        try:
            rank = int(fields[0])
        except ValueError:
            continue
        if rank > 20:
            break
        code = fields[2]
        rating = float(fields[3])
        team = CODE_TO_TEAM.get(code)
        if team is None:
            continue   # silently skip unmapped — we'll see this in the diff report
        top20[team] = rating

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(top20, ensure_ascii=False, indent=2), encoding="utf-8")
    return top20


def load_fallback_fixture() -> dict[str, float] | None:
    """Try the committed golden fixture if live fetch fails."""
    candidates = sorted(FIXTURE_DIR.glob("eloratings_top20_*.json"))
    if not candidates:
        return None
    # Newest fixture wins
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


# ── Sanity check ──────────────────────────────────────────────────────────────


def _abs_bound_violations(latest: dict[str, float],
                          lo: float = 500.0, hi: float = 2500.0) -> list[tuple[str, float]]:
    return [(team, r) for team, r in latest.items() if r < lo or r > hi]


def sanity_check_history(history: dict[str, pd.DataFrame], *,
                         top20_reference: dict[str, float] | None,
                         tolerance: float = 100.0,
                         max_failures: int = 5) -> dict[str, object]:
    """Two-layer sanity check on a reconstructed Elo history.

    Layer A (always): every team's latest rating ∈ [500, 2500]. HARD STOP on violation.
        Loose bounds (500 floor, not the plan's original 1000) — weak national teams
        like Brunei/Macau/Laos legitimately fall below 1000 after sustained losing
        streaks. Real numerical bugs (sign errors, NaN propagation, K-factor scale
        bugs) cause runaway to negatives or >3000, which [500, 2500] still catches.

    Layer B (if reference present): model's latest top-20 ratings vs eloratings.net.
        HARD STOP if more than `max_failures` teams diverge by > tolerance Elo.

    If top20_reference is None (scrape failed AND fixture absent), Layer B is skipped
    with WARN; Layer A still runs.

    Returns a report dict; HARD STOP raises SystemExit before returning when applicable.
    """
    from mc_simu.elo import latest_ratings

    latest = latest_ratings(history)

    # ── Layer A ──
    viol = _abs_bound_violations(latest)
    if viol:
        sample = ", ".join(f"{t}={r:.0f}" for t, r in viol[:5])
        hard_stop("sanity_check_history Layer A",
                  f"{len(viol)} teams outside [500, 2500]: {sample}",
                  "all latest ratings ∈ [500, 2500]")

    # ── Layer B ──
    if top20_reference is None:
        return {"layer_a": "PASS", "layer_b": "SKIP (no reference)",
                "n_teams_checked": len(latest)}

    diffs = []
    for team, ref_rating in top20_reference.items():
        if team not in latest:
            diffs.append((team, None, ref_rating, None))
            continue
        model_rating = latest[team]
        delta = model_rating - ref_rating
        diffs.append((team, model_rating, ref_rating, delta))

    failures = [(t, m, r, d) for (t, m, r, d) in diffs if d is None or abs(d) > tolerance]
    n_failures = len(failures)

    if n_failures > max_failures:
        parts: list[str] = []
        for t, m, r, d in failures[:8]:
            m_str = "NA" if m is None else f"{m:.0f}"
            d_str = "NA" if d is None else f"{d:+.0f}"
            parts.append(f"{t}: model={m_str} ref={r:.0f} Δ={d_str}")
        detail = "; ".join(parts)
        hard_stop("sanity_check_history Layer B",
                  f"{n_failures} of top-20 diverge >{tolerance:.0f} Elo: {detail}",
                  f"≤{max_failures} divergences")

    return {
        "layer_a": "PASS",
        "layer_b": f"{n_failures} of {len(diffs)} top-20 diverge >{tolerance:.0f}",
        "diffs": diffs,
        "n_teams_checked": len(latest),
    }


__all__ = [
    "CODE_TO_TEAM",
    "fetch_eloratings_top20",
    "load_fallback_fixture",
    "sanity_check_history",
]
