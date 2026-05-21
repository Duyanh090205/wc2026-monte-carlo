"""Bracket adapter — UEFA Euro 16-team format (Euro 2004, 2008, 2012).

Format:
    16 teams → 4 groups (A-D) × 4 → top 2 each (8) → QF → SF → Final
    No R16, no best-thirds. Group runners-up advance directly to QF.

Standard cross-bracket QF pairings (UEFA convention since 1996):
    QF1: Winner A vs Runner-up B
    QF2: Winner C vs Runner-up D
    QF3: Winner B vs Runner-up A
    QF4: Winner D vs Runner-up C
    SF1: W_QF1 vs W_QF2  → upper half
    SF2: W_QF3 vs W_QF4  → lower half
    Final: W_SF1 vs W_SF2

Group assignments per edition are hardcoded — historical fact, won't change.
make_round_robin() generates the 6 group matches per group on the fly.

Same harness primitives as WC2026 (rank_group, sample_scores_batch, etc.) —
this adapter only knows about its bracket shape; the model plugged in is
arbitrary per the Predictor protocol.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from mc_simu._common import banner  # noqa: E402,F401 — UTF-8 stdout side-effect
from mc_simu.simulator import (  # noqa: E402
    HostInfo,
    Predictor,
    build_group_samplers,
    build_ko_advance_table,
    make_elo_predictor,
    sample_knockout_winner,
    sample_scores_batch,
)
from mc_simu.single_game import ModelParams  # noqa: E402
from mc_simu.standings import (  # noqa: E402
    GroupMatch,
    compute_stats,
    rank_group,
)


GROUP_LETTERS = list("ABCD")  # 4 groups

# QF bracket: (qf_match_id, left_source, right_source)
# Source = ("W"|"RU", group_letter)
QF_BRACKET: list[tuple[int, tuple[str, str], tuple[str, str]]] = [
    (1, ("W",  "A"), ("RU", "B")),
    (2, ("W",  "C"), ("RU", "D")),
    (3, ("W",  "B"), ("RU", "A")),
    (4, ("W",  "D"), ("RU", "C")),
]

# Later rounds: (match_id, prev_left, prev_right).
# SF mapping per UEFA cross-bracket convention: A/B half stays separated from
# C/D half until Final. Verified against Euro 2008 (GER[QF1]+TUR[QF3] → SF1;
# RUS[QF2]+ESP[QF4] → SF2) and Euro 2012 (POR+ESP → SF1; GER+ITA → SF2).
#
# Earlier draft had SF1=W(QF1)vsW(QF2) — wrong (would let W_A and W_C meet in
# SF, but UEFA's convention keeps them on opposite halves until Final).
# Corrected per Phase 3 web-verification 2026-05-20 (decision #36).
LATER_ROUNDS: list[tuple[int, int, int]] = [
    (5, 1, 3),   # SF1 — A/B half: W(W_A vs RU_B) vs W(W_B vs RU_A)
    (6, 2, 4),   # SF2 — C/D half: W(W_C vs RU_D) vs W(W_D vs RU_C)
    (7, 5, 6),   # Final
]


# ── Historical editions data ──────────────────────────────────────────────────

# Each entry: year → dict with groups + hosts. Group teams use names matching
# the elo_history.csv canonical entries (Kaggle international_results).
EDITIONS: dict[int, dict[str, Any]] = {
    2004: {
        "host_countries": ["Portugal"],
        "host_confederation": "UEFA",
        "groups": {
            "A": ["Portugal", "Greece", "Spain", "Russia"],
            "B": ["France", "England", "Croatia", "Switzerland"],
            "C": ["Sweden", "Denmark", "Italy", "Bulgaria"],
            "D": ["Czech Republic", "Netherlands", "Germany", "Latvia"],
        },
    },
    2008: {
        "host_countries": ["Austria", "Switzerland"],
        "host_confederation": "UEFA",
        "groups": {
            "A": ["Switzerland", "Czech Republic", "Portugal", "Turkey"],
            "B": ["Austria", "Croatia", "Germany", "Poland"],
            "C": ["Netherlands", "Italy", "Romania", "France"],
            "D": ["Spain", "Sweden", "Greece", "Russia"],
        },
    },
    2012: {
        "host_countries": ["Poland", "Ukraine"],
        "host_confederation": "UEFA",
        "groups": {
            "A": ["Poland", "Greece", "Russia", "Czech Republic"],
            "B": ["Netherlands", "Denmark", "Germany", "Portugal"],
            "C": ["Spain", "Italy", "Republic of Ireland", "Croatia"],
            "D": ["Ukraine", "Sweden", "France", "England"],
        },
    },
}


# ── Round-robin fixture generator ─────────────────────────────────────────────


def make_round_robin(teams: list[str]) -> list[tuple[str, str]]:
    """Return all C(4,2)=6 unique pairings within a 4-team group.

    Order is (i, j) with i < j by input list position; treat first team as
    "home" arbitrarily (venue logic handles HFA separately).
    """
    pairs: list[tuple[str, str]] = []
    n = len(teams)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((teams[i], teams[j]))
    return pairs


# ── Bundle + sim context ─────────────────────────────────────────────────────


@dataclass
class Euro16Bundle:
    """Static data for one edition."""

    year: int
    host: HostInfo
    group_membership: dict[str, list[str]]   # letter → 4 teams
    ratings: dict[str, float]
    params: ModelParams


def load_edition_bundle(
    year: int,
    ratings: dict[str, float],
    params: ModelParams | None = None,
) -> Euro16Bundle:
    """Load a historical Euro edition.

    Args:
        year:    one of EDITIONS keys (2004/2008/2012)
        ratings: pre-tournament Elo snapshot for all 16 teams (caller's
                 responsibility to use as-of date BEFORE tournament start).
    """
    if year not in EDITIONS:
        raise ValueError(f"No data for Euro {year}; available: {sorted(EDITIONS)}")
    if params is None:
        params = ModelParams()
    ed = EDITIONS[year]
    host = HostInfo(
        host_countries=list(ed["host_countries"]),
        host_confederation=ed["host_confederation"],
    )
    return Euro16Bundle(
        year=year,
        host=host,
        group_membership={g: list(teams) for g, teams in ed["groups"].items()},
        ratings=ratings,
        params=params,
    )


@dataclass
class Euro16SimContext:
    group_cdfs_per_group: dict[str, np.ndarray]
    group_team_pairs: dict[str, list[tuple[str, str]]]
    ko_advance: dict[tuple[str, str], float]
    fifa_ranking: dict[str, float]
    teams: list[str]


def build_sim_context(
    bundle: Euro16Bundle,
    predictor: Predictor | None = None,
) -> Euro16SimContext:
    """Pre-compute 24 group CDFs + 16×15 KO advance table."""
    if predictor is None:
        predictor = make_elo_predictor(bundle.params)

    group_team_pairs: dict[str, list[tuple[str, str]]] = {}
    group_cdfs_per_group: dict[str, np.ndarray] = {}

    for letter in GROUP_LETTERS:
        teams = bundle.group_membership[letter]
        pairs = make_round_robin(teams)
        # Choose venue: if a host is in the match, set venue=that host. Else "".
        host_set = set(bundle.host.host_countries)
        fixtures = []
        for h, a in pairs:
            host_in_match = host_set & {h, a}
            venue = next(iter(host_in_match)) if host_in_match else ""
            fixtures.append({
                "home_team": h,
                "away_team": a,
                "venue_country": venue,
                "tournament_type": "continental_final",
                "attendance_pct": 1.0,
            })
        samplers = build_group_samplers(
            fixtures=fixtures,
            ratings=bundle.ratings,
            host=bundle.host,
            predictor=predictor,
        )
        group_team_pairs[letter] = [(s.home_team, s.away_team) for s in samplers]
        group_cdfs_per_group[letter] = np.stack([s.cdf_flat for s in samplers])

    teams = [t for letter in GROUP_LETTERS for t in bundle.group_membership[letter]]
    ko_advance = build_ko_advance_table(
        teams=teams,
        ratings=bundle.ratings,
        host=bundle.host,
        predictor=predictor,
    )
    fifa_ranking = dict(bundle.ratings)

    return Euro16SimContext(
        group_cdfs_per_group=group_cdfs_per_group,
        group_team_pairs=group_team_pairs,
        ko_advance=ko_advance,
        fifa_ranking=fifa_ranking,
        teams=teams,
    )


# ── Single-iteration sim ──────────────────────────────────────────────────────


def simulate_one(
    sim: Euro16SimContext,
    rng: np.random.Generator,
) -> tuple[str, dict[str, str]]:
    """One Monte Carlo iteration — returns (champion, group_winners_by_letter)."""
    # 1. Sim 24 group matches
    matches_by_group: dict[str, list[GroupMatch]] = {}
    for letter in GROUP_LETTERS:
        scores = sample_scores_batch(sim.group_cdfs_per_group[letter], rng)
        pairs = sim.group_team_pairs[letter]
        matches_by_group[letter] = [
            GroupMatch(home_team=h, away_team=a,
                       home_goals=int(scores[i, 0]), away_goals=int(scores[i, 1]))
            for i, (h, a) in enumerate(pairs)
        ]

    # 2. Rank each group
    winners: dict[str, str] = {}
    runners_up: dict[str, str] = {}
    for letter in GROUP_LETTERS:
        stats = compute_stats(matches_by_group[letter])
        ranked = rank_group(
            matches_by_group[letter],
            final_tiebreaker=sim.fifa_ranking,
            stats=stats,
        )
        winners[letter] = ranked[0]
        runners_up[letter] = ranked[1]

    # 3. QF (no best-thirds, no R16 — go straight from group to QF)
    def resolve(source: tuple[str, str]) -> str:
        kind, key = source
        if kind == "W":
            return winners[key]
        if kind == "RU":
            return runners_up[key]
        raise ValueError(f"Unknown source kind: {kind}")

    match_winners: dict[int, str] = {}
    for match_id, left, right in QF_BRACKET:
        team_a = resolve(left)
        team_b = resolve(right)
        match_winners[match_id] = sample_knockout_winner(
            team_a, team_b, sim.ko_advance[(team_a, team_b)], rng,
        )

    # 4. SF + Final
    for match_id, prev_left, prev_right in LATER_ROUNDS:
        team_a = match_winners[prev_left]
        team_b = match_winners[prev_right]
        match_winners[match_id] = sample_knockout_winner(
            team_a, team_b, sim.ko_advance[(team_a, team_b)], rng,
        )

    champion = match_winners[7]
    return champion, winners


# ── Monte Carlo loop ──────────────────────────────────────────────────────────


def run_monte_carlo(
    bundle: Euro16Bundle,
    n_iterations: int = 10_000,
    seed: int = 42,
    progress: bool = False,
    predictor: Predictor | None = None,
) -> dict[str, Any]:
    """Aggregate champion + group-winner frequencies across N iterations.

    Returns same structure as wc2026.run_monte_carlo:
        {'champion': {team: {mc_fair_prob, mc_se, n_iterations}},
         'group_winners': {letter: {team: ...}}}
    """
    sim = build_sim_context(bundle, predictor=predictor)
    rng = np.random.default_rng(seed)

    champion_counter: dict[str, int] = defaultdict(int)
    group_winner_counter: dict[str, dict[str, int]] = {
        g: defaultdict(int) for g in GROUP_LETTERS
    }

    iterator = range(n_iterations)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"Euro{bundle.year} MC ({n_iterations:,})")
        except ImportError:
            pass

    for _ in iterator:
        champion, group_winners = simulate_one(sim, rng)
        champion_counter[champion] += 1
        for g, w in group_winners.items():
            group_winner_counter[g][w] += 1

    def to_results(counter: dict[str, int], n: int) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for team, count in counter.items():
            p = count / n
            out[team] = {
                "mc_fair_prob": p,
                "mc_se": float(np.sqrt(p * (1 - p) / n)),
                "n_iterations": n,
            }
        return out

    return {
        "champion": to_results(champion_counter, n_iterations),
        "group_winners": {
            g: to_results(group_winner_counter[g], n_iterations) for g in GROUP_LETTERS
        },
    }
