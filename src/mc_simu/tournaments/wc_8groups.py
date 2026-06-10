"""Bracket adapter — FIFA World Cup 8-group format (WC 2002–2022).

Format:
    32 teams → 8 groups (A-H) × 4 → top 2 each (16) → R16 → QF → SF → Final
    No best-thirds. Group runners-up advance directly to R16.

Bracket structure (paired adjacent groups + standard cross-bracket):

    R16 matches (8, paired by adjacent group pairs):
        pair (A,B): 1A-2B, 2A-1B
        pair (C,D): 1C-2D, 2C-1D
        pair (E,F): 1E-2F, 2E-1F
        pair (G,H): 1G-2H, 2G-1H

    QF matches (4, each combining winners from 2 paired group-pairs):
        QF_9:  W(1A-2B) vs W(1C-2D)   — upper-left quadrant
        QF_10: W(2A-1B) vs W(2C-1D)   — lower-left quadrant
        QF_11: W(1E-2F) vs W(1G-2H)   — upper-right quadrant
        QF_12: W(2E-1F) vs W(2G-1H)   — lower-right quadrant

    SF matches:
        SF_13: W(QF_9)  vs W(QF_11)   — upper half (UL vs UR)
        SF_14: W(QF_10) vs W(QF_12)   — lower half (LL vs LR)

    Final: W(SF_13) vs W(SF_14)

Verified against WC 2022 actual bracket: SF1 = Argentina (1C→QF_9) vs Croatia
(2F→QF_11) ✓; SF2 = France (1D→QF_10) vs Morocco (1F→QF_12) ✓.

Hardcoded EDITIONS covers WC 2002, 2006, 2010, 2014, 2018, 2022.
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
from mc_simu.standings import GroupMatch, compute_stats, rank_group  # noqa: E402


GROUP_LETTERS = list("ABCDEFGH")  # 8 groups

# R16 bracket — paired by adjacent groups.
# (match_id, left_source, right_source). Source = ("W"|"RU", group_letter).
R16_BRACKET: list[tuple[int, tuple[str, str], tuple[str, str]]] = [
    (1, ("W",  "A"), ("RU", "B")),
    (2, ("W",  "C"), ("RU", "D")),
    (3, ("W",  "B"), ("RU", "A")),
    (4, ("W",  "D"), ("RU", "C")),
    (5, ("W",  "E"), ("RU", "F")),
    (6, ("W",  "G"), ("RU", "H")),
    (7, ("W",  "F"), ("RU", "E")),
    (8, ("W",  "H"), ("RU", "G")),
]

# Later rounds: (match_id, prev_left, prev_right)
LATER_ROUNDS: list[tuple[int, int, int]] = [
    # QF: each pairs 2 R16 winners from different paired-group-pairs
    (9,  1, 2),    # QF — upper-left quadrant (1A-2B winner vs 1C-2D winner)
    (10, 3, 4),    # QF — lower-left quadrant (2A-1B vs 2C-1D)
    (11, 5, 6),    # QF — upper-right quadrant (1E-2F vs 1G-2H)
    (12, 7, 8),    # QF — lower-right quadrant (2E-1F vs 2G-1H)
    # SF: upper QFs together, lower QFs together
    (13, 9, 11),   # SF — upper half (UL vs UR)
    (14, 10, 12),  # SF — lower half (LL vs LR)
    # Final
    (15, 13, 14),
]

# Bracket level of each KO match_id — used by survivor-set conditioning (backtest).
KO_ROUND_OF = {**{i: "R16" for i in range(1, 9)}, 9: "QF", 10: "QF", 11: "QF", 12: "QF",
               13: "SF", 14: "SF", 15: "Final"}


# ── Name aliases (modern → elo_history canonical) ────────────────────────────

NAME_ALIASES: dict[str, str] = {
    # Serbia + Montenegro disbanded 2006; Kaggle records as Serbia post-split
    "Serbia and Montenegro": "Serbia",
    # FIFA renamings
    "Czechia": "Czech Republic",
}


# ── Historical editions data ──────────────────────────────────────────────────

EDITIONS: dict[int, dict[str, Any]] = {
    2002: {
        "host_countries": ["South Korea", "Japan"],
        "host_confederation": "AFC",
        "groups": {
            "A": ["France", "Senegal", "Uruguay", "Denmark"],
            "B": ["Spain", "Slovenia", "Paraguay", "South Africa"],
            "C": ["Brazil", "Turkey", "China PR", "Costa Rica"],
            "D": ["South Korea", "Poland", "United States", "Portugal"],
            "E": ["Germany", "Saudi Arabia", "Republic of Ireland", "Cameroon"],
            "F": ["Argentina", "Nigeria", "England", "Sweden"],
            "G": ["Italy", "Ecuador", "Croatia", "Mexico"],
            "H": ["Japan", "Belgium", "Russia", "Tunisia"],
        },
    },
    2006: {
        "host_countries": ["Germany"],
        "host_confederation": "UEFA",
        "groups": {
            "A": ["Germany", "Costa Rica", "Poland", "Ecuador"],
            "B": ["England", "Paraguay", "Trinidad and Tobago", "Sweden"],
            "C": ["Argentina", "Ivory Coast", "Serbia and Montenegro", "Netherlands"],
            "D": ["Mexico", "Iran", "Angola", "Portugal"],
            "E": ["Italy", "Ghana", "United States", "Czech Republic"],
            "F": ["Brazil", "Croatia", "Australia", "Japan"],
            "G": ["France", "Switzerland", "South Korea", "Togo"],
            "H": ["Spain", "Ukraine", "Tunisia", "Saudi Arabia"],
        },
    },
    2010: {
        "host_countries": ["South Africa"],
        "host_confederation": "CAF",
        "groups": {
            "A": ["South Africa", "Mexico", "Uruguay", "France"],
            "B": ["Argentina", "Nigeria", "South Korea", "Greece"],
            "C": ["England", "United States", "Algeria", "Slovenia"],
            "D": ["Germany", "Australia", "Serbia", "Ghana"],
            "E": ["Netherlands", "Denmark", "Japan", "Cameroon"],
            "F": ["Italy", "Paraguay", "New Zealand", "Slovakia"],
            "G": ["Brazil", "North Korea", "Ivory Coast", "Portugal"],
            "H": ["Spain", "Switzerland", "Honduras", "Chile"],
        },
    },
    2014: {
        "host_countries": ["Brazil"],
        "host_confederation": "CONMEBOL",
        "groups": {
            "A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
            "B": ["Spain", "Netherlands", "Chile", "Australia"],
            "C": ["Colombia", "Greece", "Ivory Coast", "Japan"],
            "D": ["Uruguay", "Costa Rica", "England", "Italy"],
            "E": ["Switzerland", "Ecuador", "France", "Honduras"],
            "F": ["Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria"],
            "G": ["Germany", "Portugal", "Ghana", "United States"],
            "H": ["Belgium", "Algeria", "Russia", "South Korea"],
        },
    },
    2018: {
        "host_countries": ["Russia"],
        "host_confederation": "UEFA",
        "groups": {
            "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
            "B": ["Portugal", "Spain", "Morocco", "Iran"],
            "C": ["France", "Australia", "Peru", "Denmark"],
            "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
            "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
            "F": ["Germany", "Mexico", "Sweden", "South Korea"],
            "G": ["Belgium", "Panama", "Tunisia", "England"],
            "H": ["Poland", "Senegal", "Colombia", "Japan"],
        },
    },
    2022: {
        "host_countries": ["Qatar"],
        "host_confederation": "AFC",
        "groups": {
            "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
            "B": ["England", "Iran", "United States", "Wales"],
            "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
            "D": ["France", "Australia", "Denmark", "Tunisia"],
            "E": ["Spain", "Costa Rica", "Germany", "Japan"],
            "F": ["Belgium", "Canada", "Morocco", "Croatia"],
            "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
            "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
        },
    },
}


def make_round_robin(teams: list[str]) -> list[tuple[str, str]]:
    """All C(4,2)=6 unique pairings within a 4-team group."""
    pairs: list[tuple[str, str]] = []
    n = len(teams)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((teams[i], teams[j]))
    return pairs


# ── Bundle + sim context ─────────────────────────────────────────────────────


@dataclass
class WC8GroupsBundle:
    year: int
    host: HostInfo
    group_membership: dict[str, list[str]]
    ratings: dict[str, float]
    params: ModelParams


def load_edition_bundle(
    year: int,
    ratings: dict[str, float],
    params: ModelParams | None = None,
) -> WC8GroupsBundle:
    if year not in EDITIONS:
        raise ValueError(f"No data for WC {year}; available: {sorted(EDITIONS)}")
    if params is None:
        params = ModelParams()
    ed = EDITIONS[year]

    # Resolve aliases — push edition team-name → elo_history canonical
    ratings_resolved: dict[str, float] = dict(ratings)
    for ed_name, elo_name in NAME_ALIASES.items():
        if ed_name not in ratings_resolved and elo_name in ratings_resolved:
            ratings_resolved[ed_name] = ratings_resolved[elo_name]

    host = HostInfo(
        host_countries=list(ed["host_countries"]),
        host_confederation=ed["host_confederation"],
    )
    return WC8GroupsBundle(
        year=year,
        host=host,
        group_membership={g: list(teams) for g, teams in ed["groups"].items()},
        ratings=ratings_resolved,
        params=params,
    )


@dataclass
class WC8GroupsSimContext:
    group_cdfs_per_group: dict[str, np.ndarray]
    group_team_pairs: dict[str, list[tuple[str, str]]]
    ko_advance: dict[tuple[str, str], float]
    fifa_ranking: dict[str, float]
    teams: list[str]


def build_sim_context(
    bundle: WC8GroupsBundle,
    predictor: Predictor | None = None,
) -> WC8GroupsSimContext:
    if predictor is None:
        predictor = make_elo_predictor(bundle.params)

    group_team_pairs: dict[str, list[tuple[str, str]]] = {}
    group_cdfs_per_group: dict[str, np.ndarray] = {}

    host_set = set(bundle.host.host_countries)
    for letter in GROUP_LETTERS:
        teams = bundle.group_membership[letter]
        pairs = make_round_robin(teams)
        fixtures = []
        for h, a in pairs:
            host_in_match = host_set & {h, a}
            venue = next(iter(host_in_match)) if host_in_match else ""
            fixtures.append({
                "home_team": h,
                "away_team": a,
                "venue_country": venue,
                "tournament_type": "world_cup_final",
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

    return WC8GroupsSimContext(
        group_cdfs_per_group=group_cdfs_per_group,
        group_team_pairs=group_team_pairs,
        ko_advance=ko_advance,
        fifa_ranking=fifa_ranking,
        teams=teams,
    )


# ── Single-iteration sim ──────────────────────────────────────────────────────


def simulate_one(
    sim: WC8GroupsSimContext,
    rng: np.random.Generator,
    group_scores: dict | None = None,
    survivors: dict | None = None,
) -> tuple[str, dict[str, str]]:
    """One MC iteration. Optional conditioning for backtest/daily rerun:
        group_scores: {frozenset({home, away}): {team: goals}} — lock played group matches.
        survivors:    {round_label: set(teams that advanced past that round)} — lock KO
                      by who really survived each round, NOT by matchup. Robust to bracket
                      / tiebreaker quirks (e.g. WC2018 group H decided on fair-play, which
                      the model can't reproduce): a team absent from a round's survivor set
                      always loses that round's match; the sole survivor always wins.
    Empty/None = full unconditioned sim.
    """
    group_scores = group_scores or {}
    survivors = survivors or {}

    # 1. Sim 48 group matches (8 groups × 6); lock real scores for played fixtures
    matches_by_group: dict[str, list[GroupMatch]] = {}
    for letter in GROUP_LETTERS:
        scores = sample_scores_batch(sim.group_cdfs_per_group[letter], rng)
        pairs = sim.group_team_pairs[letter]
        gm: list[GroupMatch] = []
        for i, (h, a) in enumerate(pairs):
            real = group_scores.get(frozenset((h, a)))
            hg = real[h] if real else int(scores[i, 0])
            ag = real[a] if real else int(scores[i, 1])
            gm.append(GroupMatch(home_team=h, away_team=a, home_goals=hg, away_goals=ag))
        matches_by_group[letter] = gm

    # 2. Rank each group → winner + runner-up
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

    # 3. R16 (8 matches)
    def resolve(source: tuple[str, str]) -> str:
        kind, key = source
        if kind == "W":
            return winners[key]
        if kind == "RU":
            return runners_up[key]
        raise ValueError(f"Unknown source kind: {kind}")

    def _ko(match_id: int, a: str, b: str) -> str:
        S = survivors.get(KO_ROUND_OF[match_id])
        if S is not None:
            if a in S and b not in S:
                return a
            if b in S and a not in S:
                return b
        return sample_knockout_winner(a, b, sim.ko_advance[(a, b)], rng)

    match_winners: dict[int, str] = {}
    for match_id, left, right in R16_BRACKET:
        match_winners[match_id] = _ko(match_id, resolve(left), resolve(right))

    # 4. QF / SF / Final
    for match_id, prev_left, prev_right in LATER_ROUNDS:
        match_winners[match_id] = _ko(match_id, match_winners[prev_left], match_winners[prev_right])

    champion = match_winners[15]
    return champion, winners


# ── Monte Carlo loop ──────────────────────────────────────────────────────────


def run_monte_carlo(
    bundle: WC8GroupsBundle,
    n_iterations: int = 10_000,
    seed: int = 42,
    progress: bool = False,
    predictor: Predictor | None = None,
    group_scores: dict | None = None,
    survivors: dict | None = None,
) -> dict[str, Any]:
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
            iterator = tqdm(iterator, desc=f"WC{bundle.year} MC ({n_iterations:,})")
        except ImportError:
            pass

    for _ in iterator:
        champion, group_winners = simulate_one(sim, rng, group_scores, survivors)
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
