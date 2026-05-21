"""Bracket adapter — UEFA Euro 24-team format (Euro 2016, 2020, 2024).

Format:
    24 teams → 6 groups (A-F) × 4 → top 2 (12) + 4 best-thirds → R16 → QF → SF → F

Bracket structure (verified against Euro 2024 actual):

    R16 (8 matches):
        M1 (FIFA M37):  2A vs 2B                — static
        M2 (FIFA M38):  1A vs 2C                — static
        M3 (FIFA M39):  1C vs 3rd-from-{D,E,F}  — depends on which thirds advance
        M4 (FIFA M40):  1B vs 3rd-from-{A,D,E,F}
        M5 (FIFA M41):  2D vs 2E                — static
        M6 (FIFA M42):  1F vs 3rd-from-{A,B,C}
        M7 (FIFA M43):  1E vs 3rd-from-{A,B,C,D}
        M8 (FIFA M44):  1D vs 2F                — static

    QF:
        M9  (FIFA M45):  W4 vs W2  — upper-left  (Spain/Germany side, Euro 2024)
        M10 (FIFA M46):  W5 vs W6  — lower-right (France/Portugal side)
        M11 (FIFA M47):  W1 vs W3  — upper-right (Switzerland/England side)
        M12 (FIFA M48):  W7 vs W8  — lower-left  (Netherlands/Turkey side)

    SF: M13 = W9 vs W10, M14 = W11 vs W12
    Final: M15 = W13 vs W14

Third-place seeding: 4 of 6 groups contribute a 3rd-place team (C(6,4)=15 combos).
Each winner with a 3rd slot has an "allowed" set of groups it can face — UEFA
designs these sets to guarantee ≥1 valid pairing per combo. We use a bipartite
constraint solver (24-permutation enumeration) to pick a deterministic valid
assignment; this may not exactly match the UEFA-published Annex table per combo
but produces a valid bracket every time. Phase 4 can refine if precise replay
matters for LOTO-CV.

For combo {C,D,E,F}, three slots are forced (1F→3C, 1E→3D, 1C must take from
{E,F}) and one is free (1B can take E or F, 1C takes the other). UEFA's
official Annex picks 1B→3F (Euro 2024 actual: Spain vs Georgia). The greedy
solver here picks 1B→3E first; equally valid bracket, just not the UEFA
canonical choice for this single combo. **Phase 4 should hardcode the 15-row
UEFA Annex table if exact-replay accuracy materially affects Brier on
historical Euros** (current v1 uses any valid pairing — sufficient for
structural / smoke / approximate-prob purposes).
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
    TeamStats,
    compute_stats,
    rank_best_thirds,
    rank_group,
)


GROUP_LETTERS = list("ABCDEF")  # 6 groups

# UEFA "allowed sets" — which groups' 3rd-place team each named winner can face.
# Read off Euro 2024 R16 match descriptions (Wikipedia).
THIRD_ALLOWED_PER_WINNER: dict[str, frozenset[str]] = {
    "B": frozenset("ADEF"),
    "C": frozenset("DEF"),
    "E": frozenset("ABCD"),
    "F": frozenset("ABC"),
}

# UEFA Annex C — official 15-row best-thirds assignment table for Euro 2020/2024
# bracket geometry. Verified against UEFA Annex via Wikipedia "UEFA Euro 2024
# knockout stage" 2026-05-20. Each key = frozenset of 4 group letters whose
# 3rd-place teams advance; value = {winner_letter: third_group}.
#
# Earlier draft used a greedy permutation solver that produced VALID but not
# UEFA-CANONICAL pairings for some combos (e.g., CDEF: greedy → 1B→3E,
# UEFA → 1B→3F). Hardcoded table per Phase 3 decision #37 to enable exact
# replay in Phase 4 LOTO-CV. For Euro 2016 the column header is DIFFERENT
# (1A/1B/1C/1D instead of 1B/1C/1E/1F) — that adapter is a separate concern.
UEFA_ANNEX_C: dict[frozenset[str], dict[str, str]] = {
    frozenset("ABCD"): {"B": "A", "C": "D", "E": "B", "F": "C"},
    frozenset("ABCE"): {"B": "A", "C": "E", "E": "B", "F": "C"},
    frozenset("ABCF"): {"B": "A", "C": "F", "E": "B", "F": "C"},
    frozenset("ABDE"): {"B": "D", "C": "E", "E": "A", "F": "B"},
    frozenset("ABDF"): {"B": "D", "C": "F", "E": "A", "F": "B"},
    frozenset("ABEF"): {"B": "E", "C": "F", "E": "B", "F": "A"},
    frozenset("ACDE"): {"B": "E", "C": "D", "E": "C", "F": "A"},
    frozenset("ACDF"): {"B": "F", "C": "D", "E": "C", "F": "A"},
    frozenset("ACEF"): {"B": "E", "C": "F", "E": "C", "F": "A"},
    frozenset("ADEF"): {"B": "E", "C": "F", "E": "D", "F": "A"},
    frozenset("BCDE"): {"B": "E", "C": "D", "E": "B", "F": "C"},
    frozenset("BCDF"): {"B": "F", "C": "D", "E": "C", "F": "B"},
    frozenset("BCEF"): {"B": "F", "C": "E", "E": "C", "F": "B"},
    frozenset("BDEF"): {"B": "F", "C": "E", "E": "D", "F": "B"},
    frozenset("CDEF"): {"B": "F", "C": "E", "E": "D", "F": "C"},
}

# R16 bracket. Source = ("W"|"RU", group) for static slots, ("T3", winner_key)
# for 3rd-place slots (resolved per-iteration via combo → seeding map).
R16_BRACKET: list[tuple[int, tuple[str, str], tuple[str, str]]] = [
    (1, ("RU", "A"), ("RU", "B")),
    (2, ("W",  "A"), ("RU", "C")),
    (3, ("W",  "C"), ("T3", "C")),
    (4, ("W",  "B"), ("T3", "B")),
    (5, ("RU", "D"), ("RU", "E")),
    (6, ("W",  "F"), ("T3", "F")),
    (7, ("W",  "E"), ("T3", "E")),
    (8, ("W",  "D"), ("RU", "F")),
]

# Later rounds: (match_id, prev_left, prev_right)
LATER_ROUNDS: list[tuple[int, int, int]] = [
    # QF (FIFA M45-M48)
    (9,  4, 2),    # QF1 — W(1B vs 3) vs W(1A vs 2C) — upper-left
    (10, 5, 6),    # QF2 — W(2D vs 2E) vs W(1F vs 3) — lower-right (Euro2024: France vs Portugal)
    (11, 1, 3),    # QF3 — W(2A vs 2B) vs W(1C vs 3) — upper-right (Swiss vs England)
    (12, 7, 8),    # QF4 — W(1E vs 3) vs W(1D vs 2F) — lower-left (NL vs Turkey)
    # SF
    (13, 9, 10),   # SF1 (Euro2024: Spain vs France)
    (14, 11, 12),  # SF2 (Euro2024: England vs Netherlands)
    # Final
    (15, 13, 14),
]


# ── Third-place seeding solver ────────────────────────────────────────────────


def seed_thirds(combo: frozenset[str]) -> dict[str, str]:
    """Return UEFA-canonical {winner_key: third_group} for `combo`.

    Args:
        combo: frozenset of 4 group letters whose 3rd-place teams advance
               (e.g., frozenset("CDEF") for Euro 2024).

    Returns:
        Dict mapping winner key in {"B", "C", "E", "F"} → group letter of the
        3rd-place team it faces. The 4 mapped groups equal `combo`. Lookup is
        the verbatim UEFA Annex C table (UEFA_ANNEX_C constant), not a solver
        result — exact replay for LOTO-CV.

    Raises:
        KeyError if combo has != 4 groups or contains unknown letter.
    """
    if len(combo) != 4:
        raise ValueError(f"combo must have 4 groups, got {len(combo)}: {combo}")
    return dict(UEFA_ANNEX_C[combo])


# ── Historical editions data ─────────────────────────────────────────────────

NAME_ALIASES: dict[str, str] = {
    "Czechia": "Czech Republic",
}

# NOTE: Euro 2016 is intentionally NOT in EDITIONS here. Per Phase 3
# web-verification 2026-05-20 (decision #38), Euro 2016 used a DIFFERENT R16
# geometry where 3rd-place teams faced 1A/1B/1C/1D winners (not the 1B/1C/1E/1F
# pattern used in Euro 2020/2024). A separate adapter or branched bracket would
# be needed for Euro 2016. Phase 4 LOTO-CV can decide if Euro 2016 historical
# replay justifies the extra adapter, or accept the 2-edition coverage v1 ships.
EDITIONS: dict[int, dict[str, Any]] = {
    2020: {
        # Multi-host pan-European — no single host country gets α. β cancels
        # symmetrically since all UEFA teams. Played in 2021 due to COVID.
        "host_countries": [],
        "host_confederation": "UEFA",
        "groups": {
            "A": ["Italy", "Turkey", "Wales", "Switzerland"],
            "B": ["Belgium", "Russia", "Finland", "Denmark"],
            "C": ["Netherlands", "North Macedonia", "Ukraine", "Austria"],
            "D": ["England", "Croatia", "Scotland", "Czech Republic"],
            "E": ["Spain", "Sweden", "Slovakia", "Poland"],
            "F": ["Hungary", "Portugal", "France", "Germany"],
        },
    },
    2024: {
        "host_countries": ["Germany"],
        "host_confederation": "UEFA",
        "groups": {
            "A": ["Germany", "Scotland", "Hungary", "Switzerland"],
            "B": ["Spain", "Croatia", "Italy", "Albania"],
            "C": ["Slovenia", "Denmark", "Serbia", "England"],
            "D": ["Poland", "Netherlands", "Austria", "France"],
            "E": ["Belgium", "Slovakia", "Romania", "Ukraine"],
            "F": ["Turkey", "Georgia", "Portugal", "Czech Republic"],
        },
    },
}


def make_round_robin(teams: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            pairs.append((teams[i], teams[j]))
    return pairs


# ── Bundle + sim context ─────────────────────────────────────────────────────


@dataclass
class Euro24Bundle:
    year: int
    host: HostInfo
    group_membership: dict[str, list[str]]
    ratings: dict[str, float]
    params: ModelParams


def load_edition_bundle(
    year: int,
    ratings: dict[str, float],
    params: ModelParams | None = None,
) -> Euro24Bundle:
    if year not in EDITIONS:
        raise ValueError(f"No data for Euro {year}; available: {sorted(EDITIONS)}")
    if params is None:
        params = ModelParams()
    ed = EDITIONS[year]

    ratings_resolved: dict[str, float] = dict(ratings)
    for ed_name, elo_name in NAME_ALIASES.items():
        if ed_name not in ratings_resolved and elo_name in ratings_resolved:
            ratings_resolved[ed_name] = ratings_resolved[elo_name]

    host = HostInfo(
        host_countries=list(ed["host_countries"]),
        host_confederation=ed["host_confederation"],
    )
    return Euro24Bundle(
        year=year,
        host=host,
        group_membership={g: list(teams) for g, teams in ed["groups"].items()},
        ratings=ratings_resolved,
        params=params,
    )


@dataclass
class Euro24SimContext:
    group_cdfs_per_group: dict[str, np.ndarray]
    group_team_pairs: dict[str, list[tuple[str, str]]]
    ko_advance: dict[tuple[str, str], float]
    fifa_ranking: dict[str, float]
    teams: list[str]


def build_sim_context(
    bundle: Euro24Bundle,
    predictor: Predictor | None = None,
) -> Euro24SimContext:
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

    return Euro24SimContext(
        group_cdfs_per_group=group_cdfs_per_group,
        group_team_pairs=group_team_pairs,
        ko_advance=ko_advance,
        fifa_ranking=fifa_ranking,
        teams=teams,
    )


# ── Single-iteration sim ──────────────────────────────────────────────────────


def simulate_one(
    sim: Euro24SimContext,
    rng: np.random.Generator,
) -> tuple[str, dict[str, str]]:
    # 1. Sim 36 group matches (6 groups × 6)
    matches_by_group: dict[str, list[GroupMatch]] = {}
    for letter in GROUP_LETTERS:
        scores = sample_scores_batch(sim.group_cdfs_per_group[letter], rng)
        pairs = sim.group_team_pairs[letter]
        matches_by_group[letter] = [
            GroupMatch(home_team=h, away_team=a,
                       home_goals=int(scores[i, 0]), away_goals=int(scores[i, 1]))
            for i, (h, a) in enumerate(pairs)
        ]

    # 2. Rank each group — capture winner, runner-up, third (+ stats for best-thirds)
    winners: dict[str, str] = {}
    runners_up: dict[str, str] = {}
    thirds: list[tuple[str, str]] = []
    all_stats: dict[str, TeamStats] = {}

    for letter in GROUP_LETTERS:
        group_stats = compute_stats(matches_by_group[letter])
        ranked = rank_group(
            matches_by_group[letter],
            final_tiebreaker=sim.fifa_ranking,
            stats=group_stats,
        )
        winners[letter] = ranked[0]
        runners_up[letter] = ranked[1]
        thirds.append((ranked[2], letter))
        all_stats.update(group_stats)

    # 3. Rank thirds, take top 4
    ranked_thirds = rank_best_thirds(thirds, all_stats, sim.fifa_ranking)
    top4_thirds = ranked_thirds[:4]
    third_combo = frozenset(g for _team, g in top4_thirds)
    third_by_group: dict[str, str] = {g: t for t, g in top4_thirds}

    # 4. Resolve which 3rd plays which winner
    seeding = seed_thirds(third_combo)  # {"B": "F", "C": "E", "E": "D", "F": "C"}

    # 5. R16 bracket
    def resolve(source: tuple[str, str]) -> str:
        kind, key = source
        if kind == "W":
            return winners[key]
        if kind == "RU":
            return runners_up[key]
        if kind == "T3":
            third_group = seeding[key]
            return third_by_group[third_group]
        raise ValueError(f"Unknown source kind: {kind}")

    match_winners: dict[int, str] = {}
    for match_id, left, right in R16_BRACKET:
        team_a = resolve(left)
        team_b = resolve(right)
        match_winners[match_id] = sample_knockout_winner(
            team_a, team_b, sim.ko_advance[(team_a, team_b)], rng,
        )

    for match_id, prev_left, prev_right in LATER_ROUNDS:
        team_a = match_winners[prev_left]
        team_b = match_winners[prev_right]
        match_winners[match_id] = sample_knockout_winner(
            team_a, team_b, sim.ko_advance[(team_a, team_b)], rng,
        )

    champion = match_winners[15]
    return champion, winners


# ── Monte Carlo loop ──────────────────────────────────────────────────────────


def run_monte_carlo(
    bundle: Euro24Bundle,
    n_iterations: int = 10_000,
    seed: int = 42,
    progress: bool = False,
    predictor: Predictor | None = None,
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
