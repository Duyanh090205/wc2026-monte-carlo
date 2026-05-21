"""Tests for src/mc_simu/standings.py — Phase 3 §4.1 Steps 3.1 + 3.2.

Three categories:
1. Mechanics — fair_play_score formula, compute_stats, collect_thirds
2. rank_group — real (Euro 2016 Group F) + synthetic discriminators
3. rank_best_thirds + R32 Annex C regression — validates that
   data/mc_simu/r32_seeding_table.json (parsed Phase 0 from Wikipedia)
   respects the 8 FIFA Annex C match constraints surfaced via independent
   verification (claude.ai 2026-05-20).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mc_simu.standings import (
    GroupMatch,
    TeamStats,
    collect_thirds,
    compute_stats,
    fair_play_score,
    rank_best_thirds,
    rank_group,
)


# ── Fair-play scorer ──────────────────────────────────────────────────────────


class TestFairPlayScore:
    def test_zero_cards_zero_score(self) -> None:
        assert fair_play_score() == 0

    def test_one_yellow_is_minus_one(self) -> None:
        assert fair_play_score(yellows=1) == -1

    def test_indirect_red_is_minus_three(self) -> None:
        assert fair_play_score(indirect_reds=1) == -3

    def test_direct_red_is_minus_four(self) -> None:
        assert fair_play_score(direct_reds=1) == -4

    def test_yellow_plus_direct_red_is_minus_five(self) -> None:
        """Single player got yellow earlier in match, then direct red later."""
        assert fair_play_score(yellow_plus_reds=1) == -5

    def test_mixed_5_yellows_2_direct_reds(self) -> None:
        # 5 × -1 + 2 × -4 = -13
        assert fair_play_score(yellows=5, direct_reds=2) == -13


# ── compute_stats mechanics ───────────────────────────────────────────────────


class TestComputeStats:
    def test_single_match_winner_loser(self) -> None:
        matches = [GroupMatch("A", "B", 2, 0)]
        s = compute_stats(matches)
        assert s["A"].points == 3
        assert s["A"].goal_difference == 2
        assert s["A"].goals_for == 2
        assert s["A"].played == 1
        assert s["B"].points == 0
        assert s["B"].goal_difference == -2

    def test_draw_one_point_each(self) -> None:
        matches = [GroupMatch("A", "B", 1, 1)]
        s = compute_stats(matches)
        assert s["A"].points == 1
        assert s["B"].points == 1
        assert s["A"].goal_difference == 0

    def test_fair_play_aggregates_across_matches(self) -> None:
        matches = [
            GroupMatch("A", "B", 0, 0, home_yellows=2, away_direct_reds=1),
            GroupMatch("A", "C", 0, 0, home_yellows=1),
        ]
        s = compute_stats(matches)
        assert s["A"].fair_play == -3  # 3 yellows total
        assert s["B"].fair_play == -4  # 1 direct red
        assert s["C"].fair_play == 0


# ── rank_group: real-world Euro 2016 Group F ──────────────────────────────────


def _euro_2016_group_f() -> list[GroupMatch]:
    """Real results from Euro 2016 Group F.

    Final standings (per UEFA/Wikipedia):
      1. Hungary  5 pts  GF=6 GA=4 GD=+2
      2. Iceland  5 pts  GF=4 GA=3 GD=+1
      3. Portugal 3 pts  GF=4 GA=4 GD= 0
      4. Austria  1 pt   GF=1 GA=4 GD=-3
    """
    return [
        GroupMatch("Austria", "Hungary", 0, 2),
        GroupMatch("Portugal", "Iceland", 1, 1),
        GroupMatch("Iceland", "Hungary", 1, 1),
        GroupMatch("Portugal", "Austria", 0, 0),
        GroupMatch("Iceland", "Austria", 2, 1),
        GroupMatch("Hungary", "Portugal", 3, 3),
    ]


class TestRankGroupEuro2016F:
    """Spec §4.1 Step 3.1 unit test: Hungary 1st > Iceland 2nd > Portugal 3rd > Austria 4th.

    Under FIFA 2026 chain: Hungary/Iceland tied on 5 pts. H2H 1-1 draw →
    H2H pts/GD/GF all equal → fall to Overall GD: Hungary +2 > Iceland +1.
    """

    def test_full_ranking(self) -> None:
        ranked = rank_group(
            _euro_2016_group_f(),
            final_tiebreaker={"Hungary": 1800, "Iceland": 1750, "Portugal": 1900, "Austria": 1700},
        )
        assert ranked == ["Hungary", "Iceland", "Portugal", "Austria"]


# ── rank_group: synthetic H2H-first vs GD-first discriminator ─────────────────


def _h2h_vs_gd_discriminator() -> list[GroupMatch]:
    """Hand-crafted 2-way tie. A beats B in H2H; B has much better Overall GD.

    Standings:
      A: W vs B (2-1), W vs C (3-0), L vs D (0-2) → 6 pts, GF=5  GA=3  GD=+2
      B: L vs A (1-2), W vs C (5-0), W vs D (5-0) → 6 pts, GF=11 GA=2  GD=+9
      C: L,L,D                                    → 1 pt
      D: W vs A (2-0), L vs B (0-5), D vs C (0-0) → 4 pts (NOT tied with A/B)

    A and B tied at 6 pts. H2H: A won 2-1 → A H2H pts=3, GD=+1, GF=2; B 0/-1/1.
    Under FIFA 2026 (H2H first): A 1st, B 2nd.
    Under WRONG GD-first chain (old spec): B (+9) > A (+2), so B 1st, A 2nd.

    This test ANCHORS the corrected chain — if it ever flips, the chain regressed.
    """
    return [
        GroupMatch("A", "B", 2, 1),
        GroupMatch("A", "C", 3, 0),
        GroupMatch("A", "D", 0, 2),
        GroupMatch("B", "C", 5, 0),
        GroupMatch("B", "D", 5, 0),
        GroupMatch("C", "D", 0, 0),
    ]


class TestRankGroupH2HFirstChain:
    def test_h2h_wins_over_overall_gd(self) -> None:
        """Anchor: FIFA 2026 chain uses H2H BEFORE Overall GD (per decision #22)."""
        matches = _h2h_vs_gd_discriminator()
        ranked = rank_group(
            matches,
            final_tiebreaker={"A": 1800, "B": 1750, "C": 1700, "D": 1650},
        )
        # A wins H2H 1-0 → A is 1st despite worse overall GD
        assert ranked[0] == "A"
        assert ranked[1] == "B"


# ── rank_group: 3-way tie with recursive H2H ─────────────────────────────────


def _three_way_tie_partial_h2h() -> list[GroupMatch]:
    """3 teams (A, B, C) on 6 pts; D on 0.

    Build it so:
      - All three beat D (3 pts each)
      - Round-robin among {A, B, C}: A beat B, B beat C, C beat A (rock-paper-scissors)
        Each has 1W+1L within H2H → all have H2H pts=3, H2H GD depends on margins.

    Margins:
      A 1-0 B   → A H2H GD +1, GF 1
      B 2-0 C   → B H2H GD +2, GF 2
      C 3-1 A   → C H2H GD +2, GF 3 (against A -2 GD, 1 GF)

    H2H pts: A=3, B=3, C=3. Tied.
    H2H GD: A = +1-2 = -1. B = -1+2 = +1. C = -2+2 = 0.
    Order by H2H GD: B (+1) > C (0) > A (-1). B 1st, C 2nd, A 3rd. D 4th.

    No further recursion needed — H2H GD fully separated the 3 tied teams.
    """
    return [
        GroupMatch("A", "B", 1, 0),
        GroupMatch("B", "C", 2, 0),
        GroupMatch("C", "A", 3, 1),
        GroupMatch("A", "D", 5, 0),
        GroupMatch("B", "D", 5, 0),
        GroupMatch("C", "D", 5, 0),
    ]


class TestRankGroupThreeWayTie:
    def test_h2h_gd_separates_3_way_tie(self) -> None:
        ranked = rank_group(
            _three_way_tie_partial_h2h(),
            final_tiebreaker={"A": 1800, "B": 1750, "C": 1700, "D": 1650},
        )
        assert ranked == ["B", "C", "A", "D"]


# ── rank_group: all-4-tied falls to overall criteria ─────────────────────────


def _all_four_tied_on_points() -> list[GroupMatch]:
    """All 4 teams end on 4 pts (1W 1D 1L pattern).

    Realistic scenario: each team wins one, loses one, draws one in a cycle.
      A beats B 1-0, draws C 0-0, loses to D 0-1
      B beats C 1-0, draws D 0-0, loses to A (above)
      C beats D 1-0, draws A (above), loses to B (above)
      D beats A (above), draws B (above), loses to C (above)

    Each: 1W (3) + 1D (1) + 1L (0) = 4 pts.

    When ALL 4 are tied, H2H subset == full group, so H2H stats == overall.
    H2H block can't separate (all equal) → falls to Overall GD/GF.

    Overall GD: A: 1-0+0-0+0-1 = 0. B: 1-0+0-0+0-1 = 0. C: 1-0+0-0+0-1 = 0.
    D: 1-0+0-0+0-1 = 0. All zero. → Overall GF: A: 1+0+0=1, B: 1+0+0=1,
    C: 1+0+0=1, D: 1+0+0=1. All same. → Fair-play (all 0) → FIFA ranking.

    Assert: order matches final_tiebreaker descending.
    """
    return [
        GroupMatch("A", "B", 1, 0),
        GroupMatch("A", "C", 0, 0),
        GroupMatch("D", "A", 1, 0),
        GroupMatch("B", "C", 1, 0),
        GroupMatch("B", "D", 0, 0),
        GroupMatch("C", "D", 1, 0),
    ]


class TestRankGroupAllFourTied:
    def test_all_tied_falls_to_fifa_ranking(self) -> None:
        ranked = rank_group(
            _all_four_tied_on_points(),
            final_tiebreaker={"A": 2000, "B": 1900, "C": 1800, "D": 1700},
        )
        assert ranked == ["A", "B", "C", "D"]

    def test_fifa_ranking_reversed_flips_order(self) -> None:
        """Sanity: swap FIFA ranking, order flips end-to-end."""
        ranked = rank_group(
            _all_four_tied_on_points(),
            final_tiebreaker={"A": 1700, "B": 1800, "C": 1900, "D": 2000},
        )
        assert ranked == ["D", "C", "B", "A"]


# ── rank_group: fair-play breaks tie before FIFA ranking ─────────────────────


class TestRankGroupFairPlayTiebreaker:
    def test_fair_play_wins_over_fifa_ranking(self) -> None:
        """Two teams identical on all overall criteria. Better conduct (less negative)
        ranks higher even if FIFA ranking would say otherwise.
        """
        matches = [
            GroupMatch("A", "B", 1, 1, home_yellows=0, away_yellows=5),
            GroupMatch("A", "C", 2, 0),
            GroupMatch("B", "C", 2, 0),
        ]
        # A: 1D + 1W = 4 pts, GD=+2, GF=3, fair_play=0
        # B: 1D + 1W = 4 pts, GD=+2, GF=3, fair_play=-5
        # C: 0 pts
        ranked = rank_group(matches, final_tiebreaker={"A": 1500, "B": 1800, "C": 1700})
        assert ranked[0] == "A"  # fair-play 0 beats fair-play -5 (B has higher Elo)
        assert ranked[1] == "B"


# ── rank_best_thirds: no H2H ─────────────────────────────────────────────────


class TestRankBestThirds:
    def test_orders_by_points_then_gd(self) -> None:
        thirds = [("X", "A"), ("Y", "B"), ("Z", "C")]
        stats = {
            "X": TeamStats(team="X", points=4, goals_for=3, goals_against=2),
            "Y": TeamStats(team="Y", points=4, goals_for=5, goals_against=2),  # better GD
            "Z": TeamStats(team="Z", points=3, goals_for=10, goals_against=0),  # lower pts
        }
        ranked = rank_best_thirds(thirds, stats, final_tiebreaker={"X": 0, "Y": 0, "Z": 0})
        # Y (+3 GD, 4 pts) > X (+1 GD, 4 pts) > Z (3 pts)
        assert ranked == [("Y", "B"), ("X", "A"), ("Z", "C")]


# ── collect_thirds helper ─────────────────────────────────────────────────────


class TestCollectThirds:
    def test_extracts_third_place_per_group(self) -> None:
        rankings = {
            "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
            "B": ["Canada", "Switzerland", "Qatar", "BiH"],
        }
        thirds = collect_thirds(rankings)
        assert thirds == [("South Korea", "A"), ("Qatar", "B")]


# ── R32 Annex C 8-constraint regression test ──────────────────────────────────


# Verified via independent claude.ai cross-check 2026-05-20. Each entry:
# "1X" → set of group letters whose 3rd-place finisher Group X's winner CAN face.
# (From FIFA Regulations Annex C; only winners A/B/D/E/G/I/K/L pair with thirds.)
FIFA_ANNEX_C_ALLOWED = {
    "1A": set("CEFHI"),  # Match 79
    "1B": set("EFGIJ"),  # Match 85
    "1D": set("BEFIJ"),  # Match 81
    "1E": set("ABCDF"),  # Match 74
    "1G": set("AEHIJ"),  # Match 82
    "1I": set("CDFGH"),  # Match 77
    "1K": set("DEIJL"),  # Match 87
    "1L": set("EHIJK"),  # Match 80
}


@pytest.fixture
def r32_table(mc_simu_data_dir: Path) -> dict:
    path = mc_simu_data_dir / "r32_seeding_table.json"
    with path.open() as f:
        return json.load(f)


class TestR32AnnexCRegression:
    def test_495_combinations_present(self, r32_table: dict) -> None:
        """C(12, 8) = 495 ways to pick 8 third-place groups from 12."""
        assert len(r32_table) == 495

    def test_all_keys_are_8_unique_sorted_group_letters(self, r32_table: dict) -> None:
        for key in r32_table:
            assert len(key) == 8
            assert len(set(key)) == 8
            assert all(c in "ABCDEFGHIJKL" for c in key)
            assert key == "".join(sorted(key))

    def test_each_combo_has_exactly_8_winner_slots(self, r32_table: dict) -> None:
        expected = set(FIFA_ANNEX_C_ALLOWED.keys())
        for combo, mapping in r32_table.items():
            assert set(mapping.keys()) == expected, f"combo {combo} missing/extra winners"

    def test_thirds_within_combo_used_exactly_once(self, r32_table: dict) -> None:
        for combo, mapping in r32_table.items():
            used = {v[1] for v in mapping.values()}  # "3H" → "H"
            assert used == set(combo), f"combo {combo}: thirds {used} != combo {set(combo)}"

    def test_all_pairings_respect_fifa_allowed_sets(self, r32_table: dict) -> None:
        """The 8 schema constraints (matches 74/77/79/80/81/82/85/87) must hold
        for every one of the 495 combinations.
        """
        violations = []
        for combo, mapping in r32_table.items():
            for winner, third in mapping.items():
                third_letter = third[1]
                if third_letter not in FIFA_ANNEX_C_ALLOWED[winner]:
                    violations.append((combo, winner, third))
        assert not violations, f"FIFA Annex C violations: {violations[:5]}"
