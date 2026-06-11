"""Tests for src/mc_simu/fetch_played.py — football-data.org -> wc2026_played.csv.

All offline: API payloads are synthetic dicts mirroring the v4 /matches schema.
R32 resolution runs against the real r32_seeding_table.json with synthetic
group results constructed so rankings are decisive (no tiebreaker reliance).
"""

from __future__ import annotations

import json
from pathlib import Path

from mc_simu.fetch_played import (
    canonical_team,
    find_ko_match_id,
    group_fixture_issue,
    ko_winner,
    load_group_fixture_dates,
    merge_rows,
    plausible_score,
    resolve_r32_pairings,
)
from mc_simu.tournaments.wc2026 import GROUP_LETTERS, R32_BRACKET

DATA = Path(__file__).resolve().parent.parent.parent / "data" / "mc_simu"


# ── Team name normalization ───────────────────────────────────────────────────


class TestCanonicalTeam:
    BY_NORM = {"czechia": "Czechia", "united states": "United States",
               "ivory coast": "Ivory Coast", "south korea": "South Korea"}

    def test_exact_model_name(self) -> None:
        assert canonical_team({"name": "Czechia"}, self.BY_NORM) == "Czechia"

    def test_alias_czech_republic(self) -> None:
        assert canonical_team({"name": "Czech Republic"}, self.BY_NORM) == "Czechia"

    def test_alias_diacritics(self) -> None:
        assert canonical_team({"name": "Côte d'Ivoire"}, self.BY_NORM) == "Ivory Coast"

    def test_alias_usa(self) -> None:
        assert canonical_team({"name": "USA"}, self.BY_NORM) == "United States"

    def test_shortname_fallback(self) -> None:
        team = {"name": "Korea Republic FA XI", "shortName": "South Korea"}
        assert canonical_team(team, self.BY_NORM) == "South Korea"

    def test_unknown_returns_none(self) -> None:
        assert canonical_team({"name": "Atlantis"}, self.BY_NORM) is None


# ── KO winner extraction ──────────────────────────────────────────────────────


class TestKoWinner:
    def test_home_win(self) -> None:
        m = {"score": {"winner": "HOME_TEAM"}}
        assert ko_winner(m, "France", "Senegal") == "France"

    def test_away_win(self) -> None:
        m = {"score": {"winner": "AWAY_TEAM"}}
        assert ko_winner(m, "France", "Senegal") == "Senegal"

    def test_penalties_decide_when_winner_missing(self) -> None:
        m = {"score": {"winner": "DRAW", "penalties": {"home": 3, "away": 4}}}
        assert ko_winner(m, "France", "Senegal") == "Senegal"

    def test_no_winner_no_penalties(self) -> None:
        assert ko_winner({"score": {"winner": "DRAW"}}, "A", "B") is None


# ── R32 bracket resolution from group results ─────────────────────────────────


def _synthetic_groups() -> tuple[dict[str, list[str]], dict[frozenset, dict[str, int]]]:
    """12 groups of fake teams with decisive results.

    In group g: g1 > g2 > g3 > g4 strictly by points. The third's win margin
    decreases with group index, so thirds from groups A-H advance on goal
    difference (points all equal at 3).
    """
    membership = {g: [f"{g}{i}" for i in range(1, 5)] for g in GROUP_LETTERS}
    scores: dict[frozenset, dict[str, int]] = {}
    for gi, g in enumerate(GROUP_LETTERS):
        t1, t2, t3, t4 = membership[g]
        results = [
            (t1, t2, 3, 0), (t1, t3, 3, 0), (t1, t4, 3, 0),
            (t2, t3, 2, 0), (t2, t4, 2, 0),
            (t3, t4, 12 - gi, 0),
        ]
        for h, a, hg, ag in results:
            scores[frozenset((h, a))] = {h: hg, a: ag}
    return membership, scores


class TestResolveR32:
    def setup_method(self) -> None:
        self.membership, self.scores = _synthetic_groups()
        with (DATA / "r32_seeding_table.json").open() as f:
            self.r32_table = json.load(f)
        self.tiebreaker = {t: 1500.0 for ts in self.membership.values() for t in ts}

    def test_incomplete_groups_returns_none(self) -> None:
        partial = dict(self.scores)
        partial.pop(frozenset(("A1", "A2")))
        assert resolve_r32_pairings(partial, self.membership,
                                    self.r32_table, self.tiebreaker) is None

    def test_full_groups_resolve_16_pairs_covering_32_teams(self) -> None:
        pairs = resolve_r32_pairings(self.scores, self.membership,
                                     self.r32_table, self.tiebreaker)
        assert pairs is not None
        assert sorted(pairs) == [mid for mid, _, _ in R32_BRACKET]
        all_teams = set().union(*pairs.values())
        assert len(all_teams) == 32

    def test_static_and_winner_slots(self) -> None:
        pairs = resolve_r32_pairings(self.scores, self.membership,
                                     self.r32_table, self.tiebreaker)
        assert pairs[73] == frozenset(("A2", "B2"))
        assert "E1" in pairs[74]
        third = next(iter(pairs[74] - {"E1"}))
        assert third in {f"{g}3" for g in "ABCDEFGH"}


# ── Source validation guards ──────────────────────────────────────────────────


class TestSourceGuards:
    GROUP_OF = {"Mexico": "A", "South Africa": "A", "Canada": "B"}
    DATES = {frozenset(("Mexico", "South Africa")): "2026-06-11"}

    def test_plausible_scores(self) -> None:
        assert plausible_score(2, 1)
        assert plausible_score(0, 0)
        assert not plausible_score(16, 0)
        assert not plausible_score(-1, 2)

    def test_same_group_on_schedule_is_valid(self) -> None:
        assert group_fixture_issue("Mexico", "South Africa", "2026-06-11T20:00:00Z",
                                   self.GROUP_OF, self.DATES) is None

    def test_cross_group_pairing_rejected(self) -> None:
        issue = group_fixture_issue("Mexico", "Canada", "2026-06-11T20:00:00Z",
                                    self.GROUP_OF, self.DATES)
        assert issue is not None and "not a scheduled group fixture" in issue

    def test_unknown_team_rejected(self) -> None:
        assert group_fixture_issue("Atlantis", "Mexico", "2026-06-11T20:00:00Z",
                                   self.GROUP_OF, self.DATES) is not None

    def test_date_weeks_off_rejected(self) -> None:
        issue = group_fixture_issue("Mexico", "South Africa", "2026-07-30T20:00:00Z",
                                    self.GROUP_OF, self.DATES)
        assert issue is not None and "scheduled" in issue

    def test_one_day_slack_tolerated(self) -> None:
        assert group_fixture_issue("Mexico", "South Africa", "2026-06-12T01:00:00Z",
                                   self.GROUP_OF, self.DATES) is None

    def test_real_fixture_table_covers_all_72(self) -> None:
        dates = load_group_fixture_dates()
        assert len(dates) == 72
        assert all(d and d != "TBD" for d in dates.values())


# ── KO match id assignment ────────────────────────────────────────────────────


class TestFindKoMatchId:
    def test_r32_lookup(self) -> None:
        r32 = {73: frozenset(("A2", "B2")), 75: frozenset(("F1", "C2"))}
        assert find_ko_match_id(frozenset(("A2", "B2")), r32, {}) == 73

    def test_later_round_from_winners(self) -> None:
        ko = {73: "A2", 75: "F1"}
        assert find_ko_match_id(frozenset(("A2", "F1")), None, ko) == 89

    def test_unplaceable_returns_none(self) -> None:
        assert find_ko_match_id(frozenset(("X", "Y")), None, {}) is None


# ── CSV merge ─────────────────────────────────────────────────────────────────


class TestMergeRows:
    GROUP_ROW = {"stage": "group", "match_id": "", "home_team": "Mexico",
                 "away_team": "South Africa", "home_goals": "2",
                 "away_goals": "1", "winner": ""}

    def test_upsert_overwrites_same_key(self) -> None:
        stale = dict(self.GROUP_ROW, home_goals="0")
        rows, n = merge_rows([stale], [self.GROUP_ROW])
        assert n == 1
        assert rows == [self.GROUP_ROW]

    def test_identical_row_is_noop(self) -> None:
        rows, n = merge_rows([self.GROUP_ROW], [dict(self.GROUP_ROW)])
        assert n == 0
        assert rows == [self.GROUP_ROW]

    def test_manual_rows_preserved_and_ko_sorted_last(self) -> None:
        manual_ko = {"stage": "ko", "match_id": "73", "home_team": "A",
                     "away_team": "B", "home_goals": "1", "away_goals": "0",
                     "winner": "A"}
        rows, n = merge_rows([manual_ko], [self.GROUP_ROW])
        assert n == 1
        assert rows == [self.GROUP_ROW, manual_ko]
