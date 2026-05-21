"""Tests for src/mc_simu/tournaments/euro_24teams.py (Euro 2020/2024).

Euro 2016 dropped from this adapter per Phase 3 decision #38 — its R16 geometry
puts thirds against 1A/1B/1C/1D winners, not the 1B/1C/1E/1F pattern adopted
from 2020 onward. Adding Euro 2016 requires a separate adapter or a branched
bracket; deferred until Phase 4 LOTO-CV demonstrates the missing edition matters.

Coverage:
1. Editions data — 24 teams per edition, 6 groups × 4
2. Bracket constants — R16 (8) + later (7) = 15
3. seed_thirds — all 15 of C(6,4) combos resolve, output respects allowed sets
4. seed_thirds is bijective (4 thirds → 4 winners, no duplicates)
5. Per-edition smoke
6. Probability invariants
7. Reproducibility
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mc_simu.tournaments.euro_24teams import (
    EDITIONS,
    GROUP_LETTERS,
    LATER_ROUNDS,
    R16_BRACKET,
    THIRD_ALLOWED_PER_WINNER,
    UEFA_ANNEX_C,
    build_sim_context,
    load_edition_bundle,
    run_monte_carlo,
    seed_thirds,
    simulate_one,
)


# ── Editions data integrity ──────────────────────────────────────────────────


class TestEditions:
    @pytest.mark.parametrize("year", [2020, 2024])
    def test_24_teams_6_groups(self, year: int) -> None:
        ed = EDITIONS[year]
        groups = ed["groups"]
        assert set(groups.keys()) == set(GROUP_LETTERS)
        all_teams = [t for grp in groups.values() for t in grp]
        assert len(all_teams) == 24
        assert len(set(all_teams)) == 24
        for letter, teams in groups.items():
            assert len(teams) == 4


# ── Bracket constants ────────────────────────────────────────────────────────


class TestBracketDefinition:
    def test_r16_has_8_matches(self) -> None:
        assert len(R16_BRACKET) == 8
        assert sorted(m[0] for m in R16_BRACKET) == list(range(1, 9))

    def test_r16_has_4_static_and_4_third_slots(self) -> None:
        """8 R16 matches: 4 with static W vs RU + 4 with W vs T3."""
        t3_count = sum(1 for _id, l, r in R16_BRACKET for kind, _ in (l, r) if kind == "T3")
        ru_count = sum(1 for _id, l, r in R16_BRACKET for kind, _ in (l, r) if kind == "RU")
        w_count = sum(1 for _id, l, r in R16_BRACKET for kind, _ in (l, r) if kind == "W")
        assert t3_count == 4
        assert ru_count == 6  # 4 static W-vs-RU matches have 1 RU each = 4; plus 2 RU-vs-RU = 4
        # Static: M1 (RU,RU), M2 (W,RU), M5 (RU,RU), M8 (W,RU) → 4 RU positions from W-RU + 4 from RU-RU = ... hmm let me recount
        # M1: (RU,A) (RU,B) → 2 RUs
        # M2: (W,A) (RU,C) → 1 RU, 1 W
        # M3: (W,C) (T3,C) → 1 W, 1 T3
        # M4: (W,B) (T3,B) → 1 W, 1 T3
        # M5: (RU,D) (RU,E) → 2 RUs
        # M6: (W,F) (T3,F) → 1 W, 1 T3
        # M7: (W,E) (T3,E) → 1 W, 1 T3
        # M8: (W,D) (RU,F) → 1 W, 1 RU
        # Total: RU=2+1+2+1=6, W=1+1+1+1+1+1=6, T3=4 ✓
        assert w_count == 6

    def test_t3_sources_use_only_allowed_winner_keys(self) -> None:
        """T3 sources should be keyed by the 4 winners that have 3rd-slot fixtures."""
        t3_keys = {key for _id, l, r in R16_BRACKET for kind, key in (l, r) if kind == "T3"}
        assert t3_keys == set(THIRD_ALLOWED_PER_WINNER.keys())  # {B, C, E, F}

    def test_later_rounds_7_matches(self) -> None:
        assert len(LATER_ROUNDS) == 7
        assert sorted(m[0] for m in LATER_ROUNDS) == list(range(9, 16))


# ── seed_thirds — constraint solver ─────────────────────────────────────────


class TestSeedThirds:
    def test_all_15_combos_resolve(self) -> None:
        combos = list(combinations("ABCDEF", 4))
        assert len(combos) == 15
        for c in combos:
            assignment = seed_thirds(frozenset(c))
            # 4 winners → 4 thirds, bijective
            assert set(assignment.keys()) == set("BCEF")
            assert set(assignment.values()) == set(c)
            # Each assignment respects allowed set
            for winner, third in assignment.items():
                assert third in THIRD_ALLOWED_PER_WINNER[winner], (
                    f"combo {c}: 1{winner} cannot face 3{third} per allowed set"
                )

    def test_invalid_combo_size_raises(self) -> None:
        with pytest.raises(ValueError, match="must have 4 groups"):
            seed_thirds(frozenset("ABC"))

    def test_uefa_annex_c_table_complete(self) -> None:
        """All 15 combos from C(6,4) are present in the hardcoded UEFA table."""
        combos = {frozenset(c) for c in combinations("ABCDEF", 4)}
        assert set(UEFA_ANNEX_C.keys()) == combos

    def test_euro_2024_canonical_pairing(self) -> None:
        """Euro 2024 actual: combo {C,D,E,F} →
        Spain(1B) vs Georgia(3F), England(1C) vs Slovakia(3E),
        Romania(1E) vs Netherlands(3D), Portugal(1F) vs Slovenia(3C).
        """
        assignment = seed_thirds(frozenset("CDEF"))
        assert assignment == {"B": "F", "C": "E", "E": "D", "F": "C"}


# ── Per-edition smoke ───────────────────────────────────────────────────────


def _ratings_before(year: int, mc_simu_data_dir: Path) -> dict[str, float]:
    elo_df = pd.read_csv(mc_simu_data_dir / "elo_history.csv")
    elo_df["date"] = pd.to_datetime(elo_df["date"])
    # Euro 2020 actually played June 2021
    cutoff_year = 2021 if year == 2020 else year
    cutoff = pd.Timestamp(year=cutoff_year, month=6, day=1)
    snap = elo_df[elo_df["date"] < cutoff]
    return snap.sort_values("date").groupby("team").tail(1).set_index("team")["rating_after"].to_dict()


class TestPerEditionSmoke:
    @pytest.mark.parametrize("year", [2020, 2024])
    def test_all_teams_have_ratings(self, year: int, mc_simu_data_dir: Path) -> None:
        ratings = _ratings_before(year, mc_simu_data_dir)
        bundle = load_edition_bundle(year, ratings)
        all_teams = {t for grp in EDITIONS[year]["groups"].values() for t in grp}
        missing = all_teams - set(bundle.ratings)
        assert not missing, f"Euro {year}: missing Elo: {missing}"

    @pytest.mark.parametrize("year", [2020, 2024])
    def test_one_iter_smoke(self, year: int, mc_simu_data_dir: Path) -> None:
        ratings = _ratings_before(year, mc_simu_data_dir)
        bundle = load_edition_bundle(year, ratings)
        sim = build_sim_context(bundle)
        rng = np.random.default_rng(42)
        champion, group_winners = simulate_one(sim, rng)
        all_teams = {t for grp in EDITIONS[year]["groups"].values() for t in grp}
        assert champion in all_teams
        assert set(group_winners.keys()) == set(GROUP_LETTERS)

    @pytest.mark.parametrize("year", [2020, 2024])  # cheaper subset
    def test_mc_invariants(self, year: int, mc_simu_data_dir: Path) -> None:
        ratings = _ratings_before(year, mc_simu_data_dir)
        bundle = load_edition_bundle(year, ratings)
        result = run_monte_carlo(bundle, n_iterations=300, seed=42)
        total = sum(s["mc_fair_prob"] for s in result["champion"].values())
        assert abs(total - 1.0) < 1e-9
        for g in GROUP_LETTERS:
            assert abs(sum(s["mc_fair_prob"] for s in result["group_winners"][g].values()) - 1.0) < 1e-9


# ── Reproducibility ──────────────────────────────────────────────────────────


class TestReproducibility:
    def test_same_seed_same_output(self, mc_simu_data_dir: Path) -> None:
        ratings = _ratings_before(2024, mc_simu_data_dir)
        bundle = load_edition_bundle(2024, ratings)
        r1 = run_monte_carlo(bundle, n_iterations=200, seed=42)
        r2 = run_monte_carlo(bundle, n_iterations=200, seed=42)
        for team in r1["champion"]:
            assert r1["champion"][team]["mc_fair_prob"] == r2["champion"][team]["mc_fair_prob"]
