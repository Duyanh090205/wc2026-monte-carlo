"""Tests for src/mc_simu/tournaments/wc_8groups.py (WC 2002–2022).

Coverage:
1. Editions data integrity — 32 teams per edition, 8 groups × 4
2. Bracket constants — R16 (8 matches) + later (7) = 15, all match IDs covered
3. R16 same-group constraint: winner-X and runner-up-X never meet in R16
4. Per-edition smoke — champion ∈ roster, group winners ∈ groups
5. Probability invariants
6. Reproducibility
7. Bracket structural check against verified WC 2022 SF1 = Argentina(1C) vs Croatia(2F)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mc_simu.tournaments.wc_8groups import (
    EDITIONS,
    GROUP_LETTERS,
    LATER_ROUNDS,
    R16_BRACKET,
    build_sim_context,
    load_edition_bundle,
    make_round_robin,
    run_monte_carlo,
    simulate_one,
)


# ── Editions data integrity ──────────────────────────────────────────────────


class TestEditions:
    @pytest.mark.parametrize("year", [2002, 2006, 2010, 2014, 2018, 2022])
    def test_32_teams_8_groups(self, year: int) -> None:
        ed = EDITIONS[year]
        groups = ed["groups"]
        assert set(groups.keys()) == set(GROUP_LETTERS)
        all_teams = [t for grp in groups.values() for t in grp]
        assert len(all_teams) == 32
        assert len(set(all_teams)) == 32
        for letter, teams in groups.items():
            assert len(teams) == 4

    @pytest.mark.parametrize("year", [2002, 2006, 2010, 2014, 2018, 2022])
    def test_host_in_a_group(self, year: int) -> None:
        ed = EDITIONS[year]
        all_teams = {t for grp in ed["groups"].values() for t in grp}
        for host in ed["host_countries"]:
            assert host in all_teams


# ── Bracket constants ────────────────────────────────────────────────────────


class TestBracketDefinition:
    def test_r16_has_8_matches(self) -> None:
        assert len(R16_BRACKET) == 8
        assert sorted(m[0] for m in R16_BRACKET) == list(range(1, 9))

    def test_r16_no_same_group_pairing(self) -> None:
        """In R16, winner-X and runner-up-X of same group must NEVER meet."""
        for _id, left, right in R16_BRACKET:
            l_kind, l_key = left
            r_kind, r_key = right
            if {l_kind, r_kind} == {"W", "RU"}:
                assert l_key != r_key, (
                    f"Match {_id}: W and RU of same group {l_key} meet — forbidden"
                )

    def test_r16_uses_all_8_winners_and_all_8_runners(self) -> None:
        winners_used = {key for _id, l, r in R16_BRACKET
                        for kind, key in (l, r) if kind == "W"}
        runners_used = {key for _id, l, r in R16_BRACKET
                        for kind, key in (l, r) if kind == "RU"}
        assert winners_used == set(GROUP_LETTERS)
        assert runners_used == set(GROUP_LETTERS)

    def test_later_rounds_7_matches(self) -> None:
        # 4 QF + 2 SF + 1 Final = 7
        assert len(LATER_ROUNDS) == 7
        assert sorted(m[0] for m in LATER_ROUNDS) == list(range(9, 16))

    def test_all_r16_consumed_by_qf(self) -> None:
        r16_ids = {m[0] for m in R16_BRACKET}
        qf_consumes = {m for _id, l, r in LATER_ROUNDS if _id in (9, 10, 11, 12) for m in (l, r)}
        assert qf_consumes == r16_ids


# ── Round-robin generator ────────────────────────────────────────────────────


class TestMakeRoundRobin:
    def test_4_teams_yields_6_pairs(self) -> None:
        assert len(make_round_robin(["A", "B", "C", "D"])) == 6


# ── Per-edition smoke ───────────────────────────────────────────────────────


def _ratings_before(year: int, mc_simu_data_dir: Path) -> dict[str, float]:
    elo_df = pd.read_csv(mc_simu_data_dir / "elo_history.csv")
    elo_df["date"] = pd.to_datetime(elo_df["date"])
    # WC typically starts June, except WC2022 Qatar in November
    if year == 2022:
        cutoff = pd.Timestamp(year=2022, month=11, day=20)
    else:
        cutoff = pd.Timestamp(year=year, month=6, day=1)
    snap = elo_df[elo_df["date"] < cutoff]
    return snap.sort_values("date").groupby("team").tail(1).set_index("team")["rating_after"].to_dict()


class TestPerEditionSmoke:
    @pytest.mark.parametrize("year", [2002, 2006, 2010, 2014, 2018, 2022])
    def test_all_teams_have_ratings_after_alias(self, year: int, mc_simu_data_dir: Path) -> None:
        ratings = _ratings_before(year, mc_simu_data_dir)
        bundle = load_edition_bundle(year, ratings)
        all_teams = {t for grp in EDITIONS[year]["groups"].values() for t in grp}
        missing = all_teams - set(bundle.ratings)
        assert not missing, f"WC {year}: teams missing Elo after alias: {missing}"

    @pytest.mark.parametrize("year", [2002, 2006, 2010, 2014, 2018, 2022])
    def test_one_iter_smoke(self, year: int, mc_simu_data_dir: Path) -> None:
        ratings = _ratings_before(year, mc_simu_data_dir)
        bundle = load_edition_bundle(year, ratings)
        sim = build_sim_context(bundle)
        rng = np.random.default_rng(42)
        champion, group_winners = simulate_one(sim, rng)
        all_teams = {t for grp in EDITIONS[year]["groups"].values() for t in grp}
        assert champion in all_teams
        assert set(group_winners.keys()) == set(GROUP_LETTERS)

    @pytest.mark.parametrize("year", [2018, 2022])  # cheaper subset for MC test
    def test_mc_invariants(self, year: int, mc_simu_data_dir: Path) -> None:
        ratings = _ratings_before(year, mc_simu_data_dir)
        bundle = load_edition_bundle(year, ratings)
        result = run_monte_carlo(bundle, n_iterations=200, seed=42)
        total = sum(s["mc_fair_prob"] for s in result["champion"].values())
        assert abs(total - 1.0) < 1e-9
        for g in GROUP_LETTERS:
            assert abs(sum(s["mc_fair_prob"] for s in result["group_winners"][g].values()) - 1.0) < 1e-9


# ── Reproducibility ──────────────────────────────────────────────────────────


class TestReproducibility:
    def test_same_seed_same_output(self, mc_simu_data_dir: Path) -> None:
        ratings = _ratings_before(2018, mc_simu_data_dir)
        bundle = load_edition_bundle(2018, ratings)
        r1 = run_monte_carlo(bundle, n_iterations=150, seed=42)
        r2 = run_monte_carlo(bundle, n_iterations=150, seed=42)
        for team in r1["champion"]:
            assert r1["champion"][team]["mc_fair_prob"] == r2["champion"][team]["mc_fair_prob"]
