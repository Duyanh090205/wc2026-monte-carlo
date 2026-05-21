"""Tests for src/mc_simu/tournaments/euro_16teams.py (Euro 2004/2008/2012).

Coverage:
1. Editions data integrity — 16 teams per edition, 4 groups × 4
2. Bracket constants — QF (4 matches) + SF (2) + Final (1) = 7 matches; all match IDs referenced exactly once in later rounds
3. Round-robin generator — 6 unique pairs per 4-team group
4. Single-iter smoke per edition — champion ∈ roster, group winners ∈ group
5. Probability invariants — Σ champion = 1, per-group winner sums = 1
6. Reproducibility — same seed → same output
7. Baseline plugs in
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mc_simu.baseline import predict_match_40_40_20
from mc_simu.tournaments.euro_16teams import (
    EDITIONS,
    GROUP_LETTERS,
    LATER_ROUNDS,
    QF_BRACKET,
    build_sim_context,
    load_edition_bundle,
    make_round_robin,
    run_monte_carlo,
    simulate_one,
)


# ── Editions data integrity ──────────────────────────────────────────────────


class TestEditions:
    @pytest.mark.parametrize("year", [2004, 2008, 2012])
    def test_16_teams_4_groups(self, year: int) -> None:
        ed = EDITIONS[year]
        groups = ed["groups"]
        assert set(groups.keys()) == set(GROUP_LETTERS)
        all_teams = [t for grp in groups.values() for t in grp]
        assert len(all_teams) == 16
        assert len(set(all_teams)) == 16  # no duplicates across groups
        for letter, teams in groups.items():
            assert len(teams) == 4, f"Group {letter} ({year}) has {len(teams)} teams"

    @pytest.mark.parametrize("year", [2004, 2008, 2012])
    def test_host_in_a_group(self, year: int) -> None:
        ed = EDITIONS[year]
        all_teams = {t for grp in ed["groups"].values() for t in grp}
        for host in ed["host_countries"]:
            assert host in all_teams, f"Host {host} not in any Euro {year} group"


# ── Bracket constants ────────────────────────────────────────────────────────


class TestBracketDefinition:
    def test_qf_has_4_matches(self) -> None:
        assert len(QF_BRACKET) == 4
        assert sorted(m[0] for m in QF_BRACKET) == [1, 2, 3, 4]

    def test_qf_sources_cover_all_8_advancing_teams(self) -> None:
        """4 winners + 4 runners-up across all 4 groups."""
        sources = [s for _id, l, r in QF_BRACKET for s in (l, r)]
        winners_used = {key for kind, key in sources if kind == "W"}
        runners_used = {key for kind, key in sources if kind == "RU"}
        assert winners_used == set(GROUP_LETTERS)
        assert runners_used == set(GROUP_LETTERS)

    def test_qf_cross_bracket_constraint(self) -> None:
        """No (W, A) vs (RU, A) — winner and runner-up of SAME group don't meet in QF."""
        for _id, left, right in QF_BRACKET:
            if left[0] == "W" and right[0] == "RU":
                assert left[1] != right[1], "Same-group W vs RU should not meet in QF"
            if left[0] == "RU" and right[0] == "W":
                assert left[1] != right[1]

    def test_later_rounds_3_matches(self) -> None:
        assert len(LATER_ROUNDS) == 3
        assert sorted(m[0] for m in LATER_ROUNDS) == [5, 6, 7]

    def test_all_qf_consumed_by_sf(self) -> None:
        qf_ids = {m[0] for m in QF_BRACKET}
        sf_consumes = {m for _id, l, r in LATER_ROUNDS if _id in (5, 6) for m in (l, r)}
        assert sf_consumes == qf_ids

    def test_sf_cross_bracket_preserves_group_halves(self) -> None:
        """UEFA cross-bracket: A/B half stays separated from C/D half until Final.

        Anchor against accidental regression to SF1=W(QF1)vs W(QF2). Verified
        against Euro 2008 actual (GER[A-side] vs TUR[A-side]; RUS[C-side] vs
        ESP[C-side]) and Euro 2012 (POR/ESP, GER/ITA).

        QF1 (W_A vs RU_B) and QF3 (W_B vs RU_A) both involve groups A+B.
        QF2 (W_C vs RU_D) and QF4 (W_D vs RU_C) both involve groups C+D.
        SF1 must pair QF1+QF3 (A/B half); SF2 must pair QF2+QF4 (C/D half).
        """
        sf_pairings = [(m[1], m[2]) for m in LATER_ROUNDS if m[0] in (5, 6)]
        sf_pairings_canonical = [tuple(sorted(p)) for p in sf_pairings]
        assert (1, 3) in sf_pairings_canonical, "SF1 should pair QF1+QF3 (A/B half)"
        assert (2, 4) in sf_pairings_canonical, "SF2 should pair QF2+QF4 (C/D half)"


# ── Round-robin generator ────────────────────────────────────────────────────


class TestMakeRoundRobin:
    def test_4_teams_yields_6_pairs(self) -> None:
        pairs = make_round_robin(["A", "B", "C", "D"])
        assert len(pairs) == 6

    def test_no_self_pair(self) -> None:
        pairs = make_round_robin(["A", "B", "C", "D"])
        assert all(h != a for h, a in pairs)

    def test_all_pairs_distinct(self) -> None:
        pairs = make_round_robin(["A", "B", "C", "D"])
        # Treat (X, Y) == (Y, X) for distinctness check
        canonical = {frozenset(p) for p in pairs}
        assert len(canonical) == 6


# ── Per-edition smoke (all 3 editions) ──────────────────────────────────────


def _ratings_before(year: int, mc_simu_data_dir: Path) -> dict[str, float]:
    """Latest Elo per team strictly before the year's tournament start."""
    elo_df = pd.read_csv(mc_simu_data_dir / "elo_history.csv")
    elo_df["date"] = pd.to_datetime(elo_df["date"])
    # All Euros start June or thereabouts; June 1 is a safe cutoff
    cutoff = pd.Timestamp(year=year, month=6, day=1)
    snap = elo_df[elo_df["date"] < cutoff]
    return snap.sort_values("date").groupby("team").tail(1).set_index("team")["rating_after"].to_dict()


class TestPerEditionSmoke:
    @pytest.mark.parametrize("year", [2004, 2008, 2012])
    def test_all_teams_have_ratings(self, year: int, mc_simu_data_dir: Path) -> None:
        ratings = _ratings_before(year, mc_simu_data_dir)
        teams = {t for grp in EDITIONS[year]["groups"].values() for t in grp}
        missing = teams - set(ratings)
        assert not missing, f"Euro {year}: teams missing Elo: {missing}"

    @pytest.mark.parametrize("year", [2004, 2008, 2012])
    def test_one_iter_returns_valid_champion(self, year: int, mc_simu_data_dir: Path) -> None:
        ratings = _ratings_before(year, mc_simu_data_dir)
        bundle = load_edition_bundle(year, ratings)
        sim = build_sim_context(bundle)
        rng = np.random.default_rng(42)
        champion, group_winners = simulate_one(sim, rng)
        all_teams = {t for grp in EDITIONS[year]["groups"].values() for t in grp}
        assert champion in all_teams
        assert set(group_winners.keys()) == set(GROUP_LETTERS)
        for letter, winner in group_winners.items():
            assert winner in EDITIONS[year]["groups"][letter]

    @pytest.mark.parametrize("year", [2004, 2008, 2012])
    def test_mc_invariants(self, year: int, mc_simu_data_dir: Path) -> None:
        ratings = _ratings_before(year, mc_simu_data_dir)
        bundle = load_edition_bundle(year, ratings)
        result = run_monte_carlo(bundle, n_iterations=300, seed=42)
        total_champ = sum(s["mc_fair_prob"] for s in result["champion"].values())
        assert abs(total_champ - 1.0) < 1e-9
        for g in GROUP_LETTERS:
            group_total = sum(s["mc_fair_prob"] for s in result["group_winners"][g].values())
            assert abs(group_total - 1.0) < 1e-9, f"Group {g} sum: {group_total}"


# ── Reproducibility ──────────────────────────────────────────────────────────


class TestReproducibility:
    def test_same_seed_same_output(self, mc_simu_data_dir: Path) -> None:
        ratings = _ratings_before(2004, mc_simu_data_dir)
        bundle = load_edition_bundle(2004, ratings)
        r1 = run_monte_carlo(bundle, n_iterations=200, seed=42)
        r2 = run_monte_carlo(bundle, n_iterations=200, seed=42)
        for team in r1["champion"]:
            assert r1["champion"][team]["mc_fair_prob"] == r2["champion"][team]["mc_fair_prob"]


# ── Baseline plugs in ────────────────────────────────────────────────────────


class TestBaselinePluggable:
    def test_run_with_baseline_predictor(self, mc_simu_data_dir: Path) -> None:
        ratings = _ratings_before(2004, mc_simu_data_dir)
        bundle = load_edition_bundle(2004, ratings)
        result = run_monte_carlo(
            bundle, n_iterations=300, seed=42, predictor=predict_match_40_40_20,
        )
        # Baseline should NOT have strong concentration on any team
        top_prob = max(s["mc_fair_prob"] for s in result["champion"].values())
        # 1/16 = 0.0625 if perfectly uniform; baseline still concentrates on cross-bracket luck
        assert top_prob < 0.20, f"baseline top prob {top_prob:.3f} too concentrated"
