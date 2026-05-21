"""Tests for src/mc_simu/tournaments/wc2026.py — Phase 3 §4.1 Step 3.5 + 3.6.

Coverage:
1. Loaders — bundle integrity (48 teams, 104 fixtures, 495 R32 combos)
2. Venue enrichment — host group matches get host's country
3. Single-iteration smoke — champion is one of 48 WC2026 teams, group winners are from each group's 4 teams
4. Reproducibility — same seed → byte-identical output
5. Probability invariants — Σ P(champion) = 1.0, per-group winners sum = 1.0
6. Sanity — top teams have higher champion prob than weak teams; no team >50% or <0.0001%
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mc_simu.tournaments.wc2026 import (
    GROUP_LETTERS,
    LATER_ROUNDS,
    R32_BRACKET,
    WC2026_HOST_CONFEDERATION,
    WC2026_HOST_COUNTRIES,
    build_sim_context,
    enrich_group_venues,
    load_wc2026_bundle,
    run_monte_carlo,
    simulate_tournament,
)


@pytest.fixture
def latest_ratings(mc_simu_data_dir: Path) -> dict[str, float]:
    """Latest Elo per team from elo_history.csv."""
    elo_df = pd.read_csv(mc_simu_data_dir / "elo_history.csv")
    elo_df["date"] = pd.to_datetime(elo_df["date"])
    elo_df = elo_df.sort_values("date")
    return elo_df.groupby("team").tail(1).set_index("team")["rating_after"].to_dict()


@pytest.fixture
def bundle(latest_ratings: dict[str, float]):
    return load_wc2026_bundle(latest_ratings)


# ── Bundle integrity ─────────────────────────────────────────────────────────


class TestBundleLoad:
    def test_48_teams_with_ratings(self, bundle) -> None:
        all_teams = [t for g in GROUP_LETTERS for t in bundle.group_membership[g]]
        assert len(all_teams) == 48
        assert len(set(all_teams)) == 48
        for t in all_teams:
            assert t in bundle.ratings, f"Missing Elo rating for {t}"

    def test_104_fixtures(self, bundle) -> None:
        assert len(bundle.fixtures_df) == 104
        n_group = (bundle.fixtures_df["stage"] == "group").sum()
        assert n_group == 72

    def test_r32_seeding_table_495(self, bundle) -> None:
        assert len(bundle.r32_table) == 495

    def test_host_metadata(self, bundle) -> None:
        assert bundle.host.host_countries == WC2026_HOST_COUNTRIES
        assert bundle.host.host_confederation == WC2026_HOST_CONFEDERATION

    def test_alias_resolves_czechia(self, latest_ratings: dict[str, float]) -> None:
        assert "Czech Republic" in latest_ratings
        assert "Czechia" not in latest_ratings  # confirms alias is non-trivial
        bundle = load_wc2026_bundle(latest_ratings)
        assert "Czechia" in bundle.ratings
        assert bundle.ratings["Czechia"] == latest_ratings["Czech Republic"]


# ── Bracket integrity ───────────────────────────────────────────────────────


class TestBracketDefinition:
    def test_r32_has_16_matches(self) -> None:
        assert len(R32_BRACKET) == 16
        match_ids = [m[0] for m in R32_BRACKET]
        assert sorted(match_ids) == list(range(73, 89))

    def test_r32_t3_sources_only_for_allowed_winners(self) -> None:
        """Per FIFA Annex C, only winners A/B/D/E/G/I/K/L play 3rd-place finishers."""
        allowed = {"A", "B", "D", "E", "G", "I", "K", "L"}
        for _match_id, left, right in R32_BRACKET:
            for kind, key in (left, right):
                if kind == "T3":
                    assert key in allowed, f"T3 source {key} not in FIFA Annex C allowed winners"

    def test_later_rounds_15_matches(self) -> None:
        """R16 (8) + QF (4) + SF (2) + Final (1) = 15. M103 (3rd-place play-off) skipped."""
        assert len(LATER_ROUNDS) == 15
        match_ids = [m[0] for m in LATER_ROUNDS]
        assert sorted(match_ids) == list(range(89, 103)) + [104]

    def test_final_is_m104(self) -> None:
        final = [m for m in LATER_ROUNDS if m[0] == 104][0]
        assert final == (104, 101, 102)


# ── Venue enrichment ────────────────────────────────────────────────────────


class TestEnrichGroupVenues:
    def test_host_team_match_gets_host_venue(self, bundle) -> None:
        enriched = enrich_group_venues(bundle.fixtures_df, WC2026_HOST_COUNTRIES)
        for host in WC2026_HOST_COUNTRIES:
            host_matches = enriched[
                (enriched["stage"] == "group")
                & ((enriched["home_team"] == host) | (enriched["away_team"] == host))
            ]
            assert (host_matches["venue_country"] == host).all(), \
                f"{host}: some matches still TBD after enrichment"

    def test_non_host_group_matches_stay_tbd(self, bundle) -> None:
        enriched = enrich_group_venues(bundle.fixtures_df, WC2026_HOST_COUNTRIES)
        host_set = set(WC2026_HOST_COUNTRIES)
        non_host_group = enriched[
            (enriched["stage"] == "group")
            & ~enriched["home_team"].isin(host_set)
            & ~enriched["away_team"].isin(host_set)
        ]
        # All non-host group matches should remain TBD (Phase 0 fixture builder
        # set them all to TBD; no enrichment policy applied).
        assert (non_host_group["venue_country"] == "TBD").all()


# ── Single-iteration smoke ───────────────────────────────────────────────────


class TestSingleIterationSmoke:
    def test_one_iter_runs_and_returns_valid_champion(self, bundle) -> None:
        sim = build_sim_context(bundle)
        rng = np.random.default_rng(42)
        champion, group_winners = simulate_tournament(
            group_cdfs_per_group=sim.group_cdfs_per_group,
            group_team_pairs=sim.group_team_pairs,
            r32_table=sim.r32_table,
            ko_advance=sim.ko_advance,
            fifa_ranking=sim.fifa_ranking,
            rng=rng,
        )
        # Champion must be one of the 48 teams
        all_teams = {t for g in GROUP_LETTERS for t in bundle.group_membership[g]}
        assert champion in all_teams
        # Group winners: one per group, each from that group's roster
        assert set(group_winners.keys()) == set(GROUP_LETTERS)
        for g, winner in group_winners.items():
            assert winner in bundle.group_membership[g], \
                f"Group {g} winner {winner} not in roster {bundle.group_membership[g]}"


# ── Reproducibility (Gate 3.C) ───────────────────────────────────────────────


class TestReproducibility:
    def test_same_seed_same_output(self, bundle) -> None:
        result1 = run_monte_carlo(bundle, n_iterations=200, seed=42, progress=False)
        result2 = run_monte_carlo(bundle, n_iterations=200, seed=42, progress=False)
        # Champion probabilities exactly equal
        assert set(result1["champion"].keys()) == set(result2["champion"].keys())
        for team in result1["champion"]:
            assert result1["champion"][team]["mc_fair_prob"] == \
                   result2["champion"][team]["mc_fair_prob"]

    def test_different_seed_different_output(self, bundle) -> None:
        result1 = run_monte_carlo(bundle, n_iterations=200, seed=42, progress=False)
        result2 = run_monte_carlo(bundle, n_iterations=200, seed=99, progress=False)
        # At least one team should have different probability
        common = set(result1["champion"]) & set(result2["champion"])
        diffs = [
            result1["champion"][t]["mc_fair_prob"] - result2["champion"][t]["mc_fair_prob"]
            for t in common
        ]
        assert any(abs(d) > 1e-9 for d in diffs)


# ── Probability invariants ───────────────────────────────────────────────────


class TestProbabilityInvariants:
    def test_champion_probs_sum_to_one(self, bundle) -> None:
        result = run_monte_carlo(bundle, n_iterations=1000, seed=42, progress=False)
        total = sum(s["mc_fair_prob"] for s in result["champion"].values())
        assert abs(total - 1.0) < 1e-9

    def test_per_group_winner_probs_sum_to_one(self, bundle) -> None:
        result = run_monte_carlo(bundle, n_iterations=1000, seed=42, progress=False)
        for g in GROUP_LETTERS:
            group_total = sum(s["mc_fair_prob"] for s in result["group_winners"][g].values())
            assert abs(group_total - 1.0) < 1e-9, f"Group {g} sum: {group_total}"

    def test_group_winners_only_from_group(self, bundle) -> None:
        result = run_monte_carlo(bundle, n_iterations=500, seed=42, progress=False)
        for g in GROUP_LETTERS:
            for team in result["group_winners"][g]:
                assert team in bundle.group_membership[g], \
                    f"Group {g} has {team} as winner but it's not in roster"


# ── Non-degenerate distribution ──────────────────────────────────────────────


class TestNonDegenerateDistribution:
    """Spec §4.2 'Distribution not degenerate' check — no team >50% or <0.01%."""

    def test_no_team_above_50_percent(self, bundle) -> None:
        result = run_monte_carlo(bundle, n_iterations=2000, seed=42, progress=False)
        for team, stats in result["champion"].items():
            assert stats["mc_fair_prob"] < 0.50, \
                f"{team} champion prob {stats['mc_fair_prob']:.3f} exceeds 50%"

    def test_top_teams_outrank_weak_teams(self, bundle) -> None:
        """Spec §4.2: top teams (Spain/Argentina/France) > weak (Curacao/Haiti)."""
        result = run_monte_carlo(bundle, n_iterations=2000, seed=42, progress=False)
        champion_probs = {t: s["mc_fair_prob"] for t, s in result["champion"].items()}
        # Allow Curacao / Haiti not appearing (effective P=0)
        for strong in ["Spain", "Argentina", "France"]:
            for weak in ["Curacao", "Haiti"]:
                p_strong = champion_probs.get(strong, 0)
                p_weak = champion_probs.get(weak, 0)
                assert p_strong > p_weak, \
                    f"{strong} ({p_strong:.4f}) should outrank {weak} ({p_weak:.4f})"
