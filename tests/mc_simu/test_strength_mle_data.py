"""Phase 1 data-QC gates (T1.x) for the mle_strength rating source."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mc_simu.strength_mle import (
    HALF_PERIOD_DAYS,
    IMPORTANCE_WEIGHTS,
    MODELING_COLUMNS,
    PROJECT_ROOT,
    WINDOW_START,
    load_matches,
    match_weights,
    resolve_team_name,
    time_decay_weights,
)

AS_OF = pd.Timestamp("2026-06-10")


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return load_matches(as_of=AS_OF)


@pytest.fixture(scope="module")
def wc2026_teams() -> list[str]:
    path = PROJECT_ROOT / "data" / "mc_simu" / "wc2026_groups.json"
    groups = json.loads(path.read_text(encoding="utf-8"))["groups"]
    return [team for quad in groups.values() for team in quad]


class TestT1LoadAndNulls:
    def test_t11_row_count_band(self, matches: pd.DataFrame) -> None:
        assert 15_000 <= len(matches) <= 40_000

    def test_t12_no_nulls_in_modeling_columns(self, matches: pd.DataFrame) -> None:
        assert matches[MODELING_COLUMNS].notna().all().all()

    def test_t12_scores_nonnegative_int(self, matches: pd.DataFrame) -> None:
        for col in ("home_score", "away_score"):
            assert pd.api.types.is_integer_dtype(matches[col])
            assert (matches[col] >= 0).all()

    def test_t12_dates_within_window(self, matches: pd.DataFrame) -> None:
        assert matches["date"].min() >= WINDOW_START
        assert matches["date"].max() < AS_OF

    def test_t13_no_duplicate_fixtures(self, matches: pd.DataFrame) -> None:
        dups = matches.duplicated(subset=["date", "home_team", "away_team"])
        assert int(dups.sum()) == 0


class TestT14SanityBands:
    def test_mean_total_goals(self, matches: pd.DataFrame) -> None:
        mean_goals = (matches["home_score"] + matches["away_score"]).mean()
        assert 2.3 <= mean_goals <= 3.1

    def test_draw_rate(self, matches: pd.DataFrame) -> None:
        draw_rate = (matches["home_score"] == matches["away_score"]).mean()
        assert 0.18 <= draw_rate <= 0.30

    def test_home_advantage_on_nonneutral(self, matches: pd.DataFrame) -> None:
        nn = matches[~matches["neutral_bool"]]
        home_wins = (nn["home_score"] > nn["away_score"]).mean()
        away_wins = (nn["home_score"] < nn["away_score"]).mean()
        assert home_wins > away_wins


class TestT15Wc2026Coverage:
    def test_all_48_teams_present(self, matches: pd.DataFrame,
                                  wc2026_teams: list[str]) -> None:
        assert len(wc2026_teams) == 48
        seen = set(matches["home_team"]) | set(matches["away_team"])
        missing = [t for t in wc2026_teams if resolve_team_name(t) not in seen]
        assert missing == []

    def test_each_team_min_30_matches(self, matches: pd.DataFrame,
                                      wc2026_teams: list[str]) -> None:
        counts = (matches["home_team"].value_counts()
                  .add(matches["away_team"].value_counts(), fill_value=0))
        thin = {t: int(counts.get(resolve_team_name(t), 0)) for t in wc2026_teams
                if counts.get(resolve_team_name(t), 0) < 30}
        assert thin == {}


class TestT16NeutralFlag:
    def test_nonneutral_wc_finals_rows_are_host_home_games(
            self, matches: pd.DataFrame) -> None:
        wc = matches[matches["tournament_type"] == "world_cup_final"]
        nonneutral = wc[~wc["neutral_bool"]]
        host_home = (nonneutral["home_team"] == nonneutral["country"]).mean()
        assert host_home >= 0.95


class TestT18Importance:
    def test_full_coverage(self, matches: pd.DataFrame) -> None:
        assert matches["importance"].notna().all()
        assert set(matches["importance"].unique()) <= set(IMPORTANCE_WEIGHTS.values())

    @pytest.mark.parametrize("tournament,expected", [
        ("FIFA World Cup", 4.0),
        ("FIFA World Cup qualification", 2.5),
        ("UEFA Euro", 3.0),
        ("Copa América", 3.0),
        ("UEFA Nations League", 2.5),
        ("Friendly", 1.0),
    ])
    def test_spot_weights(self, matches: pd.DataFrame,
                          tournament: str, expected: float) -> None:
        rows = matches[matches["tournament"] == tournament]
        assert not rows.empty
        assert (rows["importance"] == expected).all()


class TestWeights:
    def test_half_period_gives_half(self) -> None:
        dates = pd.Series([AS_OF - pd.Timedelta(days=HALF_PERIOD_DAYS)])
        w = time_decay_weights(dates, AS_OF)
        assert w[0] == pytest.approx(0.5, abs=1e-12)

    def test_wc_match_weighs_4x_equal_dated_friendly(self) -> None:
        day = AS_OF - pd.Timedelta(days=100)
        df = pd.DataFrame({
            "date": [day, day],
            "importance": [4.0, 1.0],
        })
        w = match_weights(df, AS_OF)
        assert w[0] / w[1] == pytest.approx(4.0, abs=1e-12)

    def test_weights_positive_and_decreasing_with_age(self) -> None:
        df = pd.DataFrame({
            "date": [AS_OF - pd.Timedelta(days=d) for d in (10, 1000, 5000)],
            "importance": [1.0, 1.0, 1.0],
        })
        w = match_weights(df, AS_OF)
        assert (w > 0).all()
        assert w[0] > w[1] > w[2]
