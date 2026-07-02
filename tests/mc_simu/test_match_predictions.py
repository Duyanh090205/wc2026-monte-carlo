"""Tests for run_daily.export_match_predictions — bracket tab data feed."""

from __future__ import annotations

import pandas as pd
import pytest

from mc_simu.run_daily import DATA, export_match_predictions
from mc_simu.tournaments.wc2026 import PlayedResults, load_played_results


def test_unconditioned_export_has_72_group_rows(tmp_path) -> None:
    out = tmp_path / "pred.csv"
    n = export_match_predictions(PlayedResults(), out)
    df = pd.read_csv(out)
    assert n == len(df) == 72
    assert (df["stage"] == "group").all()
    probs = df["p_home"] + df["p_draw"] + df["p_away"]
    assert ((probs - 1).abs() < 0.01).all()
    assert len(set(df["home_team"]) | set(df["away_team"])) == 48
    assert df["score_pred"].str.fullmatch(r"\d-\d").all()
    assert ((df["score_prob"] > 0) & (df["score_prob"] < 0.5)).all()


def test_played_result_merged_into_row(tmp_path) -> None:
    pr = PlayedResults()
    pr.group_scores[frozenset(("Mexico", "South Africa"))] = {"Mexico": 2, "South Africa": 1}
    out = tmp_path / "pred.csv"
    export_match_predictions(pr, out)
    df = pd.read_csv(out)
    mask = ((df["home_team"] == "Mexico") & (df["away_team"] == "South Africa")) | \
           ((df["home_team"] == "South Africa") & (df["away_team"] == "Mexico"))
    row = df[mask].iloc[0]
    goals = {row["home_team"]: int(row["home_goals"]), row["away_team"]: int(row["away_goals"])}
    assert goals == {"Mexico": 2, "South Africa": 1}
    others = df[~mask]
    assert others["home_goals"].isna().all()


def test_ko_rows_carry_modal_score(tmp_path) -> None:
    played_csv = DATA / "wc2026_played.csv"
    if not played_csv.exists():
        pytest.skip("no played results file")
    played = load_played_results(played_csv)
    out = tmp_path / "pred.csv"
    export_match_predictions(played, out)
    df = pd.read_csv(out)
    ko = df[df["stage"] != "group"]
    if ko.empty:
        pytest.skip("no KO pairings resolved yet")
    assert ko["score_pred"].str.fullmatch(r"\d-\d").all()
    assert ((ko["score_prob"] > 0) & (ko["score_prob"] < 0.5)).all()
    assert ((ko["p_home"] + ko["p_away"] - 1).abs() < 0.01).all()
    decided = ko[ko["winner"].notna()]
    assert decided["home_goals"].notna().all()
    assert decided["away_goals"].notna().all()


def test_load_played_results_ko_scores(tmp_path) -> None:
    csv = tmp_path / "played.csv"
    csv.write_text(
        "stage,match_id,home_team,away_team,home_goals,away_goals,"
        "duration,pen_home,pen_away,winner\n"
        "ko,73,South Africa,Canada,0,1,REGULAR,,,Canada\n"
        "ko,74,Germany,Paraguay,1,1,PENALTY_SHOOTOUT,4,5,Paraguay\n",
        encoding="utf-8")
    pr = load_played_results(csv)
    assert pr.ko_winners == {73: "Canada", 74: "Paraguay"}
    assert pr.ko_scores[73] == {"goals": {"South Africa": 0, "Canada": 1},
                                "duration": "REGULAR", "pens": None}
    assert pr.ko_scores[74] == {"goals": {"Germany": 1, "Paraguay": 1},
                                "duration": "PENALTY_SHOOTOUT",
                                "pens": {"Germany": 4, "Paraguay": 5}}


def test_load_played_results_legacy_columns(tmp_path) -> None:
    csv = tmp_path / "played.csv"
    csv.write_text(
        "stage,match_id,home_team,away_team,home_goals,away_goals,winner\n"
        "ko,73,South Africa,Canada,0,1,Canada\n",
        encoding="utf-8")
    pr = load_played_results(csv)
    assert pr.ko_winners == {73: "Canada"}
    assert pr.ko_scores[73] == {"goals": {"South Africa": 0, "Canada": 1},
                                "duration": "", "pens": None}
