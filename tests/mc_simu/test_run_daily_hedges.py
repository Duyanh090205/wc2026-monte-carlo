"""Tests for run_daily failure hedges — market retry + mle-optional rows."""

from __future__ import annotations

import pandas as pd

import mc_simu.run_daily as rd
from mc_simu.tournaments.wc2026 import PlayedResults


class TestFetchMarketWithRetry:
    def test_exhausts_attempts_on_exception(self, monkeypatch) -> None:
        calls = []

        def boom(api):
            calls.append(api)
            raise RuntimeError("connection refused")

        monkeypatch.setattr(rd, "live_market", boom)
        assert rd.fetch_market_with_retry("http://x", attempts=3, wait_s=0) is None
        assert len(calls) == 3

    def test_empty_parse_counts_as_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(rd, "live_market", lambda api: {})
        assert rd.fetch_market_with_retry("http://x", attempts=2, wait_s=0) is None

    def test_success_returns_immediately(self, monkeypatch) -> None:
        calls = []
        payload = {"Spain": {"pm": 0.16, "kalshi": 0.17, "consensus": 0.165}}

        def ok(api):
            calls.append(api)
            return payload

        monkeypatch.setattr(rd, "live_market", ok)
        assert rd.fetch_market_with_retry("http://x", attempts=3, wait_s=0) == payload
        assert len(calls) == 1

    def test_recovers_on_second_attempt(self, monkeypatch) -> None:
        payload = {"Spain": {"pm": 0.16, "kalshi": None, "consensus": 0.16}}
        responses = iter([RuntimeError("503"), payload])

        def flaky(api):
            r = next(responses)
            if isinstance(r, Exception):
                raise r
            return r

        monkeypatch.setattr(rd, "live_market", flaky)
        assert rd.fetch_market_with_retry("http://x", attempts=3, wait_s=0) == payload


class TestSnapshotRows:
    COMMON = ["Spain", "Panama"]
    MDL = {"Spain": 0.95, "Panama": 0.05}
    MKT = {"Spain": 0.90, "Panama": 0.10}
    MARKET = {"Spain": {"pm": 0.91, "kalshi": 0.89, "consensus": 0.90,
                        "pm_bid": 0.90, "pm_ask": 0.92,
                        "kalshi_bid": 0.88, "kalshi_ask": 0.90},
              "Panama": {"pm": 0.10, "kalshi": None, "consensus": 0.10,
                         "pm_bid": 0.09, "pm_ask": 0.11,
                         "kalshi_bid": None, "kalshi_ask": None}}

    def test_mle_failure_leaves_columns_blank(self) -> None:
        rows = rd.snapshot_rows("2026-06-12", self.COMMON, self.MDL, self.MKT,
                                self.MARKET, mle_n=None, pool=None)
        assert [r["team"] for r in rows] == ["Spain", "Panama"]
        assert all(r["mle_pct"] == "" and r["pool_pct"] == "" for r in rows)
        assert rows[0]["model_pct"] == 95.0
        assert rows[0]["abs_pp"] == 5.0

    def test_full_rows_carry_mle_and_pool(self) -> None:
        mle_n = {"Spain": 0.9, "Panama": 0.1}
        pool = {"Spain": 0.92, "Panama": 0.08}
        rows = rd.snapshot_rows("2026-06-12", self.COMMON, self.MDL, self.MKT,
                                self.MARKET, mle_n=mle_n, pool=pool)
        assert rows[0]["mle_pct"] == 90.0
        assert rows[0]["pool_pct"] == 92.0
        assert rows[1]["kalshi_pct"] == ""

    def test_bid_ask_logged_and_blank_when_absent(self) -> None:
        rows = rd.snapshot_rows("2026-06-12", self.COMMON, self.MDL, self.MKT,
                                self.MARKET, mle_n=None, pool=None)
        assert rows[0]["pm_bid"] == 90.0 and rows[0]["pm_ask"] == 92.0
        assert rows[0]["kalshi_bid"] == 88.0 and rows[0]["kalshi_ask"] == 90.0
        assert rows[1]["kalshi_bid"] == "" and rows[1]["kalshi_ask"] == ""
        assert rows[1]["pm_ask"] == 11.0


class TestExportSurvivesMleFailure:
    def test_missing_artifact_blanks_only_mle_columns(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(rd, "STRENGTH_ARTIFACT", tmp_path / "missing.json")
        out = tmp_path / "pred.csv"
        n = rd.export_match_predictions(PlayedResults(), out)
        assert n == 72
        df = pd.read_csv(out)
        assert df["p_home_mle"].isna().all()
        assert df["p_home"].notna().all()
