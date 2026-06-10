"""Per-match Brier on historical WC matches: ELO vs +MV vs +MV+star vs +MV+UEFA-disc.

Answers Sam/user (2026-06-09): which MV variant predicts ACTUAL World Cup match
outcomes best — the star-bonus model (production) or the Pele-style UEFA-discount
model? This is the accuracy test (vs realized W/D/L), NOT closeness-to-market.

Configs (all alpha=0.5 blend, same single_game params, neutral ctx):
  elo       : eloratings only
  mv        : blend(Elo, raw squad MV)
  mv_star   : mv, then +15 Elo per "star" (approx: player as-of MV >= EUR 70m)
  mv_disc   : blend(Elo, UEFA-discounted squad MV) — UEFA-club valuations x0.70

Historical squad signals are rebuilt point-in-time from jfjelstul rosters +
dcaribou player_valuations (as-of tournament kickoff), which carry the player's
club domestic competition -> confederation -> UEFA flag. WC2010/14/18/22.

CAVEATS: star count is an MV>=70m approximation (production uses curated criteria);
MV source is dcaribou top-of-squad, not the official scrape. Directional test.

Usage: PYTHONPATH=src python -m mc_simu.compare_mv_variants_history
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mc_simu.single_game as sg  # noqa: E402
from mc_simu._common import banner  # noqa: E402
from mc_simu.build_historical_mv import CACHE, SEASON_TO_WC, _dl, _find_player, _norm  # noqa: E402
from mc_simu.mv_blend import blend_elo_with_mv  # noqa: E402
from mc_simu.single_game import ModelParams, predict_match  # noqa: E402
from mc_simu.validate_mv_history import _brier, _elo_before, _neutral_ctx  # noqa: E402

DATA = PROJECT_ROOT / "data" / "mc_simu"
STAR_MV = 70_000_000.0
STAR_X = 15.0


def edition_signals(season: int, cutoff: pd.Timestamp, squads, players, pv, uefa_ids):
    """Per-team {raw, disc, stars} for one edition, point-in-time."""
    wc_id = SEASON_TO_WC[season]
    roster = squads[squads["tournament_id"] == wc_id]
    pvc = (pv[pv["date"] <= cutoff].sort_values("date")
           .groupby("player_id").tail(1).set_index("player_id"))
    valued = set(pvc.index)
    sig: dict[str, dict] = {}
    for team, grp in roster.groupby("team_name"):
        raw = disc = 0.0
        stars = 0
        for _, r in grp.iterrows():
            pid = _find_player(players, team, r["given_name"], r["family_name"], valued)
            if pid is None or pid not in pvc.index:
                continue
            row = pvc.loc[pid]
            mv = float(row["market_value_in_eur"])
            raw += mv
            is_uefa = row["player_club_domestic_competition_id"] in uefa_ids
            disc += mv * (0.70 if is_uefa else 1.0)
            if mv >= STAR_MV:
                stars += 1
        sig[team] = {"raw": raw, "disc": disc, "stars": stars}
    return sig


def main() -> int:
    sg.ELO_GOALS_DENOMINATOR = 1400.0
    params = ModelParams(diagonal_inflation=0.20)
    elo = pd.read_csv(DATA / "elo_history_eloratings.csv")
    elo["date"] = pd.to_datetime(elo["date"])
    games = pd.read_csv(_dl("games.csv.gz"), low_memory=False)
    games["date"] = pd.to_datetime(games["date"])
    players = pd.read_csv(_dl("players.csv.gz"), low_memory=False)
    players["nn"] = players["name"].map(_norm)
    players["fam"] = players["nn"].map(lambda x: x.split()[-1] if x else "")
    pv = pd.read_csv(_dl("player_valuations.csv.gz"))
    pv["date"] = pd.to_datetime(pv["date"])
    comp = pd.read_csv(_dl("competitions.csv.gz"))
    uefa_ids = set(comp[comp["confederation"] == "europa"]["competition_id"])
    squads = pd.read_csv("https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/squads.csv")

    configs = ["elo", "mv", "mv_star", "mv_disc"]
    agg = {c: [] for c in configs}
    per_ed = []
    for season in [2009, 2013, 2017, 2021]:
        wc = f"WC{SEASON_TO_WC[season].split('-')[1]}"
        ed = games[(games["competition_id"] == "FIWC") & (games["season"] == season)]
        cutoff = ed["date"].min()
        elo_d = _elo_before(elo, cutoff)
        sig = edition_signals(season, cutoff, squads, players, pv, uefa_ids)
        field = {t: elo_d[t] for t in sig if t in elo_d}
        mv_raw = {t: sig[t]["raw"] for t in field if sig[t]["raw"] > 0}
        mv_disc = {t: sig[t]["disc"] for t in field if sig[t]["disc"] > 0}

        rate = {"elo": dict(field)}
        b_mv, _ = blend_elo_with_mv(field, mv_raw, alpha=0.5)
        rate["mv"] = b_mv
        rate["mv_star"] = {t: b_mv[t] + STAR_X * sig.get(t, {}).get("stars", 0) for t in b_mv}
        b_disc, _ = blend_elo_with_mv(field, mv_disc, alpha=0.5)
        rate["mv_disc"] = b_disc

        ed_brier = {c: [] for c in configs}
        for _, g in ed.iterrows():
            a, b = g["home_club_name"], g["away_club_name"]
            if a not in field or b not in field or pd.isna(g["home_club_goals"]):
                continue
            gh, ga = g["home_club_goals"], g["away_club_goals"]
            outcome = 0 if gh > ga else (1 if gh == ga else 2)
            ctx = _neutral_ctx(a, b)
            for c in configs:
                p = predict_match(rate[c][a], rate[c][b], ctx, params)
                ed_brier[c].append(_brier((p.p_home, p.p_draw, p.p_away), outcome))
        row = {"edition": wc, "n": len(ed_brier["elo"])}
        for c in configs:
            row[c] = float(np.mean(ed_brier[c]))
            agg[c].extend(ed_brier[c])
        per_ed.append(row)

    banner("Per-match Brier vs ACTUAL outcomes — WC2010-2022 (lower = better)")
    print(f"{'Edition':<9}{'n':>4}{'ELO':>9}{'+MV':>9}{'+MV+star':>10}{'+MV+disc':>10}")
    for r in per_ed:
        print(f"{r['edition']:<9}{r['n']:>4}{r['elo']:>9.4f}{r['mv']:>9.4f}{r['mv_star']:>10.4f}{r['mv_disc']:>10.4f}")
    print("-" * 51)
    n = len(agg["elo"])
    means = {c: float(np.mean(agg[c])) for c in configs}
    print(f"{'POOLED':<9}{n:>4}{means['elo']:>9.4f}{means['mv']:>9.4f}{means['mv_star']:>10.4f}{means['mv_disc']:>10.4f}")
    print(f"\n  vs ELO baseline (Δ, − = better):")
    for c in ["mv", "mv_star", "mv_disc"]:
        d = np.array(agg[c]) - np.array(agg["elo"])
        z = d.mean() / (d.std(ddof=1) / np.sqrt(n))
        print(f"    {c:<9} ΔBrier {d.mean():+.5f}  z={z:+.2f}")
    best = min(means, key=means.get)
    print(f"\n  Best (lowest pooled Brier): {best} ({means[best]:.4f})")
    return 0


__all__ = ["edition_signals", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
