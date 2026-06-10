"""Build point-in-time squad market value per historical World Cup edition.

Combines two free, no-auth sources (proven feasible 2026-06-09):
  - rosters:    jfjelstul/worldcup `squads.csv` (23-man squads per WC, 1930-2022)
  - valuations: dcaribou/transfermarkt-datasets `player_valuations` (dated MV)
                + `players` (name -> player_id, country) + `games` (FIWC dates)

For each edition we take each squad player's LATEST Transfermarkt market value
dated <= the tournament kickoff (true point-in-time, not current). Players are
matched jfjelstul-name -> dcaribou-player_id, scoped by country of citizenship
(match rate ~96-100% for major teams; degrades for minnows whose squad MV is
~0 anyway).

Raw dcaribou CSVs are cached under data/mc_simu/cache/tmdata/ (gitignored).
Output: data/mc_simu/historical_squad_mv.csv (edition, team, squad_mv_eur, ...).

Usage:
    PYTHONPATH=src python -m mc_simu.build_historical_mv            # all WC editions
    PYTHONPATH=src python -m mc_simu.build_historical_mv --min-season 2009
"""

from __future__ import annotations

import argparse
import os
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from mc_simu._common import banner  # noqa: E402

R2_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
JFJELSTUL_SQUADS = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/squads.csv"
CACHE = PROJECT_ROOT / "data" / "mc_simu" / "cache" / "tmdata"
OUT_CSV = PROJECT_ROOT / "data" / "mc_simu" / "historical_squad_mv.csv"

# FIWC season label (year before) -> jfjelstul tournament_id.
SEASON_TO_WC = {2005: "WC-2006", 2009: "WC-2010", 2013: "WC-2014",
                2017: "WC-2018", 2021: "WC-2022"}


def _dl(fname: str) -> Path:
    """Download a dcaribou R2 CSV to cache if missing; return local path."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / fname
    if p.exists() and p.stat().st_size > 100:
        return p
    print(f"  downloading {fname} ...", flush=True)
    with requests.get(R2_BASE + fname, stream=True, timeout=300) as r:
        r.raise_for_status()
        with p.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    return p


def _norm(s: object) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.split())


def build(min_season: int = 2009) -> pd.DataFrame:
    games = pd.read_csv(_dl("games.csv.gz"), low_memory=False)
    players = pd.read_csv(_dl("players.csv.gz"), low_memory=False)
    pv = pd.read_csv(_dl("player_valuations.csv.gz"))
    pv["date"] = pd.to_datetime(pv["date"])
    squads = pd.read_csv(JFJELSTUL_SQUADS)

    players["nn"] = players["name"].map(_norm)
    players["fam"] = players["nn"].map(lambda x: x.split()[-1] if x else "")

    fiwc = games[games["competition_id"] == "FIWC"]
    rows: list[dict] = []
    for season, wc_id in sorted(SEASON_TO_WC.items()):
        if season < min_season:
            continue
        edition = games[(games["competition_id"] == "FIWC") & (games["season"] == season)]
        if edition.empty:
            continue
        cutoff = pd.Timestamp(edition["date"].min())
        roster = squads[squads["tournament_id"] == wc_id]
        if roster.empty:
            print(f"  WARN no jfjelstul roster for {wc_id}")
            continue
        # one as-of valuation lookup per edition (latest <= cutoff)
        pvc = (pv[pv["date"] <= cutoff].sort_values("date")
               .groupby("player_id").tail(1).set_index("player_id")["market_value_in_eur"])
        valued_ids = set(pvc.index)
        wc_year = wc_id.split("-")[1]
        for team, grp in roster.groupby("team_name"):
            matched, valued, total_mv = 0, 0, 0.0
            for _, r in grp.iterrows():
                pid = _find_player(players, team, r["given_name"], r["family_name"], valued_ids)
                if pid is None:
                    continue
                matched += 1
                if pid in pvc.index:
                    valued += 1
                    total_mv += float(pvc[pid])
            rows.append({"edition": f"WC{wc_year}", "season": season, "cutoff": cutoff.date(),
                         "team": team, "squad_mv_eur": round(total_mv),
                         "squad_size": len(grp), "matched": matched, "valued": valued})
        print(f"  {wc_id}: {roster['team_name'].nunique()} teams, cutoff {cutoff.date()}")
    return pd.DataFrame(rows)


def _find_player(players: pd.DataFrame, country: str, given: str, family: str,
                 valued_ids: set) -> object | None:
    full, famn = _norm(f"{given} {family}"), _norm(family)
    pool = players[players["country_of_citizenship"] == country]
    if pool.empty:
        pool = players
    m = pool[pool["nn"] == full]
    if m.empty:
        m = pool[pool["fam"] == famn]
    if m.empty:
        return None
    # prefer a candidate that actually has a valuation (disambiguates namesakes)
    pref = m[m["player_id"].isin(valued_ids)]
    return (pref if not pref.empty else m).iloc[0]["player_id"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-season", type=int, default=2009,
                   help="lowest FIWC season to include (2009=WC2010; 2005=WC2006 sparse)")
    p.add_argument("--out", type=Path, default=OUT_CSV)
    args = p.parse_args(argv)

    banner("Build point-in-time squad MV per WC edition")
    df = build(args.min_season)
    if df.empty:
        raise SystemExit("no editions built")
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")

    for ed in sorted(df["edition"].unique()):
        sub = df[df["edition"] == ed].sort_values("squad_mv_eur", ascending=False)
        banner(f"{ed} — top 6 / bottom 3 (point-in-time squad MV)")
        show = pd.concat([sub.head(6), sub.tail(3)])
        for _, r in show.iterrows():
            print(f"  {r['team']:<16} €{r['squad_mv_eur']/1e9:5.3f}bn  "
                  f"matched {r['matched']:>2}/{r['squad_size']}  valued {r['valued']:>2}")
    return 0


__all__ = ["build", "main", "SEASON_TO_WC"]


if __name__ == "__main__":
    raise SystemExit(main())
