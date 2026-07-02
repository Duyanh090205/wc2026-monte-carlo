"""Daily conditional rerun — production model (ELO+MV+star) vs LIVE market.

Each day of WC2026: lock real results played so far (conditioning harness),
re-sim the remainder with the STATIC production model, pull the live market,
and APPEND the model-vs-market snapshot to a time-series log. This log is the
raw material for the edge layer (characterize the model's stable bias vs market
-> flag days the divergence is anomalous). Model never re-trains; only the
bracket is conditioned on facts.

    PYTHONPATH=src python -m mc_simu.run_daily
    PYTHONPATH=src python -m mc_simu.run_daily --played-csv data/mc_simu/wc2026_played.csv --date 2026-06-20

Output (append-only, gitignored): data/mc_simu/daily_log.csv
    date, team, model_pct, pm_pct, kalshi_pct, consensus_pct, abs_pp, rel_pct,
    mle_pct, pool_pct, pm_bid, pm_ask, kalshi_bid, kalshi_ask
mle_pct = parallel mle_strength source (frozen artifact, same conditioning);
pool_pct = 50/50 log-opinion pool of the two models. abs_pp/rel_pct keep their
original production-vs-consensus meaning. pm/kalshi_bid/ask = executable spread
per platform (mids still drive consensus) — logged so a later backtest can tell
tradeable edge from edge inside the spread. Pre-tournament -> unconditioned sim.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mc_simu.single_game as sg  # noqa: E402
from mc_simu._common import banner  # noqa: E402
from mc_simu.mv_blend import blend_elo_with_mv, load_mv_table  # noqa: E402
from mc_simu.run_phase3_baselines import load_ratings_for_source  # noqa: E402
from mc_simu.single_game import ModelParams  # noqa: E402
from mc_simu.star_presence import apply_star_bonus, load_star_counts  # noqa: E402
from mc_simu.tournaments.wc2026 import (  # noqa: E402
    LATER_ROUNDS, build_sim_context, load_played_results, load_wc2026_bundle,
    run_monte_carlo,
)
from mc_simu.tune_to_market import jsd, normalize  # noqa: E402
from mc_simu.wc2026_vs_multi import load_prices_full  # noqa: E402

DATA = PROJECT_ROOT / "data" / "mc_simu"
DEFAULT_API = "https://seal-app-yatxw.ondigitalocean.app/api"
LOG_COLS = ["date", "team", "model_pct", "pm_pct", "kalshi_pct",
            "consensus_pct", "abs_pp", "rel_pct", "mle_pct", "pool_pct",
            "pm_bid", "pm_ask", "kalshi_bid", "kalshi_ask"]
# Tiered fallback for partially-migrated Supabase tables: try full schema, then
# the mle/pool era (10 cols), then the original launch schema (8 cols). Keeps
# the most data flowing whichever ALTER TABLE migrations have been applied.
COLS_V2 = LOG_COLS[:10]
LEGACY_LOG_COLS = LOG_COLS[:8]
STRENGTH_ARTIFACT = DATA / "strength_mle_2026-06-11.json"


def push_supabase(rows: list[dict]) -> bool:
    """Upsert the day's rows into Supabase table `daily_log` (on date,team).

    Needs SUPABASE_URL + SUPABASE_SERVICE_KEY in the environment; silently
    skipped when absent (local runs keep the CSV log only).
    """
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return False
    import requests

    def _post(cols):
        payload = [{c: (None if r[c] == "" else r[c]) for c in cols} for r in rows]
        return requests.post(
            f"{url.rstrip('/')}/rest/v1/daily_log?on_conflict=date,team",
            headers={"apikey": key, "Authorization": f"Bearer {key}",
                     "Content-Type": "application/json",
                     "Prefer": "resolution=merge-duplicates,return=minimal"},
            json=payload, timeout=30,
        )

    for cols in (LOG_COLS, COLS_V2, LEGACY_LOG_COLS):
        resp = _post(cols)
        if not (resp.status_code == 400 and "column" in resp.text.lower()):
            break
        if cols is not LEGACY_LOG_COLS:
            print(f"Supabase missing columns beyond {len(cols)} (run "
                  "deploy/supabase_schema.sql migration) — retrying narrower schema")
    resp.raise_for_status()
    return True


def _teams48():
    return [t for g in json.load(open(DATA / "wc2026_groups.json"))["groups"].values() for t in g]


def production_ratings() -> dict[str, float]:
    """ELO(eloratings) + MV blend alpha=0.5 + star X=15 — the locked production config."""
    teams48 = _teams48()
    ratings, _ = load_ratings_for_source("eloratings", pd.Timestamp.now(), DATA, teams48)
    elo_sub = {t: ratings[t] for t in teams48 if t in ratings}
    blended, _ = blend_elo_with_mv(elo_sub, load_mv_table(), alpha=0.5)
    ratings.update(blended)
    return apply_star_bonus(ratings, bonus_X=15.0, star_counts=load_star_counts())


def run_production_sim(played, n: int, seed: int) -> dict:
    """Full production MC result (champion + group_winners) on the conditioned
    bracket. Shared so the daily group-winner log reuses the champion sim instead
    of paying for a second 1M run."""
    sg.ELO_GOALS_DENOMINATOR = 1400.0
    bundle = load_wc2026_bundle(production_ratings(), params=ModelParams(diagonal_inflation=0.20))
    return run_monte_carlo(bundle, n_iterations=n, seed=seed, progress=False, played=played)


def model_champion_probs(played, n: int, seed: int) -> dict[str, float]:
    res = run_production_sim(played, n, seed)
    return normalize({t: s["mc_fair_prob"] for t, s in res["champion"].items()})


def mle_champion_probs(played, n: int, seed: int) -> dict[str, float]:
    """Parallel mle_strength run — same bundle/conditioning, frozen artifact.

    Bundle is built from production ratings so bracket mechanics (R32 seeding)
    are identical across models; the predictor ignores those ratings entirely.
    """
    import hashlib

    from mc_simu.strength_mle import (
        load_strengths_artifact, make_mle_predictor, resolve_team_name,
    )
    art = load_strengths_artifact(STRENGTH_ARTIFACT)
    missing = [t for t in _teams48() if resolve_team_name(t) not in art["strengths"]]
    if missing:
        raise KeyError(f"strength artifact missing WC2026 teams: {missing}")
    current = hashlib.sha256((DATA / "results.csv").read_bytes()).hexdigest()
    if art.get("data_sha256") != current:
        print("WARNING: strength artifact data_sha256 != current results.csv "
              "(artifact stays frozen; refresh is intentional only pre-kickoff)")
    sg.ELO_GOALS_DENOMINATOR = 1400.0
    bundle = load_wc2026_bundle(production_ratings(), params=ModelParams(diagonal_inflation=0.20))
    res = run_monte_carlo(bundle, n_iterations=n, seed=seed, progress=False,
                          played=played, predictor=make_mle_predictor(art))
    return normalize({t: s["mc_fair_prob"] for t, s in res["champion"].items()})


KO_STAGE = {**{m: "r32" for m in range(73, 89)}, **{m: "r16" for m in range(89, 97)},
            **{m: "qf" for m in range(97, 101)}, 101: "sf", 102: "sf", 104: "final"}
PRED_COLS = ["stage", "group", "match_id", "home_team", "away_team",
             "p_home", "p_draw", "p_away",
             "p_home_mle", "p_draw_mle", "p_away_mle",
             "score_pred", "score_prob", "score_pred_mle",
             "home_goals", "away_goals", "duration", "pen_home", "pen_away",
             "winner"]


def _modal_from_grid(grid: np.ndarray) -> tuple[str, float]:
    h, a = np.unravel_index(int(grid.argmax()), grid.shape)
    return f"{h}-{a}", round(float(grid[h, a]), 4)


def _wdl_from_cdf(cdfs, i) -> tuple[float, float, float]:
    grid = np.diff(np.concatenate([[0.0], cdfs[i]])).reshape(9, 9)
    return (round(float(np.tril(grid, -1).sum()), 4),
            round(float(np.trace(grid)), 4),
            round(float(np.triu(grid, 1).sum()), 4))


def _modal_score(cdfs, i) -> tuple[str, float]:
    return _modal_from_grid(np.diff(np.concatenate([[0.0], cdfs[i]])).reshape(9, 9))


def export_match_predictions(played, out_csv: Path) -> int:
    """Per-match view + results for the dashboard bracket tab, both sources.

    Group W/D/L probs are static (locked models, known fixtures). KO rows appear
    as the bracket resolves from played results; p_home = advance probability,
    score_pred = the modal 90-minute score (extra time / penalties excluded).
    p_*_mle columns carry the parallel mle_strength source over the same bundle.
    """
    from mc_simu.simulator import _build_ko_match_context, make_elo_predictor

    sg.ELO_GOALS_DENOMINATOR = 1400.0
    bundle = load_wc2026_bundle(production_ratings(), params=ModelParams(diagonal_inflation=0.20))
    predictor = make_elo_predictor(bundle.params)
    sim = build_sim_context(bundle, predictor=predictor)
    try:
        from mc_simu.strength_mle import load_strengths_artifact, make_mle_predictor
        mle_predictor = make_mle_predictor(load_strengths_artifact(STRENGTH_ARTIFACT))
        sim_mle = build_sim_context(bundle, predictor=mle_predictor)
    except Exception as e:
        print(f"export: mle_strength source unavailable (production columns "
              f"unaffected, mle columns blank): {e}")
        sim_mle = mle_predictor = None

    def ko_modal(pred_fn, a: str, b: str) -> tuple[str, float]:
        ctx = _build_ko_match_context(a, b, bundle.host)
        return _modal_from_grid(pred_fn(bundle.ratings[a], bundle.ratings[b], ctx).goal_grid)

    rows = []
    for g in sorted(sim.group_team_pairs):
        cdfs = sim.group_cdfs_per_group[g]
        for i, (h, a) in enumerate(sim.group_team_pairs[g]):
            ph, pd_, pa = _wdl_from_cdf(cdfs, i)
            score, score_p = _modal_score(cdfs, i)
            if sim_mle is not None:
                ph_m, pd_m, pa_m = _wdl_from_cdf(sim_mle.group_cdfs_per_group[g], i)
                score_m, _ = _modal_score(sim_mle.group_cdfs_per_group[g], i)
            else:
                ph_m = pd_m = pa_m = score_m = ""
            real = played.group_scores.get(frozenset((h, a)))
            rows.append({
                "stage": "group", "group": g, "match_id": "",
                "home_team": h, "away_team": a,
                "p_home": ph, "p_draw": pd_, "p_away": pa,
                "p_home_mle": ph_m, "p_draw_mle": pd_m, "p_away_mle": pa_m,
                "score_pred": score, "score_prob": score_p, "score_pred_mle": score_m,
                "home_goals": real[h] if real else "",
                "away_goals": real[a] if real else "",
                "duration": "", "pen_home": "", "pen_away": "",
                "winner": "",
            })

    from mc_simu.fetch_played import resolve_r32_pairings
    known = resolve_r32_pairings(played.group_scores, bundle.group_membership,
                                 bundle.r32_table, bundle.ratings) or {}
    for mid, left, right in LATER_ROUNDS:
        if left in played.ko_winners and right in played.ko_winners:
            known[mid] = frozenset((played.ko_winners[left], played.ko_winners[right]))
    for mid in sorted(known):
        a, b = sorted(known[mid])
        adv = sim.ko_advance[(a, b)]
        adv_m = sim_mle.ko_advance[(a, b)] if sim_mle is not None else None
        score, score_p = ko_modal(predictor, a, b)
        score_m = ko_modal(mle_predictor, a, b)[0] if sim_mle is not None else ""
        ks = played.ko_scores.get(mid)
        pens = ks["pens"] if ks else None
        rows.append({
            "stage": KO_STAGE[mid], "group": "", "match_id": mid,
            "home_team": a, "away_team": b,
            "p_home": round(adv, 4), "p_draw": "",
            "p_away": round(1 - adv, 4),
            "p_home_mle": round(adv_m, 4) if adv_m is not None else "",
            "p_draw_mle": "",
            "p_away_mle": round(1 - adv_m, 4) if adv_m is not None else "",
            "score_pred": score, "score_prob": score_p, "score_pred_mle": score_m,
            "home_goals": ks["goals"].get(a, "") if ks else "",
            "away_goals": ks["goals"].get(b, "") if ks else "",
            "duration": ks["duration"] if ks else "",
            "pen_home": pens.get(a, "") if pens else "",
            "pen_away": pens.get(b, "") if pens else "",
            "winner": played.ko_winners.get(mid, ""),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PRED_COLS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def live_market(api: str) -> dict[str, dict[str, float]]:
    """{team: {pm, kalshi, consensus, pm_bid, pm_ask, kalshi_bid, kalshi_ask}}.

    Mids drive consensus (unchanged); bid/ask are logged so a later backtest can
    tell tradeable edge from edge that sits inside the spread.
    """
    prices = load_prices_full(api)
    out = {}
    for t, row in prices.items():
        pm = row.get("Polymarket", {}).get("mid") if "Polymarket" in row else None
        kal = row.get("Kalshi", {}).get("mid") if "Kalshi" in row else None
        vals = [v for v in (pm, kal) if v is not None]
        if not vals:
            continue
        out[t] = {
            "pm": pm, "kalshi": kal, "consensus": float(np.median(vals)),
            "pm_bid": row.get("Polymarket", {}).get("bid"),
            "pm_ask": row.get("Polymarket", {}).get("ask"),
            "kalshi_bid": row.get("Kalshi", {}).get("bid"),
            "kalshi_ask": row.get("Kalshi", {}).get("ask"),
        }
    return out


def fetch_market_with_retry(api: str, attempts: int = 3,
                            wait_s: float = 60) -> dict[str, dict[str, float]] | None:
    """None after exhausting retries. An empty parse also counts as failure so a
    silent /prices schema change fails loudly instead of logging a day of blanks."""
    import time
    for i in range(1, attempts + 1):
        try:
            market = live_market(api)
            if market:
                return market
            print(f"market fetch {i}/{attempts}: response parsed to 0 teams")
        except Exception as e:
            print(f"market fetch {i}/{attempts} failed: {e}")
        if i < attempts:
            time.sleep(wait_s)
    return None


def snapshot_rows(date: str, common: list[str], mdl: dict, mkt_cons: dict,
                  market: dict, mle_n: dict | None, pool: dict | None) -> list[dict]:
    rows = []
    for t in sorted(common, key=lambda x: -mdl[x]):
        m, c = mdl[t], mkt_cons[t]
        rows.append({
            "date": date, "team": t,
            "model_pct": round(m * 100, 3),
            "pm_pct": round(market[t]["pm"] * 100, 3) if market[t]["pm"] else "",
            "kalshi_pct": round(market[t]["kalshi"] * 100, 3) if market[t]["kalshi"] else "",
            "consensus_pct": round(c * 100, 3),
            "abs_pp": round((m - c) * 100, 3),
            "rel_pct": round((c - m) / m * 100, 1) if m > 0 else "",
            "mle_pct": round(mle_n[t] * 100, 3) if mle_n is not None else "",
            "pool_pct": round(pool[t] * 100, 3) if pool is not None else "",
            "pm_bid": round(market[t]["pm_bid"] * 100, 3) if market[t].get("pm_bid") else "",
            "pm_ask": round(market[t]["pm_ask"] * 100, 3) if market[t].get("pm_ask") else "",
            "kalshi_bid": round(market[t]["kalshi_bid"] * 100, 3) if market[t].get("kalshi_bid") else "",
            "kalshi_ask": round(market[t]["kalshi_ask"] * 100, 3) if market[t].get("kalshi_ask") else "",
        })
    return rows


GW_GROUP_LETTERS = "ABCDEFGHIJKL"
POLY_GAMMA = "https://gamma-api.polymarket.com"


def fetch_poly_group_winner_now(timeout: int = 30) -> dict[str, dict[str, float]]:
    """{group: {team: last price}} current Polymarket group-winner markets.

    Polymarket is the only venue with group-winner markets (Kalshi has none).
    Degrades per-group on any failure — a missing group just drops out."""
    import requests

    from mc_simu.tune_to_market import _norm
    out: dict[str, dict[str, float]] = {}
    for letter in GW_GROUP_LETTERS:
        try:
            r = requests.get(
                f"{POLY_GAMMA}/events/slug/world-cup-group-{letter.lower()}-winner",
                timeout=timeout)
            if r.status_code != 200:
                continue
            for m in r.json().get("markets", []):
                team = _norm(m.get("groupItemTitle") or "")
                px = m.get("lastTradePrice")
                if not team or team == "Other" or px is None or not (0 < float(px) <= 1):
                    continue
                out.setdefault(letter, {})[team] = float(px)
        except Exception as e:
            print(f"group-winner Poly {letter} fetch failed: {e}")
    return out


def push_group_winner(rows: list[dict]) -> bool:
    """Upsert group-winner rows into Supabase `group_winner_log` (date,group,team)."""
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return False
    import requests
    resp = requests.post(
        f"{url.rstrip('/')}/rest/v1/group_winner_log?on_conflict=date,group,team",
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "Prefer": "resolution=merge-duplicates,return=minimal"},
        json=rows, timeout=30)
    resp.raise_for_status()
    return True


def log_group_winner(res: dict, date: str, played) -> int:
    """One day of group-winner (model vs Polymarket) -> Supabase. Reuses the
    champion sim's result; devigs Polymarket within each group. Returns rows pushed."""
    groups = json.load(open(DATA / "wc2026_groups.json"))["groups"]
    model = {g: {t: s["mc_fair_prob"] for t, s in teams.items()}
             for g, teams in res["group_winners"].items()}
    poly = fetch_poly_group_winner_now()
    matches = len(played.group_scores)
    rows = []
    for g, teams in groups.items():
        mdl = model.get(g, {})
        pm_raw = {t: v for t, v in poly.get(g, {}).items() if t in teams}
        # winner's market resolved -> survivors don't sum near 1 -> devig is bogus
        if sum(pm_raw.values()) < 0.5:
            pm_raw = {}
        s = sum(pm_raw.values())
        pm_dv = {t: v / s for t, v in pm_raw.items()} if s > 0 else {}
        for t in teams:
            if t not in mdl and t not in pm_raw:
                continue
            rows.append({
                "date": date, "group": g, "team": t,
                "model_pct": round(mdl.get(t, 0.0) * 100, 3) if mdl else None,
                "model_state_matches": matches,
                "pm_raw_pct": round(pm_raw[t] * 100, 3) if t in pm_raw else None,
                "pm_devig_pct": round(pm_dv[t] * 100, 3) if t in pm_dv else None,
            })
    return len(rows) if rows and push_group_winner(rows) else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default=_dt.date.today().isoformat())
    p.add_argument("--played-csv", type=Path, default=DATA / "wc2026_played.csv")
    p.add_argument("--log-csv", type=Path, default=DATA / "daily_log.csv")
    p.add_argument("--api-url", default=DEFAULT_API)
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    played = load_played_results(args.played_csv)
    n_locked = len(played.group_scores) + len(played.ko_winners)
    banner(f"Daily rerun {args.date} — conditioned on {n_locked} played matches "
           f"({'pre-tournament' if n_locked == 0 else args.played_csv.name})")

    # Market first: fail fast (and skip two 1M sims) when FairLine is down.
    market = fetch_market_with_retry(args.api_url)
    if market is None:
        banner(f"MARKET UNAVAILABLE {args.date} — daily_log snapshot skipped")
        print("FairLine /prices unreachable or empty after retries. Model is static "
              "and seeded, so a later workflow_dispatch rerun backfills this day. "
              "Match predictions still refresh below.")
        try:
            n_pred = export_match_predictions(played, DATA / "wc2026_match_predictions.csv")
            print(f"Wrote {n_pred} match predictions to wc2026_match_predictions.csv")
        except Exception as e:
            print(f"Match predictions export FAILED: {e}")
        return 0

    res = run_production_sim(played, args.n, args.seed)
    model = normalize({t: s["mc_fair_prob"] for t, s in res["champion"].items()})
    try:
        mle = mle_champion_probs(played, args.n, args.seed)
    except Exception as e:
        print(f"mle_strength run FAILED (production unaffected, "
              f"mle/pool columns blank today): {e}")
        mle = None
    # log every WC2026 team that the market quotes — eliminated teams show model 0%
    # (clean time series: a knocked-out favorite reads 19%->0%, not "vanished").
    common = [t for t in _teams48() if t in market]
    mkt_cons = normalize({t: market[t]["consensus"] for t in common})
    mdl = normalize({t: model.get(t, 0.0) for t in common})
    mle_n = normalize({t: mle.get(t, 0.0) for t in common}) if mle is not None else None
    pool = (normalize({t: (mdl[t] * mle_n[t]) ** 0.5 for t in common})
            if mle_n is not None else None)

    rows = snapshot_rows(args.date, common, mdl, mkt_cons, market, mle_n, pool)

    args.log_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not args.log_csv.exists()
    with args.log_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"Appended {len(rows)} rows to {args.log_csv}")
    try:
        if push_supabase(rows):
            print(f"Upserted {len(rows)} rows to Supabase daily_log")
    except Exception as e:
        print(f"Supabase push FAILED (CSV log intact): {e}")

    try:
        n_pred = export_match_predictions(played, DATA / "wc2026_match_predictions.csv")
        print(f"Wrote {n_pred} match predictions to wc2026_match_predictions.csv")
    except Exception as e:
        print(f"Match predictions export FAILED (daily log intact): {e}")

    # Group-winner (model vs Polymarket) — only meaningful while the group stage
    # is live; once all 72 group matches are in, the winners are decided (0/1) and
    # Polymarket's markets resolve. Reuses res, so no extra sim.
    if len(played.group_scores) < 72:
        try:
            n_gw = log_group_winner(res, args.date, played)
            if n_gw:
                print(f"Upserted {n_gw} rows to Supabase group_winner_log")
            else:
                print("Group-winner: no rows pushed (no Supabase creds or empty market)")
        except Exception as e:
            print(f"Group-winner log FAILED (daily log intact): {e}")
    else:
        print("Group stage complete (72/72) — group-winner logging stopped")

    # daily summary
    j = jsd(mdl, mkt_cons, common)
    l1 = sum(abs(mdl[t] - mkt_cons[t]) for t in common) * 100
    if mle_n is not None:
        mle_s = (f"mle={jsd(mle_n, mkt_cons, common):.4f} "
                 f"prod-vs-mle={jsd(mdl, mle_n, common):.4f}")
    else:
        mle_s = "mle=FAILED"
    banner(f"Snapshot {args.date} — JSD prod={j:.4f} {mle_s}  L1(prod)={l1:.1f}pp")
    print(f"{'Team':<16}{'model%':>8}{'mkt%':>8}{'abs pp':>9}{'rel %':>8}")
    by_abs = sorted(rows, key=lambda r: -abs(r["abs_pp"]))
    for r in by_abs[:8]:
        rel = r["rel_pct"]
        rel_s = f"{rel:>+7.0f}%" if isinstance(rel, (int, float)) else f"{'—':>8}"
        print(f"{r['team']:<16}{r['model_pct']:>7.2f}%{r['consensus_pct']:>7.2f}%"
              f"{r['abs_pp']:>+8.2f}{rel_s}")
    return 0


__all__ = ["production_ratings", "run_production_sim", "model_champion_probs",
           "mle_champion_probs", "live_market", "fetch_market_with_retry",
           "snapshot_rows", "push_supabase", "fetch_poly_group_winner_now",
           "push_group_winner", "log_group_winner", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
