"""Experiment (2026-06-09): replace star bonus with a Pele-style 30% UEFA-club
MV discount, and see how the model handles favorites vs market.

Motivation (Nate Silver's "Pele" model): Transfermarkt values are inflated for
players at UEFA clubs (richest transfer market). Pele shaves ~30% off UEFA-club
valuations so MV reflects talent, not which league a player happens to be in.
This lifts squads with domestic / non-European contingents (Saudi, Mexico,
Uruguay, ...) relative to the all-European top.

Build squad MV per WC2026 team from dcaribou players (top-23 by current MV,
algorithmic roster a la Pele); discount UEFA-club players' MV by 30%. Then run
the model with alpha=0.5 and NO star bonus, vs the current model (alpha=0.5 +
star X=15, official team-aggregate MV).

CAVEAT: the MV source changes too (dcaribou top-23-by-citizenship vs the official
participant-page aggregate), so this is a directional experiment, not a clean
one-variable A/B. Discount effect is isolated by also running the undiscounted
dcaribou MV.

Outputs:
  mc_simu/figures/wc2026_uefa_disc_abs_vs_rel.png   (abs + rel edge, 2-panel)
  console comparison table (current vs new, key teams)
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mc_simu.single_game as sg  # noqa: E402
from mc_simu._common import banner  # noqa: E402
from mc_simu.mv_blend import blend_elo_with_mv  # noqa: E402
from mc_simu.run_phase3_baselines import load_ratings_for_source  # noqa: E402
from mc_simu.single_game import ModelParams  # noqa: E402
from mc_simu.tournaments.wc2026 import load_wc2026_bundle, run_monte_carlo  # noqa: E402
from mc_simu.tune_to_market import normalize  # noqa: E402
from mc_simu.wc2026_vs_multi import load_prices_endpoint  # noqa: E402

DATA = PROJECT_ROOT / "data" / "mc_simu"
TMD = DATA / "cache" / "tmdata"
DEFAULT_API = "https://seal-app-yatxw.ondigitalocean.app/api"
DISCOUNT = 0.70  # keep 70% of a UEFA-club player's MV


def _norm(s: object) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = "".join(c if c.isalnum() or c == " " else " " for c in s)
    return " ".join(s.split())


CITIZENSHIP_ALIAS = {
    "czechia": "czech republic", "bosnia and herzegovina": "bosnia herzegovina",
    "ivory coast": "cote divoire", "south korea": "korea south",
    "united states": "united states", "dr congo": "dr congo",
}


def build_uefa_mv() -> dict[str, dict]:
    teams48 = [t for g in json.load(open(DATA / "wc2026_groups.json"))["groups"].values() for t in g]
    pl = pd.read_csv(TMD / "players.csv.gz", low_memory=False).dropna(subset=["market_value_in_eur"])
    comp = pd.read_csv(TMD / "competitions.csv.gz")
    uefa_ids = set(comp[comp["confederation"] == "europa"]["competition_id"])
    pl["cit"] = pl["country_of_citizenship"].map(_norm)
    pl["is_uefa"] = pl["current_club_domestic_competition_id"].isin(uefa_ids)

    out: dict[str, dict] = {}
    for t in teams48:
        key = CITIZENSHIP_ALIAS.get(_norm(t), _norm(t))
        cand = pl[pl["cit"] == key].sort_values("market_value_in_eur", ascending=False).head(23)
        if cand.empty:
            continue
        mv = cand["market_value_in_eur"].to_numpy()
        uefa = cand["is_uefa"].to_numpy()
        out[t] = {
            "raw": float(mv.sum()),
            "disc": float((mv * np.where(uefa, DISCOUNT, 1.0)).sum()),
            "n": int(len(cand)),
            "uefa_share": float(mv[uefa].sum() / mv.sum()) if mv.sum() else 0.0,
        }
    return out


def champion_probs(mv_dict: dict[str, float], n: int, seed: int) -> dict[str, float]:
    """alpha=0.5 blend, NO star bonus. mv_dict: team -> MV (eur)."""
    teams48 = [t for g in json.load(open(DATA / "wc2026_groups.json"))["groups"].values() for t in g]
    ratings, _ = load_ratings_for_source("eloratings", pd.Timestamp("2026-06-09"), DATA, teams48)
    elo_subset = {t: ratings[t] for t in teams48 if t in ratings}
    mv_pos = {t: v for t, v in mv_dict.items() if v and v > 0}
    blended, _ = blend_elo_with_mv(elo_subset, mv_pos, alpha=0.5)
    ratings.update(blended)
    sg.ELO_GOALS_DENOMINATOR = 1400.0
    bundle = load_wc2026_bundle(ratings, params=ModelParams(diagonal_inflation=0.20))
    res = run_monte_carlo(bundle, n_iterations=n, seed=seed, progress=False)
    return {t: s["mc_fair_prob"] for t, s in res["champion"].items()}


def _colour(v):
    return "#2a8a2a" if v > 0 else "#cc3b2f"


def figure(model, mkt, common, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = []
    for t in common:
        m, k = model[t], mkt[t]
        data.append((t, (m - k) * 100, (m / k - 1) * 100 if k > 0 else 0.0))
    abs_s = sorted(data, key=lambda r: r[1], reverse=True)
    rel_s = sorted(data, key=lambda r: r[2], reverse=True)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(15, 13))
    fig.suptitle("WC2026 — model with UEFA-30%-discount MV + NO star bonus  vs LIVE Polymarket\n"
                 "ABSOLUTE (pp) vs RELATIVE (%)   (+ = model above market)",
                 fontsize=12, fontweight="bold")
    ys = list(range(len(data)))
    axa.barh(ys, [r[1] for r in abs_s], color=[_colour(r[1]) for r in abs_s], edgecolor="black", linewidth=0.4)
    axa.set_yticks(ys); axa.set_yticklabels([r[0] for r in abs_s], fontsize=7)
    axa.invert_yaxis(); axa.axvline(0, color="black", lw=0.8)
    av = [r[1] for r in abs_s]; axa.set_xlim(min(av) - 1.4, max(av) + 1.4)
    for i, r in enumerate(abs_s):
        axa.annotate(f"{r[1]:+.2f}", (r[1], i), fontsize=6, va="center",
                     ha="left" if r[1] >= 0 else "right", xytext=(2 if r[1] >= 0 else -2, 0),
                     textcoords="offset points")
    axa.set_xlabel("model − Polymarket  (pp)"); axa.set_title("ABSOLUTE (pp) — sorted by pp")

    CAP_HI, CAP_LO = 250.0, -110.0
    draw = [max(CAP_LO, min(CAP_HI, r[2])) for r in rel_s]
    axb.barh(ys, draw, color=[_colour(r[2]) for r in rel_s], edgecolor="black", linewidth=0.4)
    axb.set_yticks(ys); axb.set_yticklabels([r[0] for r in rel_s], fontsize=7)
    axb.invert_yaxis(); axb.axvline(0, color="black", lw=0.8); axb.set_xlim(CAP_LO - 15, CAP_HI + 55)
    for i, r in enumerate(rel_s):
        d = draw[i]
        axb.annotate(f"{r[2]:+.0f}%", (d, i), fontsize=6, va="center",
                     ha="left" if d >= 0 else "right", xytext=(2 if d >= 0 else -2, 0),
                     textcoords="offset points")
    axb.set_xlabel("(model / Polymarket − 1)  (%)   [clipped]"); axb.set_title("RELATIVE (%) — sorted by %")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Wrote figure: {out}")


def main() -> int:
    banner("Build UEFA-discounted squad MV (dcaribou top-23)")
    mv = build_uefa_mv()
    print(f"  teams with MV: {len(mv)}/48 | thin (<15 players): "
          f"{[t for t, d in mv.items() if d['n'] < 15]}")

    banner("Run model — NEW (UEFA-disc MV, X=0) + ref (undiscounted, X=0)")
    p_new = champion_probs({t: d["disc"] for t, d in mv.items()}, n=20000, seed=42)
    p_undisc = champion_probs({t: d["raw"] for t, d in mv.items()}, n=20000, seed=42)

    banner("Fetch LIVE Polymarket")
    prices = load_prices_endpoint(DEFAULT_API)
    mkt = normalize({t: r["Polymarket"] for t, r in prices.items() if "Polymarket" in r})
    new = normalize(p_new)
    common = [t for t in new if t in mkt]
    figure(new, mkt, common, PROJECT_ROOT / "mc_simu" / "figures" / "wc2026_uefa_disc_abs_vs_rel.png")

    # comparison vs current production model (star X=15, official MV)
    cur = pd.read_csv(DATA / "phase3_baselines" / "wc2026_final_mv_star_vs_market.csv")
    cur_p = normalize(dict(zip(cur["team"], cur["mc_prob"])))
    und = normalize(p_undisc)
    banner("Key teams — champion prob & edge vs LIVE market (current X=15  vs  NEW disc-X=0)")
    print(f"{'Team':<14}{'mkt%':>7}{'curX15%':>9}{'edge':>7} | {'undisc%':>8}{'NEW%':>7}{'edge':>7}{'rel%':>7}")
    order = sorted(common, key=lambda t: -mkt[t])
    for t in order[:16]:
        e_cur = (cur_p.get(t, 0) - mkt[t]) * 100
        e_new = (new[t] - mkt[t]) * 100
        rel = (new[t] / mkt[t] - 1) * 100 if mkt[t] else 0
        print(f"{t:<14}{mkt[t]*100:>6.2f}%{cur_p.get(t,0)*100:>8.2f}%{e_cur:>+6.2f} | "
              f"{und.get(t,0)*100:>7.2f}%{new[t]*100:>6.2f}%{e_new:>+6.2f}{rel:>+6.0f}%")
    return 0


__all__ = ["build_uefa_mv", "champion_probs", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
