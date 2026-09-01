"""Recompute every headline number claimed for this project, from committed data only.

    python verify_scorecard.py

No dependencies beyond the Python standard library. Reads the frozen campaign
record in data/mc_simu/ and prints the figures used in the write-up, so any
reader can check them without running a simulation.
"""
import csv
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "data", "mc_simu")
LOCK_DATE = "2026-06-10"          # model frozen here, before the first match
TOP4 = 4


def rows(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def champion_market():
    log = [r for r in rows(os.path.join(D, "closed_record", "daily_log.csv"))
           if r["date"] == LOCK_DATE]
    log.sort(key=lambda r: -num(r["model_pct"]))
    print(f"\nCHAMPION MARKET, locked {LOCK_DATE}")
    print(f"  {'team':<12}{'model':>8}{'PM':>8}{'Kalshi':>8}{'consensus':>11}")
    for r in log[:4]:
        print(f"  {r['team']:<12}{num(r['model_pct']):>8.2f}"
              f"{num(r['pm_pct'] or 0):>8.2f}{num(r['kalshi_pct'] or 0):>8.2f}"
              f"{num(r['consensus_pct'] or 0):>11.2f}")
    last = max(r["date"] for r in rows(os.path.join(D, "closed_record", "daily_log.csv")))
    champ = [r for r in rows(os.path.join(D, "closed_record", "daily_log.csv"))
             if r["date"] == last and num(r["model_pct"]) > 99.9]
    print(f"  champion: {champ[0]['team'] if champ else '?'}")


def semifinal_reach():
    rl = rows(os.path.join(D, "closed_record", "reach_log.csv"))
    last = max(r["date"] for r in rl)
    actual = {r["team"] for r in rl
              if r["date"] == last and r["round"] == "sf" and num(r["model_pct"]) > 99.9}
    pre = sorted([r for r in rl if r["date"] == LOCK_DATE and r["round"] == "sf"],
                 key=lambda r: -num(r["model_pct"]))
    print(f"\nSEMIFINAL REACH, model's top {TOP4} locked {LOCK_DATE}")
    print(f"  {'team':<12}{'model':>8}{'PM de-vig':>11}   reached")
    hit = 0
    for r in pre[:TOP4]:
        ok = r["team"] in actual
        hit += ok
        print(f"  {r['team']:<12}{num(r['model_pct']):>8.2f}"
              f"{num(r['pm_devig_pct'] or 0):>11.2f}   {'yes' if ok else 'no'}")
    print(f"  -> {hit} of {TOP4} reached the semifinals")


def match_scorecard():
    mp = rows(os.path.join(D, "wc2026_match_predictions.csv"))
    played = [r for r in mp if (r.get("home_goals") or "").strip() not in ("", "NA")]

    grp = [r for r in played if r["stage"] == "group"]
    hits = brier = 0.0
    for r in grp:
        p = [num(r["p_home"]), num(r["p_draw"]), num(r["p_away"])]
        hg, ag = int(float(r["home_goals"])), int(float(r["away_goals"]))
        actual = "h" if hg > ag else ("a" if hg < ag else "d")
        oh = {"h": [1, 0, 0], "d": [0, 1, 0], "a": [0, 0, 1]}[actual]
        brier += sum((pi - oi) ** 2 for pi, oi in zip(p, oh))
        hits += ["h", "d", "a"][p.index(max(p))] == actual
    n = len(grp)
    uniform = sum((1 / 3 - o) ** 2 for o in (1, 0, 0))
    print(f"\nGROUP STAGE  n={n}")
    print(f"  outcome hit rate     {hits / n * 100:.1f}%")
    print(f"  multiclass Brier     {brier / n:.3f}   (uniform 1/3 baseline {uniform:.3f})")
    print("  note: the uniform baseline is a sanity check, not a benchmark.")

    ko = [r for r in played if r["stage"] != "group" and (r.get("winner") or "").strip()]
    kh = 0
    sq = 0.0
    for r in ko:
        ph, pa = num(r["p_home"]), num(r["p_away"])
        fav = r["home_team"] if ph >= pa else r["away_team"]
        kh += fav == r["winner"]
        pw = ph if r["winner"] == r["home_team"] else pa
        sq += (1 - pw) ** 2
    print(f"\nKNOCKOUT  n={len(ko)}")
    print(f"  ties called correctly  {kh} of {len(ko)}  ({kh / len(ko) * 100:.1f}%)")
    print(f"  two-way Brier          {sq / len(ko):.3f}   (coin-flip baseline 0.250)")
    by = {}
    for r in ko:
        ph, pa = num(r["p_home"]), num(r["p_away"])
        fav = r["home_team"] if ph >= pa else r["away_team"]
        s = by.setdefault(r["stage"], [0, 0])
        s[0] += fav == r["winner"]
        s[1] += 1
    print("  by round: " + "  ".join(f"{k} {v[0]}/{v[1]}"
                                     for k, v in by.items()))


if __name__ == "__main__":
    print("Reproducing the published scorecard from the frozen campaign record.")
    champion_market()
    semifinal_reach()
    match_scorecard()
    print("\nAll figures above come from data/mc_simu/ as committed. "
          "Nothing here re-runs the simulation.")
