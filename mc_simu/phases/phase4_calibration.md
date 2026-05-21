## 5. Phase 4 — Model Comparison & MC Output (4+2 days)

**Goal:** Plug 3 models into Phase 3 barebone simulator → run historical replay on 12 giải → tune params per Elo source via LOTO-CV → write WC2026 outputs (one CSV per source).

**Files:** `src/mc_simu/calibration.py`, `src/mc_simu/historical_replay.py`, `src/mc_simu/output.py`

### 5.0 Comparison framework (2026-05-20 ADDED)

Per Sam's note 2026-05-20 + user decisions on 2026-05-20:

**Three plug-in models** (each implements `predict_match(team_a, team_b, ctx) → MatchPrediction`):

| Model | Built in | Tunable params |
|---|---|---|
| **A. self-Elo** | Phase 1 §2.1-2.2 | α, β, diagonal_inflation (LOTO-CV per source) |
| **B. eloratings-Elo** | Phase 1 §2.4 | same 3 params, tuned independently |
| **C. Baseline 40/40/20** | Phase 3 §3.8 | none (constant) |

**Two metrics, two roles:**

| Metric | Where | Role | Sample size |
|---|---|---|---|
| **Historical tournament replay** (§5.3) | Plug each model into Phase 3 simulator; run 12 historical tournaments × 100k iters | **Primary** — picks Elo source winner per Sam's framework | n=12 (1 actual champion / giải) |
| **LOTO-CV match-Brier** (§5.1) | Existing param grid sweep per Elo source | **Secondary** — tunes (α, β, diagonal) within each source | n~760 matches |

**Decision rule (revised 2026-05-20):**
1. Run LOTO-CV separately for self-Elo and eloratings-Elo → get θ_self*, θ_eloratings*
2. Plug both tuned models + baseline into historical replay
3. Pick winner via aggregate log P(actual_champion) across 12 giải
4. If self vs eloratings within 2σ → ship both CSVs; team review decides
5. If baseline beats either Elo model → HARD STOP, model is broken

**Two-CSV output (user decision 2026-05-20):**
- `data/mc_simu/mc_fair_odds_wc2026_self_<date>.csv` — 96 rows
- `data/mc_simu/mc_fair_odds_wc2026_eloratings_<date>.csv` — 96 rows
- Optionally `mc_fair_odds_wc2026_baseline_<date>.csv` for floor reference

### 5.1 Build steps

**Step 4.1 — Parameter grid** (3 dimensions, 27 combos — `blend_match_weight` fixed at 1.0 per §0.8 Phase 2 skip)

```python
PARAM_GRID = {
    'alpha_hfa':          [0.25, 0.40, 0.50],
    'beta_confederation': [0.00, 0.13, 0.20],
    'diagonal_inflation': [0.05, 0.09, 0.12],
}
# 3 × 3 × 3 = 27 combinations
# blend_match_weight fixed at 1.0 (no roster overlay, per §0.8)
```

**Step 4.2 — LOTO-CV trên 12 tournaments**

```python
LOTO_TOURNAMENTS = [
    'FIFA World Cup 2002', 'FIFA World Cup 2006', 'FIFA World Cup 2010',
    'FIFA World Cup 2014', 'FIFA World Cup 2018', 'FIFA World Cup 2022',
    'UEFA Euro 2004', 'UEFA Euro 2008', 'UEFA Euro 2012',
    'UEFA Euro 2016', 'UEFA Euro 2020', 'UEFA Euro 2024',
]

def run_loto_cv(param_grid, tournaments):
    """For each θ, hold out each tournament, fit Elo on rest, predict matches.
    Brier per tournament, average across folds.
    Returns: {θ: {tournament: brier, avg: brier_mean}}"""
    results = {}
    for params_values in itertools.product(*param_grid.values()):
        θ = ModelParams(**dict(zip(param_grid.keys(), params_values)))
        per_tournament = {}
        for held_out in tournaments:
            # Fit Elo on data BEFORE held_out tournament start (no lookahead)
            train_cutoff = start_date_of(held_out)
            elo_history = build_rating_history_until(train_cutoff)
            
            briers = []
            for m in matches_in(held_out):
                elo_h = elo_history[m.home_team].at(m.date - timedelta(days=1))
                elo_a = elo_history[m.away_team].at(m.date - timedelta(days=1))
                pred = predict_match(elo_h, elo_a, m.context, params=θ)
                
                # actual: 1-hot encoding of W/D/L from scoreboard
                if m.home_score > m.away_score:   actual = (1, 0, 0)
                elif m.home_score == m.away_score: actual = (0, 1, 0)
                else:                              actual = (0, 0, 1)
                
                brier = sum((p - a)**2 for p, a in zip(
                    [pred.p_home, pred.p_draw, pred.p_away], actual))
                briers.append(brier)
            
            per_tournament[held_out] = np.mean(briers)
        
        results[tuple(params_values)] = {
            **per_tournament,
            'avg': np.mean(list(per_tournament.values()))
        }
    return results
```

**Step 4.3 — Select θ* with 2σ check**

```python
def select_optimal_params(cv_results):
    sorted_by = sorted(cv_results.items(), key=lambda x: x[1]['avg'])
    best, second = sorted_by[0], sorted_by[1]
    
    se = np.std([best[1][t] for t in LOTO_TOURNAMENTS]) / np.sqrt(len(LOTO_TOURNAMENTS))
    diff = second[1]['avg'] - best[1]['avg']
    
    if diff < 2 * se:
        print(f"WARNING: Best vs 2nd-best diff ({diff:.4f}) < 2σ ({2*se:.4f})")
        print(f"Falling back to defaults (α=0.40, β=0.13)")
        return DEFAULT_PARAMS
    
    return dict(best[0])
```

**Step 4.4 — Isotonic calibration**

```python
def fit_isotonic_calibration(predictions, actuals, n_bins=10):
    """Check if MC outputs need post-hoc calibration. If max bias >5pp in any bin,
    fit isotonic. Otherwise return identity mapping.
    
    Args:
        predictions: array of predicted probabilities from LOTO-CV
        actuals: array of 0/1 outcomes (binned per W/D/L)
        n_bins: deciles by default
    
    Returns: callable mapping (raw_prob → calibrated_prob) or identity
    """
    from sklearn.isotonic import IsotonicRegression
    
    # Bin predictions into deciles, compute observed frequency per bin
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bins) - 1
    
    max_deviation = 0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() < 10:  # skip sparse bins
            continue
        predicted_mean = predictions[mask].mean()
        actual_freq = actuals[mask].mean()
        deviation = abs(predicted_mean - actual_freq)
        max_deviation = max(max_deviation, deviation)
    
    if max_deviation < 0.05:  # <5pp bias → no calibration needed
        return lambda p: p
    
    # Fit isotonic regression
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(predictions, actuals)
    return iso.predict
```

**Step 4.5 — Sanity check vs market (diagnostic only, no integration)**

Two independent sanity checks (per Sam's suggestion):

**4.5a — Champion market** (1 market, 48 outcomes)
- Polymarket "Will [team] win World Cup 2026?" prices
- Compare với MC champion distribution

**4.5b — Group winner markets** (12 markets, 4 outcomes each = 48 entries total)
- Polymarket "Group [A-L] winner" markets
- Compare với MC group winner distributions
- Liquidity typically higher than champion market vì settle sớm hơn → more reliable reference

Cross-checking both catches different bug classes: champion-only check có thể miss knockout sim bugs (group sim correct compensates), nhưng group winner check expose ngay.

```python
def sanity_check_vs_market(mc_results, market_results):
    """Compare MC outputs vs Polymarket for both champion + group winners.
    
    NOTE: This is for evaluation/presentation purposes only.
    MC output does NOT feed back into market — Stream 3 is standalone in v1.
    
    Args:
        mc_results: dict from run_monte_carlo() with 'champion' + 'group_winners' keys
        market_results: dict {
            'champion': {team: polymarket_price},
            'group_winners': {group_letter: {team: polymarket_price}}
        }
    
    Decision rules (apply separately to champion check + each group):
    - All teams in market lệch <2pp → SUSPECT implicit overfit. Investigate.
    - 2-10pp lệch on 3-10 teams → expected. Potential trading opportunities.
    - >10pp lệch on >5 teams → likely model bug. HARD STOP.
    
    Writes diagnostic CSVs:
    - audit_sanity_champion.csv (48 rows)
    - audit_sanity_group_winners.csv (48 rows, by group)
    """
    # Champion check
    champion_diffs = {}
    for team, mc_data in mc_results['champion'].items():
        market_price = market_results['champion'].get(team)
        if market_price is None:
            continue
        champion_diffs[team] = abs(mc_data['mc_fair_prob'] - market_price)
    _apply_decision_rules(champion_diffs, label='champion')
    
    # Group winner checks (12 separate sanity checks)
    for group_letter, group_mc in mc_results['group_winners'].items():
        group_market = market_results['group_winners'].get(group_letter, {})
        if not group_market:
            print(f"WARNING: No Polymarket data for Group {group_letter} — skipping")
            continue
        group_diffs = {}
        for team, mc_data in group_mc.items():
            market_price = group_market.get(team)
            if market_price is None:
                continue
            group_diffs[team] = abs(mc_data['mc_fair_prob'] - market_price)
        _apply_decision_rules(group_diffs, label=f'group_{group_letter}')
```

**Step 4.6 — Freeze params**

Save θ* + isotonic to `src/mc_simu/frozen_params.json`:
```json
{
  "model_version": "v1.0",
  "frozen_at": "2026-06-06T12:00:00Z",
  "alpha_hfa": 0.40,
  "beta_confederation": 0.13,
  "diagonal_inflation": 0.09,
  "blend_match_weight": 1.0,
  "phase_2_skipped": true,
  "isotonic_mapping": [...]
}
```

**Step 4.7 — ⛔ SKIPPED v1 — `mc_fair_odds` DB table** (deferred to v2)

Per §0.6 v1 CSV-only path, **no schema modification to `src/db.py` in v1**. CSV output (Step 4.8) only. Schema in §0.6 preserved as DDL for v2 reactivation.

**Step 4.8 — MC output writer (CSV-only v1)**

```python
# src/mc_simu/output.py
import csv
from pathlib import Path

def write_mc_fair_odds_csv(results, event, params, computed_at):
    """Write 96 rows to data/mc_simu/mc_fair_odds_<event>_<YYYY-MM-DD>.csv.

    Schema columns match §0.6 DDL (for future v2 DB activation compatibility):
      event, market, team, mc_fair_prob, mc_se, n_iterations, elo_used,
      roster_elo_used, hfa_alpha, hfa_beta, diagonal_inflation,
      blend_match_weight, model_version, computed_at

    v1 invariants per §0.8:
      - roster_elo_used = NULL (empty string in CSV)
      - blend_match_weight = 1.0 (fixed, no roster blend)

    Rows: 48 champion + 12×4 group winner = 96.
    """
    out_path = Path(f"data/mc_simu/mc_fair_odds_{event}_"
                    f"{computed_at.strftime('%Y-%m-%d')}.csv")
    cols = ["event","market","team","mc_fair_prob","mc_se","n_iterations",
            "elo_used","roster_elo_used","hfa_alpha","hfa_beta",
            "diagonal_inflation","blend_match_weight","model_version","computed_at"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        # Champion market (48 rows)
        for team, data in results['champion'].items():
            w.writerow({"event": event, "market": "winner", "team": team,
                        "mc_fair_prob": data['mc_fair_prob'],
                        "mc_se": data['mc_se'], "n_iterations": data['n_iterations'],
                        "elo_used": data.get('elo_used'),
                        "roster_elo_used": "",                    # §0.8: NULL in v1
                        "hfa_alpha": params['alpha_hfa'],
                        "hfa_beta": params['beta_confederation'],
                        "diagonal_inflation": params['diagonal_inflation'],
                        "blend_match_weight": 1.0,                # §0.8: fixed
                        "model_version": "v1.0",
                        "computed_at": computed_at.isoformat()})
        # Group winner markets (48 rows)
        for g, group_data in results['group_winners'].items():
            for team, data in group_data.items():
                w.writerow({"event": event, "market": f"group_winner_{g}",
                            "team": team, "mc_fair_prob": data['mc_fair_prob'],
                            "mc_se": data['mc_se'], "n_iterations": data['n_iterations'],
                            "elo_used": data.get('elo_used'),
                            "roster_elo_used": "", "hfa_alpha": params['alpha_hfa'],
                            "hfa_beta": params['beta_confederation'],
                            "diagonal_inflation": params['diagonal_inflation'],
                            "blend_match_weight": 1.0, "model_version": "v1.0",
                            "computed_at": computed_at.isoformat()})
```

**v2 future:** add `write_mc_fair_odds_db()` companion that mirrors CSV writer into DB via `_upsert()` pattern (when team approves DB integration).

**Step 4.9 — Daily refresh script (v1 file-based, NO automated cron in v1)**

Create `src/run_mc_simu.py` as manual-trigger entry point. Daily cron deferred to v2.

```python
# src/run_mc_simu.py
"""On-demand MC simulation runner.

1. Load frozen params from frozen_params.json
2. Load current Elo ratings (built from match history)
3. Load WC2026 fixtures + R32 seeding
4. Run 100k iterations
5. Write CSV to data/mc_simu/mc_fair_odds_<event>_<date>.csv
6. Log summary to logs/mc_simu.log

v1: run manually for presentation snapshot.
v2: register with DO worker / Task Scheduler when daily refresh approved."""
```

**v2 cron registration template (NOT applied in v1):**

```
Task name:    PredMktMCSimu (Windows Task Scheduler) or DO worker entry
Trigger:      Daily at 00:30 UTC
Action:       python src/run_mc_simu.py --event world_cup_2026
Restart:      On failure every 1 min, 999 retries
```

### 5.2 Validation gates

**Test 4.A — CV sanity:**
- θ* values fall within tunable ranges (not extreme corners)
- Per-tournament Brier doesn't vary >2× (else overfit to one tournament)

**Test 4.B — Market sanity (per Step 4.5):**
- 2-10pp lệch on 3-10 teams → expected
- Not all teams <2pp (overfit)
- Not >5 teams >10pp (HARD STOP)

**Test 4.C — Standalone CSV output verification (v1):**

```bash
# 1. Trigger MC sim
python src/run_mc_simu.py --event world_cup_2026

# 2. Verify CSV exists + 96 rows + per-market sums = 1.0
python -c "
import pandas as pd
df = pd.read_csv('data/mc_simu/mc_fair_odds_world_cup_2026_2026-06-XX.csv')
assert len(df) == 96, f'Expected 96 rows (48 champion + 48 group winners), got {len(df)}'

# Champion market sum = 1.0
champ_sum = df[df['market']=='winner']['mc_fair_prob'].sum()
assert abs(champ_sum - 1.0) < 0.001, f'Champion sum={champ_sum}'

# Each group winner market sum = 1.0
for g in 'ABCDEFGHIJKL':
    s = df[df['market']==f'group_winner_{g}']['mc_fair_prob'].sum()
    assert abs(s - 1.0) < 0.001, f'Group {g} sum={s}'

# v1 invariants per §0.8
assert df['blend_match_weight'].eq(1.0).all(), 'blend_match_weight must be 1.0 in v1'
assert df['roster_elo_used'].isna().all(), 'roster_elo_used must be NULL in v1'

print('Output verification: PASS (96 rows, all markets sum to 1.0, §0.8 invariants OK)')
"

# 3. Verify NO modifications to existing DB schemas (v1 doesn't write DB at all)
git diff src/db.py
# Expected: NO diff in v1 — mc_fair_odds table NOT added until v2

# 4. Verify paper_trader.py unchanged behavior — run 1 cycle, compare baseline
python src/paper_trader.py --event champions_league_2025-26 --dry-run
# Expected: identical signal output vs pre-Phase-4 baseline (Phase 0 captures snapshot)
```

### 5.3 Historical tournament replay framework (NEW 2026-05-20)

**Primary metric for Elo source selection** per §5.0 decision rule. Implements Sam's framework: plug each model into the same Phase 3 simulator, run on 12 historical tournaments, compare how often each model gave high probability to the actual champion.

**File:** `src/mc_simu/historical_replay.py`

```python
# src/mc_simu/historical_replay.py
"""Historical tournament replay framework — primary Elo-source decision metric.

Plugs different per-game probability models into the same Phase 3 simulator,
runs full tournament sim on 12 historical giải, scores each model by how
much probability mass it assigned to the actual champion.

Per Sam's 2026-05-20 framework + user decisions same day.
"""
from mc_simu.simulator import run_monte_carlo
from mc_simu.tournaments import wc_8groups, euro_24teams, euro_16teams
from mc_simu.elo_source import load_elo_history
from mc_simu.elo import get_rating_as_of
from mc_simu.single_game import predict_match, ModelParams
from mc_simu.baseline import predict_match_40_40_20
from functools import partial
import numpy as np

HISTORICAL_TOURNAMENTS = [
    # (year, format_module, actual_champion, host_meta)
    (2002, wc_8groups,   "Brazil",    {...}),
    (2006, wc_8groups,   "Italy",     {...}),
    (2010, wc_8groups,   "Spain",     {...}),
    (2014, wc_8groups,   "Germany",   {...}),
    (2018, wc_8groups,   "France",    {...}),
    (2022, wc_8groups,   "Argentina", {...}),
    (2004, euro_16teams, "Greece",    {...}),
    (2008, euro_16teams, "Spain",     {...}),
    (2012, euro_16teams, "Spain",     {...}),
    (2016, euro_24teams, "Portugal",  {...}),
    (2020, euro_24teams, "Italy",     {...}),  # played 2021
    (2024, euro_24teams, "Spain",     {...}),
]


def score_model(predictor, model_name: str, n_iters: int = 100_000) -> dict:
    """Run all 12 historical tournaments with `predictor` plugged in.
    Return {tournament: P(actual_champion), 'log_sum': sum log P, 'sum': sum P}.

    Elo snapshot for each tournament uses ratings as-of (start_date - 1 day)
    via get_rating_as_of() — same no-lookahead rule as LOTO-CV.
    """
    results = {}
    log_sum = 0.0
    plain_sum = 0.0
    for year, fmt, actual, host_meta in HISTORICAL_TOURNAMENTS:
        fixtures = fmt.make_fixtures(year, host_meta)
        ratings_snapshot = _take_snapshot(year - 0.001, model_name)  # pre-tournament
        mc = run_monte_carlo(n_iters, fixtures=fixtures, predictor=predictor,
                             ratings=ratings_snapshot,
                             tournament_module=fmt)
        p_actual = mc['champion'].get(actual, {}).get('mc_fair_prob', 1e-6)
        results[f"{fmt.__name__.split('.')[-1]}_{year}"] = {
            'actual_champion': actual,
            'p_actual': p_actual,
        }
        log_sum += np.log(max(p_actual, 1e-6))   # clamp to avoid -inf on misses
        plain_sum += p_actual
    results['log_sum'] = log_sum
    results['sum'] = plain_sum
    return results


def run_comparison(n_iters: int = 100_000) -> pd.DataFrame:
    """Score all 3 models. Returns wide-format DataFrame for audit_replay_report.csv.

    Columns: tournament, actual_champion,
             self_p, eloratings_p, baseline_p,
             self_log, eloratings_log, baseline_log
    Plus aggregate row at bottom.
    """
    # Load tuned θ* per source from Phase 4 §5.1 LOTO-CV
    theta_self = json.load(open("src/mc_simu/frozen_params_self.json"))
    theta_elor = json.load(open("src/mc_simu/frozen_params_eloratings.json"))

    pred_self = partial(predict_match, params=ModelParams(**theta_self))
    pred_elor = partial(predict_match, params=ModelParams(**theta_elor))
    pred_base = predict_match_40_40_20

    r_self = score_model(pred_self, "self")
    r_elor = score_model(pred_elor, "eloratings")
    r_base = score_model(pred_base, "baseline")

    # ...build DataFrame, write to data/mc_simu/audit_replay_report.csv
```

**Output report** `data/mc_simu/audit_replay_report.csv`:

| tournament | actual | self_p | eloratings_p | baseline_p |
|---|---|---|---|---|
| WC 2018 | France | 0.14 | 0.16 | 0.03 |
| WC 2022 | Argentina | 0.08 | 0.12 | 0.03 |
| Euro 2024 | Spain | 0.17 | 0.15 | 0.04 |
| ... | ... | ... | ... | ... |
| **log-sum** | — | **−18.2** | **−16.8** | **−40.5** |

**Decision rule:**

| Outcome | Action |
|---|---|
| Baseline beats either Elo model | ❌ **HARD STOP** — model bug, debug before proceeding |
| Self vs eloratings log-sum diff > 2σ | ✅ Pick winner. Ship single CSV. |
| Self vs eloratings log-sum diff ≤ 2σ | ⚠ Statistical tie. Ship **both** CSVs side-by-side per user decision 2026-05-20; team review decides. |

**2σ estimation:** bootstrap 1000 resamples of the 12 tournament results to estimate SE of log-sum difference. If `\|self_log - eloratings_log\| > 2 * SE_diff` → significant.

### 5.4 Dual-source output writer update (Step 4.8 revision 2026-05-20)

Update `write_mc_fair_odds_csv(...)` signature:

```python
def write_mc_fair_odds_csv(results, event, source: str, params, computed_at):
    """source ∈ {'self', 'eloratings', 'baseline'}. Filename includes source."""
    out_path = Path(f"data/mc_simu/mc_fair_odds_{event}_{source}_"
                    f"{computed_at.strftime('%Y-%m-%d')}.csv")
    # ... rest unchanged, but write `source` to a new "elo_source" column
```

Schema column `elo_source` added (TEXT, NOT NULL). Phase 4 ships 2 CSV files (self + eloratings) + optional baseline reference per user decision 2026-05-20.

`run_mc_simu.py` (Step 4.9) takes `--elo-source` CLI flag:
```bash
python src/run_mc_simu.py --event world_cup_2026 --elo-source self
python src/run_mc_simu.py --event world_cup_2026 --elo-source eloratings
python src/run_mc_simu.py --event world_cup_2026 --elo-source baseline   # optional
```

Or `--all-sources` to run all three sequentially.

---

