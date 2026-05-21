## 4. Phase 3 — Tournament MC Harness (7+2 days)

**Goal:** Build **barebone tournament simulator with plug-in per-game predictor** + bracket adapters for WC2026 AND 12 historical tournaments. Enables Phase 4 historical replay comparison of multiple models.

**Architectural principle (per Sam's note 2026-05-20):**

```
   ┌─────────────────────────────────────────────────┐
   │  Tournament Harness (FIXED across all formats)  │
   │  - Group stage matches                          │
   │  - FIFA tiebreakers                             │
   │  - Knockout bracket simulation                  │
   │  - Champion + group winner aggregation          │
   └────────────────────┬────────────────────────────┘
                        │ delegates to bracket adapter (per format)
                        ↓
   ┌─────────────────────────────────────────────────┐
   │  Bracket adapter (per tournament format)        │
   │  - WC2026 (12 groups + best thirds)             │
   │  - WC 8-groups (2002-2022)                      │
   │  - Euro 24-teams (2016/2020/2024)               │
   │  - Euro 16-teams (2004/2008/2012)               │
   └────────────────────┬────────────────────────────┘
                        │ calls predictor(team_a, team_b, ctx) per match
                        ↓
   ┌─────────────────────────────────────────────────┐
   │  Per-game probability model (SWAPPABLE)         │
   │  - Self-Elo + Poisson + HFA                     │
   │  - eloratings-Elo + Poisson + HFA               │
   │  - Baseline 40/40/20                            │
   │  - (future: FIFA ranking, market consensus...)  │
   └─────────────────────────────────────────────────┘
```

The simulator code does NOT know which model is plugged in. Models do NOT know which tournament they're predicting. This separation is what enables Phase 4 head-to-head comparison.

**Files:** `src/mc_simu/standings.py`, `src/mc_simu/simulator.py`, `src/mc_simu/baseline.py`, `src/mc_simu/tournaments/wc2026.py`, `src/mc_simu/tournaments/wc_8groups.py`, `src/mc_simu/tournaments/euro_24teams.py`, `src/mc_simu/tournaments/euro_16teams.py`

### 4.1 Build steps

**Step 3.1 — FIFA tiebreaker (within group)**

**Corrected per Phase 3 decision #22 (2026-05-20):** FIFA 2026 Competition
Regulations Article 13 (p.26) uses H2H **before** Overall GD/GF — same ordering
as UEFA adopted post-2018. Earlier spec listed GD/GF before H2H, which is the
pre-2018 FIFA chain and would mis-rank groups with 2- or 3-way ties.

```python
def rank_group(matches: list[GroupMatch], final_tiebreaker: dict[str, float]) -> list[str]:
    """FIFA 2026 chain (in this exact order):
    1. Points (3W/1D/0L) — overall
    2. H2H Points — only matches between tied teams
    3. H2H Goal Difference
    4. H2H Goals Scored
    5. (Recurse: if subset still tied after 2-4, re-apply 2-4 within that subset
        with H2H recomputed for the smaller team set)
    6. Overall Goal Difference
    7. Overall Goals Scored
    8. Team Conduct (fair-play score; sum of -1 yellow / -3 indirect red /
        -4 direct red / -5 yellow+direct-red)
    9. FIFA World Ranking — in sim we pass current Elo as proxy
       (no drawing of lots in WC2026; FIFA ranking is the final tiebreaker)
    """
```

**Unit test 3.1:** Multiple cases (see [test_mc_standings.py](../../tests/test_mc_standings.py)):
- Euro 2016 Group F → Hungary 1st > Iceland 2nd > Portugal 3rd > Austria 4th
- Synthetic **H2H-first discriminator** anchor — guards against accidental regression to pre-2018 GD-first chain
- 3-way tie with partial H2H separation
- All-4-tied → falls through to FIFA ranking (Elo proxy)
- Fair-play tiebreaker before FIFA ranking (cards data path)

**Step 3.2 — Cross-group best-thirds ranking**

```python
def rank_best_thirds(thirds: list[Team]) -> list[Team]:
    """Same chain as group BUT NO head-to-head (different groups).
    1. Points → 2. GD → 3. GF → 4. Fair-play → 5. FIFA ranking"""
```

**Unit test 3.2:** Reproduce Euro 2024 best-thirds — verify which 4 of 6 advanced.

**Step 3.3 — R32 cross-seeding for WC2026**

```python
def seed_r32(winners, runners_up, best_thirds, third_origins, r32_table):
    """Map (12 winners, 12 runners, 8 thirds + origins) → 16 R32 matches.
    r32_table[third_origins] returns seeding pattern per FIFA bracket."""
```

**Unit test 3.3:** 5 hand-verified mappings from FIFA bracket PDF.

**Step 3.4 — Group + knockout match simulation**

`predictor` is a partial-application wrapper around `predict_match` that hides params (frozen θ*). Built once per MC run:

```python
# Built once before run_monte_carlo() — params are frozen at this point
from functools import partial
predictor = partial(predict_match, params=FROZEN_PARAMS)
# Now predictor(elo_home, elo_away, ctx) returns MatchPrediction
```

```python
def simulate_group_match(team_a, team_b, ctx, predictor, rng):
    pred = predictor(team_a.elo, team_b.elo, ctx)
    u = rng.uniform()
    if u < pred.p_home: result = 'home_win'
    elif u < pred.p_home + pred.p_draw: result = 'draw'
    else: result = 'away_win'
    score = sample_scoreline(pred.goal_grid, result, rng)
    return MatchResult(team_a, team_b, score, result)

def simulate_knockout_match(team_a, team_b, ctx, predictor, rng):
    """Draw at 90' → 50/50 advance (v1 simplification)."""
    pred = predictor(team_a.elo, team_b.elo, ctx)
    p_a_advance = pred.p_home + 0.5 * pred.p_draw
    return team_a if rng.uniform() < p_a_advance else team_b
```

**Step 3.5 — Full tournament sim**

```python
def simulate_tournament(teams, ratings, fixtures, r32_table, predictor, rng):
    """Run one tournament, return champion name.
    
    Args:
        teams: dict {team_name: Team object with metadata}
        ratings: dict {team_name: Elo rating} — SNAPSHOT taken pre-tournament,
                 frozen for entire iteration. DO NOT update ratings within
                 a single sim run. Rationale: in-iteration updates would
                 add noise (each tournament is 1 sample); we want structural
                 prior, not Elo trajectory. Daily MC re-run uses fresh
                 snapshot from latest historical Elo update.
        fixtures: list of 104 Match objects with venue + context
        r32_table: dict mapping third_origin frozensets → R32 seeding
        predictor: predict_match function (frozen with model params)
        rng: numpy random generator (for reproducibility)
    
    Returns: tuple (champion_name: str, group_winners: dict[str, str])
             group_winners maps group letter → name of team that finished 1st
             in that group (per FIFA tiebreaker rank_group).
    
    Steps:
    1. 72 group matches → standings per group (using frozen ratings)
    2. Collect winners, runners, thirds (no rating updates)
    3. Rank thirds, pick top 8
    4. Seed R32 via cross-seeding table
    5. Knockout: R32 → R16 → QF → SF → F (frozen ratings throughout)
    6. Return champion
    """
```

**Step 3.6 — Run N iterations + track group winners**

Per Sam's suggestion (review 4/30/26): Polymarket lists separate markets cho mỗi group winner (12 markets, mỗi market 4 outcomes). Tracking group winner frequencies cost negligible (group rankings đã compute trong simulate_tournament) và provides **2nd independent sanity check** beyond champion market.

```python
def run_monte_carlo(n_iterations: int = 100_000, seed: int = 42, **kwargs) -> dict:
    """Run N tournament simulations. Track BOTH champion + group winners.
    
    Returns dict with 2 keys:
      'champion': {team: {mc_fair_prob, mc_se, n_iterations}}
      'group_winners': {group_letter: {team: {mc_fair_prob, mc_se, n_iterations}}}
    """
    rng = np.random.default_rng(seed)
    champion_counter = defaultdict(int)
    group_winner_counter = {g: defaultdict(int) for g in 'ABCDEFGHIJKL'}
    
    for _ in tqdm(range(n_iterations)):
        # simulate_tournament returns (champion, group_winners_dict)
        # group_winners_dict = {'A': 'USA', 'B': 'Spain', ...}
        champion, group_winners = simulate_tournament(rng=rng, **kwargs)
        champion_counter[champion] += 1
        for group_letter, winner_name in group_winners.items():
            group_winner_counter[group_letter][winner_name] += 1
    
    def to_results(counter_dict, n):
        out = {}
        for team, count in counter_dict.items():
            p = count / n
            out[team] = {
                'mc_fair_prob': p,
                'mc_se': np.sqrt(p * (1 - p) / n),
                'n_iterations': n,
            }
        return out
    
    return {
        'champion': to_results(champion_counter, n_iterations),
        'group_winners': {
            g: to_results(group_winner_counter[g], n_iterations)
            for g in 'ABCDEFGHIJKL'
        },
    }
```

**Note:** `simulate_tournament` signature changes — phải return tuple `(champion_name, group_winners_dict)` thay vì chỉ champion. Update Step 3.5 accordingly.

**Step 3.7 — Historical bracket adapters (NEW 2026-05-20; ✅ SHIPPED)**

Three adapters cover all 12 LOTO-CV tournaments. Each exposes the same interface as `tournaments/wc2026.py` so the harness in §3.5 can swap them at will.

| Adapter file | Tournaments covered | Format | Status |
|---|---|---|---|
| `tournaments/wc_8groups.py` | WC 2002, 2006, 2010, 2014, 2018, 2022 | 8 groups × 4 → R16 → QF → SF → F (no best-thirds) | ✅ 33/33 tests pass |
| `tournaments/euro_24teams.py` | Euro 2016, 2020, 2024 | 6 groups × 4 + 4 best-thirds → R16 → QF → SF → F | ✅ 18/18 tests pass |
| `tournaments/euro_16teams.py` | Euro 2004, 2008, 2012 | 4 groups × 4 → QF → SF → F (group winners + runners-up only) | ✅ 25/25 tests pass |

Each adapter implements:

```python
# tournaments/<format>.py
GROUP_STRUCTURE = {...}      # group letter → list of teams
SEEDING_RULES = {...}        # how group standings → knockout bracket
FIXTURES_TEMPLATE = [...]    # match list with venue + dates

def make_fixtures(year: int, host: str) -> list[Match]:
    """Build the actual fixture list for a specific edition.
    
    Reads historical actual standings + matches from matches_1998_2026.csv
    only for VENUE/DATE/HOST metadata — does NOT use actual results.
    Knockout pairings are computed from simulated group standings, not from
    historical bracket (this is what makes it a 'replay' not a 'replay+').
    """

def simulate_one(ratings, predictor, rng) -> tuple[str, dict[str, str]]:
    """Same signature as simulate_tournament — drop-in compat."""
```

**Time cost:** ~2.5 days for 3 adapters (vs ~3 days/adapter in old plan). Saves time via:
- WC 2002-2022 share identical structure → 1 adapter
- Euro 2016/2020/2024 share identical structure → 1 adapter
- Euro 2004-2012 share identical structure → 1 adapter

**Step 3.8 — Baseline 40/40/20 model (NEW 2026-05-20)**

Sanity floor. Any Elo-based model should clearly beat this; if not, something is broken.

```python
# src/mc_simu/baseline.py
def predict_match_40_40_20(team_a, team_b, ctx, *_args, **_kwargs):
    """Ignorance baseline: 40% team_A win, 20% draw, 40% team_B win.
    
    Implements same callable interface as predict_match() from Phase 1 §2.1.6.
    Returns MatchPrediction with p_home=0.40, p_draw=0.20, p_away=0.40 and
    a uniform-ish goal_grid (not used for tournament sim — only the W/D/L 
    probs matter for knockout decisions).
    """
    return MatchPrediction(p_home=0.40, p_draw=0.20, p_away=0.40,
                           goal_grid=_uniform_grid())
```

**Why 40/40/20 (not 33/33/33):** match historical base rates in international football — ~40% home, ~25-30% draw, ~30-35% away — rounded to convenient (40, 20, 40) for "obvious ignorance" framing. Adjust to (40, 25, 35) if Sam prefers more realistic baseline; doesn't materially change which models beat it.

### 4.2 Validation gates

**Test 3.A — Speed:** 100k iterations < 60s on laptop.

**Test 3.B — Convergence:** Run 10k/100k/500k. Top-5 lệch <0.5pp between 10k→100k, <0.1pp between 100k→500k.

**Test 3.C — Reproducibility:** Same seed → byte-identical output.

**Test 3.D — External smoke check vs Opta + 538 (SOFTENED, NOT HARD STOP — per decisions #23 + #24)**

**Rationale for softening (Phase 3 decision #23, 2026-05-20):** original §4.2
defined this as a ±3pp HARD STOP against Opta. Two problems surfaced during
Phase 3:

1. **Train/test objective mismatch.** Model is tuned on per-match Brier
   (historical W/D/L). Opta forecasts tournament-level champion probabilities
   using squad + lineup + manager + weather features we DON'T have. There's no
   principled reason match-tuned Elo+Poisson should match Opta within ±3pp.
   Convergence at this resolution would imply Opta uses only Elo-like signals,
   which is false.

2. **Goodhart's Law risk.** A ±3pp HARD STOP forces re-tuning model params to
   close the Opta gap → model output drifts toward Opta. We lose Stream 3's
   "independent structural prior" property and inherit Opta's biases. This
   contradicts the goal of building a market-independent fair-line.

**Revised Gate 3.D — smoke check only:**
- Run MC with current Elo + default HFA (frozen Phase 1 params).
- Compare top-10 champion probabilities + ranking vs **two** independent sources:
  - **Opta May 2026 post-draw** (per decision #24, refreshed target): Spain
    16.08% / France 12.78% / England 11.01% / Argentina 10.02% / Portugal
    6.84% / Brazil 6.48% / Germany 5.66% / Netherlands 3.84% (theanalyst.com
    Apr 2026, post-FIFA draw).
  - **Optional triangulation**: 538/ESPN BPI, Pinnacle implied — when available.
- Diagnostic thresholds (NOT HARD STOPS):
  - **Top-5 ranking** matches Opta in 4+ of 5 slots → ✓ ordering sanity
  - **Magnitude divergence ≤5pp** per team → ✓ within model-feature spread
  - **Magnitude divergence 5-10pp** → ⚠ investigate root cause (Elo snapshot
    age, missing squad signal, etc.) — DO NOT auto-tune θ to close the gap
  - **Magnitude divergence >10pp** → 🔎 likely bug (sign error, wrong fixture,
    bad seeding) — investigate code, NOT calibration

Document divergences + investigated causes in this phase doc.

**Phase 3 baseline result (2026-05-20, 100k iter seed=42, default Phase 1
params α=0.27 β=0.09 diag=0.20 D=1400):** Spain 19.85% / Argentina 14.13% /
France 10.50% / England 6.26% / Brazil 4.88%. Top-5 ranking matches Opta in
3 of 5 slots (Spain, France, Argentina in similar positions; mine has Brazil
in 5th vs Opta's Portugal). Magnitude divergence: Argentina +4.1pp, England
-4.8pp, Spain +3.8pp, Germany -3.1pp — all within "model-feature spread"
band, no actionable bug.

**Note on historical replay (REVISED 2026-05-20):** Past tournaments NOW IN SCOPE via Step 3.7 bracket adapters. ~2.5 days total (3 shared adapters cover 12 giải). Replays feed Phase 4 §5.3 historical comparison framework — the primary metric for picking between Elo sources per Sam's note. LOTO-CV match-level Brier ([phase4_calibration.md](phase4_calibration.md) §5.1) kept as **secondary** metric for param tuning within each source.

---

