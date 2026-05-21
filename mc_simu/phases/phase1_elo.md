## 2. Phase 1 — Elo Engine + Single-Game Model (5 days)

**Goal:** `predict_match(team_A, team_B, context) → (P_W, P_D, P_L, goal_dist)` with Brier ≤ 0.20.

**Files:** `src/mc_simu/elo.py`, `src/mc_simu/single_game.py`

### 2.1 Build steps

**Step 1.1 — `elo.py`: expected formula**

```python
def elo_expected(rating_home: float, rating_away: float,
                 hfa_elo: float = 100.0, is_neutral: bool = False) -> float:
    """Return P(home_advance) via logistic from Elo difference."""
    rh = rating_home + (0 if is_neutral else hfa_elo)
    return 1.0 / (1.0 + 10 ** (-(rh - rating_away) / 400))
```

Unit tests: equal ratings neutral → 0.5; equal+HFA → ~0.64; +400 Elo neutral → ~0.91.

**Step 1.2 — `elo.py`: update + MoV**

```python
K_FACTORS = {
    'world_cup_final': 60, 'continental_final': 50,
    'qualifier': 40, 'nations_league': 40,
    'friendly_in_window': 30, 'friendly_out_window': 20,
}

def mov_multiplier(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    if gd == 3: return 1.75
    return 1.75 + (gd - 3) / 8

def elo_update(rating, expected, actual, K, mov_mult=1.0):
    return rating + K * mov_mult * (actual - expected)
```

**Step 1.3 — `elo.py`: history builder**

```python
def build_rating_history(matches_df, initial_rating=1500, time_halflife_months=18):
    """Process matches chronologically, return {team: DataFrame(date, rating)}."""
```

Sanity check post-build:
- Brazil peak ~2150 around 2002
- Germany peak ~2100 around 2014
- Spain peak ~2120 around 2010
- **HARD STOP** if any team Elo < 1000 or > 2500 (numerical bug)

**Step 1.4 — `single_game.py`: Elo → λ**

```python
def elo_to_lambda(elo_team, elo_opp, hfa_log_goals=0.0, league_avg_goals=2.7):
    """Map Elo difference + HFA additive to expected goals."""
    half_avg = league_avg_goals / 2
    base_lambda = half_avg * 10 ** ((elo_team - elo_opp) / 400)
    return base_lambda * np.exp(hfa_log_goals)
```

**Step 1.5 — `single_game.py`: Poisson grid + diagonal inflation**

```python
def goal_distribution(lambda_home, lambda_away, diagonal_inflation=0.09, max_goals=8):
    grid = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            grid[i, j] = poisson_pmf(i, lambda_home) * poisson_pmf(j, lambda_away)
    for k in range(max_goals + 1):
        grid[k, k] *= (1 + diagonal_inflation)
    return grid / grid.sum()
```

**Step 1.6 — `single_game.py`: predict_match with HFA context**

```python
@dataclass
class MatchContext:
    is_neutral: bool
    tournament_type: str
    home_country: str
    away_country: str
    venue_country: str
    venue_confederation: str
    home_confederation: str
    away_confederation: str
    # Tournament context — empty/None for non-tournament matches
    # (qualifiers, friendlies). For tournament matches, populate from
    # tournament metadata (e.g., WC2026: host_countries=['USA','Mexico','Canada'],
    # host_confederation='CONCACAF').
    host_countries: list[str] = field(default_factory=list)
    host_confederation: str | None = None
    attendance_pct: float = 1.0

@dataclass
class ModelParams:
    alpha: float = 0.40
    beta: float = 0.13
    diagonal_inflation: float = 0.09
    blend_match_weight: float = 0.75

def hfa_log_goals(team_country, team_confederation, ctx, params):
    own_country = team_country == ctx.venue_country
    is_host = team_country in ctx.host_countries
    same_confed_as_host = (team_confederation == ctx.host_confederation 
                           and not is_host)
    
    bonus = (params.alpha if own_country else 0) + \
            (params.beta if same_confed_as_host else 0)
    
    if ctx.attendance_pct < 0.25:
        bonus *= 0.67
    elif ctx.attendance_pct < 0.50:
        bonus *= 0.80
    
    return bonus

def predict_match(elo_home, elo_away, ctx, params):
    hfa_h = hfa_log_goals(ctx.home_country, ctx.home_confederation, ctx, params)
    hfa_a = hfa_log_goals(ctx.away_country, ctx.away_confederation, ctx, params)
    
    lambda_h = elo_to_lambda(elo_home, elo_away, hfa_h)
    lambda_a = elo_to_lambda(elo_away, elo_home, hfa_a)
    
    grid = goal_distribution(lambda_h, lambda_a, params.diagonal_inflation)
    
    p_home = np.sum(np.tril(grid, k=-1))
    p_draw = np.sum(np.diag(grid))
    p_away = np.sum(np.triu(grid, k=1))
    
    return MatchPrediction(p_home, p_draw, p_away, grid)
```

### 2.2 Validation gate — HARD STOP

**Test 1.A — Brier on validation tournaments**

Run on WC 2018, WC 2022, Euro 2020, Euro 2024 (~210 matches). Elo ratings AS OF day before each match (no lookahead).

```python
def validate_phase1():
    val_tournaments = ['FIFA World Cup 2018', 'FIFA World Cup 2022',
                       'UEFA Euro 2020', 'UEFA Euro 2024']
    briers = []
    for t in val_tournaments:
        for m in load_matches(t):
            elo_h = get_rating(m.home, as_of=m.date - timedelta(days=1))
            elo_a = get_rating(m.away, as_of=m.date - timedelta(days=1))
            pred = predict_match(elo_h, elo_a, m.context, DEFAULT_PARAMS)
            actual = (1, 0, 0) if m.home_wins else (0, 1, 0) if m.draw else (0, 0, 1)
            brier = sum((p - a)**2 for p, a in zip(
                [pred.p_home, pred.p_draw, pred.p_away], actual))
            briers.append(brier)
    return np.mean(briers)
```

**Gate:**
- Avg Brier ≤ 0.20 → ✓ proceed to Phase 2
- Avg Brier > 0.21 → **HARD STOP**, switch to Karlis-Ntzoufras (see 2.3)

### 2.3 Karlis-Ntzoufras fallback (only if 2.2 fails)

**Time budget:** 3 days.

Karlis-Ntzoufras bivariate Poisson trong `footBayes` R package:

1. **rpy2 wrapper** (preferred): `pip install rpy2`, then `R -e 'install.packages("footBayes")'`
2. **Stan direct** (fallback): Re-implement in PyStan/CmdStanPy

Decision authority for switch: confirm with Sam/Duy Anh.

---

### 2.4 Alternative Elo source: eloratings.net scrape (2026-05-20 ADDED)

**Rationale (Sam's framework note 2026-05-20):** treat per-game probability model as a plug-in. Self-computed Elo (§2.1-2.2) is one model; eloratings.net is a second source we can plug into the same simulator. Compare in Phase 4 historical replay (see [phase4_calibration.md](phase4_calibration.md) §5.3).

**File:** `src/mc_simu/eloratings_client.py`

**Source:** `https://eloratings.net/{country}` — per-team page contains full match-by-match history (Date | Match | Tournament | Rating change | Resulting rating | Rank). Confirmed via screenshot 2026-05-20: each row is a "rating-after-match" point — gives effective "as-of date X" lookup by filtering `date < X`, taking last row.

**Scope:** all ~240 nations (verified 2026-05-20 by user) on eloratings.net (per user decision 2026-05-20). Approx ~1-2h scrape with polite throttling.

```python
# src/mc_simu/eloratings_client.py — minimal sketch
def scrape_team_history(country: str) -> pd.DataFrame:
    """Fetch per-match Elo time-series for one nation.

    Returns DataFrame matching elo.build_rating_history() output schema for
    drop-in compat with get_rating_as_of():
        date, rating_after, opponent, tournament_type, K, mov, gs_self, gs_opp

    Caches HTML to data/mc_simu/cache/ like scrape_euro2020_attendance.py.
    """

def scrape_all_nations(country_list: list[str]) -> dict[str, pd.DataFrame]:
    """Iterate scrape_team_history() over ~240 nations (verified 2026-05-20 by user).
    Output: data/mc_simu/elo_history_eloratings.csv (long format)."""
```

**Name normalization:** eloratings.net uses some non-standard names (vd "USA" vs Kaggle's "United States", "Bosnia and Herzegovina" vs "Bosnia"). Reuse `entity_map.json` from `src/normalize.py` + extend per-need alias dict (similar pattern to `clubelo_client.py` NAME_ALIASES).

**Output schema (drop-in compat):**
```
data/mc_simu/elo_history_eloratings.csv
  columns: team, date, rating_after, opponent, tournament_type, K, mov, gs_self, gs_opp
  (same schema as elo.history_to_long_df() — load via pd.read_csv + groupby('team'))
```

**Loader for downstream use (Phase 3+4):**
```python
# src/mc_simu/elo_source.py — NEW abstraction
def load_elo_history(source: Literal['self', 'eloratings']) -> dict[str, pd.DataFrame]:
    """Return {team: DataFrame} usable by elo.get_rating_as_of().

    source='self':       build_rating_history(matches_1998_2026.csv)
    source='eloratings': read elo_history_eloratings.csv → groupby('team')
    """
```

**Cross-check sanity gate (HARD STOP if fails):**
- Top-20 nations as of 2026-05-01: self-Elo vs eloratings-Elo diff < 100 points
- Lệch >100 ở >5/20 đội → bug ở scraping, K-factor mismatch, hoặc tournament classification mismatch
- Document differences in this phase doc under §2.4

**Why both sources, not just eloratings:** preserves reproducibility + audit control. We control self-compute; eloratings.net could update methodology silently. Keeping both lets us spot regressions and cross-validate.

---

