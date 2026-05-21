## 1. Phase 0 — Pre-flight Audit (3 days)

**Goal:** Verify existing project state + historical data quality. **HARD STOP** before Phase 1 nếu fail.

### 1.1 Existing project pre-flight checks

**Read-only checks** — verify pipeline hiện tại còn hoạt động, KHÔNG modify gì.

**v1 invocation (file-only default):**
```bash
python src/mc_simu/preflight.py             # Runs Check 0.4 only (entity_map.json)
python src/mc_simu/preflight.py --with-db   # All 5 checks (requires DATABASE_URL)
```

Per spec §0.6 v1 CSV-only path, DB checks are **opt-in via `--with-db` flag** (not hard required in v1). Default invocation runs Check 0.4 (file-based) only. DB checks reserved for v2 production deploy verification.

```python
# src/mc_simu/preflight.py — actual v1 implementation
def run_preflight(*, with_db: bool):
    """Returns list of (label, status, detail). Status: PASS | NOT RUN | FAIL.
    HARD STOP on FAIL via _common.hard_stop()."""

    # Check 0.4 — entity_map.json (always runs; file-based)
    assert Path("data/clean/entity_map.json").exists()

    if with_db:
        # Check 0.1 — DB connectivity
        from db import get_engine
        eng = get_engine()
        if eng is None: hard_stop("0.1 DB connectivity", "None", "live Engine")

        # Check 0.2 — Required tables (5 spec + 2 new in post-merge main:
        #             api_call_log, live_price_cache). issubset check.
        # Check 0.3 — Recent fair_odds writes in last 7 days
        # Check 0.5 — fair_odds columns ⊇ {event, market, outcome, fair_prob,
        #             z_estimate, n_sources, computed_at}
        #             (actual schema includes consensus_prob added in PR #2 — superset OK)
    # Else: Checks 0.1/0.2/0.3/0.5 → NOT RUN (deferred to v2)
```

**Implementation note (from Phase 0 build):** post-merge main HEAD `0773e2c` includes 2 extra tables (`api_call_log`, `live_price_cache`) added via PR #1+#2; preflight Check 0.2 uses `issubset` semantics so extras are non-blocking. `fair_odds` schema also gained `consensus_prob` column — same superset logic applies to Check 0.5.

### 1.2 Data sources

| Dataset | Where to get | Format | Used by |
|---|---|---|---|
| Historical international matches 1872-2026 | **GitHub mirror** (used in Phase 0): `https://raw.githubusercontent.com/martj42/international_results/master/results.csv` (no auth, ~3.7 MB, 49k+ rows). Fallback: Kaggle dataset same maintainer (manual download). | CSV — actual columns: `date,home_team,away_team,home_score,away_score,tournament,city,country,neutral` | Elo training, validation |
| ClubElo ratings ⛔ v2 only | API: `http://api.clubelo.com/{ClubName}` (no key, free). Dated-endpoint `/{YYYY-MM-DD}` returns full ranking (verified 633 European clubs total). | TSV per club: `Rank,Club,Country,Level,Elo,From,To` | ~~Roster overlay~~ — SKIPPED v1 per §0.8 |
| WC2026 squads ⛔ v2 only | Wikipedia "2026 FIFA World Cup squads" (19/48 teams published by May 17; final deadline FIFA June 2) | Scraped JSON via `scrape_wc2026_squads.py` | Phase 0 Check 5 (informational) / v2 |
| WC2026 fixtures | Generated from `data/mc_simu/wc2026_groups.json` round-robin enumeration + 32 KO placeholders. Opening match (Mexico vs ZA at Estadio Azteca) + Final (MetLife) + 3rd-place (Hard Rock) populated; intermediate venues `TBD` until Phase 3 needs them. | `build_wc2026_fixtures.py` → CSV | Phase 3 simulator |
| R32 cross-seeding | **Wikipedia "2026 FIFA World Cup knockout stage"** — has full 495-row Annex C combinations table parseable via BS4 | `build_r32_seeding.py` → JSON | Phase 3 simulator |
| Past tournament fixtures + results | Same GitHub mirror, filter by `(tournament, date.year)` — note Euro 2020 played 2021 (COVID delay) | CSV | Phase 4 LOTO-CV |
| Euro 2020 attendance metadata | `scrape_euro2020_attendance.py` from Wikipedia main + group + knockout pages (51 matches). Venue capacity hardcoded for 11 Euro 2020 venues. | CSV column `attendance_pct` | COVID ghost-game scaling |

**Fallbacks:**
- If GitHub mirror lags behind Kaggle: manual Kaggle download (`https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017` → Sign in → Download → extract to `data/mc_simu/results.csv` → re-run `download_match_history.py --skip-download` for validation).
- Wikipedia scrape: raw HTML snapshots cached to `data/mc_simu/cache/` for idempotent re-parse if upstream format drifts.

### 1.3 Data quality checks (mandatory)

Create `src/mc_simu/data_audit.py`. All 6 checks must PASS:

**Check 1 — International match history completeness**

> ⚠ **Spec correction (post-Phase-0 build):** Kaggle dataset `tournament` column stores just `"FIFA World Cup"` / `"UEFA Euro"` (NO year embedded). Original spec pseudocode `df['tournament'] == 'FIFA World Cup 2002'` would always return 0 rows. Real implementation uses `(tournament, year_from_date)` tuple. Plus Euro 2020 played in 2021 (COVID delay) — special-cased.

```python
TOURNAMENT_EXPECTED = {
    # label → (kaggle_tournament_str, actual_year, expected_match_count)
    "FIFA World Cup 2002": ("FIFA World Cup", 2002, 64),
    ...
    "UEFA Euro 2020": ("UEFA Euro", 2021, 51),   # COVID delay → played 2021
    "UEFA Euro 2024": ("UEFA Euro", 2024, 51),
}

def check_1_history_completeness(df):
    required_cols = {'date', 'home_team', 'away_team', 'home_score', 'away_score',
                     'tournament', 'neutral', 'country'}
    assert required_cols.issubset(df.columns)

    dt = pd.to_datetime(df['date'])
    assert dt.min() <= pd.Timestamp('1998-01-01')   # raw 1872-2026 satisfies
    assert dt.max() >= pd.Timestamp('2026-01-01')

    for label, (tname, year, n) in TOURNAMENT_EXPECTED.items():
        sub = df[(df['tournament'] == tname) & (dt.dt.year == year)]
        assert len(sub) / n >= 0.80, f"{label}: {len(sub)}/{n}"
        assert sub[['home_score', 'away_score']].notna().all().all()
```

**Phase 0 actual result:** 12/12 tournaments → **100% coverage** (64/64 WC, 31/31 or 51/51 Euros).

**Check 2 — Tournament type → K-factor mapping**

```python
def check_tournament_types(df):
    counts = df['tournament_type'].value_counts()
    required = ['world_cup_final', 'continental_final', 'qualifier',
                'nations_league', 'friendly']
    for cat in required:
        assert cat in counts.index, f"Missing tournament_type: {cat}"
```

**Check 3 — Neutral venue flag consistency**

```python
def check_neutral_flag(df):
    """Host's matches NOT neutral, others ARE."""
    host_map = {
        'FIFA World Cup 2014': 'Brazil', 'FIFA World Cup 2018': 'Russia',
        'FIFA World Cup 2022': 'Qatar', 'UEFA Euro 2016': 'France',
        'UEFA Euro 2024': 'Germany',
    }
    for tournament, host in host_map.items():
        sub = df[df['tournament'] == tournament]
        host_matches = sub[(sub['home_team'] == host) | (sub['away_team'] == host)]
        assert host_matches['neutral'].sum() == 0, f"{tournament}: host matches misflagged"
        other = sub[(sub['home_team'] != host) & (sub['away_team'] != host)]
        assert other['neutral'].all(), f"{tournament}: non-host matches misflagged"
```

**Check 4 — Euro 2020 attendance metadata**

```python
def check_euro2020_attendance(df):
    """Euro 2020 was behind closed doors / capped. Need per-match attendance %
    for COVID ghost-game scaling per HFA spec section 0.5.
    
    Note: `df` is the same match history dataframe used in Check 1, after
    joining with euro2020_attendance.csv on (date, home_team, away_team)."""
    e2020 = df[df['tournament'] == 'UEFA Euro 2020']
    assert 'attendance_pct' in e2020.columns
    assert e2020['attendance_pct'].notna().all()
    low_attendance = (e2020['attendance_pct'] < 0.5).sum()
    assert low_attendance >= len(e2020) * 0.4, "Too few low-attendance matches"
```

**Check 5 — ClubElo coverage** ⚠ DOWNGRADED v1 (informational only, see §0.8)

Per §0.8 decision, Phase 2 SKIPPED → Check 5 result no longer blocks Phase 1. Implementation preserved for diagnostic (informs v2 revisit if better roster source identified). Status downgraded from HARD STOP to WARN: <85% per team logged but does not block gate.

```python
def check_clubelo_coverage(squads):
    """v1: WARN status only — Phase 2 skipped per §0.8 ClubElo scope analysis."""
    per_team = {}
    for team, squad in squads.items():
        mapped = sum(1 for p in squad if clubelo_get(p['club']) is not None)
        per_team[team] = mapped / len(squad)
    below_85 = {t: c for t, c in per_team.items() if c < 0.85}
    # No assert — informational only in v1
    return {"per_team": per_team, "below_85": below_85}
```

**Check 6 — WC2026 fixtures + R32 seeding integrity**

```python
def check_wc2026_structure(fixtures, r32_table):
    assert len(fixtures) == 104
    
    group_stage = fixtures[fixtures['stage'] == 'group']
    assert len(group_stage) == 72
    for group in 'ABCDEFGHIJKL':
        assert len(group_stage[group_stage['group'] == group]) == 6
    
    # R32 cross-seeding: 495 mappings (C(12,8))
    assert len(r32_table) == 495
```

### 1.4 Phase 0 deliverables (✅ COMPLETED 2026-05-17)

**Code** (all under `src/mc_simu/`):
- `__init__.py` — package exports
- `_common.py` — `hard_stop()`, `banner()`, `infer_tournament_type()`, UTF-8 stdout fix
- `preflight.py` — 5 pre-flight checks (Check 0.4 file-only default + `--with-db` opt-in for 0.1/0.2/0.3/0.5)
- `data_audit.py` — 6 data quality checks orchestrator + report writer
- `download_match_history.py` — GitHub mirror fetcher (idempotent, `--force`, `--skip-download`)
- `scrape_euro2020_attendance.py` — Wikipedia scraper (main + 6 group pages + knockout page), 11-venue capacity table hardcoded
- `scrape_wc2026_squads.py` — Wikipedia "2026 FIFA World Cup squads" parser
- `clubelo_client.py` — HTTP + 50+ aliases + JSON cache (kept as v2 byproduct per §0.8)
- `build_wc2026_fixtures.py` — round-robin enumerator from groups JSON + KO placeholders
- `build_r32_seeding.py` — Wikipedia 495-row Annex C table parser (hand-verifies sample)

**Tests** (84/84 passing):
- `tests/conftest.py` — sys.path bootstrap + `project_root`, `mc_simu_data_dir`, `mc_simu_fixtures_dir` fixtures
- `tests/test_mc_preflight.py` — 7 tests (hard_stop overridable, --with-db flag passthrough, etc.)
- `tests/test_mc_data_audit.py` — 14 tests (all 6 checks + report writer + hard-stop scenarios)

**Data** (all under `data/mc_simu/`):
- `results.csv` — raw GitHub mirror, 49,329 rows, 3.7 MB
- `matches_1998_2026.csv` — filtered + derived `tournament_type` column, 26,805 rows
- `euro2020_attendance.csv` — 51 matches, 41 low-attendance (<50% capacity)
- `wc2026_fixtures.csv` — 104 rows (72 group + 32 KO)
- `wc2026_squads_provisional.json` — 19 teams, 693 player rows (29 teams pending FIFA June 2 deadline)
- `r32_seeding_table.json` — **495 combinations** (FIFA Annex C, full coverage)
- `clubelo_cache.json` — 316 unique clubs cached
- `cache/*.html` — 10 raw Wikipedia HTML snapshots for re-parse

**Phase 0 final gate status:**

| Check | Status | Detail |
|---|---|---|
| Preflight 0.1/0.2/0.3/0.5 | NOT RUN | DB checks deferred to v2 (file-only path) |
| Preflight 0.4 | ✅ PASS | entity_map.json 6440 bytes |
| Check 1 — History | ✅ PASS | 12/12 tournaments 100% (49,329 rows) |
| Check 2 — Tournament types | ✅ PASS | 5/5 categories |
| Check 3 — Neutral flag | ✅ PASS | 5/5 hosts validated |
| Check 4 — Euro 2020 attendance | ✅ PASS | 51/51 merged, 41 low-attendance |
| Check 5 — ClubElo coverage | ⚠ WARN | 55% mean (downgraded informational per §0.8) |
| Check 6 — WC2026 structure | ✅ PASS | 104 fixtures + 495 R32 |

**HARD STOP gate:** ✅ PASS — ready for Phase 1 (1 WARN, non-blocking per §0.8 decision).

---

