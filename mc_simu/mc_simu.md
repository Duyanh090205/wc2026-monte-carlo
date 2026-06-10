# Monte Carlo Tournament Simulator — Implementation Plan

**Version:** v3.1 (Phase 2 skipped post-Phase-0 ClubElo coverage analysis) | **Last updated:** May 17, 2026 | **Target ship for review:** Before June 11, 2026

---

## Phase Navigation

This plan is split across multiple files. Each phase has its own file; this index keeps cross-cutting context (§0 scope, §7 testing, §8 file layout, §9 timeline, §10 done, §11 log).

| Phase | File | Status |
|---|---|---|
| 0 — Pre-flight Audit | [phases/phase0_audit.md](phases/phase0_audit.md) | ✅ DONE (2026-05-17) |
| 1 — Elo + Single-Game Model | [phases/phase1_elo.md](phases/phase1_elo.md) | 🔄 IN PROGRESS |
| 2 — Roster-Elo Overlay | (skipped — see §0.8) | ⛔ SKIPPED |
| 3 — Tournament MC Harness | [phases/phase3_simulator.md](phases/phase3_simulator.md) | ✅ DONE (2026-05-20) |
| 4 — Calibration & MC Output | [phases/phase4_calibration.md](phases/phase4_calibration.md) | ⏳ PENDING |
| 5 — Presentation Prep | [phases/phase5_presentation.md](phases/phase5_presentation.md) | ⏳ PENDING |

---

## 0. Context & Scope

### 0.1 What we are building

FairLine hiện có 2 sources cho `fair_prob`:

- **Stream 1 — Sportsbook devigged**: Shin + Power devigging trên 26+ books → `fair_odds` table (`src/fair_odds.py`)
- **Stream 2 — Prediction market**: Polymarket + Kalshi mid bid/ask → `pm_prices` table (`src/pm_snapshot.py`)

Cả 2 đều là market consensus. Khi cả 2 cùng sai (vd under-price tail risk), không phát hiện được.

**Stream 3 — Monte Carlo tournament** là source thứ 3 độc lập dựa vào **structural prior**:

```
Input:  team Elo ratings + tournament bracket structure
Method: simulate full tournament 100,000 times, count champion frequency
Output: mc_fair_prob cho mỗi đội vô địch → new mc_fair_odds table
```

Khi Stream 3 lệch Stream 1+2 → model sai (tune) hoặc market sai (signal to investigate). Calibration loop quyết định.

**Relationship với Sam's bracket-aware repricing framework:** Stream 3 cung cấp foundation MC distribution để Sam có thể precompute 2nd-order effects (bracket-aware repricing) — đó là separate project sau v1.

### 0.2 Build philosophy — validation first, integration later

**v1 scope (this plan):** Build + validate MC simulator as **standalone research tool**. Output to `mc_fair_odds` table only. Present results to Sam + team for review.

**v2 scope (after team approval):** Integrate vào `live_edge.py` để contribute trading signals. NOT covered in this plan.

**Why this separation:**
- Model validation (Brier, calibration, sanity checks) hoàn toàn standalone — không cần touch live engine
- Team review trước khi commit Stream 3 vào trading flow → reduce risk
- Phase 4 sản phẩm cuối là: model + frozen params + presentation materials, KHÔNG phải live trading

### 0.3 Scope

**TRONG scope:**
- Tournament-level outright market (champion to win) + 12 group-winner markets
- 12 historical tournaments cho validation: **WC 2002, 2006, 2010, 2014, 2018, 2022** (6 giải) + **Euros 2004, 2008, 2012, 2016, 2020, 2024** (6 giải). Tổng ~760 matches across 12 finals + ~10,000 qualifier/friendly để fit Elo gốc.
- Anchor: WC2026 (kickoff 11/6/2026, 104 matches, 48 teams, 12 groups + 8 best thirds → R32)
- **Barebone simulator with plug-in per-game models** (per Sam's framework 2026-05-20): tournament harness + bracket adapters are FIXED; per-game probability model is SWAPPABLE
- **Three plug-in models compared head-to-head** (2026-05-20 ADDED):
  - Self-Elo from `build_rating_history()` (Phase 1 §2.1-2.2)
  - eloratings.net scraped per-match Elo time-series (Phase 1 §2.4)
  - Baseline 40/40/20 sanity floor (Phase 3 §3.8)
- **Bracket adapters for historical replay** (2026-05-20 ADDED, formerly §4.2 out-of-scope): `wc_8groups` (6 giải), `euro_24teams` (3 giải), `euro_16teams` (3 giải) → covers all 12 LOTO-CV tournaments. Enables Phase 4 primary metric.
- Standalone MC outputs as **two parallel CSVs** per source (`mc_fair_odds_<event>_self_<date>.csv`, `mc_fair_odds_<event>_eloratings_<date>.csv`) — no DB write in v1.
- Generic framework: reuse được cho UCL, NFL Super Bowl 2027 (per-tournament module riêng)
- Presentation deliverables cho team review

**NGOÀI scope (v1):**
- Integration với `live_edge.py` (deferred to v2 after team approval)
- Modification of `paper_trader.py` (deferred to v2)
- Modification of `edge_signals` table schema (deferred to v2)
- Match-level markets (Stream 1+2 đã handle)
- Penalty shootout / extra time sub-model riêng (v1 simplify: Draw at 90' → 50/50)
- Jet lag, altitude, region bonus chung (không đủ data validate — xem 0.5)
- Bracket-aware repricing (Sam's separate project)
- **Phase 2 Roster-Elo overlay — SKIPPED v1** (pre-emptive drop based on ClubElo scope analysis; xem §0.8)

### 0.4 Tuning rules — hard constraints

**TUNE BẰNG:**
- Brier score trên kết quả 3-way W/D/L của trận đã đá (scoreboard outcomes)
- Leave-one-tournament-out cross-validation (LOTO-CV) trên 12 giải lịch sử

**KHÔNG BAO GIỜ TUNE BẰNG:**
- Output của Shin (Stream 1), Polymarket/Kalshi (Stream 2), hoặc market price nào

**TUNE STRUCTURAL PARAMETERS:**
- ✓ HFA magnitude (α, β), diagonal inflation, roster blend
- ✗ Per-team Elo offset, per-team adjustment, manual probability tweak

Market sau khi tune: **sanity check diagnostic only**, không phải training signal.

**Workflow tune cụ thể:**

```
Cho 1 bộ params θ (vd α=0.40, β=0.13, diagonal=0.09, blend=0.75):
  Cho mỗi trận trong ~760 trận của 12 giải lịch sử:
    1. Lấy Elo các đội AS OF ngày trước trận (no lookahead)
    2. MC predict bằng θ: (P_home_win, P_draw, P_away_win)
       Vd: (0.55, 0.25, 0.20)
    3. Lấy kết quả THẬT từ scoreboard
       Vd: France 4-2 Croatia (home thắng) → actual = (1, 0, 0)
    4. Brier = (0.55-1)² + (0.25-0)² + (0.20-0)² = 0.27
  Average Brier qua toàn bộ trận

Repeat cho mọi θ trong PARAM_GRID (54 combinations)
→ Chọn θ* có average Brier nhỏ nhất
```

**Tại sao KHÔNG tune bằng market:**

Nếu tune θ để match market output → MC sẽ học **giống market**. Hậu quả:
- MC ≈ Market → không có signal trading (cả 2 cùng đúng cùng sai)
- Khi market sai (vd Saudi 1-2 Argentina WC2022), MC cũng sai theo
- Stream 3 mất giá trị độc lập, thành Stream 1' (echo của Stream 1+2)

Tune bằng **scoreboard outcomes** (kết quả thật) → MC học **structural truth** từ physical reality, không phải market belief. Khi MC lệch market → đó có thể là edge thật.

### 0.5 Feature spec (chốt sau deep research)

**HFA — additive trong log-goal space:**

```
log(λ_team) = base_strength_from_Elo 
            + α × I(team đá tại stadium nước mình)
            + β × I(team confederation == host confederation AND team ≠ host)
```

| Param | Starting | Tune range | Justification |
|---|---|---|---|
| α | **+0.40** log-goals | 0.25–0.50 | 538 SPI 2018/2022 + Bilalić 2021 (0.44 nat'l team) + Pollard 56-69% |
| β | **+0.13** log-goals | 0.00–0.20 | 538 "one-third of host" both 2018+2022, single-source |

**COVID ghost-game adjustment:** Scale α và β bằng:
- `0.67` nếu attendance < 25% capacity (Bilalić proportional ~⅓ reduction)
- `0.80` nếu attendance 25-50%
- `1.00` (no scaling) nếu attendance > 50%

Apply this scaling cho Euro 2020 backtest. Use match metadata `attendance_pct` from fixtures.

**HFA application matrix for WC2026:**

| Case | α | β | Net log-goal bonus |
|---|---|---|---|
| USA đá tại US stadium | 0.40 | 0 (USA is host) | +0.40 |
| Mexico đá tại Mexico stadium | 0.40 | 0 | +0.40 |
| Canada đá tại Canada stadium | 0.40 | 0 | +0.40 |
| USA đá tại Mexico stadium | 0 | 0 | 0 |
| Mexico đá tại Canada stadium | 0 | 0 | 0 |
| Costa Rica/Jamaica/Panama/Curaçao/Haiti đá bất kỳ stadium | 0 | 0.13 | +0.13 |
| Spain/France/England đá bất kỳ stadium | 0 | 0 | 0 |
| Argentina/Brazil đá bất kỳ stadium | 0 | 0 | 0 |

**BỎ:** altitude (zero practitioner adoption), jet lag (không quantify được), region bonus chung (không đủ data).

**Single-game model components:**

| Param | Value | Mode | Justification |
|---|---|---|---|
| K-factor | eloratings.net default (60/50/40/30/20) | Fix v1 | Proven 27 năm |
| HFA α, β | per 0.5 table | Tune via LOTO-CV | See above |
| Diagonal inflation | starting 0.09 | Tune {0.05, 0.09, 0.12} | 538-style |
| Match-Elo : Roster-Elo blend | **(1.0, 0.0) — Phase 2 SKIPPED** | Fix v1 (no tune) | See §0.8 ClubElo coverage analysis |
| Time decay halflife | 18 months | Fix v1 | 538 default |
| MoV multiplier | sqrt-based per eloratings.net | Fix | Proven |

**Penalty shootout simplification (v1):** Knockout matches với Draw at 90' → `P(home advance) = P(home win 90') + 0.5 × P(Draw 90')`. Gộp ET + shootout vào 50/50.

### 0.6 Output schema (standalone — no live integration v1)

**v1 PATH: CSV-only.** Per user decision 2026-05-17, Phase 4 writes `data/mc_simu/mc_fair_odds_v1_<timestamp>.csv` thay vì DB table. DB integration deferred entirely to v2 (when team approves productionization). Schema below is the DDL for v2 (when DB write is reactivated). v1 CSV uses the same column names + types per row.

**CSV path:** `data/mc_simu/mc_fair_odds_<event>_<YYYY-MM-DD>.csv` — 96 rows (48 champion + 48 group winners).

**v2 DDL (for future activation, NOT applied in v1):**

```sql
CREATE TABLE IF NOT EXISTS mc_fair_odds (
    id                 BIGSERIAL PRIMARY KEY,
    event              TEXT             NOT NULL,
    market             TEXT             NOT NULL,
    team               TEXT             NOT NULL,
    mc_fair_prob       DOUBLE PRECISION NOT NULL,
    mc_se              DOUBLE PRECISION,
    n_iterations       INTEGER          NOT NULL,
    elo_used           DOUBLE PRECISION,
    roster_elo_used    DOUBLE PRECISION,           -- NULL in v1 (Phase 2 skipped)
    hfa_alpha          DOUBLE PRECISION,
    hfa_beta           DOUBLE PRECISION,
    diagonal_inflation DOUBLE PRECISION,
    blend_match_weight DOUBLE PRECISION,           -- always 1.0 in v1 (no roster blend)
    model_version      TEXT             NOT NULL,
    computed_at        TIMESTAMPTZ      NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS mc_fair_odds_uq
    ON mc_fair_odds (event, market, team, computed_at);
CREATE INDEX IF NOT EXISTS mc_fair_odds_lookup
    ON mc_fair_odds (event, market, computed_at DESC);
```

**Important — v1 does NOT modify these tables / code paths:**
- `edge_signals` — schema unchanged
- `paper_positions` — schema unchanged
- `live_edge.py` — code unchanged
- `paper_trader.py` — code unchanged
- `src/db.py` `_SCHEMA` — **no `mc_fair_odds` table added in v1** (deferred to v2; CSV output instead)

Stream 3 v1 hoàn toàn file-based output. Existing trading flow tiếp tục dùng Stream 1 + Stream 2 từ DB.

### 0.7 Hard stop behavior

When plan says **HARD STOP**:
1. Write failure summary to `logs/mc_simu_hard_stop.log` với timestamp + check name + actual vs expected
2. Exit script with code 2 (distinguish from runtime crash exit 1)
3. Do NOT proceed to next phase

Manual override: developer can set env var `MC_SIMU_OVERRIDE_HARD_STOP=1` to bypass; reason must be documented in the relevant phase doc.

### 0.8 Phase 2 decision — SKIPPED v1 (added 2026-05-17)

Sau khi build Phase 0 và verify thực tế ClubElo coverage, quyết định **pre-emptive drop** Phase 2 (Roster-Elo overlay). Match-Elo only ship v1.

**Empirical findings (Phase 0 Check 5):**
- ClubElo API tracks exactly **633 European clubs** (verified via `/2026-05-15` dated endpoint). Country distribution: ENG/ESP/ITA/GER/FRA/BUL/TUR/POR/POL/NED/UKR/SWE/SRB/RUS/ROM/NOR/CZE/BEL/ISR/GRE/SCO/AUT + small leagues.
- WC2026 provisional squads (19/48 teams published by deadline): **mean ClubElo coverage 55%** after all reasonable name aliases applied (Bilbao, ViktoriaPlzen, SlaviaPraha, Wolves, Fenerbahce, Salzburg, Alkmaar, etc.).
- Non-European leagues genuinely not in ClubElo's scope: Brazilian Serie A (Flamengo/Palmeiras/Cruzeiro/Boca/River/Vasco/Botafogo), Mexican Liga MX (Toluca/Cruz Azul/América/UNAM/Tijuana), Saudi Pro League (Al-Hilal/Al-Sadd/Al-Rayyan/Al-Duhail), MLS (Vancouver Whitecaps), Uzbek (Pakhtakor/Nasaf), Auckland FC, Russian Premier (post-sanctions).

**Per-team coverage (post-fix):**
- ✅ PASS (≥85%): Belgium 100%, France 96%, Ivory Coast 96%, Czechia 93% (all European-heavy rosters)
- ⚠ PARTIAL (50-84%): Argentina 69%, Brazil 58%, Sweden 81%, Japan 73%, Bosnia 77%, Haiti 65%, Tunisia 65%, S. Korea 54%, NZ 50%
- ❌ LOW (<50%): Mexico 38%, Colombia 42%, Paraguay 18%, Uzbekistan 20%, Jordan 3%, Qatar 3%

**Decision rationale:**
1. **Systematic bias risk:** Pure 1300-default fallback (spec §9) would under-rate Latin American + Asian teams. Brazil's Flamengo/Palmeiras players (real ClubElo ~1700-1850 if tracked) defaulted to 1300 → blended roster_elo biased downward → systematic under-prediction of South American team strength.
2. **Phase 2 gate (§3.2) anticipated this** — spec already includes "<1.5% Brier improvement → DROP" gate. Coverage analysis = pre-emptive drop without burning 4 days.
3. **Stream 3 design intent (§0.2):** structural prior independent of market. Match-Elo + Poisson + HFA already structurally complete. Roster overlay is enhancement, not load-bearing.
4. **Time saved (~4 days):** buffer for Phase 3 unknowns (R32 sim complexity, Opta sanity check tuning, potential Karlis-Ntzoufras pivot if Phase 1 Brier > 0.21).
5. **Presentation clarity (§6):** simpler model = easier to defend ("ClubElo scope is European-only — insufficient for WC2026 international squads; v1 ships pure match-Elo; v2 may revisit with broader roster data source").

**v2 revisit conditions:**
- Better roster data source available (FIFA player rankings, Transfermarkt valuations, league-baseline default tiers)
- Or scope reduced to European-only tournament (UCL, Euro) where ClubElo coverage is near-100%

**Impact on rest of plan:**
- Phase 2 (§3) — section marked SKIPPED, content preserved as reference
- Param grid (§5.1 Step 4.1) — `blend_match_weight` removed (4 params → 3 params, 54 combos → 27)
- Output schema (§0.6) — `roster_elo_used` and `blend_match_weight` columns always NULL in v1 writes
- File layout (§8) — `roster_elo.py` removed from v1 build; `clubelo_client.py` kept as Phase 0 Check 5 byproduct + v2 future use
- Risk register (§9) — "ClubElo coverage <85%" risk removed (resolved by skip)
- Done definition (§10) — Phase 2 item removed

---

## 7. Testing Strategy

| File | Critical tests |
|---|---|
| `tests/test_mc_preflight.py` | DB checks opt-in, file checks always run, hard_stop override (Phase 0 ✅ done) |
| `tests/test_mc_data_audit.py` | Checks 1-6, report writer, hard_stop scenarios (Phase 0 ✅ done) |
| `tests/test_mc_clubelo_client.py` | Optional Phase 0 byproduct test (kept for v2 reactivation) |
| `tests/test_mc_elo.py` | elo_expected boundaries, MoV formula, history builder ranges |
| `tests/test_mc_single_game.py` | Poisson grid sums to 1, diagonal preserves mass, HFA additive correct |
| `tests/test_mc_standings.py` | Euro 2016 Group F reproduction, FIFA tiebreaker chain |
| `tests/test_mc_simulator.py` | Convergence 10k→500k, R32 seeding samples, reproducibility |
| `tests/test_mc_calibration.py` | LOTO-CV no lookahead, 2σ θ* selection (3-param grid, no blend) |
| `tests/test_mc_output.py` | mc_fair_odds writes correct schema, sum=1, idempotent, blend_match_weight=1.0 |
| ~~`tests/test_mc_roster_elo.py`~~ | ⛔ Not in v1 — Phase 2 skipped per §0.8 |

**No integration test với live_edge / paper_trader trong v1.** These come in v2.

**Daily smoke test** (cron after MC run):
- `mc_fair_odds` row count == 96 for WC2026 (48 champion + 48 group winners)
- `sum(mc_fair_prob)` per market = 1.0 ± 0.001 (champion + 12 group winner markets)
- `max(mc_se) < 0.01` (with 100k iterations)
- `computed_at` within last 25 hours

Alert via Telegram/Slack on any failure.

---

## 8. File Layout & Conventions

```
src/mc_simu/                       # Stream 3 module (under existing src/)
├── __init__.py
├── _common.py                     # hard_stop, banner, tournament_type mapping
├── preflight.py                   # Pre-flight checks (Phase 0; moved to the trading-engine repo at the 2026-06 split)
├── data_audit.py                  # Data quality checks (Phase 0)
├── download_match_history.py      # GitHub mirror fetcher (Phase 0)
├── scrape_euro2020_attendance.py  # Wikipedia scraper (Phase 0)
├── scrape_wc2026_squads.py        # WC2026 squad scraper (Phase 0 / v2 byproduct)
├── clubelo_client.py              # ClubElo API + cache (Phase 0 Check 5 / v2 byproduct)
├── build_wc2026_fixtures.py       # WC2026 fixtures builder (Phase 0)
├── build_r32_seeding.py           # R32 seeding parser (Phase 0)
├── elo.py                         # Elo engine — self-compute (Phase 1)
├── eloratings_client.py           # eloratings.net scraper (Phase 1 §2.4, NEW 2026-05-20)
├── elo_source.py                  # Loader abstraction: {'self','eloratings'} → history (NEW)
├── single_game.py                 # Match prediction with HFA (Phase 1)
├── baseline.py                    # 40/40/20 baseline predict_match (Phase 3 §3.8, NEW)
# roster_elo.py                    # ⛔ SKIPPED v1 per §0.8 (deferred to v2)
├── standings.py                   # Tiebreaker chains (Phase 3)
├── simulator.py                   # Tournament MC harness (Phase 3, plug-in predictor)
├── historical_replay.py           # Phase 4 §5.3 primary comparison (NEW 2026-05-20)
├── calibration.py                 # LOTO-CV + isotonic — per Elo source (Phase 4 secondary)
├── output.py                      # Write 2 CSVs per source (Phase 4 §5.4, REVISED)
├── demo.py                        # Live demo for presentation (Phase 5)
├── frozen_params_self.json        # θ* for self-Elo (DO NOT EDIT post-freeze)
├── frozen_params_eloratings.json  # θ* for eloratings-Elo (NEW 2026-05-20)
└── tournaments/
    ├── wc2026.py                  # WC2026 12-groups + best-thirds (Phase 3)
    ├── wc_8groups.py              # WC 2002-2022 (8 groups → R16) (NEW 2026-05-20)
    ├── euro_24teams.py            # Euro 2016/2020/2024 (NEW 2026-05-20)
    ├── euro_16teams.py            # Euro 2004/2008/2012 (NEW 2026-05-20)
    ├── ucl_2025_26.py             # Future: UCL adapter
    └── nfl_2026_27.py             # Future: NFL adapter

data/mc_simu/                      # Stream 3 data
├── matches_1998_2026.csv          # International matches (Kaggle)
├── elo_history_eloratings.csv     # Scraped per-match Elo time-series (NEW 2026-05-20)
├── clubelo_cache.json             # ClubElo lookups
├── wc2026_fixtures.csv            # 104 matches with venue + attendance_pct
├── wc2026_squads.json             # 48 teams × 23-26 players
├── r32_seeding_table.json         # 495 mappings hand-verified
├── audit_replay_report.csv        # Phase 4 §5.3 historical replay results (NEW)
├── mc_fair_odds_<event>_self_<date>.csv         # Phase 4 dual output (NEW)
└── mc_fair_odds_<event>_eloratings_<date>.csv   # Phase 4 dual output (NEW)

docs/                              # Existing docs/ folder
└── mc_simu_v1_report.md           # Phase 5 technical report

mc_simu/presentation/              # Presentation materials
└── slides.pdf

tests/                              # Existing tests/ folder
├── test_mc_elo.py
├── test_mc_single_game.py
├── test_mc_standings.py
├── test_mc_simulator.py
├── test_mc_calibration.py
├── test_mc_output.py
└── fixtures/
    └── euro_2016_group_f.csv

dashboard/                         # Existing dashboard
└── app.py                         # Add MC tab (read-only, no logic changes elsewhere)
```

**Files NOT modified in v1:**
- `src/live_edge.py` — unchanged
- `src/paper_trader.py` — unchanged
- `src/fair_odds.py` — unchanged
- `src/normalize.py` — unchanged
- Existing `edge_signals` schema in `src/db.py` — unchanged (only ADD new `mc_fair_odds` table)

**Convention notes (from existing FairLine codebase):**
- Files under `src/`, use `sys.path.insert(0, str(Path(__file__).parent))` prelude
- Config via `.env` + `src/config.py`
- Timestamps: `datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")`
- Logging: `print()` với `===` banner style
- Supabase writes idempotent on unique constraints

---

## 9. Timeline & Risks

**Schedule** (revised 2026-05-20 — added eloratings scrape + 3 bracket adapters + historical replay):

| Phase | Days planned | Days actual | End | Hard stop check |
|---|---|---|---|---|
| 0 — Audit | 3-4 | **1** ✅ | May 17 ✅ | 5 data checks + preflight (Check 5 downgraded WARN per §0.8) |
| 1 — Elo + single-game (+§2.4 eloratings scrape) | 5 + **1** | TBD | ~ May 23 | Brier ≤ 0.20; cross-check self vs eloratings top-20 ≤ 100 Elo (else HARD STOP) |
| ~~2 — Roster~~ | ~~4~~ | ~~0~~ | — | ⛔ SKIPPED v1 (see §0.8); 4 days reallocated |
| 3 — MC harness (+§3.7 bracket adapters + §3.8 baseline) | 7 + **2** | TBD | ~ June 3 | Top-5 favs ±3pp Opta |
| 4 — Comparison + dual CSV output (+§5.3 historical replay) | 4 + **2** | TBD | ~ June 9 | Baseline must lose to both Elo models (else HARD STOP) |
| 5 — Presentation prep | 3 | TBD | ~ June 12 | Report + slides + demo ready |
| Buffer | ~1 (originally 3, minus 5 days for new framework) | TBD | June 11-12 | — |
| **Meeting with Sam + team** | — | — | Pre-WC2026 | Decide v2 integration option |

**Schedule slack analysis (revised 2026-05-20):** Phase 0 saved 2-3 days; Phase 2 skip saved 4 days; 2026-05-20 framework additions (eloratings scrape + 3 bracket adapters + historical replay + dual output) consume ~5 days. Net buffer ~1 day. Tight but feasible. K-N pivot would push delivery past June 12.

**Risk register:**

| Risk | Prob | Impact | Mitigation |
|---|---|---|---|
| Phase 1 Brier > 0.21 | Med | High | 3-day Karlis-Ntzoufras pivot allocated |
| ~~ClubElo coverage <85%~~ | ~~High~~ | ~~Med~~ | ⛔ Resolved by §0.8 Phase 2 skip — no longer relevant |
| Pure match-Elo insufficient for European-heavy teams | Med | Med | Phase 1 Brier validates; if >0.21, K-N pivot. v2 may add roster overlay with better source. |
| R32 cross-seeding bugs | Med | High | Hand-verify 5 sample mappings + unit test |
| Model implicitly overfits market | High | Critical | Train only on match results; LOTO-CV |
| Team name normalization breaks | Med | High | Reuse existing entity_map.json from normalize.py |
| Team rejects integration in v2 | Low | Low | V1 already standalone; can iterate without breaking existing pipeline |

---

## 10. Done Definition (v1)

Stream 3 v1 is "done" when:

1. Phase 0 audit report committed, all checks PASS or documented WARN (Check 5 WARN expected per §0.8) + preflight PASS
2. Phase 1 Brier ≤ 0.20 on validation tournaments for **self-Elo source**
3. **Phase 1 §2.4 (NEW 2026-05-20):** eloratings.net scrape complete; cross-check vs self-Elo top-20 within 100 Elo points
4. ~~Phase 2 either kept or documented drop~~ ⛔ Phase 2 SKIPPED v1 per §0.8 (documented in this spec)
5. Phase 3 top-5 WC2026 champions match Opta ±3pp (run with BOTH Elo sources, report both)
6. **Phase 3 §3.7-3.8 (NEW 2026-05-20):** 3 bracket adapters + baseline 40/40/20 model built and unit-tested
7. **Phase 4 §5.3 (NEW 2026-05-20):** Historical tournament replay run for all 3 models on 12 historical tournaments; `audit_replay_report.csv` committed; baseline must LOSE to both Elo models (else HARD STOP)
8. Phase 4 frozen θ* committed PER SOURCE: `frozen_params_self.json` + `frozen_params_eloratings.json`
9. **Phase 4 §5.4 (REVISED 2026-05-20):** 2 CSVs written per event (`mc_fair_odds_wc2026_self_<date>.csv` + `mc_fair_odds_wc2026_eloratings_<date>.csv`); both have 96 rows; per-market probs sum to 1.0 ± 0.001
10. All unit tests pass (excluding integration tests with live_edge — those are v2)
11. Phase 5 deliverables ready: technical report, slide deck, demo script, dashboard tab (presents BOTH Elo sources side-by-side)
12. Meeting với Sam + team scheduled
13. Existing pipeline (`live_edge.py`, `paper_trader.py`, `edge_signals` schema) UNCHANGED

**v1 explicitly excludes:** Integration with live trading flow. That work is contingent on team approval và scoped in v2 plan.

---

## 11. Spec change entries

| Date | Sections amended | Rationale |
|---|---|---|
| 2026-05-17 | §0.8 (NEW), §3 (SKIPPED), §5.1 step 4.1 (54→27 combos), §8, §10 | ClubElo coverage analysis: 55% mean across WC2026 squads → pre-emptive drop of Phase 2 roster overlay |
| 2026-05-20 | Split mc_simu.md into folder + phase files | Single 1482-line file was hard to read/execute; tách thành mc_simu/ folder with index + phases/ |
| 2026-05-20 | §0.3 (scope), §8 (file layout), §9 (timeline), §10 (done def), phase1 §2.4 (NEW), phase3 §3.7-3.8 (NEW), phase3 §4.2 note (REVISED), phase4 §5.0/§5.3/§5.4 (NEW) | Sam's barebone framework: plug-in per-game models. Added: eloratings.net scraper as 2nd Elo source, baseline 40/40/20, 3 historical bracket adapters, historical tournament replay as primary Elo-source selection metric, dual-source CSV output |
