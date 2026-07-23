# MC Tournament Simulator — how to run

Elo-based team strength + HFA + Poisson goals + MC tournament simulation, plus a
daily model-vs-market tracker for WC2026. Standalone repo (see [root README](../README.md));
the live trading engine lives in `IterLight-Lab/Prediction-Market-Project`.

**For design rationale and parameter justification, see [MODEL_SPEC.md](MODEL_SPEC.md).** This file is for running the simulator.

---

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Pre-flight data audit | ✅ Done |
| 1 | Elo engine + `predict_match` | ✅ Done |
| 2 | Roster-Elo overlay | ⛔ Skipped — ClubElo coverage 55% on WC2026 squads |
| 3 | Tournament MC harness + 3-predictor head-to-head | ✅ Done |
| 4 | Model selection (ELO+MV+star) + LOTO-CV lock (D=1400, diag=0.20) | ✅ Done |
| 5 | Conditioning harness + WC2018/22 backtest + daily tracker | ✅ Done |

---

## Requirements

- Python ≥ 3.10 (project pins 3.12.8 via `.python-version`)
- Packages from root `requirements.txt` (core: `numpy`, `scipy`, `pandas`, `requests`,
  `beautifulsoup4`; `matplotlib` for figures; `streamlit`+`plotly` only for the dashboard).
- ~35 MB free disk for committed input data; ~5 MB for generated outputs.

```powershell
pip install -r requirements.txt
```

PowerShell PYTHONPATH (Windows; needed for `python -m mc_simu.<script>` invocations):

```powershell
$env:PYTHONPATH = "src"
```

On Linux/macOS:

```bash
export PYTHONPATH=src
```

---

## Quickstart — reproduce Phase 3 baseline results

All input data is committed to `data/mc_simu/`. You do not need to download or scrape anything to reproduce the baseline run.

```powershell
# 1. Run both Elo predictors (self + eloratings) across all 11 historical editions (~70 s)
python -m mc_simu.run_phase3_baselines --n 10000 --both

# 2. Add the 40/40/20 baseline predictor (~35 s)
python -m mc_simu.run_phase3_baselines --n 10000 --predictor baseline

# 3. Score replays against actual champions
python -m mc_simu.historical_replay

# 4. Compare WC2026 predictions vs live markets (Polymarket + Kalshi + sportsbooks)
python -m mc_simu.wc2026_vs_multi
```

Outputs land in `data/mc_simu/phase3_baselines/`:

- `mc_<edition>_<model>_n10000_seed42.csv` — per-edition champion distributions (36 files: 12 editions × 3 models)
- `replay_report.md` + `replay_scores.csv` — Σ log P(actual champion) per (edition, model)
- `wc2026_vs_multi.csv` — model vs market edge table (abs pp + relative %)

These output files are **not committed** — they regenerate each run.

---

## Run full WC2026 simulation (Phase 4 target — 100k iterations)

```powershell
# Default: 100,000 iterations, seed 42 (~30-60 s)
python -m mc_simu.run_mc_simu

# Or specify
python -m mc_simu.run_mc_simu --n 100000 --seed 42
```

Output: `data/mc_simu/mc_wc2026_v1_<N>_seed<S>.csv` (champion + group winner probabilities, 96 rows).

---

## Daily tracker + backtest (production model: ELO+MV+star)

```powershell
# Daily conditional rerun vs live market — appends a snapshot row per team to
# data/mc_simu/daily_log.csv (and upserts to Supabase when SUPABASE_URL +
# SUPABASE_SERVICE_KEY are set). Locks results from data/mc_simu/wc2026_played.csv.
python -m mc_simu.run_daily --n 1000000   # CI uses 1M; default 50k for quick local runs

# Round-by-round backtest of the conditioning harness on a past WC,
# with archived oddschecker market overlay (Wayback)
python -m mc_simu.replay_wc_conditioned --year 2022 --plot

# Backfill the per-day series the dashboard's Group winner / Knockout rounds
# tabs read (model re-simulated on git as-of states vs Polymarket history)
python -m mc_simu.backfill_group_winner   # -> data/mc_simu/group_winner_log.csv
python -m mc_simu.backfill_reach          # -> data/mc_simu/reach_log.csv

# Dashboard (reads Supabase; falls back to the committed closed-record archive
# under data/mc_simu/closed_record/ when unreachable — no credentials needed)
streamlit run dashboard/mc_tracker.py
```

Automation: `.github/workflows/daily.yml` runs at 08:30 UTC daily: first
`fetch_played` pulls finished WC2026 results from football-data.org (secret
`FOOTBALL_DATA_TOKEN`; skipped when absent) and commits the updated
`data/mc_simu/wc2026_played.csv` back to main, then `run_daily` re-conditions
on it and upserts Supabase (secrets `SUPABASE_URL` + `SUPABASE_SERVICE_KEY`);
table schema in `deploy/supabase_schema.sql`. Manual fallback: edit
`wc2026_played.csv` by hand and push — any fetch failure (API down, unmapped
team name, unplaceable KO pair) degrades to a warning and leaves the CSV alone.

Failure hedges: a 13:00 UTC retry cron re-runs the day (no-op when the morning
snapshot already landed and no new results arrived); a FairLine market outage
skips that day's daily_log honestly (no fabricated prices) while match
predictions still refresh — a later `workflow_dispatch` backfills the day; an
mle_strength failure blanks only the mle/pool columns, never the production
snapshot. CI pins numpy/scipy/pandas/requests to tested minor versions.

```powershell
# Auto-fetch played results locally (same script CI runs before run_daily)
$env:FOOTBALL_DATA_TOKEN = "<token>"; python -m mc_simu.fetch_played --dry-run
```

Public read-only data API (Supabase REST over the daily log, for anyone who
wants to build on the data): see [`API_MC_SIMU.md`](API_MC_SIMU.md).

---

## Rebuild from scratch (optional)

The committed `data/mc_simu/elo_history.csv` and `elo_history_eloratings.csv` are pre-built. Rebuilding takes 5-15 minutes and is only needed if you change Elo parameters or want fresh data.

```powershell
# Re-download Kaggle match history mirror (~5 s, fetches ~50k matches)
python -m mc_simu.download_match_history

# Rebuild self-Elo from match history (~5-10 min, processes 49k matches chronologically)
python -m mc_simu.build_history

# Re-scrape eloratings.net per-match time series (~5 min — may rate-limit)
python -m mc_simu.eloratings_snapshot

# Rebuild WC2026 R32 cross-seeding table (495 combinations from Wikipedia)
python -m mc_simu.build_r32_seeding

# Rebuild WC2026 group fixtures
python -m mc_simu.build_wc2026_fixtures
```

Wikipedia HTML snapshots used for offline scraping are committed under `data/mc_simu/cache/`, so rebuilds are deterministic even if Wikipedia changes its page format.

---

## Tests

```powershell
pytest tests/mc_simu/                  # 289 tests + wiring audits, ~30 s
```

Wiring audits (`tests/mc_simu/audit_phase*_wiring.py`) catch integration-layer bugs that unit tests miss — e.g., scalar vs vectorized samplers producing different outputs given the same RNG state, or knockout symmetry violations.

---

## Repository layout

```
mc_simu/
├── README.md             ← you are here — how to run
├── MODEL_SPEC.md         ← design rationale + parameter justification
├── mc_simu.md            ← internal implementation plan (full phase log)
└── phases/               ← per-phase design notes

src/mc_simu/              ← simulator code (44 modules)
├── elo.py                ← Elo engine (rating updates, MoV, K-factor)
├── single_game.py        ← predict_match — Elo → λ → Poisson grid → W/D/L
├── simulator.py          ← MC tournament harness (vectorized hot path)
├── standings.py          ← FIFA 2026 tiebreaker chain (H2H-first)
├── mv_blend.py / star_presence.py ← market-value blend + star bonus (production model)
├── run_daily.py          ← daily conditional rerun vs live market (tracker)
├── replay_wc_conditioned.py ← WC2018/22 round-by-round backtest
├── tournaments/          ← 4 bracket adapters (WC2026, WC 8-group, Euro 24/16)
└── ...                   ← data loaders, scrapers, validators

dashboard/mc_tracker.py   ← Streamlit dashboard (Supabase or local CSV)

tests/mc_simu/            ← 16 test files + 4 audit scripts

data/mc_simu/             ← committed inputs (no results)
├── matches_1998_2026.csv ← filtered match history (4.5 MB, post-1998)
├── results.csv           ← raw Kaggle mirror (3.7 MB, 49k matches)
├── elo_history*.csv      ← pre-built Elo time series (12.7 MB combined)
├── wc2026_*.{csv,json}   ← tournament structure
├── r32_seeding_table.json← FIFA Annex C 495 R32 combinations
└── cache/                ← Wikipedia HTML snapshots for offline scraping
```

---

## What is NOT included

The following are deliberately excluded from the repo per Sam's reproducibility request — they are **outputs**, not **inputs**:

- `data/mc_simu/phase3_baselines/` — regenerated by `run_phase3_baselines.py`
- `data/mc_simu/mc_wc2026_*.csv` — regenerated by `run_mc_simu.py`
- `data/mc_simu/loto_cv_*.csv` — calibration sweep results (Phase 1 + Phase 4)
- `data/mc_simu/replay_*.csv` — historical replay scores
- `data/mc_simu/daily_log.csv` — daily tracker log (canonical copy lives in Supabase)
- `mc_simu/figures/` — generated figures

Run the Quickstart commands above to regenerate them.

---

## Integration with live trading engine

**v1 (current):** standalone repo. The only network touchpoints are the market
fetch in `run_daily.py` (FairLine `/prices` endpoint, read-only) and the tracker's
own Supabase table. Nothing here writes to the trading engine
(`IterLight-Lab/Prediction-Market-Project`) or feeds trade signals.

**v2 (post-WC2026 review):** optional integration into the engine's `live_edge.py`
as a third edge source. Decision deferred to team review after WC2026 group-stage
data accumulates in `daily_log`.

---

*Maintainer: Duy Anh (FairLine).*
