# Stream 3 — MC Tournament Simulator

Independent structural prior for FairLine: Elo-based team strength + HFA + Poisson goals + MC tournament simulation. Runs standalone — does not touch the live trading engine (Stream 1 + Stream 2 in [root README](../README.md)).

**For design rationale and parameter justification, see [MODEL_SPEC.md](MODEL_SPEC.md).** This file is for running the simulator.

---

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Pre-flight data audit | ✅ Done |
| 1 | Elo engine + `predict_match` | ✅ Done |
| 2 | Roster-Elo overlay | ⛔ Skipped — ClubElo coverage 55% on WC2026 squads |
| 3 | Tournament MC harness + 3-predictor head-to-head | ✅ Done |
| 4 | LOTO-CV tuning | ⏳ Pending |
| 5 | Team presentation | ⏳ Pending |

---

## Requirements

- Python ≥ 3.10 (project pins 3.12.8 via `.python-version`)
- Packages from project root `requirements.txt` — only `numpy`, `scipy`, `pandas`, `requests`, `beautifulsoup4` are needed for MC simulator. The full project also installs Selenium, FastAPI, Streamlit, etc. — those are NOT used by mc_simu.
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

Alternatively run scripts directly without setting `PYTHONPATH`:

```bash
python src/mc_simu/preflight.py
```

---

## Quickstart — reproduce Phase 3 baseline results

All input data is committed to `data/mc_simu/`. You do not need to download or scrape anything to reproduce the baseline run.

```powershell
# 1. Verify data integrity (~5 s)
python -m mc_simu.preflight

# 2. Run both Elo predictors (self + eloratings) across all 11 historical editions (~70 s)
python -m mc_simu.run_phase3_baselines --n 10000 --both

# 3. Add the 40/40/20 baseline predictor (~35 s)
python -m mc_simu.run_phase3_baselines --n 10000 --predictor baseline

# 4. Score replays against actual champions
python -m mc_simu.historical_replay

# 5. Compare WC2026 predictions vs current Polymarket prices
python -m mc_simu.wc2026_vs_polymarket
```

Outputs land in `data/mc_simu/phase3_baselines/`:

- `mc_<edition>_<model>_n10000_seed42.csv` — per-edition champion distributions (36 files: 12 editions × 3 models)
- `replay_report.md` + `replay_scores.csv` — Σ log P(actual champion) per (edition, model)
- `wc2026_vs_polymarket.csv` — model vs PM mid edge table

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
pytest tests/mc_simu/                  # 296 tests + 3 wiring audits, ~30 s
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

src/mc_simu/              ← simulator code (27 modules)
├── elo.py                ← Elo engine (rating updates, MoV, K-factor)
├── single_game.py        ← predict_match — Elo → λ → Poisson grid → W/D/L
├── simulator.py          ← MC tournament harness (vectorized hot path)
├── standings.py          ← FIFA 2026 tiebreaker chain (H2H-first)
├── tournaments/          ← 4 bracket adapters (WC2026, WC 8-group, Euro 24/16)
└── ...                   ← data loaders, scrapers, validators

tests/mc_simu/            ← 17 test files + 3 audit scripts

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

Run the Quickstart commands above to regenerate them.

---

## Integration with live trading engine

**v1 (current):** MC simulator is **standalone**. It does not write to Supabase, does not touch `live_edge.py` or `paper_trader.py`, and its outputs are not used for trade signals.

**v2 (post-WC2026 review):** Optional integration into `live_edge.py` as a third edge source alongside Stream 1 (sportsbook devigged) and Stream 2 (Polymarket/Kalshi). Decision deferred to team review after WC2026 group stage data accumulates.

---

*Maintainer: Duy Anh (FairLine).*
