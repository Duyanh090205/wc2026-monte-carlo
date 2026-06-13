# Project conventions — mc-simu (MC Tournament Simulator)

Standalone repo for the Monte Carlo tournament simulator (formerly Stream 3 of
`IterLight-Lab/Prediction-Market-Project`, split out 2026-06; full history preserved).
Produces an independent structural prior for WC2026 outright markets and a daily
model-vs-market tracking log. Does NOT contain or deploy the live trading engine —
that stays in the original repo.

## Where to look

- Design rationale (predict_match motivation) → [`mc_simu/MODEL_SPEC.md`](mc_simu/MODEL_SPEC.md)
- How-to-run → [`mc_simu/README.md`](mc_simu/README.md)
- Public data API (Supabase REST, read-only) → [`mc_simu/API_MC_SIMU.md`](mc_simu/API_MC_SIMU.md)
- Implementation plan + phase log → [`mc_simu/mc_simu.md`](mc_simu/mc_simu.md)
- Validation findings → [`mc_simu/audits/`](mc_simu/audits/)

## Git workflow

- `main` is ours and deploys nothing; direct commits allowed, feature branches
  (`snguyen_<feature>`) for larger work.
- Force-push only with `--force-with-lease`. Pre-commit hooks: fix the issue, never `--no-verify`.

## Architecture rules

- **Plug-in predictor protocol:** `predict_match(team_a, team_b, ctx) → MatchPrediction`.
  Tournament harness + bracket adapters are fixed; per-game probability model is
  swappable. See `mc_simu/MODEL_SPEC.md` §4.1.
- **Never tune against market prices.** LOTO-CV uses historical scoreboard W/D/L
  outcomes only. Market (PM/Kalshi/sportsbook) is a post-tune sanity check +
  distribution-closeness metric, never a training signal.
- **Must remain runnable offline.** Wikipedia HTML snapshots committed under
  `data/mc_simu/cache/`. No live-network dependencies in the predict path
  (the two deliberate network calls — market fetch in `run_daily.py`, results
  fetch in `fetch_played.py` — both degrade gracefully).
- **Production model (locked):** ELO (eloratings as-of) + MV blend alpha=0.5 +
  star X=15, D=1400, diag=0.20 (LOTO-CV validated). Static during the tournament —
  daily rerun re-conditions on the bracket, never re-fits.

## Output discipline

- **Inputs tracked:** `results.csv`, `matches_*`, `elo_history*`, `cache/`, `wc2026_*`,
  `historical_squad_mv.csv`, `r32_seeding_table.json`, `tournament_meta.json`.
- **Results gitignored:** `phase3_baselines/`, `mc_wc2026_*.csv`, `loto_cv_*.csv`,
  `daily_log.csv`, `mc_simu/figures/`. Ship code + inputs, regenerate locally.
- Internal audit drafts (`audit_report.md`, `preliminary_report.md`, `archive/`) stay local.

## Coding style

- Default to **no comments**; only when WHY is non-obvious.
- No emojis in code or commit messages.
- Prefer editing existing files over creating new ones.
- Modules end with an `__all__` export list (see `single_game.py`).
- Tests in `tests/mc_simu/`; wiring audits (`audit_phase*.py`) separate from unit tests.

## Run

```
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/mc_simu -q
PYTHONPATH=src python -m mc_simu.run_daily            # daily tracker snapshot
PYTHONPATH=src python -m mc_simu.replay_wc_conditioned --year 2022 --plot
```
