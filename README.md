# mc-simu — MC Tournament Simulator

Monte Carlo simulator for international football tournaments (WC2026 focus).
Produces an independent structural prior for outright (champion) markets and a
daily model-vs-market tracking log. Goal: track the market with a stable,
understood bias — divergence from that bias is the signal, not outright prediction.

Split out of `IterLight-Lab/Prediction-Market-Project` (Stream 3) with full git
history; the live trading engine stays there. This repo deploys nothing to the
trading stack.

**Live dashboard:** <https://mc-wc2026.streamlit.app> — the full WC2026 campaign
(2026-06-10 to 2026-07-20, closed record): daily model-vs-market edge, JSD/L1
tracking, per-round scorecard, bracket. Model was locked pre-tournament; every
line on it is out-of-sample.

## Quick start

```
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/mc_simu -q          # 296 tests
PYTHONPATH=src python -m mc_simu.run_mc_simu              # full WC2026 sim
PYTHONPATH=src python -m mc_simu.run_daily                # daily conditional rerun vs live market
PYTHONPATH=src python -m mc_simu.replay_wc_conditioned --year 2022 --plot   # backtest figure
```

## Layout

| Path | What |
|---|---|
| `src/mc_simu/` | simulator, predictors, validation + tooling scripts |
| `mc_simu/` | docs: `MODEL_SPEC.md`, `README.md` (how-to-run), `mc_simu.md` (phase log), `audits/` |
| `tests/mc_simu/` | unit tests + wiring audits |
| `data/mc_simu/` | tracked inputs (match results, Elo histories, squad MV, offline caches) |

## Production model (locked)

ELO (eloratings.net reconstruction, as-of) + squad market-value blend (alpha=0.5)
+ star presence (X=15), Poisson grid D=1400, diagonal inflation 0.20
(LOTO-CV validated over 11 tournaments). Static during a tournament — the daily
rerun re-conditions on locked results only, never re-fits. See
[`mc_simu/MODEL_SPEC.md`](mc_simu/MODEL_SPEC.md) and [`mc_simu/audits/`](mc_simu/audits/).
