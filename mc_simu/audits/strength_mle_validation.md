# mle_strength — Phase 4 LOTO validation + diversity audit

Date: 2026-06-11. Script: [`validate_strength_mle.py`](../../src/mc_simu/validate_strength_mle.py). Protocol: 12 tournaments (WC 2002–2022, Euro 2004–2024), 630 matches; per fold, strengths fit with as_of = first match date (training strictly precedes, leak-free); metric = mean-Brier (sum/3, same convention as `validate_phase1`). Baseline = eloratings as-of + production predictor at locked params (α=0.27, β=0.09, diag=0.20); MV/star excluded from baseline (validated Brier-neutral in [mv_historical_validation](mv_historical_validation.md)). Zero eloratings fallback lookups (all 630 matches' teams resolved).

## Gate results

| Gate | Result | Verdict |
|---|---|---|
| G1 accuracy | mle **0.1958** vs baseline **0.1951**; paired diff +0.0007 (SE 0.0016, z +0.46) | **PASS** (gate ≤0.2000; statistically indistinguishable) |
| G2 rating diversity | Spearman per fold 0.885–0.973, **mean 0.925**; 2/12 folds >0.95 (WC2022 0.973, Euro2004 0.959) | High overlap, flagged but expected (same underlying data) |
| G3 error decorrelation | Pearson per-match Brier **0.933**; double-fault 0.405 vs 0.192 under independence | Errors largely shared; decorrelation real but small |
| G4 ensemble lift | pool **0.1946** < min(0.1951, 0.1958) → STRONG PASS on point estimate; **but** pool−base z = −0.64, bootstrap P(pool beats both) = **0.686**, pool < baseline in only 5/12 folds | **POSITIVE BUT INCONCLUSIVE** |
| G5 class error shares | baseline home/draw/away 0.352/0.312/0.336; mle 0.349/0.314/0.337 | Identical shape — score-matrix weakness shared by construction (expected: same Poisson+diag grid) |

Per-tournament table in script output; mle beats baseline outright in 4/12 folds (WC2002 strongly: 0.1990 vs 0.2154), loses in 8.

## G7 — WC2026 outright, production vs mle_strength (20k sims each, same bundle/seed)

| Team | Production | mle_strength | Δ |
|---|---|---|---|
| Spain | 19.5% | 15.8% | −3.7pp |
| France | 15.2% | 7.4% | **−7.9pp** |
| England | 11.7% | 7.0% | −4.6pp |
| Argentina | 10.5% | 9.6% | −0.9pp |
| Brazil | 6.7% | 9.3% | +2.7pp |
| Belgium | 2.1% | 5.6% | +3.5pp |
| Colombia | 1.2% | 4.5% | +3.4pp |

Outright Spearman 0.946.

Findings:
1. **Spain divergence is rating-source-shared.** mle_strength — with zero MV/star/eloratings input — still ranks Spain #1 (15.8%). Per the pre-registered G7 rule: model-vs-market deviation on Spain must NOT be treated as two independent confirmations; it originates in the results history itself, not in the MV/star components.
2. **France is the headline disagreement** (15.2% vs 7.4%): eloratings+MV rates France far above what 3y-decayed weighted results support. This is exactly the class of signal the second model exists to surface.
3. **mle's top is flatter overall** (favorites down, mid-tier up). Directionally consistent with the only robust effect from [before_world_cup_audit](before_world_cup_audit.md) — the general favorite-longshot bias (top favorites over by ~4.4pp historically). Not by construction; emergent from the decay+importance weighting.

## Verdict vs the commit gate ("information beyond eloratings?")

Honest summary: **some, not conclusive on 630 matches.**
- For: ensemble point-estimate beats both singles; error correlation 0.933 is meaningfully below the ~0.99 a jittered clone would show; G7 isolates the MV/star contribution and surfaces France-type disagreements; flatter top matches the historically validated favorite-longshot correction.
- Against: ensemble lift is within noise (z −0.64, 5/12 folds); double-fault rate is high; class-level error shape identical.

The accuracy case alone does not clear the bar; the diagnostics case (Spain attribution, France divergence, agreement-confidence signal for daily tracking) is where the value sits. Decision on commit/deploy: user + Sam.

## Sensitivity sweeps (§6.2, §6.4, §6.5) — run 2026-06-11, full table in `logs/sweep_results.csv`

| Config | LOTO Brier (mle) | Δ vs base (fold-level, ±SE) | G2 Spearman |
|---|---|---|---|
| **BASE: H=3y, window=1995, friendlies in, diag ON** | 0.19584 | — | 0.925 |
| H=2y | 0.19522 | −0.00065 ± 0.00047 (1.4 SE) | 0.938 |
| H=2.5y | 0.19556 | −0.00029 ± 0.00020 | 0.930 |
| H=3.5y | 0.19606 | +0.00022 ± 0.00016 | 0.920 |
| H=4y | 0.19622 | +0.00039 ± 0.00028 | 0.916 |
| window=1990 | 0.19583 | +0.00003 ± 0.00013 | 0.926 |
| window=2000 | 0.19721 | +0.00119 ± 0.00101 | 0.923 |
| friendlies excluded | 0.19790 | +0.00193 ± 0.00095 (2.0 SE) | 0.904 |
| diag OFF | 0.19492 | −0.00093 ± 0.00070 (1.3 SE) | 0.925 |

Readings:
1. **H:** smooth monotone preference for shorter H; H=2y improvement is 1.4 SE — *technically* over the pre-registered 1-SE deviation threshold, but (a) magnitude is 0.0007 Brier on 630 matches, (b) 4 H-alternatives were tested (winner's-curse discount applies), (c) shorter H *increases* correlation with eloratings (G2 0.938 vs 0.925) — worse for the diversity purpose. **Recommendation: keep H=3y** (literature anchor); flagged for owner decision.
2. **Window:** 1990 indistinguishable, 2000 mildly worse — 1995 confirmed, choice second-order as predicted by weight-share analysis.
3. **Friendlies:** excluding them degrades Brier by 2 SE — keep them in at weight 1 (matches Ley et al. framework).
4. **Diag OFF beats ON by 1.3 SE** (0.19492, also beats the eloratings baseline 0.19511) — pure-Poisson mle is marginally the most accurate config tested. Kept **ON in v1** per the design adjudication (score-shape mirror = clean rating-only attribution axis); OFF goes into the v2 ablation arm. Flagged for owner decision.
5. G4 pool < both singles at every config (point estimate) — ensemble lift direction is robust to all hyperparameter choices.

## France robustness check (2026 fit, H = 2y → 5y)

Spain ranks #1 and France ranks 4–5 at every H (strength 1.52–1.62; gap to #1 −0.20 → −0.15). The production-vs-mle France divergence (15.2% vs 7.4% outright) is structural — weighted recent results genuinely place France behind the Spain/Argentina/Brazil/Portugal cluster — not an artifact of the decay hyperparameter.

## Vote diagnostics — does mle deserve a seat in the decision layer? (2026-06-11, reproduce via `--diagnostics`)

**Test 1 — disagreement is NOT a production-error alarm.** Bucketing 630 matches by model disagreement (total variation distance): production's Brier in the top-disagreement quartile is 0.1845 — *better* than its own average (0.1951, p=0.154 in the opposite direction of the alarm hypothesis). Match-mix patch: Q4 is only marginally tilted toward big-gap matches (mean fav-gap 0.317 vs 0.305), and stratifying by gap, production-in-Q4 beats production-elsewhere in **every stratum**. Within Q4 head-to-head, production also edges mle (0.1845 vs 0.1872, ns). Verdict: a hard "Class C no-trade on disagreement" rule has no empirical basis — disagreement should *discount* size, not veto.

**Test 2 — mle adds information.** Log-pool weight sweep w (production weight) 0→1: smooth U-curve, minimum at w≈0.6, plateau 0.4–0.8, both endpoints worse than the interior. Per preregistration discipline the deployed pool stays **50/50** (inside the plateau, chosen before seeing results); the curve's shape — not its argmin — is the finding.

**Test 3 — complementary niches (diagnostic only, NOT a sizing knob).** mle better on close matches (0.2117 vs 0.2156), production better on big-gap matches (0.1670 vs 0.1702). Logged as a confidence flag; context-dependent weighting would be a fitted parameter on n=630 and is deliberately not used in v1.

**France decomposition.** Pure eloratings Elo (MV/star stripped): France 11.6% — midway between production 15.4% and mle 7.4%. The gap splits ~half from the MV/star layer (never accuracy-validated → leans mle) and ~half from estimation mechanism (unadjudicated). Leave-one-out influence on France's last 12 matches: max strength swing ±0.014 (vs −0.17 gap to #1) — mle's France rating is broad-based, not noise-driven.

**Spain correction (supersedes the G7 phrasing above):** pure Elo gives Spain **21.6%** > production 19.1% > PM 16.4% > mle 15.8%. The Spain over-rating originates in the eloratings source itself; MV/star pulls *toward* market (consistent with MV's known market-fit character — not a rehabilitation of MV).

**Design tension flagged for owner decision:** the 50/50 pool puts France at ~10–11% vs PM 16.1% — the *largest* pool-vs-market divergence in the book, larger than Portugal. A discrete no-trade class for disagreement would zero the book's biggest consensus-prior edge, while Test 1 says disagreement does not predict production error and Test 2's pool lift comes precisely from disagreement matches. Continuous sizing (`Kelly(p_pool vs market) × g(agreement)`, g discounting but never zero) is the design consistent with all three tests. See [`decision_layer_design.md`](../decision_layer_design.md).

## Known limitations (restated from plan §8 + new)

1. Diversity is partial — same match-results data; only the estimation mechanism differs. G2–G4 quantify exactly this.
2. Score shape shared with production by construction (deliberate — isolates rating attribution; distribution axis deferred with quantified triggers per the design adjudication).
3. 9×9 grid truncation: for the most extreme WC2026 mismatch (λ≈4.9), ~6% of the favorite's goal mass folds back at renormalization — shared with production; W/D/L barely affected, group-GD tails slightly compressed.
4. Basque Country (non-FIFA, friendly-only sample) ranks #11 in the 2026 fit — small-sample artifact, no WC2026 impact.
5. ~~Sensitivity sweeps pending~~ — completed 2026-06-11, see section above. Two flagged-for-owner items: H=2y (1.4 SE better, less diverse) and diag OFF (1.3 SE better, breaks score-shape mirror).
