# MC Tournament Simulator — Model Specification

**Purpose:** Document the structural model behind Stream 3, with explicit attribution of which components are practitioner-standard vs derived heuristics vs empirically recalibrated.

**Audience:** Sam, Khang, team reviewers, future maintainers.

**Status:** Phases 0, 1, 3 complete. Phase 4 (LOTO-CV tuning) completed 2026-06-11 — full G1-G7
result tables retained. Phase 5 (presentation) pending.

**Last updated:** 2026-05-21

---

## 1. What problem this solves

FairLine currently derives `fair_prob` from two market-consensus sources:

- **Stream 1** — Sportsbook devigged (Shin + Power applied to 26+ books)
- **Stream 2** — Polymarket + Kalshi mid prices

Both are market beliefs. When markets are jointly wrong (rare but documented — e.g., Saudi Arabia 1-2 Argentina at WC2022 was priced ~3% pre-match), we cannot detect the error because we have no view independent of the market.

Stream 3 introduces a **structural prior independent of market consensus**:

```
Stream 3 = Elo-based team strength + HFA + Poisson goals + MC tournament simulation
```

When Stream 3 diverges from Stream 1+2, two possibilities exist:

1. **Model is wrong** → tune via LOTO-CV on historical scoreboard outcomes
2. **Market is wrong** → potential edge

The calibration loop, not narrative, distinguishes between them. Crucially, we **never** tune Stream 3 against market prices — that would create an echo chamber where Stream 3 becomes Stream 1' with extra steps.

---

## 2. Model components — origin and attribution

Every component is labelled as one of:

- **STANDARD** — direct copy from practitioner literature with citation
- **DERIVED** — heuristic constructed by us (Duy Anh / Claude), not from literature
- **EMPIRICAL** — fit or recalibrated from data, no assumed functional form OR single-parameter tune of a published form

The most important section is §2.3 (Elo → expected goals). Read that one carefully — it is where the model is most likely to be challenged.

### 2.1 Elo rating engine — STANDARD

**Source:** eloratings.net official specification (Kirill Bulygin, formerly Bob Runyan since 1997).

**Components:**

```python
# Expected win formula (logistic on Elo difference)
P_home = 1 / (1 + 10 ** (-(R_home + HFA - R_away) / 400))

# K-factor table
K = {
    "world_cup_final": 60,    "continental_final": 50,
    "qualifier": 40,           "nations_league": 40,
    "friendly": 20,
}

# Margin of Victory multiplier
mov(gd) = 1.0 if gd<=1 else 1.5 if gd==2 else 1.75 if gd==3 else 1.75 + (gd-3)/8

# HFA in Elo space (for rating UPDATES only — separate from log-goal HFA used in predict)
HFA_elo = +100 to home team rating
```

**Why this is defensible:**

- eloratings.net 5-bucket K scheme cited in Wikipedia World Football Elo Ratings article.
- Lasek et al. 2013 comparative study ranked eloratings.net **#1 of 8** international rating systems on binomial deviance (1.2634 vs FIFA's 1.3681).
- Used as input feature by 538 SPI, Groll et al., Gilch, Egidi — practitioner consensus.
- Standard Elo formula (Arpad Elo 1960s, FIDE chess) — 60+ years validated.

### 2.2 HFA specification — STANDARD with unit-conversion caveat

**What this models:**

```
log(λ_team) = base_strength_from_Elo
            + α × I(team plays in own country)
            + β × I(team in host confederation AND not host)
```

Two effects: playing in your own country (α), and playing in your continental confederation when one of your neighbours is hosting (β). These add a log-goal bonus to the team's expected goals.

**Parameters (CORRECTED — see audit trail below):**

| Param | Value | Source |
|---|---|---|
| α | **+0.27 log-goals** | Average of 538 SPI (0.260 log) and Bilalić et al. 2021 (0.282 log) after unit conversion |
| β | **+0.09 log-goals** | 538 SPI's "one-third of host bonus" after unit conversion |

**Audit trail — why these differ from 538's published numbers:**

538 SPI publishes the host bonus as **"+0.40 expected goals per match"**. Phase 1 initial implementation treated this 0.40 as a log-goals coefficient, which would make it ~50% too large. The Phase 1 deep audit caught this:

- 538 reports HFA in **additive goals/match** (the difference in expected goals before vs after applying HFA).
- Our model uses HFA as a **log-goal coefficient** (multiplicative on λ, additive on log λ).
- Conversion at baseline λ ≈ 1.35 (typical international match average): `ln((1.35 + 0.40) / 1.35) = 0.260 log-goals`.

After unit conversion:
- 538 SPI WC2018/2022: 0.260 log
- Bilalić et al. 2021 (n>4000 nat'l team matches from UEFA Nations League): 0.282 log
- **Our α = 0.27** = mean of two sources, both within tuning grid for Phase 4

For β: 538 SPI publishes "one-third of host bonus" = +0.13 additive goals/match → 0.09 log-goals.

**COVID attendance scaling** (refined Phase 1 audit, post-Wunderlich/Bilalić 2023):

| Capacity | Multiplier on (α, β) |
|---|---|
| < 25% (empty/near-empty) | 0.55× |
| 25–50% | 0.65× |
| 50–75% | 0.85× |
| ≥ 75% | 1.00× (no scaling) |

Original 3-bin {0.67, 0.80, 1.00} was too gentle vs measured COVID-era reductions (~41% HFA loss in empty stadiums per Bilalić 2021).

**Co-host handling (WC2026):** USA/Mexico/Canada get α only when playing in their own country. CONCACAF teams (non-host) get β. No double-counting when USA plays in Mexico (α=0 because USA is not in own country; β=0 because USA is itself a host).

**Defensibility:** Strong on α magnitude (multi-source agreement after unit fix). Medium on β (single-source 538). Co-host rule is a consensus convention — no published practitioner has explicitly tested it.

### 2.3 Elo → expected goals mapping — EMPIRICAL (single-parameter recalibration)

**This is the most important section.** It is where the model has been corrected twice and is most likely to be challenged in review.

**Current implementation:**

```python
λ_team = (league_avg_goals / 2) × 10 ** ((elo_team - elo_opp) / D)
                                                                 ^^^
                                                       D = 1400 (not 400)
```

**What is STANDARD:** The functional form. 538 SPI uses this exact shape. Karlis-Ntzoufras 2003 Bayesian regression with Elo as linear predictor collapses to the same shape. The base-10 exponent matches the logistic Elo formula, keeping the engine internally consistent.

**What is EMPIRICAL:** The denominator `D`. We recalibrated it from 538's published default through two iterations:

| Iteration | D value | Source | Why we rejected it |
|---|---|---|---|
| v0 | 400 | 538's published formula (literal transcription) | Mathematical error — 400 calibrates **win odds ratio** (2-way), not **goal ratio**. At ΔElo=400 → predicted goal ratio = 10× → predict_match returns (1, 0, 0) for moderate gaps. Wrong unit. |
| v1 | 800 | "+400 Elo → 91% win" anchor from eloratings.net | 91% is **2-way binary** win prob. In **3-way W/D/L** under Poisson convolution, draws absorb mass. Real-data P_home(ΔElo=400) ≈ 0.74–0.78. v1 still over-calibrated. |
| **v2** | **1400** | Brier-min sweep on validation set (230 matches across 4 historical tournaments) | Selected. Global minimum at D=1400–1500, diag=0.15–0.30. |

**Why this is defensible vs the alternative (isotonic regression on Elo_diff bins):**

A non-parametric mapping (binned isotonic regression) was the earlier Phase 1 proposal. We chose single-parameter D-recalibration instead because:

1. **Constraint counts.** Isotonic regression fits the entire mapping curve — many degrees of freedom, higher overfit risk on n~10k matches with imbalanced Elo-diff coverage.
2. **D is one scalar.** Fits one parameter to data while preserving the practitioner functional form. **More constrained, more defensible**, easier to explain in review.
3. **Tunable per Phase 4 protocol.** D is treated as a structural parameter alongside α, β, diagonal — Phase 4 LOTO-CV grid covers D ∈ {1200, 1400, 1600}. Not a free per-team adjustment.

**What this loses:** Cannot capture non-linear Elo-diff effects (e.g., is +600 ΔElo qualitatively different from +400?). If Phase 4 LOTO-CV shows residual systematic bias in extreme Elo-diff bins, fallback is Karlis-Ntzoufras bivariate Poisson via the footBayes R package.

**Honest citation caveat:** Neither 538 SPI nor eloratings.net publishes their exact Elo→λ mapping. Our D=1400 is **NOT** claimed to match 538's closed-source internal value. It is our own Brier-optimal fit on held-out validation data.

### 2.4 Poisson goal grid with diagonal inflation — STANDARD with international football calibration

**Source:** 538 SPI (WC 2022 methodology page); inflation mechanism inspired by Karlis & Ntzoufras 2003.

**Specification:**

- Two independent Poisson distributions (one per team) with λ from §2.3.
- Compute `P(home_i, away_j)` over 9×9 grid (0 to 8 goals each).
- Multiply diagonal cells `(i, i)` by `(1 + diagonal_inflation)`.
- Renormalize so total mass = 1.

**Inflation value:** `diagonal_inflation = 0.20` (not 538's 0.09).

**Why 0.20 and not 0.09:**

538's 0.09 is calibrated for **club football** (higher league averages, lower draw rate per match). International football has:

- Lower average goals per match (~2.5–2.8 vs 2.8–3.2 for top club leagues)
- Higher draw rate per match (lower-variance scorelines in qualifiers/friendlies)
- More cautious tactics in must-not-lose tournament group stages

Phase 1 Brier-min sweep on validation set found global minimum at `diag ∈ [0.15, 0.30]`. We selected 0.20.

**Underflow guard:** At very high λ (ΔElo > ~6000 under D=1400), all P(0..max_goals) values fall below float64 precision (~1e-300) and the grid sums to literal 0. The implementation returns a degenerate grid placing all mass on the corner consistent with the λ ordering — preserves W/D/L direction at pathological scorelines.

**Limitation:** This is a **simplified post-hoc diagonal boost**, NOT the full bivariate Poisson model with explicit correlation parameter λ_3 from Karlis-Ntzoufras 2003. If Phase 4 shows residual draw mis-calibration, fallback is full Karlis-Ntzoufras via footBayes (rpy2 wrapper) or PyStan port.

### 2.5 FIFA tiebreaker chain — STANDARD (2026 H2H-first chain)

**Source:** FIFA 2026 Competition Regulations Article 13.

**Within-group ordering (CORRECTED per Phase 3 decision #22):**

1. Points (3W/1D/0L) — overall
2. H2H Points — only matches between tied teams
3. H2H Goal Difference
4. H2H Goals Scored
5. (Recurse: if subset still tied after 2–4, re-apply 2–4 within smaller team set)
6. Overall Goal Difference
7. Overall Goals Scored
8. Team Conduct (fair-play score)
9. FIFA World Ranking (we pass current Elo as proxy)

**Important:** FIFA 2026 puts H2H **before** overall GD/GF — same ordering as UEFA adopted post-2018. Earlier spec listed GD/GF before H2H, which is the **pre-2018 FIFA chain** and would mis-rank groups with 2- or 3-way ties.

**Cross-group best-thirds:** Same chain but no H2H (different groups never played each other).

**Verification:**

- Unit test reproduces Euro 2016 Group F where Portugal advanced as 3rd-placed team and went on to win the tournament.
- Synthetic H2H-first discriminator test guards against accidental regression to pre-2018 GD-first chain.
- 3-way tie with partial H2H separation tested.
- All-4-tied case falls through to FIFA ranking (Elo proxy).

### 2.6 R32 cross-seeding for WC2026 — STANDARD

**Source:** FIFA official bracket PDF (post-draw December 5, 2025), reproduced on Wikipedia "2026 FIFA World Cup knockout stage".

**Specification:**

- 12 groups × 4 teams = 48 teams.
- Top 2 per group + 8 best thirds → 32 teams advance to R32.
- Cross-seeding is combinatorial: `C(12, 8) = 495` possible "which 8 groups contribute a third" combinations.
- FIFA published a fixed mapping for all 495 combinations. We parse this from Wikipedia and store in `data/mc_simu/r32_seeding_table.json`.
- Hand-verified 5 sample mappings against official bracket PDF.

**Defensibility:** Strong. Direct from FIFA. Implementation pitfall, not specification ambiguity.

### 2.7 Penalty shootout simplification — DERIVED (acknowledged)

**Specification:** Knockout matches with Draw at 90' → 50/50 advance probability.

```
P(home advance) = P(home win 90') + 0.5 × P(Draw 90')
```

**Why this is derived, not standard:**

538 uses an SPI-difference-based extra-time win probability sub-model. We chose 50/50 because:

- Historical penalty shootout outcomes: ~52/48 favouring "team with better players" → essentially 50/50.
- Extra time: slight edge to stronger team, but fatigue mitigates.
- Building a proper sub-model would take ~3 days for ~0.5–1pp accuracy gain on knockout matches.
- Below precision threshold for v1.

**Cost of simplification:** Modelled WC2026 champion probability for tournament favourite likely differs by ~0.5–1pp from a 538-style refined model. Documented as a known v1 limitation.

**Defensibility:** Medium. Defensible as v1 simplification with explicit trade-off documented. Counter-argument if pushed: knockout is ~16 of 104 matches; the simplification affects ~0.5pp accuracy on champion probability, vs ~3% improvement from getting structural Elo correct.

### 2.8 Monte Carlo iteration count — STANDARD (with current vs target distinction)

**Practitioner consensus:**

- 538: 10,000 sims for WC
- Opta: 10,000 sims
- Groll et al.: 100,000 sims (academic paper)

**Current state vs target:**

- **Phase 3 baselines: 10,000 iterations** per (tournament, model) — sufficient for top-5 champion identification, used for fast head-to-head model comparison (~108s wall time for all 36 baseline runs).
- **Phase 4 final WC2026 output: 100,000 iterations** — needed for group winner markets and conditional probabilities at small base rates (per Sam's group winner check 2026-04-30).

**Why 100k:**

Monte Carlo standard error on champion probability `p` with `N` sims is `√(p(1-p)/N)`. For a 5% favourite at 100k: SE = 0.069pp — below our 5pp signal threshold by ~70× margin. 1M sims would give 0.022pp SE; overkill for headline probabilities but warranted for tail probabilities below 1%.

---

## 4. Validation framework

### 4.1 Three-model plug-in architecture (per Sam's 2026-05-20 framework note)

The simulator has been built as a **3-layer barebone harness**:

```
   Tournament Harness (FIXED across all formats)
            │ delegates to bracket adapter (per format)
            ▼
   Bracket adapter (per tournament format)
            │ calls predictor(team_a, team_b, ctx) per match
            ▼
   Per-game probability model (SWAPPABLE)
```

**Three plug-in models compared head-to-head:**

| Model | Description | Tunable params |
|---|---|---|
| **A. self-Elo** | Elo ratings built from 49,329 historical matches (1872–2026), 336 unique teams | α, β, diagonal, D — LOTO-CV per Phase 4 |
| **B. eloratings-Elo** | eloratings.net scraped per-match ratings, 244 teams | Same 3 params, tuned independently |
| **C. Baseline 40/40/20** | Constant 40% home / 40% away / 20% draw — ignorance floor | None |

**Bracket adapters:** WC2026, WC 8-groups (2002–2022), Euro 24-teams (2020/24), Euro 16-teams (2004/08/12).

### 4.2 Two metrics, two roles

| Metric | Role | Sample size |
|---|---|---|
| **Historical tournament replay** (Σ log P(actual champion) across historical tournaments) | **PRIMARY** — picks Elo source per Sam's framework | n=11 in Phase 3 (Euro 2016 deferred — different bracket geometry from other Euros; adapter pending) |
| **LOTO-CV match-Brier** | **SECONDARY** — tunes (α, β, diagonal, D) within each Elo source | n~760 matches across all 12 target tournaments |

### 4.3 Tuning protocol (Phase 4 — COMPLETE, 2026-06-11)

**Leave-one-tournament-out cross-validation (LOTO-CV):**

1. Hold out tournament T.
2. Fit Elo ratings on all matches before T's start date.
3. Predict every match in T using parameter set θ.
4. Compute Brier score on T's matches.
5. Average across 12 folds.

**Parameter grid:**

```python
PARAM_GRID = {
    "alpha":              [0.20, 0.27, 0.35],
    "beta":               [0.00, 0.09, 0.15],
    "diagonal":           [0.12, 0.20, 0.28],
    "D":                  [1200, 1400, 1600],
}
# 81 combinations × 12 folds × 2 tunable predictors (self-Elo, eloratings-Elo) = 1944 model fits
# Baseline 40/40/20 has no params — single eval per fold (12 fits total).
```

**Selection rule:** Choose θ* minimizing average Brier across folds. **Hard check:** best vs 2nd-best Brier difference must exceed 2 standard errors. If not, fall back to current defaults (α=0.27, β=0.09, diag=0.20, D=1400). This prevents overfitting to a specific tournament composition.

### 4.4 Tuning rule — what we will and will not tune against

**Tune against:**

- Brier score on 3-way W/D/L outcomes from ~760 matches across 12 historical tournaments (WC 2002, 2006, 2010, 2014, 2018, 2022 + Euros 2004, 2008, 2012, 2016, 2020, 2024).

**Never tune against:**

- Market prices (Shin output, Polymarket, Kalshi, Pinnacle).

Market prices are used **only** as a post-tune sanity check diagnostic. If we tuned against market, Stream 3 would become a market echo and lose its independent signal value.

### 4.5 Phase 3 preliminary findings (untuned defaults, 11 historical editions)

| Model | Σ log P | Mean rank of actual champion | Top-3 hit rate |
|---|---|---|---|
| eloratings-Elo | −25.95 | 4.55 | 6/11 |
| self-Elo | −26.74 | 4.73 | 6/11 |
| baseline 40/40/20 | −35.17 | 9.64 | 3/11 |

Both Elo models clear the baseline by ~9 log-prob units (massive). Eloratings-Elo vs self-Elo separation is 0.79 log-prob units — directionally consistent but within tuning-grade noise. Phase 4 LOTO-CV will determine if the lead is robust.

### 4.6 Sanity check against market (diagnostic only)

After parameters are frozen, MC runs on WC2026 current ratings. Compare champion distribution with Polymarket + Pinnacle devigged consensus.

**Decision rules:**

- All teams differ < 2pp from market → suspect implicit overfit, investigate
- 2–10pp difference on 3–10 teams → expected, potential trade signals
- 10pp difference on > 5 teams → likely model bug, **hard stop**

Same diagnostic applied to 12 group winner markets (per Sam's suggestion 2026-04-30).

**Preliminary WC2026 vs Polymarket** (untuned, 10k iter — **DO NOT TRADE ON THESE**):

| Direction | Team | PM mid | Model (eloratings) | Edge |
|---|---|---|---|---|
| BUY | Spain | 15.60% | 20.75% | +5.15pp |
| BUY | Argentina | 8.90% | 14.01% | +5.11pp |
| FADE | England | 11.20% | 6.78% | −4.42pp |
| FADE | France | 16.65% | 12.47% | −4.18pp |
| FADE | Brazil | 8.60% | 4.53% | −4.07pp |

Phase 4 tuning will move probabilities ±2–3pp per team. Current edges may shrink or flip after tuning.

---

## 9. References

**Elo & rating systems:**

- Eloratings.net specification: https://www.eloratings.net/about
- Lasek et al. 2013 (ranking system comparison): *International Journal of Forecasting* 29:144–159.
- Wikipedia "World Football Elo Ratings": https://en.wikipedia.org/wiki/World_Football_Elo_Ratings

**538 SPI methodology:**

- WC 2022 methodology: https://fivethirtyeight.com/features/how-our-2022-world-cup-predictions-work/
- WC 2018 methodology: https://fivethirtyeight.com/features/how-our-2018-world-cup-predictions-work/

**Home advantage research:**

- Bilalić, Gula & Vaci 2021. "The double bind of expertise: When experts' HFA is reduced." *Scientific Reports* 11:21558. (n>4000 national team matches, +0.44 additive goals/match measurement.)
- Pollard & Armatas 2017. "Home advantage and the factors that influence it for football national teams." *International Journal of Performance Analysis in Sport* 17(1-2):121–135. (56–69% home win rate.)
- McCarrick et al. 2021 (COVID HFA): *Psychology of Sport and Exercise.* 4,844 games, 15 leagues.
- Almeida & Leite 2021. "COVID HFA quantification." *Biology of Sport.* 44% reduction in points-based HFA.

**Bivariate Poisson:**

- Karlis & Ntzoufras 2003. "Analysis of sports data by using bivariate Poisson models." *The Statistician* 52:381–393.
- Egidi & Palaskas footBayes R package: https://cran.r-project.org/package=footBayes

**Tournament forecasting:**

- Groll et al. 2018. "Prediction of the FIFA World Cup 2018 — a random forest approach." arXiv:1806.03208.
- Constantinou & Fenton 2012. "RPS for football." *Journal of Quantitative Analysis in Sports* 8(1).

**FIFA / tournament structure:**

- FIFA World Cup 2026 official: https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026
- FIFA 2026 Competition Regulations Article 13 (tiebreaker chain).

---

*Maintainer: Duy Anh (FairLine). Status: v1 standalone validation; Phase 4 LOTO-CV tuning pending; presentation pending Sam + team review.*
