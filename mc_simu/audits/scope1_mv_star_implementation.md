# Scope 1 — MV Blend + Star Presence Implementation Report

**Status: LOCKED — ready to ship.**
**Date: 2026-05-27**
**Final config:** MV blend α=0.5 + Star bonus X=15 + default predict_match knobs

---

## TL;DR

Sam đề xuất 2026-05-27: thêm Transfermarkt market value (MV) song song Elo để model gần PM/Kalshi hơn. Sau implementation + iteration:

- **MV blend α=0.5** (50/50 z-score blend với Elo) giảm JSD vs PM **40%** (0.0600 → 0.0360)
- **Star bonus X=15 Elo/star** thêm giảm **33%** nữa (0.0360 → 0.0235), capture aging legends (Messi/Ronaldo MV-deflation problem)
- **Title bonus** (WC titles) tested marginal (-0.4%), skipped — không justify complexity
- **Total cumulative: -61% JSD vs PM** (0.0600 → 0.0235), Brier history 0.1946 (PASS-WARN, ≤ 0.21 HARD-STOP)
- 3 residual outliers ≥ 2pp = **alpha opportunities**: Brazil/Portugal UNDER PM (bet against), Spain OVER (bet for)

---

## 1. Final locked config

```yaml
# Selection script: src/mc_simu/tune_to_market.py
mv_blend_alpha: 0.5            # MV-heavy per Sam intuition + JSD optimum
star_bonus_X: 15               # Elo per identified star (Messi/Ronaldo fix)

# predict_match knobs (unchanged from Phase 1 defaults)
D: 1400                         # Elo goals denominator
diagonal_inflation: 0.20
alpha_HFA: 0.27                 # own-country home advantage
beta_confed: 0.09               # same-confederation host bonus
```

**Reproduce final run:**
```bash
python src/mc_simu/tune_to_market.py \
  --D-grid 1400 --diag-grid 0.20 \
  --n 100000 --mv-blend-alpha 0.5 --star-bonus-X 15
```

### Performance metrics

| Metric | Value | Gate / Δ |
|---|---|---|
| **JSD vs PM** | **0.0235** | -61% vs baseline 0.0600 ⬇️ |
| JSD vs Kalshi | 0.0344 | |
| JSD vs FairLine | 0.0206 | (lowest — but Sam said don't anchor) |
| **Brier history** | **0.1946** | PASS-WARN (gate ≤ 0.21) ✓ |
| L1 vs PM+Kalshi consensus | 25.6pp | |
| Outliers \|edge\| ≥ 2pp | 3 teams | Brazil, Portugal, Spain |

---

## 2. MV Blend — Feature engineering

### Source data
- **Transfermarkt FIWC participants page**: https://www.transfermarkt.com/world-cup/teilnehmer/pokalwettbewerb/FIWC
- Scraped 2026-05-27 → 48/48 coverage
- MV range: €18m (Jordan) → €1.48bn (France), ratio 80x
- HTML cached offline at `data/mc_simu/cache/transfermarkt/wc2026_teams.html`

### Blend formula (z-score space)

```
z_elo[t]  = (elo[t] - mean(elo)) / std(elo)
z_mv[t]   = (log(mv[t]) - mean(log(mv))) / std(log(mv))
z_blend   = (1 - α) * z_elo + α * z_mv
blended_elo[t] = z_blend * std(elo) + mean(elo)
```

Design choices:
- **Log(MV)**: handles 80x right-skew, "tier difference" comparable across spread
- **Z-score**: 2 signals comparable in standardized units, alpha weights them fairly
- **Map back to Elo scale**: predict_match unchanged, blend transparent

### α sweep — JSD vs PM (n=10k MC)

| α | JSD vs PM | Brier_history | Δ JSD |
|---|---|---|---|
| 0.0 (pure Elo) | 0.0600 | 0.1974 | baseline |
| 0.3 | 0.0413 | 0.1959 | -31% |
| 0.4 | 0.0382 | — | -36% |
| **0.5** ⭐ | **0.0360** | **0.1957** | **-40%** |
| 0.55 | 0.0351 | — | -42% |
| 0.6 | 0.0371 | — | -38% |
| 0.7 | 0.0417 | 0.1963 | -30% |

**Chốt α = 0.5** — simple round number, near optimum, both JSD + Brier improve.

### Per-team impact (top 8)

| Team | PM | α=0 baseline | α=0.5 blend | edge baseline | edge blend |
|---|---|---|---|---|---|
| Spain | 16.28% | 20.75% | 15.88% | +4.47pp | -0.42pp ✓ fix |
| France | 15.99% | 12.47% | 13.04% | -3.52pp | -2.95pp partial |
| England | 10.57% | 6.78% | 8.55% | -3.79pp | -2.02pp partial |
| **Argentina** | 8.21% | **14.01%** | 9.61% | **+5.80pp** | **+1.65pp** ✓ fix |
| Portugal | 9.39% | 4.52% | 6.05% | -4.87pp | -3.33pp partial |
| Brazil | 8.77% | 4.53% | 5.47% | -4.24pp | -3.30pp partial |
| Germany | 5.06% | 2.79% | 4.56% | -2.27pp | -0.50pp ✓ fix |
| Netherlands | 3.73% | 4.01% | 4.65% | +0.28pp | +0.92pp ≈ |

**Big winners from MV blend:** Argentina (over fixed), Germany (under fixed), Spain (over fixed).
**Residual under:** France/Brazil/Portugal/England — partial only, brand premium remains.

---

## 3. Star Presence — Aging legends fix

### Motivation

MV reflects **resale value** (driven by age + contract years), NOT current playing skill:
- Messi (38, MV €15m) but elite finisher
- Ronaldo (41, MV €12m) but still scoring
- Modric (40, MV €3m) but creative engine

MV blend alone systematically under-rates teams with aging legends → "star presence" indicator adds bonus.

### Star definition (3-way union, hand-curated 2026-05-27)

A player counts as "star" if **any**:
1. Current Transfermarkt MV ≥ €70m → catches young rising stars (Yamal €200m, Wirtz €130m)
2. Peak career MV ≥ €100m → catches aging legends (Messi peak €180m, Ronaldo €120m)
3. Ballon d'Or top-30 in 2024 or 2025

**Coverage:** 49 teams, 130 total stars. Distribution:
- 8 stars (3 teams): France, Spain, England
- 7 stars (3): Portugal, Brazil, Argentina
- 6 stars (3): Germany, Netherlands, Uruguay
- 4 stars (6): Belgium, Croatia, Japan, Morocco, USA, Switzerland
- 3 stars (7): Senegal, Turkey, Sweden, Ecuador, Austria, Ivory Coast, Scotland
- 2 stars (9): Norway, Colombia, Mexico, Egypt, Algeria, S.Korea, Czechia, Iran, Canada
- 1 star (4): Ghana, Tunisia, DR Congo, Bosnia
- 0 stars (14): Saudi Arabia, Qatar, NZ, Jordan, Iraq, Cape Verde, Curacao, S.Africa, Australia, Uzbekistan, Haiti, Panama, Paraguay, Italy

### Bonus sweep — JSD + Brier (n=15k)

| X (Elo / star) | JSD vs PM | Brier_history | Status |
|---|---|---|---|
| 0 (MV blend only) | 0.0354 | 0.1957 | baseline |
| 5 | 0.0275 | 0.1948 | improved |
| 10 | 0.0252 | **0.1945** (best Brier) | improved |
| **15** ⭐ | **0.0239** (best JSD) | 0.1946 | **chốt** |
| 20 | 0.0257 | 0.1952 | starts worse |
| 30 | 0.0366 | 0.1977 | over-correction |

**Chốt X = 15** — best JSD, Brier near-optimal, all PASS-WARN.

### Per-team impact (X=15 vs X=0)

| Team | Stars | PM | X=0 (MV only) | X=15 (final) | Δ |
|---|---|---|---|---|---|
| **France** | 8 | 16.31% | 13.04% | **15.57%** | ✓ near-match (-0.74pp) |
| **Portugal** | 7 | 9.98% | 6.05% | 7.27% | +1.22pp ✓ improved |
| **Brazil** | 7 | 9.02% | 5.47% | 6.25% | +0.78pp ≈ |
| **England** | 8 | 10.75% | 8.55% | 11.89% | +3.34pp ✓ (slight over) |
| Spain | 8 | 16.60% | 15.88% | 19.51% | +3.63pp ⚠️ over |
| Argentina | 7 | 7.97% | 9.61% | 10.55% | +0.94pp ⚠️ more over |
| Germany | 6 | 4.89% | 4.56% | 4.32% | -0.24pp ≈ |
| Norway | 2 | 2.40% | 2.86% | 1.53% | -1.33pp |

**Trade-off:** Fix top brand teams (France/Portugal/Brazil), but Spain/Argentina/England slightly over-correct. Net JSD significantly better.

---

## 4. Title Bonus — Tested + Skipped

### Hypothesis

Teams with historical WC titles get "brand premium" from PM bettors. Add bonus Y Elo per title.

| Team | WC titles | Years |
|---|---|---|
| Brazil | 5 | 1958, 1962, 1970, 1994, 2002 |
| Germany | 4 | 1954, 1974, 1990, 2014 |
| Argentina | 3 | 1978, 1986, 2022 |
| France | 2 | 1998, 2018 |
| Uruguay | 2 | 1930, 1950 |
| Spain | 1 | 2010 |
| England | 1 | 1966 |

### Y sweep (on top of MV+Star)

| Y | JSD vs PM | Notes |
|---|---|---|
| 0 | 0.0239 | baseline |
| **5** | **0.0236** | marginal -1.3% (within MC noise) |
| 10 | 0.0250 | worse |
| 15 | 0.0280 | much worse |
| 20 | 0.0322 | over-correction |

### Per-team trade-off at Y=5

| Team | Edge before | Edge after Y=5 | Verdict |
|---|---|---|---|
| Brazil | -2.56pp | -1.54pp | ✓ improved 1pp |
| Germany | -0.57pp | +0.01pp | ✓ fixed |
| **Argentina** | +2.64pp | **+3.53pp** | ⚠️ WORSE |
| **Portugal** (0 titles) | -2.79pp | -3.18pp | ⚠️ KHÔNG fix (no title) |
| Spain | +2.85pp | +2.43pp | marginal |

### Why skipped

1. **JSD gain = noise floor** (0.0003 — within MC error for n=20k)
2. **Argentina over thêm 0.9pp** — already over, gets worse
3. **Portugal residual UNFIXED** (0 titles → no bonus) — biggest under stays
4. **Complexity vs gain ratio** — adding feature for marginal win

**Decision:** Skip title bonus. Lock at MV+Star only.

---

## 5. Pure Elo vs Pure MV vs Blend — Why blend wins

### Empirical comparison (n=20k)

| Setup | JSD vs PM | Spearman rank vs PM |
|---|---|---|
| Pure Elo (α=0) | 0.0613 | 0.86 |
| **50/50 blend** | **0.0341** | — |
| Pure MV (α=1) | 0.0631 | **0.91** |

**Key insight:** MV alone is NOT closer to PM than Elo (JSD ≈ same). But MV captures PM ranking slightly better. **The win is the blend** — 2 orthogonal signals combine to give JSD ≈ half of either alone.

### Where Elo vs MV disagree → PM stands in middle

| Team | PM rank | Elo rank | MV rank | PM at midpoint? |
|---|---|---|---|---|
| Spain | 1 | 1 | 3 | (Elo correct) |
| France | 2 | 3 | 1 | ✓ PM ≈ midpoint |
| Argentina | 6 | **2** (too high) | **8** (too low) | ✓ PM midpoint |
| Germany | 7 | **11** (too low) | **4** (too high) | ✓ PM midpoint |

→ **Confirmation**: PM bettors implicitly weight both signals. Blend captures this directly.

---

## 6. Research support

After implementing, surveyed literature to validate approach + check for missed features:

### Validated features (already covered or tested)
- **Hvattum & Arntzen (2010)** — Elo for football match prediction (foundational, baseline ✓)
- **Peeters (2018)** — Transfermarkt crowd valuations > FIFA/Elo for international forecasts (validates MV ✓)
- **Academic paper PMC10520731** — 3+ WC titles significant (tested, marginal)
- **Klement econometric model** (predicted 3 WCs) — uses GDP/population/temperature/FIFA + host (host already covered; others would predict against Brazil)

### Features NOT covered, ranked by potential impact
- **FiveThirtyEight SPI**: uses xG (expected goals) shot-based + non-shot-based → captures "quality of play", NOT just outcomes. Major missing structural signal but heavy lift (1-2 weeks).
- **Olympic medal count** — academic-significant, easy to add but marginal expected
- **Continental titles** (Euro/Copa wins) — would fix Portugal Euro 2016 brand premium
- **AFC region adjustment** — academic-significant, but only affects Asian teams

### Notable independent prediction for WC 2026
- **Klement (2026):** Netherlands win, Portugal runner-up
  - Match PM's bullish Portugal view
  - Suggests our model's UNDER-pricing of Portugal might be over-conservative

---

## 7. Residual outliers = Alpha opportunities

At final config (n=100k):

| Team | MC | PM | Edge | Interpretation | Trade |
|---|---|---|---|---|---|
| **Brazil** | 6.32% | 9.35% | **-3.03pp** | Brand premium PM gives 5x champions | **Bet AGAINST** Brazil ở PM |
| **Portugal** | 7.42% | 10.45% | **-3.03pp** | Ronaldo + Euro 2016 + Klement also bullish | **Bet AGAINST** Portugal ở PM |
| **Spain** | 19.25% | 17.20% | **+2.05pp** | Model amplifies Spain's strong Elo+stars | **Bet FOR** Spain ở PM |

**Per Sam philosophy** ([[sam-quant-philosophy-model-market]]):
> "Tuning EXACTLY to market → model becomes market regressor with no edge. Bad."

Residual disagreement is the **alpha target** — model should diverge from market when we have structural signal market lacks. Sam decides if/how to trade.

**Caveat:** Multiple independent sources (Klement, FiveThirtyEight-style, PM consensus) all give Portugal more weight than our model. The under-pricing **may not be model error** — it may be that we're missing the "xG quality of play" signal that other professional models use.

---

## 8. Files delivered

### New
- [src/mc_simu/mv_blend.py](../src/mc_simu/mv_blend.py) — z-score blend module
- [src/mc_simu/star_presence.py](../src/mc_simu/star_presence.py) — star bonus module
- [src/mc_simu/scrape_transfermarkt_mv.py](../src/mc_simu/scrape_transfermarkt_mv.py) — Transfermarkt scraper
- [src/mc_simu/validate_phase1_with_mv.py](../src/mc_simu/validate_phase1_with_mv.py) — Brier check w/ MV+Star

### Modified
- [src/mc_simu/tune_to_market.py](../src/mc_simu/tune_to_market.py)
  - Dropped Fairline from consensus (Sam direction: PM/Kalshi only)
  - Added `--mv-blend-alpha` CLI flag (default 0.0)
  - Added `--star-bonus-X` CLI flag (default 0.0)

### Data
- [data/mc_simu/transfermarkt_mv_wc2026.csv](../data/mc_simu/transfermarkt_mv_wc2026.csv) — 48 teams, MV in EUR
- [data/mc_simu/star_presence_wc2026.json](../data/mc_simu/star_presence_wc2026.json) — hand-curated star counts + criteria
- [data/mc_simu/cache/transfermarkt/wc2026_teams.html](../data/mc_simu/cache/transfermarkt/wc2026_teams.html) — cached HTML (offline rule)

---

## 9. Known limitations + future work

### Known limitations
1. **Star list = hand-curated** (subjective). Sam should review JSON list.
2. **MV is 2026 snapshot** — historical Brier check biased (Yamal/Wirtz were rising in 2018-2024 with different MV profile).
3. **No xG signal** — biggest structural gap vs FiveThirtyEight-style models.
4. **Aging legends still incomplete fix** — Messi peak MV €180m vs current €15m; star bonus partial fix.
5. **Brazil/Portugal residual** — could be PM over-pricing OR our model under-pricing; unresolvable without more signals.

### Future work (deferred — not for v1)
- **Squad-level top-K MV scrape** (48 page fetches, ~2h) — captures top-11 elite, may fix Brazil residual
- **Peak career MV per player** (per-player /preisverlauf scrape) — proper Messi/Ronaldo fix
- **xG-based offensive/defensive ratings** (FiveThirtyEight approach) — biggest structural signal missing, 1-2 weeks
- **Continental titles bonus** (Portugal Euro 2016, Argentina Copa 2024)
- **Historical MV per year** (to fix biased Brier check)
- **WC2026 watch list:**
  - Verify model's Brazil/Portugal under-prediction
  - If both reach SF/F → model wrong, re-open feature engineering
  - If both eliminated R16/QF → model correct, PM was over-pricing brand

---

## 10. Tóm lại 1 đoạn cho Sam

> Đợt 2 implement đầy đủ Sam's MV proposal. Hardcoded **α=0.5 + Star X=15** giảm JSD vs PM **61%** (0.0600 → 0.0235), Brier 0.1946 (PASS-WARN), all 48 teams covered. Title bonus test rồi skip (marginal +Argentina over). Star list 130 cầu thủ hand-curated (file JSON, Sam edit được). Residual 3 outliers ≥2pp: Brazil/Portugal UNDER PM (bet AGAINST = potential alpha), Spain OVER (bet FOR). Research survey confirms approach reasonable; missing xG signal là next-level work nhưng ngoài v1 scope. Sẵn sàng ship.

---

## Appendix: cumulative improvement table

| Stage | Setup | JSD vs PM | Brier | Δ JSD |
|---|---|---|---|---|
| Baseline | Pure Elo | 0.0600 | 0.1974 | — |
| Đợt 2-A | + MV blend α=0.5 | 0.0360 | 0.1957 | -40% |
| Đợt 2-B | + Star bonus X=15 | **0.0235** | **0.1946** | **-61%** |
| (rejected) | + Title bonus Y=5 | 0.0236 | (not tested) | -0.4% |
| (rejected) | D grid tune | 0.0296 | (not tested) | -8% (breaks Brier risk) |
