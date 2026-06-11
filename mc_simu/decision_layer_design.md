# Decision layer — design note for review

2026-06-11. Status: awaiting two decisions (§3); uncontested parts (§2) proceed in parallel.
Evidence base: [`audits/strength_mle_validation.md`](audits/strength_mle_validation.md) (12-tournament LOTO, 630 matches, + WC2026 outright vs Polymarket).

## 1. Context in three sentences

We now run two validated, accuracy-equivalent priors: production (eloratings+MV+star → Poisson) and mle_strength (weighted-MLE on results only, same Poisson grid). Their 50/50 log-pool beats both singles across every hyperparameter sweep (point estimate; weight-sweep U-curve confirms the pool carries information), but per-match errors correlate at 0.93 — the structural-prior space is saturated, so the next build is the decision layer, not a third model. Three live signal patterns exist today: Portugal (both models below PM by 2.5–3.2pp), Spain (production-only edge, traced to the eloratings source), France (largest model disagreement: production 15.4% / mle 7.4% / PM 16.1%).

## 2. Uncontested components (building now)

- **Tagging schema:** every paper trade carries {signal class, venue, entry edge, component attribution (Elo- vs MV-driven), exit reason}. Post-tournament question is "which signal class has real edge", not "what was P&L".
- **Entry threshold per venue:** edge must exceed fee + half-spread + MC standard error (100k sims → se < 0.01). Derived from costs and simulation noise only — no market fitting.
- **Longshot filter:** for teams with market prob below ~2–3%, required edge multiplies 2–3×; hard floor below ~0.5% market. Literature-anchored (favorite-longshot bias, Snowberg & Wolfers 2010); our own measurement: PM tail floor 0.10%, +100–300% relative premiums. Form anchored from literature, sanity-checked (not fitted) on WC2022/Euro2024 history. One-direction data flow: this layer models the market instrument, never touches the priors.
- **Consensus prior:** 50/50 log-opinion pool (preregistered weight; sweep plateau 0.4–0.8 makes any interior choice statistically equivalent).
- **Market focus:** group-winner markets (12×4 teams, settle end of June) prioritized over champion markets for sample size; advance-to-R32/R16 next.

## 3. Two decisions needed before sizing code

**D1 — Sizing function: discrete classes vs continuous discount.**
- Option A (discrete): Class A consensus → 0.25 Kelly; Class B single-model → log-only; Class C disagreement → no-trade.
- Option B (continuous): `size = fractional_Kelly(p_pool vs p_market) × g(internal_agreement)`, g decreasing in model disagreement but never zero.
- Evidence favors B: Test 1 (+match-mix patch) shows disagreement does **not** predict production error — so disagreement justifies a discount, not a veto; Test 2's pool lift comes precisely from disagreement matches. Under B, Portugal sizes full, Spain sizes near-zero naturally (small pool edge), France becomes a small-but-nonzero short instead of a forced zero.
- Why it needs an owner call: France is the live test case — pool ~10–11% vs PM 16.1% is the **largest divergence in the book**. A=France is untouchable research material; B=France is a small position. Philosophy call, not a technical one.

**D2 — Tradeable object: pool-vs-market, or production-vs-market with mle as discount input?**
- Pool-vs-market is the Test-2-consistent choice and what §2 assumes. Production-vs-market keeps the locked model as the single anchor (cleaner story, weaker use of evidence). Decide jointly with D1.

## 4. Explicitly out of scope (v1)

Context-dependent model weights (mle on close games, production on mismatches — logged as confidence flag only); any third structural prior (club-axis ClubElo aggregation stays in post-tournament backlog with league-tier defaults); any parameter fitted against current market prices.
