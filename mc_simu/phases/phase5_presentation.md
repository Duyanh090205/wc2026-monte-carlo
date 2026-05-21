## 6. Phase 5 — Presentation Prep (3 days)

**Goal:** Prepare materials for review meeting với Sam và team. Decide on v2 integration sau meeting.

**Files:** `mc_simu/presentation/` folder, `docs/mc_simu_v1_report.md`

### 6.1 Presentation deliverables

**Deliverable 5.1 — Technical report (`docs/mc_simu_v1_report.md`)**

Structure:
1. Executive summary (1 page): what we built, key metrics, recommendation
2. Methodology overview: Elo + Poisson + HFA additive + MC simulation
3. Feature spec rationale: tại sao α=0.40, β=0.13, bỏ altitude/jet lag
4. Validation results:
   - Phase 1 Brier on 4 tournaments (table)
   - Phase 2 roster overlay decision (kept/dropped, evidence)
   - Phase 3 sanity check vs Opta (table)
   - Phase 4 LOTO-CV results (top 5 θ candidates, Brier per tournament heatmap)
5. WC2026 predictions:
   - **Champion market:** Top-10 champion probabilities với MC, vs Polymarket vs Pinnacle devigged
   - **Group winner markets (per Sam's suggestion):** 12 group tables, MC vs Polymarket group winner odds
   - Notable divergences (teams where MC differs >5pp from market in either market type) — potential trade signals
6. Limitations + future work:
   - V1 simplifications: penalty shootout, no altitude, no jet lag
   - V2 recommendations: integrate vào live_edge, add altitude post-WC2026 data
7. Recommendation for v2 integration decision

**Deliverable 5.2 — Slide deck (10-15 slides)**

Slides outline:
1. Title — Stream 3: MC Tournament Simulator
2. Problem — why need Stream 3 (current 2 streams = market consensus only)
3. Approach — structural prior via Elo + MC
4. Feature spec — HFA additive, validated via deep research
5. Validation methodology — Brier + LOTO-CV
6. Phase 1-3 results
7. Phase 4 LOTO-CV result (θ* table)
8. WC2026 predictions vs market — champion + 12 group winners (dual sanity check)
9. Notable divergences — potential edges
10. Limitations
11. V2 proposal — integration decision
12. Q&A backup slides (calibration plots, sample tournament sims, code structure)

**Deliverable 5.3 — Interactive dashboard (Streamlit)**

Add new tab to existing `dashboard/app.py`:
- "MC Simulator" tab
- WC2026 champion distribution chart (top 20 teams)
- Comparison: MC vs Shin (Stream 1) vs Polymarket+Kalshi (Stream 2)
- Per-team divergence highlighted
- Filters: by confederation, by HFA-eligible (host/co-host/none)

**v1 data source:** Read CSV `data/mc_simu/mc_fair_odds_world_cup_2026_<latest>.csv` directly (no DB query). Stream 1+2 comparison data pre-computed once for presentation (snapshot CSV) — dashboard avoids realtime DB calls in v1.

Existing dashboard tabs unchanged. README mentions "Monte Carlo placeholder" slot in Odds Comparison tab — fills this slot.

**v2 future:** swap CSV read for `mc_fair_odds` DB query when DB write is reactivated.

### 6.2 Demo script for live presentation

```python
# src/mc_simu/demo.py
"""Live demo script — run during team meeting.

1. Show MC simulation running (100k iterations, ~30s)
2. Print top-10 champion probabilities
3. Compare vs current Polymarket + Pinnacle devigged
4. Highlight 3 biggest divergences
5. Open dashboard tab for visual exploration"""
```

### 6.3 V2 integration decision (post-meeting)

After presentation, team decides one of:

**Option A — Approve full integration**
- Proceed to v2 plan: modify `live_edge.py` + `edge_signals` + `paper_trader.py`
- Stream 3 contributes to trading via ensemble fair_prob
- Estimated effort: 2-3 days

**Option B — Approve shadow mode**
- Add MC reads to `live_edge.py` but log only
- No effect on paper_trader decisions
- Observe 2 weeks before deciding active integration

**Option C — Keep standalone**
- MC remains research tool
- Manual review of `mc_fair_odds` table for trade ideas
- Revisit integration after WC2026 group stage data

**Option D — Reject / iterate**
- Specific issues identified during review
- Address feedback, re-present

Plan v2 will be drafted based on this decision.

### 6.4 Phase 5 deliverables

- `docs/mc_simu_v1_report.md` — technical report
- `mc_simu/presentation/slides.pdf` — slide deck
- `dashboard/app.py` — Streamlit MC tab added (read-only)
- `src/mc_simu/demo.py` — live demo script
- Meeting scheduled với Sam + team
- Decision documented in `docs/mc_simu_v2_decision.md`

---

