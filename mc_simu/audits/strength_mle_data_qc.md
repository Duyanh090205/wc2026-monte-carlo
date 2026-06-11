# mle_strength — Phase 1 Data QC Report

Date: 2026-06-11 (updated same day after snapshot refresh). Source: [`data/mc_simu/results.csv`](../../data/mc_simu/results.csv) (martj42/international_results mirror — same file Phase 0 production pipeline downloaded; see [`mc_simu/phases/phase0_audit.md`](../phases/phase0_audit.md) for original provenance). Single source of truth shared with `elo.py`, no separate copy.

**Snapshot refreshed 2026-06-11** from upstream commit `c636851f6e38` (2026-06-11T06:44Z, 49,477 rows): adds 146 played matches 2026-05-16 → 2026-06-10 (the pre-WC warm-ups the previous snapshot lacked). Refresh diff audited before swap: schema identical; only historical change is a global team rename `China PR` → `China` (~700 rows, not a WC2026 team; NOTE: rebuilding `elo_history.csv` from this snapshot would need the self-source alias updated); one upstream entry-duplicate found (Gibraltar 4-1 Cayman Islands 2026-06-06 entered under two city spellings) — handled by a principled dedup rule in `load_matches` (drop identical (date, teams, scores) quintuples; same-day double-headers with differing scores are kept).

Scope decisions applied (vs the original plan draft):
- Reuse `results.csv` instead of a fresh `model_b/data/raw/` snapshot.
- No `name_map.py`: fitting keeps csv-native names (same convention as the self-Elo source); adapter-name divergences resolved at the lookup boundary via `strength_mle.TEAM_NAME_ALIASES` — exactly the 2 cases the self source already aliases (`Czechia` → `Czech Republic`, `Curacao` → `Curaçao`).
- Importance mapping built on the repo's existing 6-bucket classifier `_common.infer_tournament_type` (no new string enumeration logic): world_cup_final 4.0, continental_final 3.0 (incl. Confederations Cup), qualifier 2.5, **nations_league 2.5** (decision: qualifier parity, mirroring K=40 in `elo.K_FACTORS`), other_tournament 1.0, friendly 1.0.

## Gate results (T1.x) — all PASS, automated in [`tests/mc_simu/test_strength_mle_data.py`](../../tests/mc_simu/test_strength_mle_data.py)

| Gate | Check | Result |
|---|---|---|
| T1.1 | Row counts | raw 49,477; 72 NA-score rows dropped (future WC2026 fixtures); window (≥1995-01-01, < as_of 2026-06-11): **29,292 matches**, 323 teams |
| T1.2 | Nulls / types | 0 nulls in 6 modeling columns; scores int ≥ 0; all dates parse |
| T1.3 | Duplicates on (date, home, away) | **0** after entry-dup rule (1 upstream double-entry dropped, see header) |
| T1.4 | Sanity bands | mean total goals **2.771** (band 2.3–3.1 ✓); draw rate **23.3%** (0.18–0.30 ✓); non-neutral home-win 50.9% > away-win 25.9% (n=21,031) ✓ |
| T1.5 | WC2026 coverage | all 48 teams resolve; min matches in window: Curacao 165, Cape Verde 191, New Zealand 217 (gate ≥30, ample) |
| T1.6 | Neutral flag | of 448 WC-finals rows, 43 non-neutral — **100%** are host playing at home ✓ |
| T1.7 | External cross-check | not re-run: dataset already validated during production Phase 0/1 (same file builds `elo_history.csv`); offline rule keeps live re-verification out of this phase |
| T1.8 | Importance coverage | 100% of rows weighted; spot checks WC→4.0, WC-qualification→2.5, Euro/Copa América→3.0, Nations League→2.5, Friendly→1.0 all ✓ |

Weight-function unit checks (early T2.6): match exactly H=1095 days before as_of gets w_time=0.5; WC weighs 4× equal-dated friendly; weights positive, strictly decreasing with age.

## Effective sample under weights (as_of 2026-06-10, H=3y)

Weight share by recency (as_of 2026-06-11, post-refresh): matches ≥2024 carry 45.2%, ≥2020 carry 78.4%, ≥2015 carry 93.9%, ≥2010 carry 98.1%. Window-start choice (1995 vs 1990 vs 2000) is second-order as expected — pre-2000 matches carry ~0.1% of weight (sensitivity §6.4 still to be run for the record).

Bucket composition (window): qualifier 11,484 · friendly 11,185 · other_tournament 2,956 · continental_final 2,139 · nations_league 1,080 · world_cup_final 448.

## Findings / flags

1. **Snapshot staleness — RESOLVED 2026-06-11:** previous snapshot ended 2026-03-31; refreshed to upstream `c636851f6e38`, newest played match now **2026-06-10** (covers the pre-WC warm-up window). Note the full tournament-enumeration table below was generated pre-refresh; the 146 added matches are friendlies/Unity Cup/qualifiers already covered by existing buckets.
2. Non-FIFA entities present (CONIFA World Football Cup, Viva World Cup, Island Games...) — classified other_tournament/friendly at weight 1.0; their participants never intersect WC2026 teams' fitting in any material way. No action.
3. `Pacific Games` / `Southeast Asian Games` fall through to the friendly bucket (weight 1.0) — acceptable: minor multi-sport events, no WC2026 relevance.
4. Inter-confederation identification rests on WC finals + Confederations Cup + cross-confed friendlies (known limitation from the plan §8) — importance weights mitigate; to be quantified in Phase 4 diversity report.

## Full tournament-string enumeration (136 distinct in window)

Generated from the window data; mapping is `infer_tournament_type` → `IMPORTANCE_WEIGHTS`; sorted by match count.

| Tournament | Bucket | Weight | Matches |
|---|---|---|---|
| Friendly | friendly | 1.0 | 9611 |
| FIFA World Cup qualification | qualifier | 2.5 | 6482 |
| UEFA Euro qualification | qualifier | 2.5 | 1911 |
| African Cup of Nations qualification | qualifier | 2.5 | 1657 |
| UEFA Nations League | nations_league | 2.5 | 658 |
| AFC Asian Cup qualification | qualifier | 2.5 | 613 |
| African Cup of Nations | continental_final | 3.0 | 586 |
| FIFA World Cup | world_cup_final | 4.0 | 448 |
| CONCACAF Nations League | nations_league | 2.5 | 422 |
| CFU Caribbean Cup qualification | qualifier | 2.5 | 414 |
| Gold Cup | continental_final | 3.0 | 388 |
| CECAFA Cup | other_tournament | 1.0 | 372 |
| COSAFA Cup | other_tournament | 1.0 | 354 |
| Island Games | other_tournament | 1.0 | 352 |
| Copa América | continental_final | 3.0 | 326 |
| UEFA Euro | continental_final | 3.0 | 308 |
| AFF Championship | other_tournament | 1.0 | 291 |
| AFC Asian Cup | continental_final | 3.0 | 282 |
| Gulf Cup | other_tournament | 1.0 | 219 |
| CFU Caribbean Cup | other_tournament | 1.0 | 163 |
| SAFF Cup | other_tournament | 1.0 | 162 |
| UNCAF Cup | other_tournament | 1.0 | 147 |
| Confederations Cup | continental_final | 3.0 | 136 |
| EAFF Championship | other_tournament | 1.0 | 130 |
| WAFF Championship | other_tournament | 1.0 | 114 |
| Oceania Nations Cup | continental_final | 3.0 | 113 |
| Arab Cup | other_tournament | 1.0 | 102 |
| CONIFA World Football Cup | other_tournament | 1.0 | 101 |
| AFC Challenge Cup qualification | qualifier | 2.5 | 92 |
| King's Cup | other_tournament | 1.0 | 92 |
| Gold Cup qualification | qualifier | 2.5 | 88 |
| AFC Challenge Cup | other_tournament | 1.0 | 87 |
| Indian Ocean Island Games | other_tournament | 1.0 | 77 |
| Pacific Games | friendly | 1.0 | 75 |
| Southeast Asian Games | friendly | 1.0 | 72 |
| Cyprus International Tournament | other_tournament | 1.0 | 70 |
| CONCACAF Nations League qualification | qualifier | 2.5 | 68 |
| South Pacific Games | friendly | 1.0 | 68 |
| AFF Championship qualification | qualifier | 2.5 | 62 |
| Amílcar Cabral Cup | friendly | 1.0 | 60 |
| Viva World Cup | friendly | 1.0 | 60 |
| Kirin Cup | other_tournament | 1.0 | 58 |
| FIFA Series | friendly | 1.0 | 57 |
| Muratti Vase | friendly | 1.0 | 57 |
| Asian Games | friendly | 1.0 | 54 |
| Baltic Cup | friendly | 1.0 | 51 |
| CONIFA European Football Cup | other_tournament | 1.0 | 49 |
| Nehru Cup | friendly | 1.0 | 46 |
| Coupe de l'Outre-Mer | friendly | 1.0 | 42 |
| CONCACAF Series | friendly | 1.0 | 35 |
| Malta International Tournament | friendly | 1.0 | 34 |
| Merdeka Tournament | friendly | 1.0 | 34 |
| Arab Cup qualification | qualifier | 2.5 | 28 |
| COSAFA Cup qualification | qualifier | 2.5 | 27 |
| ASEAN Championship | friendly | 1.0 | 26 |
| Oceania Nations Cup qualification | qualifier | 2.5 | 25 |
| USA Cup | friendly | 1.0 | 25 |
| United Arab Emirates Friendship Tournament | friendly | 1.0 | 25 |
| Windward Islands Tournament | friendly | 1.0 | 24 |
| MSG Prime Minister's Cup | friendly | 1.0 | 23 |
| Kirin Challenge Cup | friendly | 1.0 | 22 |
| Lunar New Year Cup | friendly | 1.0 | 22 |
| Prime Minister's Cup | friendly | 1.0 | 22 |
| South Asian Games | friendly | 1.0 | 21 |
| ABCS Tournament | friendly | 1.0 | 20 |
| Melanesia Cup | friendly | 1.0 | 20 |
| Inter Games | friendly | 1.0 | 19 |
| Intercontinental Cup | friendly | 1.0 | 17 |
| Korea Cup | friendly | 1.0 | 17 |
| ELF Cup | friendly | 1.0 | 16 |
| Dunhill Cup | friendly | 1.0 | 15 |
| Millennium Cup | friendly | 1.0 | 15 |
| Pacific Mini Games | friendly | 1.0 | 15 |
| UNIFFAC Cup | friendly | 1.0 | 15 |
| Dynasty Cup | friendly | 1.0 | 14 |
| Nile Basin Tournament | friendly | 1.0 | 14 |
| AFC Solidarity Cup | friendly | 1.0 | 13 |
| King Hassan II Tournament | friendly | 1.0 | 12 |
| Simba Tournament | friendly | 1.0 | 12 |
| East Asian Games | friendly | 1.0 | 11 |
| Palestine International Championship | friendly | 1.0 | 11 |
| SKN Football Festival | friendly | 1.0 | 11 |
| CAFA Nations Cup | friendly | 1.0 | 10 |
| FIFI Wild Cup | friendly | 1.0 | 10 |
| Nordic Championship | friendly | 1.0 | 10 |
| Dragon Cup | friendly | 1.0 | 9 |
| Philippine Peace Cup | friendly | 1.0 | 9 |
| VFF Cup | friendly | 1.0 | 9 |
| CONIFA Asia Cup | other_tournament | 1.0 | 8 |
| Superclásico de las Américas | friendly | 1.0 | 8 |
| Unity Cup | friendly | 1.0 | 8 |
| International Tournament of Peoples, Cultures and Tribes | friendly | 1.0 | 7 |
| Mahinda Rajapaksa Cup | friendly | 1.0 | 7 |
| Mapinduzi Cup | friendly | 1.0 | 7 |
| Tri Nation Tournament | friendly | 1.0 | 7 |
| Morocco, Capital of African Football | friendly | 1.0 | 6 |
| Nations Cup | friendly | 1.0 | 6 |
| Tournoi de France | friendly | 1.0 | 6 |
| Tynwald Hill Tournament | friendly | 1.0 | 6 |
| CONIFA World Football Cup qualification | qualifier | 2.5 | 5 |
| Copa Paz del Chaco | friendly | 1.0 | 5 |
| EAFF Championship qualification | qualifier | 2.5 | 5 |
| Mauritius Four Nations Cup | friendly | 1.0 | 5 |
| Afro-Asian Games | friendly | 1.0 | 4 |
| Al Ain International Cup | friendly | 1.0 | 4 |
| CONIFA Africa Football Cup | other_tournament | 1.0 | 4 |
| Canadian Shield | friendly | 1.0 | 4 |
| Copa América qualification | qualifier | 2.5 | 4 |
| Copa del Pacífico | friendly | 1.0 | 4 |
| Corsica Cup | friendly | 1.0 | 4 |
| Jordan International Tournament | friendly | 1.0 | 4 |
| Navruz Cup | friendly | 1.0 | 4 |
| OSN Cup | friendly | 1.0 | 4 |
| CONIFA South America Football Cup | other_tournament | 1.0 | 3 |
| Hungary Heritage Cup | friendly | 1.0 | 3 |
| Niamh Challenge Cup | friendly | 1.0 | 3 |
| Outrigger Challenge Cup | friendly | 1.0 | 3 |
| Soccer Ashes | friendly | 1.0 | 3 |
| Three Nations Cup | friendly | 1.0 | 3 |
| Tri-Nations Series | friendly | 1.0 | 3 |
| World Unity Cup | friendly | 1.0 | 3 |
| ASEAN Championship qualification | qualifier | 2.5 | 2 |
| Atlantic Heritage Cup | friendly | 1.0 | 2 |
| Cup of Ancient Civilizations | friendly | 1.0 | 2 |
| Four Nations' Cup | friendly | 1.0 | 2 |
| Marianas Cup | friendly | 1.0 | 2 |
| Mukuru 4 Nations | friendly | 1.0 | 2 |
| Trans-Tasman Cup | friendly | 1.0 | 2 |
| Benedikt Fontana Cup | friendly | 1.0 | 1 |
| CONIFA World Cup qualification | qualifier | 2.5 | 1 |
| CONMEBOL–UEFA Cup of Champions | friendly | 1.0 | 1 |
| ConIFA Challenger Cup | other_tournament | 1.0 | 1 |
| Copa Confraternidad | friendly | 1.0 | 1 |
| South Asian Super Cup | friendly | 1.0 | 1 |
| TIFOCO Tournament | friendly | 1.0 | 1 |
| The Other Final | friendly | 1.0 | 1 |
