# MC Simu Phase 0 Audit Report

Generated: 2026-06-10 19:10:40 UTC
Spec version: mc_simu.md v3 (May 14, 2026)
Repo HEAD: see `git rev-parse HEAD`

## Preflight checks (§1.1)
- 0.1 DB connectivity: NOT RUN (opt-in via --with-db; v1 = file-only)
- 0.2 Required tables: NOT RUN — pre-verified offline: repo has 7 tables (5 spec + api_call_log, live_price_cache)
- 0.3 Recent fair_odds writes: NOT RUN
- 0.4 entity_map.json: PASS (preflight.py lives in the trading-engine repo)
- 0.5 fair_odds schema: NOT RUN — pre-verified offline: actual is superset incl new consensus_prob col
- v1 standalone path — all DB checks deferred to v2 (production integration)

## Data quality checks (§1.3)
- Check 1 — Match history completeness: ✓ PASS — date range 1872-11-30 to 2026-06-27 (49,329 rows). Per-tournament coverage:
      FIFA World Cup 2002: 64/64 (100%)
      FIFA World Cup 2006: 64/64 (100%)
      FIFA World Cup 2010: 64/64 (100%)
      FIFA World Cup 2014: 64/64 (100%)
      FIFA World Cup 2018: 64/64 (100%)
      FIFA World Cup 2022: 64/64 (100%)
      UEFA Euro 2004: 31/31 (100%)
      UEFA Euro 2008: 31/31 (100%)
      UEFA Euro 2012: 31/31 (100%)
      UEFA Euro 2016: 51/51 (100%)
      UEFA Euro 2020: 51/51 (100%)
      UEFA Euro 2024: 51/51 (100%)
- Check 2 — Tournament types: ✓ PASS — 6/6 categories: world_cup_final=1036, continental_final=3391, qualifier=15928, nations_league=1080, other_tournament=3908, friendly=23986
- Check 3 — Neutral flag consistency: ✓ PASS — 5/5 hosts validated:
      FIFA World Cup 2014 (host=Brazil): host_matches_neutral=0 (expect 0), non_host_matches=57, non-neutral_among_them=0 (expect 0)
      FIFA World Cup 2018 (host=Russia): host_matches_neutral=0 (expect 0), non_host_matches=59, non-neutral_among_them=0 (expect 0)
      FIFA World Cup 2022 (host=Qatar): host_matches_neutral=0 (expect 0), non_host_matches=61, non-neutral_among_them=0 (expect 0)
      UEFA Euro 2016 (host=France): host_matches_neutral=0 (expect 0), non_host_matches=44, non-neutral_among_them=0 (expect 0)
      UEFA Euro 2024 (host=Germany): host_matches_neutral=0 (expect 0), non_host_matches=46, non-neutral_among_them=0 (expect 0)
- Check 4 — Euro 2020 attendance: ✓ PASS — merged 51/51 Euro 2020 matches; 41 low-attendance (<50%)
- Check 5 — ClubElo coverage: ⚠ WARN — 19 teams audited; mean coverage 55.0% (381/693 players)
    15/19 teams below 85%: ['Argentina=69%', 'Bosnia and Herzegovina=77%', 'Brazil=58%', 'Colombia=42%', 'Haiti=65%', 'Japan=73%', 'Jordan=3%', 'Mexico=38%'] ...
- Check 6 — WC2026 structure: ✓ PASS — 104 fixtures (72 group + 32 KO); 12 groups × 6 matches; 495 R32 mappings (C(12,8))

## Files generated
- data/mc_simu/results.csv (raw)
- data/mc_simu/matches_1998_2026.csv (filtered, with tournament_type)

## Hard-stop status: ✅ PASS — ready for Phase 1 (1 WARN, non-blocking per spec §9 fallback)
