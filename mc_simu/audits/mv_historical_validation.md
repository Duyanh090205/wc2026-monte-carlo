# MV historical validation + absolute-vs-relative — Task 1 & 2 findings

**Date:** 2026-06-09
**Trigger:** Sam — (1) model ELO+MV có over/under-value underdog/favorite theo quy luật hay random, so ELO-only; (2) xem relative % thay vì chỉ absolute pp.
**Liên quan:** [scope3_underdog_bias_and_rerun_plan.md](scope3_underdog_bias_and_rerun_plan.md) (Task 1 plan), [before_world_cup_audit.md](before_world_cup_audit.md) (favorite-longshot, champion penalty rejected).

---

## 0. TL;DR

1. **MV = sắp xếp lại theo squad-value, CÓ quy luật (không random):** ΔMV vs squad-value gap (z_MV−z_Elo) Spearman **+0.94** (movers); vs favorite-rank chỉ +0.34. Dìm Elo-cao-squad-rẻ (Spain −4.9pp, Argentina −4.6pp), nâng MV-cao brand (England/Germany/France). **Đuôi underdog Δ≈0.** Star bonus mới nghiêng top favorite (+1.27pp).
2. **MV KHÔNG cải thiện độ chính xác (validated trên lịch sử):** per-match Brier ELO 0.5610 vs ELO+MV 0.5636 (Δ+0.0026, z+0.39, không significant); càng tăng alpha càng tệ (α=1.0 → +0.019). Tournament: hạng nhà vô địch thật ELO 2.50 = ELO+MV 2.50; P(champion thật) ELO 15.3% > ELO+MV 13.6%.
3. **MV chỉ làm model giống market hơn ~1/3** (JSD 0.0635→0.0374 = −41%; L1 48.3pp→32.0pp = −34%). → **MV = market-fitting prior, KHÔNG phải tăng-chính-xác.** Đúng pattern kỷ luật: signal có lý nhưng fail historical-accuracy bar.
4. **Absolute vs relative đo "đội lệch nhất" hoàn toàn khác:** absolute → favorite ở 2 đầu; relative → đuôi/longshot ở 2 đầu. Relative phơi mispricing đuôi mà absolute giấu.

---

## 1. Dữ liệu MV lịch sử (free, point-in-time)

Trước đây không validate MV trên lịch sử được vì **chỉ có MV hiện tại** (transfermarkt_mv_wc2026.csv). Đã build pipeline lấy **squad MV point-in-time** mỗi kỳ WC, miễn phí, không auth:

- **Roster:** [jfjelstul/worldcup](https://github.com/jfjelstul/worldcup) `squads.csv` — **chỉ World Cup** (không Euro/Copa).
- **Valuations:** [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) qua Cloudflare R2 (`player_valuations` dated từ 2000 + `players` + `games`).
- **Ghép:** fuzzy-match tên (lọc theo quốc tịch) → player_id → giá trị **gần nhất ≤ ngày khai mạc**. Match major teams 96–100%.
- **Sanity:** France 2018 = **€1.08bn** (khớp giá TM thật giữa-2018; nếu lấy nhầm giá hiện tại phải ~€1.5bn) → point-in-time đúng.
- **Phủ:** WC2010/14/18/22 (TM valuations đáng tin ~2008+; WC2002/06 thưa, bỏ). → `build_historical_mv.py`, output `data/mc_simu/historical_squad_mv.csv`.
- **Caveat:** đội yếu undercount valuation (Saudi/Panama vài cầu thủ) → MV thấp hơn thực, nhưng đuôi ~0 nên ít ảnh hưởng.

**Vì sao chỉ ~253 trận test được:** chỉ 4 kỳ WC finals × ~64 trận. Ràng buộc = MV: jfjelstul chỉ có WC; vòng loại (~200 đội) không có squad MV. Muốn ~2× mẫu → cần roster Euro/Copa (nguồn khác).

---

## 2. Task 1 — MV systematic? + over/under-value

### 2.1 MV là squad-value reorder (`elo_vs_mv_bias.py`, figure `elo_vs_mv_bias.png`)
- 3 config cùng D=1400/diag=0.20/eloratings/n=10k: ELO-only (α0) → ELO+MV (α0.5) → final (α0.5,X15).
- ΔMV per-team: Spain −4.89, Argentina −4.60 (dìm) | England +1.77, Germany +1.58, Portugal +1.57, Brazil +0.83, France +0.60 (nâng).
- Tier ΔMV: Favorite(top6) **−0.79pp**, Contender(7–14) +0.49, Dark(15–26) +0.03, Longshot(27+) +0.02. Δfinal Favorite **+1.27pp** (star).
- Spearman(ΔMV, squad-value gap) = +0.94 movers / +0.68 all; vs favorite-rank +0.34. → **systematic theo squad-value, không random; underdog tail không đổi.**

### 2.2 Calibration vs kết quả thật
- **Per-match 253 trận WC** (`validate_mv_history.py`, figure `match_reliability_elo_vs_mv.png`): điểm **trên** đường chéo đều khắp → model **under-value % thắng / over-value hòa** (do diag=0.20), **KHÔNG** bias favorite/underdog. ELO ≡ ELO+MV (Brier 0.207 ≈ 0.208).
- **Mẫu rộng (16k finals+quals, ELO-only):** under-confidence nhẹ (favorite hơi under, underdog hơi over). Khác sample (nặng vòng loại).
- **Champion-level lịch sử rộng** (before_world_cup): top favorite hơi over (−4.4pp). Không mâu thuẫn — khác cấp (trận vs danh hiệu; KO variance làm favorite flop ở cấp giải).

### 2.3 MV có tăng chính xác? — KHÔNG (`validate_mv_history.py`, `validate_mv_tournament.py`)
Per-match Brier (n=253), alpha sweep:
| α | Brier ELO | Brier ELO+MV | Δ | z |
|--:|--:|--:|--:|--:|
| 0.3 | 0.5610 | 0.5609 | −0.0001 | −0.03 |
| 0.5 | 0.5610 | 0.5636 | +0.0026 | +0.39 |
| 0.7 | 0.5610 | 0.5686 | +0.0076 | +0.81 |
| 1.0 | 0.5610 | 0.5802 | +0.0191 | +1.44 |
→ Càng nặng MV càng tệ (monotonic), không bao giờ significant theo hướng có lợi. MV beats ELO chỉ 45% trận.

Tournament (figure `mv_tournament_calibration.png`): hạng nhà vô địch thật ELO **2.50** = ELO+MV **2.50**; P(champion) ELO 15.3% > ELO+MV 13.6%. Per-edition: 2010 #1/#1, 2014 #2/#3, 2018 #5/#4 (MV giúp — France MV cao), 2022 #2/#2 (MV hại — Argentina MV thấp). Net wash.

### 2.4 Market-closeness (cái MV thực sự cải thiện)
ELO-only vs ELO+MV (WC2026, vs PM+Kalshi): JSD **0.0635 → 0.0374** (−41%); L1 **48.3pp → 32.0pp** (−34%). → MV kéo gần market đáng kể.

**Verdict Task 1:** MV reorder theo squad-value (systematic) nhưng **không validate được là tăng chính xác** — chỉ là **market-prior**. Giữ MV vì mục tiêu "model-the-market" (Sam), KHÔNG vì độ chính xác.

---

## 3. Task 2 — Absolute vs relative

- Cắm relative % vào `wc2026_vs_multi.py` (công thức Sam `(market−model)/model`). Median |rel| ~57–71%/nguồn vs mean |abs edge| ~1pp.
- `plot_model_vs_market.py`: `--compare` (2-panel abs vs rel), `--table` (bảng số), default (bar absolute). Convention compare/table: `(model/market−1)`, + = model trên market.
- **Reordering:** absolute → favorite ở 2 đầu (Spain +3.87pp, Portugal −2.85pp); relative → đuôi ở 2 đầu (Croatia +120% … Jordan −100%), favorite nén vào giữa (Spain +24%). Relative phơi mispricing đuôi (model ép longshot →0, market sàn ~0.1%).
- **Phạm vi:** model-vs-market chỉ làm được **present (live)**. **Quá khứ KHÔNG khả thi** — probe xác nhận không có odds outright lịch sử (WC2010/14/18 không có market; WC2022 Polymarket mất giá pre-tournament, history chỉ lùi tới ~13/12/2022).

---

## 4. Artifacts

**Scripts (tracked):** `src/mc_simu/elo_vs_mv_bias.py`, `build_historical_mv.py`, `validate_mv_history.py`, `validate_mv_tournament.py`, `plot_model_vs_market.py`; sửa `wc2026_vs_multi.py` (relative cols).
**Figures:** `mc_simu/audits/elo_vs_mv_bias.png`, `mv_tournament_calibration.png`, `match_reliability_elo_vs_mv.png`; `mc_simu/figures/wc2026_model_vs_market{,_abs_vs_rel,_table}.png`.
**Data (gitignored):** `data/mc_simu/historical_squad_mv.csv`, `data/mc_simu/cache/tmdata/` (raw dcaribou).

## 5. Mở rộng (nếu muốn)
- Tăng power: roster Euro/Copa → ~2× mẫu per-match.
- Tinh chỉnh draw inflation (diag) — bias over-hòa/under-thắng ở mẫu WC (độc lập MV).
