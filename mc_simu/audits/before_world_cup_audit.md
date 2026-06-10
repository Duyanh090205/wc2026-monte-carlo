# Before-World-Cup Audit — WC2026 model bias investigation

**Date:** 2026-06-03 (WC2026 kickoff 2026-06-11)
**Model under audit:** Final = MV blend α=0.5 + Star bonus X=15 + Phase-1 defaults (D=1400, diag=0.20, α_HFA=0.27, β=0.09), Elo source = eloratings, n=100k. (scope1 LOCKED.)
**Market reference:** LIVE fixed APIs — Polymarket gamma (event `world-cup-winner`/id 30615, reads `groupItemTitle` → Panama correct) + Kalshi (`KXMENWORLDCUP-26`, ask-fallback) + FairLine `/fair-odds`. See [[project_live_engine_pm_kalshi_bugs]].
**Trigger:** Sam — model "khá oke" nhưng **Spain overrated** (model ưu ái Spain do Elo cao + MV cao, market không đồng ý). Mục tiêu: hiểu favorite over-rating, đo bias, KHÔNG over-engineer.

---

## 0. TL;DR (đọc cái này là đủ)

- **Spain bị model over ~+2…+3.8pp so với cả 3 market** (PM/Kalshi/FairLine). Argentina cũng over (~+1.5…+2.5).
- Đã loại 3 cách "sửa Spain": **(i) bỏ double-count young-star** (làm JSD xấu + Spain lật under), **(ii) hạ star X toàn cục** (collateral lên France/England), **(iii) xG** (số nền Spain vốn ELITE → xG không phải lever).
- Cơ chế thật market định giá mà model thiếu: **regression của đương kim vô địch / "không tin vô địch 2 lần"** — Spain (ĐKVĐ Euro 2024) + Argentina (ĐKVĐ WC 2022 + Copa 2024) đúng profile lịch sử hay flop ở WC.
- Áp discount −30 Elo lên Spain+Argentina khớp market + cải thiện JSD — **nhưng historical validation BÁC**: fail consistency (Copa null), specificity marginal, độ lớn lịch sử (~100 Elo) >< market (~30) → **30 là market-fit, KHÔNG hợp lệ**.
- **Quyết định: KHÔNG implement penalty đương-kim-VĐ** (fragile, đúng số phận [scope2](scope2_underdog_decay_audit.md)). Effect robust duy nhất = **favorite-longshot bias chung** (mọi top favorite over ~−4.4pp) nhưng **nhẹ** (~40 Elo, Brier gain <1%).
- **Khuyến nghị:** để model nguyên; với top favorite thì **nghiêng về market consensus** (model-the-market). Theo dõi Spain/Argentina ở WC2026 thật để thêm data point.

---

## 1. Symptom — Spain maxes mọi trục model có

| Trục | Spain | Hạng/48 |
|---|---|---|
| Elo (eloratings) | 2165 | **#1** |
| MV (Transfermarkt) | €1.27bn | #3 |
| Stars | 8 | đồng #1 |

Edge model − market (final X=15):
| | Spain | Argentina |
|---|---|---|
| vs PM (live) | **+3.7pp** | +1.5pp |
| vs Kalshi | +2.3pp | +1.4pp |
| vs FairLine | +3.1pp | +1.5pp |

→ Model **không có cách nào KHÔNG đẩy Spain lên đỉnh** — nó top cả 3 tín hiệu nó biết. Sửa = phải **thêm tín hiệu model thiếu**, không phải vặn cái đã có. Bảng đầy đủ 48 đội: 11 OVER / 11 UNDER / 26 ~match. Pattern theo tầng: **Spain over; elite brand (France/Portugal/Brazil) UNDER; dark-horse tầm trung (Argentina/Netherlands/Turkey/Uruguay/Croatia) OVER; longshot đáy hơi under**.

---

## 2. Các giả thuyết đã test (mỗi cái có verdict)

### 2.1 Double-count young star (MV blend + star bonus đếm 2 lần) — REJECT
Young star (Yamal €200m…) được tính ở **cả** MV blend (current MV cao) **lẫn** star bonus (tiêu chí (a) current MV ≥ €70m). Aging legend (Messi/Ronaldo) thì vào qua (b) peak MV / (c) Ballon d'Or → KHÔNG bị bỏ nếu loại (a).

Test (cached market, n=30k):
| Config | Spain edge | JSD vs PM |
|---|--:|--:|
| Final X=15 | +2.9 | 0.0241 |
| No-star X=0 | −0.5 | 0.0333 |
| Half X=7.5 | +1.3 | 0.0251 |
| Legend-only X=15 | −0.8 | 0.0329 |

**Verdict:** KHÔNG phải đếm thừa thuần — bỏ young-star bonus làm **JSD xấu +37%** (star bonus khôi phục "talent concentration" mà MV-tổng-log-z nén mất) và **Spain lật under** (squad Spain ~0 aging legend → mất hết bonus). Reject.

### 2.2 Hạ star X toàn cục (15→7.5) — REJECT (collateral)
(Live PM, n=100k — plot đã xoá, thí nghiệm bị bác.)
Spain +3.83→+2.21 (tốt) **nhưng** France −1.25→−2.71, England +0.66→−0.79 (xấu — chúng cũng 8 sao). X là **nút toàn cục, không tách được Spain** khỏi các đội nhiều sao khác. Reject.

### 2.3 "Market định giá gì ta thiếu" — Spain KHÔNG hollow, xG KHÔNG phải lever
Recent form (since 2024-06): **Spain 21-6-0 bất bại, GF/g 2.74, GA/g 0.81, CS 52%** — số nền **elite**. Argentina 19-3-3, GA/g 0.44, CS 64%. → "kết quả ảo / gap xG" **không** giải thích discount của market. **xG không gỡ được Spain.** (Repo cũng không có pipeline xG; FiveThirtyEight SPI đã ngừng 2023.)

### 2.4 Reigning-champion / "không tin vô địch 2 lần" — pattern mạnh (descriptive)
**Đương kim VĐ World Cup → kỳ WC kế:**
| WC | ĐKVĐ | Elo rank | Kết quả |
|---|---|--:|---|
| 2002 | France | #1 | OUT vòng bảng (0 bàn) |
| 2006 | Brazil | #1 | QF |
| 2010 | Italy | #6 | OUT vòng bảng |
| 2014 | Spain | #2 | OUT vòng bảng |
| 2018 | Germany | #2 | OUT vòng bảng |
| 2022 | France | #3 | Final |
| **2026** | **Argentina** | **#2** | ? |

**ĐKVĐ Euro → kỳ WC kế:** France'02 group(0 bàn), Greece'06 DNQ, Spain'10 **WON**, Spain'14 group(thủng 7), Portugal'18 R16, Italy'22 DNQ, **Spain'26 #1**.

**Pooled (18 reigning champ WC+Euro+Copa, 2002-2022):** chỉ **3/18 vào SF+**, **12/18 finish R16-hoặc-tệ-hơn**. → Cả Spain (Euro) lẫn Argentina (WC+Copa) = ĐKVĐ vào giải Elo #1/#2 = đúng profile hay flop. Một cơ chế giải thích CẢ HAI đội bị over.

### 2.5 Lượng hoá + áp discount −30 Elo (Spain+Argentina) — đẹp nhưng CHƯA hợp lệ
Mức ĐKVĐ đá dưới Elo: **−12.5pp win-share (~−130 Elo) overall**, nhưng **TRỌN ở vòng bảng** (−19.9pp / ~−215 Elo); **KO = +1.0pp (zero)**. Lưỡng cực: France'02 −59, Germany'18 −49, Italy'10 −47, Spain'14 −43 (sảy chân group); đi sâu thì +.

Áp −30 Elo (live PM) — plot đã xoá (thí nghiệm bị bác):
| | base X=15 | +disc −30 |
|---|--:|--:|
| Spain | +3.67 | **+0.84** |
| Argentina | +1.54 | **−0.28** |
| France | −1.33 | −0.51 |
| Brazil | −1.67 | −1.32 |
| Portugal | −1.93 | −1.44 |
| JSD vs PM | 0.0199 | **0.0185** |

"Một mũi tên hai đích": hạ favorite-over + nâng brand-under + JSD tốt hơn. **Nhưng 30 là calibrate theo market.**

### 2.6 Historical validation (test quyết định) — **BÁC implement**
Pre-registered: hợp lệ nếu (champion under hơn favorite ĐẶC THÙ) AND (cải thiện Brier) AND (nhất quán WC/Euro/Copa).

**Specificity** (mọi edition WC+Euro+Copa, win-share gap):
| Nhóm | n | gap | 95% CI |
|---|--:|--:|---|
| Reigning champions | 94 | −10.6pp | [−19.1, −2.0] |
| Top-3 favorites (non-champ) | 2389 | −4.4pp | [−5.9, −2.9] |

→ **Mọi top favorite over-rate** (−4.4pp, rất chắc). Champion under hơn nhưng **CI chồng −4.4** → không tách được "champion riêng".

**Per-competition (champions):** WC −26.4pp (n=24) ✓ · Euro −17.5pp (n=25) ✓ · **Copa +1.7pp (n=45) ✗ null** → **FAIL consistency**.

**Brier:** champion-optimal **P*=100 Elo** (0.1885→0.1770), 100 Elo làm **xấu** Brier favorite (0.1414→0.1451); favorite optimal chỉ ~40. → Lịch sử đòi ~100, market chỉ ~30 → **30 = market-fit**, vi phạm "không tune theo market".

**VERDICT: 2/3 tiêu chí fail → KHÔNG implement penalty champion-specific.** Đúng pattern [scope2](scope2_underdog_decay_audit.md): signal thật nhưng fragile + n nhỏ + không nhất quán → respect rule, không over-engineer.

---

## 3. Kết luận & khuyến nghị

1. Spain over-rating **KHÔNG sửa được** bằng: star-X tweak, bỏ double-count young-star, hay xG. Đừng đụng các hướng này.
2. Effect **robust duy nhất** = **favorite-longshot bias chung** (top favorite over ~−4.4pp, n=2389) — chính là trục Task 1. Nhưng **nhẹ** (~40 Elo, Brier gain <1%).
3. **Khuyến nghị (kỷ luật):** để model nguyên; coi Spain/Argentina over là *known*; với top favorite **nghiêng về market consensus** (model-the-market). KHÔNG ship champion penalty.
4. Tuỳ chọn (nếu muốn động): general favorite shrinkage ~40 Elo validate-trên-Brier — robust nhưng lời marginal.

---

## 4. Artifacts
- Plots: đã xoá (2 hình thí nghiệm bị bác — X=15 vs X=7.5, champ discount −30); số liệu giữ trong §2.2 / §2.5.
- Model output: `data/mc_simu/phase3_baselines/wc2026_final_mv_star_vs_market.csv` (48 đội + 3 market)
- Reproduce model: `tune_to_market.py --D-grid 1400 --diag-grid 0.20 --n 100000 --mv-blend-alpha 0.5 --star-bonus-X 15`
- Validation/test scripts: ad-hoc (chạy trong session, logic ghi ở §2.6) — historical từ `results.csv` + `elo_history.csv` + `shootouts.csv`.

---

## 5. Watch list — sau khi WC2026 đá xong
- **Spain + Argentina (ĐKVĐ):** có sảy chân vòng bảng như pattern lịch sử không? → thêm 1 data point cho champion hypothesis.
  - Đi sâu (SF+) → model đúng, over là alpha. Out sớm → champion penalty được củng cố.
- Re-run validation với data WC2026: nếu pooled SF+Final của ĐKVĐ tiếp tục âm + nhất quán cả Copa → mới re-open.
- Favorite-longshot bias: re-measure vs realized outcomes sau giải.
