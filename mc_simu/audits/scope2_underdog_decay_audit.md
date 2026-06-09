# Scope 2 audit — "Underdog stage decay" (đề xuất Sam 2026-05-27)

**Status: CLOSED — không implement.**
**Audit script:** [tests/mc_simu/audit_underdog_stage_decay.py](../tests/mc_simu/audit_underdog_stage_decay.py)
**Data scope:** 869 matches từ WC + Euro + Copa América (2000-2024)
**Fixes applied:** (1) PK shootout winner override tied scoreline, (2) snapshot Elo apples-to-apples với mc_simu

---

## TL;DR (revised sau snapshot audit)

Sam đề xuất: **underdog yếu dần qua các vòng** (do mệt, chấn thương, đối thủ chơi defensive).

**Kết quả 2 audits:**
- **LIVE Elo** (Elo update từng trận): cliff -14pp ở SF+Final, significant — nhưng KHÔNG apples-to-apples với mc_simu
- **SNAPSHOT Elo** (Elo pre-tournament, cách mc_simu thực sự dùng): pooled cliff giảm xuống -2pp, **không significant**
- **WC alone** vẫn -28pp với snapshot (n=7), nhưng quá nhỏ để build feature
- **Euro ngược pattern**: underdog overperform (Greece 2004, Italy 2012, Portugal 2016)

→ Pattern không phải decay monotone + Euro ngược + n quá nhỏ → **không implement**, watch WC2026.

---

## 1. Giả thuyết test

**Sam hypothesis:**
> "Càng về sau các đội underdog sẽ càng yếu hơn vì squad strength yếu về thể lực, chấn thương, thẻ đỏ + các đội mạnh khác có thể chơi more defensive."

**Pre-registered decision rule (đặt TRƯỚC khi nhìn data):**

Implement feature nếu thỏa cả 3:
1. **Pooled gap at SF+Final < -5pp** (underdog đá tệ hơn dự đoán ≥ 5%)
2. **95% CI excludes 0** (statistically significant)
3. **Monotone decay** Group ≥ R16+QF ≥ SF+Final (yếu dần đều như Sam mô tả)

Không bend rule khi data đẹp → tránh data mining.

---

## 2. Bug tìm được trong quá trình audit

### Issue: `results.csv` lưu score sau ET, không phải sau PK

21 trận KO WC 2002-2022 đi penalty shootout. Trong `results.csv`, các trận này hiện thị score **regulation/ET** (tie 0-0, 1-1, 2-2, 3-3), không phải kết quả thật sau PK.

**Ví dụ:**
| Trận | results.csv | Kết quả thật |
|---|---|---|
| 2022 Final Argentina vs France | 3-3 | Argentina won PK 4-2 |
| 2006 Final Italy vs France | 1-1 | Italy won PK 5-3 |
| 2018 R16 Russia vs Spain | 1-1 | Russia upset on PK |

**Impact của bug:** Script ban đầu coi mỗi tie là 0.5/0.5 → 21 trận bị mã hóa sai. Đặc biệt với SF+Final, một số trận đáng ra "underdog thua" thì bị tính thành "0.5 hòa".

**Fix:** Cross-reference với [shootouts.csv](../data/mc_simu/shootouts.csv) — có cột `winner` từ 1967. Resolve tie scores bằng PK winner.

→ **Bài học:** Đây là kiểu bug **integration-layer** (hai bảng đều đúng, nhưng nối lại sai). Phù hợp với chính sách [/deep-audit-bug](../.claude/skills/) audit.

---

## 3. Kết quả sau fix

### Pooled WC + Euro + Copa (gap ≥ 50 Elo, n=703)

| Vòng | n | Underdog actual W% | Elo dự đoán W% | Chênh lệch | 95% CI upper |
|---|---|---|---|---|---|
| Group | 534 | 30.9% | 27.0% | **+3.9pp** (overperform) | +7.2pp |
| R16+QF | 128 | 28.9% | 29.5% | -0.6pp (≈ đúng) | +7.2pp |
| **SF+Final** | **41** | **17.1%** | 31.3% | **-14.2pp** | **-2.7pp** (excludes 0) |

### Per-tournament SF+Final breakdown (gap ≥ 100)

| Giải | n | W | D | L | Actual W% | Expected W% | Chênh |
|---|---|---|---|---|---|---|---|
| **FIFA World Cup** | 6 | 0 | 0 | 6 | 0% | 30.6% | **-31pp** |
| **UEFA Euro** | 9 | 3 | 0 | 6 | 33.3% | 29.3% | **+4pp** |
| **Copa América** | 12 | 1 | 0 | 11 | 8.3% | 23.7% | **-15pp** |

→ **WC và Copa: underdog sụp đổ. Euro: ngược lại — underdog đá tốt hơn dự đoán.**

---

## 4. Concrete examples (cho Sam dễ hình dung)

### Các trận WC SF+Final mà underdog thua (gap ≥ 100, 6/6 LOSS)

| Năm | Vòng | Underdog (Elo) | Favorite (Elo) | Kết quả thật |
|---|---|---|---|---|
| 2002 | SF | South Korea (1859) | Germany (1982) | Germany thắng 1-0 |
| 2002 | SF | Turkey (1866) | Brazil (2029) | Brazil thắng 1-0 |
| 2010 | SF | Uruguay (1957) | Netherlands (2134) | Netherlands thắng 3-2 |
| 2014 | Final | Argentina (2124) | Germany (2240) | Germany thắng 1-0 (ET, Götze) |
| 2018 | Final | Croatia (2022) | France (2144) | France thắng 4-2 |
| 2022 | SF | Croatia (2015) | Argentina (2173) | Argentina thắng 3-0 |

### Các trận Euro nơi underdog **thắng** (3/9 — phản đề Sam)

| Năm | Vòng | Underdog (Elo) | Favorite (Elo) | Kết quả thật |
|---|---|---|---|---|
| 2004 | SF | **Greece (1834)** | Czech Republic (2019) | Greece thắng 1-0 (Greek miracle) |
| 2012 | SF | **Italy (1932)** | Germany (2126) | Italy thắng 2-1 (Balotelli double) |
| 2016 | Final | **Portugal (1954)** | France (2059) | Portugal thắng 1-0 ET (Eder goal) |

→ Đây là 3 ví dụ KINH ĐIỂN của underdog thắng SF/F. Nếu implement "stage decay" theo Sam, model sẽ **không bao giờ** dự đoán được những upset này.

---

## 5. Tại sao **KHÔNG implement** stage decay

### Lý do 1: Pattern không match Sam's mechanism

| Sam hypothesis | Data |
|---|---|
| Yếu dần đều (Group → R16 → QF → SF → F) | Group: **+3.9pp** (mạnh hơn dự đoán!) |
| Cause: fatigue cumulative | R16+QF: -0.6pp (đúng dự đoán) |
| | SF+Final: **-14.2pp** (sụp đột ngột) |

Pattern là **vách đá**, không phải **dốc đều**. Không có dấu hiệu cumulative fatigue ở R16/QF.

### Lý do 2: WC + Copa có cliff, Euro KHÔNG

Implement 1 stage-decay factor cho toàn bộ giải sẽ:
- **Đúng** cho WC + Copa SF/F (underdog yếu hơn dự đoán)
- **SAI** cho Euro SF/F (underdog đang đá tốt hơn dự đoán → bị shrink xuống thấp = worse calibration)

3 ví dụ Euro thắng (Greece 2004, Italy 2012, Portugal 2016) sẽ bị model under-predict. Polymarket / Kalshi sẽ price chúng cao hơn model → ta sẽ luôn bet vào favorite → thua những upset huyền thoại này.

### Lý do 3: Mechanism thật có thể là Elo bug, không phải fatigue

Pattern **+3.9pp ở Group → -14pp ở SF+Final** khớp với một bug khác của Elo, không phải Sam's fatigue:

> "Khi underdog thắng 2-3 trận upset trước đó, Elo của họ tăng **giả** rất nhanh. Đến SF/F, đội mang nhãn 'underdog' theo Elo hiện tại thật ra là **đội yếu đang đỉnh cao may mắn** — Elo overrate họ."

Nếu đúng, fix phải là **Elo regularization** (giảm K-factor cho hot-streak teams), không phải shrink "underdog strength". 2 fix khác nhau, mục tiêu khác nhau, codepath khác nhau.

### Lý do 4: Sample quá nhỏ để fit shape feature

| Vòng | n (gap ≥ 100) |
|---|---|
| Group | 410 |
| R16+QF | 90 |
| SF+Final | **27** |

n=27 cho SF+Final pooled. Per-tournament: WC=6, Euro=9, Copa=12. Bất kỳ feature shape nào (linear, exp, step) fit cho 27 điểm sẽ overfit.

### Lý do 5: Pre-registered monotone rule fail

Per rule đặt **trước** khi xem data:
- Pooled gap < -5pp at SF+Final → ✅ True (-14pp)
- 95% CI excludes 0 → ✅ True (upper -2.7pp)
- **Monotone Group ≥ R16+QF ≥ SF+Final → ❌ False** (Group +3.9, R16+QF -0.6, SF+Final -14)

Rule fail → respect rule → no implementation. Đây là để tránh **data mining** (cố tìm version của rule để pass).

---

## 5b. Critical follow-up — LIVE Elo vs SNAPSHOT Elo

Sau khi viết section 1-5, user (Duy Anh) đặt câu hỏi quan trọng: **audit dùng LIVE Elo (update theo từng trận), nhưng mc_simu thực ra dùng SNAPSHOT Elo (precomputed pre-tournament, không update trong simulation)**. Có lệch không?

Verify ở [simulator.py:6-17](../src/mc_simu/simulator.py#L6-L17):
> "48 × 47 = 2256 directed (team_a, team_b) KO pairings → scalar p_advance lookup [...] all predictions for a given MC run are PRE-COMPUTED ONCE"

→ Trong mc_simu predict mode, Elo là snapshot. Bug "Elo inflate trong giải" KHÔNG xảy ra ở forward simulation. Audit cũ (LIVE Elo) **không apples-to-apples** với mc_simu.

### Re-run với snapshot Elo

Đổi lookup: thay vì `elo_before(team, match_date)`, dùng `elo_before(team, tournament_start_date)` cho mọi trận trong giải đó.

### LIVE vs SNAPSHOT comparison (pooled WC + Euro + Copa, gap ≥ 50)

| Pool | LIVE miss | SNAPSHOT miss | Δ |
|---|---|---|---|
| Group | +3.9pp | +3.2pp | gần như giống |
| R16+QF | -0.6pp | +3.5pp | nhỉnh hơn |
| **SF+Final** | **-14.2pp (significant)** | **-2.0pp (KHÔNG significant)** | **giảm 12pp** |

→ **Phần lớn pooled cliff là Elo inflation artifact.** Khi snapshot, CI bao gồm 0.

### Per-tournament SF+Final (gap ≥ 100), SNAPSHOT Elo

| Giải | n | W | D | L | Actual | Expected | Miss |
|---|---|---|---|---|---|---|---|
| **FIFA World Cup** | 7 | **0** | 0 | 7 | **0%** | 28.2% | **-28.2pp** |
| **UEFA Euro** | 10 | 3 | 0 | 7 | 30.0% | 25.1% | +4.9pp |
| **Copa América** | 13 | 2 | 0 | 11 | 15.4% | 21.6% | **-6.2pp** |

### Findings

| Cái gì | Trước (LIVE) | Sau (SNAPSHOT) | Kết luận |
|---|---|---|---|
| Copa cliff | -15pp | -6pp | **Phần lớn là Elo artifact** — biến mất |
| Pooled significance | Có (excludes 0) | Không (bao gồm 0) | **Cliff overall không robust** |
| **WC cliff** | -31pp | **-28pp** (chỉ giảm nhẹ) | **Vẫn còn** — có component thật |

### Implication mới cho mc_simu

- **Pooled cliff** chủ yếu là artifact → mc_simu (snapshot) **không** chịu effect này phần lớn
- **WC-specific cliff** vẫn -28pp với snapshot Elo → mc_simu **có thể** over-predict underdog ở WC2026 SF+F khoảng 25-30%
- n=7 cho WC alone vẫn quá nhỏ để conclude
- MV blend (Scope 1) sẽ pull underdog xuống nếu MV thấp → mitigation phần nào

### Quyết định không đổi

- Vẫn **CLOSE scope 2** — pooled effect không robust khi snapshot
- WC-specific n=7 không đủ cho feature reliable
- Watch WC2026 SF+F để add data point — sẽ có thêm 3 trận (2 SF + 1 F)

---

## 6. Watch list cho WC2026

Sau WC2026 mình sẽ có thêm 1 data point (8 KO matches: R16=8, QF=4, SF=2, F=1 = 15 trận KO). Re-run audit sau giải:

1. Nếu WC2026 SF+Final cũng cho underdog 0/2 hoặc 0/3 → strengthen WC-specific cliff hypothesis
2. Nếu Euro 2028 (24 teams) tiếp tục cho 1-2 upset → confirm Euro ≠ WC pattern
3. Nếu pooled 4-tournament SF+Final vẫn -10pp+ với CI excludes 0 + per-tournament consistency → re-open scope

**Threshold để re-open:** SF+Final pooled gap < -10pp AND per-tournament đều cùng dấu (cả 3 đều negative).

---

## 7. Methodology (cho reproducibility)

```bash
# WC-only audit
python tests/mc_simu/audit_underdog_stage_decay.py

# Multi-tournament audit (inline script, chưa committed)
# Xem section "Multi-tournament code" trong commit history hoặc rerun từ context.
```

**Data sources:**
- `data/mc_simu/results.csv` — match outcomes 1872-present
- `data/mc_simu/elo_history.csv` — per-team Elo theo từng trận
- `data/mc_simu/shootouts.csv` — PK winner cho 247 trận tied có ghi nhận

**Tournament formats (count-based stage assignment):**

| Tournament | Format | Stage counts |
|---|---|---|
| FIFA World Cup 2002-2022 | 32 teams | 48G + 8R16 + 4QF + 2SF + 1×3rd + 1F |
| UEFA Euro 2000-2012 | 16 teams | 24G + 4QF + 2SF + 1F (no R16, no 3rd) |
| UEFA Euro 2016/2021/2024 | 24 teams | 36G + 8R16 + 4QF + 2SF + 1F (no 3rd) |
| Copa América 2001-2015 | 12 teams | 18G + 4QF + 2SF + 1×3rd + 1F |
| Copa América 2016/2024 | 16 teams | 24G + 4QF + 2SF + 1×3rd + 1F |
| Copa América 2021 | 10 teams | 20G + 4QF + 2SF + 1×3rd + 1F |

3rd-place matches **dropped** (not a "stage decay" signal — cả 2 đội đã thua SF, không có stake bracket).

**Elo lookup:** Per-team binary search trên `elo_history`, tra `rating_after` của trận trước ngày match đang xét.

**2-way win share:** Tránh thiên vị draw bias bằng cách dùng W=1, D=0.5, L=0 cho cả actual và Elo expected. PK winners override tied scoreline cho KO matches.

---

## 8. Tóm lại 1 đoạn cho Sam (revised sau snapshot audit)

> Test 869 matches WC + Euro + Copa America (2000-2024) với 2 fix: (1) PK shootout winner override tied scoreline, (2) snapshot Elo apples-to-apples với mc_simu. Kết quả: **pooled cliff biến mất** với snapshot Elo (-14pp → -2pp, không significant) — phần lớn là Elo inflation artifact. **WC cliff vẫn còn** (-28pp với snapshot, n=7) nhưng quá nhỏ để build feature. Euro thì underdog overperform ở SF+F (3/10 thắng — Greece 2004, Italy 2012, Portugal 2016). Không implement stage decay vì shape không monotone + Euro ngược pattern + n quá nhỏ. mc_simu (snapshot Elo) chỉ có residual risk cho WC2026 SF+F — MV blend (Scope 1) sẽ mitigate. Watch WC2026.
