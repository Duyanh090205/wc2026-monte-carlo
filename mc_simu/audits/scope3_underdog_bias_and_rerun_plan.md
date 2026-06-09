# Scope 3 plan (DRAFT) — Underdog bias analysis + stage-based daily rerun

**Status:** Task 1 phần lớn DONE (Reference B + favorite-longshot); Task 2 chưa build (plan locked).
**Nguồn:** Sam meeting (model hiện "khá oke", Spain overrated, lo over-engineer).
**Liên quan:** [before_world_cup_audit.md](before_world_cup_audit.md) (investigation Spain over-rating — KẾT QUẢ CHÍNH của Task 1 Reference B), [scope2_underdog_decay_audit.md](scope2_underdog_decay_audit.md) (closed), [scope1_mv_star_implementation.md](scope1_mv_star_implementation.md) (MV blend).

> ⏳ Phần đánh dấu **[DR]** phụ thuộc kết quả deep-research (favorite–longshot bias methodology) — đang chạy ngoài (Gemini/ChatGPT). Điền sau.

---

## PROGRESS LOG (cập nhật 2026-06-03)

### Task 1 — Underdog/favorite bias: **phần lớn DONE**
- ✅ **Reference B (hiện tại vs market)** — đã làm sâu, ghi ở [before_world_cup_audit.md](before_world_cup_audit.md):
  - Bảng đầy đủ 48 đội over/under vs live PM/Kalshi (fixed APIs): 11 OVER / 11 UNDER / 26 ~match.
  - Pattern theo tầng: Spain over; elite brand (France/Portugal/Brazil) under; dark-horse tầm trung over; longshot đáy hơi under.
  - Spain over-rating điều tra trọn vẹn → loại star-X/double-count/xG → cơ chế = reigning-champion regression.
  - **Lưu ý method:** bucket quartile-mean **làm loãng** tín hiệu (Q1: Spain over vs brand under triệt tiêu). Bias thật ở mức per-team/per-tier, không phải trung bình quartile.
- ✅ **Reference A (lịch sử, calibration)** — đã có evidence chính: **top-3 favorite over-rate −4.4pp (n=2389)** = favorite-longshot bias robust; reigning-champ −10.6pp nhưng fragile (Copa fail). Verdict: **không implement correction** (kỷ luật như scope2).
- ⏳ **Còn lại:** (1) tích hợp deep-research [DR] khi về (đối chiếu methodology + magnitude với literature); (2) viết Reference A đầy đủ với calibration curve bucketed nếu Sam cần con số formal; (3) re-bucket cách khác (elite vs dark-horse) vì quartile-mean loãng.

### Task 2 — Stage-based daily rerun: **CHƯA build** (plan §2 dưới)
- Build item chính = **conditioning harness** (lock trận đã đá, re-sim phần còn lại). WC2026 đá 11/06 → cần sớm nếu muốn chạy live.
- Next action cụ thể: viết design doc conditioning harness (§2.2) → implement → Morocco backtest (§2.4) → sentiment playbook (§2.5).

---

## 0. Câu trả lời đã chốt (định hướng plan)

| Quyết định | Chốt |
|---|---|
| Reference đo bias | **Cả 2**: lịch sử = vs realized outcomes (calibration); hiện tại = vs market. Market gồm **PM + Kalshi + sportsbook devigged consensus** |
| Định nghĩa underdog | **Bucket theo rank title-probability** của model (vd quartile/decile) |
| "Sentiment more than data driven" (Sam) | = (a) **playbook xử lý sự kiện bất thường** (underdog thắng favorite → odds phản ứng ra sao) + (b) **quy tắc update sau mỗi trận có guardrail** chống lệch quá |
| Trình tự | **Plan-first, chốt rồi mới build.** Measurement-first + pre-registered bar (giữ kỷ luật như Scope 2) |

**Ràng buộc dữ liệu (từ map nguồn odds):**
- **KHÔNG có historical outright odds** cho giải cũ (Wayback chỉ ~2025+; `data/fair_odds_history/` rỗng). → *"Qua các mùa"* chỉ calibrate được **vs realized outcomes**, không vs market.
- *Hiện tại* (WC2026): có đủ market — Polymarket gamma, Kalshi, và **FairLine `/fair-odds`** (Shin/Power devigged consensus per team, `GET /events/world_cup_2026/fair-odds`).
- Simulator **chưa có chế độ conditioning** trên kết quả đã đá (`run_mc_simu` chỉ có `--n/--seed`; mọi pairing precompute 1 lần) → đây là **build item cốt lõi của Task 2**.

---

## 1. TASK 1 — Underdog bias analysis

### 1.1 Mục tiêu
Định lượng model đang **bias bao nhiêu** trên nhóm đội yếu (underdog), (A) qua các mùa vs kết quả thật, (B) hiện tại vs market — và xác định có **edge** ở tail dưới không. Phụ phẩm: chẩn đoán luôn **Spain overrated** (rơi ra từ bucket trên cùng).

### 1.2 Định nghĩa underdog (bucket theo rank title-prob)
- Sắp xếp 48 đội theo `champion_prob` của model → chia bucket (đề xuất: **quartile Q1..Q4**, hoặc decile nếu đủ mẫu).
- "Underdog" = các bucket dưới (Q3–Q4 / bottom deciles). "Favorite" = bucket trên.
- Per-match phụ trợ: vẫn giữ trục Elo-gap (như Scope 2) để cross-check.

### 1.3 Reference A — Historical calibration (vs realized outcomes)
- **Data:** `data/mc_simu/results.csv` + replay (`historical_replay.py`, `phase3_baselines/`). Coverage ~869 matches WC+Euro+Copa 2000–2024 (đã dùng ở Scope 2). Outright: replay champion-prob per edition vs nhà vô địch thật.
- **Method [DR]:** reliability/calibration curve theo **predicted-prob bucket**; Brier-score decomposition (reliability/resolution); pooled + bootstrap CI; đo *signed* bias mỗi bucket (model over/under so với tần suất thật). Phương pháp chính xác (cách bin, min sample, test thống kê) **chờ deep-research xác nhận**.
- **Output:** bảng "predicted-prob bucket × actual frequency × bias (pp) × CI", + plot calibration. Câu trả lời: *tail dưới của model có lệch hệ thống không, lệch bao nhiêu.*

### 1.4 Reference B — Present vs market (WC2026)
- **Model:** `mc_wc2026_*` champion probs (regenerate local).
- **Market (3 nguồn):**
  - Polymarket gamma (`world-cup-winner`, đã fix Panama/Team Z).
  - Kalshi (`KXMENWORLDCUP-26`).
  - **Sportsbook devigged consensus** — FairLine `/events/world_cup_2026/fair-odds` (Shin + Power, `fair_odds.py::process_market`).
- **Method:** model-vs-market scatter theo team đã rank; **signed divergence theo bucket** → đo bias dấu + độ lớn ở tail dưới; flag nơi model < market (under-rate underdog → có thể là edge ngược) vs model > market.
- **Spain:** nằm bucket trên → bảng sẽ chỉ rõ Spain lệch market bao nhiêu so với phần còn lại của top bucket (one-off hay favorite-inflation hệ thống).

### 1.5 Deliverables Task 1
1. Bảng bias-by-rank (A: vs realized; B: vs market 3 nguồn).
2. Plot calibration (A) + model-vs-market ranked (B).
3. 1 đoạn kết luận: *hướng + độ lớn bias underdog*, có edge không, Spain ra sao.
4. **Pre-registered bar** trước khi đề xuất bất kỳ correction nào (chống over-engineer): vd "chỉ đề xuất shrink/recalibrate nếu bias bucket dưới > X pp AND CI excludes 0 AND nhất quán nhiều mùa".

---

## 2. TASK 2 — Stage-based daily rerun

### 2.1 Mục tiêu
Trong lúc WC2026 diễn ra (11/06–19/07/2026), **rerun simulation mỗi ngày**, conditioning trên kết quả đã đá, và quan sát odds dịch chuyển ra sao — đặc biệt khi **underdog đi sâu** (ví dụ Morocco WC2022).

### 2.2 Component 1 — Conditioning harness *(build item chính)*
- **Gap:** harness hiện sim full giải từ group fixtures, không nhận "kết quả đã có".
- **Thiết kế:** input = danh sách trận đã đá (group: tỉ số → standings từng phần; KO: winner → lock nhánh); chỉ sample phần còn lại; tái dùng cơ chế precompute pairing.
- **Output:** champion/advancement prob *có điều kiện* tại mỗi as-of date.

### 2.3 Component 2 — Daily cadence
- Lịch chạy hằng ngày trong giải; snapshot prob theo ngày → time-series per team.
- Output format **[hỏi Sam]**: dashboard? CSV time-series? alert khi model lệch market > ngưỡng?

### 2.4 Component 3 — Morocco backtest (validate trước khi tin live)
- **Reframe quan trọng:** "odds Morocco vẫn thấp ở vòng cuối" rất có thể **đúng** (đội yếu vào SF vẫn có title-prob thấp thật). Phép thử thật = *khi underdog tiến sâu, conditional re-sim có đẩy odds lên **đúng mức** (calibrated update) không* — không phải "cao/thấp".
- **Cách làm:** replay WC2022, sau mỗi vòng Morocco đi tiếp, ghi title-prob conditional của model; so trajectory với (i) kết quả thật, (ii) market khi đó **nếu có** (nhiều khả năng không có → chỉ định tính).
- **Phụ thuộc:** cần Component 1 xong trước.

### 2.5 Component 4 — "Sentiment" playbook (đáp Sam's Q3)
- **4a — Unusual-event policy:** khi underdog thắng favorite, odds *nên* phản ứng thế nào? Định nghĩa kỳ vọng (Bayesian update từ kết quả) + ngưỡng "bất thường".
- **4b — Post-match update guardrails:** sau mỗi trận, rating/ratio update → làm sao kết quả sau **sát hơn mà không lệch quá** (chống overreact). Candidate: shrinkage/cap mức dịch chuyển, hoặc giữ snapshot-Elo (như mc_simu vốn làm) + chỉ cập nhật bracket. **Pre-registered band** cho mức dịch chuyển hợp lý.
- Đây chính là phần "sentiment more than data driven" — một **bộ quy tắc phản ứng**, không phải recompute cơ học.

### 2.6 Deliverables Task 2
1. Design doc của conditioning harness (trước khi code).
2. (sau sign-off) harness + daily rerun script + time-series output.
3. Morocco backtest report (calibrated update có đạt không).
4. Playbook 4a/4b dạng quy tắc rõ ràng.

---

## 3. Open decisions (cần bạn/Sam chốt trước khi build)

1. **Bucket:** quartile hay decile? (phụ thuộc đủ mẫu — chốt sau khi xem coverage).
2. **[Sam] "Sentiment"**: xác nhận cách hiểu 4a/4b có đúng ý không.
3. **[Sam] Output Task 2**: dashboard / CSV / alert?
4. **Phạm vi**: Task 2 daily rerun **ở lại mc_simu (standalone)** hay feed sang live engine (trading)? (CLAUDE.md: mc_simu không đụng live engine — mặc định standalone).
5. **Correction**: nếu Task 1 thấy bias, có làm bước correction (shrink/recalibrate) trong scope này hay tách scope sau? (Sam lo over-engineer → đề xuất tách).

---

## 4. Sequencing đề xuất

1. **Chờ deep-research** → điền method [DR] cho Task 1.3.
2. **Task 1 trước** (đo lường, không đụng model) → ra bức tranh bias + Spain. Rẻ, an toàn, informative.
3. **Task 2 Component 1** (conditioning harness) — build item lớn nhất; xong mới làm được Morocco backtest (C3).
4. **Playbook 4a/4b** song song, chốt với Sam.
5. Correction (nếu có) = scope riêng sau, có pre-registered bar.
