# SPEC: Trading Reasoning LLM (Chart → Think → Action)

Version: 0.3 — thay thế v0.2. Giữ nguyên toàn bộ grammar/tokenizer/parser (mục 2–4) không đổi so
với v0.2. Thay đổi so với v0.2, tất cả đến từ thực nghiệm GRPO thật (round test đầu tiên) và các
bug/gap phát hiện được khi đối chiếu code với spec:

- **Semantic invariants**: thêm mục **H — Bề rộng Zone** (mục 2.2) — invariant này đã tồn tại trong
  code (`SemanticChecker`/generator) từ trước nhưng CHƯA từng được ghi vào spec chính, coi như spec
  thiếu sót, giờ bổ sung lại cho đúng thực tế đã triển khai.
- **Reward Design (mục 5)**: viết lại HOÀN TOÀN gate 3 (outcome). Công thức cũ
  `R_outcome × w[trend][action_type]` chỉ là placeholder ban đầu, thực nghiệm cho thấy cần: 1 sàn
  điểm cộng cố định khi qua gate 2 (tránh mập mờ giữa "semantic sai" và "trade lỗ"), 1 cơ chế chấm
  chất lượng zone độc lập với lựa chọn SL/RR/entry của model, và 1 khoản phí giao dịch cho hành động
  thật (BUY/SELL) để phân biệt "vào lệnh dở" với "đứng ngoài đúng lúc". Version này CÓ THỂ còn đổi
  tiếp — đánh version cao hơn mỗi khi outcome thực nghiệm lộ ra lỗ hổng mới, đừng coi công thức mục
  5 là chốt cuối cùng.
- **Forward-test engine (mục 6)**: thêm **Zone Quality Probe** — cơ chế mới, tách biệt hoàn toàn
  khỏi hành động thật của model.
- **RoundConfig (mục 8.4, mới)**: chính thức hoá cấu hình tường minh theo round GRPO — gộp cả
  weight_table lẫn 2 cặp ngưỡng zone-width/SL-distance (trước đây spec mô tả 2 cặp ngưỡng này là
  "hằng số cố định set 1 lần bên ngoài" — thực tế giờ đây chúng CÓ THỂ chỉnh theo từng round GRPO,
  vì chỉ tới lúc có outcome thật mới biết nên nới/siết thế nào).
- **Dataset architecture (mục 7.1)**: sửa lại đúng thực tế vận hành — không phải "3 repo Hub tách
  biệt" như v0.2 mô tả, mà là 2 subset "raw"/"ids" trong CÙNG 1 repo, và pretrain+SFT dùng CHUNG 1
  dataset "ids" (khác nhau ở cách consume, không khác nhau ở dữ liệu).
- **`unified_reward_func` (mục 8.1)**: sửa 1 bug quan trọng trong chính pseudocode của spec — thiếu
  bước ghép `prompt` (chart) vào trước khi parse, khiến Parser luôn thấy thiếu `<chart>` (xem mục
  8.1 để biết chi tiết hệ quả).
- Mục 9 (câu hỏi mở): xoá câu #5 (trọng số R_wf/R_sem/R_outcome) — đã chốt thành công thức cụ thể ở
  mục 5, không còn là câu hỏi mở.

---

## 1. Mục tiêu

Xây dựng một LLM nhỏ, chuyên biệt, nhận vào chuỗi nến giá (đã rời rạc hóa dạng bin 0–1023) và sinh ra một chuỗi suy luận có cấu trúc (`think`) rồi kết luận hành động giao dịch (`action`). Model học qua 3 giai đoạn:

1. **Pretrain** — chart thật + think/action **random gen có kiểm soát semantic** (mục tiêu: phân bố đều trên không gian tổ hợp hợp lệ).
2. **SFT** — chart thật + think/action random gen tương tự, mục tiêu học well-form (semantic đi kèm miễn phí vì generator đảm bảo đúng "by construction").
3. **GRPO** — model tự sinh toàn bộ think/action từ chart thật, chấm điểm qua cơ chế **gate tuần tự 3 tầng** (well-form → semantic → outcome), có counterfactual reward cho action loại "cancel", và weight/zone/SL/fee điều chỉnh tay theo round qua `RoundConfig` (mục 8.4).

Tư tưởng cốt lõi xuyên suốt: outcome (kết quả thật khi forward-test) phải **lan truyền ngược** để định hình các khái niệm mở (trend, zone, `good_price_action`), nhưng chỉ khi chuỗi suy luận đã hợp lệ về cấu trúc và ngữ nghĩa — nếu không, outcome không được tính.

---

## 2. Định dạng dữ liệu (Grammar) — không đổi so với v0.2

### 2.1 Cấu trúc tổng quát

```
program      := chart_block think_block action_block

chart_block  := "<chart>" candle{50} "</chart>"
candle       := OHLC_O OHLC_H OHLC_L OHLC_C          # token dạng <O_434> <H_543> <L_543> <C_543> — ATOMIC, không digit-decompose

think_block  := "<think>" trend current_price zone? price_in_zone? good_price_action? "</think>"

trend         := "<trend>" ("UP"|"DOWN"|"RANGE") "</trend>"
current_price := "<current_price>" DIGIT{4} "</current_price>"      # BẮT BUỘC, mọi trường hợp, không điều kiện
zone          := zone_support | zone_resistance
zone_support     := "<zone_support>" DIGIT{4} ":" DIGIT{4} "</zone_support>"
zone_resistance  := "<zone_resistance>" DIGIT{4} ":" DIGIT{4} "</zone_resistance>"
price_in_zone      := "<price_in_zone>"
good_price_action  := "<good_price_action>"

action_block := "<action>" ACTION_TYPE [ "SL:" DIGIT{4} RR_TOKEN ] "</action>"
ACTION_TYPE  := "BUY" | "SELL" | "CANCEL_BUY" | "CANCEL_SELL" | "WAIT_BUY" | "WAIT_SELL" | "HOLD"
RR_TOKEN     := "<RR_1>" | "<RR_2>" | ... | "<RR_9>"
```

- `DIGIT`: 1 chữ số `0`–`9`, mỗi digit là 1 token riêng, **cách nhau bằng khoảng trắng** trong text thô. Zero-pad cố định **4 chữ số** (`DIGIT_PAD=4`).
- `RR_TOKEN`: bracket-enum **atomic**, KHÔNG digit-decompose — chỉ 9 giá trị. `rr` sau khi parse là **`int`** (không phải `float` — v0.2 ghi nhầm là `Optional[float]` ở mục AST, đã sửa lại đúng theo code ở mục 4.3).
- `BIN` (giá trị số nguyên của DIGIT{4} sau khi ghép, hoặc của candle atomic): 0–1023.
- Chỉ 1 zone / lần sinh (v1). Zone có **hướng** (support/resistance).
- `current_price` là field duy nhất **luôn bắt buộc**, không phụ thuộc bất kỳ điều kiện nào.

### 2.2 Ràng buộc ngữ nghĩa (semantic invariants) — A/B/D/E/H

**A. Trend ↔ Zone**

| trend | zone hợp lệ |
|---|---|
| UP | chỉ `zone_support` (bắt buộc phải có) |
| DOWN | chỉ `zone_resistance` (bắt buộc phải có) |
| RANGE có setup | 1 trong 2 loại |
| RANGE không setup | không có zone → action chỉ có thể là `HOLD` |

**B. Hướng của Zone ↔ current_price**

```python
zone_support:    zone_lower_bin <= current_price_bin
zone_resistance: zone_upper_bin >= current_price_bin
```
Vi phạm → semantic fail.

**C. current_price ↔ chart thật** (well-form, không phải semantic)

`current_price` bin phải khớp tuyệt đối bin `C` (Close) của nến cuối trong `chart_block`. Sai → lỗi well-form, nặng hơn lỗi cú pháp thường (phản ánh model chưa "đọc đúng" input).

**D. price_in_zone ↔ hình học thật**

```python
if zone_lower_bin <= current_price_bin <= zone_upper_bin:
    expected_price_in_zone = True
else:
    expected_price_in_zone = check_last_5_candles_touch(zone_lower_bin, zone_upper_bin, chart_candles)
```

**E. price_in_zone ↔ nhóm action hợp lệ**

| Điều kiện | Action hợp lệ |
|---|---|
| có zone, `price_in_zone = true` | `{ACTION}` hoặc `CANCEL_{ACTION}` |
| có zone, `price_in_zone = false` | chỉ `WAIT_{ACTION}` |
| RANGE, không có zone | chỉ `HOLD` |

**F. Field bắt buộc/cấm theo action_type** (well-form, không phải semantic)

| Action | current_price | good_price_action | SL/RR |
|---|---|---|---|
| BUY / SELL | bắt buộc | **bắt buộc** | **bắt buộc** |
| CANCEL_BUY / CANCEL_SELL | bắt buộc | **cấm xuất hiện** | **cấm xuất hiện** |
| WAIT_BUY / WAIT_SELL | bắt buộc | cấm xuất hiện | cấm xuất hiện |
| HOLD | bắt buộc | cấm xuất hiện | cấm xuất hiện |

**G. good_price_action** — không có rule kiểm tra nội dung, chủ ý để tránh áp đặt bias chủ quan. Ý nghĩa học hoàn toàn qua outcome reward lan truyền ngược ở GRPO.

**H. Bề rộng Zone** *(mới ở v0.3 — invariant này đã tồn tại trong code từ trước, spec v0.2 thiếu sót không ghi lại)*

```python
width = zone.upper_bin - zone.lower_bin
ZONE_WIDTH_MIN_BINS <= width <= ZONE_WIDTH_MAX_BINS
```

Vi phạm → semantic fail. Lý do cần rule này: `generator.py` (sinh data pretrain/SFT) đã luôn tôn trọng ngưỡng này khi construct zone ngẫu nhiên, nhưng trước v0.3 **không có verifier nào kiểm tra lại lúc GRPO rollout** — vi phạm chính nguyên tắc "verifier = lật ngược generator" (mục 4.4). Model GRPO có thể sinh zone rộng 1 bin (price_in_zone gần như luôn True một cách rẻ tiền) hoặc rộng hàng trăm bin mà không bị phạt gì, nếu thiếu rule này.

Ngưỡng `ZONE_WIDTH_MIN_BINS`/`ZONE_WIDTH_MAX_BINS`:
- **Generator (pretrain/SFT)**: hardcode cố định (giá trị hiện tại: 5/20) — chỉ cần đúng format khi sinh data, không cần flexible.
- **GRPO**: lấy từ `RoundConfig` hiện tại (mục 8.4), CÓ THỂ chỉnh theo round.

---

## 3. Tokenizer / Vocab — không đổi so với v0.2

(Xem `docs/tokenizer.md` cho chi tiết implementation. Quyết định thiết kế giữ nguyên: chart OHLC atomic, current_price/zone/SL digit-decompose dùng chung 10 token `0`-`9`, RR bracket-enum atomic, WordLevel + WhitespaceSplit, KHÔNG BPE.)

---

## 4. Kiến trúc Parser (Lexer → Parser → AST → Semantic Checker) — không đổi so với v0.2

### 4.1 Lexer
Regex-based, output `Token(type, value, position)`.

### 4.2 Parser
Panic-mode error recovery, `well_form_score` liên tục theo số lỗi. Bao gồm kiểm tra bảng C và F.

### 4.3 AST Nodes
```
CandleNode(o, h, l, c)
ChartNode(candles: List[CandleNode])
ZoneNode(direction: "support"|"resistance", lower_bin, upper_bin)
ThinkNode(trend, current_price_bin, zone: Optional[ZoneNode], price_in_zone: bool, good_price_action: bool)
ActionNode(action_type, sl: Optional[int], rr: Optional[int])   # rr: int — v0.2 ghi nhầm là float, sửa lại đúng code
ProgramNode(chart, think, action)
```

### 4.4 Semantic Checker

Kiểm tra bảng A/B/D/E/H trên AST đã parse thành công (trừ C/F đã ở well-form, trừ G không kiểm tra).
Output: `SemanticResult { passed, violations: List[str], score: float }`.

**Nguyên tắc**: verifier ở GRPO = "lật ngược" generator dùng để sinh data SFT/pretrain — generator đảm bảo đúng các invariant này lúc sinh, verifier chỉ cần lật ngược logic đó thành kiểm tra. Mục H là ví dụ về việc nguyên tắc này từng bị vi phạm (generator tôn trọng nhưng verifier không kiểm tra) — mọi invariant mới thêm sau này phải tự hỏi: "generator có tôn trọng cái này khi sinh không? Nếu có, verifier phải kiểm tra lại."

`zone_width_min_bins`/`zone_width_max_bins` (dùng cho rule H) là tham số của `SemanticChecker.__init__`, default = hằng số hardcode (5/20, dùng cho generator/demo); GRPO truyền tường minh từ `RoundConfig`.

---

## 5. Reward Design (GRPO) — **VIẾT LẠI HOÀN TOÀN ở v0.3**

### 5.1 Cơ chế Gate tuần tự (KHÔNG cộng tuyến tính) — nguyên tắc không đổi

```
1. Well-form check (parser + bảng F)
   FAIL → reward = well_form_score() (liên tục theo số lỗi) → DỪNG
   PASS → tiếp bước 2

2. Semantic check (bảng A/B/D/E/H) + ràng buộc SL/target bổ sung (mục 6.1, tách riêng khỏi
   SemanticChecker vì cần entry/SL/zone cùng lúc — is_sl_valid trong forward_test engine)
   FAIL → reward = R_WF_FULL + sem_score (liên tục theo số vi phạm) → DỪNG, không chấm outcome
   PASS → tiếp bước 3

3. Outcome — xem công thức đầy đủ ở mục 5.2
```

Lý do không cộng tuyến tính: tránh completion sai cấu trúc/ngữ nghĩa nhưng outcome thắng may mắn vẫn đạt điểm cao.

`R_WF_FULL = 1.0`, `R_SEM_FULL = 1.0` — hằng số cố định (không theo round).

Khi gate 2 fail RIÊNG vì ràng buộc SL bổ sung (semantic bảng A/B/D/E/H pass, nhưng `is_sl_valid` fail): trừ thêm `EXTRA_SEMANTIC_PENALTY` (= `SemanticChecker.VIOLATION_PENALTY` = 0.2) vào `sem_score`, để phân biệt với trường hợp pass hoàn toàn cả 2 phần của gate 2.

### 5.2 Công thức Outcome (Gate 3) — MỚI HOÀN TOÀN, thay thế công thức v0.2

Sau khi qua gate 2, **mọi action đều nhận 1 sàn điểm cộng cố định K** trước khi tính thêm outcome —
lý do: nếu không có sàn này, 1 completion pass hết well-form+semantic nhưng outcome tệ (vd BUY lỗ
nặng) có thể có reward thấp hơn 1 completion fail semantic nhẹ, phá vỡ đúng tinh thần "gate cứng"
(pass gate luôn phải tốt hơn fail gate, bất kể outcome tệ tới đâu).

```
base = R_WF_FULL + R_SEM_FULL + K                       # K = round_config.pass_gate2_bonus

zone_bonus = zone_quality_bonus nếu Zone Quality Probe (mục 6.3) thắng (r_multiple > 0), else 0
             — CHỈ tính khi action có zone (WAIT_*/CANCEL_*/BUY/SELL), KHÔNG áp cho HOLD

reward theo action_type:
  HOLD                  -> base
  WAIT_BUY / WAIT_SELL  -> base + zone_bonus
  CANCEL_BUY/SELL       -> base + zone_bonus + min(0, r_multiple_counterfactual × w[trend][action])
  BUY / SELL            -> base + zone_bonus + (r_multiple_thật − fee_in_r) × w[trend][action]
```

trong đó:
- `w[trend][action_type]`: lấy từ `weight_table` trong `RoundConfig` hiện tại (mục 8.4).
- `fee_in_r = round_config.trade_fee_bins / risk_bins`, với `risk_bins = |current_price_bin − SL|`
  (risk CỦA CHÍNH lệnh này — không phải hằng số cố định, vì phí quy đổi ra đơn vị R phụ thuộc SL đặt
  sát hay xa. Nguyên tắc tính giống spread: cố định theo BIN giá, ảnh hưởng R nhiều hơn khi SL sát).

**Vì sao CANCEL chỉ có thể trừ, không thể cộng** (`min(0, ...)`): tại thời điểm CANCEL, giá đã vào
zone (đã được cộng `zone_bonus` như 1 hành động thật khác trên cùng zone đó) — phần `r_multiple`
counterfactual ở đây chỉ đo **chi phí cơ hội** của việc không hành động, không phải 1 phần thưởng
độc lập. Nếu để cộng dương, model có thể học cách "CANCEL bừa" để ăn điểm dương từ counterfactual mà
không cần suy luận gì thêm — vi phạm ý định thiết kế ban đầu của CANCEL (chỉ nên dùng khi thật sự
nên tránh 1 setup tưởng tốt nhưng đã hỏng).

**Vì sao BUY/SELL cần trừ phí (`fee_in_r`)**: phân biệt 2 loại "hành động không tốt" khác hẳn nhau
về bản chất — (a) vào lệnh với setup dở/RR phi thực tế (bị timeout hoặc lỗ, `r_multiple ≈ 0` hoặc
âm) VẪN phải trả phí, trong khi (b) đứng ngoài đúng lúc (CANCEL) không trả phí gì — tạo bất đối xứng
đúng ý "tốn phí khi vào lệnh, tiết kiệm phí khi đứng ngoài". Riêng trường hợp đặt RR quá lớn khiến
giá tiến gần target rồi đảo chiều chạm SL: **KHÔNG cần rule phạt riêng** — forward-test (mục 6.1) đã
tự bắt được LOSS trong trường hợp này (mỗi nến tương lai được kiểm tra `hit_sl` VÀ `hit_tp`, nến nào
đảo chiều chạm SL sẽ trả `r_multiple=-1` như bình thường, không cần logic mới).

**TODO còn mở** (không phải thiếu sót, cố ý để ngỏ): phân biệt phí giao dịch có nên áp riêng theo
`action_type`/`symbol` khác nhau hay dùng 1 hằng số chung `trade_fee_bins` cho mọi trường hợp — hiện
tại dùng 1 hằng số chung, tinh chỉnh sau khi có dữ liệu thực nghiệm nhiều round.

### 5.3 weight_table — round-based, giờ là 1 field của RoundConfig

`weight_table[trend][action_type]` không còn là 1 dict Python độc lập sửa tay giữa các lần gọi
`trainer.train()` (như v0.2 mô tả) — giờ là 1 field bắt buộc trong `RoundConfig` (mục 8.4), load từ
file JSON tường minh mỗi round, KHÔNG sửa trực tiếp trong code/session Python.

- Thống kê **trend-conditional** qua `StatsCollector` (mục 8.4), đọc bằng `--report_only` giữa các
  lần chạy, không cần đợi round kết thúc.
- Vai trò theo round: round đầu weight gần đều (explore). Round giữa chỉnh để kéo nhánh bị bỏ quên
  lên/dìm nhánh bị lạm dụng. Round cuối anneal dần về phản ánh chất lượng thật.
- **Bất biến bắt buộc**: `pass_gate2_bonus` (K) phải lớn hơn mọi giá trị trong `weight_table` của
  chính round đó — xem chứng minh ở mục 8.4.

---

## 6. Forward-test / Counterfactual Engine

Tất cả tính toán dựa hoàn toàn trên bin (không cần decode ngược về giá thật).

### 6.1 BUY/SELL thật — không đổi so với v0.2

- Entry = bin `current_price`. SL = bin model chọn. Target = derive từ entry/SL/RR:
  `target = entry + RR × (entry - SL)` (long) hoặc `target = entry - RR × (SL - entry)` (short).
- **Ràng buộc khoảng cách SL** (kiểm tra ở gate 2, KHÔNG phải `SemanticChecker` — cần entry/SL/zone
  cùng lúc):
  ```python
  def is_sl_valid(action_type, entry_bin, sl_bin, zone, sl_min_dist_bins, sl_max_dist_bins):
      dist = abs(entry_bin - sl_bin)
      if not (sl_min_dist_bins <= dist <= sl_max_dist_bins):
          return False
      if action_type == "BUY":
          return sl_bin < zone.lower_bin
      else:  # SELL
          return sl_bin > zone.upper_bin
  ```
  `sl_min_dist_bins`/`sl_max_dist_bins`: hardcode (5/10) cho generator/demo; GRPO lấy từ `RoundConfig`.
- Forward-test trên toàn bộ 50 nến `future_bins` (không dừng sớm). Gap SL/TP cùng nến → ưu tiên SL
  (conservative). Timeout (không chạm gì hết 50 nến) → `r_multiple = 0` (trung tính, không phạt —
  để model tự tìm RR hợp lý qua outcome, không ép bằng phạt cứng). Bin bão hoà ([0,1023]) khi derive
  target → coi là setup không hợp lệ, phạt semantic.

### 6.2 CANCEL_BUY / CANCEL_SELL (counterfactual) — không đổi so với v0.2

```python
def counterfactual_outcome(action_type, zone, future_bins, current_price_bin):
    if action_type == "CANCEL_BUY":
        entry, sl, direction = current_price_bin, zone.lower_bin - 1, "long"
    else:  # CANCEL_SELL
        entry, sl, direction = current_price_bin, zone.upper_bin + 1, "short"
    target = entry ± 1 * (entry - sl)   # RR cố định = 1
    result = forward_test(entry, sl, target, future_bins, direction)
    return -result.r_multiple   # ĐẢO DẤU: CANCEL đúng khi lẽ ra sẽ thua
```
SL/RR cho CANCEL không lấy từ model output — derive tự động từ zone + buffer cố định 1 bin + RR=1.

### 6.3 Zone Quality Probe — **MỚI ở v0.3**

Kiểm chứng ĐỘC LẬP chất lượng của zone, tách biệt hoàn toàn khỏi SL/RR/entry mà model thực sự chọn
(đó là việc của outcome thật ở mục 6.1/6.2). Dùng để quyết định `zone_bonus` ở mục 5.2.

Phép thử chuẩn hoá: giả lập đặt lệnh ở **mép GẦN giá hơn** của zone, SL ở **mép còn lại + buffer**,
RR=1 cố định:

```python
ZONE_PROBE_SL_BUFFER_BINS = 1   # buffer khi mô phỏng SL "ở mép zone" — TÁCH RIÊNG hằng số so với
                                  # buffer của counterfactual_outcome (mục 6.2) dù cùng ý tưởng, vì
                                  # 2 hàm mô phỏng ENTRY KHÁC NHAU (probe: mép zone; counterfactual:
                                  # giá hiện tại) — không gộp chung 1 hàm.

def probe_zone_quality(zone, future_candles):
    if zone.direction == "support":
        entry, sl, direction = zone.upper_bin, zone.lower_bin - ZONE_PROBE_SL_BUFFER_BINS, "long"
    else:  # resistance
        entry, sl, direction = zone.lower_bin, zone.upper_bin + ZONE_PROBE_SL_BUFFER_BINS, "short"
    target = derive_target(entry, sl, rr=1.0, direction=direction)
    if target is None:
        return ForwardTestResult(status=INVALID_SETUP, r_multiple=0.0)
    return forward_test(entry, sl, target, future_candles, direction)
```

Nếu zone thật sự bám đúng support/resistance (dựng từ hình học chart thật), phép thử mép-đối-mép
chuẩn hoá này sẽ thắng thường xuyên (`r_multiple > 0`). Nếu zone bị dựng ẩu chỉ để thoả gate D (vd
luôn bao `current_price` kiểu CONTAINS, không cần chart hỗ trợ gì), phép thử này thắng/thua gần như
ngẫu nhiên — không có edge thật để khai thác.

**Không áp `is_sl_valid`/ngưỡng khoảng cách SL ở đây** — đây là 1 probe tổng hợp đo chất lượng zone,
không phải 1 lệnh thật do model chọn, nên không cần tuân theo ràng buộc khoảng cách SL của model.

Probe này chạy cho **mọi action có zone** (WAIT_*/CANCEL_*/BUY/SELL) — không chạy cho HOLD (không có
zone để chấm).

---

## 7. Dataset Design

### 7.1 Kiến trúc lưu trữ trên Hub — **SỬA LẠI so với v0.2** (v0.2 mô tả sai thực tế)

v0.2 mô tả "3 repo Hub riêng biệt" (pretrain/sft/grpo tách biệt hoàn toàn). **Thực tế vận hành**:

- Mỗi loại dataset (pretrain+sft dùng chung, hoặc grpo) là **1 repo Hub**, bên trong có **2 subset**
  (`config_name` của `datasets`):
  - `"raw"` — text thô, chưa tokenize, đúng schema mục 7.2/7.3.
  - `"ids"` — đã tokenize + mask (chỉ có ở pretrain/sft, không cần cho GRPO vì GRPOTrainer tự
    tokenize prompt lúc rollout), build 1 lần bằng script offline, KHÔNG build lại mỗi lần train.
- **Pretrain và SFT dùng CHUNG 1 dataset "ids"** — khác nhau ở cách CONSUME (pretrain: full-sequence
  loss, không mask; SFT: mask `<bos>+prompt` → chỉ tính loss trên completion), không khác nhau ở dữ
  liệu hay cách build ids. Quyết định "mask hay không" nằm ở phía Trainer/Collator lúc train, KHÔNG
  nằm ở bước build ids — build ids không được tự ý giả định trước dataset này sẽ dùng cho giai đoạn
  nào.
- Lý do tách "raw"/"ids" ra 2 subset thay vì gộp chung: đổi tokenizer sau này (vd đổi digit-pad) chỉ
  cần build lại subset "ids", không phải regen lại "raw" từ đầu.

### 7.2 Schema Pretrain/SFT — không đổi so với v0.2

```python
{"prompt": str, "completion": str}
```
Sample uniform trên toàn bộ leaf-path hợp lệ đã liệt kê tường minh (`LEAF_RECIPES`), không sample
uniform từng field độc lập rồi lọc bỏ invalid.

### 7.3 Schema GRPO — không đổi so với v0.2

```python
{"prompt": str, "future_bins": list, "symbol": str, "window_id": str}
```

### 7.4 Chống leakage — không đổi so với v0.2

Window chồng lấn (`stride=50`, `window_size=100`) → split theo khối thời gian liên tục, không chia
ngẫu nhiên theo từng window riêng lẻ. `window_id` dùng để dedup.

---

## 8. Tích hợp TRL GRPOTrainer

### 8.1 `unified_reward_func` — **SỬA BUG quan trọng so với v0.2**

**Bug đã phát hiện và sửa**: pseudocode v0.2 chỉ `Parser.parse(completion)` — KHÔNG ghép `prompt`
(chart) vào trước khi parse. TRL's `GRPOTrainer` chỉ decode `completions` từ token **sinh RA SAU
prompt** (`completion_ids`), KHÔNG bao giờ bao gồm `<chart>`. Vì grammar (mục 2.1) yêu cầu
`program := chart_block think_block action_block` — chart_block LUÔN đứng đầu và BẮT BUỘC — parse
completion một mình khiến Parser **LUÔN LUÔN** báo lỗi "thiếu `<chart>`" (đúng 1 lỗi structural cố
định, `well_form_score() ≈ 0.85` bất kể think/action bên trong đúng hay sai), che gần hết gradient
signal thật trong suốt quá trình GRPO training. Đã sửa: ghép `prompt + " " + completion` trước khi
parse (đúng convention nối chuỗi đã dùng ở data pipeline pretrain/SFT).

```python
def score_completion(prompt, completion, future_bins, ...):
    round_config = get_active_round_config()   # BẮT BUỘC đã load trước — fail-loud nếu chưa (mục 8.4)
    parse_result = Parser.from_text(prompt + " " + completion).parse()

    if not parse_result.is_well_formed():
        return parse_result.well_form_score()

    program = parse_result.ast
    semantic_result = SemanticChecker(
        zone_width_min_bins=round_config.zone_width_min_bins,
        zone_width_max_bins=round_config.zone_width_max_bins,
    ).check(program)
    extra_valid, forward_result = evaluate_outcome(
        program.action, program.think, future_bins,
        sl_min_dist_bins=round_config.sl_min_dist_bins,
        sl_max_dist_bins=round_config.sl_max_dist_bins,
    )
    if not (semantic_result.passed and extra_valid):
        ...  # xem mục 5.1
        return R_WF_FULL + sem_score

    # Gate 3 — xem công thức đầy đủ mục 5.2
    K = round_config.pass_gate2_bonus
    base = R_WF_FULL + R_SEM_FULL + K
    ...
    return reward


def unified_reward_func(prompts, completions, future_bins, **kwargs):
    return [
        score_completion(prompt, completion, fb, ...)
        for prompt, completion, fb in zip(prompts, completions, future_bins)
    ]
```

**Nguyên tắc quan trọng nhất (không đổi so với v0.2)**: KHÔNG tách 3 gate thành 3 `reward_funcs`
riêng kèm `reward_weights` (TRL cộng dồn tuyến tính, mâu thuẫn thiết kế gate cứng). Chỉ viết **1 hàm
`unified_reward_func` duy nhất**.

### 8.2 Config bắt buộc — không đổi so với v0.2

```python
GRPOConfig(..., remove_unused_columns=False, num_generations=12, ...)
```

### 8.3 Vòng lặp round-based — không đổi ý tưởng so với v0.2

Gọi `trainer.train()` nhiều lần với `resume_from_checkpoint`; giữa các lần, sửa tay `RoundConfig`
(mục 8.4) — KHÔNG dùng callback tự động.

### 8.4 RoundConfig — **MỤC MỚI ở v0.3**

1 round GRPO = 1 file JSON tường minh, **cố định cho đến hết round**, gộp toàn bộ tham số cần chỉnh
theo thực nghiệm (weight_table VÀ 2 cặp ngưỡng zone/SL, trước đây v0.2 coi 2 cặp ngưỡng này là hằng
số cố định set 1 lần bên ngoài — thực tế cần flexible theo round vì chỉ tới GRPO mới biết nên
nới/siết thế nào):

```python
@dataclass
class RoundConfig:
    round_id: str
    weight_table: Dict[str, Dict[str, float]]   # trend -> action_type -> weight
    zone_width_min_bins: int
    zone_width_max_bins: int
    sl_min_dist_bins: int
    sl_max_dist_bins: int
    pass_gate2_bonus: float      # K — mục 5.2
    zone_quality_bonus: float    # mục 5.2/6.3
    trade_fee_bins: float        # mục 5.2
```

- **KHÔNG có fallback ngầm**: thiếu file, hoặc thiếu BẤT KỲ field nào trong 8 field trên → raise
  ngay lúc `RoundConfig.load(path)`, không âm thầm dùng giá trị mặc định nào khác (kể cả giá trị
  đang hardcode ở `SemanticChecker`/`forward_test.py` cho generator/demo — 2 mục đích khác nhau,
  không dùng lẫn).
- **Bất biến bắt buộc** (`__post_init__`, raise nếu vi phạm): `pass_gate2_bonus` (K) phải **lớn hơn**
  max weight trong `weight_table` của chính round đó. Chứng minh: reward tệ nhất khi PASS gate 2
  (BUY/SELL LOSS, zone_bonus=0, bỏ qua fee cho đơn giản) = `2.0 + K − w`; reward tệ nhất khi FAIL
  gate 2 (0 vi phạm còn lại) = `1.0 + (1.0 − 0.2) = 1.8`. Cần `2.0+K−w > 1.8` ⟺ `K > w − 0.2`. Điều
  kiện `K > w` (đang áp dụng) chặt hơn, tự động thoả với mọi `VIOLATION_PENALTY ≥ 0`.
  **Lưu ý khi có `trade_fee_bins` > 0**: biên an toàn thực tế bị siết nhẹ (worst-case pass-gate-2
  giờ trừ thêm `fee_in_r × w`) — điều kiện đầy đủ hơn là
  `K > w × (1 + trade_fee_bins / sl_min_dist_bins) − 0.2`. Chưa tự động hoá check này trong code
  (`trade_fee_bins` còn hay đổi số) — giữ margin `K` dư dả (vd K=1.5 so với max weight=1.0) để an
  toàn trong thực tế, tự kiểm tra lại quan hệ này bằng tay nếu đẩy `trade_fee_bins` lên cao hoặc siết
  `sl_min_dist_bins` xuống thấp.
- `weight_table` sync vào 1 `WeightTable` singleton (`app/training/reward/reward_func.py`) qua
  `set_active_round_config()`, gọi 1 lần lúc khởi động script train — mọi rank/process load CÙNG 1
  file nên tự nhiên đồng bộ, không cần cơ chế broadcast riêng cho multi-GPU.
- `StatsCollector` (thống kê `trend × action_type` để quyết định chỉnh `weight_table` round sau)
  persist ra đĩa theo pattern **load-rồi-append**: mỗi rank tự dump 1 file riêng
  (`{output_dir}/round{N}_stats_rank{R}.json`), mỗi lần script khởi động lại (Colab bị ngắt, chạy lại
  NHIỀU LẦN trong CÙNG 1 round) đều load lại file cũ trước khi log tiếp — không mất thống kê giữa
  các lần chạy. Xem report bất cứ lúc nào bằng cách gộp mọi file rank của 1 round
  (`StatsCollector.merge_from_files`), không cần đợi round kết thúc.

---

## 9. Câu hỏi còn mở / cần chốt trước khi code

**Đã chốt (kể cả các mục mới chốt ở v0.3):**
1. Horizon forward-test → toàn bộ 50 nến `future_bins`, không dừng sớm.
2. Gap SL/TP cùng nến → ưu tiên SL.
3. Timeout → `r_multiple = 0` (trung tính).
4. Vocab RR → vocab riêng, giới hạn 1–9.
5. **(v0.3)** Công thức reward gate 3 đầy đủ — xem mục 5.2. Đánh dấu rõ: công thức này **được kỳ
   vọng còn đổi tiếp** khi thực nghiệm round sau lộ ra lỗ hổng mới (khác các mục "đã chốt" khác —
   mục này chỉ chốt ở nghĩa "đã implement và verify", không phải "sẽ không đổi nữa").
6. Ngưỡng pass gate semantic → 100%, không dùng threshold %.
7. Ngưỡng zone-width/SL-distance → số bin cố định, generator hardcode, GRPO override qua `RoundConfig`.
8. **(v0.3)** Zone Quality Probe → mục 6.3, entry mép-đối-mép, RR=1, buffer 1 bin riêng biệt với
   buffer của `counterfactual_outcome` (2 hàm entry khác nhau, không gộp).
9. **(v0.3)** Kiến trúc dataset Hub → 2 subset raw/ids trong 1 repo, pretrain+SFT dùng chung "ids".

**Còn mở:**
- Phân biệt `trade_fee_bins` có nên khác nhau theo `action_type`/`symbol` hay dùng 1 hằng số chung —
  hiện dùng 1 hằng số chung cho mọi trường hợp, tinh chỉnh sau khi có dữ liệu nhiều round.
- Giá trị cụ thể của 8 field trong `RoundConfig` cho round 2 trở đi — phụ thuộc thống kê thực
  nghiệm từ `StatsCollector`, không chốt trước.

---

## 10. Việc cần làm tiếp theo

- [x] Lexer/Parser/AST/SemanticChecker (bảng A/B/D/E/H).
- [x] Generator cho pretrain/SFT (leaf-recipes, hardcode zone/SL).
- [x] Forward-test + Counterfactual engine + Zone Quality Probe.
- [x] `unified_reward_func` + `StatsCollector` + `RoundConfig` (đã sửa bug thiếu `prompt`).
- [x] Build dataset (raw/ids, 2 subset trong 1 repo, chống leakage).
- [x] `train_pretrain.py`/`train_sft.py`/`train_grpo.py` (round-based, resumable, KV-cache khi init
      từ SFT/round trước).
- [ ] Chạy nhiều round GRPO thật, dùng `StatsCollector` để tinh chỉnh `RoundConfig` giữa các round.
- [ ] Instrument logging: tỷ lệ pass từng gate theo round, phân bố trend×action, outcome trung bình
      theo round (đã có `StatsCollector.print_summary()`, cần thói quen chạy `--report_only` định kỳ).