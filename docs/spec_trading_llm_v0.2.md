# SPEC: Trading Reasoning LLM (Chart → Think → Action)

Version: 0.2 — thay thế v0.1, gộp toàn bộ quyết định về grammar mở rộng (zone 2 hướng, CANCEL action, current_price bắt buộc), dataset schema thực tế, và cách tích hợp TRL GRPOTrainer.

---

## 1. Mục tiêu

Xây dựng một LLM nhỏ, chuyên biệt, nhận vào chuỗi nến giá (đã rời rạc hóa dạng bin 0–1023) và sinh ra một chuỗi suy luận có cấu trúc (`think`) rồi kết luận hành động giao dịch (`action`). Model học qua 3 giai đoạn:

1. **Pretrain** — chart thật + think/action **random gen có kiểm soát semantic** (mục tiêu: phân bố đều trên không gian tổ hợp hợp lệ).
2. **SFT** — chart thật + think/action random gen tương tự, mục tiêu học well-form (semantic đi kèm miễn phí vì generator đảm bảo đúng "by construction").
3. **GRPO** — model tự sinh toàn bộ think/action từ chart thật, chấm điểm qua cơ chế **gate tuần tự 3 tầng** (well-form → semantic → outcome), có counterfactual reward cho action loại "cancel", và weight điều chỉnh tay theo round.

Tư tưởng cốt lõi xuyên suốt: outcome (kết quả thật khi forward-test) phải **lan truyền ngược** để định hình các khái niệm mở (trend, zone, `good_price_action`), nhưng chỉ khi chuỗi suy luận đã hợp lệ về cấu trúc và ngữ nghĩa — nếu không, outcome không được tính.

---

## 2. Định dạng dữ liệu (Grammar) — v0.2

### 2.1 Cấu trúc tổng quát

```
program      := chart_block think_block action_block

chart_block  := "<chart>" candle{50} "</chart>"
candle       := OHLC_O OHLC_H OHLC_L OHLC_C          # token dạng <O_434> <H_543> <L_543> <C_543>

think_block  := "<think>" trend current_price zone? price_in_zone? good_price_action? "</think>"

trend         := "<trend>" ("UP"|"DOWN"|"RANGE") "</trend>"
current_price := "<current_price>" BIN "</current_price>"      # BẮT BUỘC, mọi trường hợp, không điều kiện
zone          := zone_support | zone_resistance
zone_support     := "<zone_support>" BIN ":" BIN "</zone_support>"
zone_resistance  := "<zone_resistance>" BIN ":" BIN "</zone_resistance>"
price_in_zone      := "<price_in_zone>"
good_price_action  := "<good_price_action>"

action_block := "<action>" ACTION_TYPE [ "SL:" BIN "RR:" NUM ] "</action>"
ACTION_TYPE  := "BUY" | "SELL" | "CANCEL_BUY" | "CANCEL_SELL" | "WAIT_BUY" | "WAIT_SELL" | "HOLD"
```

- `BIN`: số nguyên 0–1023 (bin đã rời rạc hóa, kế thừa từ `ChartCodec`).
- `RR`: **1 số duy nhất** (Reward-multiple). Risk luôn chuẩn hóa = 1 (khoảng cách entry→SL), không cần biểu diễn cặp `from->to` — `RR:9` nghĩa là target cách entry 9× khoảng cách entry→SL.
- Chỉ 1 zone / lần sinh (v1). Zone có **hướng** (support/resistance), không còn 1 loại chung chung.
- `current_price` là field duy nhất **luôn bắt buộc**, không phụ thuộc bất kỳ điều kiện nào (auxiliary task: ép model map token↔bin đúng, độc lập với logic quyết định trade).

### 2.2 Ràng buộc ngữ nghĩa (semantic invariants) — bảng đầy đủ v0.2

**A. Trend ↔ Zone**

| trend | zone hợp lệ |
|---|---|
| UP | chỉ `zone_support` (bắt buộc phải có) |
| DOWN | chỉ `zone_resistance` (bắt buộc phải có) |
| RANGE có setup | 1 trong 2 loại — loại nào quyết định phía action nào được xét |
| RANGE không setup | không có zone → action chỉ có thể là `HOLD` |

**B. Hướng của Zone ↔ current_price (bin arithmetic thuần túy)**

```python
zone_support:    zone_lower_bin <= current_price_bin      # zone nằm dưới hoặc chứa giá hiện tại
zone_resistance: zone_upper_bin >= current_price_bin      # zone nằm trên hoặc chứa giá hiện tại
```
Vi phạm → semantic fail (zone sinh sai hướng so với giá hiện tại).

*(Rule "khoảng cách tối thiểu entry–zone" đã bị loại bỏ — `price_in_zone` đối chiếu hình học thật đã là điều kiện đủ, không cần rule khoảng cách chồng lên.)*

**C. current_price ↔ chart thật**

`current_price` bin **phải khớp tuyệt đối** với bin `C` (Close) của nến cuối cùng trong `chart_block`. Sai giá trị → lỗi well-form, và nên tính nặng hơn lỗi cú pháp thông thường (phản ánh model chưa "đọc đúng" input, nghiêm trọng hơn lỗi format).

**D. price_in_zone ↔ hình học thật**

```python
if zone_lower_bin <= current_price_bin <= zone_upper_bin:
    expected_price_in_zone = True   # current price đã nằm trong zone → bắt buộc true
else:
    expected_price_in_zone = check_last_5_candles_touch(zone_lower_bin, zone_upper_bin, chart_candles)
```
Giá trị field `price_in_zone` (có mặt hay không) phải khớp với `expected_price_in_zone` tính từ chart thật — không phải field tự do.

**E. price_in_zone ↔ nhóm action hợp lệ**

| Điều kiện | Action hợp lệ (phía tương ứng với loại zone) |
|---|---|
| có zone, `price_in_zone = true` | `{ACTION}` (BUY/SELL — cần `good_price_action`+SL+RR) hoặc `CANCEL_{ACTION}` (không cần good_price_action/SL/RR) |
| có zone, `price_in_zone = false` | chỉ `WAIT_{ACTION}` |
| RANGE, không có zone | chỉ `HOLD` |

**F. Field bắt buộc/cấm theo action_type — kiểm tra ở tầng WELL-FORM (không phải semantic)**

| Action | current_price | good_price_action | SL/RR |
|---|---|---|---|
| BUY / SELL | bắt buộc | **bắt buộc** | **bắt buộc** |
| CANCEL_BUY / CANCEL_SELL | bắt buộc | **cấm xuất hiện** | **cấm xuất hiện** |
| WAIT_BUY / WAIT_SELL | bắt buộc | cấm xuất hiện | cấm xuất hiện |
| HOLD | bắt buộc | cấm xuất hiện | cấm xuất hiện |

Lý do xếp vào well-form chứ không phải semantic: đây là ràng buộc cấu trúc câu có điều kiện theo `ACTION_TYPE` đọc được ngay trong cùng block — grammar hơi context-sensitive (không hoàn toàn context-free), nhưng về bản chất vẫn là "đúng/sai ngữ pháp", chưa đánh giá gì về chất lượng quyết định.

**G. good_price_action — KHÔNG có rule kiểm tra nội dung**

Chỉ ràng buộc về vị trí xuất hiện (mục F). Ý nghĩa thực chất của token này hoàn toàn học được từ outcome reward lan truyền ngược qua GRPO — không viết rule đánh giá "pattern đẹp hay xấu" để tránh áp đặt bias chủ quan.

---

## 3. Tokenizer / Vocab

- Giá: bin 0–1023 (kế thừa `ChartCodec`, không thiết kế lại).
- Token cấu trúc: mỗi tag là 1 token riêng (`<chart>`, `<trend>`, `<zone_support>`, `<zone_resistance>`, `<current_price>`, `SL:`, `RR:`, `->`...).
- Enum: `UP/DOWN/RANGE`, `BUY/SELL/CANCEL_BUY/CANCEL_SELL/WAIT_BUY/WAIT_SELL/HOLD` — mỗi giá trị 1 token.
- Số trong `RR:`: **vocab riêng**, giới hạn giá trị 1–9 (không dùng chung bin 0–1023).

---

## 4. Kiến trúc Parser (Lexer → Parser → AST → Semantic Checker)

### 4.1 Lexer
Regex-based, output `Token(type, value, position)` — giữ position để tính điểm lỗi liên tục.

### 4.2 Parser
- Theo grammar mục 2.1, **panic-mode error recovery** (không hard-fail như compiler thật) — ghi nhận lỗi + vị trí, skip đến điểm đồng bộ hóa, tiếp tục parse phần còn lại.
- Bao gồm luôn bước kiểm tra field bắt buộc/cấm theo `ACTION_TYPE` (bảng 2.2.F) — vẫn tính là well-form, không tách riêng.
- Output: `ParseResult { ast, errors: List[ParseError] }`, `well_form_score` liên tục theo số lỗi.

### 4.3 AST Nodes
```
CandleNode(o, h, l, c)
ChartNode(candles: List[CandleNode])
ZoneNode(direction: "support"|"resistance", lower_bin, upper_bin)
ThinkNode(trend, current_price_bin, zone: Optional[ZoneNode], price_in_zone: bool, good_price_action: bool)
ActionNode(action_type, sl: Optional[int], rr: Optional[float])   # rr = 1 số duy nhất (risk chuẩn hóa = 1)
ProgramNode(chart, think, action)
```

### 4.4 Semantic Checker
Chạy trên AST đã parse (thành công toàn phần hoặc một phần), kiểm tra bảng 2.2 (A, B, D, E) — trừ F (đã ở well-form) và trừ G (không kiểm tra).
Output: `SemanticResult { passed, violations: List[str], score: float }` — liên tục theo số vi phạm.

**Nguyên tắc**: verifier ở GRPO = "lật ngược" generator dùng để sinh data SFT/pretrain — generator đảm bảo đúng các invariant này lúc sinh, verifier chỉ cần lật ngược logic đó thành kiểm tra.

---

## 5. Reward Design (GRPO)

### 5.1 Cơ chế Gate tuần tự (KHÔNG cộng tuyến tính)

```
1. Well-form check (parser + bảng 2.2.F)
   FAIL → reward = R_wf(liên tục theo số lỗi) → DỪNG
   PASS → tiếp bước 2

2. Semantic check (bảng 2.2 A/B/D/E)
   FAIL → reward = R_wf_pass + R_sem_fail(liên tục theo số vi phạm) → DỪNG, không chấm outcome
   PASS → tiếp bước 3

3. Outcome check (forward-test hoặc counterfactual — mục 6)
   reward = R_wf_pass + R_sem_pass + R_outcome × w[trend][action_type]
```

Lý do không cộng tuyến tính: tránh completion sai cấu trúc/ngữ nghĩa nhưng outcome thắng may mắn vẫn đạt điểm cao. Gate cứng đảm bảo outcome chỉ có ý nghĩa khi gắn với chuỗi suy luận hợp lệ.

### 5.2 Outcome theo action type

| Action | Có outcome reward? | Cách tính |
|---|---|---|
| BUY / SELL | Có | forward-test thật trên `future_bins`, R-multiple thực đạt (mục 6.1) |
| CANCEL_BUY / CANCEL_SELL | Có (**counterfactual**) | forward-test giả lập như thể đã vào lệnh, đảo dấu R-multiple (mục 6.2) |
| WAIT_BUY / WAIT_SELL | Không | chỉ well-form + semantic |
| HOLD | Không | chỉ well-form + semantic |

### 5.3 Weight điều chỉnh tay theo round — `w[trend][action_type]`

- Thống kê **trend-conditional**: `stat[trend][action] = số lần action xuất hiện khi trend=X / tổng rollout có trend=X`, đo theo rolling window (không cộng dồn từ đầu).
- Bảng `w[trend][action]` khởi tạo = 1.0, tay chỉnh giữa các round dựa trên thống kê + chất lượng outcome đi kèm (không chỉ tần suất — tần suất cao nhưng outcome tốt thì không cần dìm).
- Vai trò theo round: round đầu (entropy/temperature cao) dùng để explore, weight gần đều — mục tiêu quan sát xu hướng tự nhiên. Round giữa chỉnh weight để kéo nhánh bị bỏ quên lên / dìm nhánh bị lạm dụng. Round cuối anneal weight dần về 1.0 (hoặc phản ánh đúng chất lượng thật) — **không giữ ép cân bằng cứng mãi mãi**, vì mục tiêu cuối là đủ exploration để học, không phải phân phối đều tuyệt đối.
- Vận hành: `w` là 1 dict Python global/module-level, sửa tay giữa các lần gọi `trainer.train()` (round rời rạc, resume từ checkpoint) — không cần sửa code reward_func.

---

## 6. Forward-test / Counterfactual Engine

Tất cả tính toán dựa **hoàn toàn trên bin** (không cần decode ngược về giá thật, không cần anchor_open/anchor_atr tại thời điểm chấm điểm) — vì `future_bins` được encode bằng cùng anchor với `chart_block` (input), nên so sánh SL/TP/entry (đều là bin) với O/H/L/C tương lai (cũng bin) là phép so sánh số nguyên thuần túy.

### 6.1 BUY/SELL thật
- Entry = bin `current_price` (= Close nến cuối input).
- SL = bin trong action_block (model tự chọn).
- Target (TP) = derive từ entry, SL, RR: `target = entry + RR × (entry - SL)` (long) hoặc `target = entry - RR × (SL - entry)` (short) — risk luôn = 1 theo định nghĩa. RR không dùng trực tiếp làm căn cứ điểm — chỉ dùng để derive TP, sau đó forward-test xác nhận R-multiple thực đạt.
- **Ràng buộc khoảng cách SL (semantic check, số bin cố định do người cấu hình set ngoài, KHÔNG derive theo ATR)**:
  ```python
  SL_MIN_DIST_BINS = 5    # ví dụ, set tay
  SL_MAX_DIST_BINS = 10   # ví dụ, set tay

  def is_sl_valid(action_type, entry_bin, sl_bin, zone) -> bool:
      dist = abs(entry_bin - sl_bin)
      if not (SL_MIN_DIST_BINS <= dist <= SL_MAX_DIST_BINS):
          return False
      if action_type == "BUY":
          return sl_bin < zone.lower_bin   # SL phải nằm dưới đáy zone_support
      else:  # SELL
          return sl_bin > zone.upper_bin   # SL phải nằm trên đỉnh zone_resistance
  ```
  Vi phạm (khoảng cách sai hoặc sai phía zone) → semantic fail, không tính outcome. Lưu ý: nếu multi-symbol, cùng 1 cặp bin cố định tương ứng biên độ ATR-thực khác nhau tùy `scale` từng symbol — v1 (1 symbol) không vấn đề, cần xem lại nếu mở rộng đa symbol (có thể cần set riêng theo symbol).
- Forward-test trên **toàn bộ 50 nến `future_bins`** (horizon = hết 50 nến, không dừng sớm).
- Case biên SL/TP cùng chạm 1 nến tương lai: **ưu tiên SL** (conservative — nếu 1 nến gap qua cả 2 mức, coi như SL chạm trước).
- Case biên timeout (đi hết 50 nến mà chưa chạm SL/TP): **reward trung tính = 0**. Lý do chủ động chọn 0 thay vì phạt: nếu phạt timeout, model sẽ bị đẩy về xu hướng đứng ngoài (né mọi lệnh có nguy cơ timeout) thay vì học cách tự điều chỉnh RR nhỏ lại cho hợp lý — để model tự tìm ra RR phù hợp qua outcome, không ép bằng phạt cứng.
- **Case biên bin bị bão hòa (clip 0/1023)**: nếu target_bin tính ra nằm ngoài [0, 1023] (RR quá xa so với phạm vi mà `scale × ATR` mô tả được) → coi là setup không hợp lệ, phạt semantic, không tính outcome.

### 6.2 CANCEL_BUY / CANCEL_SELL (counterfactual)
```python
def counterfactual_outcome(action_type, zone, future_bins, current_price_bin):
    if action_type == "CANCEL_BUY":
        entry = current_price_bin
        sl = zone.lower_bin - 1        # buffer 1 bin dưới mép zone_support
        target = entry + 1 * (entry - sl)   # RR cố định = 1
        direction = "long"
    else:  # CANCEL_SELL
        entry = current_price_bin
        sl = zone.upper_bin + 1        # buffer 1 bin trên mép zone_resistance
        target = entry - 1 * (sl - entry)
        direction = "short"

    result = forward_test(entry, sl, target, future_bins, direction)  # dùng chung hàm với 6.1
    r_multiple = -result.r_multiple    # ĐẢO DẤU: CANCEL đúng khi lẽ ra sẽ thua
    return r_multiple
```
SL/RR cho CANCEL không lấy từ model output (bị cấm ở well-form) — hoàn toàn derive tự động từ zone + buffer cố định 1 bin + RR=1.

---

## 7. Dataset Design

### 7.1 Nguồn dữ liệu chung (Preprocess + ChartCodec — đã có sẵn)
- Mỗi window lấy 100 nến: 50 đầu = input (`chart_block`), 50 sau = `future_bins` (dùng cho forward-test/counterfactual).
- Encode bằng `ChartCodec`, cùng 1 `anchor_open`/`anchor_atr` cho toàn bộ 100 nến — đảm bảo input và future_bins nhất quán trong cùng hệ bin.
- `scale` cố định theo symbol + timeframe (bảng `XAUUSD_M15_SCALE`...) — v1 chỉ 1 timeframe, không cần cột `timeframe`.
- Ngưỡng zone-width (min/max) và ngưỡng khoảng cách SL: **số bin cố định do người tạo dữ liệu set trực tiếp bên ngoài** (không derive qua `atr_to_bins`/ATR-multiple — nhất quán với cách xử lý SL ở mục 6.1):
```python
ZONE_WIDTH_MIN_BINS = ...   # set tay
ZONE_WIDTH_MAX_BINS = ...   # set tay
SL_MIN_DIST_BINS = 5        # set tay
SL_MAX_DIST_BINS = 10       # set tay
```

### 7.2 Schema Pretrain/SFT (dạng prompt+completion, train cross-entropy chuẩn — KHÔNG qua GRPOTrainer)
```python
{
    "prompt": str,       # "<chart>...50 nến...</chart>"
    "completion": str,   # "<think>...</think><action>...</action>" — random gen có kiểm soát semantic
}
```
- `chart_block`: luôn lấy từ chart thật, không random.
- `current_price`: luôn tính từ chart thật (= Close nến cuối), không random.
- `trend`, `zone` (vị trí/độ rộng/hướng): random thuần túy trong ngưỡng hợp lệ.
- `price_in_zone`: **derive** từ chart thật + zone đã random (mục 2.2.D) — không random độc lập.
- `action_type`, `good_price_action`, `SL`, `RR`: random có điều kiện theo các field trên (bảng 2.2.E/F).
- Thứ tự sinh đúng: random zone → tính `price_in_zone` thật từ chart → random action phù hợp — không random `price_in_zone` độc lập rồi mới random zone (dễ tạo mẫu mâu thuẫn).
- Sample uniform trên **toàn bộ leaf-path hợp lệ** của cây ràng buộc (liệt kê tường minh trước), không sample uniform từng field rồi lọc bỏ invalid (méo phân phối).

### 7.3 Schema GRPO (chỉ prompt — model tự sinh phần còn lại)
```python
{
    "prompt": str,            # "<chart>...50 nến...</chart>"
    "future_bins": list,      # [[o,h,l,c], ...] x 50 nến, bin thô cùng hệ với input
    "symbol": str,            # tra scale constant + ngưỡng ATR-bin precomputed
    "window_id": str,         # symbol + start_index — dedup, chống leakage
}
```

### 7.4 Chống leakage
- **Window chồng lấn**: `stride=50`, `window_size=100` → future_bins của window A = input của window B kế tiếp. Cần split pretrain/SFT/GRPO theo **khối thời gian liên tục không chồng lấn** (theo mốc ngày/tháng), không chia ngẫu nhiên theo từng window riêng lẻ.
- `window_id` dùng để dedup giữa 3 splits.
- ATR tính tại thời điểm đóng nến cuối chart input — không bao gồm future_bins (leakage nhẹ nếu tính sai).
- Multi-symbol: đảm bảo mỗi symbol có mặt ở cả 3 splits theo tỷ lệ hợp lý.

---

## 8. Tích hợp TRL GRPOTrainer

### 8.1 Nguyên tắc quan trọng nhất
**KHÔNG** tách 3 gate thành 3 `reward_funcs` riêng kèm `reward_weights` — TRL cộng dồn (tổng có trọng số) các reward_funcs, đây là cộng tuyến tính, mâu thuẫn với thiết kế gate cứng ở mục 5.1.

→ Viết **1 hàm `reward_func` duy nhất**, tự implement toàn bộ gate bên trong, trả về 1 list float đã gate xong:

```python
def unified_reward_func(prompts, completions, future_bins, symbol, **kwargs):
    rewards = []
    for completion, fb, sym in zip(completions, future_bins, symbol):
        parse_result = parser.parse(completion)
        if not parse_result.is_well_formed():
            rewards.append(parse_result.well_form_score)
            continue
        sem_result = semantic_checker.check(parse_result.ast)
        if not sem_result.passed:
            rewards.append(R_WF_FULL + sem_result.score)
            continue
        action_type = parse_result.ast.action.action_type
        if action_type in ("BUY", "SELL"):
            outcome = forward_test(parse_result.ast, fb)
        elif action_type in ("CANCEL_BUY", "CANCEL_SELL"):
            outcome = counterfactual_outcome(action_type, parse_result.ast.think.zone, fb,
                                              parse_result.ast.think.current_price_bin)
        else:  # WAIT_*, HOLD
            outcome = 0.0
        trend = parse_result.ast.think.trend
        w = weight_table[trend][action_type]
        rewards.append(R_WF_FULL + R_SEM_FULL + outcome * w)
        stats_collector.log(trend, action_type, outcome)  # cho thống kê round-based
    return rewards
```

### 8.2 Config bắt buộc
```python
training_args = GRPOConfig(
    ...,
    remove_unused_columns=False,   # BẮT BUỘC — nếu không TRL tự xóa hết cột trừ "prompt"
    num_generations=12,            # group size đủ lớn cho advantage + thống kê trend/action có ý nghĩa
)
```

### 8.3 Vòng lặp round-based
Gọi `trainer.train()` nhiều lần với `resume_from_checkpoint`; giữa các lần, đọc `stats_collector` (tần suất + outcome trung bình theo `trend × action_type`), tay chỉnh `weight_table`, train tiếp. Đơn giản, dễ kiểm soát hơn callback tự động.

---

## 9. Câu hỏi còn mở / cần chốt trước khi code

**Đã chốt:**
1. ~~Horizon forward-test~~ → toàn bộ 50 nến `future_bins`, không dừng sớm.
2. ~~Gap SL/TP cùng nến~~ → ưu tiên SL (conservative).
3. ~~Timeout~~ → reward trung tính = 0 (tránh khuyến khích đứng ngoài; để model tự điều chỉnh RR qua outcome).
4. ~~Vocab RR~~ → vocab riêng, giới hạn 1–9.
6. ~~Ngưỡng pass gate semantic~~ → 100%, phải pass toàn bộ mới được tính outcome (không dùng threshold %).
7. ~~Ngưỡng zone-width~~ → số bin cố định set tay ngoài (không derive theo ATR), nhất quán với cách xử lý SL.

**Còn mở:**
5. Trọng số cụ thể giữa `R_wf_pass`, `R_sem_pass`, `R_outcome` — **quyết định sau, còn quá sớm để chốt số cụ thể** trước khi có dữ liệu thực nghiệm từ các round GRPO đầu.

---

## 10. Việc cần làm tiếp theo

- [ ] Viết Lexer (regex-based) theo token spec mục 3.
- [ ] Viết Parser với panic-mode recovery + kiểm tra field bắt buộc/cấm theo action_type (mục 4.2, bảng 2.2.F).
- [ ] Viết AST nodes (mục 4.3, đã bao gồm ZoneNode có direction).
- [ ] Viết Semantic Checker (mục 2.2 A/B/D/E → verifier, lật ngược generator).
- [ ] Viết Generator cho pretrain/SFT (mục 7.2 — thứ tự sinh đúng: zone → price_in_zone derive → action).
- [ ] Viết Forward-test + Counterfactual engine (mục 6) — sau khi chốt mục 9.1–9.3.
- [ ] Viết `unified_reward_func` + `StatsCollector` + `weight_table` (mục 8.1).
- [ ] Build dataset GRPO (mục 7.3) từ `ChartCodec`/`Preprocess` có sẵn, đảm bảo split chống leakage (mục 7.4).
- [ ] Instrument logging: tỷ lệ pass từng gate theo round, phân bố trend×action, outcome trung bình theo round.
