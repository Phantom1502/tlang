# DOC: `app/lang/` — Lexer / Parser / AST / Semantic Checker

Version: 1.0 (đi kèm `spec_trading_llm_v0.3.md`). Đây là module **core**, ổn định nhất trong toàn bộ
project — mọi module khác (`tokenizer`, `data_prepare`, `training`) đều phụ thuộc vào đây, chiều
ngược lại không xảy ra.

Phạm vi: `app/lang/tokens.py`, `lexer.py`, `ast_nodes.py`, `parser.py`, `semantic.py`. Grammar đầy đủ
xem `spec_trading_llm_v0.3.md` mục 2 — doc này chỉ nói implementation, không lặp lại grammar.

---

## 1. `tokens.py` — nguồn sự thật duy nhất cho hằng số vocab

```python
BIN_MIN = 0
BIN_MAX = 1023
DIGIT_PAD = 4
RR_MIN = 1
RR_MAX = 9
```

Mọi module khác (tokenizer, generator, forward_test) phải **import lại từ đây**, không tự định
nghĩa lại range. Đây là bài học từ 1 bug thực tế (tokenizer từng tự định nghĩa lại `DIGIT_PAD`, lệch
với bản ở đây) — xem thêm mục 5 (nguyên tắc "tokenizer không tự vá lành").

`TokenType` liệt kê đủ mọi loại token trong grammar (structural tags, candle atomic, digit,
action_type enum, RR bracket-enum, `UNKNOWN`, `EOF`).

---

## 2. `lexer.py` — regex-based, KHÔNG BAO GIỜ raise

Master pattern gộp toàn bộ token spec thành 1 regex hợp nhất (`_MASTER_RE`), mỗi alternative có
đúng 1 named group ngoài cùng (không group con lồng nhau) để `match.lastgroup` luôn trỏ đúng
`TokenType`.

Thứ tự pattern trong `_TOKEN_SPEC` có ý nghĩa: các pattern cụ thể hơn (`SL_LABEL = "SL:"`) đứng
trước pattern tổng quát hơn (`DIGIT`, `COLON`) để tránh nhầm nghĩa — dù về mặt kỹ thuật `"SL:"` không
xung đột với `[0-9]`/`":"` (ký tự đầu khác nhau), thứ tự vẫn được giữ tường minh cho dễ đọc.

**Thiết kế cố ý không raise exception khi gặp ký tự lạ** — gói thành `TokenType.UNKNOWN` (giữ
nguyên vị trí + nội dung) thay vì crash. Lý do: Lexer chạy trực tiếp trong `reward_func` của GRPO
trên completion do model tự sinh — 1 completion sai be bét vẫn phải tokenize được để `Parser` chấm
điểm `well_form_score` liên tục, thay vì toàn bộ pipeline reward sụp đổ.

---

## 3. `parser.py` — recursive-descent, panic-mode error recovery

```
program      := chart_block think_block action_block
chart_block  := "<chart>" candle{50} "</chart>"
think_block  := "<think>" trend current_price zone? price_in_zone? good_price_action? "</think>"
action_block := "<action>" ACTION_TYPE [ SL RR ] "</action>"
```

### 3.1 Panic-mode recovery

Khi gặp token sai: ghi nhận lỗi (`ParseError{message, position, severity}`) rồi bỏ qua token tới
điểm đồng bộ hoá gần nhất (`SYNC_TOKENS = {CHART_CLOSE, THINK_OPEN, THINK_CLOSE, ACTION_OPEN,
ACTION_CLOSE, EOF}`), tiếp tục parse phần còn lại — KHÔNG hard-fail như compiler thật. Nhờ vậy 1
completion nhiều lỗi vẫn có `well_form_score()` liên tục thay vì tất cả về 0 giống nhau.

### 3.2 `well_form_score` — liên tục, không nhị phân

```python
SEVERITY_PENALTY = {"structural": 0.15, "value": 0.30}
well_form_score = max(0.0, 1.0 - sum(SEVERITY_PENALTY[e.severity] for e in errors))
```

`"value"` nặng hơn `"structural"` — phản ánh model **đọc sai nội dung** (vd `current_price` không
khớp Close nến cuối, số digit sai) nghiêm trọng hơn lỗi cú pháp thuần tuý (thiếu 1 tag đóng).

**Điểm quan trọng nhất khi tích hợp GRPO** (bug thật đã gặp và sửa — xem `spec` mục 8.1): `Parser`
LUÔN cần được gọi trên **`prompt + " " + completion`** (chart + think + action đầy đủ), KHÔNG BAO
GIỜ gọi trên `completion` một mình. Vì `_parse_chart_block()` luôn chạy đầu tiên trong `parse()` và
bắt buộc phải thấy `<chart>` — thiếu chart luôn tạo ra đúng 1 lỗi structural cố định
(`well_form_score ≈ 0.85`), che mất hoàn toàn tín hiệu thật về chất lượng think/action bên trong.

### 3.3 Kiểm tra bảng C và F (well-form, không phải semantic)

- **Bảng C** (`_check_current_price_matches_chart`): `current_price` phải khớp tuyệt đối bin Close
  nến cuối — severity `"value"`.
- **Bảng F** (`_check_action_field_consistency`): field bắt buộc/cấm theo `action_type` (SL/RR/
  good_price_action) — đọc được ngay trong cùng block, về bản chất vẫn là "đúng/sai ngữ pháp có điều
  kiện" (grammar hơi context-sensitive), chưa đánh giá chất lượng quyết định.

### 3.4 AST Nodes (`ast_nodes.py`)

```python
CandleNode(o, h, l, c)
ChartNode(candles: List[CandleNode])
ZoneNode(direction: "support"|"resistance", lower_bin, upper_bin)
ThinkNode(trend, current_price_bin, zone: Optional[ZoneNode], price_in_zone: bool, good_price_action: bool)
ActionNode(action_type, sl: Optional[int], rr: Optional[int])   # rr LUÔN là int (1-9), không phải float
ProgramNode(chart, think, action)
```

---

## 4. `semantic.py` — SemanticChecker (bảng A/B/D/E/H)

```python
SemanticChecker(
    zone_width_min_bins: int = ZONE_WIDTH_MIN_BINS,   # default 5 — class constant
    zone_width_max_bins: int = ZONE_WIDTH_MAX_BINS,    # default 20
)
```

**Vì sao có tham số thay vì hardcode tuyệt đối**: `generator.py` (sinh data pretrain/SFT) và mọi
demo gọi `SemanticChecker()` không tham số — hành vi y hệt trước (default = hằng số hardcode). CHỈ
nhánh GRPO (`reward_func.py`) truyền tường minh 2 giá trị này từ `RoundConfig` hiện tại, vì chỉ tới
GRPO mới có outcome thật để biết nên nới/siết ngưỡng zone-width thế nào (xem `docs/training.md`
mục RoundConfig).

### 4.1 5 rule kiểm tra (gọi theo thứ tự trong `check()`)

| Rule | Hàm | Ý nghĩa |
|---|---|---|
| A | `_check_trend_zone` | UP→chỉ zone_support, DOWN→chỉ zone_resistance |
| B | `_check_zone_direction_vs_price` | Zone phải nằm đúng phía current_price |
| **H** | `_check_zone_width` | `zone_width_min_bins <= width <= zone_width_max_bins` |
| D | `_check_price_in_zone_geometry` | `price_in_zone` phải khớp hình học thật (current_price trong zone, hoặc 5 nến cuối chạm zone) |
| E | `_check_action_group` | Action hợp lệ theo `price_in_zone`/hướng zone |

Rule **H** (`_check_zone_width`) là bổ sung **mới nhất** — trước đây `generator.py` tự định nghĩa
`ZONE_WIDTH_MIN_BINS`/`MAX_BINS` làm bản PLACEHOLDER riêng để dùng lúc sinh data, nhưng KHÔNG verifier
nào kiểm tra lại lúc GRPO rollout — vi phạm nguyên tắc "verifier = lật ngược generator" bên dưới.
Đã fix: 2 hằng số này giờ sống ở `SemanticChecker` (class constant), `generator.py` import lại từ
đây thay vì tự định nghĩa bản riêng.

### 4.2 Nguyên tắc "verifier = lật ngược generator"

Generator (data cho pretrain/SFT) đảm bảo đúng mọi invariant NÀY lúc sinh ("by construction").
`SemanticChecker` chỉ cần lật ngược đúng logic đó thành kiểm tra. Bài học từ rule H: **mọi invariant
mới thêm sau này phải tự hỏi "generator có tôn trọng cái này khi sinh không? Nếu có, verifier phải
kiểm tra lại"** — nếu không, GRPO rollout có thể khai thác khoảng trống mà SFT/pretrain chưa từng
cho model thấy ví dụ phản diện.

Rule **G** (`good_price_action`) là ngoại lệ CHỦ Ý — không có rule kiểm tra nội dung, để tránh áp đặt
bias chủ quan; ý nghĩa học hoàn toàn qua outcome reward lan truyền ngược ở GRPO.

---

## 5. Nguyên tắc xuyên suốt: "tokenizer/lexer không tự vá lành"

Bài học từ 1 bug thực tế: bản tokenizer đầu tiên (phía `app/tokenizer/`) từng tự "vá lành" cấu trúc
hỏng trong `convert_tokens_to_string` — tự bịa giá trị/tag đóng khi thiếu digit hoặc thiếu tag, khiến
lỗi well-form bị che giấu trước khi tới `Parser`. Nguyên tắc khắc phục, áp dụng cho MỌI tầng:

- **Lexer/Parser** (ở đây) là **tầng DUY NHẤT** được phép khoan dung lỗi cấu trúc (panic-mode
  recovery), và phải làm việc đó MỘT CÁCH TƯỜNG MINH (ghi nhận lỗi vào `errors`, không âm thầm sửa).
- **Tokenizer** (`app/tokenizer/`) phải là 1 phép ánh xạ trung thực — không tự sửa bất kể model sinh
  đúng hay sai. Xem `docs/tokenizer.md` mục "Nguyên tắc ánh xạ trung thực".

---

## 6. Demo / kiểm chứng nhanh

```bash
python -m demos.lang_demo        # Lexer + Parser — các case điển hình (well-formed, thiếu field, sai giá trị...)
python -m demos.semantic_demo    # SemanticChecker — bảng A/B/D/E/H trên completion đã well-formed
```

Không phải unit test chính thức (pytest sẽ thêm sau), chỉ để kiểm tra nhanh trước khi ghép vào
`reward_func`.