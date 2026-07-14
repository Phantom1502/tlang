# DOC: Tokenizer (Model / Vocab / Train Pipeline integration)

Version: 0.1 — phụ lục cho `docs/spec_trading_llm_v0.2.md` (mục 3) và
`docs/train_pipeline_v0.1.md` (mục 2). Implement thật cho phần tokenizer đã
được mô tả (nhưng chưa code) ở 2 doc trên.

Code: `app/tokenizer/vocab_builder.py`, `app/tokenizer/build_tokenizer.py`,
`app/tokenizer/tokenizer_demo.py`.

---

## 1. Quyết định: WordLevel, vocab đóng cố định — KHÔNG BPE/Unigram/WordPiece

Đây là quyết định đã chốt sẵn trong spec (mục 3), doc này chỉ implement đúng
theo đó — không đổi hướng dù dataset pretrain quy mô ~10B token.

**Lý do BPE/Unigram không phù hợp ở đây, bất kể scale dữ liệu:**

- Ngôn ngữ think/action là 1 **grammar hình thức đóng** — toàn bộ token hợp
  lệ đã liệt kê tường minh trong `app/lang/tokens.py` (`BIN_MIN/BIN_MAX/
  DIGIT_PAD/RR_MIN/RR_MAX`) và bảng token spec của `app/lang/lexer.py`.
  Không có khái niệm "từ mới xuất hiện trong corpus" (open vocabulary) —
  tiền đề mà BPE được thiết kế để giải quyết không tồn tại ở đây.
- 10B token pretrain **không tạo ra token mới nào** ngoài ~4145 token đã
  biết trước — càng nhiều dữ liệu chỉ càng củng cố phân phối tần suất trên
  đúng tập token đó, không có OOV để BPE "học" thêm.
- Nghiêm trọng hơn: BPE học merge theo tần suất thống kê, có thể tự ý gộp 2
  digit token đứng cạnh nhau (`"0"` + `"5"` → `"05"`), **phá vỡ trực tiếp**
  bất biến digit-decompose mà spec cố tình thiết kế để model học so sánh/số
  học trên `current_price`/`zone`/`SL` (xem spec mục 3, và bài học thực tế đã
  ghi trong đó về bug tokenizer cũ tự "vá lành" cấu trúc hỏng).

**Điều 10B token thực sự thay đổi: không phải thuật toán, mà là cách implement (mục 5).**

---

## 2. Vocab design

### 2.1 Nguồn sự thật duy nhất

`app/tokenizer/vocab_builder.py` import trực tiếp `BIN_MIN, BIN_MAX, RR_MIN,
RR_MAX` từ `app.lang.tokens` — không tự định nghĩa lại range nào. Nếu range
bin hoặc RR đổi ở `app/lang/tokens.py`, vocab tự cập nhật theo mà không cần
sửa `vocab_builder.py`.

### 2.2 Bảng nhóm token (thứ tự = thứ tự gán id, xem mục 3)

| # | Nhóm | Ví dụ | Số lượng |
|---|---|---|---|
| — | special | `<unk>` `<bos>` `<eos>` `<pad>` | 4 |
| 1 | structural_tags | `<chart>` `</chart>` `<think>` `</think>` `<action>` `</action>` | 6 |
| 2 | candle_O / candle_H / candle_L / candle_C | `<O_0>`…`<O_1023>` (×4 field) | 4×1024 = 4096 |
| 3 | trend | `<trend>UP</trend>` `<trend>DOWN</trend>` `<trend>RANGE</trend>` | 3 |
| 4 | digit_field_tags | `<current_price>` `</current_price>` `<zone_support>` `</zone_support>` `<zone_resistance>` `</zone_resistance>` `SL:` | 7 |
| 5 | digit | `0`…`9` | 10 |
| 6 | colon | `:` | 1 |
| 7 | flags | `<price_in_zone>` `<good_price_action>` | 2 |
| 8 | action_type | `BUY` `SELL` `CANCEL_BUY` `CANCEL_SELL` `WAIT_BUY` `WAIT_SELL` `HOLD` | 7 |
| 9 | rr | `<RR_1>`…`<RR_9>` | 9 |

**Tổng = 4 + 4141 = 4145 token** (khớp gần đúng ước lượng "~4146" trong
`train_pipeline_v0.1.md` — chênh lệch 1 không đáng kể, do cách đếm
`SL:`/colon ở 2 bảng khác nhau 1 chút; số 4145 này là số **thật, tính ra
bằng code**, dùng số này làm chuẩn thay vì con số ước lượng tay trong doc cũ).

Chạy `python -m app.tokenizer.vocab_builder` để in lại bảng đếm này bất cứ lúc
nào (tự đối chiếu, không cần tin vào doc).

### 2.3 Vì sao KHÔNG có token digit riêng cho từng field

`current_price`, `zone.lower`, `zone.upper`, `SL` đều dùng chung 10 token
`0`–`9` (không phải `D_current_price_0`, `D_zone_0`... riêng biệt). Đây là
điểm quan trọng nhất của thiết kế digit-decompose (spec mục 3): dùng chung 1
không gian embedding giúp model học quan hệ thứ tự/so sánh giữa các field này
tự nhiên hơn, thay vì phải học lại từ đầu cho từng field trong không gian
embedding tách biệt.

### 2.4 Chart OHLC vẫn atomic — không digit-decompose

`<O_543>`, `<H_543>`... giữ nguyên 1 token/giá trị (không tách digit). Model
chỉ cần *đọc* chart, không so sánh/tính toán trực tiếp trên OHLC — digit-
decompose ở đây chỉ tốn thêm token (4 → ~16 token/nến) mà không có lợi ích
tương ứng (đúng quyết định đã chốt trong spec).

---

## 3. Special tokens & id assignment

```
id=0  <unk>
id=1  <bos>
id=2  <eos>
id=3  <pad>
id=4..  (9 nhóm còn lại, tuần tự theo thứ tự bảng 2.2)
```

Id 1/2/3 khớp đúng `LlamaConfig(bos_token_id=1, eos_token_id=2,
pad_token_id=3)` đã quy ước trong `train_pipeline_v0.1.md` mục 1.1 — không
cần sửa gì ở model config.

**QUAN TRỌNG — vocab contract:** thứ tự nhóm trong `_build_groups()` (file
`vocab_builder.py`) quyết định id tuyệt đối của từng token. Nếu sau này cần
thêm token mới (vd mở rộng `BIN_MAX`, thêm `action_type` mới), **chỉ được
thêm vào CUỐI danh sách nhóm hiện có** (hoặc thêm nhóm mới ở cuối) — không
chèn giữa, không đổi thứ tự nhóm cũ. Đổi thứ tự = đổi id = mọi checkpoint
model đã train (embedding table đã học theo id cũ) không tương thích nữa.

---

## 4. Chiến lược pre-tokenization: Whitespace-only là đủ

Không cần viết lại 1 pre-tokenizer regex mô phỏng `_MASTER_RE` của
`app/lang/lexer.py`. Lý do: generator (`app/gen/generator.py`, các hàm
`_build_completion_text`/`_build_chart_text`) luôn in **mỗi token cách nhau
đúng 1 khoảng trắng**, kể cả từng digit rời — điều kiện này đã được ghi rõ và
đảm bảo trong spec (mục 3: "generator phải in token cách nhau đúng 1 khoảng
trắng"). Vì vậy:

- Pre-tokenizer chỉ cần tách theo whitespace (`tokenizers.pre_tokenizers.
  WhitespaceSplit` — **không phải** `Whitespace` mặc định, vì `Whitespace`
  còn tách theo punctuation/`\w+` boundary, sẽ làm vỡ token như `<O_543>`
  chứa ký tự không phải `\w`).
- Sau khi tách, mỗi "từ" đã đúng là 1 token hoàn chỉnh trong vocab (hoặc
  không có trong vocab → `<unk>`, xem mục 6). `WordLevel` model chỉ cần tra
  bảng exact-match, không cần logic gì thêm.

Đây chính là thiết kế đã được dự kiến sẵn trong spec ("WordLevel +
WhitespaceSplit pre-tokenizer đơn giản, không có chỗ để hành vi ẩn xảy ra").

---

## 5. Điều thực sự cần chỉnh vì scale ~10B token

Chọn *loại* tokenizer không đổi vì scale, nhưng *cách implement* phải khác so
với một `PreTrainedTokenizer` thuần Python nếu muốn không bị nghẽn cổ chai:

1. **Dùng `tokenizers` (Rust backend), không phải `PreTrainedTokenizer` thuần
   Python.** Đã implement đúng vậy: `Tokenizer(WordLevel(...))` +
   `PreTrainedTokenizerFast` wrapper (xem `build_tokenizer.py`). Ở ~10B token
   (~40-45M sample × ~230 token/sample), regex-match bằng Python thuần cho
   từng sample sẽ là bottleneck thật khi tokenize dataset.

2. **`dataset_mode` (mục 3.3 `train_pipeline_v0.1.md`) nên nghiêng về
   `"pre_tokenized"` sớm hơn dự kiến cho pretrain/SFT** — 2 dataset này không
   đổi qua các round (khác GRPO, vốn phải encode runtime vì completion do
   model tự sinh lúc train). Ở 10B token, tokenize lại mỗi epoch bằng
   on-the-fly là phí compute không cần thiết — nên `.map()` một lần, cache
   `input_ids` (local hoặc push lên 1 Hub repo riêng).

3. **Vocab 4145 token → embedding table rất nhỏ** so với `hidden_size`
   (128–768 theo các preset model) và `tie_word_embeddings=True` đã chốt sẵn
   — không có áp lực phải nén vocab kiểu BPE để giảm tham số ở bất kỳ preset
   nào (`tiny`→`large`).

---

## 6. Nguyên tắc "ánh xạ trung thực, không tự vá lành"

Theo đúng bài học đã ghi trong spec mục 3 (bug tokenizer cũ tự bịa giá trị/
tag đóng khi thiếu digit hoặc thiếu tag): tokenizer này **không** có logic tự
sửa completion hỏng ở bất kỳ bước nào.

- Token không có trong vocab (completion rác từ GRPO rollout, hoặc bin ngoài
  `[BIN_MIN, BIN_MAX]`) → map thẳng về `<unk>`, KHÔNG bị loại bỏ âm thầm,
  KHÔNG được suy diễn lại thành token hợp lệ gần nhất.
- Chỉ 1 tầng duy nhất — Parser (`app/lang/parser.py`, đã có panic-mode
  recovery) — chịu trách nhiệm khoan dung lỗi cấu trúc để tính
  `well_form_score` liên tục. Tokenizer encode/decode đứng ngoài hoàn toàn
  logic đó.
- Đã kiểm chứng bằng `tokenizer_demo.py` mục 3 (input rác, bin ngoài vocab,
  enum lạ như `SIDEWAYS`) — không crash, `<unk>` xuất hiện đúng chỗ.

---

## 7. File & cách dùng

```
app/tokenizer/
├── vocab_builder.py     # build_vocab() -> Dict[str, int], describe_vocab()
├── build_tokenizer.py   # build_fast_tokenizer() -> PreTrainedTokenizerFast — CHỈ dùng bởi push_to_hub.py
├── push_to_hub.py       # build từ source rồi push lên HF Hub — chạy 1 lần / mỗi khi vocab đổi
├── hub.py                # load_tokenizer() -> PreTrainedTokenizerFast — NGUỒN DUY NHẤT cho mọi nơi khác
└── tokenizer_demo.py     # kiểm chứng nhanh (round-trip, seq_len, unk, digit-decompose)
```

### 7.0 Luồng chuẩn: build 1 lần, push lên Hub, mọi nơi khác load từ Hub

Từ giờ trở đi, **chỉ `push_to_hub.py` được gọi `build_fast_tokenizer()`**.
Mọi script khác (train pretrain/SFT/GRPO, data prep, demo) gọi
`app.tokenizer.hub.load_tokenizer()`.

**Lý do tập trung vào Hub thay vì build lại mỗi nơi:**

- Train chạy trên nhiều session Colab/Kaggle khác nhau, mỗi session có thể
  cài version `tokenizers`/`transformers` hơi khác nhau theo thời gian — build
  lại từ source ở mỗi máy có rủi ro (dù nhỏ) ra vocab lệch nhau. Load đúng 1
  file `tokenizer.json` đã chốt trên Hub loại bỏ rủi ro này hoàn toàn.
- Model checkpoint tie embedding theo đúng `vocab_size`/thứ tự id của
  tokenizer lúc train (mục 3 — "vocab contract") — tokenizer trên Hub là 1
  **artifact bất biến** gắn với các checkpoint đó. Đổi vocab phải là hành
  động tường minh (push version mới), không phải hệ quả ngẫu nhiên của việc
  build lại ở đâu đó.

**Bước 1 — build & push (chạy 1 lần, hoặc khi vocab thật sự đổi):**

```bash
# Xem trước sẽ push gì, KHÔNG push thật
python -m app.tokenizer.push_to_hub --repo_id <org>/trading-llm-tokenizer --dry_run

# Push thật (cần huggingface-cli login hoặc export HF_TOKEN=hf_xxx trước)
python -m app.tokenizer.push_to_hub --repo_id <org>/trading-llm-tokenizer
```

Sau khi push xong, cập nhật `DEFAULT_TOKENIZER_REPO` trong `app/tokenizer/
hub.py` thành đúng `repo_id` vừa dùng — script sẽ tự nhắc lại điều này.

**Bước 2 — mọi nơi khác load qua `hub.load_tokenizer()`:**

```python
from app.tokenizer.hub import load_tokenizer

tok = load_tokenizer()                       # dùng DEFAULT_TOKENIZER_REPO
tok = load_tokenizer(revision="v2")          # pin đúng 1 version cụ thể trên Hub
tok = load_tokenizer(repo_id="org/staging")  # test 1 repo khác trước khi đổi default
```

`load_tokenizer()` mặc định có **fallback build local** nếu không load được
từ Hub (chưa push lần nào, mất mạng, sai tên repo) — kèm cảnh báo
(`warnings.warn`) rõ ràng, CHỈ dùng cho dev/test cục bộ. Khi chạy train
thật, nên gọi `load_tokenizer(allow_local_fallback=False)` để lỗi mạng/config
sai lộ ra ngay thay vì âm thầm train bằng 1 tokenizer build-lại-tại-chỗ có
thể lệch với tokenizer đã dùng ở round trước.

### 7.1 Build & lưu tokenizer ra đĩa (chỉ dùng nội bộ bởi push_to_hub.py, hoặc để debug)

```bash
python -m app.tokenizer.build_tokenizer --out_dir ./tokenizer_out
```

Ghi ra `tokenizer.json` + `tokenizer_config.json` — dùng để kiểm tra thủ công
trước khi push, KHÔNG phải cách các script khác lấy tokenizer trong luồng
chuẩn (xem mục 7.0).

### 7.2 Dùng trong code (luồng chuẩn — mọi script sau bước push)

```python
from app.tokenizer.hub import load_tokenizer

tok = load_tokenizer()
ids = tok.encode(prompt + " " + completion)   # tự thêm <bos>/<eos>
text_back = tok.decode(ids, skip_special_tokens=True)
```

`tok.vocab_size` dùng trực tiếp cho `LlamaConfig(vocab_size=...)`
(`app/model/model_configs.py`).

### 7.3 Chạy kiểm chứng

```bash
python -m app.tokenizer.tokenizer_demo
```

Kiểm tra (bằng dữ liệu sinh thật từ `app/gen/generator.py`, không phải mock):

1. Round-trip encode→decode trung thực 100% trên mẫu hợp lệ.
2. `seq_len` mọi sample nằm trong `max_position_embeddings=512`
   (đo thực tế: min/max/avg ≈ 216/235/229 token — khớp ước lượng
   230–240 token/sample ở `train_pipeline_v0.1.md` mục 2.4).
3. Completion rác không crash, token lạ → `<unk>`.
4. Digit-decompose: mỗi digit tách đúng thành 1 token riêng (không bị gộp).

---

## 8. Việc còn để mở

- [ ] Chạy `push_to_hub.py` thật lần đầu (cần `huggingface-cli login`/`HF_TOKEN`,
      không thực hiện được trong môi trường tôi dùng để viết/test code này vì
      không có mạng ra `huggingface.co`) — sau đó cập nhật
      `DEFAULT_TOKENIZER_REPO` trong `app/tokenizer/hub.py` thành repo thật.
- [ ] Đo lại `vocab_size` chính xác sau khi build xong (đã có `describe_vocab()`
      — chạy lại nếu `BIN_MAX`/`RR_MAX` đổi) để cập nhật `LlamaConfig` cho khớp,
      rồi push version mới lên Hub (không sửa đè lên `main` nếu đã có
      checkpoint train theo vocab cũ — cân nhắc tag/`revision` riêng).
- [ ] Khi có dataset thật (không phải synthetic), đo lại phân phối `seq_len`
      trên toàn bộ 10B token corpus để xác nhận không có outlier vượt 512
      (vd sample có candle count lỗi, dù hiếm — Parser vẫn xử lý được nhưng
      cần biết trước để không cắt cụt completion khi training).
- [ ] Quyết định `dataset_mode="pre_tokenized"` cho pretrain/SFT khi bắt đầu
      chạy thật ở scale 10B (mục 5.2) — chưa implement `make_data_module`,
      đó là việc của script train (`train_pretrain.py`/`train_sft.py`), không
      thuộc phạm vi doc này. Script train nên gọi
      `load_tokenizer(allow_local_fallback=False)` để bắt lỗi mạng/config
      sai ngay lập tức thay vì âm thầm fallback.
- [ ] Môi trường train thật cần pin đúng version `tokenizers`/`transformers`
      tương thích với `trl` GRPOTrainer đang dùng — bản dựng thử ở đây dùng
      `transformers` bản mới nhất tại thời điểm viết doc, chưa kiểm tra
      tương thích ngược với version cụ thể sẽ chạy trên Colab/Kaggle.