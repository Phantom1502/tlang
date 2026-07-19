# DOC: `app/data_prepare/` — Raw OHLC → Dataset (kể cả tokenize/map)

Version: 1.0 (đi kèm `spec_trading_llm_v0.3.md` mục 7). Phạm vi: mọi bước biến đổi dữ liệu THÔ
(giá OHLC) thành dataset sẵn sàng train — bao gồm cả bước tokenize/mask offline (`build_tokenized_
dataset.py`), vì đây vẫn là "chuẩn bị dữ liệu", không phải logic tiêu thụ lúc train (đó là việc của
`app/training/data/`).

Phụ thuộc: `app/lang/` (Parser/SemanticChecker để tự-verify), `app/tokenizer/` (tokenize), và
`app/training/reward/forward_test.py` (generator dùng `evaluate_outcome`/`SL_MIN/MAX_DIST_BINS` để
tự-verify outcome hợp lệ trước khi trả sample — xem mục 2.3).

---

## 1. `preprocess.py` — ATR + scale factor

`Preprocess.preprocess(csv_path, output_path, period=100)`: tính `ATR_100` (EMA của True Range),
quét cửa sổ trượt toàn bộ lịch sử để tìm `FINAL_SCALE_FACTOR` (bách phân vị 99.9% của tỷ lệ "cửa sổ
100 nến dạt xa gấp mấy lần ATR tại thời điểm đó so với `Open` neo") — hằng số này dùng cho
`ChartCodec` (mục 2) để quy đổi giá thật ↔ bin [0,1023].

Output: CSV đã preprocess + append 1 dòng vào `scale_factor.txt` (`"{filename}: {SCALE:.2f}"`).
`dataset_builder.py:load_scale_factors()` đọc lại file này, tránh gõ tay/copy nhầm số.

---

## 2. `chartcodec.py` — quantize/dequantize + encode window

```python
ChartCodec(scale: float, n_bins: int = 1024).quantize_price(price, anchor_open, anchor_atr) -> int
```
```
norm = clip((price - anchor_open) / (scale * anchor_atr), -1, 1)
bin  = round((norm + 1) / 2 * (n_bins - 1))
```

`encode_window`/`encode_df`: windows hoá theo `window_size`/`stride`, xuất text thô KHÔNG có dấu
ngoặc quanh field (`"O_434 H_543 L_543 C_543 ..."`) — khác format atomic có ngoặc mà grammar cần
(`"<O_434>"`). `dataset_builder.py` (mục 3) là cầu nối giữa 2 format này.

**Quan trọng — window 100 nến, không phải 50**: `encode_df(window_size=100, stride=50)` — 50 nến đầu
= input (`chart_block`), 50 nến sau = `future_bins` (forward-test/counterfactual). Cùng 1
`anchor_open`/`anchor_atr` cho toàn bộ 100 nến, đảm bảo input và future nhất quán trong cùng hệ bin.

---

## 3. `generator.py` — sinh completion cho pretrain/SFT ("by construction")

### 3.1 Nguyên tắc: hardcode zone/SL range, KHÔNG cần flexible

Khác với `SemanticChecker`/`forward_test.py` (nhận `zone_width_min/max_bins`, `sl_min/max_dist_bins`
qua tham số — GRPO cần chỉnh theo round), **generator giữ hardcode** 2 cặp ngưỡng này:

```python
ZONE_WIDTH_MIN_BINS = SemanticChecker.ZONE_WIDTH_MIN_BINS   # 5 — import lại, không tự định nghĩa
ZONE_WIDTH_MAX_BINS = SemanticChecker.ZONE_WIDTH_MAX_BINS   # 20
```

Lý do: pretrain/SFT chỉ cần dữ liệu ĐÚNG FORMAT (well-form + semantic pass "by construction"), chưa
có outcome thật để biết nên nới/siết ngưỡng thế nào — đó là việc CHỈ GRPO mới làm (qua `RoundConfig`,
xem `docs/training.md`).

### 3.2 `LEAF_RECIPES` — sample uniform trên toàn bộ leaf-path hợp lệ

21 leaf-path liệt kê tường minh: `(trend, zone_side, zone_case, action_type)`. `zone_case` ∈
`{CONTAINS, TOUCH, NOTOUCH}` — quyết định `price_in_zone` mong đợi theo đúng hình học (mục D spec).
Sample UNIFORM trên danh sách này (không sample từng field độc lập rồi lọc invalid — méo phân phối).

Thứ tự sinh **đúng** (spec §7.2): random zone trước → tính `price_in_zone` THẬT từ chart → random
action phù hợp. `current_price` LUÔN = Close nến cuối (không random).

### 3.3 Tự-verify trước khi trả sample (`generate_one`)

Sau khi dựng số xong, generator tự chạy lại qua chính `Parser`/`SemanticChecker`/`evaluate_outcome`
(nguyên tắc "double-check end-to-end, không chỉ tin công thức dựng số") — trả `None` nếu bất kỳ bước
nào fail, caller thử lại với leaf/random state khác. Đảm bảo **100% sample sinh ra đều well-formed +
semantic pass**, không cần rule riêng nào ở phía SFT để lọc lại.

---

## 4. `dataset_builder.py` — cầu nối ChartCodec ↔ Grammar + augment

- `parse_window_text`/`render_chart_block`: convert 2 chiều giữa format thô của `ChartCodec`
  (không ngoặc) và format atomic grammar cần (có ngoặc `<O_x>`).
- `augment_shift`: dịch chuyển ĐỀU (uniform bin-shift) toàn bộ candle trong 1 window theo cùng 1
  delta ngẫu nhiên — giữ hình dạng tương đối (trend/zone/khoảng cách giữa nến), đổi toạ độ tuyệt đối,
  giúp model học đúng quan hệ tương đối thay vì ghi nhớ 1 vùng bin cụ thể. Augment áp dụng trên CẢ
  100 nến cùng lúc (input+future cùng 1 delta) để giữ nhất quán nội bộ 1 window.
- `build_grpo_rows`/`build_grpo_parquet`: schema `{"prompt", "future_bins", "symbol", "window_id"}`.
- `build_pretrain_sft_rows`/`build_pretrain_sft_parquet`: schema `{"prompt", "completion"}` — gọi
  `generator.generate_dataset()` trên mỗi chart (kể cả chart đã augment).

---

## 5. Kiến trúc lưu trữ Hub — 2 subset "raw"/"ids" trong CÙNG 1 repo

**Khác với thiết kế ban đầu** (spec cũ mô tả "3 repo Hub tách biệt") — thực tế vận hành:

```
<org>/tlang-pretrain          (1 repo, dùng CHUNG cho cả pretrain lẫn SFT)
  ├── config "raw"  — text thô {"prompt","completion"}
  └── config "ids"  — đã tokenize+mask {"input_ids","labels"}

<org>/tlang-grpo               (1 repo riêng — schema khác, không có "ids" vì GRPOTrainer tự tokenize prompt)
  └── config "raw"  — {"prompt","future_bins","symbol","window_id"}
```

### 5.1 `push_dataset_to_hub.py` — upload thẳng file, không tải gì về

Upload trực tiếp file parquet lên ROOT repo (không thư mục con `data/`), prefix tên file theo split:
```bash
python -m app.data_prepare.push_dataset_to_hub \
    --parquet_paths AUDUSD_15Min_pretrain_sft.parquet --repo_id <org>/tlang-pretrain
python -m app.data_prepare.push_dataset_to_hub \
    --holdout_parquet_paths XAUUSD_15Min_VAL.parquet --repo_id <org>/tlang-pretrain   # -> split "val"
```

**Naming convention split: `"train"` / `"val"`** (không phải `"validation"`) — phải khớp
`DataArguments.eval_split = "val"` (`app/training/data/arguments.py`) và YAML config đã khai báo
tường minh trên Hub repo. Đã từng lệch (`push_dataset_to_hub.py` hardcode `"validation"` trong khi
code tiêu thụ dùng `"val"`) — đã fix, giữ đúng `"val"` xuyên suốt.

Không có bước dedup/check-overlap tự động (v3 — bỏ hẳn cơ chế tải-về-gộp-lại của v2 vì OOM ở scale
vài GB). Mỗi file build ĐỘC LẬP theo symbol/khối thời gian, không tái tạo, nên không cần merge.

### 5.2 `build_tokenized_dataset.py` — build subset "ids" 1 lần, offline

```bash
python -m app.data_prepare.build_tokenized_dataset \
    --repo_id <org>/tlang-pretrain --ids_repo_id <org>/tlang-pretrain-ids \
    --kind pretrain_sft --split train --dry_run
```

**Pretrain và SFT dùng CHUNG 1 dataset "ids"** — build 1 lần, KHÔNG build riêng cho từng giai đoạn.
Quyết định "mask hay không mask" (pretrain: full-sequence loss; SFT: mask `<bos>+prompt`) nằm ở PHÍA
TIÊU THỤ (`DataCollatorForPreTokenizedCoT(is_pretrain=...)`, xem `docs/training.md`), KHÔNG nằm ở
bước build ids này — build ids không được tự ý giả định trước dataset sẽ dùng cho giai đoạn nào.

**Luôn tính mask** (cột `labels` luôn có mask sẵn) khi build "ids", bất kể sẽ dùng cho pretrain hay
SFT — phía pretrain đơn giản BỎ QUA cột `labels` đã build sẵn này, tự dựng lại `labels = input_ids`
(full-sequence) ngay lúc train. Từng cân nhắc thêm cờ `--is_pretrain` để skip tính mask cho tiết kiệm
compute lúc build — **ĐÃ BỎ Ý ĐỊNH NÀY**: dataset "ids" dùng chung vật lý cho cả 2 giai đoạn, nếu
build ids tự quyết định bỏ mask theo cờ, ai lỡ dùng lại dataset đó cho SFT sau này sẽ nhận cột
`labels` sai (loss tính cả trên chart, không mask prompt) — bug âm thầm, khó phát hiện.

Batched `.map()` (không phải 1 example/lần) — encode cả batch trong 1 lệnh, khác biệt hiệu năng lớn
nhất ở scale ~10B token. `TOKENIZERS_PARALLELISM=false` set TRƯỚC mọi import liên quan tokenizer —
tránh xung đột đa luồng Rust tokenizer với `num_proc` multiprocessing của `datasets.map()`.

**Verify bắt buộc trước khi push** (`--skip_verify` mặc định TẮT, fail-closed): decode lại các token
KHÔNG bị mask, so khớp CHÍNH XÁC với `completion` gốc — độc lập hoàn toàn với công thức mask (không
tái dùng cùng logic để tránh 2 nơi cùng sai giống nhau che lấp lẫn nhau).

Công thức mask dùng CHUNG với `DataCollatorForCoT` (`app/training/data/masking.py:compute_labels`) —
1 nguồn sự thật duy nhất, không viết tay 2 lần độc lập (bài học từ antipattern từng có).

### 5.3 Naming convention repo (tổng hợp)

| Pattern | Ý nghĩa |
|---|---|
| `{name}` | dataset raw (vd `tlang-pretrain`, `tlang-grpo`) |
| `{name}-ids` | dataset đã tokenize (vd `tlang-pretrain-ids`) — repo TÁCH RIÊNG khỏi raw, tránh mỗi lần push "ids" tạo commit mới làm cache "raw" bị stale |
| `{model}-pretrain` / `-sft` / `-grpo-round{N}` | model checkpoint repo, ĐỒNG THỜI là nơi chứa tokenizer (add sẵn trước khi train) |

Không có tự động hoá đặt tên — mọi giá trị đều truyền tay qua CLI (`--repo_id`, `--ids_repo_id`...),
xem ví dụ đầy đủ trong `scripts/*.sh`.

---

## 6. Demo / kiểm chứng nhanh

```bash
python -m demos.generator_demo         # 100% sample sinh ra well-form+semantic pass, phân phối leaf-path
python -m demos.dataset_builder_demo   # GRPO/pretrain-SFT parquet, round-trip qua đĩa
```