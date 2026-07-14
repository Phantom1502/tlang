# DOC: Data Module (Collator / make_data_module)

Version: 0.1 — phụ lục thi công cho `docs/train_pipeline_v0.1.md` mục 3 (Dataset) và mục 8 (việc
cần làm tiếp theo). Doc này chỉ phụ trách phần **load dataset + tokenize + mask loss cho
Pretrain/SFT** (`app/data/`) — KHÔNG liên quan tới dataset GRPO (đã có `app/reward/reward_func.py`
xử lý riêng qua `unified_reward_func`, không đi qua module này).

Code: `app/data/collator.py`, `app/data/data_module.py`.

Trạng thái: **đã viết xong, CHƯA chạy test thật** (môi trường viết code không có `torch`/
`transformers`/`datasets` cài sẵn). Cần ai đó chạy thử trên môi trường có đủ dependency trước khi
cắm vào `train_pretrain.py`/`train_sft.py` thật — xem mục 6.

---

## 1. Phạm vi

| Việc | Thuộc module này? |
|---|---|
| Load dataset Pretrain/SFT từ Hub (schema `{"prompt", "completion"}`) | Có |
| Tokenize + mask loss (chỉ tính loss trên `completion`) | Có |
| Toggle `dataset_mode`: `"on_the_fly"` vs `"pre_tokenized"` | Có |
| Dataset GRPO (`{"prompt", "future_bins", "symbol", "window_id"}`) | Không — xem `app/reward/` |
| Script train thật (`train_pretrain.py`, `train_sft.py`) | Không — chỉ cung cấp `make_data_module()` để script đó gọi |
| Build/push tokenizer lên Hub | Không — xem `docs/tokenizer_v0.1.md`, dùng `app.tokenizer.hub.load_tokenizer()` |

---

## 2. Nguyên tắc mask loss

Input mỗi sample: `{"prompt": str, "completion": str}` — `prompt` là `chart_block`, `completion` là
`think_block + action_block` (đúng schema mục 7.2 `spec_trading_llm_v0.2.md`).

Nối lại đúng convention generator đang dùng: `full_text = prompt + " " + completion`.

Xác định ranh giới prompt/completion **bằng cách encode `prompt` riêng** (`add_special_tokens=False`)
để lấy số token `P`, rồi so với `full_ids = tokenizer(full_text, add_special_tokens=True)`:

```
full_ids = [<bos>] + prompt_tokens (P token) + completion_tokens + [<eos>]
n_mask   = 1 + P                      # <bos> + toàn bộ prompt
labels   = [-100] * n_mask + full_ids[n_mask:]   # completion + <eos> giữ nguyên
```

**Vì sao encode riêng rồi so khớp prefix là an toàn**: tokenizer dùng `WordLevel` (exact-match) +
`WhitespaceSplit` (chỉ tách theo khoảng trắng, không theo punctuation) — không có BPE, không có
merge token qua ranh giới. Nên `tokenize(prompt)` đứng độc lập chắc chắn trùng khớp tuyệt đối với
đoạn prefix tương ứng trong `tokenize(full_text)`. Không dùng `return_offsets_mapping` (dù tokenizer
là fast/Rust backend, có hỗ trợ) vì cách encode-riêng đơn giản hơn và không cần xử lý case đặc biệt
offset `(0,0)` của special token — đúng tinh thần "không có chỗ để hành vi ẩn xảy ra" đã chọn cho
tokenizer ở `docs/tokenizer_v0.1.md`.

**Đánh đổi**: tokenize 2 lần/sample (`prompt` riêng + `full_text`) thay vì 1 lần — chấp nhận được ở
scale hiện tại (tokenizer Rust backend, rẻ). Nếu sau này đo được đây là bottleneck thật ở dataset
~10B token, cân nhắc chuyển sang `return_offsets_mapping=True` để encode 1 lần.

---

## 3. `DataCollatorForCoT` — `dataset_mode="on_the_fly"`

`app/data/collator.py`

- Nhận batch feature dạng **raw text** `{"prompt", "completion"}`.
- Tự tokenize + mask (mục 2) + pad ngay trong `__call__` — không `.map()` lưu ra trước.
- Pad thủ công: `input_ids` pad bằng `tokenizer.pad_token_id` (=3), `labels` pad bằng `-100`,
  `attention_mask` pad bằng `0`.
- Hỗ trợ `max_length` (mặc định 512, khớp `MAX_POSITION_EMBEDDINGS` ở `app/model/model_configs.py`)
  và `pad_to_multiple_of` (tối ưu tensor core, tuỳ chọn).

```python
from transformers import Trainer
from app.tokenizer.hub import load_tokenizer
from app.data.collator import DataCollatorForCoT

tok = load_tokenizer()
trainer = Trainer(..., data_collator=DataCollatorForCoT(tokenizer=tok))
```

## 4. `DataCollatorForPreTokenizedCoT` — `dataset_mode="pre_tokenized"`

Cùng file `app/data/collator.py`.

- Nhận batch feature đã có sẵn `input_ids`/`labels` (đã tính 1 lần qua `.map()` — xem mục 5).
- **Chỉ pad**, không tokenize/mask gì thêm. Nếu mask sai, lỗi nằm ở bước `.map()`
  (`_tokenize_and_mask_example` trong `app/data/data_module.py`), không phải ở collator này —
  tách trách nhiệm rõ ràng để dễ debug.
- Dùng chung hàm `_pad_encoded()` nội bộ với `DataCollatorForCoT` để 2 nhánh không lệch cách pad.

---

## 5. `make_data_module()` — `app/data/data_module.py`

Entry point script train sẽ gọi:

```python
from app.data.data_module import DataArguments, make_data_module
from app.tokenizer.hub import load_tokenizer
from transformers import Trainer, TrainingArguments

tok = load_tokenizer()
data_args = DataArguments(
    dataset_name="<org>/trading-llm-pretrain",
    dataset_mode="on_the_fly",   # hoặc "pre_tokenized"
)
data_module = make_data_module(tok, data_args, is_pretrain=True)

trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    **data_module,   # train_dataset, eval_dataset, data_collator
)
```

`DataArguments` (dataclass, CLI-configurable — không hard-code trong script train):

| Field | Default | Ghi chú |
|---|---|---|
| `dataset_name` | bắt buộc | vd `<org>/trading-llm-pretrain` hoặc `...-sft` |
| `dataset_mode` | `"on_the_fly"` | `Literal["on_the_fly", "pre_tokenized"]` |
| `eval_dataset_name` | `None` | `None` = không có eval split riêng |
| `num_proc` | `4` | chỉ dùng khi `dataset_mode="pre_tokenized"` (`.map()` đa luồng) |
| `max_length` | `512` | khớp `MAX_POSITION_EMBEDDINGS` |

Nhánh `"on_the_fly"`: trả thẳng dataset raw (chưa tokenize) + `DataCollatorForCoT`.

Nhánh `"pre_tokenized"`: `.map()` một lần bằng `_tokenize_and_mask_example()` (dùng **cùng logic
mask** với `DataCollatorForCoT.__call__` — 1 nguồn sự thật duy nhất cho rule mask, tránh 2 nơi lệch
nhau), cache Arrow local, trả dataset đã tokenize + `DataCollatorForPreTokenizedCoT`.

`load_dataset()` tự nhận diện `DatasetDict` (có split `"train"`) hay `Dataset` thẳng
(`hasattr(raw, "keys")`) — nếu chắc chắn dataset trên Hub luôn có split `"train"`, có thể bỏ check
này cho gọn.

---

## 6. Việc còn để mở / cần làm tiếp

- [ ] **Chưa chạy test thật** — môi trường viết code không có `torch`/`transformers`/`datasets`. Cần
      chạy thử: build vài sample bằng `app/gen/generator.py:generate_dataset()`, bọc thành
      `datasets.Dataset.from_list(...)`, chạy qua cả 2 nhánh `dataset_mode`, in `labels` ra kiểm tra
      vị trí `-100` kết thúc đúng ngay trước token đầu tiên của `completion` (`<think>`).
- [ ] Chưa implement nhánh "push input_ids lên 1 Hub repo riêng" nhắc tới trong
      `train_pipeline_v0.1.md` mục 3.3 (comment `# tokenize trong collator hoặc .with_transform()...`
      / `# .map() một lần, cache local hoặc push input_ids lên 1 Hub repo riêng`) — hiện
      `"pre_tokenized"` chỉ cache Arrow local qua `.map()` mặc định của `datasets`, chưa
      `push_to_hub()`. Đây là optimization "bật khi cần" theo spec, chưa chốt có cần không.
    - [ ] Theo `docs/tokenizer_v0.1.md` mục 5.2: cân nhắc bật `"pre_tokenized"` **sớm hơn dự kiến**
      cho Pretrain/SFT ở scale ~10B token, vì 2 dataset này không đổi qua các round (khác GRPO).
- [ ] Script `train_pretrain.py`/`train_sft.py` (gọi `make_data_module()`) vẫn chưa viết — thuộc
      việc tiếp theo trong `train_pipeline_v0.1.md` mục 8, không thuộc phạm vi doc này.
- [ ] Chưa benchmark chi phí tokenize-2-lần/sample ở nhánh `on_the_fly` trên scale dữ liệu thật —
      nếu là bottleneck, cân nhắc `return_offsets_mapping=True` (xem đánh đổi ở mục 2).