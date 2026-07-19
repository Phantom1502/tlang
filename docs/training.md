# DOC: `app/training/` — Model / Data Feed / Reward / Train Scripts

Version: 1.0 (đi kèm `spec_trading_llm_v0.3.md`). Nhánh lớn nhất, gộp 4 phần con:

```
app/training/
  model/      # CHỈ kiến trúc model — không biết gì về data/reward
  data/       # data FEED cho Trainer lúc train (khác app/data_prepare/ — đây là consume, không phải build)
  reward/     # GRPO reward — post-train, chấm điểm rollout
  train_*.py  # entrypoint script
  common.py   # logic dùng chung 3 script (resume checkpoint, load model + vocab check)
```

---

## 1. `model/configs.py` — kiến trúc Llama, KHÔNG biết gì về vocab_size cụ thể

```python
MODEL_PRESETS = {
    "tiny":  ModelArgs(hidden=128, layers=4,  heads=4,  kv_heads=2, inter=512),
    "small": ModelArgs(hidden=256, layers=6,  heads=8,  kv_heads=4, inter=1024),
    "base":  ModelArgs(hidden=512, layers=8,  heads=8,  kv_heads=4, inter=2048),
    "large": ModelArgs(hidden=768, layers=12, heads=12, kv_heads=4, inter=3072),
}
```

GQA (`kv_heads < heads`) áp dụng MỌI preset — seq_len ngắn (~230 token/sample, budget 512) không cần
MHA đầy đủ. `vocab_size` luôn lấy từ tokenizer thật đã build (`tok.vocab_size`), không hard-code — đổi
vocab không cần sửa file này. `estimate_param_count()` (thuần Python, không cần torch) để đối chiếu
nhanh trước khi khởi tạo model thật.

`build_llama_config()`: `pad_token_id=3, bos_token_id=1, eos_token_id=2` (khớp vocab contract ở
`docs/tokenizer.md`), `tie_word_embeddings=True` mọi size (vocab nhỏ so với hidden_size).

---

## 2. `data/` — feed dataset cho Trainer (khác `app/data_prepare/`)

### 2.1 `arguments.py` — `DataArguments`

```python
dataset_name: str
dataset_mode: Literal["on_the_fly", "pre_tokenized"] = "pre_tokenized"
train_split: str = "train"
eval_split: str = "val"          # PHẢI khớp convention "val" ở docs/data_prepare.md mục 5.1
max_length: int = 512
```

### 2.2 `masking.py` — `compute_labels()`, nguồn sự thật duy nhất cho rule mask

```python
def compute_labels(prompt_ids, full_ids, max_length=None) -> (full_ids, labels):
    # full_ids = [<bos>] + prompt_tokens + completion_tokens + [<eos>]
    # cắt max_length TRƯỚC khi tính n_mask
    n_mask = min(1 + len(prompt_ids), len(full_ids))
    labels = [-100] * n_mask + full_ids[n_mask:]
```

Dùng CHUNG bởi `DataCollatorForCoT` (nhánh `on_the_fly`, ở đây) VÀ
`app/data_prepare/build_tokenized_dataset.py` (nhánh `pre_tokenized`, build offline) — trước đây công
thức này bị viết tay 2 lần độc lập ở 2 nơi (antipattern đã sửa), giờ cả 2 gọi lại hàm này.

CHỈ áp dụng cho nhánh SFT (mask prompt). **Pretrain KHÔNG dùng hàm này** — pretrain là full-sequence
loss (học cả chart), xử lý riêng ở caller (xem mục 2.3).

**Vì sao encode riêng `prompt` rồi so khớp prefix là an toàn** (không cần
`return_offsets_mapping=True`): tokenizer dùng WordLevel (exact-match) + WhitespaceSplit (chỉ tách
theo khoảng trắng) — không có BPE, không merge token qua ranh giới. `tokenize(prompt)` độc lập chắc
chắn trùng khớp tuyệt đối với đoạn prefix tương ứng trong `tokenize(full_text)`.

### 2.3 `data_module.py` — 2 collator + `make_data_module()`

```python
DataCollatorForCoT(tokenizer, is_pretrain: bool, max_length, ...)          # dataset_mode="on_the_fly"
DataCollatorForPreTokenizedCoT(tokenizer, is_pretrain: bool, ...)          # dataset_mode="pre_tokenized"
```

- `is_pretrain=True`: labels = **toàn bộ** `full_ids` (full-sequence loss, không mask gì — pretrain
  học cả chart). `DataCollatorForPreTokenizedCoT` **bỏ qua** cột `labels` đã build sẵn (luôn có mask,
  xem `docs/data_prepare.md` mục 5.2), tự dựng lại `labels=input_ids` ngay tại chỗ.
- `is_pretrain=False` (SFT): dùng `compute_labels()` (on_the_fly) hoặc cột `labels` đã build sẵn
  (pre_tokenized) — cả 2 trường hợp đều mask `<bos>+prompt`.

`make_data_module(tokenizer, data_args, is_pretrain)` → trả `{"train_dataset", "eval_dataset",
"data_collator"}`, dùng thẳng cho `Trainer(**data_module)`. Cả 2 nhánh `dataset_mode` đều load qua
`streaming=True` (dataset lớn, ~10B token ở scale thật).

---

## 3. `reward/` — GRPO reward system

### 3.1 `round_config.py` — `RoundConfig`, cấu hình tường minh theo round

```python
@dataclass
class RoundConfig:
    round_id: str
    weight_table: Dict[str, Dict[str, float]]
    zone_width_min_bins: int
    zone_width_max_bins: int
    sl_min_dist_bins: int
    sl_max_dist_bins: int
    pass_gate2_bonus: float      # K
    zone_quality_bonus: float
    trade_fee_bins: float
```

**Fail-loud, KHÔNG fallback ngầm**: `RoundConfig.load(path)` raise ngay nếu thiếu file hoặc thiếu
BẤT KỲ field nào trong 8 field trên (`_REQUIRED_KEYS`). `__post_init__` validate bất biến
`pass_gate2_bonus > max(weight_table)` — xem chứng minh đầy đủ ở `spec` mục 8.4.

Vòng đời: 1 file JSON = 1 round, **cố định tới hết round** (Colab có thể ngắt/chạy lại nhiều lần
trong CÙNG 1 round — file JSON là nguồn duy nhất, không phụ thuộc trạng thái Python session). Chuyển
round KHÔNG tự động hoá — tay đổi `--round_config`/`--init_from_repo` khi gọi `train_grpo.py`.

### 3.2 `forward_test.py` — forward-test / counterfactual / zone quality probe

```python
forward_test(entry_bin, sl_bin, target_bin, future_candles, direction) -> ForwardTestResult
```
Check `hit_sl`/`hit_tp` MỖI nến tương lai (horizon=50, không dừng sớm). Gap cùng nến → ưu tiên SL.
Không chạm gì → `TIMEOUT, r_multiple=0` (trung tính — không cần rule phạt riêng cho "RR đặt quá lớn":
nếu giá tiến gần target rồi đảo chiều chạm SL, vòng lặp per-candle đã tự bắt được LOSS như bình
thường, không phải TIMEOUT).

```python
counterfactual_outcome(action_type, zone, current_price_bin, future_candles)  # CANCEL_*, buffer 1 bin từ giá hiện tại
probe_zone_quality(zone, future_candles)                                      # MỚI — mép-đối-mép, buffer 1 bin từ mép zone
```

2 hàm buffer đều dùng ý tưởng "lùi 1 bin ra ngoài mép" nhưng **KHÔNG gộp chung 1 hàm** — entry khác
nhau hoàn toàn (`counterfactual_outcome`: entry = giá hiện tại; `probe_zone_quality`: entry = mép
zone gần giá hơn). Hằng số buffer tách riêng (`ZONE_PROBE_SL_BUFFER_BINS` cho probe, literal `±1`
inline cho counterfactual).

`is_sl_valid`/`evaluate_outcome` nhận `sl_min_dist_bins`/`sl_max_dist_bins` qua tham số, default =
module constant (5/10, dùng cho generator/demo); GRPO truyền tường minh từ `RoundConfig`.

### 3.3 `reward_func.py` — `unified_reward_func`, công thức gate 3 đầy đủ

```python
def score_completion(prompt: str, completion: str, future_bins, ...) -> float:
    round_config = get_active_round_config()   # fail-loud nếu chưa set_active_round_config()
    parse_result = Parser.from_text(prompt + " " + completion).parse()   # PHẢI ghép prompt — xem mục 3.3.1
    ...
```

#### 3.3.1 Bug đã sửa: `prompt` bị bỏ quên

`unified_reward_func(prompts, completions, future_bins, **kwargs)` nhận `prompts` nhưng bản đầu
KHÔNG dùng — chỉ `Parser.from_text(completion)`. Vì `GRPOTrainer` (TRL) decode `completions` CHỈ từ
token sinh RA SAU prompt (không bao giờ có `<chart>`), Parser LUÔN báo lỗi thiếu chart → 1 lỗi
structural cố định → `well_form_score() ≈ 0.85` bất kể think/action đúng hay sai — che gần hết
gradient signal thật (log thực tế: `reward/mean` phẳng lì ~0.848, `frac_reward_zero_std` 93-97%).
Đã fix: `unified_reward_func` zip cả `prompts`, `score_completion` ghép `prompt + " " + completion`
trước khi parse.

#### 3.3.2 Công thức gate 3 (outcome) — xem đầy đủ ở `spec` mục 5.2

```python
K = round_config.pass_gate2_bonus
base = R_WF_FULL + R_SEM_FULL + K                                  # sàn cộng khi qua gate 2

zone_bonus = round_config.zone_quality_bonus if probe_zone_quality(zone, future).r_multiple > 0 else 0.0

HOLD          -> base
WAIT_*        -> base + zone_bonus
CANCEL_*      -> base + zone_bonus + min(0, r_multiple_counterfactual * w)      # CHỈ trừ, không cộng
BUY/SELL      -> base + zone_bonus + (r_multiple_thật - fee_in_r) * w           # fee_in_r = trade_fee_bins / risk_bins
```

`fee_in_r` CHỈ áp cho BUY/SELL (vào lệnh thật) — KHÔNG rò rỉ sang CANCEL/WAIT/HOLD. `risk_bins =
|current_price_bin - SL|` — tính theo risk CỦA CHÍNH lệnh đó (không phải hằng số cố định), giống
spread ảnh hưởng R nhiều hơn khi SL đặt sát.

**Công thức này được kỳ vọng còn đổi tiếp** — đánh version cao hơn ở `spec` mỗi khi thực nghiệm round
sau lộ ra lỗ hổng mới, không coi đây là chốt cuối cùng.

### 3.4 `StatsCollector` — persistence, resume nhiều lần/round

```python
stats_collector.save(path)                              # dump toàn bộ record tích luỹ hiện tại
StatsCollector.load(path)                                # load lại (load-rồi-append pattern)
StatsCollector.merge_from_files([path_rank0, path_rank1, ...])   # gộp report multi-GPU
```

Mỗi rank tự dump 1 file riêng (`{output_dir}/{round_id}_stats_rank{rank}.json`) — mỗi lần script
khởi động lại (Colab bị ngắt, chạy lại NHIỀU LẦN trong CÙNG 1 round) đều `load()` lại record cũ TRƯỚC
khi log tiếp, để file trên đĩa luôn phản ánh TOÀN BỘ round tính đến hiện tại, không chỉ session hiện
tại. `train_grpo.py` gọi `save()` theo đúng chu kỳ `save_steps` (qua `TrainerCallback`) — cùng triết
lý "push theo chu kỳ, không phải 1 lần lúc kết thúc" đã áp dụng cho checkpoint model.

---

## 4. `train_pretrain.py` / `train_sft.py` / `train_grpo.py` — resumable pattern

### 4.1 Luồng init chung (`common.py`)

```python
resolve_resume_checkpoint(output_dir, checkpoint_repo) -> Optional[str]
```
Ưu tiên: (1) local checkpoint dir trong `output_dir` (session chưa ngắt, chạy `.train()` 2 lần liên
tiếp) → resume tại chỗ; (2) session mới hoàn toàn → tải subfolder `last-checkpoint/` từ Hub (nếu có,
`hub_strategy="checkpoint"` push CẢ optimizer/scheduler state, không chỉ weights); (3) không có gì →
`None` (lần đầu tiên train repo này).

```python
load_model_with_vocab_check(source, vocab_size) -> model
```
`LlamaForCausalLM.from_pretrained(source)` + raise nếu `vocab_size` lệch tokenizer hiện tại — không
âm thầm train tiếp trên embedding table sai kích thước (vi phạm vocab contract).

### 4.2 Push/save theo chu kỳ (không phải 1 lần lúc kết thúc)

Colab/Kaggle session có thể ngắt bất kỳ lúc nào — `hub_strategy="checkpoint"`, `save_strategy=
"steps"` để `Trainer` tự push định kỳ (background). Cuối cùng vẫn gọi `trainer.save_model()` +
`push_to_hub()` để có 1 bản "final" tách biệt khỏi checkpoint giữa chừng.

### 4.3 Khác biệt duy nhất giữa 3 script — nguồn init khi CHƯA có checkpoint

| Script | Chưa có checkpoint → init từ |
|---|---|
| `train_pretrain.py` | from scratch (`build_llama_config`) |
| `train_sft.py` | `--pretrain_repo` (checkpoint pretrain đã train xong) |
| `train_grpo.py` | `--init_from_repo` (SFT cho round 1, checkpoint round liền trước cho round N>1 — **tay truyền, không tự động hoá việc chuyển round**) |

### 4.4 `train_grpo.py` — riêng biệt với 2 script trên

```bash
python -m app.training.train_grpo \
    --model_size tiny --round_id round1 \
    --repo_id <org>/tlang-grpo-round1 --init_from_repo <org>/tlang-sft \
    --round_config ./rounds/round1.json \
    --dataset_name <org>/tlang-grpo --output_dir ./output/grpo-round1 \
    --num_generations 12 --save_steps 50 --max_steps 500

python -m app.training.train_grpo --round_id round1 --output_dir ./output/grpo-round1 --report_only
```

**Khác biệt so với pretrain/SFT**:
- Dùng `GRPOConfig`/`GRPOTrainer` (TRL), không phải `TrainingArguments`/`Trainer`.
- `remove_unused_columns=False` **BẮT BUỘC** — nếu không TRL tự xoá hết cột trừ `"prompt"`, mất
  `future_bins`/`symbol`/`window_id` cần cho `reward_func`.
- `reward_funcs=unified_reward_func` — TRL >= một số bản dùng `processing_class=` thay vì
  `tokenizer=` khi khởi tạo `GRPOTrainer` (đổi tên API — pin version `trl` cụ thể trong
  `requirements.txt`, kiểm tra lại signature thật mỗi khi nâng cấp `trl`).
- Load `RoundConfig` + `set_active_round_config()` **1 lần lúc khởi động**, TRƯỚC `trainer.train()`.
- **KV cache**: model load từ SFT/round trước có thể mang `use_cache=False` (nếu checkpoint nguồn
  từng train với `gradient_checkpointing` bật). GRPO cần `generate()` rất nhiều lần mỗi step
  (rollout) — PHẢI set lại `model.config.use_cache = True` NGAY SAU KHI LOAD, bất kể checkpoint nguồn
  lưu gì. `GRPOConfig.gradient_checkpointing=True` (default TRL) vẫn hoạt động bình thường cho pha
  forward/backward — chỉ cần đảm bảo giá trị BAN ĐẦU không bị kẹt ở `False` từ checkpoint cũ.
- **Tokenizer quirk** (`add_eos_token=False`) — xem `docs/tokenizer.md` mục 6, chi tiết đầy đủ nhất
  nằm ở đó vì đây là vấn đề của tokenizer, không phải của training loop.
- `--report_only`: KHÔNG train, chỉ gộp mọi `{round_id}_stats_rank*.json` trong `--output_dir` +
  in `summary()` — xem bất cứ lúc nào, không cần đợi round kết thúc, không cần GPU.

---

## 5. `scripts/*.sh` — ví dụ chạy đầy đủ

```bash
scripts/tiny_pretrain.sh    # model_size=tiny, dataset_mode=pre_tokenized
scripts/base_pretrain.sh    # model_size=base
scripts/base_sft.sh         # --pretrain_repo trỏ checkpoint pretrain
scripts/base_grpo.sh        # --round_config ./rounds/round1.json, round 1 init từ SFT
```

Mọi giá trị (`--repo_id`, `--dataset_name`, batch size, LR...) truyền tay tường minh trong từng `.sh`
— không có giá trị mặc định ẩn nào khác ngoài những gì `argparse` khai báo trong từng script.