# DOC: Train Pipeline (Model / Tokenizer / Dataset / Pretrain / SFT / GRPO)

Version: 0.1 — phụ lục thi công cho `docs/spec_trading_llm_v0.2.md`. Doc này chỉ phụ trách phần
**model + tokenizer + train script (Pretrain → SFT → GRPO)**. Phần Lexer/Parser/Semantic Checker/
Forward-test là nhánh riêng, do team khác phụ trách — doc này chỉ định nghĩa **interface** cần họ
khớp theo (mục 6.2).

Framework: HuggingFace `transformers` + `trl` (GRPOTrainer). Kiến trúc: `LlamaConfig` /
`LlamaForCausalLM` chuẩn (không custom modeling), train from scratch (không fine-tune checkpoint có sẵn).

---

## 1. Model Config

### 1.1 Base

```python
config = LlamaConfig(
    vocab_size=...,                       # lấy từ tokenizer đã build (mục 2)
    hidden_size=...,
    intermediate_size=...,
    num_hidden_layers=...,
    num_attention_heads=...,
    num_key_value_heads=...,              # GQA — luôn < num_attention_heads
    max_position_embeddings=512,          # chốt cứng — đủ cho seq_len thực tế ~230-240 token/sample
    pad_token_id=3,                       # khớp SPECIAL_TOKENS: <pad>=3
    bos_token_id=1,                       # <bos>=1
    eos_token_id=2,                       # <eos>=2
    tie_word_embeddings=True,             # mặc định True mọi size — vocab nhỏ so với hidden_size
)
model = LlamaForCausalLM._from_config(config, attn_implementation="sdpa")
```

### 1.2 Presets (`model_configs.py`)

Nhiều size để test/scale dần, chọn qua CLI `--model_size {tiny,small,base,large}`. GQA
(`num_key_value_heads` < `num_attention_heads`) áp dụng mọi preset vì seq_len ngắn không cần MHA đầy đủ.

| Preset | hidden_size | num_hidden_layers | num_attention_heads | num_key_value_heads | intermediate_size | params (~) |
|---|---|---|---|---|---|---|
| tiny  | 128 | 4  | 4  | 2 | 512  | ~2–3M |
| small | 256 | 6  | 8  | 4 | 1024 | ~12–15M |
| base  | 512 | 8  | 8  | 4 | 2048 | ~45–55M |
| large | 768 | 12 | 12 | 4 | 3072 | ~110–130M |

Params thực tế phụ thuộc `vocab_size` cuối cùng sau khi tokenizer build xong (mục 2) — bảng trên là
ước lượng phần transformer block, chưa cộng embedding table (nhỏ, vì `tie_word_embeddings=True`).

`model_configs.py` implement dạng dict-of-dataclass, không hard-code trong script train:

```python
MODEL_PRESETS = {
    "tiny":  ModelArgs(hidden_size=128, num_hidden_layers=4,  num_attention_heads=4,  num_key_value_heads=2, intermediate_size=512),
    "small": ModelArgs(hidden_size=256, num_hidden_layers=6,  num_attention_heads=8,  num_key_value_heads=4, intermediate_size=1024),
    "base":  ModelArgs(hidden_size=512, num_hidden_layers=8,  num_attention_heads=8,  num_key_value_heads=4, intermediate_size=2048),
    "large": ModelArgs(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4, intermediate_size=3072),
}
```

---

## 2. Tokenizer Design

### 2.1 Nguyên tắc phân tầng

Hai mục đích khác nhau → hai cách token hoá khác nhau:

- **Chart (OHLC)**: chỉ là input context, cần nén gọn → **atomic** (1 token/giá trị).
- **Mọi field "giá" dùng trong quyết định** (`current_price`, `zone_support`/`zone_resistance`
  lower & upper, `SL:`): đây là chỗ cố ý dạy model học **map/compose số bin** (spec 2.1: auxiliary
  task ép model đọc đúng giá) và học ước lượng khoảng cách (zone width, SL distance) → **digit-level**,
  zero-pad 4 chữ số, share 1 bảng digit token cho tất cả các field này.

`<trend>` và `action_type` không phải "giá" → giữ enum atomic như cũ.

### 2.2 Bảng vocab

| Nhóm | Dạng | Token cụ thể | Số lượng |
|---|---|---|---|
| Chart O/H/L/C | atomic, mỗi field 1 dải riêng | `<O_0>`...`<O_1023>`, tương tự H/L/C | 4×1024 = 4096 |
| Digit dùng chung (current_price / zone lower / zone upper / SL) | digit-level, zero-pad 4 chữ số | `D_0`...`D_9` | 10 |
| Tag mở/đóng field digit-hoá | atomic tag | `<current_price>` `</current_price>` `<zone_support>` `</zone_support>` `<zone_resistance>` `</zone_resistance>` `SL:` | 7 |
| Separator | atomic | `:` (dùng trong zone `lower:upper`) | 1 |
| `<trend>` | fused atomic | `<trend>UP</trend>` `<trend>DOWN</trend>` `<trend>RANGE</trend>` | 3 |
| `RR:` + giá trị | tag riêng + vocab riêng (không share digit — theo spec mục 3) | `RR:` + `RR_1`...`RR_9` | 1 + 9 |
| Structural tags | atomic | `<chart>` `</chart>` `<think>` `</think>` `<action>` `</action>` `<price_in_zone>` `<good_price_action>` | 8 |
| `action_type` | atomic enum | `BUY` `SELL` `CANCEL_BUY` `CANCEL_SELL` `WAIT_BUY` `WAIT_SELL` `HOLD` | 7 |
| Special | atomic | `<pad>`(id=3) `<bos>`(id=1) `<eos>`(id=2) `<unk>` | 4 |

**Tổng ≈ 4146 token.**

### 2.3 Encode / Decode — ví dụ cụ thể

Input text (đúng format compact, giống output của Data Generator — **không đổi gì ở text layer**):
```
<zone_support>500:510</zone_support>
```
Encode:
1. Regex tách giống hệt Lexer hiện tại: `<zone_support>` `500` `:` `510` `</zone_support>`.
2. Zero-pad số → `0500`, `0510`.
3. Tách digit → `D_0 D_5 D_0 D_0`, `D_0 D_5 D_1 D_0`.
4. Kết quả token sequence: `<zone_support>` `D_0 D_5 D_0 D_0` `:` `D_0 D_5 D_1 D_0` `</zone_support>`.

Decode (ngược lại, dùng khi model generate xong để đưa qua Parser bên kia):
1. Gom các `D_x` liên tiếp trong phạm vi 1 field → ghép thành số nguyên (bỏ zero-pad khi format lại).
2. Ghép lại đúng compact text gốc: `<zone_support>500:510</zone_support>`.
3. **Đảm bảo output cuối luôn đúng compact form** — Parser bên kia không cần biết gì về digit-level, chỉ nhận text thô đúng grammar v0.2.

Tương tự cho `<current_price>512</current_price>` → `<current_price>` `D_0 D_5 D_1 D_2` `</current_price>`,
và `SL:495` → `SL:` `D_0 D_4 D_9 D_5`.

### 2.4 Seq_len ước tính (kiểm tra lại budget 512)

- Chart: 50 nến × 4 field = 200 token (atomic, không đổi).
- `<current_price>`: 2 tag + 4 digit = 6 token.
- 1 zone (nếu có): 2 tag + 4+4 digit + 1 separator = 11 token.
- `<trend>`, `<price_in_zone>`, `<good_price_action>`: ~3 token.
- Action block: `<action>` + action_type + (`SL:` + 4 digit + `RR:` + 1) + `</action>` ≈ 7–9 token.
- **Tổng ≈ 230–240 token/sample** → dư nhiều so với `max_position_embeddings=512`, không cần đổi.

### 2.5 Custom tokenizer implementation note

Không dùng tokenizer train tự động (BPE/WordPiece) vì vocab đã được liệt kê tường minh (closed-form,
không học từ corpus). Implement dạng custom `PreTrainedTokenizer` (hoặc `PreTrainedTokenizerFast` với
`tokenizers.Tokenizer` custom pre-tokenizer + fixed vocab) với `encode()`/`decode()` override theo logic
mục 2.3 — không dùng `AutoTokenizer.train_new_from_iterator`.

---

## 3. Dataset

### 3.1 Nguồn

3 repo Hub riêng biệt (không gộp subset trong 1 repo):
- `<org>/trading-llm-pretrain`
- `<org>/trading-llm-sft`
- `<org>/trading-llm-grpo`

### 3.2 Format lưu trên Hub — RAW TEXT (chưa tokenize)

Giữ nguyên đúng schema spec mục 7.2/7.3, ở dạng text thô theo grammar gốc (compact, không zero-pad,
không tách digit) — encode/decode digit-level chỉ xảy ra ở tokenizer, không ở data layer. Lý do tách
lớp: đổi tokenizer sau này (vd đổi padding 3↔4 digit) không phải regen dataset.

**Pretrain / SFT:**
```python
{
    "prompt": str,       # "<chart>...50 nến...</chart>"
    "completion": str,   # "<think>...</think><action>...</action>"
}
```

**GRPO:**
```python
{
    "prompt": str,
    "future_bins": list,   # [[o,h,l,c], ...] x 50
    "symbol": str,
    "window_id": str,
}
```

### 3.3 Load & tokenize — config toggle `dataset_mode`

Mặc định **on-the-fly**: load raw text trực tiếp từ Hub, tokenize trong `make_data_module` mỗi lần
load dataset (không cache). Đủ nhanh ở scale hiện tại (model nhỏ, dataset không quá lớn).

Thiết kế `make_data_module` với 1 flag để không phải viết lại khi cần đổi hướng:

```python
def make_data_module(tokenizer, data_args, is_pretrain: bool):
    raw = load_dataset(data_args.dataset_name)

    if data_args.dataset_mode == "on_the_fly":
        # tokenize trong collator hoặc .with_transform() — không .map() lưu ra
        ...
    elif data_args.dataset_mode == "pre_tokenized":
        # .map() một lần, cache local hoặc push input_ids lên 1 Hub repo riêng
        # bật khi thấy on-the-fly là bottleneck (không likely ở model size hiện tại)
        ...
```

`data_args.dataset_mode: Literal["on_the_fly", "pre_tokenized"] = "on_the_fly"` — để trong
`DataArguments`, không hard-code, đổi được qua CLI nếu cần sau này.

### 3.4 Data collator

`DataCollatorForCoT` — mask loss trên phần `prompt` (chart_block), chỉ tính loss trên `completion`
(think+action block) cho pretrain/SFT (giống cấu trúc code mẫu bạn đưa). GRPO không cần collator loss
mask kiểu này — `GRPOTrainer` tự xử lý qua reward function.

---

## 4. Pretrain

### 4.1 Luồng init — resumable, không phải "luôn from scratch"

Script phải tự kiểm tra Hub trước khi quyết định init mới hay resume, vì Colab/Kaggle free-tier
session có giới hạn thời gian (bị ngắt giữa chừng là chuyện thường) — không thể giả định mỗi lần
chạy là lần đầu tiên.

```python
checkpoint_repo = f"{org}/trading-llm-{size}-pretrain"

if repo_exists(checkpoint_repo):                     # vd huggingface_hub.repo_exists()
    logger.info(f"Tìm thấy checkpoint sẵn có: {checkpoint_repo} — resume")
    model = LlamaForCausalLM.from_pretrained(checkpoint_repo)
else:
    logger.info("Chưa có checkpoint — init from scratch")
    config = build_llama_config(model_args, tokenizer)
    model = LlamaForCausalLM._from_config(config, attn_implementation="sdpa")
```

Lưu ý: `repo_exists()` chỉ xác định *có repo* — không tự phân biệt "repo rỗng mới tạo" với "đã train
xong 1 phần". Cần đảm bảo model chỉ push lên Hub sau khi đã có ít nhất 1 checkpoint hợp lệ (không
push repo rỗng/placeholder), để lần chạy sau check `repo_exists()` là đáng tin cậy.

`training_args.resume_from_checkpoint` của `Trainer` (dùng cho optimizer state/step giữa 2 lần gọi
`.train()` liên tiếp *trong cùng session*) là khái niệm khác với việc load model weights từ Hub ở
trên (dùng cho việc *bắt đầu lại ở session/máy khác*, không có local checkpoint dir) — script cần hỗ
trợ cả 2 trường hợp: còn local checkpoint dir thì dùng `resume_from_checkpoint`, mất local (session
mới hoàn toàn) thì load lại từ Hub theo đoạn trên rồi train tiếp bình thường (không có optimizer state
cũ, chấp nhận warmup lại).

### 4.2 Push theo chu kỳ, không phải 1 lần lúc kết thúc

Do session Colab/Kaggle có thể bị ngắt bất kỳ lúc nào, **không thể chỉ push lúc training xong** — cần
push theo chu kỳ để không mất tiến độ. Dùng cơ chế có sẵn của `Trainer`, không tự viết callback riêng:

```python
training_args = TrainingArguments(
    ...,
    push_to_hub=True,
    hub_model_id=checkpoint_repo,
    hub_strategy="checkpoint",       # push cả optimizer/scheduler state, không chỉ model weights —
                                      # cho phép resume_from_checkpoint đúng nghĩa ở session sau
    save_strategy="steps",
    save_steps=...,                  # chu kỳ cụ thể tuỳ tốc độ/giới hạn session, set khi chạy thật
    save_total_limit=2,              # tránh phình Hub repo với quá nhiều checkpoint cũ
)
```

- `Trainer` tự động push mỗi lần `save_steps` đạt tới (background, không chặn training loop).
- Cuối cùng vẫn gọi `trainer.save_model()` + `tokenizer.save_pretrained()` + `trainer.push_to_hub()`
  lúc kết thúc để đảm bảo có 1 bản "final" rõ ràng, tách biệt khỏi các checkpoint giữa chừng.

### 4.3 Còn lại

- `Trainer` chuẩn, `data_collator=DataCollatorForCoT`.
- Dataset: `<org>/trading-llm-pretrain`, `dataset_mode="on_the_fly"`.
- Mục tiêu: học well-form + phân bố đều trên không gian tổ hợp hợp lệ (theo spec mục 1) — loss
  cross-entropy chuẩn trên `completion`.
- Checkpoint naming: `<org>/trading-llm-<size>-pretrain` (vd `trading-llm-tiny-pretrain`).

## 5. SFT

### 5.1 Luồng init — cùng pattern resumable như pretrain (mục 4.1), khác nguồn checkpoint gốc

```python
sft_repo = f"{org}/trading-llm-{size}-sft"
pretrain_repo = f"{org}/trading-llm-{size}-pretrain"

if repo_exists(sft_repo):
    logger.info(f"Đã có checkpoint SFT — resume từ {sft_repo}")
    model = LlamaForCausalLM.from_pretrained(sft_repo)
else:
    logger.info(f"Chưa có checkpoint SFT — bắt đầu từ pretrain: {pretrain_repo}")
    model = LlamaForCausalLM.from_pretrained(pretrain_repo)
```

Tức là: lần chạy đầu tiên của SFT bắt nguồn từ checkpoint **pretrain**; các lần chạy sau (session bị
ngắt, chạy tiếp) phải resume từ checkpoint **SFT** của chính nó, không load lại từ pretrain mỗi lần
(nếu không sẽ mất tiến độ SFT đã train).

- Dataset: `<org>/trading-llm-sft` — cùng schema prompt/completion, nhưng random-gen semantic-controlled
  khác (theo spec 7.2 — "semantic đi kèm miễn phí vì generator đảm bảo đúng by construction").
- `Trainer` giống hệt pretrain script (cùng `push_to_hub`/`hub_strategy="checkpoint"`/`save_steps`
  theo chu kỳ — mục 4.2), chỉ đổi nguồn model + `dataset_name`.
- Checkpoint naming: `<org>/trading-llm-<size>-sft`.

## 6. GRPO

### 6.1 Setup

- Luồng init cùng pattern resumable (mục 4.1/5.1) nhưng theo **round**: round 1 load từ checkpoint SFT
  (`<org>/trading-llm-<size>-sft`) nếu chưa có `round1` checkpoint; các round sau load từ checkpoint
  round liền trước (`round{N-1}`) bằng `resume_from_checkpoint`/tham số `--model_name_or_path` truyền
  tay khi tay đổi `weight_table` giữa các round (mục 6.3) — không tự động hoá việc chuyển round.
- Dataset: `<org>/trading-llm-grpo` — chỉ có `prompt`, model tự sinh phần còn lại.
- `remove_unused_columns=False` — **bắt buộc**, nếu không TRL tự xoá hết cột trừ `"prompt"` (mất
  `future_bins`/`symbol`/`window_id` cần cho reward func).
- `num_generations=12` (group size — theo spec, có thể chỉnh theo VRAM thực tế nếu OOM).
- `use_vllm: bool = False` (default) — để trong `GRPOConfig` như 1 field bật/tắt được qua CLI, không
  cố định cứng. Lý do giữ `False` mặc định trên hạ tầng target (Colab T4 / Kaggle T4×2):
  - Seq_len thực tế ngắn (~230–240 token) → PagedAttention của vLLM không tạo khác biệt đáng kể.
  - Model nhỏ (tiny→base, ≤~130M) → generation không phải bottleneck chính so với forward/backward.
  - vLLM cần pool KV-cache riêng chạy song song với model đang train (`colocate` mode) trên cùng GPU
    → rủi ro OOM cao trên T4 16GB single-GPU.
  - Để ngỏ bật lại (`use_vllm=True`) nếu sau này: (a) scale lên preset `large`+, (b) đổi hạ tầng GPU
    VRAM lớn hơn/multi-GPU tách riêng, (c) đo được rollout generation thực sự là bottleneck.

### 6.2 `unified_reward_func` — interface với nhánh Parser/Semantic/Forward-test

**Nguyên tắc quan trọng nhất (spec 8.1)**: KHÔNG tách 3 gate thành 3 `reward_funcs` riêng kèm
`reward_weights` — TRL cộng dồn có trọng số (linear), mâu thuẫn với thiết kế gate cứng. Chỉ viết
**1 hàm `unified_reward_func` duy nhất**.

Giả định module path bên nhánh Parser/Semantic/Forward-test expose (họ khớp interface theo sau, chưa
có nghĩa là đã tồn tại — cần xác nhận khi tích hợp thật):

```python
from app.lang.parser import Parser                                  # đã có sẵn
from app.lang.semantic import SemanticChecker                       # nhánh khác — chưa viết
from app.lang.forward_test import forward_test, counterfactual_outcome  # nhánh khác — chưa viết
```

Expected signature (để nhánh kia biết cần khớp gì):
- `SemanticChecker.check(ast) -> SemanticResult` với `SemanticResult.passed: bool`, `.score: float` (liên tục theo số vi phạm, cùng convention với `ParseResult.well_form_score()`).
- `forward_test(action_ast, future_bins) -> ForwardTestResult` với `.r_multiple: float`.
- `counterfactual_outcome(action_type, zone, future_bins, current_price_bin) -> float` (r_multiple đã đảo dấu).

```python
def unified_reward_func(prompts, completions, future_bins, symbol, **kwargs):
    rewards = []
    for completion, fb, sym in zip(completions, future_bins, symbol):
        parse_result = Parser.from_text(completion).parse()
        if not parse_result.is_well_formed():
            rewards.append(parse_result.well_form_score())
            continue

        sem_result = SemanticChecker.check(parse_result.ast)
        if not sem_result.passed:
            rewards.append(R_WF_FULL + sem_result.score)
            continue

        action_type = parse_result.ast.action.action_type
        if action_type in ("BUY", "SELL"):
            outcome = forward_test(parse_result.ast.action, fb).r_multiple
        elif action_type in ("CANCEL_BUY", "CANCEL_SELL"):
            outcome = counterfactual_outcome(
                action_type, parse_result.ast.think.zone, fb,
                parse_result.ast.think.current_price_bin,
            )
        else:  # WAIT_*, HOLD
            outcome = 0.0

        trend = parse_result.ast.think.trend
        w = weight_table[trend][action_type]
        rewards.append(R_WF_FULL + R_SEM_FULL + outcome * w)
        stats_collector.log(trend, action_type, outcome)
    return rewards
```

`R_WF_FULL`, `R_SEM_FULL`, `weight_table` — trọng số cụ thể **chưa chốt** (spec mục 9, câu hỏi mở #5),
để dạng constant/module-level dict dễ sửa tay giữa các round, không hard-code trong hàm.

### 6.3 Vòng lặp round-based

Gọi `trainer.train()` nhiều lần với `resume_from_checkpoint=True`, giữa các lần đọc `stats_collector`
(tần suất + outcome trung bình theo `trend × action_type`), sửa tay `weight_table`, train tiếp — theo
đúng spec mục 5.3/8.3, không dùng callback tự động.

Checkpoint naming: `<org>/trading-llm-<size>-grpo-round<N>`.

---

## 7. Hạ tầng

| Mục | Quyết định |
|---|---|
| GPU target | Colab T4 (single) / Kaggle T4×2 |
| Precision | `fp16=True` — T4 (Turing) không có bf16 tensor core tốt. **Nếu sau này lên A100/H100, đổi sang `bf16=True`, 1 dòng config.** |
| Multi-GPU | `Trainer`/`GRPOTrainer` mặc định qua `accelerate`/`torchrun` — đủ cho Kaggle T4×2, không cần code riêng. Model ≤130M không cần DeepSpeed/FSDP. |
| vLLM | `use_vllm=False` mặc định, để trong config, bật lại khi cần (mục 6.1). |
| Dataset load | `dataset_mode="on_the_fly"` mặc định, để trong config, chuyển `"pre_tokenized"` khi cần cache (mục 3.3). |

---

## 8. Việc cần làm tiếp theo (phần model/train, không tính nhánh Parser/Semantic)

- [ ] Viết `model_configs.py` (4 preset + `ModelArgs` dataclass).
- [ ] Viết custom tokenizer (`encode`/`decode` theo mục 2.3, vocab cố định mục 2.2).
- [ ] Viết `DataCollatorForCoT` (mask loss trên prompt).
- [ ] Viết `make_data_module` với toggle `dataset_mode`.
- [ ] Viết script pretrain (`train_pretrain.py`).
- [ ] Viết script SFT (`train_sft.py`, load checkpoint pretrain).
- [ ] Viết script GRPO (`train_grpo.py`, `unified_reward_func`, `weight_table`, `stats_collector`) —
      cần nhánh Semantic/Forward-test xong interface mục 6.2 mới chạy thật được, nhưng code khung có
      thể viết trước với mock/stub.