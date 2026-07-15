"""
app/data/data_module.py — make_data_module (train_pipeline_v0.1.md mục 3.3)

Load dataset Pretrain/SFT từ Hub (schema mục 7.2 spec: {"prompt","completion"})
và chuẩn bị train_dataset/eval_dataset/data_collator theo toggle
`dataset_mode`:

- "on_the_fly" (mặc định): giữ nguyên raw text, tokenize+mask ngay trong
  DataCollatorForCoT mỗi batch — không .map() lưu ra (mục 3.3 spec).
- "pre_tokenized": .map() một lần ra input_ids/labels, cache Arrow local
  — bật khi on-the-fly là bottleneck thật. Theo tokenizer_v0.1.md mục 5.2,
  nên cân nhắc bật SỚM cho pretrain/SFT ở scale ~10B token vì 2 dataset
  này KHÔNG đổi qua các round (khác GRPO, phải encode runtime vì
  completion do model tự sinh lúc train).

KHÔNG dùng cho GRPO — dataset GRPO chỉ có "prompt" (mục 7.3 spec), không
có "completion" để mask loss kiểu này; GRPOTrainer tự lo sinh/reward,
không đi qua make_data_module/DataCollatorForCoT.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from transformers import PreTrainedTokenizerBase

from app.data.collator import (
    LABEL_PAD_ID,
    DataCollatorForCoT,
    DataCollatorForPreTokenizedCoT,
)


@dataclass
class DataArguments:
    """CLI-configurable — KHÔNG hard-code dataset_mode trong script train."""

    dataset_name: str                                    # vd "<org>/trading-llm-pretrain" | "...-sft"
    dataset_mode: Literal["on_the_fly", "pre_tokenized"] = "on_the_fly"
    eval_dataset_name: Optional[str] = None               # None = không có eval split riêng
    num_proc: int = 4                                     # chỉ dùng khi dataset_mode="pre_tokenized" (.map())
    max_length: int = 512                                 # khớp MAX_POSITION_EMBEDDINGS (app/model/model_configs.py)


def _tokenize_and_mask_example(
    example: Dict[str, Any], tokenizer: PreTrainedTokenizerBase, max_length: int
) -> Dict[str, Any]:
    """Dùng cho .map() ở dataset_mode="pre_tokenized" — CÙNG rule mask
    (<bos>+prompt -> -100) với DataCollatorForCoT.__call__, tách riêng để
    2 nơi không lệch nhau (1 nguồn sự thật duy nhất cho rule mask)."""
    prompt, completion = example["prompt"], example["completion"]

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + " " + completion, add_special_tokens=True)["input_ids"]

    n_mask = min(1 + len(prompt_ids), len(full_ids))
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        n_mask = min(n_mask, len(full_ids))

    labels = [LABEL_PAD_ID] * n_mask + full_ids[n_mask:]
    return {"input_ids": full_ids, "labels": labels}


def make_data_module(
    tokenizer: PreTrainedTokenizerBase,
    data_args: DataArguments,
    is_pretrain: bool,
) -> Dict[str, Any]:
    """
    Trả về dict truyền thẳng vào `Trainer(**make_data_module(...))`:
        {"train_dataset": ..., "eval_dataset": ... | None, "data_collator": ...}

    `is_pretrain` hiện chỉ dùng để log rõ đang chuẩn bị data cho pretrain
    hay SFT — 2 nhánh dùng chung schema/logic (mục 7.2 spec: "SFT dùng
    cùng schema prompt/completion, chỉ khác nguồn random-gen"). Để sẵn
    tham số này phòng khi sau cần rẽ nhánh xử lý khác nhau.
    """
    from datasets import load_dataset  # import trễ — tránh ép cài `datasets` cho script không cần data

    stage = "pretrain" if is_pretrain else "sft"
    raw = load_dataset(data_args.dataset_name)
    train_raw = raw["train"] if hasattr(raw, "keys") else raw

    eval_raw = None
    if data_args.eval_dataset_name is not None:
        eval_ds = load_dataset(data_args.eval_dataset_name)
        eval_raw = eval_ds["train"] if hasattr(eval_ds, "keys") else eval_ds

    if data_args.dataset_mode == "on_the_fly":
        print(f"[make_data_module:{stage}] dataset_mode=on_the_fly — tokenize trong collator mỗi batch")
        return {
            "train_dataset": train_raw,
            "eval_dataset": eval_raw,
            "data_collator": DataCollatorForCoT(tokenizer=tokenizer, max_length=data_args.max_length),
        }

    if data_args.dataset_mode == "pre_tokenized":
        print(f"[make_data_module:{stage}] dataset_mode=pre_tokenized — .map() 1 lần, cache Arrow local")

        def _map_fn(example):
            return _tokenize_and_mask_example(example, tokenizer, data_args.max_length)

        train_tok = train_raw.map(
            _map_fn, remove_columns=train_raw.column_names, num_proc=data_args.num_proc,
            desc=f"Tokenize+mask ({stage}, pre_tokenized)",
        )
        eval_tok = None
        if eval_raw is not None:
            eval_tok = eval_raw.map(
                _map_fn, remove_columns=eval_raw.column_names, num_proc=data_args.num_proc,
                desc=f"Tokenize+mask ({stage} eval, pre_tokenized)",
            )

        return {
            "train_dataset": train_tok,
            "eval_dataset": eval_tok,
            "data_collator": DataCollatorForPreTokenizedCoT(tokenizer=tokenizer),
        }

    raise ValueError(
        f"dataset_mode không hợp lệ: {data_args.dataset_mode!r} (chỉ chấp nhận on_the_fly|pre_tokenized)"
    )