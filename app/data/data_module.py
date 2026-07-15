from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from app.data.collator import (
    LABEL_PAD_ID,
    DataCollatorForCoT,
    DataCollatorForPreTokenizedCoT,
)

logger = logging.getLogger("app.data.data_module")


@dataclass
class DataArguments:
    dataset_name: str
    # "auto" (mặc định — MỚI): thử load config "default" (ids/, đã tokenize
    #   sẵn input_ids/labels) trước; nếu repo chưa có config này (vd repo cũ,
    #   hoặc lỗi mạng) -> tự fallback config "raw" (text thô, tokenize
    #   on-the-fly). Đây là fallback THẬT (code tự thử/catch), không phải
    #   README tự làm được — README chỉ khai báo config nào default cho
    #   `load_dataset(repo)` KHÔNG chỉ định tên config, không có ý nghĩa
    #   "nếu thiếu thì tự chuyển sang config khác".
    # "on_the_fly" / "pre_tokenized": giữ lại để CHỈ ĐỊNH TAY (không auto,
    #   không đọc config "default"/"raw" — tương thích ngược với script cũ).
    dataset_mode: Literal["auto", "on_the_fly", "pre_tokenized"] = "auto"
    eval_dataset_name: Optional[str] = None
    eval_split: str = "validation"   # tên split eval TRONG CÙNG repo (ids/val.parquet, raw/val.parquet)
    num_proc: int = 4
    max_length: int = 512


def _tokenize_and_mask_example(
    example: Dict[str, Any], tokenizer: PreTrainedTokenizerBase, max_length: int
) -> Dict[str, Any]:
    prompt, completion = example["prompt"], example["completion"]

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + " " + completion, add_special_tokens=True)["input_ids"]

    n_mask = min(1 + len(prompt_ids), len(full_ids))
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        n_mask = min(n_mask, len(full_ids))

    labels = [LABEL_PAD_ID] * n_mask + full_ids[n_mask:]
    return {"input_ids": full_ids, "labels": labels}


# =====================================================================
# Fallback loader THẬT — resolve_dataset_with_fallback().
#
# Cấu trúc repo (xem app/data/publish_dataset.py):
#     ids/train.parquet, ids/val.parquet      -> config "default" (đã tokenize)
#     raw/train.parquet, raw/val.parquet      -> config "raw" (text thô)
#
# Quyết định fallback là TOÀN BỘ (all-or-nothing): nếu load config
# "default" thất bại ở bước train split, dùng "raw" cho CẢ train lẫn eval
# — không trộn train=tokenized + eval=raw (tránh lệch collator giữa 2 tập).
# =====================================================================
def resolve_dataset_with_fallback(
    dataset_name: str,
    eval_split: str = "validation",
) -> Tuple[Any, Optional[Any], bool]:
    """Trả về (train_dataset, eval_dataset_hoặc_None, is_pretokenized)."""
    from datasets import load_dataset

    def _try(config_name: str):
        train = load_dataset(dataset_name, name=config_name, split="train")
        try:
            eval_ds = load_dataset(dataset_name, name=config_name, split=eval_split)
        except Exception:
            eval_ds = None
        return train, eval_ds

    try:
        train_ds, eval_ds = _try("default")
        logger.info(f"[{dataset_name}] Load config 'default' (ids/, đã tokenize) OK -> dataset_mode=pre_tokenized (auto).")
        return train_ds, eval_ds, True
    except Exception as e:
        logger.warning(
            f"[{dataset_name}] Không load được config 'default' (ids/): {e}\n"
            f"-> Fallback config 'raw' (text thô) -> dataset_mode=on_the_fly (auto)."
        )
        train_ds, eval_ds = _try("raw")
        return train_ds, eval_ds, False


def make_data_module(
    tokenizer: PreTrainedTokenizerBase,
    data_args: DataArguments,
    is_pretrain: bool,
) -> Dict[str, Any]:
    stage = "pretrain" if is_pretrain else "sft"

    # ------------------------------------------------------------
    # Nhánh MỚI (mặc định) — auto-fallback default(ids) -> raw thật.
    # ------------------------------------------------------------
    if data_args.dataset_mode == "auto":
        train_raw, eval_raw, is_pretokenized = resolve_dataset_with_fallback(
            data_args.dataset_name, eval_split=data_args.eval_split,
        )

        # eval_dataset_name RIÊNG (khác repo hẳn) -> override, đọc đúng
        # config tương ứng (default nếu train đang pretokenized, raw nếu không)
        # để đồng nhất loại collator giữa train/eval.
        if data_args.eval_dataset_name is not None:
            from datasets import load_dataset
            config_name = "default" if is_pretokenized else "raw"
            eval_raw = load_dataset(data_args.eval_dataset_name, name=config_name, split="train")

        if is_pretokenized:
            print(f"[make_data_module:{stage}] dataset_mode=auto -> pre_tokenized (config 'default')")
            return {
                "train_dataset": train_raw,
                "eval_dataset": eval_raw,
                "data_collator": DataCollatorForPreTokenizedCoT(tokenizer=tokenizer),
            }
        else:
            print(f"[make_data_module:{stage}] dataset_mode=auto -> on_the_fly (config 'raw', fallback)")
            return {
                "train_dataset": train_raw,
                "eval_dataset": eval_raw,
                "data_collator": DataCollatorForCoT(tokenizer=tokenizer, max_length=data_args.max_length),
            }

    # ------------------------------------------------------------
    # Nhánh CŨ — chỉ định tay, không đọc config "default"/"raw", giữ lại
    # để tương thích ngược (vd dataset chưa tổ chức theo 2-config này).
    # ------------------------------------------------------------
    from datasets import load_dataset

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

    raise ValueError(f"dataset_mode không hợp lệ: {data_args.dataset_mode!r}")