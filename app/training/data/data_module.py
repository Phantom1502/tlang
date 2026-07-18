from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from app.training.data.arguments import DataArguments
from app.training.data.masking import LABEL_PAD_ID, compute_labels


def _pad_encoded(
    batch_input_ids: List[List[int]],
    batch_labels: List[List[int]],
    pad_id: int,
    label_pad_token_id: int = LABEL_PAD_ID,
    pad_to_multiple_of: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    max_len = max(len(ids) for ids in batch_input_ids)
    if pad_to_multiple_of is not None:
        remainder = max_len % pad_to_multiple_of
        if remainder != 0:
            max_len += pad_to_multiple_of - remainder

    input_ids, attention_mask, labels = [], [], []
    for ids, labs in zip(batch_input_ids, batch_labels):
        pad_n = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_n)
        attention_mask.append([1] * len(ids) + [0] * pad_n)
        labels.append(labs + [label_pad_token_id] * pad_n)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


@dataclass
class DataCollatorForCoT:
    """dataset_mode="on_the_fly" — nhận {"prompt","completion"} thô, tự
    tokenize + pad. `is_pretrain` quyết định cách dựng labels:
      - pretrain: labels = full_ids (loss trên TOÀN BỘ sequence, kể cả chart)
      - sft:      labels = mask <bos>+prompt -> -100, chỉ tính loss trên completion
    """

    tokenizer: PreTrainedTokenizerBase
    is_pretrain: bool
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = LABEL_PAD_ID

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for feat in features:
            prompt, completion = feat["prompt"], feat["completion"]
            full_ids = self.tokenizer(prompt + " " + completion, add_special_tokens=True)["input_ids"]

            if self.is_pretrain:
                # full-sequence loss — KHÔNG mask gì (pretrain học cả chart, đã chốt lại thiết kế)
                if self.max_length is not None and len(full_ids) > self.max_length:
                    full_ids = full_ids[: self.max_length]
                labels = list(full_ids)
            else:
                prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                # 1 nguồn sự thật duy nhất cho rule mask — dùng chung với
                # build_tokenized_dataset.py (nhánh pre_tokenized), xem masking.py
                full_ids, labels = compute_labels(prompt_ids, full_ids, self.max_length)

            batch_input_ids.append(full_ids)
            batch_labels.append(labels)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id là None.")
        return _pad_encoded(batch_input_ids, batch_labels, pad_id, self.label_pad_token_id, self.pad_to_multiple_of)


@dataclass
class DataCollatorForPreTokenizedCoT:
    """dataset_mode="pre_tokenized" — dataset đã có sẵn input_ids/labels
    (labels build sẵn CHO SFT, đã mask prompt). `is_pretrain` quyết định
    có DÙNG labels đã mask đó hay bỏ qua, dùng input_ids làm labels
    (full-sequence loss) thay thế."""

    tokenizer: PreTrainedTokenizerBase
    is_pretrain: bool
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = LABEL_PAD_ID

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [f["input_ids"] for f in features]
        if self.is_pretrain:
            batch_labels = [list(f["input_ids"]) for f in features]   # bỏ qua labels đã mask sẵn
        else:
            batch_labels = [f["labels"] for f in features]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id là None.")
        return _pad_encoded(batch_input_ids, batch_labels, pad_id, self.label_pad_token_id, self.pad_to_multiple_of)


def make_data_module(
    tokenizer: PreTrainedTokenizerBase,
    data_args: DataArguments,
    is_pretrain: bool,
) -> Dict[str, Any]:
    stage = "pretrain" if is_pretrain else "sft"

    if data_args.dataset_mode == "on_the_fly":
        print(f"[make_data_module:{stage}] dataset_mode=on_the_fly, repo={data_args.dataset_name}")
        dataset = load_dataset(data_args.dataset_name, streaming= True, cache_dir=data_args.cache_dir)
        train_dataset = dataset[data_args.train_split].shuffle(seed=42, buffer_size=10_000)
        #train_dataset = dataset[data_args.train_split]
        return {
            "train_dataset": train_dataset,
            "eval_dataset": dataset[data_args.eval_split],
            "data_collator": DataCollatorForCoT(
                tokenizer=tokenizer, is_pretrain=is_pretrain, max_length=data_args.max_length,
            ),
        }

    if data_args.dataset_mode == "pre_tokenized":
        # LƯU Ý: đây phải là repo RIÊNG đã chứa sẵn input_ids/labels
        # (build bởi app/data/build_tokenized_dataset.py), KHÔNG phải
        # cùng repo raw — 2 repo tách biệt để dễ quản lý (theo quyết
        # định của bạn), nên data_args.dataset_name ở nhánh này phải trỏ
        # đúng repo "ids", không phải repo "raw".
        print(f"[make_data_module:{stage}] dataset_mode=pre_tokenized, repo={data_args.dataset_name}")
        dataset = load_dataset(data_args.dataset_name, streaming= True, cache_dir=data_args.cache_dir)
        train_dataset = dataset[data_args.train_split].shuffle(seed=42, buffer_size=10_000)
        #train_dataset = dataset[data_args.train_split]
        return {
            "train_dataset": train_dataset,
            "eval_dataset": dataset[data_args.eval_split],
            "data_collator": DataCollatorForPreTokenizedCoT(
                tokenizer=tokenizer, is_pretrain=is_pretrain,
            ),
        }

    raise NotImplementedError(f"dataset_mode không hợp lệ: {data_args.dataset_mode!r}")