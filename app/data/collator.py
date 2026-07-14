"""
app/data/collator.py — DataCollatorForCoT + DataCollatorForPreTokenizedCoT
(train_pipeline_v0.1.md mục 3.4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import PreTrainedTokenizerBase

LABEL_PAD_ID = -100


def _pad_encoded(
    batch_input_ids: List[List[int]],
    batch_labels: List[List[int]],
    pad_id: int,
    label_pad_token_id: int = LABEL_PAD_ID,
    pad_to_multiple_of: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Pad input_ids/labels đã tính sẵn — dùng chung cho cả 2 collator
    (on_the_fly tự tính rồi pad ngay, pre_tokenized chỉ pad lại cái đã
    tính sẵn ở bước .map()). Tách ra đây để 2 nơi không lệch cách pad."""
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
    """dataset_mode="on_the_fly" — nhận {"prompt","completion"} (text thô),
    tự tokenize + mask (bos+prompt -> -100) + pad mỗi batch."""

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = LABEL_PAD_ID

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for feat in features:
            prompt, completion = feat["prompt"], feat["completion"]

            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(prompt + " " + completion, add_special_tokens=True)["input_ids"]

            n_mask = min(1 + len(prompt_ids), len(full_ids))
            if self.max_length is not None and len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]
                n_mask = min(n_mask, len(full_ids))

            labels = [self.label_pad_token_id] * n_mask + full_ids[n_mask:]
            batch_input_ids.append(full_ids)
            batch_labels.append(labels)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id là None — kiểm tra lại tokenizer đã set <pad> (id=3) chưa.")
        return _pad_encoded(batch_input_ids, batch_labels, pad_id, self.label_pad_token_id, self.pad_to_multiple_of)


@dataclass
class DataCollatorForPreTokenizedCoT:
    """dataset_mode="pre_tokenized" — input_ids/labels ĐÃ được tính sẵn 1
    lần qua .map() (app/data/data_module.py:_tokenize_and_mask_example).
    Collator này CHỈ pad, không tokenize/mask gì thêm — nếu mask sai thì
    lỗi nằm ở bước .map(), không phải ở đây (tách trách nhiệm rõ ràng)."""

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = LABEL_PAD_ID

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [f["input_ids"] for f in features]
        batch_labels = [f["labels"] for f in features]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id là None — kiểm tra lại tokenizer đã set <pad> (id=3) chưa.")
        return _pad_encoded(batch_input_ids, batch_labels, pad_id, self.label_pad_token_id, self.pad_to_multiple_of)