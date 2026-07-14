"""
app/data/collator.py — DataCollatorForCoT (train_pipeline_v0.1.md mục 3.4)

Mask loss trên phần prompt (chart_block), chỉ tính loss trên phần
completion (think_block + action_block). Dùng cho Pretrain/SFT — KHÔNG
dùng cho GRPO (GRPOTrainer tự xử lý qua unified_reward_func, không cần
collator kiểu mask loss này).

Nhận batch feature dạng {"prompt": str, "completion": str} (đúng schema
mục 7.2 spec_trading_llm_v0.2.md), tokenize ngay trong collator (đúng
dataset_mode="on_the_fly" mặc định, mục 3.3 train_pipeline_v0.1.md).

Nguyên tắc mask: full_text = prompt + " " + completion (đúng cách nối
generator/main.py đang dùng). Encode:
  - prompt_ids = tokenizer(prompt, add_special_tokens=False) -> độ dài P
    (không có <bos>/<eos>, chỉ dùng để đo ranh giới)
  - full_ids   = tokenizer(full_text, add_special_tokens=True)
               = [<bos>] + prompt_tokens (P token) + completion_tokens + [<eos>]

Vì pre-tokenizer là WhitespaceSplit + WordLevel exact-match (không BPE,
không merge token qua lại), tokenize(prompt) đứng riêng PHẢI trùng khớp
tuyệt đối với đoạn prefix tương ứng trong tokenize(full_text) — không có
hiện tượng token ở biên prompt/completion bị gộp lẫn nhau (xem
docs/tokenizer_v0.1.md mục 4).

-> mask [<bos>] + P token đầu (tổng P+1 token) = -100.
   Phần còn lại (completion + <eos>) giữ nguyên làm label để tính loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import PreTrainedTokenizerBase

LABEL_PAD_ID = -100


@dataclass
class DataCollatorForCoT:
    """
    Callable data collator — dùng trực tiếp:

        from app.tokenizer.hub import load_tokenizer
        tok = load_tokenizer()
        trainer = Trainer(..., data_collator=DataCollatorForCoT(tok))

    Args:
        tokenizer: PreTrainedTokenizerFast lấy qua
            `app.tokenizer.hub.load_tokenizer()` — PHẢI đúng tokenizer đã
            khớp vocab contract (mục 3 docs/tokenizer_v0.1.md). Collator
            này KHÔNG tự build tokenizer riêng.
        max_length: cắt bớt nếu full sequence vượt ngưỡng. Theo thiết kế
            seq_len thực tế ~230-240 token (dư nhiều so với
            max_position_embeddings=512), nên bình thường không nên bị
            cắt — mặc định 512 chỉ để phòng vệ dữ liệu bất thường.
        pad_to_multiple_of: pad batch lên bội số này (tối ưu tensor core),
            None = không áp dụng.
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = LABEL_PAD_ID

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for feat in features:
            prompt = feat["prompt"]
            completion = feat["completion"]

            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)

            full_text = prompt + " " + completion
            full_ids = self.tokenizer(full_text, add_special_tokens=True)["input_ids"]

            # Số token bị mask (<bos> + toàn bộ prompt) = 1 + prompt_len.
            n_mask = 1 + prompt_len
            if n_mask > len(full_ids):
                # Phòng vệ — không nên xảy ra với dữ liệu đúng schema, nhưng
                # không raise cứng để 1 sample lỗi không làm sập cả batch;
                # coi như mask toàn bộ (không còn completion nào để tính loss).
                n_mask = len(full_ids)

            if self.max_length is not None and len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]
                n_mask = min(n_mask, len(full_ids))

            labels = [self.label_pad_token_id] * n_mask + full_ids[n_mask:]

            batch_input_ids.append(full_ids)
            batch_labels.append(labels)

        return self._pad(batch_input_ids, batch_labels)

    def _pad(
        self, batch_input_ids: List[List[int]], batch_labels: List[List[int]]
    ) -> Dict[str, torch.Tensor]:
        max_len = max(len(ids) for ids in batch_input_ids)
        if self.pad_to_multiple_of is not None:
            remainder = max_len % self.pad_to_multiple_of
            if remainder != 0:
                max_len += self.pad_to_multiple_of - remainder

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id là None — kiểm tra lại tokenizer đã set <pad> (id=3) chưa.")

        input_ids, attention_mask, labels = [], [], []
        for ids, labs in zip(batch_input_ids, batch_labels):
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attention_mask.append([1] * len(ids) + [0] * pad_n)
            labels.append(labs + [self.label_pad_token_id] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }