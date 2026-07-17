from __future__ import annotations

from typing import List, Tuple

# =====================================================================
# 1 nguồn sự thật duy nhất cho rule mask loss SFT: mask <bos>+prompt
# (-100), chỉ tính loss trên phần completion + <eos>.
#
# Trước đây công thức "n_mask = 1 + len(prompt_ids)" bị viết tay 2 lần
# độc lập — app/training/data/data_module.py (DataCollatorForCoT,
# non-batched) và app/data_prepare/build_tokenized_dataset.py
# (_tokenize_and_mask_batch, batched) — đúng antipattern mà chính spec
# đã cảnh báo (xem docs/data_module_v0.1.md mục 2). Gộp về đây, sửa 1
# chỗ, cả 2 nơi ăn theo.
#
# CHỈ áp dụng cho nhánh SFT (mask prompt). Pretrain KHÔNG dùng hàm này —
# pretrain là full-sequence loss (học cả chart), xử lý riêng ở caller.
# =====================================================================
LABEL_PAD_ID = -100


def compute_labels(
    prompt_ids: List[int],
    full_ids: List[int],
    max_length: int | None = None,
) -> Tuple[List[int], List[int]]:
    """
    full_ids = [<bos>] + prompt_tokens + completion_tokens + [<eos>]
    (đã encode sẵn, add_special_tokens=True). prompt_ids = encode riêng
    prompt, add_special_tokens=False — dùng để suy ra ranh giới prompt/
    completion (xem docs/data_module_v0.1.md mục 2 — vì sao encode
    riêng rồi so khớp prefix là an toàn với tokenizer WordLevel +
    WhitespaceSplit, không cần return_offsets_mapping).

    Cắt max_length TRƯỚC khi tính n_mask (không phải sau) — khớp đúng
    hành vi gốc ở cả 2 nơi bị hợp nhất về đây.

    Trả về (full_ids đã cắt max_length nếu cần, labels tương ứng).
    """
    if max_length is not None and len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    n_mask = min(1 + len(prompt_ids), len(full_ids))
    labels = [LABEL_PAD_ID] * n_mask + full_ids[n_mask:]
    return full_ids, labels
