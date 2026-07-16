"""
app/train/common.py — Logic dùng chung giữa train_pretrain.py và
train_sft.py (docs/train_pipeline_v0.1.md mục 4.1/5.1: "cùng pattern
resumable như pretrain, khác nguồn checkpoint gốc").

Tách ra đây thay vì lặp lại ở mỗi script, để rule resume/vocab-check chỉ
có 1 nguồn sự thật duy nhất — sửa 1 chỗ, cả 2 script (và GRPO sau này) ăn
theo.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("app.train.common")

HUB_CHECKPOINT_SUBFOLDER = "last-checkpoint"  # thư mục con mà hub_strategy="checkpoint" push vào


def resolve_resume_checkpoint(
    output_dir: str, checkpoint_repo: str
) -> Optional[str]:
    """
    Tìm 1 checkpoint ĐẦY ĐỦ (weights + optimizer + scheduler + rng_state)
    để resume TRAINING CỦA CHÍNH `checkpoint_repo` NÀY (không phải nguồn
    init ban đầu — xem `load_model_with_vocab_check` cho việc đó), ưu
    tiên theo thứ tự:

    1. Local checkpoint dir trong `output_dir` (session chưa bị ngắt, chạy
       `.train()` lần 2 liên tiếp) — có sẵn optimizer/scheduler state trên
       đĩa, không cần tải gì.
    2. Session mới hoàn toàn (mất local) — `hub_strategy="checkpoint"`
       push TOÀN BỘ state (không chỉ weights) vào subfolder
       `last-checkpoint/` trên Hub repo, nên phải TẢI subfolder đó về rồi
       trả path local để `trainer.train(resume_from_checkpoint=...)` tự
       khôi phục đúng optimizer/scheduler/rng.
    3. Không có gì cả (lần đầu tiên train `checkpoint_repo` này) -> None.
    """
    from transformers.trainer_utils import get_last_checkpoint

    local_checkpoint = get_last_checkpoint(output_dir) if output_dir else None
    if local_checkpoint is not None:
        logger.info(f"Tìm thấy local checkpoint: {local_checkpoint} — resume optimizer/scheduler/rng tại chỗ")
        return local_checkpoint

    from huggingface_hub import repo_exists, snapshot_download

    if not repo_exists(checkpoint_repo):
        logger.info(f"Không có local checkpoint, chưa có repo {checkpoint_repo} trên Hub -> lần chạy đầu tiên")
        return None

    logger.info(
        f"Không có local checkpoint nhưng {checkpoint_repo} đã tồn tại trên Hub — "
        f"tải subfolder '{HUB_CHECKPOINT_SUBFOLDER}/' (full state, không chỉ weights) về local"
    )
    try:
        local_repo_dir = snapshot_download(
            repo_id=checkpoint_repo,
            allow_patterns=[f"{HUB_CHECKPOINT_SUBFOLDER}/*"],
        )
    except Exception as e:
        logger.warning(f"Tải checkpoint từ Hub thất bại ({e}) — fallback coi như chưa có gì")
        return None

    downloaded_checkpoint = os.path.join(local_repo_dir, HUB_CHECKPOINT_SUBFOLDER)
    if not os.path.isdir(downloaded_checkpoint):
        logger.warning(
            f"Repo {checkpoint_repo} tồn tại nhưng KHÔNG có subfolder '{HUB_CHECKPOINT_SUBFOLDER}/' "
            f"(có thể lần trước push bằng hub_strategy khác, hoặc chưa save_steps nào chạy) — "
            f"fallback coi như chưa có gì."
        )
        return None

    logger.info(f"Đã tải xong — resume từ: {downloaded_checkpoint}")
    return downloaded_checkpoint


def load_model_with_vocab_check(source: str, vocab_size: int):
    """
    `LlamaForCausalLM.from_pretrained(source)` + kiểm tra vocab_size khớp
    tokenizer hiện tại. Dùng cho CẢ 2 tình huống (khác nhau về Ý NGHĨA
    của `source`, giống nhau về cách load + validate):

    - Resume checkpoint của chính repo đang train (local path hoặc path
      vừa tải từ `last-checkpoint/` — xem `resolve_resume_checkpoint`).
    - Init từ checkpoint HOÀN TẤT của stage trước (vd SFT load từ
      `<org>/trading-llm-<size>-pretrain` — model ROOT của repo đó, tức
      bản "final" đã push lúc `trainer.push_to_hub()` cuối pretrain,
      KHÔNG phải subfolder `last-checkpoint/` của pretrain — vì đó là
      optimizer/scheduler state CỦA PRETRAIN, không liên quan gì tới 1
      training run mới của SFT).

    Raise ValueError nếu vocab_size lệch — vi phạm vocab contract (mục 3
    docs/tokenizer_v0.1.md), không âm thầm train tiếp trên embedding
    table sai kích thước.
    """
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(source)
    if model.config.vocab_size != vocab_size:
        raise ValueError(
            f"vocab_size của checkpoint tại {source!r} ({model.config.vocab_size}) KHÔNG khớp "
            f"vocab_size tokenizer hiện tại ({vocab_size}) — vocab đã đổi từ lần train checkpoint "
            f"này, không thể dùng an toàn (vi phạm vocab contract, xem docs/tokenizer_v0.1.md mục 3)."
        )
    return model