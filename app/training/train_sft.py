"""
train_sft.py — Script SFT (docs/train_pipeline_v0.1.md mục 5).

CÙNG PATTERN resumable như train_pretrain.py (mục 5.1: "cùng pattern
resumable như pretrain, khác nguồn checkpoint gốc") — logic resume dùng
lại y hệt qua `app.train.common.resolve_resume_checkpoint`. Điểm khác duy
nhất so với pretrain nằm ở NGUỒN INIT khi chưa có gì để resume:

    - train_pretrain.py: chưa có gì -> init from scratch (LlamaConfig).
    - train_sft.py (file này): chưa có gì -> load checkpoint đã HOÀN TẤT
      của pretrain (`<org>/trading-llm-<size>-pretrain`, bản ROOT/final —
      KHÔNG phải subfolder `last-checkpoint/` của pretrain, vì đó là
      optimizer/scheduler state của pretrain, không liên quan gì tới 1
      training run SFT mới, xem docstring `load_model_with_vocab_check`).

Cụ thể (mục 5.1):
    sft_repo tồn tại (local hoặc Hub, qua resolve_resume_checkpoint)
        -> resume TRAINING CỦA CHÍNH SFT (đã train dở, session bị ngắt)
    sft_repo CHƯA tồn tại
        -> bắt đầu SFT lần đầu, load weights từ pretrain_repo (fresh
           optimizer/scheduler — đây là 1 training run mới, không phải
           resume training cũ của pretrain)

Tức là: lần chạy đầu tiên của SFT bắt nguồn từ pretrain; các lần chạy sau
(session bị ngắt, chạy tiếp SFT) phải resume từ chính SFT checkpoint, KHÔNG
load lại từ pretrain mỗi lần (nếu không sẽ mất tiến độ SFT đã train).

Dataset: `<org>/trading-llm-sft` — cùng schema prompt/completion với
pretrain (mục 7.2: "SFT dùng cùng schema, chỉ khác nguồn random-gen").

Usage:
    python train_sft.py \
        --org my-org --model_size tiny \
        --dataset_name my-org/trading-llm-sft \
        --output_dir ./out/sft-tiny \
        --save_steps 500 --max_steps 20000

Chạy thử nhanh không push lên Hub thật (dev/test cục bộ) — LƯU Ý: vẫn cần
pretrain_repo tồn tại thật trên Hub để load nguồn init, vì --no_push_to_hub
chỉ tắt việc PUSH kết quả SFT, không thay được nguồn load pretrain:
    python train_sft.py --org my-org --model_size tiny \
        --dataset_name my-org/trading-llm-sft \
        --output_dir ./out/sft-tiny --no_push_to_hub --max_steps 20
"""
from __future__ import annotations

import argparse
import logging

logger = logging.getLogger("train_sft")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- Model / checkpoint naming (mục 1.2, mục 5.1) ---
    p.add_argument("--model_size", choices=["tiny", "small", "base", "large"], default="tiny")
    p.add_argument("--pretrain_repo", default=None, help="Pretrain repo để train sft, cần có để init sft",)

    # --- Dataset (mục 3, mục 7.2) ---
    p.add_argument("--dataset_name", required=True, help="sullivan1502/tlang-pretrain-ids for pretokenized")
    p.add_argument("--dataset_mode", choices=["on_the_fly", "pre_tokenized"], default="pre_tokenized")
    p.add_argument("--max_length", type=int, default=512, help="khớp MAX_POSITION_EMBEDDINGS")

    # --- Training loop ---
    p.add_argument("--output_dir", required=True, help="local checkpoint dir — dùng để detect resume trong-session")
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4, help="mặc định thấp hơn pretrain — SFT fine-tune từ checkpoint có sẵn")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_steps", type=int, default=-1, help="-1 = dùng num_train_epochs thay thế")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=50)

    # --- Push theo chu kỳ (mục 4.2, áp dụng lại cho SFT theo mục 5.1) ---
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--hf_token", default=None, help="HF Token")
    p.add_argument("--repo_id", default=None, help="Model Repo ID on HF Hub")

    # --- Hạ tầng (mục 7) ---
    p.add_argument("--fp16", dest="fp16", action="store_true", default=True, help="Mặc định BẬT — T4 không có bf16 tensor core tốt")
    p.add_argument("--bf16", dest="fp16", action="store_false", help="Dùng bf16 thay fp16 — chỉ bật nếu chạy trên A100/H100")

    return p


def build_model_for_resume_or_from_pretrain(resume_checkpoint, pretrain_repo: str, vocab_size: int):
    """
    Nếu có `resume_checkpoint` (SFT đã train dở, session bị ngắt — local
    hoặc vừa tải subfolder `last-checkpoint/` của chính sft_repo): load
    qua đó, giữ nguyên optimizer/scheduler state đã có (`trainer.train
    (resume_from_checkpoint=...)` sẽ khôi phục đầy đủ, việc load ở đây
    chỉ để có 1 model instance hợp lệ trước khi khởi tạo Trainer).

    Nếu chưa từng train SFT lần nào (`resume_checkpoint is None`): BẮT
    BUỘC phải có `pretrain_repo` đã train xong trên Hub — đây là nguồn
    init duy nhất cho SFT (mục 5.1: "lần chạy đầu tiên của SFT bắt nguồn
    từ checkpoint pretrain"). Raise lỗi rõ ràng nếu thiếu, thay vì âm
    thầm init from scratch (SFT KHÔNG có nhánh from-scratch — khác
    pretrain).
    """
    from app.training.common import load_model_with_vocab_check

    if resume_checkpoint is not None:
        return load_model_with_vocab_check(resume_checkpoint, vocab_size)

    from huggingface_hub import repo_exists

    if not repo_exists(pretrain_repo):
        raise RuntimeError(
            f"Chưa có checkpoint SFT nào để resume, VÀ pretrain_repo {pretrain_repo!r} chưa tồn tại "
            f"trên Hub — SFT cần checkpoint pretrain đã train xong làm nguồn init (mục 5.1). Chạy "
            f"train_pretrain.py xong trước, hoặc truyền --pretrain_repo trỏ đúng repo đã có."
        )

    logger.info(f"Chưa có checkpoint SFT nào — bắt đầu từ pretrain: {pretrain_repo}")
    return load_model_with_vocab_check(pretrain_repo, vocab_size)


def main() -> None:
    args = build_arg_parser().parse_args()

    from transformers import Trainer, TrainingArguments

    from app.training.data.data_module import DataArguments, make_data_module
    from app.tokenizer.hub import load_tokenizer
    from app.training.common import resolve_resume_checkpoint
    import os
    import torch
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
        
    push_to_hub = False
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        if args.repo_id:
            push_to_hub = True
    elif args.repo_id:
        print("Có repo_id nhưng chưa có hf_token — nhớ gọi huggingface_hub.login() thủ công trước khi chạy.")
       
    if args.pretrain_repo is None:
        print("Chưa cố checkpoint pretrain — yêu cầu set --pretrain_repo.")
        exit(1)
    # ------------------------------------------------------------
    # Tokenizer — luôn load qua Hub (app/tokenizer/hub.py), KHÔNG build lagi
    # từ source ở script train (mục 7.0 docs/tokenizer_v0.1.md).
    # ------------------------------------------------------------
    tok = load_tokenizer(repo_id=args.repo_id, allow_local_fallback=False)
    logger.info(f"tokenizer vocab_size = {tok.vocab_size}")
    
    # ------------------------------------------------------------
    # Resume checkpoint CỦA CHÍNH SFT (mục 5.1) — tìm TRƯỚC khi khởi tạo
    # Trainer. Nếu None, model sẽ init từ pretrain_repo (không phải
    # from-scratch — điểm khác biệt duy nhất so với train_pretrain.py).
    # ------------------------------------------------------------
    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, args.repo_id)
    model = build_model_for_resume_or_from_pretrain(resume_checkpoint, args.pretrain_repo, tok.vocab_size)

    # ------------------------------------------------------------
    # Data — make_data_module (docs/data_module_v0.1.md), giống hệt
    # pretrain, chỉ đổi dataset_name trỏ tới <org>/trading-llm-sft.
    # ------------------------------------------------------------
    data_args = DataArguments(
        dataset_name=args.dataset_name,
        dataset_mode=args.dataset_mode,
        max_length=args.max_length,
    )
    data_module = make_data_module(tok, data_args, is_pretrain=False)

    # ------------------------------------------------------------
    # TrainingArguments — giống hệt train_pretrain.py (mục 5.1: "Trainer
    # giống hệt pretrain script, chỉ đổi nguồn model + dataset_name"),
    # kể cả remove_unused_columns=False (lý do y hệt — xem comment trong
    # train_pretrain.py) và push theo chu kỳ (mục 4.2, áp dụng lại đây).
    # ------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=not args.fp16,
        push_to_hub=push_to_hub,
        hub_model_id=args.repo_id if push_to_hub else None,
        hub_strategy="checkpoint" if push_to_hub else "every_save",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, **data_module)

    # resume_checkpoint ở đây LUÔN thuộc về sft_repo (nếu có) — KHÔNG bao
    # giờ trỏ vào pretrain (init từ pretrain không mang theo
    # optimizer/scheduler, đúng ý "1 training run mới").
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ------------------------------------------------------------
    # Bản "final" tách biệt các checkpoint giữa chừng (mục 4.2, áp dụng
    # lại cho SFT) — luôn chạy dù push_to_hub bật hay tắt.
    # ------------------------------------------------------------
    trainer.save_model()
    tok.save_pretrained(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub(commit_message="Final SFT checkpoint")
        logger.info(f"Đã push bản final lên: https://huggingface.co/{args.repo_id}")
    else:
        logger.info(f"push_to_hub tắt (--no_push_to_hub) — checkpoint final chỉ lưu local tại {args.output_dir}")


if __name__ == "__main__":
    main()