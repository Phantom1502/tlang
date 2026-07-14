"""
train_pretrain.py — Script Pretrain (docs/train_pipeline_v0.1.md mục 4).

Luồng resumable (mục 4.1): tự kiểm tra local rồi tới Hub trước khi quyết
định init model mới hay resume — vì session Colab/Kaggle free-tier có
giới hạn thời gian, không thể giả định mỗi lần chạy là lần đầu tiên.

Phân biệt 2 loại "resume" khác nhau, CẢ HAI đều khôi phục ĐẦY ĐỦ
weights + optimizer + scheduler + rng_state (không chỉ model weights):
    - Còn local checkpoint dir (session chưa bị ngắt, chạy .train() lần 2
      liên tiếp) -> resume_from_checkpoint trỏ thẳng vào đó, không cần
      tải gì.
    - Mất local checkpoint (session mới hoàn toàn) -> vì
      hub_strategy="checkpoint" push TOÀN BỘ state (không chỉ weights)
      vào subfolder "last-checkpoint/" trên Hub repo, script tự TẢI
      subfolder đó về rồi truyền path local vào resume_from_checkpoint —
      Trainer khôi phục lại đúng optimizer/scheduler/rng đã push, KHÔNG
      phải quay về warmup từ đầu.
    - Không có gì cả ở cả 2 nơi (lần chạy đầu tiên tuyệt đối) -> init
      from scratch theo --model_size.

Push theo chu kỳ (mục 4.2), không phải 1 lần lúc kết thúc — dùng cơ chế có
sẵn của Trainer (push_to_hub=True, hub_strategy="checkpoint"), không viết
callback riêng.

Usage:
    python train_pretrain.py \
        --org my-org --model_size tiny \
        --dataset_name my-org/trading-llm-pretrain \
        --output_dir ./out/pretrain-tiny \
        --save_steps 500 --max_steps 20000

Chạy thử nhanh không push lên Hub thật (dev/test cục bộ):
    python train_pretrain.py --org my-org --model_size tiny \
        --dataset_name my-org/trading-llm-pretrain \
        --output_dir ./out/pretrain-tiny --no_push_to_hub --max_steps 20
"""
from __future__ import annotations

import argparse
import logging

logger = logging.getLogger("train_pretrain")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- Model / checkpoint naming (mục 1.2, mục 4.1) ---
    p.add_argument("--org", required=True, help="HF org/username dùng để đặt tên mọi repo (checkpoint + tokenizer)")
    p.add_argument("--model_size", choices=["tiny", "small", "base", "large"], default="tiny")
    p.add_argument(
        "--tokenizer_repo", default=None,
        help="Repo tokenizer trên Hub — mặc định dùng DEFAULT_TOKENIZER_REPO trong app/tokenizer/hub.py",
    )

    # --- Dataset (mục 3) ---
    p.add_argument("--dataset_name", required=True, help="vd <org>/trading-llm-pretrain")
    p.add_argument("--eval_dataset_name", default=None)
    p.add_argument("--dataset_mode", choices=["on_the_fly", "pre_tokenized"], default="on_the_fly")
    p.add_argument("--num_proc", type=int, default=4, help="chỉ dùng khi dataset_mode=pre_tokenized")
    p.add_argument("--max_length", type=int, default=512, help="khớp MAX_POSITION_EMBEDDINGS")

    # --- Training loop ---
    p.add_argument("--output_dir", required=True, help="local checkpoint dir — dùng để detect resume trong-session")
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_steps", type=int, default=-1, help="-1 = dùng num_train_epochs thay thế")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=50)

    # --- Push theo chu kỳ (mục 4.2) ---
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument(
        "--push_to_hub", dest="push_to_hub", action="store_true", default=True,
        help="Mặc định BẬT — đúng thiết kế mục 4.2 (push theo chu kỳ, không chỉ lúc kết thúc)",
    )
    p.add_argument(
        "--no_push_to_hub", dest="push_to_hub", action="store_false",
        help="Tắt push — CHỈ dùng cho dev/test cục bộ (vd chạy vài step kiểm tra script không crash)",
    )

    # --- Hạ tầng (mục 7) ---
    p.add_argument("--fp16", dest="fp16", action="store_true", default=True, help="Mặc định BẬT — T4 không có bf16 tensor core tốt")
    p.add_argument("--bf16", dest="fp16", action="store_false", help="Dùng bf16 thay fp16 — chỉ bật nếu chạy trên A100/H100")

    return p


def build_model_for_resume_or_scratch(resume_checkpoint, model_size: str, vocab_size: int):
    """
    Nếu có `resume_checkpoint` (local hoặc vừa tải từ Hub — xem
    `app.train.common.resolve_resume_checkpoint`): load qua
    `load_model_with_vocab_check()` để lấy đúng architecture/config đã
    lưu trong checkpoint đó (không dựa vào `--model_size` CLI đoán mò).
    Trọng số này sẽ bị `trainer.train(resume_from_checkpoint=...)` load
    lại 1 lần nữa (hơi dư nhưng vô hại) — quan trọng hơn là bước
    `resume_from_checkpoint` mới là bước khôi phục optimizer/scheduler/
    rng/global_step, không phải bước load ở đây.

    Nếu không có checkpoint nào (lần chạy đầu) -> init from scratch theo
    đúng `--model_size` đã chọn (mục 1.2) — pretrain là stage DUY NHẤT
    có nhánh "init from scratch" này (SFT/GRPO luôn init từ 1 checkpoint
    có sẵn của stage trước, xem train_sft.py).
    """
    from transformers import LlamaForCausalLM

    from app.model.model_configs import build_llama_config
    from app.train.common import load_model_with_vocab_check

    if resume_checkpoint is not None:
        return load_model_with_vocab_check(resume_checkpoint, vocab_size)

    logger.info(f"Init from scratch — model_size={model_size}")
    config = build_llama_config(model_size, vocab_size)
    return LlamaForCausalLM._from_config(config, attn_implementation="sdpa")


def main() -> None:
    args = build_arg_parser().parse_args()

    from transformers import Trainer, TrainingArguments

    from app.data.data_module import DataArguments, make_data_module
    from app.tokenizer.hub import load_tokenizer
    from app.train.common import resolve_resume_checkpoint

    checkpoint_repo = f"{args.org}/trading-llm-{args.model_size}-pretrain"

    # ------------------------------------------------------------
    # Tokenizer — luôn load qua Hub (app/tokenizer/hub.py), KHÔNG build lại
    # từ source ở script train (đúng nguyên tắc "1 nguồn duy nhất" mục 7.0
    # docs/tokenizer_v0.1.md). allow_local_fallback=False để lỗi mạng/config
    # sai lộ ra ngay thay vì âm thầm train bằng tokenizer build-lại-tại-chỗ.
    # ------------------------------------------------------------
    tok = load_tokenizer(repo_id=args.tokenizer_repo, allow_local_fallback=False)
    logger.info(f"tokenizer vocab_size = {tok.vocab_size}")

    # ------------------------------------------------------------
    # Resume checkpoint (mục 4.1) — tìm TRƯỚC khi khởi tạo Trainer, vì
    # cả model init lẫn training_args đều cần biết có đang resume hay không.
    # ------------------------------------------------------------
    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, checkpoint_repo, args.push_to_hub)
    model = build_model_for_resume_or_scratch(resume_checkpoint, args.model_size, tok.vocab_size)

    # ------------------------------------------------------------
    # Data — make_data_module (docs/data_module_v0.1.md)
    # ------------------------------------------------------------
    data_args = DataArguments(
        dataset_name=args.dataset_name,
        dataset_mode=args.dataset_mode,
        eval_dataset_name=args.eval_dataset_name,
        num_proc=args.num_proc,
        max_length=args.max_length,
    )
    data_module = make_data_module(tok, data_args, is_pretrain=True)

    # ------------------------------------------------------------
    # TrainingArguments — push theo chu kỳ (mục 4.2), fp16 mặc định (mục 7)
    #
    # remove_unused_columns=False LUÔN LUÔN cần thiết ở đây, không chỉ cho
    # GRPO: dataset_mode="on_the_fly" trả cột "prompt"/"completion" (text
    # thô) mà model.forward() không nhận trực tiếp — nếu để True, Trainer
    # sẽ tự xoá 2 cột này TRƯỚC KHI data_collator kịp thấy, làm
    # DataCollatorForCoT nhận batch rỗng. Giữ False đồng nhất cho cả
    # dataset_mode="pre_tokenized" (input_ids/labels không bị ảnh hưởng gì
    # khi remove_unused_columns=False) để hành vi không đổi giữa 2 mode.
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
        push_to_hub=args.push_to_hub,
        hub_model_id=checkpoint_repo if args.push_to_hub else None,
        hub_strategy="checkpoint" if args.push_to_hub else "every_save",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if data_module.get("eval_dataset") is not None else "no",
        eval_steps=args.save_steps if data_module.get("eval_dataset") is not None else None,
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, **data_module)

    # resume_checkpoint đã được xác định ở trên (local HOẶC vừa tải từ Hub
    # subfolder last-checkpoint/) — truyền thẳng vào đây để Trainer khôi
    # phục optimizer/scheduler/rng/global_step, không chỉ model weights.
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ------------------------------------------------------------
    # Đảm bảo có 1 bản "final" rõ ràng, tách biệt các checkpoint giữa
    # chừng (mục 4.2) — luôn chạy dù push_to_hub bật hay tắt.
    # ------------------------------------------------------------
    trainer.save_model()
    tok.save_pretrained(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub(commit_message="Final pretrain checkpoint")
        logger.info(f"Đã push bản final lên: https://huggingface.co/{checkpoint_repo}")
    else:
        logger.info(f"push_to_hub tắt (--no_push_to_hub) — checkpoint final chỉ lưu local tại {args.output_dir}")


if __name__ == "__main__":
    main()