from __future__ import annotations

import argparse
import logging

logger = logging.getLogger("train_pretrain")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--model_size", choices=["tiny", "small", "base", "large"], default="tiny")

    p.add_argument("--dataset_name", required=True, help="sullivan1502/tlang-pretrain-ids for pretokenized")
    p.add_argument("--dataset_mode", choices=["on_the_fly", "pre_tokenized"], default="pre_tokenized")
    p.add_argument("--max_length", type=int, default=512, help="khớp MAX_POSITION_EMBEDDINGS")


    p.add_argument("--output_dir", required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=50)

    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--hf_token", default=None, help="HF Token")
    p.add_argument("--repo_id", default=None, help="Model Repo ID on HF Hub")

    p.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    p.add_argument("--bf16", dest="fp16", action="store_false")

    return p


def build_model_for_resume_or_scratch(resume_checkpoint, model_size: str, vocab_size: int):
    from transformers import LlamaForCausalLM

    from app.training.model.configs import build_llama_config
    from app.training.common import load_model_with_vocab_check

    if resume_checkpoint is not None:
        return load_model_with_vocab_check(resume_checkpoint, vocab_size)

    logger.info(f"Init from scratch — model_size={model_size}")
    config = build_llama_config(model_size, vocab_size)
    return LlamaForCausalLM._from_config(config, attn_implementation="sdpa")


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

    # ------------------------------------------------------------
    # Tokenizer — luôn load qua Hub (app/tokenizer/hub.py), KHÔNG build lại
    # ------------------------------------------------------------
    tok = load_tokenizer(repo_id=args.repo_id, allow_local_fallback=False)
    logger.info(f"tokenizer vocab_size = {tok.vocab_size}")

    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, args.repo_id)
    model = build_model_for_resume_or_scratch(resume_checkpoint, args.model_size, tok.vocab_size)

    data_args = DataArguments(
        dataset_name=args.dataset_name,
        dataset_mode=args.dataset_mode,
        max_length=args.max_length,
    )
    data_module = make_data_module(tok, data_args, is_pretrain=True)

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
        hub_model_id=args.repo_id,
        hub_strategy="checkpoint" if push_to_hub else "every_save",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, **data_module)
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    trainer.save_model()
    tok.save_pretrained(args.output_dir)
    if push_to_hub:
        trainer.push_to_hub(commit_message="Final pretrain checkpoint")
        logger.info(f"Đã push bản final lên: https://huggingface.co/{args.repo_id}")
    else:
        logger.info(f"push_to_hub tắt — checkpoint final chỉ lưu local tại {args.output_dir}")


if __name__ == "__main__":
    main()