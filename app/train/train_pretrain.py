from __future__ import annotations

import argparse
import logging

logger = logging.getLogger("train_pretrain")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--org", required=True)
    p.add_argument("--model_size", choices=["tiny", "small", "base", "large"], default="tiny")
    p.add_argument("--tokenizer_repo", default=None)

    p.add_argument("--dataset_name", required=True, help="vd <org>/tlang-pretrain")
    p.add_argument("--eval_dataset_name", default=None)
    p.add_argument("--eval_split", default="validation", help="Tên split eval TRONG CÙNG repo (ids/val.parquet, raw/val.parquet)")
    p.add_argument(
        "--dataset_mode", choices=["auto", "on_the_fly", "pre_tokenized"], default="auto",
        help="'auto' (mặc định, MỚI): tự thử config 'default' (ids/, đã tokenize) trước, "
             "fallback config 'raw' nếu chưa có. 2 giá trị còn lại là chỉ định tay, không auto.",
    )
    p.add_argument("--num_proc", type=int, default=4)
    p.add_argument("--max_length", type=int, default=512)

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
    p.add_argument("--push_to_hub", dest="push_to_hub", action="store_true", default=True)
    p.add_argument("--no_push_to_hub", dest="push_to_hub", action="store_false")

    p.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    p.add_argument("--bf16", dest="fp16", action="store_false")

    return p


def build_model_for_resume_or_scratch(resume_checkpoint, model_size: str, vocab_size: int):
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

    tok = load_tokenizer(repo_id=args.tokenizer_repo, allow_local_fallback=False)
    logger.info(f"tokenizer vocab_size = {tok.vocab_size}")

    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, checkpoint_repo, args.push_to_hub)
    model = build_model_for_resume_or_scratch(resume_checkpoint, args.model_size, tok.vocab_size)

    data_args = DataArguments(
        dataset_name=args.dataset_name,
        dataset_mode=args.dataset_mode,
        eval_dataset_name=args.eval_dataset_name,
        eval_split=args.eval_split,
        num_proc=args.num_proc,
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
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    trainer.save_model()
    tok.save_pretrained(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub(commit_message="Final pretrain checkpoint")
        logger.info(f"Đã push bản final lên: https://huggingface.co/{checkpoint_repo}")
    else:
        logger.info(f"push_to_hub tắt — checkpoint final chỉ lưu local tại {args.output_dir}")


if __name__ == "__main__":
    main()