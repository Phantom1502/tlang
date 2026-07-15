"""
app/data/build_tokenized_dataset.py — Map dataset raw text (prompt/completion
hoặc prompt/future_bins/symbol/window_id) sang input_ids, rồi push lên cùng repo
nhưng nằm ở subset 'ids'.

Usage:
    # Pretrain/SFT (schema prompt/completion) -> thêm input_ids/labels,
    # xoá cột text gốc.
    python -m app.data.build_tokenized_dataset \
        --repo_id sullivan1502/tlang-pretrain \
        --kind pretrain_sft

    # GRPO (schema prompt/future_bins/symbol/window_id) -> THÊM cột
    # "prompt_input_ids" bên cạnh "prompt" (GIỮ NGUYÊN).
    python -m app.data.build_tokenized_dataset \
        --repo_id sullivan1502/tlang-grpo \
        --kind grpo

Dry-run (không push, chỉ .map() + in thống kê để kiểm tra trước):
    ... --dry_run
"""
from __future__ import annotations

import argparse
import logging
from typing import Any, Dict

logger = logging.getLogger("build_tokenized_dataset")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", required=True, help="Repo dataset trên Hub (chứa cả subset 'raw' và 'ids')")
    p.add_argument("--kind", choices=["pretrain_sft", "grpo"], required=True)
    p.add_argument("--tokenizer_repo", default=None, help="Mặc định DEFAULT_TOKENIZER_REPO trong app/tokenizer/hub.py")
    p.add_argument("--max_length", type=int, default=512, help="Khớp MAX_POSITION_EMBEDDINGS")
    p.add_argument("--num_proc", type=int, default=4)
    p.add_argument("--private", action="store_true")
    p.add_argument(
        "--dry_run", action="store_true",
        help="Chỉ chạy .map() + in thống kê seq_len, KHÔNG push lên Hub — luôn chạy trước khi push thật",
    )
    return p


def _tokenize_grpo_prompt(example: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, Any]:
    """
    Tokenize riêng cột "prompt" -> "prompt_input_ids" (add_special_tokens=False)
    """
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    if len(prompt_ids) > max_length:
        prompt_ids = prompt_ids[:max_length]
    return {"prompt_input_ids": prompt_ids, "prompt_length": len(prompt_ids)}


def main() -> None:
    args = build_arg_parser().parse_args()

    from app.tokenizer.hub import load_tokenizer

    tok = load_tokenizer(repo_id=args.tokenizer_repo, allow_local_fallback=False)
    logger.info(f"tokenizer vocab_size = {tok.vocab_size}")

    from datasets import load_dataset

    # LOAD: Chỉ định rõ cấu hình (name) là "raw" để load đúng từ thư mục raw/
    raw = load_dataset(args.repo_id, name="raw")
    train_raw = raw["train"] if hasattr(raw, "keys") else raw
    logger.info(f"Loaded {args.repo_id} (subset 'raw'): {len(train_raw)} row, columns={train_raw.column_names}")

    if args.kind == "pretrain_sft":
        from app.data.data_module import _tokenize_and_mask_example  # DÙNG LẠI, không viết lại

        def _map_fn(example):
            return _tokenize_and_mask_example(example, tok, args.max_length)

        mapped = train_raw.map(
            _map_fn, remove_columns=train_raw.column_names, num_proc=args.num_proc,
            desc="Tokenize + mask (pretrain_sft, pre_tokenized)",
        )

        lens = [len(x) for x in mapped["input_ids"]]
        logger.info(f"seq_len: min={min(lens)} max={max(lens)} avg={sum(lens) / len(lens):.1f}")
        n_truncated = sum(1 for l in lens if l >= args.max_length)
        if n_truncated:
            logger.warning(
                f"{n_truncated}/{len(lens)} sample chạm/đúng max_length={args.max_length} "
                f"— có thể đã bị cắt cụt completion, cân nhắc tăng max_length."
            )

    else:  # grpo
        def _map_fn(example):
            return _tokenize_grpo_prompt(example, tok, args.max_length)

        mapped = train_raw.map(
            _map_fn, num_proc=args.num_proc,
            desc="Tokenize prompt (grpo, thêm cột prompt_input_ids)",
        )

        lens = mapped["prompt_length"]
        logger.info(f"prompt_len: min={min(lens)} max={max(lens)} avg={sum(lens) / len(lens):.1f}")

    logger.info(f"Kết quả: {len(mapped)} row, columns={mapped.column_names}")

    if args.dry_run:
        logger.info("[DRY RUN] Không push lên Hub. Kiểm tra thống kê ở trên, chạy lại KHÔNG có --dry_run để push thật.")
        return

    # PUSH: Đẩy lên cùng repo nhưng khai báo config_name="ids"
    # Hugging Face sẽ tự tạo folder ids/ và tự update metadata cấu hình trong README.md
    mapped.push_to_hub(args.repo_id, config_name="ids", private=args.private)
    logger.info(f"Đã push dataset đã tokenize lên subset 'ids' của repo: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()