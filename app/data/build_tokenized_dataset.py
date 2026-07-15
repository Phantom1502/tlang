"""
app/data/build_tokenized_dataset.py — Map dataset raw text sang input_ids, 
rồi push lên cùng repo nhưng nằm ở subset 'ids'.

Hỗ trợ tham số --split để chạy riêng lẻ từng tập dữ liệu (train/val), tránh chạy lại từ đầu.
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
    # Thêm tham số --split để chủ động lựa chọn tập dữ liệu cần xử lý
    p.add_argument("--split", default=None, help="Chỉ định split cụ thể để xử lý (vd: train, val). Nếu để None sẽ chạy tất cả.")
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
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    if len(prompt_ids) > max_length:
        prompt_ids = prompt_ids[:max_length]
    return {"prompt_input_ids": prompt_ids, "prompt_length": len(prompt_ids)}


def main() -> None:
    args = build_arg_parser().parse_args()

    from app.tokenizer.hub import load_tokenizer
    tok = load_tokenizer(repo_id=args.tokenizer_repo, allow_local_fallback=False)
    logger.info(f"tokenizer vocab_size = {tok.vocab_size}")

    from datasets import load_dataset, DatasetDict, Dataset

    # LOAD: Sử dụng tham số split nếu người dùng truyền vào để chỉ tải đúng phần dữ liệu đó
    if args.split:
        logger.info(f"Đang tải riêng lẻ split '{args.split}' từ subset 'raw'...")
        # Kết quả trả về từ load_dataset khi có tham số split sẽ là một đối tượng Dataset đơn lẻ
        single_ds = load_dataset(args.repo_id, name="raw", split=args.split)
        raw = DatasetDict({args.split: single_ds})
    else:
        logger.info(f"Đang tải TOÀN BỘ các split từ subset 'raw'...")
        raw = load_dataset(args.repo_id, name="raw")
        if not isinstance(raw, DatasetDict):
            raw = DatasetDict({"train": raw})

    logger.info(f"Loaded {args.repo_id} (subset 'raw') | Các split sẽ xử lý: {list(raw.keys())}")

    # 1. Định nghĩa map function dựa trên loại task (kind)
    if args.kind == "pretrain_sft":
        from app.data.data_module import _tokenize_and_mask_example

        def _map_fn(example):
            return _tokenize_and_mask_example(example, tok, args.max_length)

        # map() trên DatasetDict
        mapped = raw.map(
            _map_fn, 
            remove_columns=next(iter(raw.values())).column_names, 
            num_proc=args.num_proc,
            desc=f"Tokenize + mask (pretrain_sft, split={args.split or 'all'})",
        )

        for split_name, dataset in mapped.items():
            lens = [len(x) for x in dataset["input_ids"]]
            avg_len = sum(lens) / len(lens) if lens else 0
            logger.info(f"[{split_name}] seq_len: min={min(lens) if lens else 0} max={max(lens) if lens else 0} avg={avg_len:.1f}")
            n_truncated = sum(1 for l in lens if l >= args.max_length)
            if n_truncated:
                logger.warning(f"[{split_name}] {n_truncated}/{len(lens)} sample chạm max_length.")

    else:  # grpo
        def _map_fn(example):
            return _tokenize_grpo_prompt(example, tok, args.max_length)

        mapped = raw.map(
            _map_fn, 
            num_proc=args.num_proc,
            desc=f"Tokenize prompt (grpo, split={args.split or 'all'})",
        )

        for split_name, dataset in mapped.items():
            lens = dataset["prompt_length"]
            avg_len = sum(lens) / len(lens) if lens else 0
            logger.info(f"[{split_name}] prompt_len: min={min(lens) if lens else 0} max={max(lens) if lens else 0} avg={avg_len:.1f}")

    for split_name, dataset in mapped.items():
        logger.info(f"Kết quả split '{split_name}': {len(dataset)} row, columns={dataset.column_names}")

    if args.dry_run:
        logger.info("[DRY RUN] Không push lên Hub.")
        return

    # PUSH: Đẩy phần dữ liệu đã xử lý lên subset "ids"
    # Nếu chỉ chạy riêng lẻ một split, HF Hub thông minh ở chỗ nó sẽ chỉ thêm/ghi đè split đó 
    # trong config_name="ids" mà KHÔNG làm mất các split khác đã push trước đó.
    mapped.push_to_hub(args.repo_id, config_name="ids", private=args.private)
    logger.info(f"Đã cập nhật thành công dữ liệu tokenize lên subset 'ids' của repo: {args.repo_id}")


if __name__ == "__main__":
    main()