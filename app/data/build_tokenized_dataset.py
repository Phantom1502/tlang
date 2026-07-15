"""
app/data/build_tokenized_dataset.py — Map dataset raw text (prompt/completion
hoặc prompt/future_bins/symbol/window_id) sang input_ids, rồi push lên 1 repo
Hub mới. Dùng cho `dataset_mode="pre_tokenized"` (docs/train_pipeline_v0.1.md
mục 3.3; docs/tokenizer_v0.1.md mục 5.2: nên bật SỚM cho pretrain/SFT ở scale
lớn vì 2 dataset này KHÔNG đổi qua các round GRPO, khác GRPO phải encode
runtime vì completion do model tự sinh lúc train).

QUAN TRỌNG: dùng LẠI ĐÚNG `_tokenize_and_mask_example` đã có ở
app/data/data_module.py — KHÔNG viết lại logic mask ở đây, để tránh 2 nơi
lệch nhau (bài học từ vụ tokenizer tự vá lành trước đó — chỉ 1 nguồn sự
thật duy nhất cho mỗi rule).

Usage:
    # Pretrain/SFT (schema prompt/completion) -> thêm input_ids/labels,
    # xoá cột text gốc (không cần nữa sau khi đã tokenize).
    python -m app.data.build_tokenized_dataset \\
        --source_repo sullivan1502/tlang-pretrain \\
        --target_repo sullivan1502/tlang-pretrain-tokenized \\
        --kind pretrain_sft

    # GRPO (schema prompt/future_bins/symbol/window_id) -> THÊM cột
    # "prompt_input_ids" bên cạnh "prompt" (GIỮ NGUYÊN, xem cảnh báo trong
    # docstring _tokenize_grpo_prompt bên dưới).
    python -m app.data.build_tokenized_dataset \\
        --source_repo sullivan1502/tlang-grpo \\
        --target_repo sullivan1502/tlang-grpo-tokenized \\
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
    p.add_argument("--source_repo", required=True, help="Repo dataset raw text trên Hub, vd sullivan1502/tlang-pretrain")
    p.add_argument("--target_repo", required=True, help="Repo đích để push dataset đã tokenize")
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
    Tokenize riêng cột "prompt" -> "prompt_input_ids" (add_special_tokens=
    False — prompt chưa phải hết câu, GRPOTrainer/model tự thêm BOS lúc
    generate, không cần EOS ở cuối vì đây là điểm bắt đầu sinh tiếp).

    CẢNH BÁO: cột "prompt" (text gốc) được GIỮ NGUYÊN, KHÔNG xoá — nhiều
    bản GRPOTrainer của TRL tự tokenize "prompt" tại runtime bằng chính
    tokenizer đã truyền vào Trainer, KHÔNG chắc chắn đọc thẳng
    "prompt_input_ids" này mà không cần chỉnh thêm code Trainer (tuỳ version
    TRL). Cột này chủ yếu để: (1) cache sẵn cho pipeline tuỳ biến ngoài TRL,
    (2) kiểm tra nhanh seq_len prompt trước khi train. KHÔNG giả định cắm
    thẳng vào GRPOTrainer mà chưa kiểm tra lại API bản TRL đang dùng.
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

    raw = load_dataset(args.source_repo)
    train_raw = raw["train"] if hasattr(raw, "keys") else raw
    logger.info(f"Loaded {args.source_repo}: {len(train_raw)} row, columns={train_raw.column_names}")

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

    mapped.push_to_hub(args.target_repo, private=args.private)
    logger.info(f"Đã push dataset đã tokenize lên: https://huggingface.co/datasets/{args.target_repo}")


if __name__ == "__main__":
    main()