"""
push_to_hub.py — Build tokenizer từ source (vocab_builder.py) và push lên
HF Hub. Chạy 1 LẦN (hoặc mỗi khi vocab thật sự đổi — vd đổi BIN_MAX/RR_MAX ở
app/lang/tokens.py) — sau đó MỌI script khác load tokenizer bằng
`app.tokenizer.hub.load_tokenizer()`, không build lại từ source nữa.

Yêu cầu trước khi chạy:
    huggingface-cli login
    # hoặc: export HF_TOKEN=hf_xxx

Usage:
    python -m app.tokenizer.push_to_hub --repo_id sullivan1502/base-grpo-test
    python -m app.tokenizer.push_to_hub --repo_id <org>/trading-llm-tokenizer --private
    python -m app.tokenizer.push_to_hub --repo_id <org>/trading-llm-tokenizer --dry_run
"""
from __future__ import annotations

import argparse

from app.tokenizer.build_tokenizer import build_fast_tokenizer


def push_tokenizer(
    repo_id: str,
    private: bool = False,
    commit_message: str = "Build & push tokenizer (WordLevel, closed vocab)",
    token: str | None = None,
):
    """
    Build tokenizer từ source rồi push lên Hub. Trả về chính tokenizer đã
    build (để caller in vocab_size/kiểm tra thêm nếu cần) — KHÔNG load lại
    từ Hub sau khi push trong hàm này (tách trách nhiệm build vs load,
    đúng nguyên tắc "chỉ push_to_hub.py mới build từ source").
    """
    tok = build_fast_tokenizer()
    tok.push_to_hub(repo_id, private=private, commit_message=commit_message, token=token)
    return tok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True, help="vd: my-org/trading-llm-tokenizer")
    parser.add_argument("--private", action="store_true", help="Tạo/push repo ở chế độ private")
    parser.add_argument(
        "--commit_message", default="Build & push tokenizer (WordLevel, closed vocab)",
    )
    parser.add_argument(
        "--token", default=None,
        help="HF token — nếu không truyền, dùng cached login (huggingface-cli login) hoặc biến môi trường HF_TOKEN",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Chỉ build + in thông tin vocab, KHÔNG thật sự push lên Hub (kiểm tra trước khi push thật)",
    )
    args = parser.parse_args()

    if args.dry_run:
        tok = build_fast_tokenizer()
        print("[DRY RUN] Không push lên Hub. Thông tin tokenizer sẽ được push:")
        print(f"  repo_id      = {args.repo_id}")
        print(f"  private      = {args.private}")
        print(f"  vocab_size   = {tok.vocab_size}")
        print(f"  pad/bos/eos/unk id = {tok.pad_token_id}/{tok.bos_token_id}/{tok.eos_token_id}/{tok.unk_token_id}")
        print("Chạy lại KHÔNG có --dry_run để push thật.")
        return

    tok = push_tokenizer(
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
        token=args.token,
    )

    print(f"Đã push tokenizer lên: https://huggingface.co/{args.repo_id}")
    print(f"vocab_size = {tok.vocab_size}")
    print(
        f"\nCập nhật DEFAULT_TOKENIZER_REPO trong app/tokenizer/hub.py thành "
        f"{args.repo_id!r} để mọi script khác tự động load đúng repo này."
    )


if __name__ == "__main__":
    main()