"""
build_tokenizer.py — Build tokenizer thật (Rust backend, `tokenizers`
library) từ vocab đóng ở vocab_builder.py, rồi wrap thành
`PreTrainedTokenizerFast` để dùng trực tiếp với `transformers`/`trl`.

TẠI SAO WordLevel, KHÔNG BPE:
- Vocab đã đóng, liệt kê tường minh (xem vocab_builder.py) — không có
  khái niệm "học merge từ tần suất" ở đây, khác hẳn tiền đề của BPE.
- BPE có thể tự ý merge 2 digit token đứng cạnh nhau ("0" + "5" -> "05"),
  phá vỡ đúng lợi ích digit-decompose mà spec cố tình thiết kế (mục 3:
  "để model học so sánh/số học... thay vì tra bảng 1024 embedding độc
  lập"). Đây chính là lesson đã ghi trong spec — không lặp lại.

TẠI SAO Whitespace-only pre-tokenizer là ĐỦ:
- Generator (`app/gen/generator.py`, hàm `_build_completion_text` /
  `_build_chart_text`) luôn in mỗi token cách nhau ĐÚNG 1 khoảng trắng —
  kể cả từng digit rời. Điều kiện này đã được ghi rõ trong spec mục 3.
- Vì vậy KHÔNG cần 1 pre-tokenizer regex phức tạp mô phỏng lại
  `_MASTER_RE` của Lexer — chỉ cần tách theo whitespace, sau đó
  WordLevel tra bảng exact-match. Dùng `pre_tokenizers.WhitespaceSplit`
  (chỉ tách trên khoảng trắng ASCII, KHÔNG tách theo punctuation như
  `pre_tokenizers.Whitespace` mặc định — quan trọng vì token dạng
  "<O_543>" chứa ký tự không phải \\w và phải giữ nguyên vẹn).
- Hệ quả: encode() không "tự sửa" cấu trúc hỏng. Token lạ (không có
  trong vocab, hoặc completion rác từ GRPO rollout tự sinh) map thẳng
  sang <unk> — đúng nguyên tắc "tokenizer phải là phép ánh xạ trung
  thực, không tự vá lành" (spec mục 3, bài học từ bug tokenizer cũ).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from app.tokenizer.vocab_builder import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    build_vocab,
)


def build_raw_tokenizer() -> Tokenizer:
    vocab = build_vocab()

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = WhitespaceSplit()

    bos_id = vocab[BOS_TOKEN]
    eos_id = vocab[EOS_TOKEN]

    # Thêm <bos>/<eos> tự động quanh completion khi encode 1 chuỗi đơn —
    # khớp LlamaConfig(bos_token_id=1, eos_token_id=2) trong
    # docs/train_pipeline_v0.1.md mục 1.1. Không thêm token đặc biệt nào
    # khác (không có [CLS]/[SEP] kiểu BERT — kiến trúc là causal LM).
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} {BOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[(BOS_TOKEN, bos_id), (EOS_TOKEN, eos_id)],
    )

    return tokenizer


def build_fast_tokenizer() -> PreTrainedTokenizerFast:
    raw_tokenizer = build_raw_tokenizer()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token=UNK_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
    )
    return fast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir", type=str, default="./tokenizer_out",
        help="Thư mục lưu tokenizer (tokenizer.json + tokenizer_config.json ...)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fast = build_fast_tokenizer()
    fast.save_pretrained(str(out_dir))

    print(f"Đã lưu tokenizer vào: {out_dir.resolve()}")
    print(f"vocab_size = {fast.vocab_size}")
    print(f"pad_token_id={fast.pad_token_id} bos_token_id={fast.bos_token_id} "
          f"eos_token_id={fast.eos_token_id} unk_token_id={fast.unk_token_id}")


if __name__ == "__main__":
    main()