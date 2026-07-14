"""
hub.py — NGUỒN LOAD TOKENIZER DUY NHẤT cho mọi script khác (train
pretrain/SFT/GRPO, data prep, demo, reward_func nếu cần tokenize gì đó).

Luồng chuẩn từ giờ:

    1. `python -m app.tokenizer.push_to_hub --repo_id <org>/trading-llm-tokenizer`
       — build tokenizer từ source (vocab_builder.py) và push lên HF Hub.
       Chỉ chạy lại khi vocab THẬT SỰ đổi (vd đổi BIN_MAX/RR_MAX).
    2. Mọi nơi khác gọi `load_tokenizer()` — KHÔNG gọi lại
       `build_fast_tokenizer()`/`build_vocab()` trực tiếp nữa.

Lý do tập trung vào 1 nguồn (Hub) thay vì build lại mỗi nơi:
- Train pretrain/SFT/GRPO chạy trên nhiều session Colab/Kaggle khác nhau,
  mỗi session có thể cài version `tokenizers`/`transformers` hơi khác —
  build lại từ source ở mỗi máy có rủi ro (dù nhỏ) ra vocab hơi khác nhau.
  Load đúng 1 file `tokenizer.json` đã chốt trên Hub loại bỏ rủi ro này.
- Model checkpoint tie embedding theo đúng `vocab_size`/thứ tự id của
  tokenizer lúc train — tokenizer trên Hub đóng vai trò 1 ARTIFACT BẤT
  BIẾN gắn với các checkpoint đó. Đổi vocab phải là 1 hành động tường
  minh (push version mới, đổi `DEFAULT_TOKENIZER_REPO`/`revision`), không
  phải hệ quả ngẫu nhiên của việc build lại.

Fallback local build CHỈ dành cho dev/test không có mạng tới Hub (giống
sandbox hiện tại) — luôn in cảnh báo to, không nên dùng fallback này khi
train thật.
"""
from __future__ import annotations

import warnings
from typing import Optional

from transformers import PreTrainedTokenizerFast

# =====================================================================
# TODO: đổi thành repo thật sau lần `push_to_hub.py` đầu tiên. Đặt ở đây
# (1 hằng số duy nhất) để mọi script khác không phải tự nhớ tên repo.
# =====================================================================
DEFAULT_TOKENIZER_REPO = "sullivan1502/tlang"


def load_tokenizer(
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    allow_local_fallback: bool = True,
) -> PreTrainedTokenizerFast:
    """
    Load tokenizer từ HF Hub — cách DUY NHẤT các script khác nên lấy
    tokenizer từ giờ trở đi.

    Args:
        repo_id: mặc định `DEFAULT_TOKENIZER_REPO`. Truyền tay nếu cần
            test 1 phiên bản tokenizer khác (vd repo staging trước khi
            đổi default).
        revision: commit/tag cụ thể trên Hub — dùng khi cần pin đúng 1
            phiên bản tokenizer cho 1 round train cụ thể, tránh trường
            hợp ai đó push đè lên `main` giữa chừng.
        allow_local_fallback: nếu True và load từ Hub thất bại (không có
            mạng, chưa push lần nào, sai tên repo...), fallback build
            local từ `vocab_builder.py` kèm CẢNH BÁO rõ ràng — chỉ nên
            bật cho dev/test cục bộ, KHÔNG bật khi chạy train thật (để
            lỗi mạng/config sai lộ ra ngay thay vì âm thầm train bằng 1
            tokenizer build-lại-tại-chỗ có thể lệch với tokenizer đã
            dùng ở round trước).

    Returns:
        PreTrainedTokenizerFast — dùng thẳng cho `LlamaConfig(vocab_size=
        tok.vocab_size, ...)`, `DataCollatorForCoT`, `GRPOTrainer`...
    """
    repo_id = repo_id or DEFAULT_TOKENIZER_REPO

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo_id, revision=revision)
        return tok
    except Exception as e:
        if not allow_local_fallback:
            raise RuntimeError(
                f"Không load được tokenizer từ Hub (repo_id={repo_id!r}): {e}. "
                f"allow_local_fallback=False nên KHÔNG fallback — sửa repo_id/mạng/"
                f"đăng nhập HF rồi thử lại, hoặc chạy `push_to_hub.py` nếu chưa push "
                f"lần nào."
            ) from e

        warnings.warn(
            f"\n[CẢNH BÁO] Không load được tokenizer từ Hub (repo_id={repo_id!r}): {e}\n"
            f"Đang FALLBACK build local từ app.tokenizer.vocab_builder — CHỈ dùng cho "
            f"dev/test, KHÔNG dùng kết quả này để train thật (không đảm bảo khớp id "
            f"với tokenizer đã push lên Hub / đã dùng cho checkpoint trước đó).",
            stacklevel=2,
        )
        from app.tokenizer.build_tokenizer import build_fast_tokenizer
        return build_fast_tokenizer()