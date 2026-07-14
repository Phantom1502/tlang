"""
vocab_builder.py — Xây dựng vocab CỐ ĐỊNH, ĐÓNG (closed, enumerable) cho
tokenizer của Trading Reasoning LLM.

NGUYÊN TẮC (khớp docs/spec_trading_llm_v0.2.md mục 3 và
docs/train_pipeline_v0.1.md mục 2): ngôn ngữ think/action là 1 grammar
hình thức đã liệt kê tường minh toàn bộ token hợp lệ — KHÔNG có OOV thật
sự, nên KHÔNG dùng BPE/Unigram/WordPiece (những thuật toán học merge từ
tần suất thống kê trên corpus mở). Toàn bộ vocab ở đây được *tính ra*
từ hằng số, không "học" từ dữ liệu.

Import hằng số trực tiếp từ app.lang.tokens — ĐÂY LÀ NGUỒN SỰ THẬT DUY
NHẤT cho BIN_MIN/BIN_MAX/DIGIT_PAD/RR_MIN/RR_MAX, dùng chung với nhánh
Lexer/Parser. Không tự định nghĩa lại range ở đây (bài học đã ghi trong
spec mục 3: "1 bug thực tế: bản tokenizer đầu tiên tự vá lành cấu trúc
hỏng...").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.lang.tokens import BIN_MAX, BIN_MIN, RR_MAX, RR_MIN

# =====================================================================
# Special tokens — id cố định để khớp LlamaConfig trong
# docs/train_pipeline_v0.1.md mục 1.1 (pad_token_id=3, bos_token_id=1,
# eos_token_id=2). unk lấy id=0 (quy ước phổ biến, không có ràng buộc gì
# từ LlamaConfig).
# =====================================================================
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

SPECIAL_TOKENS_IN_ID_ORDER: List[str] = [UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]  # id 0,1,2,3

ACTION_TYPES: Tuple[str, ...] = (
    "BUY", "SELL", "CANCEL_BUY", "CANCEL_SELL", "WAIT_BUY", "WAIT_SELL", "HOLD",
)

TREND_VALUES: Tuple[str, ...] = ("UP", "DOWN", "RANGE")

CANDLE_PREFIXES: Tuple[str, ...] = ("O", "H", "L", "C")  # <O_x> <H_x> <L_x> <C_x>


@dataclass(frozen=True)
class VocabGroup:
    """1 nhóm token trong bảng vocab — chỉ để in báo cáo `describe()`, không ảnh hưởng id."""
    name: str
    tokens: List[str]


def _build_groups() -> List[VocabGroup]:
    """
    Liệt kê tường minh mọi nhóm token, ĐÚNG THỨ TỰ sẽ được gán id (sau
    4 special token). Thứ tự này là 1 phần của "vocab contract" — đổi
    thứ tự đồng nghĩa đổi id, PHẢI rebuild lại mọi tokenizer.json/model
    embedding đã train nếu đổi. Xem thêm docs/tokenizer_v0.1.md mục 3.
    """
    groups: List[VocabGroup] = []

    # 1) Structural tags — chart/think/action wrapper
    groups.append(VocabGroup("structural_tags", [
        "<chart>", "</chart>",
        "<think>", "</think>",
        "<action>", "</action>",
    ]))

    # 2) Chart OHLC — ATOMIC, mỗi field 1 dải riêng [BIN_MIN, BIN_MAX].
    #    Model chỉ cần ĐỌC chart, không làm số học trực tiếp trên OHLC
    #    (spec mục 3) -> gộp tag+giá trị thành 1 token, không digit-decompose.
    for prefix in CANDLE_PREFIXES:
        groups.append(VocabGroup(
            f"candle_{prefix}",
            [f"<{prefix}_{v}>" for v in range(BIN_MIN, BIN_MAX + 1)],
        ))

    # 3) Trend — fused atomic (chỉ 3 giá trị, enum gộp hợp lý hơn tag+digit)
    groups.append(VocabGroup("trend", [f"<trend>{v}</trend>" for v in TREND_VALUES]))

    # 4) Tag mở/đóng cho field digit-decompose (current_price/zone) + SL label
    groups.append(VocabGroup("digit_field_tags", [
        "<current_price>", "</current_price>",
        "<zone_support>", "</zone_support>",
        "<zone_resistance>", "</zone_resistance>",
        "SL:",
    ]))

    # 5) Digit dùng CHUNG cho mọi field digit-decompose (current_price / zone
    #    lower / zone upper / SL) — cùng 1 không gian embedding 10 token, để
    #    model học so sánh/số học giữa các field này dễ hơn (spec mục 3).
    groups.append(VocabGroup("digit", [str(d) for d in range(10)]))

    # 6) Colon — phân cách 2 cạnh zone "lower:upper"
    groups.append(VocabGroup("colon", [":"]))

    # 7) Boolean flag tokens — chỉ có mặt hay không, không có "close tag" riêng
    groups.append(VocabGroup("flags", ["<price_in_zone>", "<good_price_action>"]))

    # 8) Action type — atomic enum
    groups.append(VocabGroup("action_type", list(ACTION_TYPES)))

    # 9) RR — bracket-enum ATOMIC (range 1-9 quá nhỏ để digit-decompose)
    groups.append(VocabGroup("rr", [f"<RR_{v}>" for v in range(RR_MIN, RR_MAX + 1)]))

    return groups


def build_vocab() -> Dict[str, int]:
    """
    Trả về dict token_str -> id, id 0..3 = special token cố định, phần
    còn lại gán tuần tự theo thứ tự nhóm trong `_build_groups()`.

    Idempotent & deterministic: gọi lại nhiều lần luôn ra đúng 1 kết quả
    (không phụ thuộc hash-order hay random) — điều kiện bắt buộc để
    tokenizer.json build lại từ source vẫn tương thích checkpoint cũ.
    """
    vocab: Dict[str, int] = {}
    next_id = 0

    for tok in SPECIAL_TOKENS_IN_ID_ORDER:
        vocab[tok] = next_id
        next_id += 1

    for group in _build_groups():
        for tok in group.tokens:
            if tok in vocab:
                raise ValueError(f"Token trùng lặp giữa các nhóm: {tok!r} (nhóm {group.name})")
            vocab[tok] = next_id
            next_id += 1

    return vocab


def describe_vocab() -> str:
    """In bảng thống kê số token mỗi nhóm — dùng để đối chiếu với bảng
    vocab trong docs/train_pipeline_v0.1.md mục 2.2 (kỳ vọng ~4146 token)."""
    lines = [f"{'special':<20} count={len(SPECIAL_TOKENS_IN_ID_ORDER)}"]
    total = len(SPECIAL_TOKENS_IN_ID_ORDER)
    for group in _build_groups():
        lines.append(f"{group.name:<20} count={len(group.tokens)}")
        total += len(group.tokens)
    lines.append(f"{'TOTAL':<20} count={total}")
    return "\n".join(lines)


if __name__ == "__main__":
    vocab = build_vocab()
    print(describe_vocab())
    print(f"\nlen(build_vocab()) = {len(vocab)}")
    assert len(vocab) == len(set(vocab.values())), "id bị trùng — bug trong build_vocab()"