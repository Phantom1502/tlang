from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

# =====================================================================
# Hằng số vocab dùng chung cho cả Lexer/Parser (nhánh model) — ĐÂY LÀ
# NGUỒN SỰ THẬT DUY NHẤT cho quy ước zero-pad/phạm vi bin. Nhánh
# tokenizer nên import trực tiếp từ đây thay vì tự định nghĩa lại, để
# tránh lệch giữa 2 nơi (bài học từ lần review tokenizer trước).
# =====================================================================
BIN_MIN = 0
BIN_MAX = 1023
DIGIT_PAD = 4        # zero-pad 4 chữ số cho mọi field digit-decompose (0000–1023)
RR_MIN = 1
RR_MAX = 9


class TokenType(enum.Enum):
    CHART_OPEN = "CHART_OPEN"
    CHART_CLOSE = "CHART_CLOSE"

    # Chart OHLC GIỮ NGUYÊN dạng atomic <O_543> — không digit-decompose,
    # vì model chỉ cần ĐỌC chart, không cần làm phép tính số học trực
    # tiếp trên OHLC (khác với current_price/zone/SL).
    CANDLE_O = "CANDLE_O"
    CANDLE_H = "CANDLE_H"
    CANDLE_L = "CANDLE_L"
    CANDLE_C = "CANDLE_C"

    THINK_OPEN = "THINK_OPEN"
    THINK_CLOSE = "THINK_CLOSE"

    TREND = "TREND"   # vẫn atomic "<trend>UP</trend>" — chỉ 3 giá trị, enum gộp là hợp lý

    # current_price/zone: tag mở/đóng tách riêng, giá trị số ở giữa là
    # chuỗi DIGIT rời (digit-decompose, zero-pad DIGIT_PAD chữ số).
    CURRENT_PRICE_OPEN = "CURRENT_PRICE_OPEN"
    CURRENT_PRICE_CLOSE = "CURRENT_PRICE_CLOSE"
    ZONE_SUPPORT_OPEN = "ZONE_SUPPORT_OPEN"
    ZONE_SUPPORT_CLOSE = "ZONE_SUPPORT_CLOSE"
    ZONE_RESISTANCE_OPEN = "ZONE_RESISTANCE_OPEN"
    ZONE_RESISTANCE_CLOSE = "ZONE_RESISTANCE_CLOSE"
    PRICE_IN_ZONE = "PRICE_IN_ZONE"
    GOOD_PRICE_ACTION = "GOOD_PRICE_ACTION"

    ACTION_OPEN = "ACTION_OPEN"
    ACTION_CLOSE = "ACTION_CLOSE"
    ACTION_TYPE = "ACTION_TYPE"

    SL_LABEL = "SL_LABEL"   # literal "SL:" — chỉ là nhãn, giá trị theo sau là DIGIT rời
    RR = "RR"               # bracket-enum ATOMIC "<RR_1>".."<RR_9>" — KHÔNG digit-decompose,
                             # vì range quá nhỏ (1-9), enum token tự mô tả rõ nghĩa hơn 1 con số trần

    DIGIT = "DIGIT"     # 1 chữ số '0'-'9' — dùng chung cho current_price/zone/SL
    COLON = "COLON"     # ':' phân cách 2 cạnh của zone

    UNKNOWN = "UNKNOWN"   # token lạ, không khớp bất kỳ pattern nào — không raise, để Parser tự xử lý
    EOF = "EOF"


@dataclass
class Token:
    type: TokenType
    value: Optional[str]   # chuỗi gốc đã match (vd "<current_price>", "5", "<RR_9>")
    position: int          # vị trí ký tự bắt đầu token trong text gốc — dùng để định vị lỗi

    def __repr__(self) -> str:
        preview = self.value if self.value is None or len(self.value) <= 40 else self.value[:37] + "..."
        return f"Token({self.type.name}, {preview!r}, pos={self.position})"