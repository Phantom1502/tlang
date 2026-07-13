from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class TokenType(enum.Enum):
    CHART_OPEN = "CHART_OPEN"
    CHART_CLOSE = "CHART_CLOSE"

    CANDLE_O = "CANDLE_O"
    CANDLE_H = "CANDLE_H"
    CANDLE_L = "CANDLE_L"
    CANDLE_C = "CANDLE_C"

    THINK_OPEN = "THINK_OPEN"
    THINK_CLOSE = "THINK_CLOSE"

    TREND = "TREND"
    CURRENT_PRICE = "CURRENT_PRICE"
    ZONE_SUPPORT = "ZONE_SUPPORT"
    ZONE_RESISTANCE = "ZONE_RESISTANCE"
    PRICE_IN_ZONE = "PRICE_IN_ZONE"
    GOOD_PRICE_ACTION = "GOOD_PRICE_ACTION"

    ACTION_OPEN = "ACTION_OPEN"
    ACTION_CLOSE = "ACTION_CLOSE"
    ACTION_TYPE = "ACTION_TYPE"
    SL = "SL"
    RR = "RR"

    UNKNOWN = "UNKNOWN"   # token lạ, không khớp bất kỳ pattern nào — không raise, để Parser tự xử lý
    EOF = "EOF"


@dataclass
class Token:
    type: TokenType
    value: Optional[str]   # chuỗi gốc đã match (vd "<current_price>512</current_price>")
    position: int          # vị trí ký tự bắt đầu token trong text gốc — dùng để định vị lỗi

    def __repr__(self) -> str:
        preview = self.value if self.value is None or len(self.value) <= 40 else self.value[:37] + "..."
        return f"Token({self.type.name}, {preview!r}, pos={self.position})"
