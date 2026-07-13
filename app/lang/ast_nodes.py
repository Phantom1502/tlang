from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CandleNode:
    o: int
    h: int
    l: int
    c: int


@dataclass
class ChartNode:
    candles: List[CandleNode] = field(default_factory=list)


@dataclass
class ZoneNode:
    direction: str        # "support" | "resistance"
    lower_bin: int
    upper_bin: int


@dataclass
class ThinkNode:
    trend: Optional[str] = None                  # "UP" | "DOWN" | "RANGE"
    current_price_bin: Optional[int] = None      # BẮT BUỘC theo spec — luôn phải có mặt
    zone: Optional[ZoneNode] = None
    price_in_zone: bool = False
    good_price_action: bool = False


@dataclass
class ActionNode:
    action_type: Optional[str] = None  # BUY | SELL | CANCEL_BUY | CANCEL_SELL | WAIT_BUY | WAIT_SELL | HOLD
    sl: Optional[int] = None
    rr: Optional[int] = None           # risk luôn chuẩn hoá = 1, rr là reward-multiple duy nhất


@dataclass
class ProgramNode:
    chart: Optional[ChartNode] = None
    think: Optional[ThinkNode] = None
    action: Optional[ActionNode] = None
