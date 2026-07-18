from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from app.lang.ast_nodes import ActionNode, ThinkNode, ZoneNode

SL_MIN_DIST_BINS = 5
SL_MAX_DIST_BINS = 10

BIN_MIN = 0
BIN_MAX = 1023

HORIZON = 50

FutureCandle = Tuple[int, int, int, int]


class OutcomeStatus(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    TIMEOUT = "TIMEOUT"
    INVALID_SETUP = "INVALID_SETUP"


@dataclass
class ForwardTestResult:
    status: OutcomeStatus
    r_multiple: float
    exit_index: Optional[int] = None


def is_sl_valid(
    action_type: str, entry_bin: int, sl_bin: int, zone: ZoneNode,
    sl_min_dist_bins: int = SL_MIN_DIST_BINS,
    sl_max_dist_bins: int = SL_MAX_DIST_BINS,
) -> bool:
    """Default = module constant (5/10) — dùng cho generator.py/demo không
    đổi gì. GRPO (reward_func.py) truyền tường minh từ RoundConfig hiện tại."""
    dist = abs(entry_bin - sl_bin)
    if not (sl_min_dist_bins <= dist <= sl_max_dist_bins):
        return False
    if action_type == "BUY":
        return sl_bin < zone.lower_bin   # SL phải nằm dưới đáy zone_support
    if action_type == "SELL":
        return sl_bin > zone.upper_bin   # SL phải nằm trên đỉnh zone_resistance
    return False


def derive_target(entry_bin: int, sl_bin: int, rr: float, direction: str) -> Optional[int]:
    if direction == "long":
        target = entry_bin + rr * (entry_bin - sl_bin)
    else:
        target = entry_bin - rr * (sl_bin - entry_bin)

    target = round(target)
    if not (BIN_MIN <= target <= BIN_MAX):
        return None
    return target


def forward_test(
    entry_bin: int,
    sl_bin: int,
    target_bin: int,
    future_candles: List[FutureCandle],
    direction: str,
) -> ForwardTestResult:
    risk = abs(entry_bin - sl_bin)
    if risk == 0:
        return ForwardTestResult(status=OutcomeStatus.INVALID_SETUP, r_multiple=0.0)

    for i, (o, h, l, c) in enumerate(future_candles[:HORIZON]):
        if direction == "long":
            hit_sl = l <= sl_bin
            hit_tp = h >= target_bin
        else:
            hit_sl = h >= sl_bin
            hit_tp = l <= target_bin

        if hit_sl:
            return ForwardTestResult(status=OutcomeStatus.LOSS, r_multiple=-1.0, exit_index=i)
        if hit_tp:
            r_multiple = abs(target_bin - entry_bin) / risk
            return ForwardTestResult(status=OutcomeStatus.WIN, r_multiple=r_multiple, exit_index=i)

    return ForwardTestResult(status=OutcomeStatus.TIMEOUT, r_multiple=0.0)


def counterfactual_outcome(
    action_type: str,
    zone: ZoneNode,
    current_price_bin: int,
    future_candles: List[FutureCandle],
) -> ForwardTestResult:
    entry = current_price_bin

    if action_type == "CANCEL_BUY":
        sl = zone.lower_bin - 1
        direction = "long"
    elif action_type == "CANCEL_SELL":
        sl = zone.upper_bin + 1
        direction = "short"
    else:
        raise ValueError(f"counterfactual_outcome chỉ áp dụng cho CANCEL_BUY/CANCEL_SELL, nhận {action_type!r}")

    target = derive_target(entry, sl, rr=1.0, direction=direction)
    if target is None:
        return ForwardTestResult(status=OutcomeStatus.INVALID_SETUP, r_multiple=0.0)

    result = forward_test(entry, sl, target, future_candles, direction)
    return ForwardTestResult(
        status=result.status,
        r_multiple=-result.r_multiple,
        exit_index=result.exit_index,
    )


def evaluate_outcome(
    action: ActionNode,
    think: ThinkNode,
    future_candles: List[FutureCandle],
    sl_min_dist_bins: int = SL_MIN_DIST_BINS,
    sl_max_dist_bins: int = SL_MAX_DIST_BINS,
) -> Tuple[bool, Optional[ForwardTestResult]]:
    """sl_min_dist_bins/sl_max_dist_bins: default = module constant (5/10),
    dùng cho generator.py/demo không đổi gì. GRPO (reward_func.py) truyền
    tường minh từ RoundConfig hiện tại (app/training/reward/round_config.py)."""
    action_type = action.action_type

    if action_type in ("WAIT_BUY", "WAIT_SELL", "HOLD"):
        return True, None

    if action_type in ("BUY", "SELL"):
        if think.zone is None or action.sl is None or action.rr is None:
            return False, None
        if not is_sl_valid(
            action_type, think.current_price_bin, action.sl, think.zone,
            sl_min_dist_bins, sl_max_dist_bins,
        ):
            return False, None
        direction = "long" if action_type == "BUY" else "short"
        target = derive_target(think.current_price_bin, action.sl, action.rr, direction)
        if target is None:
            return False, None
        result = forward_test(think.current_price_bin, action.sl, target, future_candles, direction)
        return True, result

    if action_type in ("CANCEL_BUY", "CANCEL_SELL"):
        if think.zone is None:
            return False, None
        result = counterfactual_outcome(action_type, think.zone, think.current_price_bin, future_candles)
        return True, result

    return False, None