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

ZONE_PROBE_SL_BUFFER_BINS = 1

_REMAINING_EPS = 1e-9

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
    dist = abs(entry_bin - sl_bin)
    if not (sl_min_dist_bins <= dist <= sl_max_dist_bins):
        return False
    if action_type == "BUY":
        return sl_bin < zone.lower_bin
    if action_type == "SELL":
        return sl_bin > zone.upper_bin
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
    """Nhị phân WIN/LOSS/TIMEOUT — dùng cho probe_zone_quality và
    counterfactual_outcome (CANCEL_BUY/CANCEL_SELL). KHÔNG dùng cho outcome
    thật của BUY/SELL nữa — xem partial_tp_forward_test cho việc đó."""
    risk = abs(entry_bin - sl_bin)
    if risk == 0:
        return ForwardTestResult(status=OutcomeStatus.INVALID_SETUP, r_multiple=0.0)

    rr = abs(target_bin - entry_bin) / risk

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
            return ForwardTestResult(status=OutcomeStatus.WIN, r_multiple=rr, exit_index=i)

    return ForwardTestResult(status=OutcomeStatus.TIMEOUT, r_multiple=0.0)

def partial_tp_forward_test(
    entry_bin: int,
    sl_bin: int,
    rr: int,
    future_candles: List[FutureCandle],
    direction: str,
) -> ForwardTestResult:
    """
    Outcome THẬT cho BUY/SELL — chốt lời từng phần theo `rr` mức R đều nhau
    (mỗi phần = 1/rr vị thế, chốt tại mức 1R, 2R, ..., rr*R). SL GIỮ NGUYÊN
    khoảng cách ban đầu suốt lệnh (KHÔNG dời về breakeven) — chủ ý để model
    học đặt đúng RR mong muốn: outcome đạt đỉnh khi rr khớp đúng mức giá
    thực tế đi được trước khi đảo chiều, giảm dần khi rr lệch về 2 phía
    (đặt thấp hơn -> chốt hết sớm, trung bình thấp hơn giá trị thật đi được;
    đặt cao hơn -> phần lớn vị thế còn mở khi bị đá SL).

    QUAN TRỌNG: nếu chốt hết TOÀN BỘ rr phần (WIN), r_multiple THẬT =
    (1+2+...+rr)/rr = (rr+1)/2 — KHÔNG PHẢI = rr. Đây là đánh đổi tự nhiên
    của scale-out (giảm variance, giảm expectancy đỉnh), không phải bug.

    Trong 1 nến: SL LUÔN được ưu tiên kiểm tra TRƯỚC — nếu nến đó vừa chạm
    SL vừa chạm thêm mức TP mới, KHÔNG cho chốt thêm mức TP nào của nến đó
    (giữ nguyên tinh thần "ưu tiên SL" của forward_test cũ, xử lý bảo thủ
    cho trường hợp gap).

    rr phải là int >= 1 (khớp RR_MIN..RR_MAX = 1..9 của vocab <RR_k>).
    """
    risk = abs(entry_bin - sl_bin)
    if risk == 0:
        return ForwardTestResult(status=OutcomeStatus.INVALID_SETUP, r_multiple=0.0)

    rr = int(round(rr))
    if rr < 1:
        return ForwardTestResult(status=OutcomeStatus.INVALID_SETUP, r_multiple=0.0)

    # Target bin cho từng mức k=1..rr — đơn điệu tăng dần theo k, luôn nằm
    # giữa entry và target mức rr (mức xa nhất). entry_bin đã đảm bảo nằm
    # trong [BIN_MIN, BIN_MAX] từ trước (current_price hợp lệ), nên nếu mức
    # rr (xa nhất) còn hợp lệ thì mọi mức k<rr chắc chắn cũng hợp lệ.
    level_targets: List[int] = []
    for k in range(1, rr + 1):
        t = derive_target(entry_bin, sl_bin, k, direction)
        if t is None:
            # Chỉ có thể xảy ra ở mức xa nhất (rr) do bão hoà bin — coi như
            # setup không tính được, KHÔNG suy diễn tiếp các mức thấp hơn.
            return ForwardTestResult(status=OutcomeStatus.INVALID_SETUP, r_multiple=0.0)
        level_targets.append(t)

    part_size = 1.0 / rr
    realized_r = 0.0
    remaining = 1.0
    next_level_idx = 0   # index vào level_targets, 0-based (mức k = next_level_idx+1)

    for i, (o, h, l, c) in enumerate(future_candles[:HORIZON]):
        if direction == "long":
            hit_sl = l <= sl_bin
        else:
            hit_sl = h >= sl_bin

        if hit_sl:
            realized_r += remaining * (-1.0)
            return ForwardTestResult(status=OutcomeStatus.LOSS, r_multiple=realized_r, exit_index=i)

        # Chốt MỌI mức TP mà nến này chạm được (nến dài có thể xuyên qua
        # nhiều mức 1 lúc) — không hit SL nến này nên an toàn để chốt.
        while next_level_idx < rr:
            level = next_level_idx + 1
            target = level_targets[next_level_idx]
            hit_tp_level = (h >= target) if direction == "long" else (l <= target)
            if not hit_tp_level:
                break
            realized_r += part_size * level
            remaining -= part_size
            next_level_idx += 1

        if remaining <= _REMAINING_EPS:
            return ForwardTestResult(status=OutcomeStatus.WIN, r_multiple=realized_r, exit_index=i)

    # TIMEOUT — mark-to-market phần vị thế CÒN MỞ tại Close nến cuối cùng
    # đã xét; phần đã chốt (nếu có) giữ nguyên giá trị đã khoá.
    last_close = future_candles[min(HORIZON, len(future_candles)) - 1][3] if future_candles else entry_bin
    mtm_r = (last_close - entry_bin) / risk if direction == "long" else (entry_bin - last_close) / risk
    realized_r += remaining * mtm_r
    return ForwardTestResult(status=OutcomeStatus.TIMEOUT, r_multiple=realized_r)

def probe_zone_quality(
    zone: ZoneNode,
    future_candles: List[FutureCandle],
) -> ForwardTestResult:
    if zone.direction == "support":
        entry, sl, direction = zone.upper_bin, zone.lower_bin - ZONE_PROBE_SL_BUFFER_BINS, "long"
    else:
        entry, sl, direction = zone.lower_bin, zone.upper_bin + ZONE_PROBE_SL_BUFFER_BINS, "short"

    target = derive_target(entry, sl, rr=1.0, direction=direction)
    if target is None:
        return ForwardTestResult(status=OutcomeStatus.INVALID_SETUP, r_multiple=0.0)

    return forward_test(entry, sl, target, future_candles, direction)

def _evaluate_outcome_impl(
    action: ActionNode,
    think: ThinkNode,
    future_candles: List[FutureCandle],
    sl_min_dist_bins: int,
    sl_max_dist_bins: int,
) -> Tuple[bool, Optional[ForwardTestResult], Optional[bool]]:
    """Logic gate DÙNG CHUNG cho evaluate_outcome() (reward GRPO) và
    evaluate_true_outcome() (eval P&L) — giờ 2 hàm này GIỐNG HỆT NHAU
    (không còn khác biệt train/eval). CANCEL_BUY/CANCEL_SELL KHÔNG có
    outcome (giống WAIT_BUY/WAIT_SELL/HOLD) — quyết định đã chốt: phản-thực
    (counterfactual) không mang giá trị thống kê nào, đã bỏ hẳn."""
    action_type = action.action_type

    if action_type in ("WAIT_BUY", "WAIT_SELL", "HOLD"):
        return True, None, None

    if action_type in ("BUY", "SELL"):
        if think.zone is None or action.sl is None or action.rr is None:
            return False, None, None

        sl_valid = is_sl_valid(
            action_type, think.current_price_bin, action.sl, think.zone,
            sl_min_dist_bins, sl_max_dist_bins,
        )
        direction = "long" if action_type == "BUY" else "short"
        result = partial_tp_forward_test(
            think.current_price_bin, action.sl, action.rr, future_candles, direction,
        )
        if result.status == OutcomeStatus.INVALID_SETUP:
            return False, None, sl_valid
        return True, result, sl_valid

    if action_type in ("CANCEL_BUY", "CANCEL_SELL"):
        if think.zone is None:
            return False, None, None
        return True, None, None

    return False, None, None


def evaluate_outcome(action, think, future_candles, sl_min_dist_bins=SL_MIN_DIST_BINS, sl_max_dist_bins=SL_MAX_DIST_BINS):
    return _evaluate_outcome_impl(action, think, future_candles, sl_min_dist_bins, sl_max_dist_bins)


def evaluate_true_outcome(action, think, future_candles, sl_min_dist_bins=SL_MIN_DIST_BINS, sl_max_dist_bins=SL_MAX_DIST_BINS):
    return _evaluate_outcome_impl(action, think, future_candles, sl_min_dist_bins, sl_max_dist_bins)