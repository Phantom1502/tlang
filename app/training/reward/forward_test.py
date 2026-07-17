from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from app.lang.ast_nodes import ActionNode, ThinkNode, ZoneNode

# =====================================================================
# Config — số bin CỐ ĐỊNH set tay ngoài (KHÔNG derive theo ATR), nhất
# quán với cách xử lý zone-width. Điều chỉnh trực tiếp 2 hằng số này
# theo dữ liệu/symbol đang dùng.
# =====================================================================
SL_MIN_DIST_BINS = 5
SL_MAX_DIST_BINS = 10

BIN_MIN = 0
BIN_MAX = 1023

HORIZON = 50   # đi hết toàn bộ future_bins, không dừng sớm (đã chốt trong spec)

FutureCandle = Tuple[int, int, int, int]   # (o, h, l, c) — cùng hệ bin với input, cùng anchor


class OutcomeStatus(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    TIMEOUT = "TIMEOUT"
    INVALID_SETUP = "INVALID_SETUP"   # SL sai khoảng cách/phía zone, hoặc target bị bão hoà bin


@dataclass
class ForwardTestResult:
    status: OutcomeStatus
    r_multiple: float                  # dương=thắng theo R đạt được, âm=thua (-1.0), 0=timeout/invalid
    exit_index: Optional[int] = None   # nến thứ mấy trong future_candles gây thoát lệnh


# =====================================================================
# Ràng buộc SL (khoảng cách cố định + đúng phía zone) — về bản chất là
# 1 semantic check gắn với cơ chế thực thi lệnh, tính riêng ở đây vì
# cần entry/SL/zone cùng lúc. Caller (unified reward func) cần cộng kết
# quả này vào gate 2 (semantic), KHÔNG coi thất bại ở đây là "outcome=0".
# =====================================================================
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
    """
    target = entry + RR × (entry - SL)  (long)
    target = entry - RR × (SL - entry)  (short)
    Risk luôn chuẩn hoá = 1 theo định nghĩa RR (xem spec mục 2.1/6.1).

    Trả None nếu target vượt ra ngoài [0, 1023] — bin bị bão hoà, coi là
    setup không hợp lệ (không tính outcome, phạt như 1 case invalid).
    """
    if direction == "long":
        target = entry_bin + rr * (entry_bin - sl_bin)
    else:  # short
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
    """
    Chạy forward-test trên toàn bộ future_candles (horizon = hết 50 nến,
    không dừng sớm). Toàn bộ tính bằng bin nguyên — không cần decode giá
    thật/ATR, vì future_candles đã được encode cùng anchor với input chart.

    Case biên:
    - SL và TP cùng chạm trong 1 nến (gap) -> ưu tiên SL (conservative).
    - Hết horizon mà chưa chạm SL/TP (timeout) -> reward trung tính = 0
      (chủ ý không phạt, để tránh đẩy model về xu hướng đứng ngoài; model
      tự học cách chọn RR hợp lý hơn qua outcome, không bị ép bằng phạt cứng).
    """
    risk = abs(entry_bin - sl_bin)
    if risk == 0:
        # Phòng vệ — không nên xảy ra nếu is_sl_valid đã chặn SL trùng entry.
        return ForwardTestResult(status=OutcomeStatus.INVALID_SETUP, r_multiple=0.0)

    for i, (o, h, l, c) in enumerate(future_candles[:HORIZON]):
        if direction == "long":
            hit_sl = l <= sl_bin
            hit_tp = h >= target_bin
        else:  # short
            hit_sl = h >= sl_bin
            hit_tp = l <= target_bin

        if hit_sl:
            # Ưu tiên SL bất kể hit_tp có true hay không (gap qua cả 2 mức trong cùng 1 nến).
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
    """
    CANCEL_BUY / CANCEL_SELL: SL/RR KHÔNG lấy từ model output (bị cấm ở
    well-form) — derive tự động từ zone + buffer cố định 1 bin + RR=1.

    Dùng chung hàm forward_test với BUY/SELL thật (cùng logic gap-SL-
    priority, cùng xử lý timeout), rồi ĐẢO DẤU r_multiple: CANCEL đúng
    khi lẽ ra sẽ THUA (r_multiple gốc âm -> reward dương), CANCEL sai khi
    lẽ ra sẽ THẮNG (r_multiple gốc dương -> reward âm). Timeout giữ 0.
    """
    entry = current_price_bin

    if action_type == "CANCEL_BUY":
        sl = zone.lower_bin - 1   # buffer 1 bin dưới mép zone_support
        direction = "long"
    elif action_type == "CANCEL_SELL":
        sl = zone.upper_bin + 1   # buffer 1 bin trên mép zone_resistance
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
) -> Tuple[bool, Optional[ForwardTestResult]]:
    """
    Entry point tổng hợp — dùng trực tiếp trong unified_reward_func (gate 3).

    Trả về (extra_semantic_valid, forward_test_result):
    - extra_semantic_valid=False: vi phạm ràng buộc SL (khoảng cách/phía
      zone) hoặc target bị bão hoà bin. Đây KHÔNG phải "outcome tệ", mà
      là 1 dạng semantic violation bổ sung — caller cần cộng vào gate 2
      (semantic), KHÔNG tính outcome=0 đơn thuần cho case này.
    - forward_test_result=None cho WAIT_BUY/WAIT_SELL/HOLD (không có gì
      để forward-test).
    """
    action_type = action.action_type

    if action_type in ("WAIT_BUY", "WAIT_SELL", "HOLD"):
        return True, None

    if action_type in ("BUY", "SELL"):
        if think.zone is None or action.sl is None or action.rr is None:
            return False, None
        if not is_sl_valid(action_type, think.current_price_bin, action.sl, think.zone):
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

    return False, None   # action_type lạ/None — không nên xảy ra nếu well-form đã pass