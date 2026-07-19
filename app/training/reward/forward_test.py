from __future__ import annotations

import math
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

# =====================================================================
# Partial credit cho case LOSS (SL chạm TRƯỚC target) — thay vì luôn phạt
# cứng r_multiple=-1.0 bất kể giá đã đi được bao xa favorable trước khi
# đảo chiều. Với 1 chart cố định (entry/SL cố định, chỉ RR đổi),
# reward(RR) có dạng "hình chuông": tăng dần khi RR còn trong tầm với
# thực tế (RR <= MFE thật -> WIN, trả đủ RR), đỉnh tại RR = MFE thật, rồi
# giảm dần khi RR vượt quá xa MFE (đặt target viển vông) — cho model
# gradient signal để tự chỉnh RR hợp lý thay vì mọi RR-quá-cao đều lãnh
# -1 y hệt nhau.
#
# LOSS_CREDIT_DECAY_RATE — placeholder, tinh chỉnh sau khi có dữ liệu
# thực nghiệm GRPO (cùng tinh thần VIOLATION_PENALTY/SEVERITY_PENALTY).
# =====================================================================
LOSS_CREDIT_DECAY_RATE = 0.35

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


# =====================================================================
# Partial credit cho case LOSS — CHỈ áp dụng nếu MFE (max favorable
# excursion, tính bằng R, đạt được TRƯỚC nến làm SL bị chạm) đã đạt
# CHUẨN SÀN >= 1R (tức nếu đã vào lệnh, ít nhất từng lãi được 1R trước
# khi bị đá ngược). KHÔNG đạt chuẩn sàn này -> phạt thẳng tay -1.0, y hệt
# hành vi cũ, không có partial credit.
#
# Khi đã đạt chuẩn sàn (mfe_r >= 1):
#   Pha 1 (r > 1):  TUYẾN TÍNH dốc -1 (giống độ dốc bên nhánh WIN),
#                    r = mfe_r - overshoot
#   Pha 2 (r <= 1):  DECAY hàm mũ, tiệm cận FLOOR = 0 (không phải -1 nữa
#                    — vì đã từng đạt chuẩn 1R, "tệ nhất" chỉ là huề vốn,
#                    không bị coi ngang hàng 1 lệnh thua sạch từ đầu).
#
# 2 pha nối liên tục tại overshoot_b = mfe_r - 1 (luôn >= 0 vì mfe_r>=1).
# =====================================================================
LOSS_CREDIT_DECAY_RATE = 0.7
LOSS_CREDIT_MIN_MFE = 1.0   # chuẩn sàn — dưới mức này không có partial credit


def _loss_partial_credit(mfe_r: float, rr: float, decay_rate: float = LOSS_CREDIT_DECAY_RATE) -> float:
    """
    - mfe_r < LOSS_CREDIT_MIN_MFE (kể cả mfe_r <= 0 — LOSS "sạch" hoặc
      chưa từng lãi đủ 1R) -> -1.0, phạt thẳng tay, KHÔNG có partial
      credit — giữ tương thích ngược với case SL chạm ngay nến đầu.
    - mfe_r >= LOSS_CREDIT_MIN_MFE -> pha 1 tuyến tính (dốc -1, khi giá
      trị còn > 1), pha 2 decay hàm mũ (khi giá trị đã tụt xuống <= 1),
      tiệm cận FLOOR = 0 (không phải -1).
    """
    if mfe_r < LOSS_CREDIT_MIN_MFE:
        return -1.0

    overshoot = max(0.0, rr - mfe_r)
    overshoot_b = mfe_r - 1.0   # overshoot tại đó pha 1 vừa chạm r=1 (>=0 vì mfe_r>=1)

    if overshoot <= overshoot_b:
        return mfe_r - overshoot   # pha 1 — tuyến tính, dốc -1

    # Pha 2 — decay hàm mũ từ r=1 tại điểm nối, tiệm cận floor=0.
    return max(0.0, math.exp(-decay_rate * (overshoot - overshoot_b)))


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

    rr = abs(target_bin - entry_bin) / risk
    mfe_r = 0.0   # max favorable excursion (R) — chỉ cộng dồn từ nến ĐÃ QUA AN TOÀN

    for i, (o, h, l, c) in enumerate(future_candles[:HORIZON]):
        if direction == "long":
            hit_sl = l <= sl_bin
            hit_tp = h >= target_bin
        else:
            hit_sl = h >= sl_bin
            hit_tp = l <= target_bin

        if hit_sl:
            r_multiple = _loss_partial_credit(mfe_r, rr)
            return ForwardTestResult(status=OutcomeStatus.LOSS, r_multiple=r_multiple, exit_index=i)
        if hit_tp:
            r_multiple = abs(target_bin - entry_bin) / risk
            return ForwardTestResult(status=OutcomeStatus.WIN, r_multiple=r_multiple, exit_index=i)

        # Nến này KHÔNG chạm SL/TP -> an toàn để cộng dồn MFE. Nến gây
        # LOSS/WIN (2 nhánh trên) KHÔNG được cộng — nhất quán nguyên tắc
        # "ưu tiên SL" khi gap 1 nến chạm cả 2 mức (không suy diễn favorable
        # move đã xảy ra trước trong chính nến đó).
        candle_r = (h - entry_bin) / risk if direction == "long" else (entry_bin - l) / risk
        mfe_r = max(mfe_r, candle_r)

    return ForwardTestResult(status=OutcomeStatus.TIMEOUT, r_multiple=0.0)

# =====================================================================
# BẢN "OUTCOME THẬT" — dùng cho scripts/eval_val.py (đo P&L thật để báo
# cáo win_rate/avg_R), KHÔNG dùng cho reward GRPO (forward_test() ở trên
# vẫn giữ nguyên, có partial-credit/shaping — đó là tín hiệu train, không
# phải P&L thật).
#
# Khác biệt duy nhất so với forward_test():
#   - LOSS (chạm SL trước): r_multiple = -1.0 CỐ ĐỊNH, không partial credit.
#   - WIN  (chạm target trước): giống hệt — r_multiple = +rr thật.
#   - TIMEOUT (hết horizon không chạm gì): KHÔNG gán cứng 0 — mark-to-market
#     tại giá Close của nến CUỐI CÙNG trong horizon đã xét, đúng P&L nếu
#     phải đóng lệnh tại thời điểm đó.
# =====================================================================
def true_forward_test(
    entry_bin: int,
    sl_bin: int,
    target_bin: int,
    future_candles: List[FutureCandle],
    direction: str,
) -> ForwardTestResult:
    risk = abs(entry_bin - sl_bin)
    if risk == 0:
        return ForwardTestResult(status=OutcomeStatus.INVALID_SETUP, r_multiple=0.0)

    rr = abs(target_bin - entry_bin) / risk
    last_close: Optional[int] = None

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

        last_close = c

    if last_close is None:
        # future_candles rỗng — không có gì để mark-to-market.
        return ForwardTestResult(status=OutcomeStatus.TIMEOUT, r_multiple=0.0)

    if direction == "long":
        r_multiple = (last_close - entry_bin) / risk
    else:
        r_multiple = (entry_bin - last_close) / risk

    return ForwardTestResult(status=OutcomeStatus.TIMEOUT, r_multiple=r_multiple)

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

    # Chặn ở 0 TRƯỚC khi đảo dấu nếu status=LOSS: sau khi forward_test có
    # partial credit theo MFE, 1 lệnh LOSS vẫn có thể có r_multiple dương
    # nhẹ (giá đi gần target rồi đảo chiều). CANCEL một lệnh RỐT CUỘC THUA
    # luôn là quyết định đúng — không được phép bị phạt chỉ vì nó "suýt
    # thắng". KHÔNG áp rule này cho WIN (CANCEL 1 lệnh lẽ ra thắng vẫn phải
    # bị phạt đúng như cũ).
    raw_r = result.r_multiple
    if result.status == OutcomeStatus.LOSS:
        raw_r = min(raw_r, 0.0)

    return ForwardTestResult(
        status=result.status,
        r_multiple=-raw_r,
        exit_index=result.exit_index,
    )


def evaluate_outcome(
    action: ActionNode,
    think: ThinkNode,
    future_candles: List[FutureCandle],
    sl_min_dist_bins: int = SL_MIN_DIST_BINS,
    sl_max_dist_bins: int = SL_MAX_DIST_BINS,
) -> Tuple[bool, Optional[ForwardTestResult], Optional[bool]]:
    """
    Trả về (extra_valid, forward_result, sl_valid).

    extra_valid: setup có TÍNH ĐƯỢC hay không (thiếu zone/sl/rr, hoặc target
    bị bão hoà bin [0,1023]) — lỗi cứng thật sự, không có gì để forward-test.
    KHÔNG còn phụ thuộc is_sl_valid — SL sai khoảng cách/sai phía zone vẫn
    tính target bình thường (chỉ là phép cộng/trừ bin), vẫn chạy forward_test
    thật để lấy tín hiệu outcome thay vì vứt bỏ toàn bộ (bug đã fix — trước
    đây SL invalid làm mất hết outcome, tạo bias né BUY/SELL).

    sl_valid: chỉ có ý nghĩa với BUY/SELL — None cho action khác. Dùng ở
    reward_func.py để cộng/trừ ĐỐI XỨNG (giống zone_quality_bonus/penalty),
    KHÔNG dùng để gate outcome.
    """
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
        target = derive_target(think.current_price_bin, action.sl, action.rr, direction)
        if target is None:
            return False, None, sl_valid   # bin bão hoà — lỗi cứng thật, độc lập với is_sl_valid
        result = forward_test(think.current_price_bin, action.sl, target, future_candles, direction)
        return True, result, sl_valid

    if action_type in ("CANCEL_BUY", "CANCEL_SELL"):
        if think.zone is None:
            return False, None, None
        result = counterfactual_outcome(action_type, think.zone, think.current_price_bin, future_candles)
        return True, result, None

    return False, None, None

def _evaluate_outcome_impl(
    action: ActionNode,
    think: ThinkNode,
    future_candles: List[FutureCandle],
    sl_min_dist_bins: int,
    sl_max_dist_bins: int,
    forward_fn,
    counterfactual_fn=None,   # None -> CANCEL_BUY/CANCEL_SELL coi như không có outcome (giống WAIT/HOLD)
) -> Tuple[bool, Optional[ForwardTestResult], Optional[bool]]:
    """Logic gate DÙNG CHUNG cho evaluate_outcome() (reward, có shaping) và
    evaluate_true_outcome() (eval, P&L thật) — chỉ khác forward_fn/
    counterfactual_fn truyền vào. Sửa gate ở đây, cả 2 nơi ăn theo."""
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
        target = derive_target(think.current_price_bin, action.sl, action.rr, direction)
        if target is None:
            return False, None, sl_valid
        result = forward_fn(think.current_price_bin, action.sl, target, future_candles, direction)
        return True, result, sl_valid

    if action_type in ("CANCEL_BUY", "CANCEL_SELL"):
        if think.zone is None:
            return False, None, None
        if counterfactual_fn is None:
            # Eval thật: không suy đoán phản-thực, chỉ cần biết setup hợp lệ
            # để đếm vào thống kê (well-form/semantic pass) — không có outcome.
            return True, None, None
        result = counterfactual_fn(action_type, think.zone, think.current_price_bin, future_candles)
        return True, result, None

    return False, None, None

def evaluate_true_outcome(
    action: ActionNode,
    think: ThinkNode,
    future_candles: List[FutureCandle],
    sl_min_dist_bins: int = SL_MIN_DIST_BINS,
    sl_max_dist_bins: int = SL_MAX_DIST_BINS,
) -> Tuple[bool, Optional[ForwardTestResult], Optional[bool]]:
    """Bản DÙNG CHO EVAL (scripts/eval_val.py) — P&L THẬT cho BUY/SELL
    (LOSS luôn -1.0, WIN luôn +rr, TIMEOUT mark-to-market tại Close nến
    cuối horizon — xem true_forward_test). CANCEL_BUY/CANCEL_SELL KHÔNG
    có outcome (giống WAIT/HOLD) — phản-thực không có ý nghĩa cho báo cáo
    P&L, chỉ cần thống kê tần suất/tỷ lệ action này qua summarize()."""
    return _evaluate_outcome_impl(
        action, think, future_candles, sl_min_dist_bins, sl_max_dist_bins,
        forward_fn=true_forward_test, counterfactual_fn=None,
    )