from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from app.lang.ast_nodes import ActionNode, ThinkNode, ZoneNode
from app.lang.parser import Parser
from app.lang.semantic import SemanticChecker
from app.lang.tokens import BIN_MAX, BIN_MIN, DIGIT_PAD, RR_MAX, RR_MIN
from app.training.reward.forward_test import SL_MAX_DIST_BINS, SL_MIN_DIST_BINS, derive_target, evaluate_outcome

Candle = Tuple[int, int, int, int]   # (o, h, l, c)

# =====================================================================
# Config — PLACEHOLDER. Ngưỡng zone-width là "số bin cố định do người
# tạo dữ liệu set trực tiếp bên ngoài" (đã chốt trong spec mục 5.2/7.1)
# — bạn tự có rule riêng, đây chỉ là giá trị mặc định để generator này
# chạy độc lập được. Đổi 2 hằng số dưới đây theo đúng ngưỡng bạn dùng.
# =====================================================================
ZONE_WIDTH_MIN_BINS = 5
ZONE_WIDTH_MAX_BINS = 20

LAST_N_CANDLES_TOUCH = 5   # khớp SemanticChecker.LAST_N_CANDLES_TOUCH


@dataclass
class GeneratedSample:
    prompt: str        # "<chart>...</chart>"
    completion: str     # "<think>...</think><action>...</action>"
    leaf_recipe: str     # tên leaf-path đã dùng — để kiểm tra phân phối (mục A, lượt trước)


# =====================================================================
# Cây leaf-path hợp lệ — LIỆT KÊ TƯỜNG MINH trước khi sample, đúng
# nguyên tắc "sample uniform trên toàn bộ leaf-path hợp lệ, không sample
# từng field độc lập rồi lọc bỏ invalid" (spec mục 7.2).
#
# Mỗi leaf = (trend, zone_side, zone_case, action_type)
#   zone_side: "support" | "resistance" | None (RANGE không zone)
#   zone_case: "CONTAINS" | "TOUCH" | "NOTOUCH" | None
#     CONTAINS -> current_price nằm trong zone -> price_in_zone bắt buộc True
#     TOUCH    -> zone nằm ngoài current_price nhưng 1 trong 5 nến cuối chạm -> price_in_zone=True
#     NOTOUCH  -> zone nằm ngoài, không nến nào chạm -> price_in_zone=False
# =====================================================================
LEAF_RECIPES: List[Tuple[str, Optional[str], Optional[str], str]] = [
    # trend=UP — chỉ zone_support, action phía buy
    ("UP", "support", "CONTAINS", "BUY"),
    ("UP", "support", "CONTAINS", "CANCEL_BUY"),
    ("UP", "support", "TOUCH", "BUY"),
    ("UP", "support", "TOUCH", "CANCEL_BUY"),
    ("UP", "support", "NOTOUCH", "WAIT_BUY"),
    # trend=DOWN — chỉ zone_resistance, action phía sell
    ("DOWN", "resistance", "CONTAINS", "SELL"),
    ("DOWN", "resistance", "CONTAINS", "CANCEL_SELL"),
    ("DOWN", "resistance", "TOUCH", "SELL"),
    ("DOWN", "resistance", "TOUCH", "CANCEL_SELL"),
    ("DOWN", "resistance", "NOTOUCH", "WAIT_SELL"),
    # trend=RANGE — có thể có zone_support HOẶC zone_resistance HOẶC không zone
    ("RANGE", "support", "CONTAINS", "BUY"),
    ("RANGE", "support", "CONTAINS", "CANCEL_BUY"),
    ("RANGE", "support", "TOUCH", "BUY"),
    ("RANGE", "support", "TOUCH", "CANCEL_BUY"),
    ("RANGE", "support", "NOTOUCH", "WAIT_BUY"),
    ("RANGE", "resistance", "CONTAINS", "SELL"),
    ("RANGE", "resistance", "CONTAINS", "CANCEL_SELL"),
    ("RANGE", "resistance", "TOUCH", "SELL"),
    ("RANGE", "resistance", "TOUCH", "CANCEL_SELL"),
    ("RANGE", "resistance", "NOTOUCH", "WAIT_SELL"),
    ("RANGE", None, None, "HOLD"),
]


def _digits(n: int, pad: int = DIGIT_PAD) -> List[str]:
    return list(str(n).zfill(pad))


# ---------------------------------------------------------------------
# Construct zone THEO ĐÚNG zone_case — deterministic by construction,
# trả None nếu KHÔNG THỂ dựng được zone thỏa mãn case này trên chart cụ
# thể đang xét (vd không nến nào trong 5 nến cuối đủ điều kiện cho TOUCH)
# — caller sẽ thử lại với random state khác hoặc leaf khác, KHÔNG dùng
# reject-sampling trên toàn bộ leaf-path, chỉ retry ở tầng hiện thực số.
# ---------------------------------------------------------------------
def _pick_zone(
    rng: random.Random,
    side: str,
    case: str,
    current_price: int,
    last5: Sequence[Candle],
) -> Optional[ZoneNode]:
    width = rng.randint(ZONE_WIDTH_MIN_BINS, ZONE_WIDTH_MAX_BINS)

    if side == "support":
        if case == "CONTAINS":
            k = rng.randint(0, width)
            lower = current_price - k
            upper = lower + width
            if lower < BIN_MIN or upper > BIN_MAX:
                return None
            return ZoneNode(direction="support", lower_bin=lower, upper_bin=upper)

        if case == "TOUCH":
            candidates = [c for c in last5 if c[3] < current_price]
            if not candidates:
                return None
            anchor = rng.choice(candidates)[3]
            upper = min(anchor + rng.randint(0, width), current_price - 1)
            lower = upper - width
            if lower < BIN_MIN or upper < BIN_MIN or upper >= current_price:
                return None
            return ZoneNode(direction="support", lower_bin=lower, upper_bin=upper)

        if case == "NOTOUCH":
            min_low_last5 = min(c[2] for c in last5)
            ceiling = min(min_low_last5, current_price) - 1
            upper = ceiling
            lower = upper - width
            if lower < BIN_MIN or upper < BIN_MIN:
                return None
            return ZoneNode(direction="support", lower_bin=lower, upper_bin=upper)

    elif side == "resistance":
        if case == "CONTAINS":
            k = rng.randint(0, width)
            upper = current_price + k
            lower = upper - width
            if lower < BIN_MIN or upper > BIN_MAX:
                return None
            return ZoneNode(direction="resistance", lower_bin=lower, upper_bin=upper)

        if case == "TOUCH":
            candidates = [c for c in last5 if c[3] > current_price]
            if not candidates:
                return None
            anchor = rng.choice(candidates)[3]
            lower = max(anchor - rng.randint(0, width), current_price + 1)
            upper = lower + width
            if upper > BIN_MAX or lower <= current_price:
                return None
            return ZoneNode(direction="resistance", lower_bin=lower, upper_bin=upper)

        if case == "NOTOUCH":
            max_high_last5 = max(c[1] for c in last5)
            floor = max(max_high_last5, current_price) + 1
            lower = floor
            upper = lower + width
            if upper > BIN_MAX or lower > BIN_MAX:
                return None
            return ZoneNode(direction="resistance", lower_bin=lower, upper_bin=upper)

    return None


# ---------------------------------------------------------------------
# Construct SL/RR cho BUY/SELL — thỏa is_sl_valid (khoảng cách cố định
# từ ENTRY + đúng phía zone) và derive_target không bị bão hoà bin.
# ---------------------------------------------------------------------
def _pick_sl_rr(rng: random.Random, action_type: str, current_price: int, zone: ZoneNode) -> Optional[Tuple[int, int]]:
    direction = "long" if action_type == "BUY" else "short"
    dist_candidates = list(range(SL_MIN_DIST_BINS, SL_MAX_DIST_BINS + 1))
    rng.shuffle(dist_candidates)

    for dist in dist_candidates:
        if direction == "long":
            sl = current_price - dist
            if not (BIN_MIN <= sl < zone.lower_bin):
                continue
        else:
            sl = current_price + dist
            if not (sl > zone.upper_bin and sl <= BIN_MAX):
                continue

        rr_candidates = list(range(RR_MIN, RR_MAX + 1))
        rng.shuffle(rr_candidates)
        for rr in rr_candidates:
            if derive_target(current_price, sl, rr, direction) is not None:
                return sl, rr

    return None


def _build_completion_text(think: ThinkNode, action: ActionNode) -> str:
    parts = ["<think>", f"<trend>{think.trend}</trend>",
             "<current_price>", *_digits(think.current_price_bin), "</current_price>"]

    if think.zone is not None:
        tag = "zone_support" if think.zone.direction == "support" else "zone_resistance"
        parts += [f"<{tag}>", *_digits(think.zone.lower_bin), ":", *_digits(think.zone.upper_bin), f"</{tag}>"]

    if think.price_in_zone:
        parts.append("<price_in_zone>")
    if think.good_price_action:
        parts.append("<good_price_action>")
    parts.append("</think>")

    parts += ["<action>", action.action_type]
    if action.sl is not None and action.rr is not None:
        parts += ["SL:", *_digits(action.sl), f"<RR_{action.rr}>"]
    parts.append("</action>")

    return " ".join(parts)


def _build_chart_text(candles: Sequence[Candle]) -> str:
    parts = ["<chart>"]
    for o, h, l, c in candles:
        parts.extend([f"<O_{o}>", f"<H_{h}>", f"<L_{l}>", f"<C_{c}>"])
    parts.append("</chart>")
    return " ".join(parts)


def generate_one(
    candles: Sequence[Candle],
    rng: random.Random,
    max_attempts: int = 30,
) -> Optional[GeneratedSample]:
    """
    Sinh 1 mẫu (prompt, completion) cho 1 chart thật cố định (candles).

    Thứ tự sinh ĐÚNG theo spec 7.2: random zone trước -> tính price_in_zone
    THẬT từ chart -> random action phù hợp. current_price luôn = Close nến
    cuối (không random). Leaf-path (trend/zone_side/zone_case/action) được
    sample UNIFORM trên toàn bộ danh sách LEAF_RECIPES đã liệt kê tường minh.

    Trả None nếu sau `max_attempts` lần thử vẫn không dựng được số liệu
    hình học thoả mãn leaf đã chọn trên chính chart này (vd không nến nào
    trong 5 nến cuối phù hợp cho case TOUCH) — caller nên thử lại (leaf
    khác hoặc random state khác), KHÔNG phải lỗi của generator.
    """
    current_price = candles[-1][3]
    last5 = candles[-LAST_N_CANDLES_TOUCH:]

    for _ in range(max_attempts):
        trend, side, case, action_type = rng.choice(LEAF_RECIPES)

        think = ThinkNode(trend=trend, current_price_bin=current_price)
        action = ActionNode(action_type=action_type)

        if side is None:
            # RANGE không zone -> HOLD
            pass
        else:
            zone = _pick_zone(rng, side, case, current_price, last5)
            if zone is None:
                continue
            think.zone = zone
            think.price_in_zone = case in ("CONTAINS", "TOUCH")

            if action_type in ("BUY", "SELL"):
                think.good_price_action = True
                sl_rr = _pick_sl_rr(rng, action_type, current_price, zone)
                if sl_rr is None:
                    continue
                action.sl, action.rr = sl_rr
            # CANCEL_*/WAIT_*: không set thêm gì (đúng bảng 2.2.F — cấm các field này)

        completion = _build_completion_text(think, action)
        prompt = _build_chart_text(candles)

        # Tự verify lại bằng chính Parser/SemanticChecker/evaluate_outcome
        # (nguyên tắc "verifier = lật ngược generator") trước khi trả về —
        # phòng vệ cho các trường hợp biên mà công thức dựng số ở trên bỏ sót.
        full_text = prompt + " " + completion
        parse_result = Parser.from_text(full_text).parse()
        if not parse_result.is_well_formed():
            continue
        sem_result = SemanticChecker().check(parse_result.ast)
        if not sem_result.passed:
            continue
        if action_type in ("BUY", "SELL", "CANCEL_BUY", "CANCEL_SELL"):
            extra_valid, _ = evaluate_outcome(parse_result.ast.action, parse_result.ast.think, candles)
            if not extra_valid:
                continue

        leaf_name = f"{trend}|{side}|{case}|{action_type}"
        return GeneratedSample(prompt=prompt, completion=completion, leaf_recipe=leaf_name)

    return None


def generate_dataset(
    charts: Sequence[Sequence[Candle]],
    samples_per_chart: int = 4,
    seed: Optional[int] = None,
    max_attempts: int = 30,
) -> List[GeneratedSample]:
    """
    Sinh dataset pretrain/SFT: với MỖI chart thật, sinh `samples_per_chart`
    mẫu think/action khác nhau (random leaf mỗi lần) — tăng số lượng mẫu
    mà không cần thêm chart mới, vì current_price luôn nhất quán = Close
    nến cuối của chính chart đó ở mọi lần lặp lại.
    """
    rng = random.Random(seed)
    samples: List[GeneratedSample] = []

    for candles in charts:
        for _ in range(samples_per_chart):
            sample = generate_one(candles, rng, max_attempts=max_attempts)
            if sample is not None:
                samples.append(sample)

    return samples