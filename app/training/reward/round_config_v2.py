from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

# =====================================================================
# round_config_v2.py — RoundConfig cho reward v2 (xem docs thiết kế: gate
# chung = FULL SemanticChecker.check() (A/B/B2/D/E), sau đó rẽ 2 task ĐỘC
# LẬP (zone/action), mỗi task có buff riêng (KHÔNG còn K/pass_gate2_bonus/
# zone_score_scale/ZONE_GATE_FULL/ACTION_GATE_FULL như các bản nháp trước —
# đã bỏ vì task action fold thẳng vào nhánh "semantic fail" khi SL invalid,
# không cần hằng số gate riêng nào để giữ bất biến worst>best nữa).
#
# NGUỒN SỰ THẬT DUY NHẤT cho "NO_ZONE ratio của task zone": common gate
# (bảng A + E của SemanticChecker) ép CỨNG bất biến "zone is None ⟺
# trend=RANGE và action_type=HOLD" cho MỌI completion pass gate (rule A:
# UP/DOWN bắt buộc phải có zone; rule E: zone is None mà trend=RANGE thì
# action PHẢI là HOLD, và zone is not None thì HOLD KHÔNG BAO GIỜ nằm
# trong valid_actions của bất kỳ nhánh nào). Do đó tỉ lệ "no-zone" quan sát
# được ở task zone LUÔN đúng bằng tỉ lệ "HOLD" quan sát được ở task action
# — không phải quy ước tiện lợi, mà là hệ quả tất yếu của gate. Vì vậy
# KHÔNG có field JSON riêng cho no_zone_ratio — suy ra bằng property từ
# action_hold_target_ratio, tránh 2 con số có thể lệch nhau do sửa tay.
# =====================================================================

GROUPS_ACTION: Tuple[str, ...] = (
    "HOLD", "BUY", "SELL", "CANCEL_BUY", "CANCEL_SELL", "WAIT_BUY", "WAIT_SELL",
)
GROUPS_ZONE: Tuple[str, ...] = ("HAS_ZONE", "NO_ZONE")

_RATIO_SUM_EPS = 1e-6

_ACTION_TARGET_FIELDS: Tuple[str, ...] = (
    "action_hold_target_ratio",
    "action_buy_target_ratio",
    "action_sell_target_ratio",
    "action_cancel_buy_target_ratio",
    "action_cancel_sell_target_ratio",
    "action_wait_buy_target_ratio",
    "action_wait_sell_target_ratio",
)

_REQUIRED_KEYS: Tuple[str, ...] = (
    "round_id",
    "zone_width_min_bins", "zone_width_max_bins",
    "sl_min_dist_bins", "sl_max_dist_bins",
    "trade_fee_bins", "entry_score_weight",

    "action_ema_alpha", "action_buff_kp", "action_buff_kd", "action_buff_step_max",
) + _ACTION_TARGET_FIELDS + (
    "action_hold_buff_min", "action_hold_buff_max",
    "action_buy_buff_min", "action_buy_buff_max",
    "action_sell_buff_min", "action_sell_buff_max",
    "action_cancel_buy_buff_min", "action_cancel_buy_buff_max",
    "action_cancel_sell_buff_min", "action_cancel_sell_buff_max",
    "action_wait_buy_buff_min", "action_wait_buy_buff_max",
    "action_wait_sell_buff_min", "action_wait_sell_buff_max",

    "zone_ema_alpha", "zone_buff_kp", "zone_buff_kd", "zone_buff_step_max",
    "zone_has_zone_buff_min", "zone_has_zone_buff_max",
    "zone_no_zone_buff_min", "zone_no_zone_buff_max",

    "rr_entropy_floor", "rr_entropy_ema_alpha", "rr_entropy_kp", "rr_entropy_kd",
    "rr_entropy_bonus_step_max", "rr_entropy_bonus_cap",
)


@dataclass
class RoundConfigV2:
    round_id: str

    # --- Common semantic gate (tái dùng nguyên vẹn SemanticChecker) ---
    zone_width_min_bins: int
    zone_width_max_bins: int
    sl_min_dist_bins: int
    sl_max_dist_bins: int

    # --- Task action: phí giao dịch + trọng số nhánh entry-quality ---
    trade_fee_bins: float
    entry_score_weight: float   # biên độ khuyến nghị [0,2], giá trị thường dùng ~0.2

    # --- Buff PD params: action (7 nhóm, dùng chung 1 bộ tham số PD) ---
    action_ema_alpha: float
    action_buff_kp: float
    action_buff_kd: float
    action_buff_step_max: float

    action_hold_target_ratio: float
    action_buy_target_ratio: float
    action_sell_target_ratio: float
    action_cancel_buy_target_ratio: float
    action_cancel_sell_target_ratio: float
    action_wait_buy_target_ratio: float
    action_wait_sell_target_ratio: float

    action_hold_buff_min: float
    action_hold_buff_max: float
    action_buy_buff_min: float
    action_buy_buff_max: float
    action_sell_buff_min: float
    action_sell_buff_max: float
    action_cancel_buy_buff_min: float
    action_cancel_buy_buff_max: float
    action_cancel_sell_buff_min: float
    action_cancel_sell_buff_max: float
    action_wait_buy_buff_min: float
    action_wait_buy_buff_max: float
    action_wait_sell_buff_min: float
    action_wait_sell_buff_max: float

    # --- Buff PD params: zone (2 nhóm HAS_ZONE/NO_ZONE, vòng điều khiển
    # RIÊNG BIỆT với action — tốc độ hội tụ có thể khác, không dùng chung
    # action_ema_alpha/action_buff_kp/kd/step_max) ---
    zone_ema_alpha: float
    zone_buff_kp: float
    zone_buff_kd: float
    zone_buff_step_max: float

    zone_has_zone_buff_min: float
    zone_has_zone_buff_max: float
    zone_no_zone_buff_min: float
    zone_no_zone_buff_max: float

    # --- RR entropy controller (giữ nguyên ý nghĩa như v1 — xem
    # app/training/reward/reward_func.py:RREntropyController) ---
    rr_entropy_floor: float
    rr_entropy_ema_alpha: float
    rr_entropy_kp: float
    rr_entropy_kd: float
    rr_entropy_bonus_step_max: float
    rr_entropy_bonus_cap: float

    # --- init optional — mặc định = min nếu không truyền, giống quy ước v1 ---
    action_hold_buff_init: Optional[float] = None
    action_buy_buff_init: Optional[float] = None
    action_sell_buff_init: Optional[float] = None
    action_cancel_buy_buff_init: Optional[float] = None
    action_cancel_sell_buff_init: Optional[float] = None
    action_wait_buy_buff_init: Optional[float] = None
    action_wait_sell_buff_init: Optional[float] = None
    zone_has_zone_buff_init: Optional[float] = None
    zone_no_zone_buff_init: Optional[float] = None

    def __post_init__(self) -> None:
        # --- init mặc định = min (giữ hành vi cũ của RoundConfig v1) ---
        if self.action_hold_buff_init is None:
            self.action_hold_buff_init = self.action_hold_buff_min
        if self.action_buy_buff_init is None:
            self.action_buy_buff_init = self.action_buy_buff_min
        if self.action_sell_buff_init is None:
            self.action_sell_buff_init = self.action_sell_buff_min
        if self.action_cancel_buy_buff_init is None:
            self.action_cancel_buy_buff_init = self.action_cancel_buy_buff_min
        if self.action_cancel_sell_buff_init is None:
            self.action_cancel_sell_buff_init = self.action_cancel_sell_buff_min
        if self.action_wait_buy_buff_init is None:
            self.action_wait_buy_buff_init = self.action_wait_buy_buff_min
        if self.action_wait_sell_buff_init is None:
            self.action_wait_sell_buff_init = self.action_wait_sell_buff_min
        if self.zone_has_zone_buff_init is None:
            self.zone_has_zone_buff_init = self.zone_has_zone_buff_min
        if self.zone_no_zone_buff_init is None:
            self.zone_no_zone_buff_init = self.zone_no_zone_buff_min

        # --- validate min<=init<=max cho 7 nhóm action + 2 nhóm zone ---
        for group in GROUPS_ACTION:
            lo, hi = self.group_range("action", group)
            init = self.group_init("action", group)
            if lo > hi:
                raise ValueError(f"action_{group.lower()}_buff_min ({lo}) phải <= _buff_max ({hi}).")
            if not (lo <= init <= hi):
                raise ValueError(
                    f"action_{group.lower()}_buff_init ({init}) phải nằm trong [{lo},{hi}]."
                )
        for group in GROUPS_ZONE:
            lo, hi = self.group_range("zone", group)
            init = self.group_init("zone", group)
            if lo > hi:
                raise ValueError(f"zone_{group.lower()}_buff_min ({lo}) phải <= _buff_max ({hi}).")
            if not (lo <= init <= hi):
                raise ValueError(
                    f"zone_{group.lower()}_buff_init ({init}) phải nằm trong [{lo},{hi}]."
                )

        # --- 7 target ratio của action PHẢI cộng đúng 1.0 (không còn nhóm
        # ẩn suy phần dư như v1 — sai lệch > eps là raise ngay) ---
        ratios = {f: getattr(self, f) for f in _ACTION_TARGET_FIELDS}
        for f, v in ratios.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{f} phải nằm trong [0,1], nhận {v}.")
        ratio_sum = sum(ratios.values())
        if abs(ratio_sum - 1.0) > _RATIO_SUM_EPS:
            raise ValueError(
                f"Tổng 7 action_*_target_ratio = {ratio_sum:.6f}, PHẢI đúng bằng 1.0 "
                f"(sai lệch {abs(ratio_sum - 1.0):.6f} > eps={_RATIO_SUM_EPS}). "
                f"Giá trị từng field: {ratios}"
            )

        # --- các validate cơ bản khác ---
        if not (0.0 <= self.entry_score_weight <= 2.0):
            raise ValueError(f"entry_score_weight nên nằm trong [0,2], nhận {self.entry_score_weight}.")
        if self.trade_fee_bins < 0:
            raise ValueError(f"trade_fee_bins phải >= 0, nhận {self.trade_fee_bins}.")
        if self.sl_min_dist_bins <= 0:
            raise ValueError(f"sl_min_dist_bins phải > 0, nhận {self.sl_min_dist_bins}.")

        for namespace, alpha in (("action", self.action_ema_alpha), ("zone", self.zone_ema_alpha)):
            if not (0.0 <= alpha < 1.0):
                raise ValueError(f"{namespace}_ema_alpha phải nằm trong [0,1), nhận {alpha}.")
        for namespace, kp, kd, step_max in (
            ("action", self.action_buff_kp, self.action_buff_kd, self.action_buff_step_max),
            ("zone", self.zone_buff_kp, self.zone_buff_kd, self.zone_buff_step_max),
        ):
            if kp < 0 or kd < 0 or step_max < 0:
                raise ValueError(f"{namespace}_buff_kp/kd/step_max phải >= 0 (nhận kp={kp}, kd={kd}, step_max={step_max}).")

        if self.rr_entropy_floor < 0:
            raise ValueError(f"rr_entropy_floor phải >= 0, nhận {self.rr_entropy_floor}.")
        if not (0.0 <= self.rr_entropy_ema_alpha < 1.0):
            raise ValueError(f"rr_entropy_ema_alpha phải nằm trong [0,1), nhận {self.rr_entropy_ema_alpha}.")
        if self.rr_entropy_kp < 0 or self.rr_entropy_kd < 0:
            raise ValueError("rr_entropy_kp/kd phải >= 0.")
        if self.rr_entropy_bonus_step_max < 0 or self.rr_entropy_bonus_cap < 0:
            raise ValueError("rr_entropy_bonus_step_max/cap phải >= 0.")

    # ------------------------------------------------------------
    # Zone target — SUY RA từ action_hold_target_ratio (xem giải thích ở
    # docstring module) — KHÔNG phải field JSON riêng.
    # ------------------------------------------------------------
    @property
    def zone_target_no_zone_ratio(self) -> float:
        return self.action_hold_target_ratio

    @property
    def zone_target_has_zone_ratio(self) -> float:
        return 1.0 - self.action_hold_target_ratio

    # ------------------------------------------------------------
    # Dispatch theo namespace ("action" | "zone") — dùng chung bởi
    # EMABuffControllerV2 cho CẢ 2 buff controller độc lập.
    # ------------------------------------------------------------
    def group_target(self, namespace: str, group: str) -> float:
        if namespace == "zone":
            if group == "NO_ZONE":
                return self.zone_target_no_zone_ratio
            if group == "HAS_ZONE":
                return self.zone_target_has_zone_ratio
            raise ValueError(f"Nhóm zone không hợp lệ: {group!r}")
        if namespace == "action":
            return getattr(self, f"action_{group.lower()}_target_ratio")
        raise ValueError(f"namespace không hợp lệ: {namespace!r}")

    def group_range(self, namespace: str, group: str) -> Tuple[float, float]:
        prefix = "zone" if namespace == "zone" else "action"
        lo = getattr(self, f"{prefix}_{group.lower()}_buff_min")
        hi = getattr(self, f"{prefix}_{group.lower()}_buff_max")
        return lo, hi

    def group_init(self, namespace: str, group: str) -> float:
        prefix = "zone" if namespace == "zone" else "action"
        return getattr(self, f"{prefix}_{group.lower()}_buff_init")

    def group_ema_alpha(self, namespace: str) -> float:
        return self.zone_ema_alpha if namespace == "zone" else self.action_ema_alpha

    def group_pd_params(self, namespace: str) -> Tuple[float, float, float]:
        if namespace == "zone":
            return self.zone_buff_kp, self.zone_buff_kd, self.zone_buff_step_max
        return self.action_buff_kp, self.action_buff_kd, self.action_buff_step_max

    # ------------------------------------------------------------
    # load/save — cùng pattern RoundConfig v1
    # ------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "RoundConfigV2":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy round config v2 tại {path!r}.")
        data = json.loads(p.read_text(encoding="utf-8"))
        missing = [k for k in _REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"Round config v2 tại {path!r} THIẾU field bắt buộc: {missing}.")

        kwargs = {k: data[k] for k in _REQUIRED_KEYS}
        kwargs["round_id"] = str(kwargs["round_id"])
        for int_field in ("zone_width_min_bins", "zone_width_max_bins", "sl_min_dist_bins", "sl_max_dist_bins"):
            kwargs[int_field] = int(kwargs[int_field])
        for k in list(kwargs.keys()):
            if k not in ("round_id",) and k not in ("zone_width_min_bins", "zone_width_max_bins", "sl_min_dist_bins", "sl_max_dist_bins"):
                kwargs[k] = float(kwargs[k])

        for opt_field in (
            "action_hold_buff_init", "action_buy_buff_init", "action_sell_buff_init",
            "action_cancel_buy_buff_init", "action_cancel_sell_buff_init",
            "action_wait_buy_buff_init", "action_wait_sell_buff_init",
            "zone_has_zone_buff_init", "zone_no_zone_buff_init",
        ):
            if opt_field in data and data[opt_field] is not None:
                kwargs[opt_field] = float(data[opt_field])

        return cls(**kwargs)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")