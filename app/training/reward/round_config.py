from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

_REQUIRED_KEYS = (
    "round_id",
    "zone_width_min_bins",
    "zone_width_max_bins",
    "sl_min_dist_bins",
    "sl_max_dist_bins",
    "pass_gate2_bonus",
    "zone_score_scale",
    "sl_valid_bonus",
    "sl_valid_penalty",
    "trade_fee_bins",
    "target_hold_ratio",
    "target_buy_ratio",
    "target_sell_ratio",
    "target_cancel_ratio",
    "ema_alpha",
    "buff_kp",
    "buff_kd",
    "buff_step_max",
    "buy_buff_min", "buy_buff_max",
    "sell_buff_min", "sell_buff_max",
    "hold_buff_min", "hold_buff_max",
    "cancel_buff_min", "cancel_buff_max",
    "wait_buff_min", "wait_buff_max",
    # =================================================================
    # THÊM MỚI — RR entropy floor (chống "exploration collapse" cho RR
    # của BUY/SELL, TÁCH BIỆT hoàn toàn khỏi buff nhóm action ở trên).
    #
    # KHÁC BẢN CHẤT với buff nhóm: đây là 1 FLOOR MỘT CHIỀU (không phải
    # target 2 chiều) — RR không có "tỉ lệ đúng" cố định, entropy cao hay
    # thấp còn tuỳ context input; chỉ can thiệp khi entropy Shannon của
    # phân phối RR TRONG 1 NHÓM ROLLOUT (num_generations completion cùng
    # 1 prompt) tụt xuống dưới floor — dấu hiệu model gần như luôn sinh
    # đúng 1 giá trị RR bất kể context, khiến advantage trong nhóm không
    # còn gì để so sánh (exploration đã chết), không phải vì đó là lựa
    # chọn tối ưu theo outcome thật.
    #
    # rr_entropy_floor tính bằng NATURAL LOG (nats) — với RR có tối đa 9
    # giá trị (1..9, xem RR_MIN/RR_MAX ở app/lang/tokens.py), entropy tối
    # đa lý thuyết = ln(9) ≈ 2.197 nats khi phân phối đều tuyệt đối.
    # =================================================================
    "rr_entropy_floor",
    "rr_entropy_ema_alpha",
    "rr_entropy_kp",
    "rr_entropy_kd",
    "rr_entropy_bonus_step_max",
    "rr_entropy_bonus_cap",
)

# PHẢI khớp R_WF_FULL/R_SEM_FULL trong reward_func.py — không import trực tiếp
# (circular import, reward_func.py import RoundConfig từ đây ở top-level).
_R_WF_FULL = 1.0
_R_SEM_FULL = 1.0


@dataclass
class RoundConfig:
    round_id: str

    zone_width_min_bins: int
    zone_width_max_bins: int
    sl_min_dist_bins: int
    sl_max_dist_bins: int

    pass_gate2_bonus: float
    zone_score_scale: float
    sl_valid_bonus: float
    sl_valid_penalty: float
    trade_fee_bins: float

    target_hold_ratio: float
    target_buy_ratio: float
    target_sell_ratio: float
    target_cancel_ratio: float

    ema_alpha: float
    buff_kp: float
    buff_kd: float
    buff_step_max: float

    buy_buff_min: float
    buy_buff_max: float
    sell_buff_min: float
    sell_buff_max: float
    hold_buff_min: float
    hold_buff_max: float
    cancel_buff_min: float
    cancel_buff_max: float
    wait_buff_min: float
    wait_buff_max: float

    # THÊM MỚI — xem giải thích ở _REQUIRED_KEYS phía trên.
    rr_entropy_floor: float           # ngưỡng sàn (nats) — dưới ngưỡng này bonus bắt đầu kích hoạt
    rr_entropy_ema_alpha: float       # EMA riêng cho entropy reading, ĐỘC LẬP với ema_alpha của buff nhóm
    rr_entropy_kp: float              # P-term: delta = kp * max(0, floor - ema_entropy)
    rr_entropy_kd: float              # D-term: "phanh"/decay sớm khi entropy đang hồi phục nhanh
    rr_entropy_bonus_step_max: float  # trần |delta| mỗi lần update (1 lần/optimizer step)
    rr_entropy_bonus_cap: float       # trần TUYỆT ĐỐI của bonus (sàn dưới luôn = 0.0, không cần field riêng)

    # init=None -> mặc định = min (giữ hành vi cũ của buff_init trước đây)
    buy_buff_init: Optional[float] = None
    sell_buff_init: Optional[float] = None
    hold_buff_init: Optional[float] = None
    cancel_buff_init: Optional[float] = None
    wait_buff_init: Optional[float] = None

    def __post_init__(self) -> None:
        if self.buy_buff_init is None:
            self.buy_buff_init = self.buy_buff_min
        if self.sell_buff_init is None:
            self.sell_buff_init = self.sell_buff_min
        if self.hold_buff_init is None:
            self.hold_buff_init = self.hold_buff_min
        if self.cancel_buff_init is None:
            self.cancel_buff_init = self.cancel_buff_min
        if self.wait_buff_init is None:
            self.wait_buff_init = self.wait_buff_min

        for name, lo, val, hi in (
            ("buy", self.buy_buff_min, self.buy_buff_init, self.buy_buff_max),
            ("sell", self.sell_buff_min, self.sell_buff_init, self.sell_buff_max),
            ("hold", self.hold_buff_min, self.hold_buff_init, self.hold_buff_max),
            ("cancel", self.cancel_buff_min, self.cancel_buff_init, self.cancel_buff_max),
            ("wait", self.wait_buff_min, self.wait_buff_init, self.wait_buff_max),
        ):
            if lo > hi:
                raise ValueError(f"{name}_buff_min ({lo}) phải <= {name}_buff_max ({hi}).")
            if not (lo <= val <= hi):
                raise ValueError(
                    f"{name}_buff_init ({val}) phải nằm trong [{name}_buff_min, {name}_buff_max] "
                    f"= [{lo},{hi}]."
                )

        group_sum = self.target_hold_ratio + self.target_buy_ratio + self.target_sell_ratio + self.target_cancel_ratio
        if not (0.0 <= group_sum <= 1.0):
            raise ValueError(
                f"target_hold_ratio + target_buy_ratio + target_sell_ratio + target_cancel_ratio = {group_sum:.4f}, "
                f"phải nằm trong [0,1] (phần còn lại tự suy ra cho WAIT)."
            )
        self.target_wait_ratio = 1.0 - group_sum

        if not (0.0 <= self.ema_alpha < 1.0):
            raise ValueError(f"ema_alpha phải nằm trong [0,1), nhận {self.ema_alpha}.")
        if self.buff_kp < 0:
            raise ValueError(f"buff_kp phải >= 0, nhận {self.buff_kp}.")
        if self.buff_kd < 0:
            raise ValueError(f"buff_kd phải >= 0, nhận {self.buff_kd}.")
        if self.buff_step_max < 0:
            raise ValueError(f"buff_step_max phải >= 0, nhận {self.buff_step_max}.")
        if self.zone_score_scale < 0:
            raise ValueError(f"zone_score_scale phải >= 0, nhận {self.zone_score_scale}.")
        if self.sl_valid_bonus < 0:
            raise ValueError(f"sl_valid_bonus phải >= 0, nhận {self.sl_valid_bonus}.")
        if self.sl_valid_penalty < 0:
            raise ValueError(f"sl_valid_penalty phải >= 0, nhận {self.sl_valid_penalty}.")
        if self.trade_fee_bins < 0:
            raise ValueError(f"trade_fee_bins phải >= 0, nhận {self.trade_fee_bins}.")
        if self.sl_min_dist_bins <= 0:
            raise ValueError(f"sl_min_dist_bins phải > 0, nhận {self.sl_min_dist_bins}.")

        # THÊM MỚI — validation cho RR entropy controller. Không có bất
        # biến nào cần thêm vào worst_by_group bên dưới: bonus này CỘNG
        # THÊM, LUÔN >= 0 (sàn dưới cố định 0.0, không có field min riêng),
        # nên worst-case reward khi PASS gate KHÔNG bị giảm bởi cơ chế
        # này — bất biến "worst pass > gate2_fail_max" vẫn đúng nguyên
        # vẹn như trước khi thêm entropy bonus (worst case = bonus 0).
        if self.rr_entropy_floor < 0:
            raise ValueError(f"rr_entropy_floor phải >= 0, nhận {self.rr_entropy_floor}.")
        if not (0.0 <= self.rr_entropy_ema_alpha < 1.0):
            raise ValueError(f"rr_entropy_ema_alpha phải nằm trong [0,1), nhận {self.rr_entropy_ema_alpha}.")
        if self.rr_entropy_kp < 0:
            raise ValueError(f"rr_entropy_kp phải >= 0, nhận {self.rr_entropy_kp}.")
        if self.rr_entropy_kd < 0:
            raise ValueError(f"rr_entropy_kd phải >= 0, nhận {self.rr_entropy_kd}.")
        if self.rr_entropy_bonus_step_max < 0:
            raise ValueError(f"rr_entropy_bonus_step_max phải >= 0, nhận {self.rr_entropy_bonus_step_max}.")
        if self.rr_entropy_bonus_cap < 0:
            raise ValueError(f"rr_entropy_bonus_cap phải >= 0, nhận {self.rr_entropy_bonus_cap}.")

        fee_worst = self.trade_fee_bins / self.sl_min_dist_bins
        worst_zone_score = -1.0 * self.zone_score_scale
        worst_outcome_score = -1.0 - fee_worst
        gate2_fail_max = _R_WF_FULL + _R_SEM_FULL + self.sl_valid_bonus

        worst_by_group = {
            "BUY": self.pass_gate2_bonus + worst_zone_score + worst_outcome_score + self.buy_buff_min,
            "SELL": self.pass_gate2_bonus + worst_zone_score + worst_outcome_score + self.sell_buff_min,
            "HOLD": self.pass_gate2_bonus + self.hold_buff_min,
            "CANCEL": self.pass_gate2_bonus + worst_zone_score + self.cancel_buff_min,
            "WAIT": self.pass_gate2_bonus + worst_zone_score + self.wait_buff_min,
        }
        for group_name, worst in worst_by_group.items():
            if worst <= gate2_fail_max:
                raise ValueError(
                    f"[round {self.round_id!r}] nhóm {group_name}: worst-case reward khi PASS gate "
                    f"= {worst:.3f}, phải LỚN HƠN gate2_fail_max ({gate2_fail_max:.3f}) — nếu không, "
                    f"1 completion PASS gate với outcome/buff tệ nhất của nhóm {group_name} có thể có "
                    f"reward THẤP HƠN hoặc BẰNG 1 completion FAIL gate nhẹ, phá vỡ gate cứng. Tăng "
                    f"pass_gate2_bonus hoặc nâng {group_name.lower()}_buff_min để sửa."
                )

    @classmethod
    def load(cls, path: str) -> "RoundConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy round config tại {path!r}.")
        data = json.loads(p.read_text(encoding="utf-8"))
        missing = [k for k in _REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"Round config tại {path!r} THIẾU field bắt buộc: {missing}.")
        return cls(
            round_id=str(data["round_id"]),
            zone_width_min_bins=int(data["zone_width_min_bins"]),
            zone_width_max_bins=int(data["zone_width_max_bins"]),
            sl_min_dist_bins=int(data["sl_min_dist_bins"]),
            sl_max_dist_bins=int(data["sl_max_dist_bins"]),
            pass_gate2_bonus=float(data["pass_gate2_bonus"]),
            zone_score_scale=float(data["zone_score_scale"]),
            sl_valid_bonus=float(data["sl_valid_bonus"]),
            sl_valid_penalty=float(data["sl_valid_penalty"]),
            trade_fee_bins=float(data["trade_fee_bins"]),
            target_hold_ratio=float(data["target_hold_ratio"]),
            target_buy_ratio=float(data["target_buy_ratio"]),
            target_sell_ratio=float(data["target_sell_ratio"]),
            target_cancel_ratio=float(data["target_cancel_ratio"]),
            ema_alpha=float(data["ema_alpha"]),
            buff_kp=float(data["buff_kp"]),
            buff_kd=float(data["buff_kd"]),
            buff_step_max=float(data["buff_step_max"]),
            buy_buff_min=float(data["buy_buff_min"]),
            buy_buff_max=float(data["buy_buff_max"]),
            sell_buff_min=float(data["sell_buff_min"]),
            sell_buff_max=float(data["sell_buff_max"]),
            hold_buff_min=float(data["hold_buff_min"]),
            hold_buff_max=float(data["hold_buff_max"]),
            cancel_buff_min=float(data["cancel_buff_min"]),
            cancel_buff_max=float(data["cancel_buff_max"]),
            wait_buff_min=float(data["wait_buff_min"]),
            wait_buff_max=float(data["wait_buff_max"]),
            rr_entropy_floor=float(data["rr_entropy_floor"]),
            rr_entropy_ema_alpha=float(data["rr_entropy_ema_alpha"]),
            rr_entropy_kp=float(data["rr_entropy_kp"]),
            rr_entropy_kd=float(data["rr_entropy_kd"]),
            rr_entropy_bonus_step_max=float(data["rr_entropy_bonus_step_max"]),
            rr_entropy_bonus_cap=float(data["rr_entropy_bonus_cap"]),
            buy_buff_init=data.get("buy_buff_init"),
            sell_buff_init=data.get("sell_buff_init"),
            hold_buff_init=data.get("hold_buff_init"),
            cancel_buff_init=data.get("cancel_buff_init"),
            wait_buff_init=data.get("wait_buff_init"),
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")