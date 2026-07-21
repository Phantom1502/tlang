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
    # 3 nhóm target tường minh — WAIT tự bù = 1 - tổng 3 nhóm này (mục 3, chốt lượt trước)
    "target_hold_ratio",
    "target_trade_ratio",
    "target_cancel_ratio",
    # EMA + proportional control — DÙNG CHUNG cho cả 4 nhóm
    "ema_alpha",
    "buff_kp",
    "buff_step_max",
    # range riêng từng nhóm
    "trade_buff_min", "trade_buff_max",
    "hold_buff_min", "hold_buff_max",
    "cancel_buff_min", "cancel_buff_max",
    "wait_buff_min", "wait_buff_max",
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

    pass_gate2_bonus: float       # K — sàn tuyệt đối khi pass gate well-form + semantic
    zone_score_scale: float
    sl_valid_bonus: float
    sl_valid_penalty: float
    trade_fee_bins: float

    target_hold_ratio: float
    target_trade_ratio: float     # BUY+SELL gộp chung 1 nhóm
    target_cancel_ratio: float
    # target_wait_ratio KHÔNG khai báo field — suy ra trong __post_init__,
    # KHÔNG đọc/ghi trực tiếp từ JSON (đây là chốt thiết kế: 3 nhóm tường
    # minh, 1 nhóm tự bù).

    ema_alpha: float              # ema_new = (1-alpha)*rate_step + alpha*ema_old
    buff_kp: float                # delta = kp * (target - ema_ratio)
    buff_step_max: float          # trần |delta| mỗi lần update (1 lần / optimizer step)

    trade_buff_min: float
    trade_buff_max: float
    hold_buff_min: float
    hold_buff_max: float
    cancel_buff_min: float
    cancel_buff_max: float
    wait_buff_min: float
    wait_buff_max: float

    # init=None -> mặc định = min (giữ hành vi cũ của buff_init trước đây)
    trade_buff_init: Optional[float] = None
    hold_buff_init: Optional[float] = None
    cancel_buff_init: Optional[float] = None
    wait_buff_init: Optional[float] = None

    def __post_init__(self) -> None:
        if self.trade_buff_init is None:
            self.trade_buff_init = self.trade_buff_min
        if self.hold_buff_init is None:
            self.hold_buff_init = self.hold_buff_min
        if self.cancel_buff_init is None:
            self.cancel_buff_init = self.cancel_buff_min
        if self.wait_buff_init is None:
            self.wait_buff_init = self.wait_buff_min

        for name, lo, val, hi in (
            ("trade", self.trade_buff_min, self.trade_buff_init, self.trade_buff_max),
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

        group_sum = self.target_hold_ratio + self.target_trade_ratio + self.target_cancel_ratio
        if not (0.0 <= group_sum <= 1.0):
            raise ValueError(
                f"target_hold_ratio + target_trade_ratio + target_cancel_ratio = {group_sum:.4f}, "
                f"phải nằm trong [0,1] (phần còn lại tự suy ra cho WAIT)."
            )
        self.target_wait_ratio = 1.0 - group_sum   # attribute suy ra, KHÔNG phải dataclass field

        if not (0.0 <= self.ema_alpha < 1.0):
            raise ValueError(f"ema_alpha phải nằm trong [0,1), nhận {self.ema_alpha}.")
        if self.buff_kp < 0:
            raise ValueError(f"buff_kp phải >= 0, nhận {self.buff_kp}.")
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

        # ==============================================================
        # Bất biến bắt buộc, MỞ RỘNG cho đủ 4 nhóm (trước đây chỉ check
        # TRADE/HOLD) — worst-case reward khi PASS gate của MỖI nhóm phải
        # LỚN HƠN worst-case khi FAIL gate nhẹ nhất (semantic fail).
        #
        # worst_zone_score áp dụng cho MỌI action có zone (TRADE/CANCEL/WAIT
        # đều có zone; HOLD thì KHÔNG — RANGE không zone).
        # worst_outcome_score CHỈ áp dụng cho TRADE (BUY/SELL là action duy
        # nhất có outcome_score, xem compute_outcome_score).
        # ==============================================================
        fee_worst = self.trade_fee_bins / self.sl_min_dist_bins
        worst_zone_score = -1.0 * self.zone_score_scale
        worst_outcome_score = -1.0 - fee_worst
        gate2_fail_max = _R_WF_FULL + _R_SEM_FULL + self.sl_valid_bonus

        worst_by_group = {
            "TRADE": self.pass_gate2_bonus + worst_zone_score + worst_outcome_score + self.trade_buff_min,
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
            target_trade_ratio=float(data["target_trade_ratio"]),
            target_cancel_ratio=float(data["target_cancel_ratio"]),
            ema_alpha=float(data["ema_alpha"]),
            buff_kp=float(data["buff_kp"]),
            buff_step_max=float(data["buff_step_max"]),
            trade_buff_min=float(data["trade_buff_min"]),
            trade_buff_max=float(data["trade_buff_max"]),
            hold_buff_min=float(data["hold_buff_min"]),
            hold_buff_max=float(data["hold_buff_max"]),
            cancel_buff_min=float(data["cancel_buff_min"]),
            cancel_buff_max=float(data["cancel_buff_max"]),
            wait_buff_min=float(data["wait_buff_min"]),
            wait_buff_max=float(data["wait_buff_max"]),
            trade_buff_init=data.get("trade_buff_init"),
            hold_buff_init=data.get("hold_buff_init"),
            cancel_buff_init=data.get("cancel_buff_init"),
            wait_buff_init=data.get("wait_buff_init"),
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # target_wait_ratio là attribute suy ra (không phải dataclass field
        # khai báo) -> asdict() KHÔNG tự gồm nó, không cần pop tay.
        p.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")