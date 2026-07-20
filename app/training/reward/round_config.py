from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

_REQUIRED_KEYS = (
    "round_id",
    "target_action_ratio",   # float trong [0,1] — tỉ lệ mong muốn (BUY+SELL)/tổng action
    "zone_width_min_bins",
    "zone_width_max_bins",
    "sl_min_dist_bins",
    "sl_max_dist_bins",
    "pass_gate2_bonus",      # K — sàn tuyệt đối khi pass gate well-form + semantic
    "zone_score_scale",      # scale nhân với r_multiple của probe_zone_quality
    "sl_valid_bonus",        # thưởng đối xứng khi SL đúng luật (chỉ BUY/SELL)
    "sl_valid_penalty",      # phạt đối xứng khi SL sai luật (chỉ BUY/SELL)
    "trade_fee_bins",
    "buff_step",             # mỗi lần update_buffs_from_stats(): tăng/giảm buff bao nhiêu
    "buff_max",
    "buff_min",
    "target_hold_ratio",
    "hold_buff_step", 
    "hold_buff_max", 
    "hold_buff_min",
)

# =====================================================================
# 2 hằng số này PHẢI khớp R_WF_FULL/R_SEM_FULL trong
# app/training/reward/reward_func.py — không import trực tiếp từ đó để
# tránh circular import (reward_func.py đã import RoundConfig từ module
# này ở top-level). Nếu đổi R_WF_FULL/R_SEM_FULL bên reward_func.py,
# PHẢI sửa lại 2 số dưới đây cho khớp, nếu không invariant ở
# __post_init__ sẽ tính sai gate2_fail_max.
# =====================================================================
_R_WF_FULL = 1.0
_R_SEM_FULL = 1.0


@dataclass
class RoundConfig:
    round_id: str
    target_action_ratio: float
    zone_width_min_bins: int
    zone_width_max_bins: int
    sl_min_dist_bins: int
    sl_max_dist_bins: int
    pass_gate2_bonus: float
    zone_score_scale: float
    sl_valid_bonus: float
    sl_valid_penalty: float
    trade_fee_bins: float
    buff_step: float
    buff_max: float
    buff_min: float = 0.0
    target_hold_ratio: float = 0.10
    hold_buff_step: float = 0.0
    hold_buff_max: float = 0.0
    hold_buff_min: float = 0.0

    def __post_init__(self) -> None:
        """
        Bất biến bắt buộc — reward khi PASS gate (K + zone_score + outcome_score)
        phải LUÔN LỚN HƠN reward khi FAIL gate nhẹ nhất (semantic fail,
        R_WF_FULL + sem_score, sem_score có thể chạm ~1.0 + sl_valid_bonus).

        worst_zone_score    = -1.0 * zone_score_scale
            (probe_zone_quality LOSS -> r_multiple sàn -1.0, xem forward_test.py)
        worst_outcome_score = -1.0 - fee_worst + buff_min
            (BUY/SELL LOSS -> r_multiple=-1.0; fee_worst = trade_fee_bins /
            sl_min_dist_bins, risk nhỏ nhất -> fee quy đổi ra R lớn nhất;
            buff_min mặc định 0.0 -> KHÔNG cứu được worst-case)
        gate2_fail_max = R_WF_FULL + R_SEM_FULL + sl_valid_bonus
            (semantic fail nhưng sl_valid=True vẫn được cộng sl_valid_bonus)

        Cần: K + worst_zone_score + worst_outcome_score > gate2_fail_max
        """
        if not (0.0 <= self.target_action_ratio <= 1.0):
            raise ValueError(f"target_action_ratio phải trong [0,1], nhận {self.target_action_ratio}.")
        if self.zone_score_scale < 0:
            raise ValueError(f"zone_score_scale phải >= 0, nhận {self.zone_score_scale}.")
        if self.sl_valid_bonus < 0:
            raise ValueError(f"sl_valid_bonus phải >= 0, nhận {self.sl_valid_bonus}.")
        if self.sl_valid_penalty < 0:
            raise ValueError(f"sl_valid_penalty phải >= 0, nhận {self.sl_valid_penalty}.")
        if self.trade_fee_bins < 0:
            raise ValueError(f"trade_fee_bins phải >= 0 (phí, không phải bonus), nhận {self.trade_fee_bins}.")
        if self.buff_step < 0:
            raise ValueError(f"buff_step phải >= 0, nhận {self.buff_step}.")
        if self.buff_min > self.buff_max:
            raise ValueError(f"buff_min ({self.buff_min}) phải <= buff_max ({self.buff_max}).")
        if not (0.0 <= self.target_hold_ratio <= 1.0):
            raise ValueError(f"target_hold_ratio phải trong [0,1], nhận {self.target_hold_ratio}.")
        if self.hold_buff_step < 0:
            raise ValueError(f"hold_buff_step phải >= 0, nhận {self.hold_buff_step}.")
        if self.hold_buff_min > self.hold_buff_max:
            raise ValueError(f"hold_buff_min ({self.hold_buff_min}) phải <= hold_buff_max ({self.hold_buff_max}).")
        if self.sl_min_dist_bins <= 0:
            raise ValueError(f"sl_min_dist_bins phải > 0, nhận {self.sl_min_dist_bins}.")

        fee_worst = self.trade_fee_bins / self.sl_min_dist_bins
        worst_zone_score = -1.0 * self.zone_score_scale
        worst_outcome_score = -1.0 - fee_worst + self.buff_min
        gate2_fail_max = _R_WF_FULL + _R_SEM_FULL + self.sl_valid_bonus

        worst_pass = self.pass_gate2_bonus + worst_zone_score + worst_outcome_score
        if worst_pass <= gate2_fail_max:
            raise ValueError(
                f"[round {self.round_id!r}] pass_gate2_bonus (K={self.pass_gate2_bonus}) KHÔNG đủ lớn: "
                f"worst-case reward khi PASS gate = K({self.pass_gate2_bonus}) + worst_zone_score"
                f"({worst_zone_score:.3f}) + worst_outcome_score({worst_outcome_score:.3f}) = "
                f"{worst_pass:.3f}, phải LỚN HƠN gate2_fail_max ({gate2_fail_max:.3f}) — nếu không, "
                f"1 completion PASS gate với outcome/zone tệ nhất có thể có reward THẤP HƠN hoặc BẰNG "
                f"1 completion FAIL gate nhẹ, phá vỡ gate cứng. Tăng pass_gate2_bonus hoặc giảm "
                f"zone_score_scale/trade_fee_bins/sl_valid_bonus để sửa."
            )
        worst_hold = self.pass_gate2_bonus + self.hold_buff_min
        if worst_hold <= gate2_fail_max:
            raise ValueError(
                f"[round {self.round_id!r}] hold_buff_min ({self.hold_buff_min}) khiến worst-case "
                f"reward của HOLD khi PASS gate = K({self.pass_gate2_bonus}) + hold_buff_min"
                f"({self.hold_buff_min}) = {worst_hold:.3f}, phải LỚN HƠN gate2_fail_max "
                f"({gate2_fail_max:.3f}). Tăng pass_gate2_bonus hoặc nâng hold_buff_min."
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
            target_action_ratio=float(data["target_action_ratio"]),
            zone_width_min_bins=int(data["zone_width_min_bins"]),
            zone_width_max_bins=int(data["zone_width_max_bins"]),
            sl_min_dist_bins=int(data["sl_min_dist_bins"]),
            sl_max_dist_bins=int(data["sl_max_dist_bins"]),
            pass_gate2_bonus=float(data["pass_gate2_bonus"]),
            zone_score_scale=float(data["zone_score_scale"]),
            sl_valid_bonus=float(data["sl_valid_bonus"]),
            sl_valid_penalty=float(data["sl_valid_penalty"]),
            trade_fee_bins=float(data["trade_fee_bins"]),
            buff_step=float(data["buff_step"]),
            buff_max=float(data["buff_max"]),
            buff_min=float(data.get("buff_min", 0.0)),
            target_hold_ratio=float(data["target_hold_ratio"]),
            hold_buff_step=float(data["hold_buff_step"]),
            hold_buff_max=float(data["hold_buff_max"]),
            hold_buff_min=float(data["hold_buff_min"]),
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")