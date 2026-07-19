from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

_REQUIRED_KEYS = (
    "round_id",
    "weight_table",
    "zone_width_min_bins",
    "zone_width_max_bins",
    "sl_min_dist_bins",
    "sl_max_dist_bins",
    "pass_gate2_bonus",      # K
    "zone_quality_bonus",
    "zone_quality_penalty",
    "sl_valid_bonus",        # thưởng đối xứng khi SL đúng luật (BUY/SELL)
    "sl_valid_penalty",      # phạt đối xứng khi SL sai luật (BUY/SELL)
    "trade_fee_bins",
    "buff_action",
)

# "pure_outcome_mode" CỐ Ý KHÔNG nằm trong _REQUIRED_KEYS — đây là 1 cờ bật/tắt
# chế độ, không phải tham số số học. Thiếu field này trong file JSON cũ (vd
# round1.json đã chạy trước khi field này tồn tại) phải an toàn quay về hành vi
# gate-3-shaping cũ (K + zone_bonus + sl_bonus + timing*w), KHÔNG fail-loud —
# khác với 8 field còn lại (số học, thiếu 1 cái là sai lệch âm thầm nếu có default).


@dataclass
class RoundConfig:
    round_id: str
    weight_table: Dict[str, Dict[str, float]]
    zone_width_min_bins: int
    zone_width_max_bins: int
    sl_min_dist_bins: int
    sl_max_dist_bins: int
    pass_gate2_bonus: float
    zone_quality_bonus: float
    zone_quality_penalty: float
    sl_valid_bonus: float
    sl_valid_penalty: float
    trade_fee_bins: float
    buff_action: float
    pure_outcome_mode: bool = False

    def __post_init__(self) -> None:
        """
        Bất biến bắt buộc — khác nhau tuỳ mode:

        pure_outcome_mode=False (mặc định, hành vi round1 cũ — gate 3 vẫn
        cộng dồn K + zone_bonus + sl_bonus + timing*w):
            K phải LỚN HƠN (zone_quality_penalty + sl_valid_penalty + max_w),
            xem chứng minh đầy đủ ở bản gốc — worst-case pass-gate-2 vẫn phải
            > worst-case fail-gate-2 nhẹ (1.8).

        pure_outcome_mode=True (round 2 — K là SÀN TUYỆT ĐỐI DUY NHẤT khi pass,
        fail well-form/semantic = 0.0 thẳng, KHÔNG còn R_WF_FULL/R_SEM_FULL/
        zone_bonus/sl_bonus nào cộng thêm):
            reward tệ nhất khi PASS  = K + (-1.0) * max_w   (LOSS thật, r_multiple=-1,
                                                              bỏ qua fee cho margin chặt hơn)
            reward khi FAIL          = 0.0
            Cần: K - max_w > 0  <=>  K > max_w.
        """
        max_w = max(
            (w for actions in self.weight_table.values() for w in actions.values()),
            default=0.0,
        )

        if self.pure_outcome_mode:
            if self.pass_gate2_bonus <= max_w:
                raise ValueError(
                    f"[pure_outcome_mode] pass_gate2_bonus (K={self.pass_gate2_bonus}) phải LỚN HƠN "
                    f"max weight trong weight_table ({max_w}) của round {self.round_id!r} — nếu không, "
                    f"1 completion pass gate với outcome LOSS (r_multiple=-1) có thể có reward <= 0, "
                    f"bằng hoặc thấp hơn reward fail gate (luôn = 0.0 ở mode này), phá vỡ gate cứng."
                )
        else:
            if self.pass_gate2_bonus <= self.zone_quality_penalty + self.sl_valid_penalty + max_w:
                raise ValueError(
                    f"pass_gate2_bonus (K={self.pass_gate2_bonus}) phải LỚN HƠN "
                    f"zone_quality_penalty ({self.zone_quality_penalty}) + sl_valid_penalty "
                    f"({self.sl_valid_penalty}) + max weight trong weight_table ({max_w}) của "
                    f"round {self.round_id!r} — nếu không, worst-case pass-gate-2 có thể có "
                    f"reward THẤP HƠN fail-gate-2 nhẹ, phá vỡ gate cứng (xem docstring)."
                )

        if self.zone_quality_bonus < 0:
            raise ValueError(f"zone_quality_bonus phải >= 0, nhận {self.zone_quality_bonus}.")
        if self.zone_quality_penalty < 0:
            raise ValueError(f"zone_quality_penalty phải >= 0, nhận {self.zone_quality_penalty}.")
        if self.sl_valid_bonus < 0:
            raise ValueError(f"sl_valid_bonus phải >= 0, nhận {self.sl_valid_bonus}.")
        if self.sl_valid_penalty < 0:
            raise ValueError(f"sl_valid_penalty phải >= 0, nhận {self.sl_valid_penalty}.")
        if self.trade_fee_bins < 0:
            raise ValueError(f"trade_fee_bins phải >= 0 (phí, không phải bonus), nhận {self.trade_fee_bins}.")
        if self.buff_action < 0:
            raise ValueError(f"buff_action phải >= 0, nhận {self.buff_action}.")

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
            weight_table=data["weight_table"],
            zone_width_min_bins=int(data["zone_width_min_bins"]),
            zone_width_max_bins=int(data["zone_width_max_bins"]),
            sl_min_dist_bins=int(data["sl_min_dist_bins"]),
            sl_max_dist_bins=int(data["sl_max_dist_bins"]),
            pass_gate2_bonus=float(data["pass_gate2_bonus"]),
            zone_quality_bonus=float(data["zone_quality_bonus"]),
            zone_quality_penalty=float(data["zone_quality_penalty"]),
            sl_valid_bonus=float(data["sl_valid_bonus"]),
            sl_valid_penalty=float(data["sl_valid_penalty"]),
            trade_fee_bins=float(data["trade_fee_bins"]),
            buff_action=float(data["buff_action"]),
            pure_outcome_mode=bool(data.get("pure_outcome_mode", False)),
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")