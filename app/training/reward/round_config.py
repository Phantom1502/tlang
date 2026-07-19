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
    "zone_quality_penalty",  # MỚI — phạt khi probe_zone_quality LOSS, symmetric với bonus
    "trade_fee_bins",
)


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
    zone_quality_penalty: float   # MỚI — trừ khi probe LOSS (WIN/LOSS symmetric, TIMEOUT trung tính)
    trade_fee_bins: float

    def __post_init__(self) -> None:
        """
        Bất biến bắt buộc: K PHẢI lớn hơn (zone_quality_penalty + max weight
        trong weight_table của chính round này). Nếu không, 1 completion pass
        hết well-form + semantic nhưng rơi vào worst-case ở gate 3 (zone LOSS
        VÀ timing LOSS, r_multiple=-1) sẽ có reward THẤP HƠN 1 completion fail
        semantic nhẹ — phá vỡ đúng nguyên tắc gate cứng (spec mục 5.1: pass
        gate 2 phải luôn tốt hơn fail gate 2, bất kể outcome ở gate 3 tệ tới đâu).

        reward tệ nhất khi pass gate 2 (BUY/SELL LOSS, zone LOSS):
            R_WF_FULL + R_SEM_FULL + K + (-zone_quality_penalty) + (-1 * w)
            = 2.0 + K - zone_quality_penalty - w
        reward tệ nhất khi fail gate 2 (0 vi phạm còn lại -> fail nhẹ nhất):
            R_WF_FULL + (R_SEM_FULL - VIOLATION_PENALTY) = 1.0 + 0.8 = 1.8
        Cần: 2.0 + K - zone_quality_penalty - w > 1.8
             <=> K > zone_quality_penalty + w - 0.2
        Chọn điều kiện CHẶT hơn (an toàn hơn, không phụ thuộc số 0.2 hardcode
        ở SemanticChecker): K > zone_quality_penalty + max_w — tự động thoả
        điều kiện trên với mọi VIOLATION_PENALTY >= 0.
        """
        max_w = max(
            (w for actions in self.weight_table.values() for w in actions.values()),
            default=0.0,
        )
        if self.pass_gate2_bonus <= self.zone_quality_penalty + max_w:
            raise ValueError(
                f"pass_gate2_bonus (K={self.pass_gate2_bonus}) phải LỚN HƠN "
                f"zone_quality_penalty ({self.zone_quality_penalty}) + max weight "
                f"trong weight_table ({max_w}) của round {self.round_id!r} — nếu không, "
                f"1 completion pass gate 2 nhưng worst-case (zone LOSS + timing LOSS) có thể "
                f"có reward THẤP HƠN 1 completion fail semantic nhẹ, phá vỡ gate cứng "
                f"(xem docstring __post_init__)."
            )
        if self.zone_quality_bonus < 0:
            raise ValueError(
                f"zone_quality_bonus phải >= 0 (chỉ CỘNG khi probe thắng, không bao giờ trừ) "
                f"— nhận {self.zone_quality_bonus}."
            )
        if self.zone_quality_penalty < 0:
            raise ValueError(
                f"zone_quality_penalty phải >= 0 (độ lớn phạt, dấu trừ áp ở reward_func.py, "
                f"không tự mang dấu âm ở đây) — nhận {self.zone_quality_penalty}."
            )
        if self.trade_fee_bins < 0:
            raise ValueError(f"trade_fee_bins phải >= 0 (phí, không phải bonus), nhận {self.trade_fee_bins}.")

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
            trade_fee_bins=float(data["trade_fee_bins"]),
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")