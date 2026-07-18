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
    "pass_gate2_bonus",
    "zone_quality_bonus",
    "trade_fee_bins",     # phí giao dịch kiểu spread — CHỈ áp cho BUY/SELL thật (không CANCEL/WAIT/HOLD)
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
    trade_fee_bins: float   # phí cố định (bin), quy đổi ra R-multiple theo risk CỦA CHÍNH lệnh đó — vd 0.5

    def __post_init__(self) -> None:
        max_w = max(
            (w for actions in self.weight_table.values() for w in actions.values()),
            default=0.0,
        )
        if self.pass_gate2_bonus <= max_w:
            raise ValueError(
                f"pass_gate2_bonus (K={self.pass_gate2_bonus}) phải LỚN HƠN max weight "
                f"trong weight_table ({max_w}) của round {self.round_id!r} — nếu không, "
                f"1 completion LOSS (r_multiple=-1) có thể có reward THẤP HƠN 1 completion "
                f"fail semantic, phá vỡ gate cứng."
            )
        if self.zone_quality_bonus < 0:
            raise ValueError(f"zone_quality_bonus phải >= 0, nhận {self.zone_quality_bonus}.")
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
            trade_fee_bins=float(data["trade_fee_bins"]),
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")