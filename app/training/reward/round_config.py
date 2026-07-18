from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

# =====================================================================
# 1 round GRPO = 1 RoundConfig, tường minh và CỐ ĐỊNH cho tới hết round
# (Colab có thể ngắt/chạy lại nhiều lần trong CÙNG 1 round — file này là
# nguồn duy nhất, không phụ thuộc trạng thái Python session nào).
#
# CHỈ áp dụng cho GRPO (app/training/reward/*). Generator (data cho
# pretrain/SFT, app/data_prepare/generator.py) và SemanticChecker khi
# gọi không tham số vẫn giữ nguyên hardcode cũ — round config KHÔNG ảnh
# hưởng 2 giai đoạn train đầu, vì lúc đó chỉ cần đúng format, chưa có
# outcome thật để biết nên nới/siết zone/SL range thế nào.
#
# KHÔNG có fallback ngầm: thiếu file, hoặc thiếu bất kỳ field bắt buộc
# nào trong file, đều raise ngay — không âm thầm dùng giá trị mặc định
# nào khác (kể cả 5/20, 5/10 đang là default của SemanticChecker/
# forward_test.py cho mục đích KHÁC — generator/demo).
# =====================================================================
_REQUIRED_KEYS = (
    "round_id",
    "weight_table",
    "zone_width_min_bins",
    "zone_width_max_bins",
    "sl_min_dist_bins",
    "sl_max_dist_bins",
    "pass_gate2_bonus",      # K — mới
    "zone_quality_bonus",    # mới
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
    pass_gate2_bonus: float      # K — sàn cộng thêm khi pass gate 2, ÁP DỤNG MỌI action
    zone_quality_bonus: float    # cộng thêm nếu probe_zone_quality thắng (WAIT/CANCEL/BUY/SELL)
    trade_fee_bins: float   # phí cố định (bin), quy đổi ra R-multiple theo risk CỦA CHÍNH lệnh đó — vd 0.5

    def __post_init__(self) -> None:
        """
        Bất biến bắt buộc: K PHẢI lớn hơn max weight trong weight_table
        của chính round này. Nếu không, 1 completion pass hết well-form +
        semantic nhưng LOSS (r_multiple=-1, nặng nhất có thể) sẽ có reward
        THẤP HƠN 1 completion fail semantic nhẹ — phá vỡ đúng nguyên tắc
        gate cứng (spec mục 5.1: pass gate 2 phải luôn tốt hơn fail gate 2,
        bất kể outcome ở gate 3 tệ tới đâu).

        reward tệ nhất khi pass gate 2 (BUY/SELL LOSS, zone_bonus=0):
            R_WF_FULL + R_SEM_FULL + K + 0 + (-1 * w) = 2.0 + K - w
        reward tệ nhất khi fail gate 2 (0 vi phạm còn lại → fail nhẹ nhất):
            R_WF_FULL + (R_SEM_FULL - VIOLATION_PENALTY) = 1.0 + 0.8 = 1.8
        Cần: 2.0 + K - w > 1.8  <=>  K > w - 0.2. Chọn điều kiện CHẶT hơn
        (an toàn hơn, không phụ thuộc số 0.2 hardcode ở SemanticChecker):
        K > w  — tự động thoả điều kiện trên với mọi VIOLATION_PENALTY >= 0.
        """
        max_w = max(
            (w for actions in self.weight_table.values() for w in actions.values()),
            default=0.0,
        )
        if self.pass_gate2_bonus <= max_w:
            raise ValueError(
                f"pass_gate2_bonus (K={self.pass_gate2_bonus}) phải LỚN HƠN max weight "
                f"trong weight_table ({max_w}) của round {self.round_id!r} — nếu không, "
                f"1 completion LOSS (r_multiple=-1) có thể có reward THẤP HƠN 1 completion "
                f"fail semantic, phá vỡ gate cứng (xem docstring __post_init__)."
            )
        if self.zone_quality_bonus < 0:
            raise ValueError(
                f"zone_quality_bonus phải >= 0 (chỉ CỘNG khi probe thắng, không bao giờ trừ) "
                f"— nhận {self.zone_quality_bonus}."
            )
        if self.trade_fee_bins < 0:
            raise ValueError(f"trade_fee_bins phải >= 0 (phí, không phải bonus), nhận {self.trade_fee_bins}.")

    @classmethod
    def load(cls, path: str) -> "RoundConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Không tìm thấy round config tại {path!r}. Mỗi round GRPO PHẢI có "
                f"config tường minh (zone/SL range + weight_table + K + zone_quality_bonus) "
                f"trước khi train — không có giá trị mặc định ngầm cho GRPO."
            )
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Round config tại {path!r} không phải JSON hợp lệ: {e}") from e

        missing = [k for k in _REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(
                f"Round config tại {path!r} THIẾU field bắt buộc: {missing}. "
                f"Yêu cầu đủ {list(_REQUIRED_KEYS)} — không cho phép fallback ngầm."
            )

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

    @classmethod
    def new_default(cls, round_id: str) -> "RoundConfig":
        return cls(
            round_id=round_id,
            weight_table={},
            zone_width_min_bins=5,
            zone_width_max_bins=20,
            sl_min_dist_bins=5,
            sl_max_dist_bins=10,
            pass_gate2_bonus=1.5,
            zone_quality_bonus=0.5,
        )