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
)


@dataclass
class RoundConfig:
    round_id: str
    weight_table: Dict[str, Dict[str, float]]   # trend -> action_type -> weight
    zone_width_min_bins: int
    zone_width_max_bins: int
    sl_min_dist_bins: int
    sl_max_dist_bins: int

    @classmethod
    def load(cls, path: str) -> "RoundConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Không tìm thấy round config tại {path!r}. Mỗi round GRPO PHẢI có "
                f"config tường minh (zone/SL range + weight_table) trước khi train — "
                f"không có giá trị mặc định ngầm cho GRPO (khác generator/demo)."
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
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def new_default(cls, round_id: str) -> "RoundConfig":
        """Tiện ích tạo 1 bản khởi điểm (vd cho round 1) — dùng giá trị
        đang hardcode ở SemanticChecker/forward_test.py làm điểm xuất
        phát, rồi bạn tay chỉnh file JSON từ đây trở đi. KHÔNG dùng hàm
        này thay cho load() trong code train — chỉ để bootstrap/CLI."""
        return cls(
            round_id=round_id,
            weight_table={},
            zone_width_min_bins=5,
            zone_width_max_bins=20,
            sl_min_dist_bins=5,
            sl_max_dist_bins=10,
        )