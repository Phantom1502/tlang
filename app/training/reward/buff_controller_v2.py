from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

from app.training.reward.round_config_v2 import RoundConfigV2

# =====================================================================
# EMABuffControllerV2 — THAY THẾ HOÀN TOÀN cách buff bị trộn giữa raw-score
# và if/elif rải rác của v1 (xem thảo luận thiết kế). Nguyên tắc:
#
#   - Tầng 1 (raw score, forward_test.py + reward_func_v2.py:compute_*)
#     KHÔNG BAO GIỜ gọi tới class này — hoàn toàn không biết buff tồn tại.
#   - Tầng 2 (class này) chỉ làm 1 việc: giữ 1 con số buff/nhóm, cập nhật
#     theo PD-controller mỗi optimizer step, và trả buff khi được hỏi.
#   - Tầng 2 KHÔNG tự đếm sample nữa (khác EMABuffController v1 — record()
#     đã bị XOÁ HẲN). counts/total luôn do caller cung cấp ở on_step_end(),
#     lấy từ StatsCollectorV2.counts_since_step_boundary() — 1 NGUỒN SỰ
#     THẬT DUY NHẤT cho "mẫu nào eligible tính vào tỉ lệ", dùng lại được
#     cho CẢ report lẫn buff, không có 2 bộ đếm trùng lặp logic.
#
# 1 instance của class này CHỈ nên phục vụ 1 namespace duy nhất
# ("action" 7 nhóm HOLD/BUY/SELL/CANCEL_BUY/CANCEL_SELL/WAIT_BUY/WAIT_SELL,
# hoặc "zone" 2 nhóm HAS_ZONE/NO_ZONE) — 2 vòng điều khiển ĐỘC LẬP hoàn
# toàn, mỗi cái có PD-params + target riêng lấy qua RoundConfigV2 dispatch
# theo namespace (xem group_target/group_range/group_init/group_ema_alpha/
# group_pd_params trong round_config_v2.py).
# =====================================================================


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class GroupBuffState:
    ema_ratio: float
    buff: float
    prev_error: float = 0.0


class EMABuffControllerV2:
    def __init__(self, groups: Sequence[str], namespace: str):
        self.groups = tuple(groups)
        self.namespace = namespace
        self.states: Dict[str, GroupBuffState] = {}

    def seed_from_round_config(self, round_config: RoundConfigV2) -> None:
        """Dùng khi round MỚI bắt đầu (không có state cũ để resume) — seed
        ema_ratio = target (tránh vài trăm step đầu phản ứng nhầm hướng do
        ema=0), buff = group_init, prev_error = 0.0 — cùng triết lý seed
        với EMABuffController v1."""
        for group in self.groups:
            self.states[group] = GroupBuffState(
                ema_ratio=round_config.group_target(self.namespace, group),
                buff=round_config.group_init(self.namespace, group),
                prev_error=0.0,
            )

    def on_step_end(self, round_config: RoundConfigV2, counts: Dict[str, int], total: int) -> None:
        """counts/total PHẢI được caller tính từ StatsCollectorV2 (đếm số
        mẫu eligible kể từ watermark step trước) — class này không tự đếm
        gì cả. Nếu total==0 (step này không có mẫu eligible nào cho
        namespace hiện tại) -> bỏ qua, giữ nguyên state cũ, tránh update
        dựa trên rate=0/0 vô nghĩa."""
        if total == 0:
            return
        alpha = round_config.group_ema_alpha(self.namespace)
        kp, kd, step_max = round_config.group_pd_params(self.namespace)

        for group in self.groups:
            rate_this_step = counts.get(group, 0) / total
            st = self.states[group]
            st.ema_ratio = (1.0 - alpha) * rate_this_step + alpha * st.ema_ratio

            lo, hi = round_config.group_range(self.namespace, group)
            target = round_config.group_target(self.namespace, group)
            error = target - st.ema_ratio
            d_error = error - st.prev_error
            st.prev_error = error

            delta = kp * error + kd * d_error
            delta = _clip(delta, -step_max, step_max)
            st.buff = _clip(st.buff + delta, lo, hi)

    def get_buff(self, key: Optional[str]) -> float:
        if key is None or key not in self.states:
            return 0.0
        return self.states[key].buff

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            g: {"ema_ratio": s.ema_ratio, "buff": s.buff, "prev_error": s.prev_error}
            for g, s in self.states.items()
        }

    def state_dict(self) -> Dict[str, Dict[str, float]]:
        return self.snapshot()

    def load_state_dict(self, data: Dict[str, Dict[str, float]]) -> None:
        for group, d in data.items():
            self.states[group] = GroupBuffState(
                ema_ratio=float(d["ema_ratio"]),
                buff=float(d["buff"]),
                prev_error=float(d.get("prev_error", 0.0)),
            )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.state_dict(), ensure_ascii=False), encoding="utf-8")

    def load(self, path: str) -> bool:
        """Trả True nếu load thành công. Caller PHẢI gọi
        seed_from_round_config() khi trả về False — KHÔNG được để states
        rỗng (get_buff sẽ âm thầm trả 0.0 cho group thiếu)."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.load_state_dict(data)
            return True
        except Exception:
            return False