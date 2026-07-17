#!/usr/bin/env python3
"""
fix_round_config_wiring.py — Nối RoundConfig (zone/SL range + weight_table,
tường minh + cố định trong 1 round GRPO) vào SemanticChecker/forward_test.py/
reward_func.py. Cộng thêm StatsCollector.save()/load() để resume nhiều lần
trên 1 round (Colab session bị ngắt giữa chừng) không mất thống kê.

Chạy SAU migrate.sh + fix_zone_width.py (semantic.py phải đã có
ZONE_WIDTH_MIN_BINS/MAX_BINS + _check_zone_width như patch trước để lại).

    python3 fix_round_config_wiring.py --check
    python3 fix_round_config_wiring.py
    git diff app/lang/semantic.py app/training/reward/forward_test.py app/training/reward/reward_func.py

QUAN TRỌNG: generator.py / demos gọi SemanticChecker() hay evaluate_outcome()
KHÔNG truyền tham số mới vẫn chạy y hệt trước (default = giá trị hardcode cũ)
— round config CHỈ ảnh hưởng khi GRPO chủ động gọi
set_active_round_config(RoundConfig.load(path)) trước khi train.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEMANTIC_PY = ROOT / "app/lang/semantic.py"
FORWARD_TEST_PY = ROOT / "app/training/reward/forward_test.py"
REWARD_FUNC_PY = ROOT / "app/training/reward/reward_func.py"


def _replace_exact(path: Path, old: str, new: str, label: str, check_only: bool) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        print(f"  [SKIP] {label}: không tìm thấy đoạn text mong đợi trong {path} — kiểm tra tay.")
        return
    if check_only:
        print(f"  [OK-CHECK] {label}")
        return
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"  [PATCHED] {label}")


def patch_semantic(check_only: bool) -> None:
    print("=== 1) app/lang/semantic.py: SemanticChecker nhận zone bounds qua __init__ ===")
    _replace_exact(
        SEMANTIC_PY,
        '    BUY_SIDE_ACTIONS = {"BUY", "CANCEL_BUY"}\n'
        '    SELL_SIDE_ACTIONS = {"SELL", "CANCEL_SELL"}\n\n'
        '    def check(self, program: ProgramNode) -> SemanticResult:',
        '    BUY_SIDE_ACTIONS = {"BUY", "CANCEL_BUY"}\n'
        '    SELL_SIDE_ACTIONS = {"SELL", "CANCEL_SELL"}\n\n'
        '    def __init__(\n'
        '        self,\n'
        '        zone_width_min_bins: int = ZONE_WIDTH_MIN_BINS,\n'
        '        zone_width_max_bins: int = ZONE_WIDTH_MAX_BINS,\n'
        '    ) -> None:\n'
        '        """\n'
        '        Default = class constant (5/20) — KHÔNG đổi gì cho generator.py/demo\n'
        '        (SemanticChecker() không tham số vẫn y hệt trước). Chỉ nhánh GRPO\n'
        '        (app/training/reward/reward_func.py) truyền tường minh 2 giá trị này\n'
        '        từ RoundConfig của round hiện tại (app/training/reward/round_config.py)\n'
        '        — zone range CHỈ được phép chỉnh ở GRPO, nơi outcome thật mới cho biết\n'
        '        nên nới/siết thế nào; generator (data pretrain/SFT) giữ hardcode vì chỉ\n'
        '        cần đúng format.\n'
        '        """\n'
        '        self.zone_width_min_bins = zone_width_min_bins\n'
        '        self.zone_width_max_bins = zone_width_max_bins\n\n'
        '    def check(self, program: ProgramNode) -> SemanticResult:',
        "thêm __init__(zone_width_min_bins, zone_width_max_bins)",
        check_only,
    )
    _replace_exact(
        SEMANTIC_PY,
        '    def _check_zone_width(self, think: ThinkNode, violations: List[str]) -> None:\n'
        '        zone = think.zone\n'
        '        if zone is None:\n'
        '            return\n'
        '        width = zone.upper_bin - zone.lower_bin\n'
        '        if not (self.ZONE_WIDTH_MIN_BINS <= width <= self.ZONE_WIDTH_MAX_BINS):\n'
        '            violations.append(\n'
        '                f"zone={zone.direction} ({zone.lower_bin}:{zone.upper_bin}) có width={width} bin, "\n'
        '                f"ngoài phạm vi hợp lệ [{self.ZONE_WIDTH_MIN_BINS},{self.ZONE_WIDTH_MAX_BINS}]"\n'
        '            )',
        '    def _check_zone_width(self, think: ThinkNode, violations: List[str]) -> None:\n'
        '        zone = think.zone\n'
        '        if zone is None:\n'
        '            return\n'
        '        width = zone.upper_bin - zone.lower_bin\n'
        '        if not (self.zone_width_min_bins <= width <= self.zone_width_max_bins):\n'
        '            violations.append(\n'
        '                f"zone={zone.direction} ({zone.lower_bin}:{zone.upper_bin}) có width={width} bin, "\n'
        '                f"ngoài phạm vi hợp lệ [{self.zone_width_min_bins},{self.zone_width_max_bins}]"\n'
        '            )',
        "_check_zone_width dùng self.zone_width_min/max_bins (instance) thay vì class constant",
        check_only,
    )


def patch_forward_test(check_only: bool) -> None:
    print("\n=== 2) app/training/reward/forward_test.py: is_sl_valid/evaluate_outcome nhận SL bounds qua tham số ===")
    _replace_exact(
        FORWARD_TEST_PY,
        'def is_sl_valid(action_type: str, entry_bin: int, sl_bin: int, zone: ZoneNode) -> bool:\n'
        '    dist = abs(entry_bin - sl_bin)\n'
        '    if not (SL_MIN_DIST_BINS <= dist <= SL_MAX_DIST_BINS):\n'
        '        return False',
        'def is_sl_valid(\n'
        '    action_type: str, entry_bin: int, sl_bin: int, zone: ZoneNode,\n'
        '    sl_min_dist_bins: int = SL_MIN_DIST_BINS,\n'
        '    sl_max_dist_bins: int = SL_MAX_DIST_BINS,\n'
        ') -> bool:\n'
        '    """Default = module constant (5/10) — dùng cho generator.py/demo không\n'
        '    đổi gì. GRPO (reward_func.py) truyền tường minh từ RoundConfig hiện tại."""\n'
        '    dist = abs(entry_bin - sl_bin)\n'
        '    if not (sl_min_dist_bins <= dist <= sl_max_dist_bins):\n'
        '        return False',
        "is_sl_valid nhận sl_min_dist_bins/sl_max_dist_bins",
        check_only,
    )
    _replace_exact(
        FORWARD_TEST_PY,
        'def evaluate_outcome(\n'
        '    action: ActionNode,\n'
        '    think: ThinkNode,\n'
        '    future_candles: List[FutureCandle],\n'
        ') -> Tuple[bool, Optional[ForwardTestResult]]:\n'
        '    action_type = action.action_type\n\n'
        '    if action_type in ("WAIT_BUY", "WAIT_SELL", "HOLD"):\n'
        '        return True, None\n\n'
        '    if action_type in ("BUY", "SELL"):\n'
        '        if think.zone is None or action.sl is None or action.rr is None:\n'
        '            return False, None\n'
        '        if not is_sl_valid(action_type, think.current_price_bin, action.sl, think.zone):\n'
        '            return False, None',
        'def evaluate_outcome(\n'
        '    action: ActionNode,\n'
        '    think: ThinkNode,\n'
        '    future_candles: List[FutureCandle],\n'
        '    sl_min_dist_bins: int = SL_MIN_DIST_BINS,\n'
        '    sl_max_dist_bins: int = SL_MAX_DIST_BINS,\n'
        ') -> Tuple[bool, Optional[ForwardTestResult]]:\n'
        '    """sl_min_dist_bins/sl_max_dist_bins: default = module constant (5/10),\n'
        '    dùng cho generator.py/demo không đổi gì. GRPO (reward_func.py) truyền\n'
        '    tường minh từ RoundConfig hiện tại (app/training/reward/round_config.py)."""\n'
        '    action_type = action.action_type\n\n'
        '    if action_type in ("WAIT_BUY", "WAIT_SELL", "HOLD"):\n'
        '        return True, None\n\n'
        '    if action_type in ("BUY", "SELL"):\n'
        '        if think.zone is None or action.sl is None or action.rr is None:\n'
        '            return False, None\n'
        '        if not is_sl_valid(\n'
        '            action_type, think.current_price_bin, action.sl, think.zone,\n'
        '            sl_min_dist_bins, sl_max_dist_bins,\n'
        '        ):\n'
        '            return False, None',
        "evaluate_outcome nhận + thread sl_min_dist_bins/sl_max_dist_bins vào is_sl_valid",
        check_only,
    )


def patch_reward_func(check_only: bool) -> None:
    print("\n=== 3) app/training/reward/reward_func.py: RoundConfig singleton (fail-loud) + StatsCollector persistence ===")

    _replace_exact(
        REWARD_FUNC_PY,
        'from __future__ import annotations\n\n'
        'from collections import defaultdict\n'
        'from dataclasses import dataclass, field\n'
        'from typing import Any, Dict, List, Optional, Sequence, Tuple\n\n'
        'from app.lang.parser import Parser\n'
        'from app.lang.semantic import SemanticChecker\n'
        'from app.training.reward.forward_test import FutureCandle, evaluate_outcome',
        'from __future__ import annotations\n\n'
        'import json\n'
        'from collections import defaultdict\n'
        'from dataclasses import asdict, dataclass, field\n'
        'from pathlib import Path\n'
        'from typing import Any, Dict, List, Optional, Sequence, Tuple\n\n'
        'from app.lang.parser import Parser\n'
        'from app.lang.semantic import SemanticChecker\n'
        'from app.training.reward.forward_test import FutureCandle, evaluate_outcome\n'
        'from app.training.reward.round_config import RoundConfig',
        "import json/Path/asdict/RoundConfig",
        check_only,
    )

    _replace_exact(
        REWARD_FUNC_PY,
        'weight_table = WeightTable()   # singleton — import và sửa tay giữa các round',
        'weight_table = WeightTable()   # singleton — import và sửa tay giữa các round\n\n\n'
        '# =====================================================================\n'
        '# RoundConfig singleton — mỗi round GRPO PHẢI load 1 RoundConfig tường\n'
        '# minh (zone_width/sl_dist range + weight_table) TRƯỚC khi train, cố\n'
        '# định cho tới hết round (không fallback ngầm — xem round_config.py).\n'
        '# =====================================================================\n'
        '_active_round_config: Optional[RoundConfig] = None\n\n\n'
        'def set_active_round_config(config: RoundConfig) -> None:\n'
        '    """Gọi 1 lần lúc khởi động train_grpo.py — mọi rank/process load CÙNG\n'
        '    1 file config nên tự nhiên đồng bộ, không cần cơ chế broadcast riêng.\n'
        '    Đồng bộ luôn weight_table singleton từ config.weight_table."""\n'
        '    global _active_round_config\n'
        '    _active_round_config = config\n'
        '    weight_table.reset()\n'
        '    weight_table.set_many(config.weight_table)\n\n\n'
        'def get_active_round_config() -> RoundConfig:\n'
        '    if _active_round_config is None:\n'
        '        raise RuntimeError(\n'
        '            "Chưa load RoundConfig cho round hiện tại — gọi "\n'
        '            "set_active_round_config(RoundConfig.load(path)) TRƯỚC khi train GRPO. "\n'
        '            "Zone/SL range KHÔNG có giá trị mặc định ngầm ở đây (xem round_config.py)."\n'
        '        )\n'
        '    return _active_round_config',
        "thêm RoundConfig singleton + set/get_active_round_config (fail-loud)",
        check_only,
    )

    _replace_exact(
        REWARD_FUNC_PY,
        '    def print_summary(self) -> None:\n'
        '        summary = self.summary()\n'
        '        print("=== Rollout stats (trend -> action) ===")\n'
        '        for trend, actions in summary.items():\n'
        '            print(f"trend={trend}")\n'
        '            for action_type, stat in actions.items():\n'
        '                avg_r = f"{stat[\'avg_r_multiple\']:.2f}" if stat["avg_r_multiple"] is not None else "-"\n'
        '                win_rate = f"{stat[\'win_rate\'] * 100:.0f}%" if stat["win_rate"] is not None else "-"\n'
        '                print(\n'
        '                    f"  {action_type:<12} count={stat[\'count\']:<4} "\n'
        '                    f"freq={stat[\'freq_within_trend\'] * 100:5.1f}%  avg_R={avg_r:>6}  win_rate={win_rate}"\n'
        '                )',
        '    def print_summary(self) -> None:\n'
        '        summary = self.summary()\n'
        '        print("=== Rollout stats (trend -> action) ===")\n'
        '        for trend, actions in summary.items():\n'
        '            print(f"trend={trend}")\n'
        '            for action_type, stat in actions.items():\n'
        '                avg_r = f"{stat[\'avg_r_multiple\']:.2f}" if stat["avg_r_multiple"] is not None else "-"\n'
        '                win_rate = f"{stat[\'win_rate\'] * 100:.0f}%" if stat["win_rate"] is not None else "-"\n'
        '                print(\n'
        '                    f"  {action_type:<12} count={stat[\'count\']:<4} "\n'
        '                    f"freq={stat[\'freq_within_trend\'] * 100:5.1f}%  avg_R={avg_r:>6}  win_rate={win_rate}"\n'
        '                )\n\n'
        '    # ------------------------------------------------------------------\n'
        '    # Persistence — cần thiết vì Colab session có thể bị ngắt/chạy lại\n'
        '    # NHIỀU LẦN trong CÙNG 1 round. Load-rồi-append: mỗi lần khởi động,\n'
        '    # nạp lại records đã dump trước khi log tiếp, để file trên đĩa luôn\n'
        '    # phản ánh TOÀN BỘ round tính đến hiện tại, không chỉ session này.\n'
        '    # ------------------------------------------------------------------\n'
        '    def to_list(self) -> List[Dict[str, Any]]:\n'
        '        return [asdict(r) for r in self._records]\n\n'
        '    def save(self, path: str) -> None:\n'
        '        p = Path(path)\n'
        '        p.parent.mkdir(parents=True, exist_ok=True)\n'
        '        p.write_text(json.dumps(self.to_list(), ensure_ascii=False), encoding="utf-8")\n\n'
        '    @classmethod\n'
        '    def load(cls, path: str) -> "StatsCollector":\n'
        '        """Dùng lúc khởi động 1 session MỚI cho round đang chạy dở (Colab bị\n'
        '        ngắt) — tiếp tục cộng dồn đúng, không mất thống kê các lần chạy trước\n'
        '        trong CÙNG round. File chưa tồn tại (lần đầu của round) -> collector rỗng."""\n'
        '        collector = cls()\n'
        '        p = Path(path)\n'
        '        if p.exists():\n'
        '            for d in json.loads(p.read_text(encoding="utf-8")):\n'
        '                collector.log(RolloutRecord(**d))\n'
        '        return collector\n\n'
        '    @classmethod\n'
        '    def merge_from_files(cls, paths: Sequence[str]) -> "StatsCollector":\n'
        '        """Gộp nhiều file rank-riêng (multi-GPU, mỗi rank tự dump 1 file theo\n'
        '        pattern vd f\'{output_dir}/round{N}_stats_rank{rank}.json\') thành 1\n'
        '        StatsCollector duy nhất — summary() lúc này mới đúng trên TOÀN BỘ\n'
        '        round, không chỉ 1 rank. File không tồn tại (rank chưa log gì) -> bỏ qua."""\n'
        '        collector = cls()\n'
        '        for path in paths:\n'
        '            p = Path(path)\n'
        '            if not p.exists():\n'
        '                continue\n'
        '            for d in json.loads(p.read_text(encoding="utf-8")):\n'
        '                collector.log(RolloutRecord(**d))\n'
        '        return collector',
        "thêm StatsCollector.to_list()/save()/load()/merge_from_files()",
        check_only,
    )

    _replace_exact(
        REWARD_FUNC_PY,
        '    weights = weights if weights is not None else weight_table\n'
        '    future_candles: List[FutureCandle] = [tuple(c) for c in future_bins]  # type: ignore[misc]',
        '    weights = weights if weights is not None else weight_table\n'
        '    round_config = get_active_round_config()   # fail loud nếu chưa set — xem set_active_round_config()\n'
        '    future_candles: List[FutureCandle] = [tuple(c) for c in future_bins]  # type: ignore[misc]',
        "score_completion: lấy round_config hiện tại (fail-loud nếu chưa set)",
        check_only,
    )
    _replace_exact(
        REWARD_FUNC_PY,
        '    # --- Gate 2: semantic (bảng A/B/D/E) + ràng buộc SL/target bổ sung ---\n'
        '    semantic_result = SemanticChecker().check(program)\n'
        '    extra_valid, forward_result = evaluate_outcome(program.action, program.think, future_candles)\n'
        '    overall_semantic_passed = semantic_result.passed and extra_valid',
        '    # --- Gate 2: semantic (bảng A/B/D/E) + ràng buộc SL/target bổ sung ---\n'
        '    # zone_width/sl_dist range LẤY TỪ round_config hiện tại (KHÔNG dùng\n'
        '    # default hardcode của SemanticChecker/evaluate_outcome — round GRPO\n'
        '    # luôn phải tường minh, xem round_config.py).\n'
        '    semantic_result = SemanticChecker(\n'
        '        zone_width_min_bins=round_config.zone_width_min_bins,\n'
        '        zone_width_max_bins=round_config.zone_width_max_bins,\n'
        '    ).check(program)\n'
        '    extra_valid, forward_result = evaluate_outcome(\n'
        '        program.action, program.think, future_candles,\n'
        '        sl_min_dist_bins=round_config.sl_min_dist_bins,\n'
        '        sl_max_dist_bins=round_config.sl_max_dist_bins,\n'
        '    )\n'
        '    overall_semantic_passed = semantic_result.passed and extra_valid',
        "score_completion: SemanticChecker/evaluate_outcome dùng bounds từ round_config",
        check_only,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    for p in (SEMANTIC_PY, FORWARD_TEST_PY, REWARD_FUNC_PY):
        if not p.exists():
            print(f"Không thấy {p}. Chạy script này từ root repo, SAU migrate.sh + fix_zone_width.py.", file=sys.stderr)
            sys.exit(1)

    patch_semantic(args.check)
    patch_forward_test(args.check)
    patch_reward_func(args.check)

    print("\n=== XONG. Review: git diff app/lang/semantic.py app/training/reward/forward_test.py app/training/reward/reward_func.py ===")


if __name__ == "__main__":
    main()