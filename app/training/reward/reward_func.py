from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.lang.parser import Parser
from app.lang.semantic import SemanticChecker
from app.training.reward.forward_test import FutureCandle, evaluate_outcome

# =====================================================================
# Trọng số giữa 3 nhóm (well-form / semantic / outcome) — PLACEHOLDER.
# Đây chính là câu hỏi còn mở #5 trong spec mục 9: "quyết định sau, còn
# quá sớm để chốt số cụ thể trước khi có dữ liệu thực nghiệm từ các
# round GRPO đầu". Đổi 2 hằng số này khi có đủ dữ liệu, không cần sửa
# logic gate bên dưới.
# =====================================================================
R_WF_FULL = 1.0
R_SEM_FULL = 1.0

# Phạt bổ sung khi is_sl_valid/target-saturation fail nhưng SemanticChecker
# (bảng A/B/D/E) đã pass riêng — coi như tương đương 1 vi phạm semantic
# thông thường (cùng độ lớn với SemanticChecker.VIOLATION_PENALTY).
EXTRA_SEMANTIC_PENALTY = SemanticChecker.VIOLATION_PENALTY

DEFAULT_WEIGHT = 1.0


# =====================================================================
# WeightTable — w[trend][action_type], khởi tạo 1.0 mọi ô. Đây là 1 dict
# global/module-level (singleton `weight_table` bên dưới) — sửa tay giữa
# các round GRPO (đọc từ StatsCollector), KHÔNG cần sửa code reward_func.
# Vai trò theo round (xem spec mục 5.3):
#   - Round đầu (entropy/temperature cao): giữ ~1.0 mọi ô, mục tiêu quan
#     sát xu hướng tự nhiên của model.
#   - Round giữa: kéo nhánh bị bỏ quên lên / dìm nhánh bị lạm dụng.
#   - Round cuối: anneal dần về 1.0 (hoặc phản ánh đúng chất lượng thật)
#     — KHÔNG ép cân bằng cứng mãi mãi.
# =====================================================================
class WeightTable:
    def __init__(self, default: float = DEFAULT_WEIGHT):
        self._default = default
        self._table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: self._default))

    def get(self, trend: Optional[str], action_type: Optional[str]) -> float:
        if trend is None or action_type is None:
            return self._default
        return self._table[trend][action_type]

    def set(self, trend: str, action_type: str, weight: float) -> None:
        self._table[trend][action_type] = weight

    def set_many(self, updates: Dict[str, Dict[str, float]]) -> None:
        for trend, actions in updates.items():
            for action_type, w in actions.items():
                self.set(trend, action_type, w)

    def reset(self) -> None:
        self._table.clear()

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {t: dict(a) for t, a in self._table.items()}


weight_table = WeightTable()   # singleton — import và sửa tay giữa các round


# =====================================================================
# StatsCollector — log rollout theo round để tính stat[trend][action]
# (tần suất trong nhánh trend + outcome trung bình đi kèm). Đây là dữ
# liệu bạn nhìn để tự tay chỉnh weight_table (không tự động hoá quyết
# định weight — chỉ tự động hoá việc THU THẬP thống kê).
# =====================================================================
@dataclass
class RolloutRecord:
    trend: Optional[str]
    action_type: Optional[str]
    outcome_status: Optional[str]     # "WIN" | "LOSS" | "TIMEOUT" | "INVALID_SETUP" | None
    r_multiple: Optional[float]
    well_formed: bool
    semantic_passed: bool


class StatsCollector:
    def __init__(self):
        self._records: List[RolloutRecord] = []

    def log(self, record: RolloutRecord) -> None:
        self._records.append(record)

    def reset(self) -> None:
        self._records.clear()

    def summary(self) -> Dict[str, Dict[str, dict]]:
        """
        trend -> action_type -> {count, freq_within_trend, avg_r_multiple, win_rate}

        Chỉ tính trên record đã pass CẢ well-form lẫn semantic — vì chỉ
        những case đó outcome mới đáng tin để đưa vào thống kê (case fail
        gate 1/2 không có outcome thật, đưa vào sẽ làm méo win_rate/avg_R).
        """
        by_trend_total: Dict[str, int] = defaultdict(int)
        raw: Dict[str, Dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {"count": 0, "r_multiples": []}))

        for r in self._records:
            if r.trend is None or r.action_type is None:
                continue
            if not (r.well_formed and r.semantic_passed):
                continue
            by_trend_total[r.trend] += 1
            entry = raw[r.trend][r.action_type]
            entry["count"] += 1
            if r.r_multiple is not None:
                entry["r_multiples"].append(r.r_multiple)

        result: Dict[str, Dict[str, dict]] = {}
        for trend, actions in raw.items():
            result[trend] = {}
            total = by_trend_total[trend]
            for action_type, entry in actions.items():
                rms = entry["r_multiples"]
                avg_r = sum(rms) / len(rms) if rms else None
                win_rate = (sum(1 for x in rms if x > 0) / len(rms)) if rms else None
                result[trend][action_type] = {
                    "count": entry["count"],
                    "freq_within_trend": entry["count"] / total if total else 0.0,
                    "avg_r_multiple": avg_r,
                    "win_rate": win_rate,
                }
        return result

    def print_summary(self) -> None:
        summary = self.summary()
        print("=== Rollout stats (trend -> action) ===")
        for trend, actions in summary.items():
            print(f"trend={trend}")
            for action_type, stat in actions.items():
                avg_r = f"{stat['avg_r_multiple']:.2f}" if stat["avg_r_multiple"] is not None else "-"
                win_rate = f"{stat['win_rate'] * 100:.0f}%" if stat["win_rate"] is not None else "-"
                print(
                    f"  {action_type:<12} count={stat['count']:<4} "
                    f"freq={stat['freq_within_trend'] * 100:5.1f}%  avg_R={avg_r:>6}  win_rate={win_rate}"
                )


stats_collector = StatsCollector()   # singleton — reset() giữa các round nếu muốn thống kê rolling


# =====================================================================
# score_completion — chấm 1 completion theo gate tuần tự 3 tầng (mục 5.1).
# KHÔNG cộng tuyến tính: mỗi gate fail thì DỪNG NGAY, không tính điểm
# gate sau (kể cả khi outcome giả định sẽ tốt).
# =====================================================================
def score_completion(
    completion: str,
    future_bins: Sequence[Sequence[int]],
    stats: Optional[StatsCollector] = None,
    weights: Optional[WeightTable] = None,
) -> float:
    weights = weights if weights is not None else weight_table
    future_candles: List[FutureCandle] = [tuple(c) for c in future_bins]  # type: ignore[misc]

    parse_result = Parser.from_text(completion).parse()

    # --- Gate 1: well-form ---
    if not parse_result.is_well_formed():
        if stats is not None:
            program = parse_result.ast
            stats.log(RolloutRecord(
                trend=program.think.trend if program and program.think else None,
                action_type=program.action.action_type if program and program.action else None,
                outcome_status=None, r_multiple=None,
                well_formed=False, semantic_passed=False,
            ))
        return parse_result.well_form_score()

    program = parse_result.ast
    trend = program.think.trend
    action_type = program.action.action_type

    # --- Gate 2: semantic (bảng A/B/D/E) + ràng buộc SL/target bổ sung ---
    semantic_result = SemanticChecker().check(program)
    extra_valid, forward_result = evaluate_outcome(program.action, program.think, future_candles)
    overall_semantic_passed = semantic_result.passed and extra_valid

    if not overall_semantic_passed:
        sem_score = semantic_result.score
        if semantic_result.passed and not extra_valid:
            # SemanticChecker tự nó pass, nhưng vi phạm SL-distance/target-saturation
            # (mục 6.1) — cộng thêm đúng 1 mức phạt tương đương 1 vi phạm thông thường.
            sem_score = max(0.0, sem_score - EXTRA_SEMANTIC_PENALTY)
        if stats is not None:
            stats.log(RolloutRecord(
                trend=trend, action_type=action_type, outcome_status=None, r_multiple=None,
                well_formed=True, semantic_passed=False,
            ))
        return R_WF_FULL + sem_score

    # --- Gate 3: outcome (chỉ áp dụng cho BUY/SELL/CANCEL_BUY/CANCEL_SELL) ---
    if forward_result is None:
        # WAIT_BUY/WAIT_SELL/HOLD — không có outcome để chấm.
        if stats is not None:
            stats.log(RolloutRecord(
                trend=trend, action_type=action_type, outcome_status=None, r_multiple=None,
                well_formed=True, semantic_passed=True,
            ))
        return R_WF_FULL + R_SEM_FULL

    w = weights.get(trend, action_type)
    reward = R_WF_FULL + R_SEM_FULL + forward_result.r_multiple * w

    if stats is not None:
        stats.log(RolloutRecord(
            trend=trend, action_type=action_type,
            outcome_status=forward_result.status.value, r_multiple=forward_result.r_multiple,
            well_formed=True, semantic_passed=True,
        ))

    return reward


# =====================================================================
# unified_reward_func — entry point trực tiếp cho TRL:
#     GRPOTrainer(..., reward_funcs=unified_reward_func)
#
# QUAN TRỌNG (mục 8.1 spec): đây PHẢI là 1 hàm reward_func DUY NHẤT.
# Không tách well-form/semantic/outcome thành 3 reward_funcs riêng kèm
# reward_weights — TRL cộng dồn (tổng có trọng số) các reward_funcs, đó
# là cộng tuyến tính, mâu thuẫn với gate cứng đã thiết kế ở đây.
#
# Dataset GRPO cần cột `future_bins` (list [[o,h,l,c],...] x 50 nến) và
# PHẢI set `remove_unused_columns=False` trong GRPOConfig, nếu không
# TRL sẽ tự xoá cột này trước khi tới reward_func.
# =====================================================================
def unified_reward_func(
    prompts: Sequence[Any],
    completions: Sequence[str],
    future_bins: Sequence[Sequence[Sequence[int]]],
    **kwargs: Any,
) -> List[float]:
    return [
        score_completion(completion, fb, stats=stats_collector, weights=weight_table)
        for completion, fb in zip(completions, future_bins)
    ]