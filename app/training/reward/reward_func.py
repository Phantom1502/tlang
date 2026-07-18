from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.lang.parser import Parser
from app.lang.semantic import SemanticChecker
from app.training.reward.forward_test import FutureCandle, evaluate_outcome
from app.training.reward.round_config import RoundConfig

R_WF_FULL = 1.0
R_SEM_FULL = 1.0

EXTRA_SEMANTIC_PENALTY = SemanticChecker.VIOLATION_PENALTY

DEFAULT_WEIGHT = 1.0


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
# RoundConfig singleton — mỗi round GRPO PHẢI load 1 RoundConfig tường
# minh (zone_width/sl_dist range + weight_table) TRƯỚC khi train, cố
# định cho tới hết round (không fallback ngầm — xem round_config.py).
# =====================================================================
_active_round_config: Optional[RoundConfig] = None


def set_active_round_config(config: RoundConfig) -> None:
    """Gọi 1 lần lúc khởi động train_grpo.py — mọi rank/process load CÙNG
    1 file config nên tự nhiên đồng bộ, không cần cơ chế broadcast riêng.
    Đồng bộ luôn weight_table singleton từ config.weight_table."""
    global _active_round_config
    _active_round_config = config
    weight_table.reset()
    weight_table.set_many(config.weight_table)


def get_active_round_config() -> RoundConfig:
    if _active_round_config is None:
        raise RuntimeError(
            "Chưa load RoundConfig cho round hiện tại — gọi "
            "set_active_round_config(RoundConfig.load(path)) TRƯỚC khi train GRPO. "
            "Zone/SL range KHÔNG có giá trị mặc định ngầm ở đây (xem round_config.py)."
        )
    return _active_round_config


@dataclass
class RolloutRecord:
    trend: Optional[str]
    action_type: Optional[str]
    outcome_status: Optional[str]
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

    # ------------------------------------------------------------------
    # Persistence — cần thiết vì Colab session có thể bị ngắt/chạy lại
    # NHIỀU LẦN trong CÙNG 1 round. Load-rồi-append: mỗi lần khởi động,
    # nạp lại records đã dump trước khi log tiếp, để file trên đĩa luôn
    # phản ánh TOÀN BỘ round tính đến hiện tại, không chỉ session này.
    # ------------------------------------------------------------------
    def to_list(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self._records]

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_list(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "StatsCollector":
        """Dùng lúc khởi động 1 session MỚI cho round đang chạy dở (Colab bị
        ngắt) — tiếp tục cộng dồn đúng, không mất thống kê các lần chạy trước
        trong CÙNG round. File chưa tồn tại (lần đầu của round) -> collector rỗng."""
        collector = cls()
        p = Path(path)
        if p.exists():
            for d in json.loads(p.read_text(encoding="utf-8")):
                collector.log(RolloutRecord(**d))
        return collector

    @classmethod
    def merge_from_files(cls, paths: Sequence[str]) -> "StatsCollector":
        """Gộp nhiều file rank-riêng (multi-GPU, mỗi rank tự dump 1 file theo
        pattern vd f'{output_dir}/round{N}_stats_rank{rank}.json') thành 1
        StatsCollector duy nhất — summary() lúc này mới đúng trên TOÀN BỘ
        round, không chỉ 1 rank. File không tồn tại (rank chưa log gì) -> bỏ qua."""
        collector = cls()
        for path in paths:
            p = Path(path)
            if not p.exists():
                continue
            for d in json.loads(p.read_text(encoding="utf-8")):
                collector.log(RolloutRecord(**d))
        return collector


stats_collector = StatsCollector()   # singleton — reset() giữa các round nếu muốn thống kê rolling


def score_completion(
    prompt: str,
    completion: str,
    future_bins: Sequence[Sequence[int]],
    stats: Optional[StatsCollector] = None,
    weights: Optional[WeightTable] = None,
) -> float:
    """
    prompt PHẢI được truyền — grammar (app/lang/parser.py) yêu cầu
    program := chart_block think_block action_block, chart_block LUÔN
    đứng đầu. GRPOTrainer (TRL) decode "completions" CHỈ từ token sinh
    RA SAU prompt (completion_ids), KHÔNG bao gồm chart — nếu parse
    completion một mình, Parser sẽ LUÔN báo thiếu <chart> (đúng 1 lỗi
    structural cố định, well_form_score()=0.85 bất kể think/action bên
    trong đúng hay sai), che gần hết gradient signal thật. Phải ghép lại
    prompt + completion trước khi parse — đúng convention nối chuỗi đã
    dùng ở app/training/data/data_module.py (full_text = prompt + " " + completion).
    """
    weights = weights if weights is not None else weight_table
    round_config = get_active_round_config()   # fail loud nếu chưa set — xem set_active_round_config()
    future_candles: List[FutureCandle] = [tuple(c) for c in future_bins]  # type: ignore[misc]

    parse_result = Parser.from_text(prompt + " " + completion).parse()

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
    # zone_width/sl_dist range LẤY TỪ round_config hiện tại (KHÔNG dùng
    # default hardcode của SemanticChecker/evaluate_outcome — round GRPO
    # luôn phải tường minh, xem round_config.py).
    semantic_result = SemanticChecker(
        zone_width_min_bins=round_config.zone_width_min_bins,
        zone_width_max_bins=round_config.zone_width_max_bins,
    ).check(program)
    extra_valid, forward_result = evaluate_outcome(
        program.action, program.think, future_candles,
        sl_min_dist_bins=round_config.sl_min_dist_bins,
        sl_max_dist_bins=round_config.sl_max_dist_bins,
    )
    overall_semantic_passed = semantic_result.passed and extra_valid

    if not overall_semantic_passed:
        sem_score = semantic_result.score
        if semantic_result.passed and not extra_valid:
            sem_score = max(0.0, sem_score - EXTRA_SEMANTIC_PENALTY)
        if stats is not None:
            stats.log(RolloutRecord(
                trend=trend, action_type=action_type, outcome_status=None, r_multiple=None,
                well_formed=True, semantic_passed=False,
            ))
        return R_WF_FULL + sem_score

    # --- Gate 3: outcome ---
    if forward_result is None:
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


def unified_reward_func(
    prompts: Sequence[Any],
    completions: Sequence[str],
    future_bins: Sequence[Sequence[Sequence[int]]],
    **kwargs: Any,
) -> List[float]:
    return [
        score_completion(prompt, completion, fb, stats=stats_collector, weights=weight_table)
        for prompt, completion, fb in zip(prompts, completions, future_bins)
    ]