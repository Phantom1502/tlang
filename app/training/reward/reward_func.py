from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.lang.parser import Parser
from app.lang.semantic import SemanticChecker
from app.training.reward.forward_test import FutureCandle, OutcomeStatus, evaluate_outcome, probe_zone_quality
from app.training.reward.round_config import RoundConfig

R_WF_FULL = 1.0
R_SEM_FULL = 1.0

EXTRA_SEMANTIC_PENALTY = SemanticChecker.VIOLATION_PENALTY

DEFAULT_WEIGHT = 1.0


class WeightTable:
    def __init__(self, default: float = DEFAULT_WEIGHT):
        self._default = default
        self._table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: self._default))

    def get(self, trend, action_type) -> float:
        if trend is None or action_type is None:
            return self._default
        return self._table[trend][action_type]

    def set(self, trend, action_type, weight) -> None:
        self._table[trend][action_type] = weight

    def set_many(self, updates) -> None:
        for trend, actions in updates.items():
            for action_type, w in actions.items():
                self.set(trend, action_type, w)

    def reset(self) -> None:
        self._table.clear()

    def snapshot(self):
        return {t: dict(a) for t, a in self._table.items()}


weight_table = WeightTable()

_active_round_config: Optional[RoundConfig] = None


def set_active_round_config(config: RoundConfig) -> None:
    global _active_round_config
    _active_round_config = config
    weight_table.reset()
    weight_table.set_many(config.weight_table)


def get_active_round_config() -> RoundConfig:
    if _active_round_config is None:
        raise RuntimeError("Chưa load RoundConfig — gọi set_active_round_config(RoundConfig.load(path)) trước.")
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

    def summary(self):
        by_trend_total = defaultdict(int)
        raw = defaultdict(lambda: defaultdict(lambda: {"count": 0, "r_multiples": []}))
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
        result = {}
        for trend, actions in raw.items():
            result[trend] = {}
            total = by_trend_total[trend]
            for action_type, entry in actions.items():
                rms = entry["r_multiples"]
                avg_r = sum(rms) / len(rms) if rms else None
                win_rate = (sum(1 for x in rms if x > 0) / len(rms)) if rms else None
                result[trend][action_type] = {
                    "count": entry["count"], "freq_within_trend": entry["count"] / total if total else 0.0,
                    "avg_r_multiple": avg_r, "win_rate": win_rate,
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
                print(f"  {action_type:<12} count={stat['count']:<4} freq={stat['freq_within_trend']*100:5.1f}%  avg_R={avg_r:>6}  win_rate={win_rate}")

    def to_list(self):
        return [asdict(r) for r in self._records]

    def save(self, path: str) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_list(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "StatsCollector":
        collector = cls(); p = Path(path)
        if p.exists():
            for d in json.loads(p.read_text(encoding="utf-8")):
                collector.log(RolloutRecord(**d))
        return collector

    @classmethod
    def merge_from_files(cls, paths) -> "StatsCollector":
        collector = cls()
        for path in paths:
            p = Path(path)
            if not p.exists(): continue
            for d in json.loads(p.read_text(encoding="utf-8")):
                collector.log(RolloutRecord(**d))
        return collector


stats_collector = StatsCollector()

def _compute_zone_bonus(probe_result, round_config: RoundConfig) -> float:
    """Điểm zone-quality có dấu — symmetric bonus/penalty. TIMEOUT/INVALID_SETUP
    trung tính (0): zone không liên quan đến việc thị trường có đi đủ xa trong
    horizon hay không, phạt TIMEOUT sẽ đẩy model né các zone hợp lý nhưng chỉ
    tình cờ đi ngang sau đó — không phải tín hiệu zone thật."""
    if probe_result.status == OutcomeStatus.WIN:
        return round_config.zone_quality_bonus
    if probe_result.status == OutcomeStatus.LOSS:
        return -round_config.zone_quality_penalty
    return 0.0

def score_completion(
    prompt: str,
    completion: str,
    future_bins: Sequence[Sequence[int]],
    stats: Optional[StatsCollector] = None,
    weights: Optional[WeightTable] = None,
) -> float:
    weights = weights if weights is not None else weight_table
    round_config = get_active_round_config()
    future_candles: List[FutureCandle] = [tuple(c) for c in future_bins]

    parse_result = Parser.from_text(prompt + " " + completion).parse()

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
    think, action = program.think, program.action
    trend = think.trend
    action_type = action.action_type

    semantic_result = SemanticChecker(
        zone_width_min_bins=round_config.zone_width_min_bins,
        zone_width_max_bins=round_config.zone_width_max_bins,
    ).check(program)
    extra_valid, forward_result = evaluate_outcome(
        action, think, future_candles,
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

    K = round_config.pass_gate2_bonus
    base = R_WF_FULL + R_SEM_FULL + K

    if action_type == "HOLD":
        reward = base
        
    elif action_type in ("WAIT_BUY", "WAIT_SELL"):
        probe = probe_zone_quality(think.zone, future_candles)
        zone_bonus = _compute_zone_bonus(probe, round_config)
        reward = base + zone_bonus

    elif action_type in ("CANCEL_BUY", "CANCEL_SELL"):
        probe = probe_zone_quality(think.zone, future_candles)
        zone_bonus = _compute_zone_bonus(probe, round_config)
        w = weights.get(trend, action_type)
        timing_score = forward_result.r_multiple * w
        reward = base + zone_bonus + min(0.0, timing_score)

    elif action_type in ("BUY", "SELL"):
        probe = probe_zone_quality(think.zone, future_candles)
        zone_bonus = _compute_zone_bonus(probe, round_config)
        w = weights.get(trend, action_type)

        # Phí giao dịch kiểu spread — cố định theo BIN (round_config.trade_fee_bins),
        # quy đổi ra R-multiple theo risk CỦA CHÍNH lệnh này (risk_bins = |entry-SL|),
        # giống spread ảnh hưởng R nhiều hơn khi SL đặt sát (risk nhỏ). CHỈ áp cho
        # BUY/SELL thật (vào lệnh thật) — CANCEL/WAIT/HOLD không trả phí vì không
        # thật sự có vị thế nào được mở.
        risk_bins = abs(think.current_price_bin - action.sl)
        fee_in_r = round_config.trade_fee_bins / risk_bins if risk_bins > 0 else 0.0
        net_r_multiple = forward_result.r_multiple - fee_in_r

        timing_score = net_r_multiple * w
        reward = base + zone_bonus + timing_score

    else:
        reward = base

    if stats is not None:
        stats.log(RolloutRecord(
            trend=trend, action_type=action_type,
            outcome_status=forward_result.status.value if forward_result else None,
            r_multiple=forward_result.r_multiple if forward_result else None,
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