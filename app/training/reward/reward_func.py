from __future__ import annotations

import json
import re
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

# =====================================================================
# Best-effort "ý định" của model, đọc trực tiếp trên raw completion text —
# KHÔNG cần parse thành công. Dùng để tách "model chọn action X" khỏi
# "model định làm X nhưng sinh sai cú pháp" (2 nguyên nhân hoàn toàn khác
# nhau khi đọc lệch tần suất trong StatsCollector — xem docs/reward_design.md
# mục "intended vs realized action").
# =====================================================================
_ACTION_TYPE_RE = re.compile(r"\b(CANCEL_BUY|CANCEL_SELL|WAIT_BUY|WAIT_SELL|BUY|SELL|HOLD)\b")


def _extract_intended_action(completion: str) -> Optional[str]:
    """Token ACTION_TYPE đầu tiên xuất hiện trong completion, bất kể completion
    có well-formed hay không — best-effort, không dùng để tính reward, CHỈ
    dùng cho thống kê."""
    m = _ACTION_TYPE_RE.search(completion)
    return m.group(1) if m else None


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
    action_type: Optional[str]              # action_type SAU parse — None nếu well-form fail
    intended_action_type: Optional[str]     # MỚI — regex thô trên raw completion, có ngay cả khi well-form fail
    outcome_status: Optional[str]
    r_multiple: Optional[float]
    well_formed: bool
    semantic_passed: bool
    sl_valid: Optional[bool] = None         # MỚI — chỉ có ý nghĩa với BUY/SELL


class StatsCollector:
    def __init__(self):
        self._records: List[RolloutRecord] = []

    def log(self, record: RolloutRecord) -> None:
        self._records.append(record)

    def reset(self) -> None:
        self._records.clear()

    def summary(self):
        """Thống kê theo action_type SAU parse — chỉ record well_formed+semantic_passed.
        Dùng để đọc outcome/timing thật, KHÔNG dùng để đọc tần suất ý định (xem
        summary_by_intended_action bên dưới cho việc đó)."""
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

    def summary_by_intended_action(self) -> Dict[str, Dict[str, Any]]:
        """Well-form rate theo Ý ĐỊNH (intended_action_type, có ngay cả khi
        parse fail) — tách 'model chọn action X nhiều hơn' khỏi 'action X dễ
        sinh sai cú pháp hơn nên bị lọc rớt nhiều hơn ở summary() thường.

        So sánh well_form_rate giữa 2 action cùng nhóm (vd BUY vs CANCEL_BUY):
          - Nếu xấp xỉ nhau -> lệch tần suất trong summary() là hành vi thật.
          - Nếu lệch rõ (vd BUY << CANCEL_BUY) -> phần lớn lệch trong
            summary() là nhiễu cú pháp (BUY cần sinh thêm SL+RR, nhiều cơ hội
            sai hơn) — vấn đề thuộc SFT/generator, KHÔNG phải reward design.
        """
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "well_formed": 0})
        for r in self._records:
            if r.intended_action_type is None:
                continue
            entry = counts[r.intended_action_type]
            entry["total"] += 1
            if r.well_formed:
                entry["well_formed"] += 1
        return {
            action: {
                "total": e["total"],
                "well_formed": e["well_formed"],
                "well_form_rate": e["well_formed"] / e["total"] if e["total"] else 0.0,
            }
            for action, e in counts.items()
        }

    def print_summary(self) -> None:
        summary = self.summary()
        print("=== Rollout stats (trend -> action, chỉ well-formed + semantic pass) ===")
        for trend, actions in summary.items():
            print(f"trend={trend}")
            for action_type, stat in actions.items():
                avg_r = f"{stat['avg_r_multiple']:.2f}" if stat["avg_r_multiple"] is not None else "-"
                win_rate = f"{stat['win_rate'] * 100:.0f}%" if stat["win_rate"] is not None else "-"
                print(f"  {action_type:<12} count={stat['count']:<4} freq={stat['freq_within_trend']*100:5.1f}%  avg_R={avg_r:>6}  win_rate={win_rate}")

        print("\n=== Well-form rate theo Ý ĐỊNH (intended_action_type — kể cả parse fail) ===")
        intended = self.summary_by_intended_action()
        for action, stat in sorted(intended.items()):
            print(f"  {action:<12} total={stat['total']:<6} well_formed={stat['well_formed']:<6} rate={stat['well_form_rate']*100:5.1f}%")

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
    trung tính (0)."""
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
    intended_action = _extract_intended_action(completion)

    parse_result = Parser.from_text(prompt + " " + completion).parse()

    # ------------------------------------------------------------
    # Fail well-form
    # ------------------------------------------------------------
    if not parse_result.is_well_formed():
        if stats is not None:
            program = parse_result.ast
            stats.log(RolloutRecord(
                trend=program.think.trend if program and program.think else None,
                action_type=program.action.action_type if program and program.action else None,
                intended_action_type=intended_action,
                outcome_status=None, r_multiple=None,
                well_formed=False, semantic_passed=False,
            ))
        if round_config.pure_outcome_mode:
            return 0.0
        return parse_result.well_form_score()

    program = parse_result.ast
    think, action = program.think, program.action
    trend = think.trend
    action_type = action.action_type

    semantic_result = SemanticChecker(
        zone_width_min_bins=round_config.zone_width_min_bins,
        zone_width_max_bins=round_config.zone_width_max_bins,
    ).check(program)
    extra_valid, forward_result, sl_valid = evaluate_outcome(
        action, think, future_candles,
        sl_min_dist_bins=round_config.sl_min_dist_bins,
        sl_max_dist_bins=round_config.sl_max_dist_bins,
    )
    overall_semantic_passed = semantic_result.passed and extra_valid

    # ------------------------------------------------------------
    # sl_valid giờ là 1 PHẦN của semantic score — áp dụng KHÔNG điều kiện
    # (bất kể phần semantic còn lại pass hay fail), CHỈ có ý nghĩa khi
    # action_type là BUY/SELL (sl_valid is None cho action khác -> không đổi
    # gì). Cộng khi đúng luật (dist + đúng phía zone), trừ khi sai — nhưng
    # KHÔNG gate vào extra_valid/overall_semantic_passed (khác bug cũ đã fix:
    # SL sai không còn làm mất sạch outcome/mất pass gate 2, chỉ ảnh hưởng
    # đúng phần điểm số này).
    # ------------------------------------------------------------
    sem_score = semantic_result.score
    if semantic_result.passed and not extra_valid:
        sem_score = max(0.0, sem_score - EXTRA_SEMANTIC_PENALTY)
    if sl_valid is True:
        sem_score += round_config.sl_valid_bonus
    elif sl_valid is False:
        sem_score -= round_config.sl_valid_penalty
    sem_score = max(0.0, sem_score)

    # ------------------------------------------------------------
    # Fail semantic (bao gồm sl_valid đã gộp ở trên)
    # ------------------------------------------------------------
    if not overall_semantic_passed:
        if stats is not None:
            stats.log(RolloutRecord(
                trend=trend, action_type=action_type, intended_action_type=intended_action,
                outcome_status=None, r_multiple=None,
                well_formed=True, semantic_passed=False, sl_valid=sl_valid,
            ))
        if round_config.pure_outcome_mode:
            return 0.0
        return R_WF_FULL + sem_score

    # ==============================================================
    # PASS cả 2 gate — rẽ nhánh theo mode
    # ==============================================================
    if round_config.pure_outcome_mode:
        # Round 2: điểm ngữ nghĩa không còn ý nghĩa nữa (đã dùng hết ở round 1)
        # — mọi mẫu pass gate đều có SÀN TUYỆT ĐỐI = K, không cộng R_WF_FULL/
        # R_SEM_FULL/zone_bonus/sl_bonus nào cả. Chỉ còn outcome thật quyết định.
        K = round_config.pass_gate2_bonus
        w = weights.get(trend, action_type)

        if action_type in ("HOLD", "WAIT_BUY", "WAIT_SELL"):
            reward = K   # không có outcome để tính — sàn tuyệt đối, không hơn không kém

        elif action_type in ("CANCEL_BUY", "CANCEL_SELL"):
            reward = K + forward_result.r_multiple * w

        elif action_type in ("BUY", "SELL"):
            risk_bins = abs(think.current_price_bin - action.sl)
            fee_in_r = round_config.trade_fee_bins / risk_bins if risk_bins > 0 else 0.0
            net_r_multiple = forward_result.r_multiple - fee_in_r
            reward = K + net_r_multiple * w

        else:
            reward = K

        if stats is not None:
            stats.log(RolloutRecord(
                trend=trend, action_type=action_type, intended_action_type=intended_action,
                outcome_status=forward_result.status.value if forward_result else None,
                r_multiple=forward_result.r_multiple if forward_result else None,
                well_formed=True, semantic_passed=True, sl_valid=sl_valid,
            ))
        return reward

    # ------------------------------------------------------------
    # Round 1 (mặc định, hành vi cũ) — K + zone_bonus + sl_bonus + timing*w.
    # sl_valid_bonus/penalty ĐÃ được gộp vào sem_score phía trên; ở nhánh
    # round1-style base vẫn dùng R_SEM_FULL cố định (không dùng sem_score) —
    # giữ đúng hành vi gốc, vì round1 không đổi công thức base, chỉ đổi công
    # thức của nhánh FAIL (nơi sem_score thực sự được dùng).
    # ------------------------------------------------------------
    K = round_config.pass_gate2_bonus
    base = R_WF_FULL + R_SEM_FULL + K

    return base


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