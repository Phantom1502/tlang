from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.lang.ast_nodes import ActionNode, ThinkNode
from app.lang.parser import Parser
from app.lang.semantic import SemanticChecker
from app.training.reward.forward_test import (
    FutureCandle,
    OutcomeStatus,
    evaluate_outcome,
    probe_zone_quality,
)
from app.training.reward.round_config import RoundConfig

R_WF_FULL = 1.0
R_SEM_FULL = 1.0

# LƯU Ý: nếu đổi 2 hằng số trên, phải sửa lại _R_WF_FULL/_R_SEM_FULL (bản copy
# cục bộ, không import được vì circular) trong app/training/reward/round_config.py.

EXTRA_SEMANTIC_PENALTY = SemanticChecker.VIOLATION_PENALTY

# Chỉ 2 action này thật sự thực thi lệnh (BUY/SELL) — CANCEL_BUY/CANCEL_SELL/
# WAIT_BUY/WAIT_SELL/HOLD KHÔNG đóng góp vào outcome_score (quyết định đã chốt:
# "cancel action ko đóng góp vào outcome").
OUTCOME_ACTIONS = ("BUY", "SELL")

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


# =====================================================================
# ActionBuffTable — THAY HOÀN TOÀN cho WeightTable cũ (nhân trọng số theo
# trend+action_type). Lý do bỏ WeightTable: outcome BUY/SELL ~ zero-sum
# (thêm phí thì hơi âm), nhân với 1 hệ số cố định không tạo tín hiệu học có
# ý nghĩa, chỉ scale noise. ActionBuffTable CỘNG (không nhân) 1 giá trị học
# động vào outcome_score, chỉ có 2 key "BUY"/"SELL" (không tách theo trend —
# tỉ lệ target là tổng gộp toàn cục, xem update_buffs_from_stats).
# =====================================================================
class ActionBuffTable:
    def __init__(self):
        self._table: Dict[str, float] = defaultdict(float)

    def get(self, action_type: Optional[str]) -> float:
        if action_type is None:
            return 0.0
        return self._table.get(action_type, 0.0)

    def set(self, action_type: str, value: float) -> None:
        self._table[action_type] = value

    def reset(self) -> None:
        self._table.clear()

    def snapshot(self) -> Dict[str, float]:
        return dict(self._table)


action_buffs = ActionBuffTable()

_active_round_config: Optional[RoundConfig] = None

class HoldBuff:
    def __init__(self):
        self._value: float = 0.0

    def get(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        self._value = value

    def reset(self) -> None:
        self._value = 0.0


hold_buff = HoldBuff()

def set_active_round_config(config: RoundConfig) -> None:
    global _active_round_config
    _active_round_config = config
    action_buffs.reset()
    hold_buff.reset()
    
def get_active_round_config() -> RoundConfig:
    if _active_round_config is None:
        raise RuntimeError("Chưa load RoundConfig — gọi set_active_round_config(RoundConfig.load(path)) trước.")
    return _active_round_config


@dataclass
class RolloutRecord:
    trend: Optional[str]
    action_type: Optional[str]              # action_type SAU parse — None nếu well-form fail
    intended_action_type: Optional[str]     # regex thô trên raw completion, có ngay cả khi well-form fail
    outcome_status: Optional[str]
    r_multiple: Optional[float]
    well_formed: bool
    semantic_passed: bool
    sl_valid: Optional[bool] = None         # chỉ có ý nghĩa với BUY/SELL


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

    def action_counts(self) -> Dict[str, int]:
        """Đếm action_type (well_formed+semantic_passed), GỘP MỌI TREND — dùng cho
        update_buffs_from_stats(). Tỉ lệ target là (BUY+SELL)/tổng toàn cục, không
        tách theo trend/family — đơn giản hoá theo quyết định đã chốt."""
        counts: Dict[str, int] = defaultdict(int)
        for r in self._records:
            if r.action_type is None or not (r.well_formed and r.semantic_passed):
                continue
            counts[r.action_type] += 1
        return dict(counts)

    def summary_by_intended_action(self) -> Dict[str, Dict[str, Any]]:
        """Well-form rate theo Ý ĐỊNH (intended_action_type, có ngay cả khi
        parse fail) — tách 'model chọn action X nhiều hơn' khỏi 'action X dễ
        sinh sai cú pháp hơn nên bị lọc rớt nhiều hơn ở summary() thường."""
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

        counts = self.action_counts()
        total = sum(counts.values())
        buy_sell = counts.get("BUY", 0) + counts.get("SELL", 0)
        ratio = buy_sell / total if total else 0.0
        print(f"\n=== Tỉ lệ (BUY+SELL)/tổng action (dùng cho buff) = {ratio*100:.1f}% (n={total}) ===")

    def to_list(self):
        return [asdict(r) for r in self._records]
        
    def save(self, path: str, buffs=None, hold=None) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": self.to_list()}
        if buffs is not None:
            payload["action_buffs"] = buffs.snapshot()
        if hold is not None:
            payload["hold_buff"] = hold.get()
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str):
        """Trả về (collector, buffs_dict, hold_value). Hỗ trợ NGƯỢC format cũ
        (bare list, không có action_buffs/hold_buff) — file cũ từ trước khi đổi
        format vẫn load được, chỉ là buffs_dict rỗng và hold_value=0.0."""
        collector = cls()
        p = Path(path)
        buffs_dict: Dict[str, float] = {}
        hold_value: float = 0.0
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):          # format cũ — bare list record
                records = data
            else:                                # format mới — {"records":..., "action_buffs":..., "hold_buff":...}
                records = data.get("records", [])
                buffs_dict = data.get("action_buffs", {})
                hold_value = data.get("hold_buff", 0.0)
            for d in records:
                collector.log(RolloutRecord(**d))

        return collector, buffs_dict, hold_value
    
    @classmethod
    def merge_from_files(cls, paths) -> "StatsCollector":
        """report_only giờ chỉ gộp được records của CHU KỲ CUỐI mỗi rank (vì
        file bị reset mỗi save_steps) — không còn full-round history, đây là
        đánh đổi chủ ý để tránh file phình to qua hàng nghìn step."""
        collector = cls()
        for path in paths:
            p = Path(path)
            if not p.exists(): continue
            data = json.loads(p.read_text(encoding="utf-8"))
            records = data if isinstance(data, list) else data.get("records", [])
            for d in records:
                collector.log(RolloutRecord(**d))
        return collector


stats_collector = StatsCollector()

RATIO_TOLERANCE = 1e-9


def update_buffs_from_stats(
    stats: StatsCollector,
    round_config: RoundConfig,
    buffs: Optional[ActionBuffTable] = None,
    hold: Optional[HoldBuff] = None,
) -> None:
    buffs = buffs if buffs is not None else action_buffs
    hold = hold if hold is not None else hold_buff
    counts = stats.action_counts()
    total = sum(counts.values())
    if total == 0:
        return

    # --- BUY/SELL buff (giữ nguyên logic cũ) ---
    actual_ratio = (counts.get("BUY", 0) + counts.get("SELL", 0)) / total
    target_ratio = round_config.target_action_ratio
    if actual_ratio < target_ratio - RATIO_TOLERANCE:
        delta = round_config.buff_step
    elif actual_ratio > target_ratio + RATIO_TOLERANCE:
        delta = -round_config.buff_step
    else:
        delta = 0.0
    for action_type in OUTCOME_ACTIONS:
        new_buff = min(round_config.buff_max, max(round_config.buff_min, buffs.get(action_type) + delta))
        buffs.set(action_type, new_buff)

    # --- HOLD buff — mẫu số = TOÀN BỘ action (mọi trend), độc lập BUY/SELL ---
    actual_hold_ratio = counts.get("HOLD", 0) / total
    target_hold_ratio = round_config.target_hold_ratio
    if actual_hold_ratio < target_hold_ratio - RATIO_TOLERANCE:
        hold_delta = round_config.hold_buff_step
    elif actual_hold_ratio > target_hold_ratio + RATIO_TOLERANCE:
        hold_delta = -round_config.hold_buff_step
    else:
        hold_delta = 0.0
    new_hold = min(round_config.hold_buff_max, max(round_config.hold_buff_min, hold.get() + hold_delta))
    hold.set(new_hold)


# =====================================================================
# Nhánh 1 — zone quality. Áp dụng cho MỌI action có think.zone (BUY, SELL,
# CANCEL_BUY, CANCEL_SELL, WAIT_BUY, WAIT_SELL) — HOLD (RANGE không zone)
# luôn nhận 0.0. Liên tục theo r_multiple của probe_zone_quality (không còn
# nhị phân bonus/penalty cố định như bản cũ).
# =====================================================================
def compute_zone_score(
    think: ThinkNode,
    future_candles: List[FutureCandle],
    round_config: RoundConfig,
) -> float:
    if think.zone is None:
        return 0.0
    probe = probe_zone_quality(think.zone, future_candles)
    if probe.status == OutcomeStatus.INVALID_SETUP:
        return 0.0
    return probe.r_multiple * round_config.zone_score_scale


# =====================================================================
# Nhánh 2 — outcome. CHỈ BUY/SELL (action.action_type in OUTCOME_ACTIONS).
# CANCEL_BUY/CANCEL_SELL/WAIT_BUY/WAIT_SELL/HOLD -> 0.0 KỂ CẢ khi
# forward_result có giá trị (CANCEL vẫn có counterfactual outcome từ
# evaluate_outcome(), nhưng theo quyết định đã chốt, CANCEL không đóng góp
# vào outcome_score nữa).
# =====================================================================
def compute_outcome_score(
    action: ActionNode,
    think: ThinkNode,
    forward_result,
    round_config: RoundConfig,
    buffs: ActionBuffTable,
) -> float:
    if action.action_type not in OUTCOME_ACTIONS:
        return 0.0
    if forward_result is None:
        return 0.0

    buff = buffs.get(action.action_type)

    risk_bins = abs(think.current_price_bin - action.sl) if action.sl is not None else 0.0
    fee_in_r = round_config.trade_fee_bins / risk_bins if risk_bins > 0 else 0.0

    return forward_result.r_multiple - fee_in_r + buff


def score_completion(
    prompt: str,
    completion: str,
    future_bins: Sequence[Sequence[int]],
    stats: Optional[StatsCollector] = None,
    buffs: Optional[ActionBuffTable] = None,
) -> float:
    buffs = buffs if buffs is not None else action_buffs
    round_config = get_active_round_config()
    future_candles: List[FutureCandle] = [tuple(c) for c in future_bins]
    intended_action = _extract_intended_action(completion)

    parse_result = Parser.from_text(prompt + " " + completion).parse()

    # ------------------------------------------------------------
    # Bước 1: gate well-formed — fail thì GIỮ NGUYÊN well_form_score, return ngay.
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

    # sl_valid là 1 phần của semantic score — áp dụng KHÔNG điều kiện (bất kể phần
    # semantic còn lại pass hay fail), CHỈ có ý nghĩa khi action_type là BUY/SELL.
    sem_score = semantic_result.score
    if semantic_result.passed and not extra_valid:
        sem_score = max(0.0, sem_score - EXTRA_SEMANTIC_PENALTY)
    if sl_valid is True:
        sem_score += round_config.sl_valid_bonus
    elif sl_valid is False:
        sem_score -= round_config.sl_valid_penalty
    sem_score = max(0.0, sem_score)

    # ------------------------------------------------------------
    # Bước 1 (tiếp): gate semantic — fail thì GIỮ NGUYÊN R_WF_FULL + sem_score, return.
    # ------------------------------------------------------------
    if not overall_semantic_passed:
        if stats is not None:
            stats.log(RolloutRecord(
                trend=trend, action_type=action_type, intended_action_type=intended_action,
                outcome_status=None, r_multiple=None,
                well_formed=True, semantic_passed=False, sl_valid=sl_valid,
            ))
        return R_WF_FULL + sem_score

    # ==============================================================
    # Bước 1 PASS cả 2 gate — K đồng đều (mọi mẫu tới đây coi như nhau ở mức
    # sàn), rồi cộng 2 nhánh SONG SONG (bước 2):
    #   - zone_score:    mọi action có zone (BUY/SELL/CANCEL_*/WAIT_*)
    #   - outcome_score: CHỈ BUY/SELL
    # Bước 3: reward = K + zone_score + outcome_score
    # ==============================================================
    K = round_config.pass_gate2_bonus
    zone_score = compute_zone_score(think, future_candles, round_config)
    outcome_score = compute_outcome_score(action, think, forward_result, round_config, buffs)
    reward = K + zone_score + outcome_score
    if action_type == "HOLD":
        reward += hold_buff.get()
        
    if stats is not None:
        stats.log(RolloutRecord(
            trend=trend, action_type=action_type, intended_action_type=intended_action,
            outcome_status=forward_result.status.value if forward_result else None,
            r_multiple=forward_result.r_multiple if forward_result else None,
            well_formed=True, semantic_passed=True, sl_valid=sl_valid,
        ))
    return reward


def unified_reward_func(
    prompts: Sequence[Any],
    completions: Sequence[str],
    future_bins: Sequence[Sequence[Sequence[int]]],
    **kwargs: Any,
) -> List[float]:
    return [
        score_completion(prompt, completion, fb, stats=stats_collector, buffs=action_buffs)
        for prompt, completion, fb in zip(prompts, completions, future_bins)
    ]