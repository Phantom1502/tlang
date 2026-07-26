from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.lang.ast_nodes import ActionNode, ProgramNode, ThinkNode
from app.lang.parser import Parser
from app.lang.semantic import SemanticChecker, SemanticResult
from app.tokenizer.vocab_builder import ACTION_TYPES
from app.training.reward.buff_controller_v2 import EMABuffControllerV2
from app.training.reward.forward_test import (
    FutureCandle,
    ForwardTestResult,
    OutcomeStatus,
    is_sl_valid,
    measure_max_favorable_r,
    partial_tp_forward_test,
    probe_zone_quality,
)
from app.training.reward.round_config_v2 import GROUPS_ACTION, GROUPS_ZONE, RoundConfigV2
# Tái dùng nguyên vẹn 3 hằng số + 1 controller pure từ v1 — KHÔNG duplicate
# logic, đây đều là những phần đã tách đúng tầng ngay từ v1 (không dính
# code-smell buff-trộn-raw-score mà bản v2 này sửa).
from app.training.reward.reward_func import (
    EXTRA_SEMANTIC_PENALTY,
    MIN_SAMPLES_FOR_RR_ENTROPY,
    OUTCOME_ACTIONS,
    R_SEM_FULL,
    R_WF_FULL,
    RREntropyController,
    shannon_entropy_nats,
)

TASK_ZONE = "zone"
TASK_ACTION = "action"
TASKS: Tuple[str, ...] = (TASK_ZONE, TASK_ACTION)

_ACTION_TYPE_RE = re.compile(r"\b(" + "|".join(ACTION_TYPES) + r")\b")


def _extract_intended_action(completion: str) -> Optional[str]:
    """Token ACTION_TYPE đầu tiên xuất hiện trong completion, bất kể có
    well-formed hay không — best-effort, chỉ dùng cho thống kê/report.
    Dùng ACTION_TYPES từ vocab_builder làm nguồn sự thật duy nhất cho danh
    sách action type, không tự liệt kê lại."""
    m = _ACTION_TYPE_RE.search(completion)
    return m.group(1) if m else None


def _entropy_and_probs(values: Sequence[int]) -> Tuple[float, Dict[int, float]]:
    n = len(values)
    counts: Dict[int, int] = defaultdict(int)
    for v in values:
        counts[v] += 1
    probs = {v: c / n for v, c in counts.items()}
    h = -sum(p * math.log(p) for p in probs.values())
    return h, probs


# =====================================================================
# TẦNG 1 — Raw score. PURE: không buff, không side-effect, không ghi log.
# =====================================================================

@dataclass
class CommonGateResult:
    program: Optional[ProgramNode]
    well_formed: bool
    semantic_result: Optional[SemanticResult]
    passed: bool                 # well_formed AND semantic_result.passed
    gate_score: float


def common_gate(prompt: str, completion: str, round_config: RoundConfigV2) -> CommonGateResult:
    """
    Gate DÙNG CHUNG cho CẢ 2 task — tái sử dụng NGUYÊN VẸN
    Parser.well_form_score() và SemanticChecker.check() (đầy đủ A/B/B2/D/E,
    1 lần gọi, 1 danh sách violations) — KHÔNG viết logic gate mới nào.

    Vì D (price_in_zone đúng sự thật) và E (action_group hợp lệ theo
    price_in_zone) LUÔN chạy cùng 1 lần gọi check() này, model khai
    price_in_zone sai sự thật sẽ bị D bắt ngay, passed=False, chặn đứng
    CẢ 2 task tại đây — không task nào cần tự tính lại giá trị "grounded"
    để chống hack riêng nữa.
    """
    parse_result = Parser.from_text(prompt + " " + completion).parse()
    program = parse_result.ast

    if not parse_result.is_well_formed():
        return CommonGateResult(
            program=program, well_formed=False, semantic_result=None,
            passed=False, gate_score=parse_result.well_form_score(),
        )

    semantic_result = SemanticChecker(
        zone_width_min_bins=round_config.zone_width_min_bins,
        zone_width_max_bins=round_config.zone_width_max_bins,
    ).check(program)

    if not semantic_result.passed:
        return CommonGateResult(
            program=program, well_formed=True, semantic_result=semantic_result,
            passed=False, gate_score=R_WF_FULL + semantic_result.score,
        )

    return CommonGateResult(
        program=program, well_formed=True, semantic_result=semantic_result,
        passed=True, gate_score=R_WF_FULL + R_SEM_FULL,   # hằng số cố định — MỌI mẫu pass đều bằng nhau
    )


@dataclass
class ZoneTaskScore:
    zone_quality: float          # r_multiple của probe, 0.0 nếu không có zone hoặc INVALID_SETUP
    probe: Optional[ForwardTestResult]
    has_zone: bool


def compute_zone_task_score(program: ProgramNode, future_bins: Sequence[Sequence[int]]) -> ZoneTaskScore:
    """
    Mục tiêu DUY NHẤT: model nhận diện đúng 1 vùng giá có ý nghĩa —
    KHÔNG nhận round_config, KHÔNG nhận buffs, chỉ cần chart+zone.
    """
    think = program.think
    if think.zone is None:
        return ZoneTaskScore(zone_quality=0.0, probe=None, has_zone=False)

    future_candles: List[FutureCandle] = [tuple(c) for c in future_bins]
    probe = probe_zone_quality(think.zone, future_candles)
    if probe.status == OutcomeStatus.INVALID_SETUP:
        return ZoneTaskScore(zone_quality=0.0, probe=probe, has_zone=True)
    return ZoneTaskScore(zone_quality=probe.r_multiple, probe=probe, has_zone=True)


@dataclass
class ActionTaskScore:
    task_passed: bool             # False CHỈ khi action_type in (BUY,SELL) và sl_valid=False
    action_type: Optional[str]
    sl_valid: Optional[bool]
    entry_quality: float          # 0.0 nếu không phải BUY/SELL
    outcome: float                 # 0.0 nếu không phải BUY/SELL
    raw_score: float                # = entry_quality + outcome — KHÔNG có buff trong này
    forward_result: Optional[ForwardTestResult]
    rr: Optional[int]


def compute_action_task_score(
    action: ActionNode,
    think: ThinkNode,
    future_bins: Sequence[Sequence[int]],
    round_config: RoundConfigV2,
) -> ActionTaskScore:
    """
    Mục tiêu DUY NHẤT: biết trước zone đã cho là đúng (common gate đã pass
    A/B/B2/D/E), xét logic hành động + SL có hợp lý hay không, rồi mới đo
    chất lượng. KHÔNG gọi buffs.record()/get_buff() ở đây — buff hoàn toàn
    thuộc tầng khác (xem apply_action_buff bên dưới).

    Bước 1 (gate) của task action, sau khi rút gọn nhờ common gate đã lo
    hết rule A/B/B2/D/E, CHỈ CÒN đúng 1 điều kiện mới: SL hợp lệ
    (is_sl_valid) — đúng hướng zone + đã chạm zone đã nằm trong E/D rồi.
    """
    action_type = action.action_type

    if action_type not in OUTCOME_ACTIONS:   # CANCEL_*/WAIT_*/HOLD — không có gì để gate/đo
        return ActionTaskScore(
            task_passed=True, action_type=action_type, sl_valid=None,
            entry_quality=0.0, outcome=0.0, raw_score=0.0,
            forward_result=None, rr=None,
        )

    # Phòng vệ: BUY/SELL đã qua common gate (E) chắc chắn có zone/sl/rr,
    # nhưng vẫn guard nhẹ để không crash nếu có edge-case nào lọt qua.
    if think.zone is None or action.sl is None or action.rr is None:
        return ActionTaskScore(
            task_passed=False, action_type=action_type, sl_valid=False,
            entry_quality=0.0, outcome=0.0, raw_score=0.0,
            forward_result=None, rr=action.rr,
        )

    sl_valid = is_sl_valid(
        action_type, think.current_price_bin, action.sl, think.zone,
        round_config.sl_min_dist_bins, round_config.sl_max_dist_bins,
    )
    if not sl_valid:
        return ActionTaskScore(
            task_passed=False, action_type=action_type, sl_valid=False,
            entry_quality=0.0, outcome=0.0, raw_score=0.0,
            forward_result=None, rr=action.rr,
        )

    future_candles: List[FutureCandle] = [tuple(c) for c in future_bins]
    direction = "long" if action_type == "BUY" else "short"

    entry_quality = measure_max_favorable_r(
        think.current_price_bin, action.sl, future_candles, direction,
    ) * round_config.entry_score_weight

    fr = partial_tp_forward_test(think.current_price_bin, action.sl, action.rr, future_candles, direction)
    risk_bins = abs(think.current_price_bin - action.sl)
    fee_in_r = round_config.trade_fee_bins / risk_bins if risk_bins > 0 else 0.0
    outcome = fr.r_multiple - fee_in_r

    return ActionTaskScore(
        task_passed=True, action_type=action_type, sl_valid=True,
        entry_quality=entry_quality, outcome=outcome, raw_score=entry_quality + outcome,
        forward_result=fr, rr=action.rr,
    )


# =====================================================================
# TẦNG 2 — Buff. Áp dụng qua ĐÚNG 1 hàm/loại, không if/elif rải rác.
# =====================================================================

def apply_action_buff(action_type: Optional[str], raw_score: float, buffs: EMABuffControllerV2) -> float:
    if action_type is None:
        return raw_score
    return raw_score + buffs.get_buff(action_type)


def apply_zone_buff(has_zone: bool, raw_score: float, buffs: EMABuffControllerV2) -> float:
    key = "HAS_ZONE" if has_zone else "NO_ZONE"
    return raw_score + buffs.get_buff(key)


# =====================================================================
# TẦNG 3 — Tracking. Nguồn DUY NHẤT, dùng lại được cho cả report lẫn buff
# (qua watermark step_boundary, KHÔNG cần 2 bộ đếm trùng lặp).
# =====================================================================

@dataclass
class TaskRolloutMeta:
    task_id: str
    trend: Optional[str]
    action_type: Optional[str]
    intended_action_type: Optional[str]
    well_formed: bool
    semantic_passed: bool
    task_passed: Optional[bool]     # None nếu task=zone, hoặc common gate đã fail
    sl_valid: Optional[bool]
    rr: Optional[int]
    has_zone: Optional[bool]         # chỉ populate ở nhánh task=zone pass gate
    zone_quality: Optional[float]
    entry_quality: Optional[float]
    outcome: Optional[float]
    buff_applied: Optional[float]    # = reward_sau_buff - raw_score (để audit riêng phần buff đóng góp)
    reward: float


class StatsCollectorV2:
    """
    Nguồn DUY NHẤT cho cả report (theo nhịp save_steps, xem
    print_summary()/summary()) LẪN nuôi buff (theo nhịp optimizer step, xem
    counts_since_step_boundary()) — không có 2 bộ đếm tách rời như v1
    (record() của buff_controller cũ và StatsCollector.log() từng độc lập).

    mark_step_boundary() chỉ dịch 1 con trỏ index, KHÔNG xoá gì —
    reset() (gọi ở on_save, cùng nhịp save_steps) mới thật sự xoá records
    VÀ đưa watermark về 0.
    """

    def __init__(self) -> None:
        self._records: List[TaskRolloutMeta] = []
        self._step_boundary: int = 0

    def log(self, meta: TaskRolloutMeta) -> None:
        self._records.append(meta)

    def reset(self) -> None:
        self._records.clear()
        self._step_boundary = 0

    def mark_step_boundary(self) -> None:
        self._step_boundary = len(self._records)

    @staticmethod
    def _filter_and_count(records: Sequence[TaskRolloutMeta], task_id: str, key_fn) -> Tuple[Dict[str, int], int]:
        counts: Dict[str, int] = defaultdict(int)
        total = 0
        for r in records:
            if r.task_id != task_id or not r.well_formed or not r.semantic_passed:
                continue
            if task_id == TASK_ACTION and r.task_passed is not True:
                continue
            key = key_fn(r)
            if key is None:
                continue
            counts[key] += 1
            total += 1
        return dict(counts), total

    def counts_since_step_boundary(self, task_id: str, key_fn) -> Tuple[Dict[str, int], int]:
        """Dùng để nuôi buff — CHỈ đếm records kể từ watermark step trước."""
        return self._filter_and_count(self._records[self._step_boundary:], task_id, key_fn)

    def full_history_counts(self, task_id: str, key_fn) -> Tuple[Dict[str, int], int]:
        """Dùng cho report — đếm TOÀN BỘ records kể từ lần reset() gần nhất."""
        return self._filter_and_count(self._records, task_id, key_fn)

    def well_form_rate_by_intended_action(self) -> Dict[str, Dict[str, Any]]:
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "well_formed": 0})
        for r in self._records:
            if r.intended_action_type is None:
                continue
            entry = counts[r.intended_action_type]
            entry["total"] += 1
            if r.well_formed:
                entry["well_formed"] += 1
        return {
            a: {**e, "well_form_rate": (e["well_formed"] / e["total"] if e["total"] else 0.0)}
            for a, e in counts.items()
        }

    def print_summary(self) -> None:
        print("=== [reward v2] StatsCollectorV2 summary ===")
        for task_id in TASKS:
            n_task = sum(1 for r in self._records if r.task_id == task_id)
            n_wf = sum(1 for r in self._records if r.task_id == task_id and r.well_formed)
            n_sem = sum(1 for r in self._records if r.task_id == task_id and r.well_formed and r.semantic_passed)
            print(f"\n--- task={task_id} (n={n_task}) ---")
            if n_task:
                print(f"  well_form_rate = {n_wf / n_task * 100:.1f}%")
            if n_wf:
                print(f"  semantic_pass_rate (trong số well-formed) = {n_sem / n_wf * 100:.1f}%")

        print("\n-- Action group counts (7 nhóm, toàn bộ lịch sử từ lần reset gần nhất) --")
        action_counts, action_total = self.full_history_counts(TASK_ACTION, key_fn=lambda r: r.action_type)
        for g in GROUPS_ACTION:
            n = action_counts.get(g, 0)
            ratio = n / action_total if action_total else 0.0
            print(f"  {g:<12} count={n:<6} ratio={ratio * 100:5.1f}%")

        print("\n-- Zone group counts (HAS_ZONE/NO_ZONE) --")
        zone_counts, zone_total = self.full_history_counts(
            TASK_ZONE,
            key_fn=lambda r: "HAS_ZONE" if r.has_zone else ("NO_ZONE" if r.has_zone is False else None),
        )
        for g in GROUPS_ZONE:
            n = zone_counts.get(g, 0)
            ratio = n / zone_total if zone_total else 0.0
            print(f"  {g:<12} count={n:<6} ratio={ratio * 100:5.1f}%")

        print("\n-- Well-form rate theo Ý ĐỊNH action (kể cả parse fail) --")
        for action, stat in sorted(self.well_form_rate_by_intended_action().items()):
            print(
                f"  {action:<12} total={stat['total']:<6} well_formed={stat['well_formed']:<6} "
                f"rate={stat['well_form_rate'] * 100:5.1f}%"
            )

    def to_list(self) -> List[dict]:
        return [asdict(r) for r in self._records]

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"records": self.to_list()}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "StatsCollectorV2":
        collector = cls()
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for d in data.get("records", []):
                collector.log(TaskRolloutMeta(**d))
        return collector

    @classmethod
    def merge_from_files(cls, paths) -> "StatsCollectorV2":
        collector = cls()
        for path in paths:
            p = Path(path)
            if not p.exists():
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            for d in data.get("records", []):
                collector.log(TaskRolloutMeta(**d))
        return collector


# =====================================================================
# Instance toàn cục — action/zone buff TÁCH RIÊNG, RREntropyController
# TÁCH RIÊNG khỏi instance của v1 (KHÔNG share global với reward_func.py)
# để cô lập state, tránh 2 hệ thống (nếu cùng import trong 1 process) vô
# tình ảnh hưởng lẫn nhau.
# =====================================================================
action_buff_controller = EMABuffControllerV2(groups=GROUPS_ACTION, namespace="action")
zone_buff_controller = EMABuffControllerV2(groups=GROUPS_ZONE, namespace="zone")
rr_entropy_controller_v2 = RREntropyController()
stats_collector_v2 = StatsCollectorV2()

_active_round_config_v2: Optional[RoundConfigV2] = None


def set_active_round_config_v2(config: RoundConfigV2) -> None:
    global _active_round_config_v2
    _active_round_config_v2 = config


def get_active_round_config_v2() -> RoundConfigV2:
    if _active_round_config_v2 is None:
        raise RuntimeError("Chưa load RoundConfigV2 — gọi set_active_round_config_v2(RoundConfigV2.load(path)) trước.")
    return _active_round_config_v2


# =====================================================================
# Orchestration — nơi DUY NHẤT ráp tầng 1+2+3 lại với nhau.
# =====================================================================

def score_completion_v2(
    prompt: str,
    completion: str,
    future_bins: Sequence[Sequence[int]],
    task_id: str,
    round_config: RoundConfigV2,
    action_buffs: EMABuffControllerV2,
    zone_buffs: EMABuffControllerV2,
    stats: Optional[StatsCollectorV2] = None,
) -> Tuple[float, TaskRolloutMeta]:
    intended_action = _extract_intended_action(completion)
    gate = common_gate(prompt, completion, round_config)

    if not gate.passed:
        program = gate.program
        meta = TaskRolloutMeta(
            task_id=task_id,
            trend=program.think.trend if (program and program.think) else None,
            action_type=program.action.action_type if (program and program.action) else None,
            intended_action_type=intended_action,
            well_formed=gate.well_formed, semantic_passed=False, task_passed=None,
            sl_valid=None, rr=None, has_zone=None,
            zone_quality=None, entry_quality=None, outcome=None,
            buff_applied=None, reward=gate.gate_score,
        )
        if stats is not None:
            stats.log(meta)
        return gate.gate_score, meta

    think, action = gate.program.think, gate.program.action

    if task_id == TASK_ZONE:
        zone_score = compute_zone_task_score(gate.program, future_bins)
        buffed = apply_zone_buff(zone_score.has_zone, zone_score.zone_quality, zone_buffs)
        reward = gate.gate_score + buffed
        meta = TaskRolloutMeta(
            task_id=task_id, trend=think.trend, action_type=action.action_type,
            intended_action_type=intended_action,
            well_formed=True, semantic_passed=True, task_passed=None,
            sl_valid=None, rr=None, has_zone=zone_score.has_zone,
            zone_quality=zone_score.zone_quality, entry_quality=None, outcome=None,
            buff_applied=buffed - zone_score.zone_quality, reward=reward,
        )
        if stats is not None:
            stats.log(meta)
        return reward, meta

    # task_id == TASK_ACTION
    action_score = compute_action_task_score(action, think, future_bins, round_config)

    if not action_score.task_passed:
        # SL invalid (hoặc guard edge-case) -> fold vào NHÁNH SEMANTIC FAIL,
        # y hệt cách v1 xử lý overall_semantic_passed=False khi extra_valid
        # fail — dùng lại ĐÚNG EXTRA_SEMANTIC_PENALTY, không có hằng số
        # riêng nào khác cho SL.
        folded = max(0.0, gate.semantic_result.score - EXTRA_SEMANTIC_PENALTY)
        reward = R_WF_FULL + folded
        meta = TaskRolloutMeta(
            task_id=task_id, trend=think.trend, action_type=action.action_type,
            intended_action_type=intended_action,
            well_formed=True, semantic_passed=False, task_passed=False,
            sl_valid=action_score.sl_valid, rr=action_score.rr, has_zone=None,
            zone_quality=None, entry_quality=None, outcome=None,
            buff_applied=None, reward=reward,
        )
        if stats is not None:
            stats.log(meta)
        return reward, meta

    buffed = apply_action_buff(action_score.action_type, action_score.raw_score, action_buffs)
    reward = gate.gate_score + buffed
    meta = TaskRolloutMeta(
        task_id=task_id, trend=think.trend, action_type=action.action_type,
        intended_action_type=intended_action,
        well_formed=True, semantic_passed=True, task_passed=True,
        sl_valid=action_score.sl_valid, rr=action_score.rr, has_zone=None,
        zone_quality=None, entry_quality=action_score.entry_quality, outcome=action_score.outcome,
        buff_applied=buffed - action_score.raw_score, reward=reward,
    )
    if stats is not None:
        stats.log(meta)
    return reward, meta


# =====================================================================
# TẦNG 4 — Group-level shaping (RR entropy). KHÁC HẲN buff: input là toàn
# bộ rollout-group (mọi completion cùng 1 prompt trong batch), không phải
# 1 sample đơn lẻ — không thể/không nên nhét vào compute_action_task_score
# hay apply_action_buff (2 hàm đó chỉ thấy 1 sample tại 1 thời điểm).
# =====================================================================

def unified_reward_func_v2(
    prompts: Sequence[Any],
    completions: Sequence[str],
    future_bins: Sequence[Sequence[Sequence[int]]],
    task_id: Sequence[str],
    **kwargs: Any,
) -> List[float]:
    round_config = get_active_round_config_v2()
    n = len(prompts)

    rewards: List[float] = [0.0] * n
    metas: List[Optional[TaskRolloutMeta]] = [None] * n

    for i in range(n):
        reward, meta = score_completion_v2(
            prompts[i], completions[i], future_bins[i], task_id[i],
            round_config, action_buff_controller, zone_buff_controller, stats_collector_v2,
        )
        rewards[i] = reward
        metas[i] = meta

    # Group theo prompt CHỈ TRONG PHẠM VI task=action (RR chỉ tồn tại ở
    # action-task) — lọc task_id TRƯỚC khi group để tránh trộn nhầm với
    # task=zone dù 2 row có thể cùng prompt-string (xem cảnh báo thiết kế
    # về hành vi group-by-prompt của GRPOTrainer — CHƯA verify bằng test
    # thực nghiệm, cần làm trước khi tin tưởng tuyệt đối).
    groups_idx: Dict[Any, List[int]] = defaultdict(list)
    for i in range(n):
        if task_id[i] == TASK_ACTION and metas[i].rr is not None:
            groups_idx[prompts[i]].append(i)

    strength = rr_entropy_controller_v2.get_bonus()
    for idx_list in groups_idx.values():
        if len(idx_list) < MIN_SAMPLES_FOR_RR_ENTROPY:
            continue

        rr_list = [metas[i].rr for i in idx_list]
        h, probs = _entropy_and_probs(rr_list)
        rr_entropy_controller_v2.record_entropy(h)

        if strength <= 0.0:
            continue

        for i in idx_list:
            rr_i = metas[i].rr
            surprisal = -math.log(probs[rr_i])
            rewards[i] += strength * surprisal

    return rewards