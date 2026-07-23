from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
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
# WAIT_BUY/WAIT_SELL/HOLD KHÔNG đóng góp vào outcome_score.
OUTCOME_ACTIONS = ("BUY", "SELL")

_ACTION_TYPE_RE = re.compile(r"\b(CANCEL_BUY|CANCEL_SELL|WAIT_BUY|WAIT_SELL|BUY|SELL|HOLD)\b")


def _extract_intended_action(completion: str) -> Optional[str]:
    """Token ACTION_TYPE đầu tiên xuất hiện trong completion, bất kể completion
    có well-formed hay không — best-effort, chỉ dùng cho thống kê/report."""
    m = _ACTION_TYPE_RE.search(completion)
    return m.group(1) if m else None


# =====================================================================
# EMABuffController — THAY HOÀN TOÀN cho ActionBuffTable + HoldBuff +
# update_buffs_from_stats (bang-bang theo chu kỳ save_steps) trước đây.
#
# 4 nhóm action: HOLD | TRADE (BUY+SELL gộp chung 1 buff) | CANCEL
# (CANCEL_BUY+CANCEL_SELL) | WAIT (WAIT_BUY+WAIT_SELL). BUY/SELL gộp chung
# vì trong cơ chế cũ, buff của 2 key "BUY"/"SELL" LUÔN được cộng cùng 1
# delta (cùng actual_ratio = (BUY+SELL)/tổng) — tức về bản chất đã luôn là
# 1 buff dùng chung, chỉ lưu dưới 2 key trùng giá trị. Gộp lại ở đây là
# đơn giản hoá hợp lệ, không đổi ngữ nghĩa.
#
# Cơ chế, TÁCH RỜI hẳn khỏi save_steps:
#   1. record(action_type): gọi trong score_completion() cho MỖI sample
#      well_formed + semantic_passed — đúng logic action_counts() cũ.
#   2. on_step_end(round_config): gọi 1 LẦN / optimizer step (KHÔNG phải
#      mỗi save_steps) — với MỖI nhóm, dùng bộ điều khiển PD (Proportional +
#      Derivative, KHÔNG có thành phần I):
#         rate_this_step = count_nhóm_trong_step / tổng_count_trong_step
#         ema_ratio = (1-alpha)*rate_this_step + alpha*ema_ratio_cũ
#         error      = target - ema_ratio
#         d_error    = error - prev_error          # THÊM — tốc độ error đang co/giãn
#         delta = clip(kp*error + kd*d_error, -step_max, +step_max)
#         buff = clip(buff + delta, group_min, group_max)
#         prev_error = error                        # lưu lại cho step sau
#
#      Ý NGHĨA D-TERM (buff_kd): P-only (buff_kd=0, hành vi cũ) chỉ phản ứng
#      theo ĐỘ LỆCH hiện tại — khi ratio đang lao nhanh về phía target, nó vẫn
#      tiếp tục đẩy buff cùng chiều tới tận khi VƯỢT QUA target mới bắt đầu
#      kéo ngược lại (overshoot, dao động không tắt dần — quan sát thấy rõ ở
#      HOLD trong log thực tế). D-term "phanh sớm": khi error đang co lại
#      nhanh (d_error âm lớn — ratio đang tiến rất nhanh về target), nó chủ
#      động trừ bớt delta NGAY TỪ TRƯỚC khi ratio kịp vượt target, giảm biên
#      độ overshoot mà không cần hạ buff_kp/trần (đổi lại: nhạy nhiễu hơn P
#      thuần — vì d_error tính trên ema_ratio đã làm mượt qua EMA nên đỡ giật
#      hơn nhiều so với lấy đạo hàm trên raw rate_this_step trực tiếp).
#
# State (ema_ratio + buff + prev_error, MỌI nhóm) PHẢI persist vào
# reward_state.json BÊN TRONG thư mục checkpoint (không phải ở output_dir
# rời như StatsCollector trước đây) — xem save()/load() và cách gọi ở
# train_grpo.py. Đây là state DUY NHẤT cần sống sót qua resume; StatsCollector
# giờ CHỈ còn vai trò report (xem docstring StatsCollector bên dưới).
#
# TƯƠNG THÍCH NGƯỢC: reward_state.json cũ (trước khi thêm D-term) không có
# "prev_error" — load_state_dict() dùng .get("prev_error", 0.0) nên load vẫn
# chạy được bình thường, chỉ là D-term coi như "khởi động lại từ 0" ở step
# đầu tiên sau resume (KHÔNG crash, KHÔNG mất buff/ema_ratio đã có).
# =====================================================================
GROUPS = ("HOLD", "TRADE", "CANCEL", "WAIT")

GROUP_OF_ACTION: Dict[str, str] = {
    "HOLD": "HOLD",
    "BUY": "TRADE", "SELL": "TRADE",
    "CANCEL_BUY": "CANCEL", "CANCEL_SELL": "CANCEL",
    "WAIT_BUY": "WAIT", "WAIT_SELL": "WAIT",
}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _group_target(group: str, rc: RoundConfig) -> float:
    return {
        "HOLD": rc.target_hold_ratio, "TRADE": rc.target_trade_ratio,
        "CANCEL": rc.target_cancel_ratio, "WAIT": rc.target_wait_ratio,
    }[group]


def _group_range(group: str, rc: RoundConfig) -> Tuple[float, float]:
    return {
        "HOLD": (rc.hold_buff_min, rc.hold_buff_max),
        "TRADE": (rc.trade_buff_min, rc.trade_buff_max),
        "CANCEL": (rc.cancel_buff_min, rc.cancel_buff_max),
        "WAIT": (rc.wait_buff_min, rc.wait_buff_max),
    }[group]


def _group_init(group: str, rc: RoundConfig) -> float:
    return {
        "HOLD": rc.hold_buff_init, "TRADE": rc.trade_buff_init,
        "CANCEL": rc.cancel_buff_init, "WAIT": rc.wait_buff_init,
    }[group]


@dataclass
class GroupBuffState:
    ema_ratio: float
    buff: float
    prev_error: float = 0.0   # THÊM — error (target - ema_ratio) của lần update trước, dùng cho D-term


class EMABuffController:
    def __init__(self):
        self.states: Dict[str, GroupBuffState] = {}
        self._counts: Dict[str, int] = defaultdict(int)
        self._total: int = 0

    def seed_from_round_config(self, round_config: RoundConfig) -> None:
        """Dùng khi round MỚI bắt đầu (không có state cũ để resume) — seed
        ema_ratio = target (giả định 'đã ở điểm cân bằng' lúc khởi động, tránh
        vài trăm step đầu buff phản ứng nhầm hướng do ema=0), buff = group_init
        (dò bằng thực nghiệm trước khi train thật), prev_error = 0.0 (chưa có
        error nào trước đó — D-term coi như "phẳng" ở step đầu tiên, đúng
        hành vi mong đợi khi seed lần đầu)."""
        for group in GROUPS:
            self.states[group] = GroupBuffState(
                ema_ratio=_group_target(group, round_config),
                buff=_group_init(group, round_config),
                prev_error=0.0,
            )
        self._counts.clear()
        self._total = 0

    def record(self, action_type: Optional[str]) -> None:
        group = GROUP_OF_ACTION.get(action_type) if action_type else None
        if group is None:
            return
        self._counts[group] += 1
        self._total += 1

    def on_step_end(self, round_config: RoundConfig) -> None:
        """Gọi 1 lần / optimizer step (TrainerCallback.on_step_end). Nếu step
        này KHÔNG có sample nào well_formed+semantic_passed (total=0) thì bỏ
        qua — giữ nguyên ema/buff/prev_error cũ, tránh update dựa trên rate=0/0
        vô nghĩa (và tránh d_error bị tính sai do "khoảng trống" không phản
        ánh biến động thật)."""
        if self._total == 0:
            return
        for group in GROUPS:
            rate_this_step = self._counts.get(group, 0) / self._total
            st = self.states[group]
            st.ema_ratio = (1.0 - round_config.ema_alpha) * rate_this_step + round_config.ema_alpha * st.ema_ratio
            lo, hi = _group_range(group, round_config)

            error = _group_target(group, round_config) - st.ema_ratio
            d_error = error - st.prev_error
            st.prev_error = error

            delta = round_config.buff_kp * error + round_config.buff_kd * d_error
            delta = _clip(delta, -round_config.buff_step_max, round_config.buff_step_max)
            st.buff = _clip(st.buff + delta, lo, hi)
        self._counts.clear()
        self._total = 0

    def get_buff(self, action_type: Optional[str]) -> float:
        group = GROUP_OF_ACTION.get(action_type) if action_type else None
        if group is None or group not in self.states:
            return 0.0
        return self.states[group].buff

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
                # .get() — tương thích ngược reward_state.json cũ (trước khi có
                # D-term) không có field này. Thiếu -> coi như 0.0, KHÔNG crash.
                prev_error=float(d.get("prev_error", 0.0)),
            )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.state_dict(), ensure_ascii=False), encoding="utf-8")

    def load(self, path: str) -> bool:
        """Trả True nếu load thành công, False nếu không (thiếu file hoặc lỗi
        parse). Caller (train_grpo.py) PHẢI gọi seed_from_round_config() khi
        trả về False — KHÔNG được để states rỗng (get_buff sẽ âm thầm trả 0.0
        cho group thiếu, che mất bug thay vì báo lỗi rõ ràng)."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.load_state_dict(data)
            return True
        except Exception:
            return False


buff_controller = EMABuffController()

_active_round_config: Optional[RoundConfig] = None


def set_active_round_config(config: RoundConfig) -> None:
    """CHỈ set config — KHÔNG tự động seed/reset buff_controller (khác hành
    vi cũ của hàm này). Seed (round mới) hay load (resume) state là quyết
    định CẦN BIẾT NGỮ CẢNH CHECKPOINT, thuộc về train_grpo.py — xem docstring
    EMABuffController."""
    global _active_round_config
    _active_round_config = config


def get_active_round_config() -> RoundConfig:
    if _active_round_config is None:
        raise RuntimeError("Chưa load RoundConfig — gọi set_active_round_config(RoundConfig.load(path)) trước.")
    return _active_round_config


@dataclass
class RolloutRecord:
    trend: Optional[str]
    action_type: Optional[str]
    intended_action_type: Optional[str]
    outcome_status: Optional[str]
    r_multiple: Optional[float]
    well_formed: bool
    semantic_passed: bool
    sl_valid: Optional[bool] = None
    rr: Optional[int] = None   # THÊM — action.rr thật (chỉ có ở BUY/SELL), để theo dõi phân phối RR
                                 # trực tiếp thay vì ước lượng ngược qua avg_R/win_rate.


class StatsCollector:
    """CHỈ còn vai trò REPORT (in summary, đọc phân phối trend/action/outcome)
    — KHÔNG còn liên quan gì tới buff nữa (buff giờ do EMABuffController quản
    lý, persist riêng trong checkpoint, xem module-level docstring phía trên).
    save()/load() giờ chỉ còn field "records", không còn "action_buffs"/
    "hold_buff" như bản cũ."""

    def __init__(self):
        self._records: List[RolloutRecord] = []

    def log(self, record: RolloutRecord) -> None:
        self._records.append(record)

    def reset(self) -> None:
        self._records.clear()

    def summary(self):
        by_trend_total = defaultdict(int)
        raw = defaultdict(lambda: defaultdict(lambda: {"count": 0, "r_multiples": [], "rrs": []}))
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
            if r.rr is not None:
                entry["rrs"].append(r.rr)
        result = {}
        for trend, actions in raw.items():
            result[trend] = {}
            total = by_trend_total[trend]
            for action_type, entry in actions.items():
                rms = entry["r_multiples"]
                rrs = entry["rrs"]
                avg_r = sum(rms) / len(rms) if rms else None
                win_rate = (sum(1 for x in rms if x > 0) / len(rms)) if rms else None
                avg_rr = sum(rrs) / len(rrs) if rrs else None
                result[trend][action_type] = {
                    "count": entry["count"], "freq_within_trend": entry["count"] / total if total else 0.0,
                    "avg_r_multiple": avg_r, "win_rate": win_rate,
                    "avg_rr": avg_rr, "rr_distribution": dict(sorted(Counter(rrs).items())) if rrs else None,
                }
        return result

    def action_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for r in self._records:
            if r.action_type is None or not (r.well_formed and r.semantic_passed):
                continue
            counts[r.action_type] += 1
        return dict(counts)

    def group_counts(self) -> Dict[str, int]:
        """Đếm theo 4 nhóm (HOLD/TRADE/CANCEL/WAIT) — khớp đúng cách
        EMABuffController.record() đang đếm, tiện đối chiếu khi debug."""
        counts: Dict[str, int] = defaultdict(int)
        for action_type, n in self.action_counts().items():
            group = GROUP_OF_ACTION.get(action_type)
            if group is not None:
                counts[group] += n
        return dict(counts)

    def summary_by_intended_action(self) -> Dict[str, Dict[str, Any]]:
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
                avg_rr = f"{stat['avg_rr']:.2f}" if stat.get("avg_rr") is not None else "-"
                line = f"  {action_type:<12} count={stat['count']:<4} freq={stat['freq_within_trend']*100:5.1f}%  avg_R={avg_r:>6}  win_rate={win_rate}  avg_RR={avg_rr:>5}"
                dist = stat.get("rr_distribution")
                if dist:
                    dist_str = " ".join(f"{k}:{v}" for k, v in dist.items())
                    line += f"  rr_dist=[{dist_str}]"
                print(line)

        print("\n=== Well-form rate theo Ý ĐỊNH (intended_action_type — kể cả parse fail) ===")
        intended = self.summary_by_intended_action()
        for action, stat in sorted(intended.items()):
            print(f"  {action:<12} total={stat['total']:<6} well_formed={stat['well_formed']:<6} rate={stat['well_form_rate']*100:5.1f}%")

        gcounts = self.group_counts()
        total = sum(gcounts.values())
        print(f"\n=== Tỉ lệ theo NHÓM (dùng cho EMABuffController, n={total}) ===")
        for group in GROUPS:
            n = gcounts.get(group, 0)
            ratio = n / total if total else 0.0
            print(f"  {group:<8} count={n:<6} ratio={ratio*100:5.1f}%")

    def to_list(self):
        return [asdict(r) for r in self._records]

    def save(self, path: str) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"records": self.to_list()}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "StatsCollector":
        """Hỗ trợ NGƯỢC cả format cũ (bare list, hoặc dict có thêm
        "action_buffs"/"hold_buff" — 2 field đó giờ bị BỎ QUA, không dùng
        nữa)."""
        collector = cls()
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            records = data if isinstance(data, list) else data.get("records", [])
            for d in records:
                collector.log(RolloutRecord(**d))
        return collector

    @classmethod
    def merge_from_files(cls, paths) -> "StatsCollector":
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


# =====================================================================
# Nhánh 1 — zone quality. Áp dụng cho MỌI action có think.zone (BUY, SELL,
# CANCEL_BUY, CANCEL_SELL, WAIT_BUY, WAIT_SELL) — HOLD (RANGE không zone)
# luôn nhận 0.0.
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
# Nhánh 2 — outcome. CHỈ BUY/SELL. buffs.get_buff("BUY"/"SELL") -> cùng trả
# về buff của nhóm TRADE (xem GROUP_OF_ACTION).
# =====================================================================
def compute_outcome_score(
    action: ActionNode,
    think: ThinkNode,
    forward_result,
    round_config: RoundConfig,
    buffs: EMABuffController,
) -> float:
    if action.action_type not in OUTCOME_ACTIONS:
        return 0.0
    if forward_result is None:
        return 0.0

    buff = buffs.get_buff(action.action_type)

    risk_bins = abs(think.current_price_bin - action.sl) if action.sl is not None else 0.0
    fee_in_r = round_config.trade_fee_bins / risk_bins if risk_bins > 0 else 0.0

    #TODO: temp rebalance
    if action.action_type == "SELL":
        return forward_result.r_multiple - fee_in_r
    return forward_result.r_multiple - fee_in_r + buff


def score_completion(
    prompt: str,
    completion: str,
    future_bins: Sequence[Sequence[int]],
    stats: Optional[StatsCollector] = None,
    buffs: Optional[EMABuffController] = None,
) -> float:
    buffs = buffs if buffs is not None else buff_controller
    round_config = get_active_round_config()
    future_candles: List[FutureCandle] = [tuple(c) for c in future_bins]
    intended_action = _extract_intended_action(completion)

    parse_result = Parser.from_text(prompt + " " + completion).parse()

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

    sem_score = semantic_result.score
    if semantic_result.passed and not extra_valid:
        sem_score = max(0.0, sem_score - EXTRA_SEMANTIC_PENALTY)
    if sl_valid is True:
        sem_score += round_config.sl_valid_bonus
    elif sl_valid is False:
        sem_score -= round_config.sl_valid_penalty
    sem_score = max(0.0, sem_score)

    if not overall_semantic_passed:
        if stats is not None:
            stats.log(RolloutRecord(
                trend=trend, action_type=action_type, intended_action_type=intended_action,
                outcome_status=None, r_multiple=None,
                well_formed=True, semantic_passed=False, sl_valid=sl_valid,
            ))
        return R_WF_FULL + sem_score

    K = round_config.pass_gate2_bonus
    zone_score = compute_zone_score(think, future_candles, round_config)
    outcome_score = compute_outcome_score(action, think, forward_result, round_config, buffs)
    reward = K + zone_score + outcome_score

    if action_type == "HOLD":
        reward += buffs.get_buff("HOLD")
    elif action_type in ("CANCEL_BUY", "CANCEL_SELL"):
        reward += buffs.get_buff(action_type)
    elif action_type in ("WAIT_BUY", "WAIT_SELL"):
        reward += buffs.get_buff(action_type)

    # record() gọi trên CHÍNH object `buffs` đang dùng (không hardcode global)
    # — quan trọng để unit test truyền buffs riêng không đụng global thật.
    buffs.record(action_type)

    if stats is not None:
        stats.log(RolloutRecord(
            trend=trend, action_type=action_type, intended_action_type=intended_action,
            outcome_status=forward_result.status.value if forward_result else None,
            r_multiple=forward_result.r_multiple if forward_result else None,
            well_formed=True, semantic_passed=True, sl_valid=sl_valid,
            rr=action.rr if action_type in OUTCOME_ACTIONS else None,
        ))
    return reward


def unified_reward_func(
    prompts: Sequence[Any],
    completions: Sequence[str],
    future_bins: Sequence[Sequence[Sequence[int]]],
    **kwargs: Any,
) -> List[float]:
    return [
        score_completion(prompt, completion, fb, stats=stats_collector, buffs=buff_controller)
        for prompt, completion, fb in zip(prompts, completions, future_bins)
    ]