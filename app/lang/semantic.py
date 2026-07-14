from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.lang.ast_nodes import ChartNode, ProgramNode, ThinkNode, ActionNode


# =====================================================================
# SemanticResult — passed CHỈ true khi KHÔNG có vi phạm nào (100%, theo
# quyết định đã chốt: gate 2 yêu cầu pass toàn bộ mới cho phép tính
# outcome, không dùng ngưỡng %). `score` vẫn liên tục (dùng cho nhánh
# fail, R_sem_fail) để reward không quá thưa.
# =====================================================================
@dataclass
class SemanticResult:
    passed: bool
    violations: List[str] = field(default_factory=list)
    score: float = 1.0


class SemanticChecker:
    """
    Kiểm tra bảng 2.2 (A, B, D, E) trên AST đã parse thành công.

    KHÔNG kiểm tra bảng F (field bắt buộc/cấm theo action_type — đã ở
    well-form, thuộc Parser) và KHÔNG kiểm tra mục G (good_price_action
    không có rule nội dung, chủ ý để tránh áp đặt bias chủ quan).

    Nguyên tắc: verifier này = "lật ngược" generator dùng để sinh dữ
    liệu SFT/pretrain — generator đảm bảo đúng các invariant này lúc
    sinh, verifier chỉ cần lật ngược logic đó thành kiểm tra.
    """

    VIOLATION_PENALTY = 0.2       # placeholder — tinh chỉnh sau khi có dữ liệu GRPO thực nghiệm
    LAST_N_CANDLES_TOUCH = 5

    BUY_SIDE_ACTIONS = {"BUY", "CANCEL_BUY"}
    SELL_SIDE_ACTIONS = {"SELL", "CANCEL_SELL"}

    def check(self, program: ProgramNode) -> SemanticResult:
        chart, think, action = program.chart, program.think, program.action
        violations: List[str] = []

        # Phòng vệ: thiếu thành phần cơ bản để đánh giá — lẽ ra đã bị
        # well-form chặn từ trước (Semantic Checker chỉ nên chạy khi
        # well-form đã pass), nhưng vẫn xử lý an toàn nếu bị gọi độc lập.
        if chart is None or think is None or action is None:
            return SemanticResult(passed=False, violations=["Thiếu chart/think/action — không thể kiểm tra semantic"], score=0.0)
        if not chart.candles or think.trend is None or think.current_price_bin is None or action.action_type is None:
            return SemanticResult(
                passed=False,
                violations=["Thiếu trend/current_price/action_type/candles — không thể kiểm tra semantic"],
                score=0.0,
            )

        self._check_trend_zone(think, violations)
        self._check_zone_direction_vs_price(think, violations)
        expected_price_in_zone = self._check_price_in_zone_geometry(chart, think, violations)
        self._check_action_group(think, action, violations, expected_price_in_zone)

        passed = len(violations) == 0
        score = max(0.0, 1.0 - self.VIOLATION_PENALTY * len(violations))
        return SemanticResult(passed=passed, violations=violations, score=score)

    # ------------------------------------------------------------------
    # A. Trend ↔ Zone
    # ------------------------------------------------------------------
    def _check_trend_zone(self, think: ThinkNode, violations: List[str]) -> None:
        trend = think.trend
        zone = think.zone

        if trend == "UP":
            if zone is None:
                violations.append("trend=UP nhưng thiếu zone (bắt buộc phải có zone_support)")
            elif zone.direction != "support":
                violations.append(f"trend=UP nhưng zone lại là {zone.direction} (phải là zone_support)")

        elif trend == "DOWN":
            if zone is None:
                violations.append("trend=DOWN nhưng thiếu zone (bắt buộc phải có zone_resistance)")
            elif zone.direction != "resistance":
                violations.append(f"trend=DOWN nhưng zone lại là {zone.direction} (phải là zone_resistance)")

        elif trend == "RANGE":
            # RANGE: zone tùy chọn, cả 2 hướng đều hợp lệ nếu có — không có vi phạm ở mục A.
            pass

    # ------------------------------------------------------------------
    # B. Hướng của Zone ↔ current_price (bin arithmetic thuần túy)
    # ------------------------------------------------------------------
    def _check_zone_direction_vs_price(self, think: ThinkNode, violations: List[str]) -> None:
        zone = think.zone
        if zone is None:
            return
        current = think.current_price_bin

        if zone.direction == "support":
            if not (zone.lower_bin <= current):
                violations.append(
                    f"zone_support ({zone.lower_bin}:{zone.upper_bin}) nằm hoàn toàn trên current_price "
                    f"({current}) — zone_support phải nằm dưới hoặc chứa giá hiện tại"
                )
        else:  # resistance
            if not (zone.upper_bin >= current):
                violations.append(
                    f"zone_resistance ({zone.lower_bin}:{zone.upper_bin}) nằm hoàn toàn dưới current_price "
                    f"({current}) — zone_resistance phải nằm trên hoặc chứa giá hiện tại"
                )

    # ------------------------------------------------------------------
    # D. price_in_zone ↔ hình học thật
    # ------------------------------------------------------------------
    def _check_price_in_zone_geometry(
        self, chart: ChartNode, think: ThinkNode, violations: List[str]
    ) -> Optional[bool]:
        zone = think.zone
        if zone is None:
            # Không có zone thì price_in_zone không có ý nghĩa — nếu model vẫn set thì coi là vi phạm nhẹ.
            if think.price_in_zone:
                violations.append("<price_in_zone> xuất hiện nhưng think_block không có zone nào")
            return None

        current = think.current_price_bin
        if zone.lower_bin <= current <= zone.upper_bin:
            expected = True
        else:
            expected = self._last_n_candles_touch_zone(chart, zone.lower_bin, zone.upper_bin)

        if think.price_in_zone != expected:
            violations.append(
                f"<price_in_zone>={think.price_in_zone} không khớp sự thật hình học "
                f"(mong đợi={expected}, zone={zone.lower_bin}:{zone.upper_bin}, current_price={current})"
            )
        return expected

    def _last_n_candles_touch_zone(self, chart: ChartNode, zone_lower: int, zone_upper: int) -> bool:
        last_candles = chart.candles[-self.LAST_N_CANDLES_TOUCH:]
        for candle in last_candles:
            # Nến "chạm" zone nếu khoảng [low, high] của nến giao với [zone_lower, zone_upper]
            if candle.l <= zone_upper and candle.h >= zone_lower:
                return True
        return False

    # ------------------------------------------------------------------
    # E. price_in_zone ↔ nhóm action hợp lệ
    # ------------------------------------------------------------------
    def _check_action_group(
        self,
        think: ThinkNode,
        action: ActionNode,
        violations: List[str],
        expected_price_in_zone: Optional[bool],
    ) -> None:
        zone = think.zone
        action_type = action.action_type

        if zone is None:
            # RANGE không có zone -> chỉ HOLD được phép.
            # UP/DOWN thiếu zone đã bị mục A bắt rồi — không kiểm tra trùng ở đây để tránh
            # báo lỗi 2 lần cho cùng 1 nguyên nhân gốc.
            if think.trend == "RANGE" and action_type != "HOLD":
                violations.append(f"RANGE không có zone thì action phải là HOLD, nhận được {action_type}")
            return

        # Dùng đúng field price_in_zone (những gì model đã khẳng định), không phải giá trị
        # "thật" — vì mục D đã tách riêng việc kiểm tra field có khớp sự thật hay không.
        piz = think.price_in_zone

        if zone.direction == "support":
            valid_actions = self.BUY_SIDE_ACTIONS if piz else {"WAIT_BUY"}
        else:  # resistance
            valid_actions = self.SELL_SIDE_ACTIONS if piz else {"WAIT_SELL"}

        if action_type not in valid_actions:
            violations.append(
                f"zone={zone.direction}, price_in_zone={piz} thì action hợp lệ phải thuộc "
                f"{sorted(valid_actions)}, nhận được {action_type}"
            )