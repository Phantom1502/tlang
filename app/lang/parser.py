from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.lang.ast_nodes import (
    ActionNode,
    CandleNode,
    ChartNode,
    ProgramNode,
    ThinkNode,
    ZoneNode,
)
from app.lang.lexer import Lexer
from app.lang.tokens import Token, TokenType


# =====================================================================
# Kết quả parse — well-form score LIÊN TỤC theo số lỗi & mức độ nghiêm
# trọng, không phải nhị phân 0/1 (để GRPO reward có gradient mượt thay
# vì reward cực thưa).
#
# Trọng số phạt cụ thể (SEVERITY_PENALTY) là placeholder ban đầu — sẽ
# tinh chỉnh sau khi có dữ liệu thực nghiệm từ vài round GRPO đầu
# (xem spec mục 9, câu hỏi còn mở #5 về trọng số reward).
# =====================================================================
@dataclass
class ParseError:
    message: str
    position: int
    severity: str = "structural"   # "structural" (lỗi cú pháp thường) | "value" (lỗi nội dung, nặng hơn)


@dataclass
class ParseResult:
    ast: Optional[ProgramNode]
    errors: List[ParseError] = field(default_factory=list)

    SEVERITY_PENALTY = {"structural": 0.15, "value": 0.30}

    def is_well_formed(self) -> bool:
        return len(self.errors) == 0

    def well_form_score(self) -> float:
        penalty = sum(self.SEVERITY_PENALTY.get(e.severity, 0.15) for e in self.errors)
        return max(0.0, 1.0 - penalty)


class Parser:
    """
    Recursive-descent parser cho grammar:

        program      := chart_block think_block action_block
        chart_block   := "<chart>" candle{50} "</chart>"
        candle        := CANDLE_O CANDLE_H CANDLE_L CANDLE_C
        think_block   := "<think>" trend current_price zone? price_in_zone? good_price_action? "</think>"
        action_block  := "<action>" ACTION_TYPE [ SL RR ] "</action>"

    Dùng panic-mode error recovery (không hard-fail như compiler thật):
    khi gặp token sai, ghi nhận lỗi rồi bỏ qua token tới điểm đồng bộ hoá
    gần nhất, tiếp tục parse phần còn lại — để 1 completion nhiều lỗi vẫn
    có well_form_score liên tục thay vì tất cả về 0 giống nhau.

    Bảng 2.2.C (current_price phải khớp Close nến cuối) và bảng 2.2.F
    (field bắt buộc/cấm theo ACTION_TYPE) được kiểm tra ngay trong lớp
    này — về bản chất vẫn là "đúng/sai ngữ pháp có điều kiện", chưa đánh
    giá chất lượng quyết định (đó là việc của Semantic Checker riêng,
    kiểm tra bảng A/B/D/E).
    """

    EXPECTED_CANDLE_COUNT = 50
    VALID_BIN_RANGE: Tuple[int, int] = (0, 1023)
    VALID_RR_RANGE: Tuple[int, int] = (1, 9)

    # Token dùng để đồng bộ hoá khi panic-mode — đều là ranh giới block rõ ràng.
    SYNC_TOKENS = {
        TokenType.CHART_CLOSE,
        TokenType.THINK_OPEN,
        TokenType.THINK_CLOSE,
        TokenType.ACTION_OPEN,
        TokenType.ACTION_CLOSE,
        TokenType.EOF,
    }

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.errors: List[ParseError] = []

    @classmethod
    def from_text(cls, text: str) -> "Parser":
        return cls(Lexer(text).tokenize())

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.type != TokenType.EOF:
            self.pos += 1
        return tok

    def _check(self, *types: TokenType) -> bool:
        return self._current().type in types

    def _error(self, message: str, severity: str = "structural") -> None:
        self.errors.append(ParseError(message=message, position=self._current().position, severity=severity))

    def _synchronize(self) -> None:
        """Bỏ token cho tới khi gặp điểm đồng bộ hoá (luôn dừng lại vì EOF nằm trong SYNC_TOKENS)."""
        while not self._check(*self.SYNC_TOKENS):
            self._advance()

    # ------------------------------------------------------------------
    # Grammar rules
    # ------------------------------------------------------------------
    def parse(self) -> ParseResult:
        chart = self._parse_chart_block()
        think = self._parse_think_block()
        action = self._parse_action_block()

        program = ProgramNode(chart=chart, think=think, action=action)

        # Bảng 2.2.C + 2.2.F — vẫn thuộc well-form, chạy ngay sau khi có đủ AST.
        self._check_current_price_matches_chart(chart, think)
        self._check_action_field_consistency(think, action)

        if not self._check(TokenType.EOF):
            self._error(f"Dư thừa token sau khi parse hết action_block: {self._current().type.name}")

        return ParseResult(ast=program, errors=self.errors)

    def _parse_chart_block(self) -> Optional[ChartNode]:
        if not self._check(TokenType.CHART_OPEN):
            self._error(f"Mong đợi <chart>, nhận được {self._current().type.name}")
            self._synchronize()
            return None
        self._advance()

        candles: List[CandleNode] = []
        while self._check(TokenType.CANDLE_O):
            candle = self._parse_candle()
            if candle is not None:
                candles.append(candle)

        if len(candles) != self.EXPECTED_CANDLE_COUNT:
            self._error(
                f"Số nến trong chart_block = {len(candles)}, mong đợi {self.EXPECTED_CANDLE_COUNT}",
                severity="value",
            )

        if not self._check(TokenType.CHART_CLOSE):
            self._error(f"Mong đợi </chart>, nhận được {self._current().type.name}")
            self._synchronize()
        else:
            self._advance()

        return ChartNode(candles=candles)

    def _parse_candle(self) -> Optional[CandleNode]:
        o = self._expect_bin(TokenType.CANDLE_O, "O")
        h = self._expect_bin(TokenType.CANDLE_H, "H")
        l = self._expect_bin(TokenType.CANDLE_L, "L")
        c = self._expect_bin(TokenType.CANDLE_C, "C")
        if None in (o, h, l, c):
            return None
        return CandleNode(o=o, h=h, l=l, c=c)

    def _expect_bin(self, token_type: TokenType, label: str) -> Optional[int]:
        if not self._check(token_type):
            self._error(f"Thiếu token {label} trong candle (nhận {self._current().type.name})")
            return None
        tok = self._advance()
        value = self._extract_int(tok.value)
        if value is None or not (self.VALID_BIN_RANGE[0] <= value <= self.VALID_BIN_RANGE[1]):
            self._error(f"Giá trị bin {label} ngoài phạm vi hợp lệ [0,1023]: {tok.value}", severity="value")
            return None
        return value

    def _parse_think_block(self) -> Optional[ThinkNode]:
        if not self._check(TokenType.THINK_OPEN):
            self._error(f"Mong đợi <think>, nhận được {self._current().type.name}")
            self._synchronize()
            return None
        self._advance()

        think = ThinkNode()

        if not self._check(TokenType.TREND):
            self._error("Thiếu <trend> trong think_block")
        else:
            tok = self._advance()
            think.trend = self._extract_enum(tok.value, ("UP", "DOWN", "RANGE"))

        if not self._check(TokenType.CURRENT_PRICE):
            self._error("Thiếu <current_price> — field này BẮT BUỘC trong mọi think_block", severity="value")
        else:
            tok = self._advance()
            think.current_price_bin = self._extract_int(tok.value)

        if self._check(TokenType.ZONE_SUPPORT, TokenType.ZONE_RESISTANCE):
            tok = self._advance()
            direction = "support" if tok.type == TokenType.ZONE_SUPPORT else "resistance"
            lower, upper = self._extract_two_ints(tok.value)
            think.zone = ZoneNode(direction=direction, lower_bin=lower, upper_bin=upper)

        if self._check(TokenType.PRICE_IN_ZONE):
            self._advance()
            think.price_in_zone = True

        if self._check(TokenType.GOOD_PRICE_ACTION):
            self._advance()
            think.good_price_action = True

        if not self._check(TokenType.THINK_CLOSE):
            self._error(f"Mong đợi </think>, nhận được {self._current().type.name}")
            self._synchronize()
        else:
            self._advance()

        return think

    def _parse_action_block(self) -> Optional[ActionNode]:
        if not self._check(TokenType.ACTION_OPEN):
            self._error(f"Mong đợi <action>, nhận được {self._current().type.name}")
            self._synchronize()
            return None
        self._advance()

        action = ActionNode()

        if not self._check(TokenType.ACTION_TYPE):
            self._error("Thiếu ACTION_TYPE trong action_block")
        else:
            tok = self._advance()
            action.action_type = tok.value.strip()

        if self._check(TokenType.SL):
            tok = self._advance()
            action.sl = self._extract_int(tok.value)

        if self._check(TokenType.RR):
            tok = self._advance()
            rr_val = self._extract_int(tok.value)
            if rr_val is not None and not (self.VALID_RR_RANGE[0] <= rr_val <= self.VALID_RR_RANGE[1]):
                self._error(f"RR={rr_val} ngoài phạm vi vocab hợp lệ (1-9)", severity="value")
            action.rr = rr_val

        if not self._check(TokenType.ACTION_CLOSE):
            self._error(f"Mong đợi </action>, nhận được {self._current().type.name}")
            self._synchronize()
        else:
            self._advance()

        return action

    # ------------------------------------------------------------------
    # Bảng 2.2.C — current_price phải khớp tuyệt đối Close nến cuối
    # ------------------------------------------------------------------
    def _check_current_price_matches_chart(self, chart: Optional[ChartNode], think: Optional[ThinkNode]) -> None:
        if chart is None or think is None:
            return
        if not chart.candles or think.current_price_bin is None:
            return
        real_close = chart.candles[-1].c
        if think.current_price_bin != real_close:
            self._error(
                f"current_price ({think.current_price_bin}) không khớp Close nến cuối thực tế ({real_close})",
                severity="value",  # nặng hơn lỗi cú pháp thuần tuý — phản ánh model đọc sai input
            )

    # ------------------------------------------------------------------
    # Bảng 2.2.F — field bắt buộc/cấm theo ACTION_TYPE
    # ------------------------------------------------------------------
    def _check_action_field_consistency(self, think: Optional[ThinkNode], action: Optional[ActionNode]) -> None:
        if think is None or action is None or action.action_type is None:
            return

        action_type = action.action_type
        requires_full = action_type in ("BUY", "SELL")
        requires_empty = action_type in ("CANCEL_BUY", "CANCEL_SELL", "WAIT_BUY", "WAIT_SELL", "HOLD")

        if requires_full:
            if action.sl is None:
                self._error(f"Thiếu SL bắt buộc cho action={action_type}")
            if action.rr is None:
                self._error(f"Thiếu RR bắt buộc cho action={action_type}")
            if not think.good_price_action:
                self._error(f"Thiếu <good_price_action> bắt buộc cho action={action_type}")

        if requires_empty:
            if action.sl is not None:
                self._error(f"SL không được xuất hiện với action={action_type}")
            if action.rr is not None:
                self._error(f"RR không được xuất hiện với action={action_type}")
            if think.good_price_action:
                self._error(f"<good_price_action> không được xuất hiện với action={action_type}")

    # ------------------------------------------------------------------
    # Tiện ích trích xuất giá trị từ raw token text
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_int(raw: Optional[str]) -> Optional[int]:
        if raw is None:
            return None
        m = re.search(r"\d+", raw)
        return int(m.group()) if m else None

    @staticmethod
    def _extract_two_ints(raw: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
        if raw is None:
            return None, None
        nums = re.findall(r"\d+", raw)
        if len(nums) < 2:
            return None, None
        return int(nums[0]), int(nums[1])

    @staticmethod
    def _extract_enum(raw: Optional[str], choices: Tuple[str, ...]) -> Optional[str]:
        if raw is None:
            return None
        for choice in choices:
            if choice in raw:
                return choice
        return None
