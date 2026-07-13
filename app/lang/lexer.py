from __future__ import annotations

import re
from typing import List

from app.lang.tokens import Token, TokenType

# =====================================================================
# Bảng token spec — mỗi mục là (TokenType, regex pattern KHÔNG có group
# con để việc xác định token type qua match.lastgroup luôn chính xác).
#
# Thứ tự không quan trọng để tránh nhầm lẫn (mỗi pattern bắt đầu bằng
# tiền tố khác nhau: '<chart', '<think', '<trend', 'SL:', 'RR:', tên
# ACTION_TYPE...), nhưng vẫn giữ nhóm liên quan gần nhau cho dễ đọc.
# =====================================================================
_TOKEN_SPEC = [
    (TokenType.CHART_OPEN, r"<chart>"),
    (TokenType.CHART_CLOSE, r"</chart>"),

    (TokenType.CANDLE_O, r"<O_\d+>"),
    (TokenType.CANDLE_H, r"<H_\d+>"),
    (TokenType.CANDLE_L, r"<L_\d+>"),
    (TokenType.CANDLE_C, r"<C_\d+>"),

    (TokenType.THINK_OPEN, r"<think>"),
    (TokenType.THINK_CLOSE, r"</think>"),

    (TokenType.TREND, r"<trend>(?:UP|DOWN|RANGE)</trend>"),
    (TokenType.CURRENT_PRICE, r"<current_price>\d+</current_price>"),
    (TokenType.ZONE_SUPPORT, r"<zone_support>\d+:\d+</zone_support>"),
    (TokenType.ZONE_RESISTANCE, r"<zone_resistance>\d+:\d+</zone_resistance>"),
    (TokenType.PRICE_IN_ZONE, r"<price_in_zone>"),
    (TokenType.GOOD_PRICE_ACTION, r"<good_price_action>"),

    (TokenType.ACTION_OPEN, r"<action>"),
    (TokenType.ACTION_CLOSE, r"</action>"),
    # CANCEL_*/WAIT_* phải đứng trước BUY/SELL trong danh sách để dễ đọc,
    # dù thực tế alternation không xung đột vì ký tự đầu khác nhau.
    (TokenType.ACTION_TYPE, r"\b(?:CANCEL_BUY|CANCEL_SELL|WAIT_BUY|WAIT_SELL|BUY|SELL|HOLD)\b"),
    (TokenType.SL, r"SL:\d+"),
    (TokenType.RR, r"RR:\d+"),
]

_WS_RE = re.compile(r"\s+")

# Ghép thành 1 regex hợp nhất — mỗi alternative có ĐÚNG 1 named group bọc
# ngoài (không có group con nào khác) nên match.lastgroup luôn trỏ đúng
# TokenType đã khớp, không bị lệch bởi group lồng nhau.
_MASTER_RE = re.compile(
    "|".join(f"(?P<{token_type.name}>{pattern})" for token_type, pattern in _TOKEN_SPEC)
)


class Lexer:
    """
    Regex-based lexer cho ngôn ngữ <chart>...<think>...<action>...

    Thiết kế để KHÔNG BAO GIỜ raise exception khi gặp ký tự lạ — mọi đoạn
    text không khớp pattern nào được gói thành TokenType.UNKNOWN (giữ
    nguyên vị trí + nội dung) thay vì crash. Điều này quan trọng vì lexer
    này sẽ chạy trực tiếp trong reward_func của GRPO trên completion do
    model tự sinh — một completion sai be bét vẫn phải tokenize được để
    Parser có thể chấm điểm well-form liên tục thay vì toàn bộ pipeline
    reward sụp đổ.
    """

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self._length = len(text)

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []

        while self.pos < self._length:
            ws_match = _WS_RE.match(self.text, self.pos)
            if ws_match:
                self.pos = ws_match.end()
                continue
            if self.pos >= self._length:
                break

            match = _MASTER_RE.match(self.text, self.pos)
            if match:
                token_type = TokenType[match.lastgroup]
                tokens.append(
                    Token(type=token_type, value=match.group(0), position=self.pos)
                )
                self.pos = match.end()
            else:
                start = self.pos
                end = self._consume_unknown(start)
                tokens.append(
                    Token(type=TokenType.UNKNOWN, value=self.text[start:end], position=start)
                )
                self.pos = end

        tokens.append(Token(type=TokenType.EOF, value=None, position=self.pos))
        return tokens

    def _consume_unknown(self, start: int) -> int:
        """Gom các ký tự lạ liên tiếp thành 1 token UNKNOWN (thay vì tách
        từng ký tự) để danh sách lỗi đỡ vụn — dừng lại ở whitespace hoặc
        ký tự '<' tiếp theo (điểm có khả năng bắt đầu 1 token hợp lệ)."""
        i = start + 1
        while i < self._length and self.text[i] not in " \t\n<":
            i += 1
        return i
