"""
Demo nhanh cho Lexer + Parser — chạy: python -m demos.demo

Không phải unit test chính thức (sẽ thêm sau bằng pytest), chỉ để
kiểm tra nhanh các case điển hình trước khi ghép vào reward_func.
"""
from app.lang.parser import Parser


def fmt_bin(n: int, pad: int = 4) -> str:
    """Format 1 số bin thành chuỗi digit CÁCH KHOẢNG TRẮNG (zero-pad),
    đúng quy ước digit-decompose cho current_price/zone/SL."""
    return " ".join(str(n).zfill(pad))


def make_chart(n: int = 50, close_last: int = 512) -> str:
    """Sinh nhanh 1 chart_block n nến, ép Close nến cuối = close_last để
    test rule current_price khớp chart thật. Chart OHLC vẫn atomic <O_x>."""
    candles = []
    for i in range(n):
        c = close_last if i == n - 1 else 500 + i
        candles.append(f"<O_{c}> <H_{c+5}> <L_{c-5}> <C_{c}>")
    return "<chart> " + " ".join(candles) + " </chart>"


CASES = {
    "well_formed_buy": (
        make_chart(close_last=512)
        + f" <think> <trend>UP</trend> <current_price> {fmt_bin(512)} </current_price> "
          f"<zone_support> {fmt_bin(500)} : {fmt_bin(510)} </zone_support> <price_in_zone> <good_price_action> </think>"
        + f" <action> BUY SL: {fmt_bin(495)} <RR_5> </action>"
    ),
    "well_formed_cancel_buy": (
        make_chart(close_last=512)
        + f" <think> <trend>UP</trend> <current_price> {fmt_bin(512)} </current_price> "
          f"<zone_support> {fmt_bin(500)} : {fmt_bin(510)} </zone_support> <price_in_zone> </think>"
        + " <action> CANCEL_BUY </action>"
    ),
    "well_formed_hold": (
        make_chart(close_last=512)
        + f" <think> <trend>RANGE</trend> <current_price> {fmt_bin(512)} </current_price> </think>"
        + " <action> HOLD </action>"
    ),
    "missing_current_price": (
        make_chart(close_last=512)
        + f" <think> <trend>UP</trend> "
          f"<zone_support> {fmt_bin(500)} : {fmt_bin(510)} </zone_support> <price_in_zone> <good_price_action> </think>"
        + f" <action> BUY SL: {fmt_bin(495)} <RR_5> </action>"
    ),
    "wrong_current_price_value": (
        make_chart(close_last=512)
        + f" <think> <trend>UP</trend> <current_price> {fmt_bin(999)} </current_price> "
          f"<zone_support> {fmt_bin(500)} : {fmt_bin(510)} </zone_support> <price_in_zone> <good_price_action> </think>"
        + f" <action> BUY SL: {fmt_bin(495)} <RR_5> </action>"
    ),
    "cancel_with_forbidden_fields": (
        make_chart(close_last=512)
        + f" <think> <trend>UP</trend> <current_price> {fmt_bin(512)} </current_price> "
          f"<zone_support> {fmt_bin(500)} : {fmt_bin(510)} </zone_support> <price_in_zone> <good_price_action> </think>"
        + f" <action> CANCEL_BUY SL: {fmt_bin(495)} <RR_5> </action>"
    ),
    "buy_missing_sl_rr": (
        make_chart(close_last=512)
        + f" <think> <trend>UP</trend> <current_price> {fmt_bin(512)} </current_price> "
          f"<zone_support> {fmt_bin(500)} : {fmt_bin(510)} </zone_support> <price_in_zone> <good_price_action> </think>"
        + " <action> BUY </action>"
    ),
    "rr_out_of_vocab_range": (
        # <RR_15> không khớp pattern <RR_[1-9]> -> lexer trả UNKNOWN -> parser báo "thiếu RR"
        make_chart(close_last=512)
        + f" <think> <trend>UP</trend> <current_price> {fmt_bin(512)} </current_price> "
          f"<zone_support> {fmt_bin(500)} : {fmt_bin(510)} </zone_support> <price_in_zone> <good_price_action> </think>"
        + f" <action> BUY SL: {fmt_bin(495)} <RR_15> </action>"
    ),
    "digit_count_wrong": (
        # current_price chỉ có 2 digit thay vì 4 -> lỗi "value" nhưng vẫn parse tiếp được
        make_chart(close_last=512)
        + " <think> <trend>UP</trend> <current_price> 5 1 2 </current_price> "
        + f"<zone_support> {fmt_bin(500)} : {fmt_bin(510)} </zone_support> <price_in_zone> <good_price_action> </think>"
        + f" <action> BUY SL: {fmt_bin(495)} <RR_5> </action>"
    ),
    "garbage_completion": "day la mot doan text hoan toan random khong theo grammar gi ca <chart broken",
    "wrong_candle_count": (
        make_chart(n=10, close_last=512)
        + f" <think> <trend>UP</trend> <current_price> {fmt_bin(512)} </current_price> "
          f"<zone_support> {fmt_bin(500)} : {fmt_bin(510)} </zone_support> <price_in_zone> <good_price_action> </think>"
        + f" <action> BUY SL: {fmt_bin(495)} <RR_5> </action>"
    ),
}


def run() -> None:
    for name, text in CASES.items():
        result = Parser.from_text(text).parse()
        print(f"\n=== {name} ===")
        print(f"well_formed = {result.is_well_formed()}  score = {result.well_form_score():.2f}")
        for err in result.errors:
            print(f"  [{err.severity}] pos={err.position}: {err.message}")
        if result.ast and result.ast.action:
            a = result.ast.action
            print(f"  action_type={a.action_type} sl={a.sl} rr={a.rr}")


if __name__ == "__main__":
    run()