"""
Demo nhanh cho Semantic Checker — chạy: python -m app.lang.semantic_demo

Test trên các completion ĐÃ well-formed (parse thành công), tập trung
kiểm tra bảng 2.2 A/B/D/E.
"""
from app.lang.parser import Parser
from app.lang.semantic import SemanticChecker


def make_chart(closes) -> str:
    """closes: list giá Close cho từng nến, nến cuối cùng = closes[-1]."""
    candles = []
    for c in closes:
        candles.append(f"<O_{c}> <H_{c+3}> <L_{c-3}> <C_{c}>")
    # pad cho đủ 50 nến bằng cách lặp nến đầu (không ảnh hưởng test semantic,
    # chỉ ảnh hưởng well-form candle-count nếu thiếu — ở đây pad đủ 50).
    while len(candles) < 50:
        candles.insert(0, f"<O_500> <H_503> <L_497> <C_500>")
    return "<chart> " + " ".join(candles) + " </chart>"


CASES = {
    # Hợp lệ hoàn toàn: UP, zone_support dưới giá, giá đã trong zone -> BUY
    "valid_up_buy": (
        make_chart([500] * 45 + [505, 506, 507, 508, 505])  # nến cuối Close=505, nằm trong zone 500:510
        + " <think> <trend>UP</trend> <current_price>505</current_price> "
          "<zone_support>500:510</zone_support> <price_in_zone> <good_price_action> </think>"
        + " <action> BUY SL:495 RR:5 </action>"
    ),
    # Vi phạm A: trend UP nhưng lại có zone_resistance
    "trend_zone_mismatch": (
        make_chart([500] * 45 + [505, 506, 507, 508, 505])
        + " <think> <trend>UP</trend> <current_price>505</current_price> "
          "<zone_resistance>500:510</zone_resistance> <price_in_zone> <good_price_action> </think>"
        + " <action> BUY SL:495 RR:5 </action>"
    ),
    # Vi phạm B: zone_support nhưng lại nằm hoàn toàn TRÊN current_price
    "zone_direction_wrong_side": (
        make_chart([500] * 45 + [505, 506, 507, 508, 480])  # current_price=480
        + " <think> <trend>UP</trend> <current_price>480</current_price> "
          "<zone_support>500:510</zone_support> </think>"
        + " <action> WAIT_BUY </action>"
    ),
    # Vi phạm D: giá KHÔNG trong zone, 5 nến cuối cũng không chạm zone, nhưng model vẫn set price_in_zone
    "price_in_zone_geometry_wrong": (
        make_chart([500] * 45 + [300, 301, 302, 303, 300])  # xa zone 500:510, không chạm
        + " <think> <trend>UP</trend> <current_price>300</current_price> "
          "<zone_support>500:510</zone_support> <price_in_zone> <good_price_action> </think>"
        + " <action> BUY SL:290 RR:5 </action>"
    ),
    # Vi phạm E: có zone_support, price_in_zone=true, nhưng action lại là WAIT_BUY (sai nhóm)
    "action_group_wrong": (
        make_chart([500] * 45 + [505, 506, 507, 508, 505])
        + " <think> <trend>UP</trend> <current_price>505</current_price> "
          "<zone_support>500:510</zone_support> <price_in_zone> </think>"
        + " <action> WAIT_BUY </action>"
    ),
    # Hợp lệ: CANCEL_BUY khi price_in_zone=true (không cần good_price_action)
    "valid_cancel_buy": (
        make_chart([500] * 45 + [505, 506, 507, 508, 505])
        + " <think> <trend>UP</trend> <current_price>505</current_price> "
          "<zone_support>500:510</zone_support> <price_in_zone> </think>"
        + " <action> CANCEL_BUY </action>"
    ),
    # Hợp lệ: RANGE không zone -> HOLD
    "valid_range_hold": (
        make_chart([500] * 50)
        + " <think> <trend>RANGE</trend> <current_price>500</current_price> </think>"
        + " <action> HOLD </action>"
    ),
    # Vi phạm E: RANGE không zone nhưng action lại không phải HOLD
    "range_no_zone_wrong_action": (
        make_chart([500] * 50)
        + " <think> <trend>RANGE</trend> <current_price>500</current_price> </think>"
        + " <action> WAIT_BUY </action>"
    ),
}


def run() -> None:
    checker = SemanticChecker()
    for name, text in CASES.items():
        parse_result = Parser.from_text(text).parse()
        print(f"\n=== {name} ===")
        if not parse_result.is_well_formed():
            print("  [SKIP] không well-formed, in lỗi parser:")
            for e in parse_result.errors:
                print(f"    [{e.severity}] {e.message}")
            continue

        sem_result = checker.check(parse_result.ast)
        print(f"  semantic passed = {sem_result.passed}  score = {sem_result.score:.2f}")
        for v in sem_result.violations:
            print(f"    - {v}")


if __name__ == "__main__":
    run()
