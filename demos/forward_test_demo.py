"""
Demo nhanh cho Forward-test / Counterfactual engine — chạy:
    python -m app.reward.forward_test_demo

Test trực tiếp trên các hàm thuần túy (không cần qua Lexer/Parser) vì
input ở đây chỉ là bin số nguyên + future_candles giả lập.
"""
from app.lang.ast_nodes import ZoneNode
from app.training.reward.forward_test import (
    OutcomeStatus,
    derive_target,
    forward_test,
    is_sl_valid,
    counterfactual_outcome,
)


def flat_candles(n: int, o: int, h: int, l: int, c: int):
    return [(o, h, l, c)] * n


def run() -> None:
    print("=== forward_test: WIN (chạm target trước SL) ===")
    candles = [
        (505, 510, 498, 505),         # chưa chạm gì
        (505, 525, 500, 522),         # chạm target=520 (h=525>=520), không chạm sl=490
    ] + flat_candles(48, 505, 510, 498, 505)
    r = forward_test(entry_bin=500, sl_bin=490, target_bin=520, future_candles=candles, direction="long")
    print(f"  status={r.status.value} r_multiple={r.r_multiple} exit_index={r.exit_index}")
    assert r.status == OutcomeStatus.WIN and abs(r.r_multiple - 2.0) < 1e-9

    print("\n=== forward_test: LOSS (chạm SL trước) ===")
    candles = [(500, 505, 488, 490)] + flat_candles(49, 500, 505, 498, 500)
    r = forward_test(entry_bin=500, sl_bin=490, target_bin=520, future_candles=candles, direction="long")
    print(f"  status={r.status.value} r_multiple={r.r_multiple} exit_index={r.exit_index}")
    assert r.status == OutcomeStatus.LOSS and r.r_multiple == -1.0

    print("\n=== forward_test: TIMEOUT (không chạm gì suốt horizon) ===")
    candles = flat_candles(50, 500, 505, 495, 500)
    r = forward_test(entry_bin=500, sl_bin=490, target_bin=520, future_candles=candles, direction="long")
    print(f"  status={r.status.value} r_multiple={r.r_multiple}")
    assert r.status == OutcomeStatus.TIMEOUT and r.r_multiple == 0.0

    print("\n=== forward_test: GAP cùng nến (SL và TP đều chạm) -> ưu tiên SL ===")
    candles = [(500, 525, 485, 505)] + flat_candles(49, 500, 505, 498, 500)
    r = forward_test(entry_bin=500, sl_bin=490, target_bin=520, future_candles=candles, direction="long")
    print(f"  status={r.status.value} r_multiple={r.r_multiple} exit_index={r.exit_index}")
    assert r.status == OutcomeStatus.LOSS and r.exit_index == 0

    print("\n=== is_sl_valid: SL quá sát (dist=3 < min=5) ===")
    zone = ZoneNode(direction="support", lower_bin=495, upper_bin=505)
    valid = is_sl_valid("BUY", entry_bin=500, sl_bin=497, zone=zone)
    print(f"  valid={valid}")
    assert valid is False

    print("\n=== is_sl_valid: SL quá xa (dist=15 > max=10) ===")
    valid = is_sl_valid("BUY", entry_bin=500, sl_bin=485, zone=zone)
    print(f"  valid={valid}")
    assert valid is False

    print("\n=== is_sl_valid: đúng khoảng cách nhưng SAI phía zone (SL nằm trong zone thay vì dưới đáy) ===")
    zone2 = ZoneNode(direction="support", lower_bin=480, upper_bin=490)
    valid = is_sl_valid("BUY", entry_bin=500, sl_bin=492, zone=zone2)  # dist=8 hợp lệ, nhưng 492 >= 480
    print(f"  valid={valid}")
    assert valid is False

    print("\n=== is_sl_valid: hợp lệ hoàn toàn (đúng khoảng cách + đúng phía) ===")
    valid = is_sl_valid("BUY", entry_bin=500, sl_bin=490, zone=zone)  # dist=10, 490<495 zone.lower
    print(f"  valid={valid}")
    assert valid is True

    print("\n=== derive_target: bin bị bão hoà (target > 1023) -> None ===")
    target = derive_target(entry_bin=1020, sl_bin=1010, rr=9, direction="long")
    print(f"  target={target}")
    assert target is None

    print("\n=== counterfactual_outcome: CANCEL_BUY ĐÚNG (tránh được lệnh thua) ===")
    zone3 = ZoneNode(direction="support", lower_bin=495, upper_bin=505)
    # sl derive = zone.lower-1 = 494, target = entry+1*(entry-sl) = 500+6 = 506
    candles = [(500, 498, 490, 492)] + flat_candles(49, 500, 505, 498, 500)  # chạm sl=494 trước (l=490<=494)
    r = counterfactual_outcome("CANCEL_BUY", zone3, current_price_bin=500, future_candles=candles)
    print(f"  status={r.status.value} r_multiple={r.r_multiple}  (kỳ vọng dương — CANCEL đúng)")
    assert r.r_multiple > 0

    print("\n=== counterfactual_outcome: CANCEL_BUY SAI (lẽ ra sẽ thắng) ===")
    candles = [(500, 510, 497, 505)] + flat_candles(49, 500, 505, 498, 500)  # chạm target=506 trước
    r = counterfactual_outcome("CANCEL_BUY", zone3, current_price_bin=500, future_candles=candles)
    print(f"  status={r.status.value} r_multiple={r.r_multiple}  (kỳ vọng âm — CANCEL sai, né mất lệnh thắng)")
    assert r.r_multiple < 0

    print("\nTất cả assertion PASS.")


if __name__ == "__main__":
    run()