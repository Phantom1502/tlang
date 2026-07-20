"""
Demo nhanh cho Forward-test engine — chạy:
    python -m demos.forward_test_demo

Test trực tiếp trên các hàm thuần túy (không cần qua Lexer/Parser) vì
input ở đây chỉ là bin số nguyên + future_candles giả lập.

LƯU Ý: counterfactual_outcome (CANCEL_BUY/CANCEL_SELL) đã bị XOÁ HẲN —
quyết định đã chốt: thống kê phản-thực không mang giá trị gì, CANCEL giờ
không có outcome (giống WAIT_BUY/WAIT_SELL/HOLD). Không còn case test nào
cho việc này.
"""
from app.lang.ast_nodes import ZoneNode
from app.training.reward.forward_test import (
    OutcomeStatus,
    derive_target,
    forward_test,
    partial_tp_forward_test,
    is_sl_valid,
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

    # ------------------------------------------------------------
    # partial_tp_forward_test — outcome THẬT cho BUY/SELL (scale-out rr phần)
    # ------------------------------------------------------------
    print("\n=== partial_tp_forward_test: RR5, chốt 1R/2R/3R rồi quay đầu chạm SL ===")
    # entry=500 sl=490 (risk=10). Level targets: 1R=510 2R=520 3R=530 4R=540 5R=550
    candles = [
        (500, 512, 498, 510),   # chạm 1R
        (510, 522, 505, 520),   # chạm 2R
        (520, 532, 515, 530),   # chạm 3R
        (530, 535, 489, 490),   # quay đầu, chạm SL — KHÔNG cho chốt thêm dù chưa tới 4R
    ] + flat_candles(46, 500, 505, 495, 500)
    r = partial_tp_forward_test(entry_bin=500, sl_bin=490, rr=5, future_candles=candles, direction="long")
    expected = (1 + 2 + 3) / 5.0 - (2 / 5.0) * 1.0   # 3 phần đã chốt, 2 phần còn lại lỗ full -1
    print(f"  status={r.status.value} r_multiple={r.r_multiple:.4f} (kỳ vọng {expected:.4f})")
    assert r.status == OutcomeStatus.LOSS and abs(r.r_multiple - expected) < 1e-9

    print("\n=== partial_tp_forward_test: WIN full 5 mức -> r=(1+2+3+4+5)/5=3.0, KHÔNG PHẢI 5.0 ===")
    candles = [(500, 560, 495, 555)] + flat_candles(49, 500, 505, 495, 500)
    r = partial_tp_forward_test(entry_bin=500, sl_bin=490, rr=5, future_candles=candles, direction="long")
    print(f"  status={r.status.value} r_multiple={r.r_multiple:.4f}")
    assert r.status == OutcomeStatus.WIN and abs(r.r_multiple - 3.0) < 1e-9

    print("\n=== partial_tp_forward_test: LOSS ngay nến đầu, chưa chốt gì -> -1.0 ===")
    candles = [(500, 505, 488, 490)] + flat_candles(49, 500, 505, 495, 500)
    r = partial_tp_forward_test(entry_bin=500, sl_bin=490, rr=5, future_candles=candles, direction="long")
    print(f"  status={r.status.value} r_multiple={r.r_multiple:.4f}")
    assert r.status == OutcomeStatus.LOSS and abs(r.r_multiple - (-1.0)) < 1e-9

    print("\n=== partial_tp_forward_test: SL và mức TP mới CÙNG 1 nến -> ưu tiên SL, không chốt thêm ===")
    candles = [(500, 525, 489, 505)] + flat_candles(49, 500, 505, 495, 500)  # h chạm 2R NHƯNG l cũng chạm SL
    r = partial_tp_forward_test(entry_bin=500, sl_bin=490, rr=5, future_candles=candles, direction="long")
    print(f"  status={r.status.value} r_multiple={r.r_multiple:.4f} exit_index={r.exit_index}")
    assert r.status == OutcomeStatus.LOSS and abs(r.r_multiple - (-1.0)) < 1e-9 and r.exit_index == 0

    print("\n=== partial_tp_forward_test: TIMEOUT có partial fill -> mark-to-market phần còn mở ===")
    candles = [(500, 512, 498, 510), (510, 522, 505, 520)] + flat_candles(48, 515, 518, 512, 515)
    r = partial_tp_forward_test(entry_bin=500, sl_bin=490, rr=5, future_candles=candles, direction="long")
    expected = (1 + 2) / 5.0 + (3 / 5.0) * ((515 - 500) / 10.0)
    print(f"  status={r.status.value} r_multiple={r.r_multiple:.4f} (kỳ vọng {expected:.4f})")
    assert r.status == OutcomeStatus.TIMEOUT and abs(r.r_multiple - expected) < 1e-9

    print("\n=== partial_tp_forward_test: rr=1 tương đương forward_test nhị phân cũ ===")
    candles = [(500, 512, 498, 505)] + flat_candles(49, 500, 505, 495, 500)
    r = partial_tp_forward_test(entry_bin=500, sl_bin=490, rr=1, future_candles=candles, direction="long")
    print(f"  status={r.status.value} r_multiple={r.r_multiple:.4f}")
    assert r.status == OutcomeStatus.WIN and abs(r.r_multiple - 1.0) < 1e-9

    print("\n=== partial_tp_forward_test: SELL (short) — mirror đúng chiều ===")
    # entry=500 sl=510 (risk=10, short). 1R=490 2R=480 3R=470
    candles = [(500, 505, 478, 480)] + flat_candles(49, 500, 505, 495, 500)  # chạm cả 1R,2R cùng nến
    r = partial_tp_forward_test(entry_bin=500, sl_bin=510, rr=3, future_candles=candles, direction="short")
    expected = (1 + 2) / 3.0 + (1 / 3.0) * ((500 - 500) / 10.0)  # còn 1/3 mở, mtm tại close=500 (nến sau flat)
    print(f"  status={r.status.value} r_multiple={r.r_multiple:.4f} (kỳ vọng {expected:.4f})")
    assert r.status == OutcomeStatus.TIMEOUT and abs(r.r_multiple - expected) < 1e-9

    print("\nTất cả assertion PASS.")


if __name__ == "__main__":
    run()