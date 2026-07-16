"""
Demo nhanh cho unified_reward_func — chạy: python -m app.reward.reward_func_demo

Test cả 3 gate (well-form fail / semantic fail / outcome), cơ chế
weight_table theo trend+action, và StatsCollector.print_summary().
"""
from app.training.reward.reward_func import (
    R_SEM_FULL,
    R_WF_FULL,
    StatsCollector,
    WeightTable,
    score_completion,
)


def fmt_bin(n: int, pad: int = 4) -> str:
    return " ".join(str(n).zfill(pad))


def make_chart(closes) -> str:
    candles = []
    for c in closes:
        candles.append(f"<O_{c}> <H_{c + 15}> <L_{c - 15}> <C_{c}>")
    while len(candles) < 50:
        candles.insert(0, "<O_500> <H_503> <L_497> <C_500>")
    return "<chart> " + " ".join(candles) + " </chart>"


def flat_future(n: int, o: int, h: int, l: int, c: int):
    return [[o, h, l, c]] * n


def run() -> None:
    stats = StatsCollector()
    weights = WeightTable()

    # ------------------------------------------------------------
    # Case 1: well-form fail (garbage hoàn toàn) -> reward thấp, không log semantic
    # ------------------------------------------------------------
    r1 = score_completion("hoan toan khong theo grammar", future_bins=[[0, 0, 0, 0]] * 50, stats=stats, weights=weights)
    print(f"[garbage]                reward={r1:.3f}  (kỳ vọng < {R_WF_FULL})")
    assert r1 < R_WF_FULL

    # ------------------------------------------------------------
    # Case 2: well-form pass, semantic fail (trend UP nhưng zone_resistance)
    # ------------------------------------------------------------
    completion = (
        make_chart([500] * 49 + [505])
        + f" <think> <trend>UP</trend> <current_price> {fmt_bin(505)} </current_price> "
          f"<zone_resistance> {fmt_bin(500)} : {fmt_bin(510)} </zone_resistance> <price_in_zone> <good_price_action> </think>"
        + f" <action> BUY SL: {fmt_bin(495)} <RR_5> </action>"
    )
    r2 = score_completion(completion, future_bins=[[500, 505, 495, 500]] * 50, stats=stats, weights=weights)
    print(f"[semantic fail]          reward={r2:.3f}  (kỳ vọng trong khoảng [{R_WF_FULL}, {R_WF_FULL + R_SEM_FULL}))")
    assert R_WF_FULL <= r2 < R_WF_FULL + R_SEM_FULL

    # ------------------------------------------------------------
    # Case 3: pass cả 2 gate, SL vi phạm khoảng cách (dist=2, quá sát) -> extra_semantic fail
    # ------------------------------------------------------------
    completion = (
        make_chart([500] * 49 + [505])
        + f" <think> <trend>UP</trend> <current_price> {fmt_bin(505)} </current_price> "
          f"<zone_support> {fmt_bin(500)} : {fmt_bin(510)} </zone_support> <price_in_zone> <good_price_action> </think>"
        + f" <action> BUY SL: {fmt_bin(503)} <RR_5> </action>"   # dist=2 < SL_MIN_DIST_BINS=5
    )
    r3 = score_completion(completion, future_bins=[[500, 505, 495, 500]] * 50, stats=stats, weights=weights)
    print(f"[SL-distance fail]       reward={r3:.3f}  (kỳ vọng trong khoảng [{R_WF_FULL}, {R_WF_FULL + R_SEM_FULL}))")
    assert R_WF_FULL <= r3 < R_WF_FULL + R_SEM_FULL

    # ------------------------------------------------------------
    # Case 4: HOLD hợp lệ (RANGE, không zone) -> chỉ well-form + semantic, không outcome
    # ------------------------------------------------------------
    completion = (
        make_chart([500] * 50)
        + f" <think> <trend>RANGE</trend> <current_price> {fmt_bin(500)} </current_price> </think>"
        + " <action> HOLD </action>"
    )
    r4 = score_completion(completion, future_bins=[[500, 505, 495, 500]] * 50, stats=stats, weights=weights)
    print(f"[HOLD hợp lệ]            reward={r4:.3f}  (kỳ vọng đúng = {R_WF_FULL + R_SEM_FULL})")
    assert abs(r4 - (R_WF_FULL + R_SEM_FULL)) < 1e-9

    # ------------------------------------------------------------
    # Case 5: BUY hợp lệ, THẮNG. entry=505, sl=495 (dist=10, sl<zone.lower=500 ok),
    # rr=5 -> target=505+5*10=555. Chạm target chính xác -> r_multiple=5.0 (=rr).
    # ------------------------------------------------------------
    completion = (
        make_chart([500] * 49 + [505])
        + f" <think> <trend>UP</trend> <current_price> {fmt_bin(505)} </current_price> "
          f"<zone_support> {fmt_bin(500)} : {fmt_bin(510)} </zone_support> <price_in_zone> <good_price_action> </think>"
        + f" <action> BUY SL: {fmt_bin(495)} <RR_5> </action>"
    )
    future = flat_future(48, 505, 510, 500, 505) + [[505, 560, 500, 555]] + flat_future(1, 505, 510, 500, 505)
    r5 = score_completion(completion, future_bins=future, stats=stats, weights=weights)
    print(f"[BUY thắng r=5.0]        reward={r5:.3f}  (kỳ vọng = {R_WF_FULL + R_SEM_FULL + 5.0})")
    assert abs(r5 - (R_WF_FULL + R_SEM_FULL + 5.0)) < 1e-6

    # ------------------------------------------------------------
    # Case 6: cùng setup nhưng weight_table["UP"]["BUY"] = 0.5 -> outcome bị giảm 1 nửa
    # ------------------------------------------------------------
    weights.set("UP", "BUY", 0.5)
    r6 = score_completion(completion, future_bins=future, stats=stats, weights=weights)
    print(f"[BUY thắng, w=0.5]       reward={r6:.3f}  (kỳ vọng = {R_WF_FULL + R_SEM_FULL + 5.0 * 0.5})")
    assert abs(r6 - (R_WF_FULL + R_SEM_FULL + 2.5)) < 1e-6

    print("\nTất cả assertion PASS.\n")
    stats.print_summary()


if __name__ == "__main__":
    run()