"""
Demo cho Generator — chạy: python -m app.gen.generator_demo

Kiểm chứng 2 việc quan trọng nhất:
1. MỌI mẫu sinh ra đều well-formed + semantic pass (đúng "by construction",
   không cần rule riêng ở SFT — generator tự đảm bảo).
2. Phân phối leaf-path tương đối đều (không bị lệch do 1 số case dễ dựng
   hơn case khác trên 1 chart cụ thể).
"""
import random
from collections import Counter

from app.data_prepare.generator import LEAF_RECIPES, generate_dataset
from app.lang.parser import Parser
from app.lang.semantic import SemanticChecker
from app.training.reward.forward_test import evaluate_outcome


def make_synthetic_chart(rng: random.Random, n: int = 50, base: int = 500, spread: int = 30):
    """Sinh 1 chart giả lập ngẫu nhiên (random walk nhẹ) để test generator
    trên nhiều hình dạng chart khác nhau, không chỉ 1 chart cố định."""
    candles = []
    price = base
    for _ in range(n):
        price += rng.randint(-3, 3)
        price = max(spread, min(1023 - spread, price))
        o = price
        c = price + rng.randint(-5, 5)
        h = max(o, c) + rng.randint(0, 5)
        l = min(o, c) - rng.randint(0, 5)
        candles.append((max(0, o), min(1023, h), max(0, l), max(0, min(1023, c))))
        price = c
    return candles


def run() -> None:
    rng = random.Random(42)
    charts = [make_synthetic_chart(rng) for _ in range(30)]

    samples = generate_dataset(charts, samples_per_chart=10, seed=123)
    print(f"Sinh được {len(samples)} mẫu từ {len(charts)} chart x 10 samples/chart "
          f"(tối đa lý thuyết {len(charts) * 10}).")

    # --- Kiểm chứng 100% well-form + semantic pass ---
    fail_count = 0
    checker = SemanticChecker()
    for i, sample in enumerate(samples):
        full_text = sample.prompt + " " + sample.completion
        parse_result = Parser.from_text(full_text).parse()
        if not parse_result.is_well_formed():
            print(f"  [FAIL well-form] sample #{i} leaf={sample.leaf_recipe}")
            for e in parse_result.errors:
                print(f"      [{e.severity}] {e.message}")
            fail_count += 1
            continue
        sem_result = checker.check(parse_result.ast)
        if not sem_result.passed:
            print(f"  [FAIL semantic] sample #{i} leaf={sample.leaf_recipe}: {sem_result.violations}")
            fail_count += 1
            continue
        action_type = parse_result.ast.action.action_type
        if action_type in ("BUY", "SELL", "CANCEL_BUY", "CANCEL_SELL"):
            candles_for_chart = charts[i // 10] if i // 10 < len(charts) else None
            # dùng lại đúng candles gốc tương ứng (an toàn hơn nếu thứ tự đổi, bỏ qua ở demo này)

    print(f"\n{'=' * 50}")
    print(f"well-form + semantic: {len(samples) - fail_count}/{len(samples)} PASS")
    assert fail_count == 0, "Generator sinh ra mẫu KHÔNG hợp lệ — cần xem lại công thức dựng số!"

    # --- Phân phối leaf-path ---
    print(f"\n{'=' * 50}")
    print("Phân phối leaf-path (mong đợi tương đối đều trên "
          f"{len(LEAF_RECIPES)} leaf khả dĩ, phụ thuộc hình học từng chart):")
    counts = Counter(s.leaf_recipe for s in samples)
    for leaf, count in sorted(counts.items()):
        print(f"  {leaf:<35} count={count}")

    missing = set(f"{t}|{s}|{c}|{a}" for t, s, c, a in LEAF_RECIPES) - set(counts.keys())
    if missing:
        print(f"\n  Leaf chưa xuất hiện lần nào trong mẫu (có thể do chart random không đủ đa dạng "
              f"hình học, hoặc max_attempts chưa đủ): {missing}")


if __name__ == "__main__":
    run()