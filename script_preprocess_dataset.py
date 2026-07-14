from transformers import AutoTokenizer
import random

from app.gen.generator import generate_dataset
from app.lang.parser import Parser
from app.tokenizer.hub import load_tokenizer


def make_synthetic_chart(rng: random.Random, n: int = 50, base: int = 500, spread: int = 30):
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

# =====================================================================
# CHƯƠNG TRÌNH CHẠY THỬ (DEMO)
# =====================================================================
if __name__ == "__main__":
    # Biểu thức kiểm tra (hỗ trợ cả độ ưu tiên toán tử và dấu ngoặc)
    prompt = "<chart></chart>"
    completion = ""

    from app.tokenizer.hub import load_tokenizer

    tok = load_tokenizer()
    print(f"tokenizer.vocab_size = {tok.vocab_size}\n")

    # ------------------------------------------------------------
    # 1) Round-trip trên mẫu hợp lệ sinh từ generator thật
    # ------------------------------------------------------------
    print("=== 1) Round-trip trên sample hợp lệ (generator by-construction) ===")
    rng = random.Random(7)
    charts = [make_synthetic_chart(rng) for _ in range(5)]
    samples = generate_dataset(charts, samples_per_chart=3, seed=42)
    assert samples, "generator không sinh được mẫu nào — kiểm tra lại app/gen/generator.py"
    
    seq_lens = []
    roundtrip_fail = 0
    for i, sample in enumerate(samples):
        full_text = sample.prompt + " " + sample.completion

        # sanity: mẫu generator sinh ra phải well-formed (đúng "by construction")
        parse_result = Parser.from_text(full_text).parse()
        assert parse_result.is_well_formed(), f"sample #{i} không well-formed — lỗi generator, không phải tokenizer"

        ids = tok.encode(full_text)  # tự thêm <bos>/<eos>
        seq_lens.append(len(ids))

        decoded = tok.decode(ids, skip_special_tokens=True)
        
        print(f"sample #{i}: {full_text}")
        print(f"    decoded = {decoded}")
        print(f"    ids = {ids}")
        print(f"    len(ids) = {len(ids)}")
        print(f"    len(decoded) = {len(decoded)}")
        print(f"    decoded == full_text = {decoded == full_text}")