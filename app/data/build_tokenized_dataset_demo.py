"""
Demo cho build_tokenized_dataset — chạy: python -m app.data.build_tokenized_dataset_demo

Kiểm chứng LOGIC của script (không push lên Hub thật, không cần mạng):
1. kind=pretrain_sft: input_ids/labels đúng — phần mask (-100) dừng đúng
   ngay trước token đầu tiên của completion, phần còn lại decode lại phải
   khớp đúng completion gốc.
2. kind=grpo: prompt_input_ids tokenize đúng, decode lại khớp prompt gốc.
"""
import random

from datasets import Dataset

from app.data.data_module import _tokenize_and_mask_example
from app.data.build_tokenized_dataset import _tokenize_grpo_prompt
from app.gen.dataset_builder import build_grpo_rows, render_chart_block
from app.gen.generator import generate_dataset
from app.tokenizer.hub import load_tokenizer


def make_synthetic_chart(rng, n=50, base=500, spread=30):
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
    tok = load_tokenizer()   # không có mạng tới Hub thật -> tự fallback build local, có cảnh báo
    print(f"tokenizer.vocab_size = {tok.vocab_size}\n")

    rng = random.Random(11)
    charts = [make_synthetic_chart(rng) for _ in range(5)]

    # ------------------------------------------------------------
    # 1) kind=pretrain_sft
    # ------------------------------------------------------------
    print("=== 1) pretrain_sft: _tokenize_and_mask_example ===")
    samples = generate_dataset(charts, samples_per_chart=3, seed=99)
    assert samples, "generator không sinh được mẫu nào"

    raw_ds = Dataset.from_list([{"prompt": s.prompt, "completion": s.completion} for s in samples])

    mismatch = 0
    for i, example in enumerate(raw_ds):
        out = _tokenize_and_mask_example(example, tok, max_length=512)
        input_ids, labels = out["input_ids"], out["labels"]
        assert len(input_ids) == len(labels)

        # Phần labels != -100 phải decode lại đúng khớp completion + <eos>
        unmasked_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
        decoded_completion = tok.decode(unmasked_ids, skip_special_tokens=True).strip()
        if decoded_completion != example["completion"].strip():
            mismatch += 1
            if mismatch <= 2:
                print(f"  [MISMATCH] sample #{i}")
                print(f"    gốc   : {example['completion'][:100]}...")
                print(f"    decode: {decoded_completion[:100]}...")

        # Phần bị mask (-100) phải đúng bằng đúng đoạn <bos>+prompt
        n_mask = sum(1 for lab in labels if lab == -100)
        prompt_ids = tok(example["prompt"], add_special_tokens=False)["input_ids"]
        assert n_mask == 1 + len(prompt_ids), (
            f"sample #{i}: n_mask={n_mask} nhưng kỳ vọng {1 + len(prompt_ids)} (1 bos + {len(prompt_ids)} prompt)"
        )

    print(f"  Tổng {len(raw_ds)} mẫu, mismatch={mismatch}")
    assert mismatch == 0, "Decode phần completion KHÔNG khớp gốc — lỗi mask boundary!"
    print("  -> PASS: mask boundary đúng, decode lại khớp 100% completion gốc.\n")

    # ------------------------------------------------------------
    # 2) kind=grpo
    # ------------------------------------------------------------
    print("=== 2) grpo: _tokenize_grpo_prompt ===")
    import pandas as pd
    encoded_rows = []
    for chart in charts:
        # giả lập 1 window 100 nến (chart input 50 + 50 nến future y hệt, đủ dùng cho test này)
        full_100 = list(chart) + list(chart)
        text = "<chart> " + " ".join(f"O_{o} H_{h} L_{l} C_{c}" for o, h, l, c in full_100) + " </chart>"
        encoded_rows.append({"text": text})
    encoded_df = pd.DataFrame(encoded_rows)

    grpo_rows = build_grpo_rows(encoded_df, symbol="TEST", n_augments=0, seed=1)
    grpo_ds = Dataset.from_list(grpo_rows)

    for i, example in enumerate(grpo_ds):
        out = _tokenize_grpo_prompt(example, tok, max_length=512)
        prompt_ids = out["prompt_input_ids"]
        decoded = tok.decode(prompt_ids, skip_special_tokens=True).strip()
        assert decoded == example["prompt"].strip(), f"row #{i}: decode prompt không khớp gốc"

    print(f"  Tổng {len(grpo_ds)} row, tất cả prompt_input_ids decode lại khớp 100% prompt gốc.")
    print("  -> PASS.\n")

    print("Tất cả assertion PASS.")


if __name__ == "__main__":
    run()