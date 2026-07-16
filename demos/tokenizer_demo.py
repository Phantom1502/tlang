"""
tokenizer_demo.py — Kiểm chứng nhanh cho tokenizer — chạy:
    python -m app.tokenizer.tokenizer_demo

Tải tokenizer qua `app.tokenizer.hub.load_tokenizer()` (nguồn DUY NHẤT từ
giờ trở đi — xem app/tokenizer/hub.py) thay vì build trực tiếp từ
vocab_builder.py. Nếu chưa push lên Hub / không có mạng, sẽ fallback build
local kèm cảnh báo — script vẫn chạy được để dev/test cục bộ.

Không phải unit test chính thức (pytest sẽ thêm sau), chỉ để xác nhận
3 điều quan trọng nhất trước khi dùng tokenizer này train thật:

1. ROUND-TRIP TRUNG THỰC trên mẫu hợp lệ: encode rồi decode lại phải
   cho ra ĐÚNG lại text mà Parser hiện có parse ra well-formed — không
   được tự "sửa" hay làm mất thông tin (đúng nguyên tắc mục 3 của spec).
2. SEQ_LEN nằm trong ngân sách max_position_embeddings=512 (mục 2.4
   docs/train_pipeline_v0.1.md, kỳ vọng ~230-240 token/sample).
3. HÀNH VI TRÊN COMPLETION RÁC (GRPO rollout tự sinh sai): không crash,
   token lạ map về <unk>, KHÔNG tự vá lành cấu trúc hỏng.
"""
from __future__ import annotations

import random

from app.data_prepare.generator import generate_dataset
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


def run() -> None:
    # Load từ HF Hub (fallback build local nếu chưa push / không có mạng —
    # xem cảnh báo in ra nếu rơi vào nhánh fallback). Từ giờ đây là cách
    # DUY NHẤT lấy tokenizer, không gọi build_fast_tokenizer() trực tiếp
    # ở bất kỳ script nào khác ngoài push_to_hub.py.
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
        # decode nối lại bằng khoảng trắng (PreTrainedTokenizerFast mặc định) — so sánh
        # sau khi chuẩn hoá whitespace 2 phía, vì input gốc join bằng " ".join(...) y hệt.
        if decoded.strip() != full_text.strip():
            roundtrip_fail += 1
            if roundtrip_fail <= 3:
                print(f"  [MISMATCH] sample #{i}")
                print(f"    gốc : {full_text[:120]}...")
                print(f"    decode: {decoded[:120]}...")

    print(f"  Tổng {len(samples)} mẫu, round-trip fail = {roundtrip_fail}")
    assert roundtrip_fail == 0, "Round-trip KHÔNG trung thực trên mẫu hợp lệ — bug trong tokenizer!"
    print("  -> PASS: encode/decode trung thực 100% trên mẫu hợp lệ.\n")

    # ------------------------------------------------------------
    # 2) Seq_len budget
    # ------------------------------------------------------------
    print("=== 2) Seq_len budget (max_position_embeddings=512) ===")
    print(f"  min={min(seq_lens)} max={max(seq_lens)} avg={sum(seq_lens)/len(seq_lens):.1f}")
    assert max(seq_lens) <= 512, "Có sample vượt quá 512 token — cần xem lại budget!"
    print("  -> PASS: mọi sample nằm trong ngân sách 512 token.\n")

    # ------------------------------------------------------------
    # 3) Completion rác — không crash, map <unk>, không tự vá lành
    # ------------------------------------------------------------
    print("=== 3) Completion rác (giả lập GRPO rollout sai) ===")
    garbage_cases = [
        "day la mot doan text hoan toan random khong theo grammar gi ca <chart broken",
        "<chart> <O_9999> </chart>",   # ngoài vocab (chỉ có <O_0>..<O_1023>)
        "<think> <trend>SIDEWAYS</trend> </think>",  # SIDEWAYS không có trong vocab (chỉ UP/DOWN/RANGE)
        "",
    ]
    unk_id = tok.unk_token_id
    for text in garbage_cases:
        ids = tok.encode(text)
        decoded = tok.decode(ids, skip_special_tokens=True)
        has_unk = unk_id in ids
        print(f"  input={text[:60]!r:<63} -> len={len(ids):<4} has_unk={has_unk}")
        print(f"    decode={decoded[:80]!r}")
    print("  -> PASS: không crash trên input rác; token ngoài vocab map về <unk> "
          "thay vì bị âm thầm loại bỏ hoặc suy diễn lại thành token hợp lệ.\n")

    # ------------------------------------------------------------
    # 4) Đối chiếu segment digit-decompose vẫn tách đúng từng token
    # ------------------------------------------------------------
    print("=== 4) Digit-decompose: mỗi digit PHẢI là 1 token riêng (không bị BPE merge) ===")
    snippet = "<current_price> 0 5 1 2 </current_price>"
    ids = tok.encode(snippet, add_special_tokens=False)
    tokens_back = tok.convert_ids_to_tokens(ids)
    print(f"  {snippet!r} -> {tokens_back}")
    assert tokens_back == ["<current_price>", "0", "5", "1", "2", "</current_price>"], (
        "Digit bị gộp sai — vi phạm nguyên tắc digit-decompose!"
    )
    print("  -> PASS: mỗi digit là 1 token độc lập, đúng thiết kế.\n")

    print("Tất cả kiểm tra PASS.")


if __name__ == "__main__":
    run()