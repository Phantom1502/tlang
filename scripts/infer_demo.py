"""
infer_demo.py — Inference nhanh cho Trading Reasoning LLM, chạy trên Colab.

Giả định:
  - Đang đứng trong repo (vd /content/tlang) — cần import được app.*
  - Đã có 1 checkpoint model trên HF Hub (pretrain/SFT/GRPO đều load được
    theo cùng cách, vì cùng kiến trúc LlamaForCausalLM + cùng tokenizer).

Cách chạy trong Colab (1 cell):
    %cd /content/tlang
    !python -m scripts.infer_demo --model_repo sullivan1502/base-grpo-test --n_samples 3

Hoặc paste thẳng nội dung dưới vào 1 cell, sửa MODEL_REPO ở main().
"""
from __future__ import annotations

import argparse
import random

import torch
from transformers import LlamaForCausalLM

from app.tokenizer.hub import load_tokenizer
from app.lang.parser import Parser
from app.lang.semantic import SemanticChecker


# =====================================================================
# 1) Sinh 1 chart giả lập (random walk) — thay bằng chart thật (qua
# ChartCodec.encode_window) nếu bạn có OHLC thật muốn test.
# =====================================================================
def make_synthetic_chart_text(rng: random.Random, n: int = 50, base: int = 500) -> str:
    parts = ["<chart>"]
    price = base
    for _ in range(n):
        price += rng.randint(-3, 3)
        price = max(30, min(1023 - 30, price))
        o = price
        c = price + rng.randint(-5, 5)
        h = max(o, c) + rng.randint(0, 5)
        l = min(o, c) - rng.randint(0, 5)
        o, h, l, c = (max(0, o), min(1023, h), max(0, l), max(0, min(1023, c)))
        parts.extend([f"<O_{o}>", f"<H_{h}>", f"<L_{l}>", f"<C_{c}>"])
        price = c
    parts.append("</chart>")
    return " ".join(parts)


# =====================================================================
# 2) Generate + parse + chấm nhanh well-form/semantic cho 1 chart
# =====================================================================
def run_one(model, tokenizer, device, chart_text: str, max_new_tokens: int = 200, do_sample: bool = True):
    prompt_ids = tokenizer(chart_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        out_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.8 if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Chỉ lấy phần model tự sinh (bỏ phần prompt đã có sẵn)
    gen_ids = out_ids[0][prompt_ids.shape[1]:]
    completion_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    full_text = chart_text + " " + completion_text
    parse_result = Parser.from_text(full_text).parse()

    print("-" * 70)
    print(f"[completion] {completion_text[:200]}")
    print(f"well_formed = {parse_result.is_well_formed()}  well_form_score = {parse_result.well_form_score():.2f}")
    for err in parse_result.errors[:5]:
        print(f"  [{err.severity}] {err.message}")

    if parse_result.is_well_formed():
        sem_result = SemanticChecker().check(parse_result.ast)
        print(f"semantic_passed = {sem_result.passed}  semantic_score = {sem_result.score:.2f}")
        for v in sem_result.violations[:5]:
            print(f"  - {v}")

        action = parse_result.ast.action
        think = parse_result.ast.think
        print(f"trend={think.trend} action_type={action.action_type} sl={action.sl} rr={action.rr}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_repo", required=True, help="vd sullivan1502/tiny-pretrain")
    p.add_argument("--tokenizer_repo", default=None, help="mặc định DEFAULT_TOKENIZER_REPO trong app/tokenizer/hub.py")
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--greedy", action="store_true", help="tắt sampling, dùng greedy decode")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = load_tokenizer(repo_id=args.tokenizer_repo)
    model = LlamaForCausalLM.from_pretrained(args.model_repo).to(device)
    model.eval()

    print(f"model_repo={args.model_repo}  vocab_size(tokenizer)={tokenizer.vocab_size}  "
          f"vocab_size(model.config)={model.config.vocab_size}")

    rng = random.Random(args.seed)
    for i in range(args.n_samples):
        chart_text = make_synthetic_chart_text(rng)
        print(f"\n=== sample #{i} ===")
        run_one(
            model, tokenizer, device, chart_text,
            max_new_tokens=args.max_new_tokens, do_sample=not args.greedy,
        )


if __name__ == "__main__":
    main()