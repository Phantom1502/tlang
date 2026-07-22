"""
infer_demo.py — Inference nhanh cho Trading Reasoning LLM, chạy trên Colab.

Giả định:
  - Đang đứng trong repo (vd /content/tlang) — cần import được app.*
  - Đã có 1 checkpoint model trên HF Hub (pretrain/SFT/GRPO đều load được
    theo cùng cách, vì cùng kiến trúc LlamaForCausalLM + cùng tokenizer).

Cách chạy trong Colab (1 cell):
    %cd /content/tlang
    python -m scripts.infer_demo --model_repo sullivan1502/base-grpo-round1

Hoặc paste thẳng nội dung dưới vào 1 cell, sửa MODEL_REPO ở main().

=== THÊM MỚI ===
Với MỖI sample well_formed=True VÀ semantic_passed=True (output "đúng"),
dựng lại chart 50 nến input từ chính AST đã parse, vẽ đè zone
(support/resistance) và, nếu action là BUY/SELL, vẽ thêm SL/TP (TP tính
qua derive_target — cùng công thức đang dùng ở
app/training/reward/forward_test.py, không suy diễn lại cách khác).
Output KHÔNG well-formed/semantic fail vẫn chỉ in text như cũ, KHÔNG vẽ
(vẽ 1 chart sai coi như minh hoạ 1 lệnh hợp lệ là sai lệch, không nên).

Ảnh lưu vào --plot_dir (mặc định ./infer_plots), tắt bằng --no_plot.
Cần cài thêm matplotlib (chưa có trong requirements.txt):
    pip install matplotlib
"""
from __future__ import annotations

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")  # headless — Colab/server không có display
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from transformers import LlamaForCausalLM

from app.tokenizer.hub import load_tokenizer
from app.lang.parser import Parser
from app.lang.semantic import SemanticChecker
from app.training.reward.forward_test import derive_target

EXAMPLES = [
    "<chart> <O_512> <H_514> <L_497> <C_499> <O_499> <H_509> <L_497> <C_509> <O_512> <H_512> <L_497> <C_509> <O_504> <H_507> <L_489> <C_497> <O_492> <H_497> <L_487> <C_492> <O_492> <H_497> <L_475> <C_477> <O_475> <H_479> <L_467> <C_467> <O_467> <H_487> <L_467> <C_484> <O_482> <H_494> <L_477> <C_477> <O_477> <H_477> <L_470> <C_475> <O_475> <H_482> <L_462> <C_465> <O_467> <H_487> <L_467> <C_472> <O_477> <H_482> <L_470> <C_475> <O_472> <H_494> <L_472> <C_492> <O_492> <H_502> <L_492> <C_497> <O_502> <H_502> <L_487> <C_487> <O_492> <H_524> <L_492> <C_512> <O_507> <H_509> <L_494> <C_497> <O_499> <H_502> <L_494> <C_494> <O_494> <H_507> <L_467> <C_467> <O_467> <H_470> <L_440> <C_467> <O_470> <H_475> <L_457> <C_460> <O_462> <H_484> <L_457> <C_479> <O_484> <H_499> <L_470> <C_470> <O_470> <H_489> <L_462> <C_482> <O_482> <H_487> <L_475> <C_477> <O_475> <H_499> <L_462> <C_497> <O_499> <H_499> <L_484> <C_494> <O_494> <H_499> <L_484> <C_487> <O_487> <H_489> <L_467> <C_475> <O_475> <H_497> <L_475> <C_492> <O_492> <H_492> <L_475> <C_484> <O_482> <H_497> <L_475> <C_492> <O_494> <H_494> <L_470> <C_482> <O_484> <H_499> <L_472> <C_475> <O_470> <H_475> <L_462> <C_472> <O_470> <H_475> <L_447> <C_447> <O_450> <H_462> <L_447> <C_447> <O_447> <H_460> <L_433> <C_460> <O_457> <H_457> <L_423> <C_425> <O_418> <H_420> <L_401> <C_418> <O_415> <H_423> <L_411> <C_413> <O_408> <H_420> <L_401> <C_408> <O_411> <H_423> <L_403> <C_411> <O_411> <H_425> <L_398> <C_411> <O_411> <H_455> <L_406> <C_447> <O_445> <H_445> <L_406> <C_413> <O_413> <H_438> <L_413> <C_438> <O_438> <H_460> <L_435> <C_457> <O_457> <H_467> <L_447> <C_465> </chart>",
    "<chart> <O_512> <H_530> <L_501> <C_525> <O_525> <H_526> <L_509> <C_523> <O_523> <H_523> <L_501> <C_512> <O_511> <H_552> <L_508> <C_550> <O_548> <H_552> <L_528> <C_547> <O_551> <H_580> <L_529> <C_558> <O_555> <H_556> <L_526> <C_533> <O_534> <H_589> <L_531> <C_589> <O_590> <H_620> <L_590> <C_599> <O_601> <H_619> <L_595> <C_599> <O_601> <H_615> <L_593> <C_609> <O_611> <H_649> <L_602> <C_645> <O_646> <H_656> <L_644> <C_651> <O_651> <H_652> <L_637> <C_648> <O_647> <H_647> <L_626> <C_634> <O_635> <H_638> <L_628> <C_636> <O_634> <H_636> <L_627> <C_633> <O_634> <H_644> <L_633> <C_633> <O_632> <H_635> <L_612> <C_614> <O_614> <H_615> <L_602> <C_603> <O_602> <H_624> <L_602> <C_624> <O_622> <H_623> <L_615> <C_616> <O_616> <H_616> <L_602> <C_604> <O_604> <H_617> <L_603> <C_613> <O_613> <H_614> <L_607> <C_607> <O_608> <H_608> <L_590> <C_598> <O_598> <H_598> <L_596> <C_596> <O_596> <H_600> <L_596> <C_597> <O_598> <H_608> <L_596> <C_607> <O_612> <H_619> <L_612> <C_613> <O_612> <H_614> <L_603> <C_603> <O_603> <H_603> <L_576> <C_578> <O_577> <H_578> <L_569> <C_576> <O_572> <H_596> <L_569> <C_586> <O_585> <H_602> <L_583> <C_601> <O_600> <H_609> <L_596> <C_609> <O_609> <H_614> <L_599> <C_603> <O_603> <H_614> <L_603> <C_613> <O_613> <H_613> <L_608> <C_609> <O_609> <H_609> <L_595> <C_599> <O_597> <H_598> <L_591> <C_594> <O_593> <H_605> <L_584> <C_599> <O_598> <H_603> <L_593> <C_603> <O_603> <H_610> <L_602> <C_610> <O_613> <H_621> <L_609> <C_619> <O_619> <H_628> <L_616> <C_619> <O_620> <H_628> <L_618> <C_621> <O_621> <H_631> <L_619> <C_627> <O_627> <H_656> <L_627> <C_647> <O_643> <H_655> <L_643> <C_652> </chart>",
    "<chart> <O_512> <H_529> <L_509> <C_527> <O_527> <H_527> <L_496> <C_506> <O_506> <H_507> <L_485> <C_485> <O_486> <H_497> <L_470> <C_482> <O_482> <H_498> <L_477> <C_488> <O_488> <H_511> <L_483> <C_511> <O_511> <H_513> <L_490> <C_491> <O_490> <H_490> <L_447> <C_466> <O_465> <H_492> <L_460> <C_485> <O_485> <H_489> <L_469> <C_482> <O_482> <H_489> <L_472> <C_488> <O_488> <H_490> <L_470> <C_471> <O_471> <H_477> <L_460> <C_462> <O_462> <H_498> <L_459> <C_486> <O_487> <H_487> <L_467> <C_467> <O_467> <H_481> <L_464> <C_480> <O_480> <H_490> <L_473> <C_474> <O_474> <H_484> <L_470> <C_478> <O_479> <H_503> <L_476> <C_503> <O_504> <H_508> <L_490> <C_500> <O_500> <H_505> <L_498> <C_499> <O_500> <H_508> <L_489> <C_501> <O_500> <H_502> <L_489> <C_501> <O_501> <H_508> <L_495> <C_501> <O_501> <H_519> <L_498> <C_518> <O_518> <H_522> <L_512> <C_521> <O_522> <H_532> <L_522> <C_524> <O_524> <H_524> <L_505> <C_513> <O_512> <H_516> <L_504> <C_507> <O_507> <H_519> <L_507> <C_508> <O_508> <H_538> <L_508> <C_529> <O_529> <H_535> <L_517> <C_518> <O_518> <H_532> <L_513> <C_532> <O_532> <H_533> <L_523> <C_527> <O_527> <H_527> <L_507> <C_515> <O_515> <H_527> <L_511> <C_522> <O_522> <H_529> <L_515> <C_515> <O_516> <H_518> <L_508> <C_514> <O_514> <H_532> <L_511> <C_531> <O_531> <H_544> <L_528> <C_533> <O_532> <H_542> <L_526> <C_540> <O_540> <H_549> <L_539> <C_544> <O_544> <H_552> <L_536> <C_536> <O_536> <H_547> <L_533> <C_547> <O_547> <H_591> <L_546> <C_586> <O_586> <H_593> <L_571> <C_575> <O_578> <H_616> <L_578> <C_606> <O_606> <H_618> <L_589> <C_607> <O_608> <H_646> <L_603> <C_640> <O_643> <H_651> <L_636> <C_642> </chart>"
]


# =====================================================================
# 1) Vẽ lại chart + zone + entry/SL/TP từ chính AST đã parse
# =====================================================================
def plot_result(chart, think, action, out_path: str, title_prefix: str = "") -> None:
    """
    Dựng lại candlestick 50 nến input, vẽ đè zone_support/zone_resistance
    (nếu có), current_price, và SL/TP nếu action là BUY/SELL (TP suy ra
    qua derive_target — CÙNG hàm dùng ở forward_test.py, không tính lại
    theo công thức khác để tránh lệch 2 nơi).

    CHỈ nên gọi khi parse_result.is_well_formed() và sem_result.passed đều
    True — hàm này không tự kiểm tra lại, caller (run_one) chịu trách nhiệm
    quyết định có gọi hay không.
    """
    candles = chart.candles

    fig, ax = plt.subplots(figsize=(16, 6))

    # --- Candlestick đơn giản (wick = line, body = rectangle) ---
    for i, cnd in enumerate(candles):
        color = "tab:green" if cnd.c >= cnd.o else "tab:red"
        ax.plot([i, i], [cnd.l, cnd.h], color=color, linewidth=1)
        lower_body = min(cnd.o, cnd.c)
        height = max(abs(cnd.c - cnd.o), 1)  # tối thiểu 1 bin cho dễ nhìn nếu doji
        ax.add_patch(patches.Rectangle((i - 0.3, lower_body), 0.6, height, color=color))

    # --- Zone (support/resistance) ---
    if think.zone is not None:
        zone_color = "tab:blue" if think.zone.direction == "support" else "tab:orange"
        ax.axhspan(
            think.zone.lower_bin, think.zone.upper_bin, color=zone_color, alpha=0.15,
            label=f"zone_{think.zone.direction} [{think.zone.lower_bin}:{think.zone.upper_bin}]",
        )

    # --- current_price ---
    ax.axhline(
        think.current_price_bin, color="black", linestyle="--", linewidth=1,
        label=f"current_price={think.current_price_bin}",
    )

    # --- Entry/SL/TP nếu là lệnh thật (BUY/SELL) ---
    if action.action_type in ("BUY", "SELL") and action.sl is not None and action.rr is not None:
        entry = think.current_price_bin
        direction = "long" if action.action_type == "BUY" else "short"
        target = derive_target(entry, action.sl, action.rr, direction)

        ax.axhline(action.sl, color="red", linestyle=":", linewidth=1.5, label=f"SL={action.sl}")
        if target is not None:
            ax.axhline(
                target, color="tab:green", linestyle=":", linewidth=1.5,
                label=f"TP(RR{action.rr})={target}",
            )

    ax.set_title(
        f"{title_prefix}trend={think.trend} action={action.action_type} "
        f"sl={action.sl} rr={action.rr}"
    )
    ax.set_xlabel("candle index")
    ax.set_ylabel("bin")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 2) Generate + parse + chấm nhanh well-form/semantic cho 1 chart
# =====================================================================
def run_one(
    model, tokenizer, device, chart_text: str, max_new_tokens: int = 200, do_sample: bool = True,
    plot_dir: str | None = None, sample_idx: int = 0,
):
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
    print(f"[completion] {completion_text}")
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
        chart = parse_result.ast.chart
        print(f"trend={think.trend} action_type={action.action_type} sl={action.sl} rr={action.rr}")

        # --- CHỈ vẽ khi output "đúng" (well-formed VÀ semantic pass) ---
        if sem_result.passed and plot_dir is not None:
            out_path = os.path.join(plot_dir, f"sample_{sample_idx}.png")
            plot_result(chart, think, action, out_path, title_prefix=f"[sample {sample_idx}] ")
            print(f"  -> Đã lưu chart: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_repo", required=True, help="vd sullivan1502/tiny-pretrain")
    p.add_argument("--tokenizer_repo", default=None, help="mặc định DEFAULT_TOKENIZER_REPO trong app/tokenizer/hub.py")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--greedy", action="store_true", help="tắt sampling, dùng greedy decode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--plot_dir", default="./output/infer_plots",
        help="Thư mục lưu chart cho các sample well-formed + semantic pass (mặc định ./infer_plots)",
    )
    p.add_argument("--no_plot", action="store_true", help="Tắt hẳn việc vẽ chart")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = load_tokenizer(repo_id=args.tokenizer_repo)
    model = LlamaForCausalLM.from_pretrained(args.model_repo).to(device)
    model.eval()

    print(f"model_repo={args.model_repo}  vocab_size(tokenizer)={tokenizer.vocab_size}  "
          f"vocab_size(model.config)={model.config.vocab_size}")

    plot_dir = None if args.no_plot else args.plot_dir
    if plot_dir is not None:
        print(f"Chart của sample well-formed + semantic pass sẽ được lưu vào: {plot_dir}")

    for i, chart_text in enumerate(EXAMPLES):
        run_one(
            model, tokenizer, device, chart_text,
            max_new_tokens=args.max_new_tokens, do_sample=not args.greedy,
            plot_dir=plot_dir, sample_idx=i,
        )


if __name__ == "__main__":
    main()