"""
scripts/eval_val.py — Evaluate 1 model checkpoint trên val split của dataset GRPO
(schema mục 7.3 spec_trading_llm_v0.2.md: prompt/future_bins/symbol/window_id) —
model TỰ SINH think/action qua model.generate(), rồi parse + semantic-check +
evaluate_outcome (forward-test/counterfactual), CÙNG 1 nguồn logic với
app/training/reward/reward_func.py lúc train GRPO.

KHÁC với reward_func.py: KHÔNG áp round_config reward-shaping (K, zone_quality_bonus,
trade_fee_bins, weight_table) — mục đích ở đây là ĐO OUTCOME THẬT (well-form rate,
semantic pass rate, win rate, avg R-multiple theo trend x action_type), không phải
tính reward RL. --round_config chỉ dùng để khớp đúng zone_width/SL-distance range
của round đã train model này (ảnh hưởng semantic gate + is_sl_valid), KHÔNG dùng
K/zone_quality_bonus/trade_fee_bins/weight_table của config đó.

Cần dataset có future_bins (schema GRPO) — pretrain/SFT val KHÔNG dùng được ở đây
vì không có gì để forward-test.

Usage:
    python -m scripts.eval_val \\
        --model_repo sullivan1502/base-grpo-test \\
        --dataset_name sullivan1502/tlang-grpo \\
        --split val \\
        --batch_size 16 --max_new_tokens 64 \\
        --output_json ./eval_out/base_grpo_val.json

    # Khớp đúng zone/SL range của round1 (nếu model được train với round1.json):
    python -m scripts.eval_val --model_repo sullivan1502/base-grpo-test --dataset_name sullivan1502/tlang-grpo --split val --round_config ./rounds/round1.json --output_json ./output/eval_out/base_grpo_round1_val.json

    # Debug nhanh trên 20 sample đầu, greedy decode:
    python -m scripts.eval_val --model_repo ... --dataset_name ... --split val \\
        --limit 20 --greedy
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("eval_val")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_repo", required=True, help="Checkpoint model trên HF Hub (hoặc local dir)")
    p.add_argument("--tokenizer_repo", default=None, help="Mặc định DEFAULT_TOKENIZER_REPO (app/tokenizer/hub.py)")
    p.add_argument("--dataset_name", required=True, help="Dataset GRPO (schema prompt/future_bins/symbol/window_id)")
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int, default=None, help="Chỉ eval N sample đầu (debug nhanh) — mặc định cả split")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=64, help="think+action ngắn (~30-40 token thực đo)")
    p.add_argument("--greedy", action="store_true", help="Tắt sampling, dùng greedy decode")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--round_config", default=None,
        help="Path RoundConfig JSON — nếu truyền, dùng đúng zone_width/SL-distance range của round đó "
             "(KHÔNG dùng weight_table/K/zone_quality_bonus/trade_fee_bins của config này, chỉ dùng 2 range).",
    )
    p.add_argument("--output_json", default=None, help="Path lưu kết quả chi tiết + summary (JSON)")
    return p


@dataclass
class EvalRecord:
    window_id: Optional[str]
    symbol: Optional[str]
    prompt: str
    completion: str
    well_formed: bool
    well_form_score: float
    semantic_passed: Optional[bool]
    semantic_score: Optional[float]
    trend: Optional[str]
    action_type: Optional[str]
    outcome_status: Optional[str]
    r_multiple: Optional[float]


def _batched(seq: List[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i: i + n]


def evaluate_batch(
    model, tokenizer, device, rows: List[Dict[str, Any]],
    max_new_tokens: int, do_sample: bool, temperature: float, top_p: float,
    zone_width_min_bins: int, zone_width_max_bins: int,
    sl_min_dist_bins: int, sl_max_dist_bins: int,
) -> List[EvalRecord]:
    import torch

    from app.lang.parser import Parser
    from app.lang.semantic import SemanticChecker
    from app.training.reward.forward_test import evaluate_true_outcome

    prompts = [r["prompt"] for r in rows]

    # padding_side="left" (set 1 lần ở main()) + add_eos_token=False/add_bos_token=True
    # (cũng set ở main()) -> mỗi prompt encode thành <bos>+chart, không <eos> chèn giữa.
    # Nhờ left-padding, phần completion model tự sinh LUÔN bắt đầu đúng tại
    # input_ids.shape[1] cho toàn batch, không lẫn vào vùng pad.
    enc = tokenizer(prompts, add_special_tokens=True, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    gen_ids = out_ids[:, input_ids.shape[1]:]
    completions = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    checker = SemanticChecker(zone_width_min_bins=zone_width_min_bins, zone_width_max_bins=zone_width_max_bins)

    records: List[EvalRecord] = []
    for row, completion in zip(rows, completions):
        prompt = row["prompt"]
        full_text = prompt + " " + completion
        parse_result = Parser.from_text(full_text).parse()

        if not parse_result.is_well_formed():
            program = parse_result.ast
            records.append(EvalRecord(
                window_id=row.get("window_id"), symbol=row.get("symbol"),
                prompt=prompt, completion=completion,
                well_formed=False, well_form_score=parse_result.well_form_score(),
                semantic_passed=None, semantic_score=None,
                trend=program.think.trend if program and program.think else None,
                action_type=program.action.action_type if program and program.action else None,
                outcome_status=None, r_multiple=None,
            ))
            continue

        program = parse_result.ast
        think, action = program.think, program.action

        sem_result = checker.check(program)
        # evaluate_outcome tự trả (True, None) cho WAIT_*/HOLD (không có outcome),
        # và bao gồm luôn is_sl_valid (khoảng cách SL + đúng phía zone) cho BUY/SELL
        # — đây là ràng buộc "extra semantic" nằm ngoài bảng A/B/D/E của SemanticChecker
        # (xem docs/spec_trading_llm_v0.2.md mục 6.1), nên phải AND lại với sem_result.passed.
        extra_valid, forward_result, sl_valid = evaluate_true_outcome(
            action, think, [tuple(c) for c in row["future_bins"]],
            sl_min_dist_bins=sl_min_dist_bins, sl_max_dist_bins=sl_max_dist_bins,
        )
        semantic_passed = sem_result.passed and extra_valid

        records.append(EvalRecord(
            window_id=row.get("window_id"), symbol=row.get("symbol"),
            prompt=prompt, completion=completion,
            well_formed=True, well_form_score=parse_result.well_form_score(),
            semantic_passed=semantic_passed, semantic_score=sem_result.score,
            trend=think.trend, action_type=action.action_type,
            outcome_status=forward_result.status.value if (semantic_passed and forward_result) else None,
            r_multiple=forward_result.r_multiple if (semantic_passed and forward_result) else None,
        ))

    return records


def summarize(records: List[EvalRecord]) -> Dict[str, Any]:
    n = len(records)
    n_well_formed = sum(1 for r in records if r.well_formed)
    n_semantic_passed = sum(1 for r in records if r.semantic_passed)

    by_trend_action: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "r_multiples": [], "statuses": []})
    )
    for r in records:
        if not (r.well_formed and r.semantic_passed and r.trend and r.action_type):
            continue
        entry = by_trend_action[r.trend][r.action_type]
        entry["count"] += 1
        if r.r_multiple is not None:
            entry["r_multiples"].append(r.r_multiple)
        if r.outcome_status is not None:
            entry["statuses"].append(r.outcome_status)

    breakdown: Dict[str, Dict[str, dict]] = {}
    for trend, actions in by_trend_action.items():
        breakdown[trend] = {}
        for action_type, entry in actions.items():
            rms = entry["r_multiples"]
            statuses = entry["statuses"]
            n_status = len(statuses) or 1
            breakdown[trend][action_type] = {
                "count": entry["count"],
                "avg_r_multiple": (sum(rms) / len(rms)) if rms else None,
                "win_rate": (statuses.count("WIN") / n_status) if statuses else None,
                "loss_rate": (statuses.count("LOSS") / n_status) if statuses else None,
                "timeout_rate": (statuses.count("TIMEOUT") / n_status) if statuses else None,
            }

    return {
        "n_samples": n,
        "well_form_rate": (n_well_formed / n) if n else 0.0,
        "semantic_pass_rate_given_well_formed": (n_semantic_passed / n_well_formed) if n_well_formed else 0.0,
        "semantic_pass_rate_overall": (n_semantic_passed / n) if n else 0.0,
        "by_trend_action": breakdown,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Eval summary ===")
    print(f"n_samples = {summary['n_samples']}")
    print(f"well_form_rate = {summary['well_form_rate'] * 100:.1f}%")
    print(f"semantic_pass_rate (trong số well-formed) = {summary['semantic_pass_rate_given_well_formed'] * 100:.1f}%")
    print(f"semantic_pass_rate (trên toàn bộ split)    = {summary['semantic_pass_rate_overall'] * 100:.1f}%")
    print("\n-- Outcome theo trend x action_type (chỉ sample well-formed + semantic pass) --")
    for trend, actions in summary["by_trend_action"].items():
        print(f"trend={trend}")
        for action_type, stat in actions.items():
            avg_r = f"{stat['avg_r_multiple']:.2f}" if stat["avg_r_multiple"] is not None else "-"
            win_rate = f"{stat['win_rate'] * 100:.0f}%" if stat["win_rate"] is not None else "-"
            timeout_rate = f"{stat['timeout_rate'] * 100:.0f}%" if stat["timeout_rate"] is not None else "-"
            print(
                f"  {action_type:<12} count={stat['count']:<4} avg_R={avg_r:>6}  "
                f"win_rate={win_rate:>5}  timeout={timeout_rate:>5}"
            )


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch
    from datasets import load_dataset
    from transformers import LlamaForCausalLM

    from app.lang.semantic import SemanticChecker
    from app.tokenizer.hub import load_tokenizer
    from app.training.reward.forward_test import SL_MAX_DIST_BINS, SL_MIN_DIST_BINS

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU : {torch.cuda.get_device_name(0)}")

    # --- zone_width/SL-distance range: khớp round GRPO cụ thể nếu truyền --round_config,
    # ngược lại dùng hằng số module (SemanticChecker/forward_test.py). CHỈ lấy 2 range này
    # từ RoundConfig — K/zone_quality_bonus/trade_fee_bins/weight_table KHÔNG dùng ở đây
    # (đó là reward-shaping cho RL, không liên quan tới đo outcome thô).
    if args.round_config:
        from app.training.reward.round_config import RoundConfig
        rc = RoundConfig.load(args.round_config)
        zone_width_min_bins, zone_width_max_bins = rc.zone_width_min_bins, rc.zone_width_max_bins
        sl_min_dist_bins, sl_max_dist_bins = rc.sl_min_dist_bins, rc.sl_max_dist_bins
        logger.info(
            f"Dùng RoundConfig từ {args.round_config}: "
            f"zone=[{zone_width_min_bins},{zone_width_max_bins}] sl=[{sl_min_dist_bins},{sl_max_dist_bins}]"
        )
    else:
        zone_width_min_bins = SemanticChecker.ZONE_WIDTH_MIN_BINS
        zone_width_max_bins = SemanticChecker.ZONE_WIDTH_MAX_BINS
        sl_min_dist_bins, sl_max_dist_bins = SL_MIN_DIST_BINS, SL_MAX_DIST_BINS

    # --- Tokenizer — load qua Hub (mục 7.0 tokenizer_v0.1.md), KHÔNG build lại từ source.
    # add_eos_token=False/add_bos_token=True: khi encode prompt để generate, chỉ muốn
    # <bos>+chart (không <eos> chèn giữa chart và think) — cùng kỹ thuật với train_grpo.py.
    # padding_side="left": bắt buộc cho batch generate để phần completion model tự sinh
    # luôn nối tiếp đúng ngay sau input_ids.shape[1] cho MỌI sample trong batch.
    tok = load_tokenizer(repo_id=args.tokenizer_repo, allow_local_fallback=False)
    tok.add_eos_token = False
    tok.add_bos_token = True
    tok.padding_side = "left"
    logger.info(f"tokenizer vocab_size = {tok.vocab_size}")

    model = LlamaForCausalLM.from_pretrained(args.model_repo).to(device)
    model.eval()
    if model.config.vocab_size != tok.vocab_size:
        raise ValueError(
            f"vocab_size model ({model.config.vocab_size}) khác vocab_size tokenizer "
            f"({tok.vocab_size}) — checkpoint và tokenizer không khớp (vi phạm vocab contract)."
        )

    #ds = load_dataset(args.dataset_name, split=args.split)
    ds = load_dataset("parquet", data_files="data/dataset/XAUUSD_M1_Val_grpo_dataset.parquet", split='train')
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    logger.info(f"Loaded {len(ds)} sample từ {args.dataset_name} split={args.split}")

    rows = list(ds)
    all_records: List[EvalRecord] = []
    for batch in _batched(rows, args.batch_size):
        records = evaluate_batch(
            model, tok, device, batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            temperature=args.temperature, top_p=args.top_p,
            zone_width_min_bins=zone_width_min_bins, zone_width_max_bins=zone_width_max_bins,
            sl_min_dist_bins=sl_min_dist_bins, sl_max_dist_bins=sl_max_dist_bins,
        )
        all_records.extend(records)
        logger.info(f"  đã eval {len(all_records)}/{len(rows)}")

    summary = summarize(all_records)
    print_summary(summary)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"summary": summary, "records": [asdict(r) for r in all_records]},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"Đã lưu kết quả chi tiết -> {out_path}")


if __name__ == "__main__":
    main()