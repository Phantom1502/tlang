"""
train_grpo_v2.py — Script GRPO cho reward v2 (2 task `zone`/`action` tách
biệt, common gate = full SemanticChecker, xem app/training/reward/reward_func_v2.py
và app/training/reward/round_config_v2.py cho toàn bộ logic tính điểm).

CÙNG PATTERN resumable/theo-round như train_grpo.py (v1) — xem docstring
module đó cho giải thích đầy đủ về resume_checkpoint vs init_from_repo,
KV cache, seed theo round_id. KHÔNG lặp lại ở đây, chỉ nêu phần KHÁC v1:

1. DATASET: mỗi row GRPO gốc (prompt/future_bins/symbol/window_id) được
   NHÂN ĐÔI thành 2 row (task_id="zone"/"action", window_id hậu tố tương
   ứng) qua app.data_prepare.task_dataset.add_task_id_columns — dùng bản
   VẬT CHẤT HOÁ THẬT (datasets.Dataset thật, không phải object Python tự
   chế) để tương thích chắc chắn với GRPOTrainer/remove_unused_columns.

   CẢNH BÁO CHƯA GIẢI QUYẾT: 2 row cùng 1 window có `prompt` GIỐNG HỆT
   NHAU. GRPOTrainer gom completion theo prompt để tính advantage — NẾU
   trl gom theo prompt-string thô, 2 completion set của 2 task (chấm 2
   công thức khác thang) có nguy cơ bị trộn vào cùng 1 nhóm. CHƯA verify
   hành vi thật của trl==1.8.0 bằng test thực nghiệm (log group-key nội
   bộ) — nên coi phần này là RỦI RO CHƯA ĐÓNG, không phải đã xác nhận an
   toàn. Nếu xác nhận có rủi ro thật, phương án dự phòng: bỏ nhân đôi,
   mỗi row gốc chỉ sinh 1 dòng, đổi 1 hàm rẽ nhánh trong reward_func_v2.py
   sang blend 0.5*reward_zone + 0.5*reward_action.

2. REWARD: dùng `unified_reward_func_v2` (nhận thêm cột "task_id" —
   remove_unused_columns=False bắt buộc, giống future_bins).

3. STATE CẦN PERSIST THEO CHECKPOINT — GẤP ĐÔI so với v1 (2 buff
   controller ĐỘC LẬP thay vì 1):
     - action_buff_controller  -> "action_buff_state_v2.json"
     - zone_buff_controller    -> "zone_buff_state_v2.json"
     - rr_entropy_controller_v2 -> "rr_entropy_state_v2.json"
   Cả 3 chỉ load khi RESUME CHÍNH ROUND NÀY — round mới init từ round
   trước (--init_from_repo) LUÔN seed lại từ round_config (target/range
   có thể đã đổi giữa các round), không mang state buff/entropy của round
   khác sang.

4. STATS: StatsCollectorV2 (không phải StatsCollector v1) — persist theo
   rank giống v1, dùng watermark (mark_step_boundary) để nuôi CẢ 2 buff
   controller từ CÙNG 1 nguồn records, không có 2 bộ đếm tách rời.

Usage:
    python -m app.training.train_grpo_v2 \\
        --model_size tiny --round_id round1_v2 \\
        --repo_id my-org/tlang-grpo-v2-round1 \\
        --init_from_repo my-org/tlang-sft \\
        --round_config ./rounds/round1_v2.json \\
        --dataset_name my-org/tlang-grpo \\
        --output_dir ./output/grpo-v2-round1 \\
        --save_steps 50 --max_steps 500

Xem report (không train, chỉ gộp stats đã có + in summary):
    python -m app.training.train_grpo_v2 --round_id round1_v2 \\
        --output_dir ./output/grpo-v2-round1 --report_only
"""
from __future__ import annotations

import argparse
import glob
import logging
import os

logger = logging.getLogger("train_grpo_v2")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--model_size", choices=["tiny", "small", "base", "large"], default="tiny")
    p.add_argument("--repo_id", default=None, help="Checkpoint repo của CHÍNH round này (resume-from/push-to)")
    p.add_argument(
        "--init_from_repo", default=None,
        help="Nguồn init NẾU round này chưa có checkpoint nào: SFT repo (round 1) "
             "hoặc checkpoint round liền trước (round N>1) — truyền tay, không tự động.",
    )

    p.add_argument("--round_id", required=True, help="vd: round1_v2, round2_v2")
    p.add_argument("--round_config", default=None, help="Path tới RoundConfigV2 JSON — BẮT BUỘC nếu không --report_only")

    p.add_argument("--dataset_name", default=None, help="Dataset GRPO gốc (schema prompt/future_bins/symbol/window_id)")
    p.add_argument("--train_split", default="train")
    p.add_argument(
        "--dataset_shuffle_seed", type=int, default=None,
        help="Seed để shuffle dataset SAU KHI nhân đôi task (add_task_id_columns) — mặc định None "
             "(không shuffle thêm, dựa hoàn toàn vào Trainer). Khuyến nghị đặt 1 giá trị cụ thể để "
             "tránh 2 nửa zone/action nằm liền khối.",
    )

    p.add_argument("--output_dir", required=True, help="local dir — dùng để detect resume trong-session + lưu stats")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--max_completion_length", type=int, default=64)

    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--min_p", type=float, default=0.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)

    p.add_argument("--num_generations", type=int, default=12)
    p.add_argument("--use_vllm", action="store_true", default=False)

    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--hf_token", default=None)

    p.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    p.add_argument("--bf16", dest="fp16", action="store_false")

    p.add_argument(
        "--report_only", action="store_true",
        help="Bỏ qua train hoàn toàn — chỉ gộp mọi {round_id}_stats_v2_rank*.json trong "
             "--output_dir rồi in summary(). Dùng để xem thống kê giữa chừng bất cứ lúc nào.",
    )

    return p


def _stats_glob_pattern(output_dir: str, round_id: str) -> str:
    return os.path.join(output_dir, f"{round_id}_stats_v2_rank*.json")


def _stats_path_for_rank(output_dir: str, round_id: str, rank: int) -> str:
    return os.path.join(output_dir, f"{round_id}_stats_v2_rank{rank}.json")


def run_report_only(output_dir: str, round_id: str) -> None:
    from app.training.reward.reward_func_v2 import StatsCollectorV2

    pattern = _stats_glob_pattern(output_dir, round_id)
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"Không tìm thấy file stats nào khớp {pattern!r} — round chưa chạy step nào, hoặc sai --output_dir/--round_id.")
        return
    print(f"Gộp {len(paths)} file: {paths}")
    merged = StatsCollectorV2.merge_from_files(paths)
    merged.print_summary()


def build_model_for_round(resume_checkpoint, init_from_repo: str | None, model_size: str, vocab_size: int):
    """Giống hệt logic v1 (xem train_grpo.py:build_model_for_round) — không
    đổi gì, GRPO v2 vẫn KHÔNG có nhánh from-scratch."""
    from app.training.common import load_model_with_vocab_check

    if resume_checkpoint is not None:
        model = load_model_with_vocab_check(resume_checkpoint, vocab_size)
    else:
        if not init_from_repo:
            raise RuntimeError(
                "Chưa có checkpoint GRPO nào để resume cho round này, VÀ --init_from_repo "
                "không được truyền — GRPO v2 cần 1 checkpoint nguồn (SFT cho round 1, hoặc round "
                "liền trước cho round N>1). Không có nhánh from-scratch."
            )
        logger.info(f"Chưa có checkpoint GRPO round này — init từ: {init_from_repo}")
        model = load_model_with_vocab_check(init_from_repo, vocab_size)

    model.config.use_cache = True   # BẮT BUỘC — GRPO cần generate() nhanh cho rollout (xem train_grpo.py v1)
    logger.info(f"model.config.use_cache = {model.config.use_cache}")
    return model


def _seed_from_round_id(round_id: str) -> int:
    import hashlib
    return int(hashlib.md5(round_id.encode()).hexdigest(), 16) % (2**31)


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.report_only:
        run_report_only(args.output_dir, args.round_id)
        return

    if not args.round_config:
        print("--round_config bắt buộc khi train thật (không --report_only). "
              "Xem app/training/reward/round_config_v2.py cho schema JSON.")
        raise SystemExit(1)
    if not args.repo_id or not args.dataset_name:
        print("--repo_id và --dataset_name bắt buộc khi train thật.")
        raise SystemExit(1)

    import torch
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    from app.training.task_dataset import add_task_id_columns
    from app.training.reward.reward_func_v2 import (
        StatsCollectorV2,
        action_buff_controller,
        zone_buff_controller,
        rr_entropy_controller_v2,
        stats_collector_v2,
        unified_reward_func_v2,
    )
    from app.training.reward.round_config_v2 import RoundConfigV2
    import app.training.reward.reward_func_v2 as reward_func_v2_module

    from app.tokenizer.hub import load_tokenizer
    from app.training.common import resolve_resume_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    push_to_hub = False
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        push_to_hub = True
    else:
        logger.warning("Không có --hf_token — checkpoint round này sẽ chỉ lưu local, không push.")

    # ------------------------------------------------------------
    # RoundConfigV2 — BẮT BUỘC tường minh, fail-loud nếu thiếu field/sai
    # invariant (xem round_config_v2.py __post_init__).
    # ------------------------------------------------------------
    round_config = RoundConfigV2.load(args.round_config)
    if round_config.round_id != args.round_id:
        logger.warning(
            f"round_config.round_id={round_config.round_id!r} khác --round_id={args.round_id!r} "
            f"truyền vào — vẫn dùng config đã load, kiểm tra lại có nhầm file round không."
        )
    reward_func_v2_module.set_active_round_config_v2(round_config)
    logger.info(
        f"RoundConfigV2 đã load: zone_width=[{round_config.zone_width_min_bins},{round_config.zone_width_max_bins}] "
        f"sl_dist=[{round_config.sl_min_dist_bins},{round_config.sl_max_dist_bins}] "
        f"entry_score_weight={round_config.entry_score_weight} trade_fee_bins={round_config.trade_fee_bins}\n"
        f"action targets: hold={round_config.action_hold_target_ratio} buy={round_config.action_buy_target_ratio} "
        f"sell={round_config.action_sell_target_ratio} cancel_buy={round_config.action_cancel_buy_target_ratio} "
        f"cancel_sell={round_config.action_cancel_sell_target_ratio} wait_buy={round_config.action_wait_buy_target_ratio} "
        f"wait_sell={round_config.action_wait_sell_target_ratio}\n"
        f"zone targets (suy từ action_hold_target_ratio): has_zone={round_config.zone_target_has_zone_ratio} "
        f"no_zone={round_config.zone_target_no_zone_ratio}"
    )

    # ------------------------------------------------------------
    # Tokenizer — giống hệt v1 (xem giải thích add_eos_token/add_bos_token
    # quirk trong train_grpo.py v1, không lặp lại).
    # ------------------------------------------------------------
    tok = load_tokenizer(repo_id=args.repo_id, allow_local_fallback=False)
    logger.info(f"tokenizer vocab_size = {tok.vocab_size}")
    tok.add_eos_token = False
    tok.add_bos_token = True

    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, args.repo_id)
    model = build_model_for_round(resume_checkpoint, args.init_from_repo, args.model_size, tok.vocab_size)

    # ------------------------------------------------------------
    # StatsCollectorV2 — load lại records đã dump của round này (nếu Colab
    # bị ngắt và đây là lần chạy lại) TRƯỚC khi log tiếp.
    # ------------------------------------------------------------
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    stats_path = _stats_path_for_rank(args.output_dir, args.round_id, rank)
    stats_collector_v2._records = StatsCollectorV2.load(stats_path)._records
    logger.info(f"[rank={rank}] StatsCollectorV2: nạp lại {len(stats_collector_v2._records)} record cũ.")

    # ------------------------------------------------------------
    # action_buff_controller / zone_buff_controller state — GẮN VÀO
    # CHECKPOINT (2 file JSON riêng), CHỈ load khi RESUME CHÍNH round này.
    # Round MỚI init từ round trước (--init_from_repo) LUÔN seed lại từ
    # round_config — không mang state buff của round khác sang.
    # ------------------------------------------------------------
    def _load_or_seed(controller, filename: str, label: str) -> None:
        loaded = False
        if resume_checkpoint is not None:
            state_path = os.path.join(resume_checkpoint, filename)
            loaded = controller.load(state_path)
            if loaded:
                logger.info(f"[rank={rank}] Đã khôi phục {label} từ {state_path}: {controller.snapshot()}")
            else:
                logger.warning(
                    f"[rank={rank}] Không đọc được {state_path} — fallback seed lại từ round_config "
                    f"(MẤT continuity của {label}, KHÔNG mất tính đúng đắn)."
                )
        if not loaded:
            controller.seed_from_round_config(round_config)
            logger.info(f"[rank={rank}] Seed {label} từ round_config: {controller.snapshot()}")

    _load_or_seed(action_buff_controller, "action_buff_state_v2.json", "action_buff_controller")
    _load_or_seed(zone_buff_controller, "zone_buff_state_v2.json", "zone_buff_controller")

    rr_entropy_loaded = False
    if resume_checkpoint is not None:
        rr_state_path = os.path.join(resume_checkpoint, "rr_entropy_state_v2.json")
        rr_entropy_loaded = rr_entropy_controller_v2.load(rr_state_path)
        if rr_entropy_loaded:
            logger.info(f"[rank={rank}] Đã khôi phục rr_entropy_controller_v2 từ {rr_state_path}: {rr_entropy_controller_v2.snapshot()}")
        else:
            logger.warning(f"[rank={rank}] Không đọc được {rr_state_path} — fallback seed lại từ round_config.")
    if not rr_entropy_loaded:
        rr_entropy_controller_v2.seed_from_round_config(round_config)
        logger.info(f"[rank={rank}] Seed rr_entropy_controller_v2 từ round_config: {rr_entropy_controller_v2.snapshot()}")

    # ------------------------------------------------------------
    # Dataset — load raw GRPO gốc rồi nhân đôi theo task_id (xem cảnh báo
    # ở docstring module về rủi ro group-by-prompt của GRPOTrainer).
    # remove_unused_columns PHẢI False (cần cả future_bins lẫn task_id).
    # ------------------------------------------------------------
    raw = load_dataset(args.dataset_name, split=args.train_split)
    dataset_seed = args.dataset_shuffle_seed if args.dataset_shuffle_seed is not None else _seed_from_round_id(args.round_id)
    expanded = add_task_id_columns(raw, shuffle_seed=dataset_seed)
    logger.info(f"Dataset gốc {len(raw)} row -> sau nhân đôi task {len(expanded)} row (shuffle_seed={dataset_seed}).")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        seed=_seed_from_round_id(args.round_id),
        remove_unused_columns=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        max_completion_length=args.max_completion_length,

        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else None,
        min_p=args.min_p if args.min_p > 0 else None,
        repetition_penalty=args.repetition_penalty,

        num_generations=args.num_generations,
        use_vllm=args.use_vllm,
        fp16=args.fp16,
        bf16=not args.fp16,
        push_to_hub=push_to_hub,
        hub_model_id=args.repo_id if push_to_hub else None,
        hub_strategy="checkpoint" if push_to_hub else "every_save",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=[],
    )

    class StatsPersistCallbackV2(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            # Nuôi buff từ ĐÚNG 1 nguồn (StatsCollectorV2), KHÔNG có bộ đếm
            # riêng nào khác — mỗi namespace tự lọc theo key_fn của mình.
            action_counts, action_total = stats_collector_v2.counts_since_step_boundary(
                "action", key_fn=lambda r: r.action_type
            )
            zone_counts, zone_total = stats_collector_v2.counts_since_step_boundary(
                "zone", key_fn=lambda r: "HAS_ZONE" if r.has_zone else ("NO_ZONE" if r.has_zone is False else None)
            )
            action_buff_controller.on_step_end(round_config, action_counts, action_total)
            zone_buff_controller.on_step_end(round_config, zone_counts, zone_total)
            rr_entropy_controller_v2.on_step_end(round_config)
            stats_collector_v2.mark_step_boundary()

        def on_log(self, args, state, control, **kwargs):
            print("\n=== ACTION BUFF CONTROLLER ===")
            for group, metrics in action_buff_controller.snapshot().items():
                print(f"{group}: ema_ratio={metrics['ema_ratio']:.4f}, buff={metrics['buff']:.4f}, prev_error={metrics['prev_error']:.4f}")
            print("\n=== ZONE BUFF CONTROLLER ===")
            for group, metrics in zone_buff_controller.snapshot().items():
                print(f"{group}: ema_ratio={metrics['ema_ratio']:.4f}, buff={metrics['buff']:.4f}, prev_error={metrics['prev_error']:.4f}")
            rr_snap = rr_entropy_controller_v2.snapshot()
            if rr_snap:
                print(f"\nRR_ENTROPY: ema_entropy={rr_snap['ema_entropy']:.4f}, bonus={rr_snap['bonus']:.4f}, prev_error={rr_snap['prev_error']:.4f}")

        def on_save(self, args, state, control, **kwargs):
            n_records = len(stats_collector_v2._records)
            print(f"\n=== [step={state.global_step}] Chu kỳ report vừa xong ({n_records} record) ===")
            stats_collector_v2.print_summary()
            print(f"action_buff_controller hiện tại: {action_buff_controller.snapshot()}")
            print(f"zone_buff_controller hiện tại: {zone_buff_controller.snapshot()}")
            print(f"rr_entropy_controller_v2 hiện tại: {rr_entropy_controller_v2.snapshot()}\n")

            stats_collector_v2.save(stats_path)
            stats_collector_v2.reset()

            ckpt_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            if os.path.isdir(ckpt_dir):
                action_buff_controller.save(os.path.join(ckpt_dir, "action_buff_state_v2.json"))
                zone_buff_controller.save(os.path.join(ckpt_dir, "zone_buff_state_v2.json"))
                rr_entropy_controller_v2.save(os.path.join(ckpt_dir, "rr_entropy_state_v2.json"))
                logger.info(f"[rank={rank}] Đã lưu action/zone_buff_state + rr_entropy_state -> {ckpt_dir}/")
            else:
                logger.warning(f"[rank={rank}] Checkpoint dir {ckpt_dir} chưa tồn tại lúc on_save — bỏ qua lưu state.")

        def on_train_end(self, args, state, control, **kwargs):
            print("\n=== [train_end] Chu kỳ report cuối cùng ===")
            stats_collector_v2.print_summary()
            print(f"action_buff_controller cuối: {action_buff_controller.snapshot()}")
            print(f"zone_buff_controller cuối: {zone_buff_controller.snapshot()}")
            print(f"rr_entropy_controller_v2 cuối: {rr_entropy_controller_v2.snapshot()}\n")
            stats_collector_v2.save(stats_path)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=unified_reward_func_v2,
        args=training_args,
        train_dataset=expanded,
        processing_class=tok,
        callbacks=[StatsPersistCallbackV2()],
    )

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    trainer.save_model()
    canonical_tok = load_tokenizer(repo_id=args.repo_id, allow_local_fallback=False)
    canonical_tok.save_pretrained(args.output_dir)
    action_buff_controller.save(os.path.join(args.output_dir, "action_buff_state_v2.json"))
    zone_buff_controller.save(os.path.join(args.output_dir, "zone_buff_state_v2.json"))
    rr_entropy_controller_v2.save(os.path.join(args.output_dir, "rr_entropy_state_v2.json"))
    stats_collector_v2.save(stats_path)

    if push_to_hub:
        trainer.push_to_hub(commit_message=f"GRPO v2 {args.round_id} checkpoint")
        logger.info(f"Đã push lên: https://huggingface.co/{args.repo_id}")

    if trainer.is_world_process_zero():
        print(f"\n=== Report round {args.round_id} (rank {rank} — chạy lại với --report_only để gộp mọi rank) ===")
        stats_collector_v2.print_summary()


if __name__ == "__main__":
    main()