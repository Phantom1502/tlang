"""
train_grpo.py — Script GRPO (docs/train_pipeline_v0.1.md mục 6, spec mục 8).

CÙNG PATTERN resumable như train_pretrain.py/train_sft.py (mục 4.1/5.1), nhưng
theo ROUND thay vì tuyến tính (mục 6.1/6.3):

    round_repo (this round's own checkpoint) đã tồn tại (local hoặc Hub)
        -> resume TRAINING CỦA CHÍNH ROUND NÀY (session Colab bị ngắt, chạy tiếp)
    round_repo CHƯA tồn tại
        -> init từ --init_from_repo (SFT checkpoint cho round 1, hoặc checkpoint
           của round liền trước cho round N>1 — TAY truyền, KHÔNG tự động hoá
           việc chuyển round, đúng quyết định spec mục 6.1/6.3)

RoundConfig (zone_width/SL range + target_action_ratio + buff step/cap) BẮT BUỘC
tường minh qua --round_config, fail-loud nếu thiếu (xem
app/training/reward/round_config.py) — generator/pretrain/SFT không liên quan gì
tới config này, CHỈ GRPO mới cần vì chỉ tới đây outcome thật mới cho biết nên
nới/siết zone/SL/tỉ lệ BUY-SELL thế nào.

REWARD DESIGN (đã đổi so với bản trước — xem app/training/reward/reward_func.py):
    1. Gate well-formed + semantic: fail -> giữ nguyên điểm gate, return ngay.
       Pass cả 2 -> set về K = round_config.pass_gate2_bonus (đồng đều mọi mẫu).
    2. Từ K, cộng 2 nhánh SONG SONG:
       - zone_score:    mọi action có zone (BUY/SELL/CANCEL_*/WAIT_*), liên tục
                         theo r_multiple của probe_zone_quality.
       - outcome_score: CHỈ BUY/SELL, = r_multiple thật - phí + buff động.
    3. Buff động (ActionBuffTable, KHÔNG còn WeightTable nhân trọng số cũ):
       mỗi save_steps, so tỉ lệ (BUY+SELL)/tổng action thực tế với
       round_config.target_action_ratio — thiếu thì tăng buff BUY+SELL, dư thì
       giảm, cùng 1 delta cho cả 2 (xem update_buffs_from_stats()).

StatsCollector persist ra đĩa theo pattern load-rồi-append (mục đã thống nhất
khi thiết kế round_config.py): mỗi rank tự dump 1 file riêng, mỗi lần script
khởi động lại (Colab bị ngắt, chạy lại NHIỀU LẦN trong CÙNG 1 round) đều load
lại file cũ trước khi log tiếp — không mất thống kê giữa các lần chạy.
ActionBuffTable KHÔNG persist ra đĩa — chỉ sống trong process hiện tại, tự học
lại từ đầu (reset về 0) nếu Colab bị ngắt và chạy lại (chấp nhận được vì
update_buffs_from_stats() tính lại ngay ở lần on_save đầu tiên dựa trên
StatsCollector đã load lại đầy đủ).

KV CACHE: model load từ SFT/round trước có thể mang theo use_cache=False (nếu
checkpoint nguồn từng train với gradient_checkpointing bật, cache tự bị tắt
lúc lưu config) — GRPO cần generate() rất nhiều lần mỗi step (rollout), nên
PHẢI bật lại use_cache=True ngay sau khi load model, bất kể checkpoint nguồn
lưu gì. GRPOConfig.gradient_checkpointing=True (default của TRL) vẫn hoạt
động bình thường cho pha forward/backward — Trainer tự tắt cache lúc cần cho
training step, bật lại cho generation step; ta chỉ cần đảm bảo GIÁ TRỊ BAN
ĐẦU không bị kẹt ở False từ checkpoint cũ.

Usage:
    python train_grpo.py \\
        --model_size tiny --round_id round1 \\
        --repo_id my-org/tlang-grpo-round1 \\
        --init_from_repo my-org/tlang-sft \\
        --round_config ./rounds/round1.json \\
        --dataset_name my-org/tlang-grpo \\
        --output_dir ./output/grpo-round1 \\
        --save_steps 50 --max_steps 500

Xem report (không train, chỉ gộp stats đã có + in summary):
    python train_grpo.py --round_id round1 --output_dir ./output/grpo-round1 --report_only
"""
from __future__ import annotations

import argparse
import glob
import logging
import os

logger = logging.getLogger("train_grpo")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- Model / checkpoint naming (mục 6.1) ---
    p.add_argument("--model_size", choices=["tiny", "small", "base", "large"], default="tiny")
    p.add_argument("--repo_id", default=None, help="Checkpoint repo của CHÍNH round này (resume-from/push-to)")
    p.add_argument(
        "--init_from_repo", default=None,
        help="Nguồn init NẾU round này chưa có checkpoint nào: SFT repo (round 1) "
             "hoặc checkpoint round liền trước (round N>1) — truyền tay, không tự động.",
    )

    # --- RoundConfig (zone/SL range + target_action_ratio + buff step/cap, tường minh) ---
    p.add_argument("--round_id", required=True, help="vd: round1, round2 — dùng đặt tên file stats/report")
    p.add_argument("--round_config", default=None, help="Path tới RoundConfig JSON — BẮT BUỘC nếu không --report_only")

    # --- Dataset GRPO (schema mục 7.3: prompt/future_bins/symbol/window_id) ---
    p.add_argument("--dataset_name", default=None, help="vd: my-org/tlang-grpo")
    p.add_argument("--train_split", default="train")

    # --- Training loop ---
    p.add_argument("--output_dir", required=True, help="local dir — dùng để detect resume trong-session + lưu stats")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-6, help="GRPO thường cần LR nhỏ hơn nhiều so với SFT")
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--max_completion_length", type=int, default=64, help="think+action block ngắn (~30-40 token thực đo)")

    p.add_argument("--temperature", type=float, default=1.1, help="Tăng >1.0 để rollout đa dạng hơn giữa num_generations completions")
    p.add_argument("--top_p", type=float, default=1.0, help="1.0 = không cắt tail, giữ tối đa đa dạng")
    p.add_argument("--top_k", type=int, default=0, help="0/None = tắt top_k filtering")
    p.add_argument("--min_p", type=float, default=0.0, help="0.0 = tắt")
    p.add_argument("--repetition_penalty", type=float, default=1.0)

    # --- GRPO-specific (mục 6.1) ---
    p.add_argument("--num_generations", type=int, default=12, help="group size — mục 6.1, chỉnh theo VRAM nếu OOM")
    p.add_argument(
        "--use_vllm", action="store_true", default=False,
        help="Mặc định TẮT (mục 6.1: seq_len ngắn + model nhỏ + T4 single-GPU risk OOM colocate mode). "
             "Bật lại nếu scale lên large+/đổi hạ tầng VRAM lớn hơn.",
    )

    # --- Push theo chu kỳ (mục 4.2, áp dụng lại cho GRPO) ---
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--hf_token", default=None)

    # --- Hạ tầng ---
    p.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    p.add_argument("--bf16", dest="fp16", action="store_false")

    # --- Report-only mode: KHÔNG train, chỉ gộp stats đã dump + in summary ---
    p.add_argument(
        "--report_only", action="store_true",
        help="Bỏ qua train hoàn toàn — chỉ gộp mọi round{round_id}_stats_rank*.json trong "
             "--output_dir rồi in summary(). Dùng để xem thống kê giữa chừng bất cứ lúc nào.",
    )

    return p


def _stats_glob_pattern(output_dir: str, round_id: str) -> str:
    return os.path.join(output_dir, f"{round_id}_stats_rank*.json")


def _stats_path_for_rank(output_dir: str, round_id: str, rank: int) -> str:
    return os.path.join(output_dir, f"{round_id}_stats_rank{rank}.json")


def run_report_only(output_dir: str, round_id: str) -> None:
    from app.training.reward.reward_func import StatsCollector

    pattern = _stats_glob_pattern(output_dir, round_id)
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"Không tìm thấy file stats nào khớp {pattern!r} — round chưa chạy step nào, hoặc sai --output_dir/--round_id.")
        return
    print(f"Gộp {len(paths)} file: {paths}")
    merged = StatsCollector.merge_from_files(paths)
    merged.print_summary()


def build_model_for_round(resume_checkpoint, init_from_repo: str | None, model_size: str, vocab_size: int):
    """
    resume_checkpoint có sẵn (round này đã train dở, session bị ngắt) -> load
    qua đó, giữ optimizer/scheduler state (trainer.train(resume_from_checkpoint=...)
    tự khôi phục, việc load ở đây chỉ để có 1 model instance hợp lệ).

    resume_checkpoint=None (round này CHƯA từng train) -> BẮT BUỘC có
    --init_from_repo (SFT checkpoint cho round 1, checkpoint round trước cho
    round N>1) — GRPO KHÔNG có nhánh from-scratch, giống SFT.

    LUÔN bật lại use_cache=True sau khi load, bất kể checkpoint nguồn lưu gì
    (xem giải thích ở docstring module) — GRPO cần generate() nhanh cho rollout.
    """
    from app.training.common import load_model_with_vocab_check

    if resume_checkpoint is not None:
        model = load_model_with_vocab_check(resume_checkpoint, vocab_size)
    else:
        if not init_from_repo:
            raise RuntimeError(
                "Chưa có checkpoint GRPO nào để resume cho round này, VÀ --init_from_repo "
                "không được truyền — GRPO cần 1 checkpoint nguồn (SFT cho round 1, hoặc round "
                "liền trước cho round N>1, xem mục 6.1). Không có nhánh from-scratch."
            )
        logger.info(f"Chưa có checkpoint GRPO round này — init từ: {init_from_repo}")
        model = load_model_with_vocab_check(init_from_repo, vocab_size)

    model.config.use_cache = True   # BẮT BUỘC — xem giải thích KV CACHE ở docstring module
    logger.info(f"model.config.use_cache = {model.config.use_cache}")
    return model


def _seed_from_round_id(round_id: str) -> int:
    """Derive 1 seed int ổn định từ round_id — mỗi round_id khác nhau
    -> seed khác nhau -> shuffle khác nhau, nhưng CÙNG round_id chạy lại
    (session bị ngắt giữa chừng, resume) vẫn ra cùng seed -> không phá
    vỡ tính resumable đã thiết kế (mục 6.1/6.3)."""
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
              "Mỗi round GRPO cần config tường minh (zone/SL range + target_action_ratio + "
              "buff step/cap), xem app/training/reward/round_config.py.")
        raise SystemExit(1)
    if not args.repo_id or not args.dataset_name:
        print("--repo_id và --dataset_name bắt buộc khi train thật.")
        raise SystemExit(1)

    import torch
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback

    from app.tokenizer.hub import load_tokenizer
    from app.training.common import resolve_resume_checkpoint
    from app.training.reward.reward_func import (
        StatsCollector,
        action_buffs,
        hold_buff,
        stats_collector,
        unified_reward_func,
        update_buffs_from_stats,
    )
    from app.training.reward.round_config import RoundConfig
    import app.training.reward.reward_func as reward_func_module

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
    # RoundConfig — BẮT BUỘC tường minh, fail-loud nếu thiếu field
    # (xem round_config.py). set_active_round_config() reset action_buffs
    # về 0 cho round mới (buff KHÔNG mang từ round trước sang).
    # ------------------------------------------------------------
    round_config = RoundConfig.load(args.round_config)
    if round_config.round_id != args.round_id:
        logger.warning(
            f"round_config.round_id={round_config.round_id!r} khác --round_id={args.round_id!r} "
            f"truyền vào — vẫn dùng config đã load, nhưng kiểm tra lại có nhầm file round không."
        )
    reward_func_module.set_active_round_config(round_config)
    logger.info(
        f"RoundConfig đã load: zone_width=[{round_config.zone_width_min_bins},{round_config.zone_width_max_bins}] "
        f"sl_dist=[{round_config.sl_min_dist_bins},{round_config.sl_max_dist_bins}] "
        f"target_action_ratio={round_config.target_action_ratio} "
        f"buff_step={round_config.buff_step} buff_range=[{round_config.buff_min},{round_config.buff_max}] "
        f"K={round_config.pass_gate2_bonus} zone_score_scale={round_config.zone_score_scale}"
    )

    # ------------------------------------------------------------
    # Tokenizer — luôn load qua Hub từ CHÍNH repo_id của round này (tokenizer
    # đã được add sẵn vào mỗi model repo trước khi train — quy ước quản lý,
    # xem app/tokenizer/hub.py), KHÔNG build lại từ source.
    # ------------------------------------------------------------
    tok = load_tokenizer(repo_id=args.repo_id, allow_local_fallback=False)
    logger.info(f"tokenizer vocab_size = {tok.vocab_size}")

    # ------------------------------------------------------------
    # GRPO-only tokenizer quirk: TRL gọi `self.processing_class(text=prompts)`
    # KHÔNG truyền add_special_tokens (mặc định True) để tokenize PROMPT lúc
    # rollout — post_processor mặc định của mình bọc CẢ <bos> lẫn <eos> quanh
    # prompt, khiến model thấy 1 <eos> giữa chart và think (sai — <eos> chỉ
    # nên đứng cuối CẢ sequence, đúng convention lúc SFT: <bos>+prompt+
    # completion+<eos>). Set add_eos_token=False (giữ add_bos_token=True) để
    # tokenizer tự rebuild post_processor thành "chỉ <bos>, không <eos>" cho
    # MỌI lần encode với add_special_tokens=True sau đây — cần thiết vì TRL
    # không cho cách nào khác để chỉnh add_special_tokens từ ngoài.
    #
    # ĐÁNH ĐỔI: đây là mutation VĨNH VIỄN trên chính object `tok` — Trainer
    # tự động lưu `processing_class` (= tok đã mutate) vào MỌI checkpoint
    # theo save_steps, nên tokenizer.json trong các checkpoint GRPO từ giờ
    # sẽ KHÁC bản canonical trên Hub (thiếu eos-wrap). Không ảnh hưởng pipeline
    # CỦA CHÍNH TA (mọi lần chạy lại đều load_tokenizer() thẳng từ Hub, không
    # đọc lại tokenizer từ checkpoint local) — chỉ ảnh hưởng nếu ai đó load
    # RIÊNG tokenizer từ 1 checkpoint GRPO cụ thể cho việc khác. Artifact CUỐI
    # CÙNG được vá lại đúng bằng canonical_tok bên dưới (xem cuối main()).
    tok.add_eos_token = False
    tok.add_bos_token = True

    # ------------------------------------------------------------
    # Resume checkpoint CỦA CHÍNH ROUND NÀY — tìm TRƯỚC khi khởi tạo Trainer.
    # None -> model init từ --init_from_repo (SFT hoặc round trước).
    # ------------------------------------------------------------
    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, args.repo_id)
    model = build_model_for_round(resume_checkpoint, args.init_from_repo, args.model_size, tok.vocab_size)

    # ------------------------------------------------------------
    # StatsCollector — load lại records đã dump của round này (nếu Colab bị
    # ngắt và đây là lần chạy lại) TRƯỚC khi log tiếp, để file cuối cùng luôn
    # tích luỹ ĐỦ toàn bộ round, không chỉ session hiện tại. Path theo rank
    # (mỗi process 1 file riêng — reward_func chạy độc lập trên từng process).
    # rank tạm dùng RANK/LOCAL_RANK env (chuẩn torchrun/accelerate) — an toàn
    # kể cả trước khi GRPOTrainer khởi tạo xong distributed state.
    # ------------------------------------------------------------
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    stats_path = _stats_path_for_rank(args.output_dir, args.round_id, rank)
    loaded_stats, loaded_buffs, loaded_hold = StatsCollector.load(stats_path)
    stats_collector._records = loaded_stats._records
    for action_type, value in loaded_buffs.items():
        action_buffs.set(action_type, value)
    hold_buff.set(loaded_hold)
    logger.info(
        f"[rank={rank}] StatsCollector: nạp lại {len(stats_collector._records)} record của chu kỳ dở "
        f"(nếu session bị ngắt giữa chừng); action_buffs khôi phục = {action_buffs.snapshot()}; "
        f"hold_buff khôi phục = {hold_buff.get()}"
    )

    # Nạp lại buff ngay từ record cũ (nếu resume giữa round) để buff không bị
    # reset về 0 rồi mất vài chu kỳ save_steps mới bắt kịp lại trạng thái cũ.
    if stats_collector._records:
        update_buffs_from_stats(stats_collector, round_config, action_buffs, hold=hold_buff)
        logger.info(
            f"[rank={rank}] Buff nạp lại từ stats cũ: {action_buffs.snapshot()}; "
            f"hold_buff = {hold_buff.get()}"
        )

    # ------------------------------------------------------------
    # Dataset GRPO — chỉ cần "prompt" (model tự sinh phần còn lại) +
    # future_bins/symbol/window_id cho reward_func — remove_unused_columns
    # PHẢI False (mục 6.1/8.2), nếu không TRL tự xoá hết cột trừ "prompt".
    # ------------------------------------------------------------
    raw = load_dataset(args.dataset_name, split=args.train_split)

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

        # sample method
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

    # ------------------------------------------------------------
    # Persist stats + update buff theo đúng chu kỳ save_steps (mục 4.2:
    # "push theo chu kỳ, không phải 1 lần lúc kết thúc" — áp dụng lại triết
    # lý đó cho stats VÀ cho buff động).
    # ------------------------------------------------------------
    class StatsPersistCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            n_records = len(stats_collector._records)

            update_buffs_from_stats(stats_collector, round_config, action_buffs, hold=hold_buff)

            # In thống kê CHU KỲ VỪA XONG ra console — không cần --report_only nữa,
            # vì stats_collector sắp bị reset() ngay sau đây (chỉ chứa đúng chu kỳ này).
            print(f"\n=== [step={state.global_step}] Chu kỳ vừa xong ({n_records} record) ===")
            stats_collector.print_summary()
            print(f"action_buffs sau update: {action_buffs.snapshot()}  hold_buff: {hold_buff.get()}\n")

            stats_collector.save(stats_path, buffs=action_buffs, hold=hold_buff)
            logger.info(
                f"[rank={rank}] Đã lưu {n_records} record -> {stats_path}; "
                f"action_buffs = {action_buffs.snapshot()}; hold_buff = {hold_buff.get()}"
            )
            stats_collector.reset()

        def on_train_end(self, args, state, control, **kwargs):
            update_buffs_from_stats(stats_collector, round_config, action_buffs, hold=hold_buff)
            print(f"\n=== [train_end] Chu kỳ cuối cùng ===")
            stats_collector.print_summary()
            print(f"action_buffs cuối: {action_buffs.snapshot()}  hold_buff cuối: {hold_buff.get()}\n")
            stats_collector.save(stats_path, buffs=action_buffs, hold=hold_buff)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=unified_reward_func,
        args=training_args,
        train_dataset=raw,
        processing_class=tok,   # TRL >=0.14: "tokenizer=" đã đổi tên thành "processing_class="
        callbacks=[StatsPersistCallback()],
    )

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    trainer.save_model()
    # QUAN TRỌNG: lưu tokenizer CANONICAL (load lại fresh, KHÔNG dùng `tok` đã
    # bị mutate add_eos_token/add_bos_token ở trên) — artifact cuối cùng push
    # lên Hub phải khớp đúng tokenizer chuẩn (docs/tokenizer_v0.1.md: tokenizer
    # là artifact bất biến), không mang theo quirk chỉ cần cho lúc rollout nội
    # bộ GRPOTrainer. Các checkpoint TRUNG GIAN (save_steps) vẫn mang tokenizer
    # đã mutate — chấp nhận được vì pipeline không tự đọc lại từ đó (xem giải
    # thích ở chỗ set add_eos_token phía trên).
    canonical_tok = load_tokenizer(repo_id=args.repo_id, allow_local_fallback=False)
    canonical_tok.save_pretrained(args.output_dir)
    update_buffs_from_stats(stats_collector, round_config, action_buffs, hold=hold_buff)
    stats_collector.save(stats_path, buffs=action_buffs, hold=hold_buff)   # chu kỳ cuối cùng chưa kịp chạm save_steps
    
    if push_to_hub:
        trainer.push_to_hub(commit_message=f"GRPO {args.round_id} checkpoint")
        logger.info(f"Đã push lên: https://huggingface.co/{args.repo_id}")

    if trainer.is_world_process_zero():
        print(f"\n=== Report round {args.round_id} (rank {rank} — chạy lại với --report_only để gộp mọi rank) ===")
        stats_collector.print_summary()
        print(f"\naction_buffs cuối round: {action_buffs.snapshot()}  hold_buff cuối round: {hold_buff.get()}")


if __name__ == "__main__":
    main()