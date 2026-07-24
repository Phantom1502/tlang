#!/usr/bin/env bash
set -euo pipefail

# Chạy 1 lần trước, không nằm trong script:
#   huggingface-cli login   # hoặc: export HF_TOKEN=hf_xxx
#
# Trước khi chạy, model repo đích (--repo_id) PHẢI đã có tokenizer add sẵn
# (quy ước quản lý — xem app/tokenizer/hub.py), giống hệt convention của
# train_pretrain.py/train_sft.py.
#
# round1: init từ checkpoint SFT (--init_from_repo), chưa có checkpoint GRPO
# nào của round này -> script tự nhận ra và init từ SFT (xem
# build_model_for_round trong train_grpo.py). Round 2 trở đi: đổi
# --init_from_repo trỏ sang checkpoint round liền trước, --repo_id/--round_id
# đổi sang round2, và SỬA TAY rounds/round2.json theo thống kê đọc được từ
# --report_only của round1 (mục 5.3 spec — không tự động hoá việc chuyển round).

python -m app.training.train_grpo \
    --model_size base \
    --round_id round2 \
    --repo_id "sullivan1502/base-grpo-round2" \
    --init_from_repo "sullivan1502/base-grpo-round1" \
    --round_config "./rounds/round2.json" \
    \
    --dataset_name "sullivan1502/tlang-grpo" \
    --train_split train \
    \
    --output_dir "./output/base_grpo_round2" \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-6 \
    --warmup_ratio 0.02 \
    --max_steps 50000 \
    --logging_steps 10 \
    --max_completion_length 40 \
    \
    --temperature 1.1 \
    --top_p 1.0 \
    --top_k 0 \
    \
    --num_generations 16 \
    \
    --save_steps 100 \
    --save_total_limit 2 \
    --hf_token "$HF_TOKEN" \
    \
    --fp16

# Xem report giữa chừng bất cứ lúc nào (không cần đợi train xong, không cần GPU):
#   python -m app.training.train_grpo --round_id round1 --output_dir ./output/tiny_grpo_round1 --report_only