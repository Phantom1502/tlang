#!/usr/bin/env bash
set -euo pipefail

# Chạy 1 lần trước, không nằm trong script:
#   huggingface-cli login   # hoặc: export HF_TOKEN=hf_xxx

python -m app.train.train_pretrain \
    --model_size base \
    --dataset_name "sullivan1502/tlang-pretrain-ids" \
    --dataset_mode pre_tokenized \
    \
    --output_dir "./output/base_pretrain" \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 32 \
    --learning_rate 3e-4 \
    --warmup_ratio 0.03 \
    --max_steps 7000 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    \
    --save_steps 100 \
    --save_total_limit 2 \
    --repo_id "sullivan1502/base-pretrain" \
    \
    --fp16 \