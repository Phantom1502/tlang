#!/usr/bin/env bash
set -euo pipefail

# Chạy 1 lần trước, không nằm trong script:
#   huggingface-cli login   # hoặc: export HF_TOKEN=hf_xxx

python -m app.train.train_pretrain \
    --model_size tiny \
    --dataset_name "sullivan1502/tlang-pretrain-ids" \
    --dataset_mode pre_tokenized \
    \
    --output_dir "./output/tiny_pretrain" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --warmup_ratio 0.03 \
    --max_steps 200 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    \
    --save_steps 100 \
    --save_total_limit 2 \
    \
    --fp16 \