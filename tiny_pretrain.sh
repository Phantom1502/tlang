#!/usr/bin/env bash
set -euo pipefail

# Chạy 1 lần trước, không nằm trong script:
#   huggingface-cli login   # hoặc: export HF_TOKEN=hf_xxx

python -m app.train.train_pretrain \
    --org "sullivan1502" \
    --model_size tiny \
    --dataset_name "sullivan1502/tlang-pretrain" \
    --output_dir "./output/tiny_pretrain" \
    \
    --dataset_mode pre_tokenized \
    --max_length 512 \
    \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --warmup_ratio 0.05 \
    --max_steps 200 \
    --logging_steps 10 \
    \
    --save_steps 100 \
    --save_total_limit 2 \
    \
    --fp16 \
    --no_push_to_hub