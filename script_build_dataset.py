"""
Demo cho dataset_builder — chạy: python -m app.gen.dataset_builder_demo

Tạo 1 CSV OHLC giả lập (đã có cột ATR_100, mô phỏng output của
app/preprocess/preprocess.py) rồi chạy full pipeline -> parquet, kiểm
chứng:
1. GRPO parquet: đúng schema, future_bins đủ 50 nến, augment sinh thêm
   window_id mới với bin dịch chuyển hợp lệ.
2. Pretrain/SFT parquet: mọi completion sinh ra đều parse well-formed
   (double-check end-to-end, không chỉ tin generator).
"""
import random

import numpy as np
import pandas as pd

from app.gen.dataset_builder import build_grpo_parquet, build_pretrain_sft_parquet, load_scale_factors
import os

def build_dataset(input_csv: str, output_folder: str):
    base_name = os.path.basename(input_csv)  # Trả về: "AUDUSD_15Min.csv"
    filename = os.path.splitext(base_name)[0]  # Trả về: "AUDUSD_15Min"

    scales = load_scale_factors("data/preprocessed/train/scale_factor.txt")
    scale = scales[filename]
    
    # 1. GRPO dataset — có augment
    print("=== 1) Build GRPO parquet (kèm augment) ===")
    print(f"Input: {input_csv}, scale: {scale}")
    
    output_path = f"{output_folder}/{filename}_grpo_dataset.parquet"
    grpo_df = build_grpo_parquet(
        csv_paths_with_scale=[(input_csv, filename, scale)],
        output_path=output_path,
        n_augments=10,
        seed=42,
    )
    
    # 2. Pretrain/SFT dataset
    print("=== 2) Build Pretrain/SFT parquet ===")
    print(f"Input: {input_csv}, scale: {scale}")
    
    output_path = f"{output_folder}/{filename}_pretrain_sft_dataset.parquet"
    sft_df = build_pretrain_sft_parquet(
        csv_paths_with_scale=[(input_csv, filename, scale)],
        output_path="/tmp/sft_dataset.parquet",
        samples_per_chart=5,
        n_augments=10,
        seed=42,
    )

    print(f"Build completed for {input_csv}")

if __name__ == "__main__":
    build_dataset(
        input_csv = "data/preprocessed/train/AUDUSD_15Min_preprocessed.csv",
        output_folder="data/dataset/"    
    )