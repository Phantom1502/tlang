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

from app.data_prepare.dataset_builder import build_grpo_parquet, build_pretrain_sft_parquet, load_scale_factors
import os
from typing import Optional
import glob

def build_grpo_dataset(input_csv: str, output_folder: str, filename: str, scale: float, stride: Optional[int] = None, n_augments: int = 5, seed: int = 42):
    print(f"Input: {input_csv}, scale: {scale}")
    
    output_path = f"{output_folder}/{filename}_grpo_dataset.parquet"
    build_grpo_parquet(
        csv_paths_with_scale=[(input_csv, filename, scale)],
        output_path=output_path,
        stride=stride if stride is not None else 50,
        n_augments=n_augments,
        seed=seed,
    )
    
def build_pretrain_sft_dataset(input_csv: str, output_folder: str, filename: str, scale: float, samples_per_chart: int = 10, n_augments: int = 10, seed: int = 42):
    print(f"Input: {input_csv}, scale: {scale}")
    
    output_path = f"{output_folder}/{filename}_pretrain_sft_dataset.parquet"
    build_pretrain_sft_parquet(
        csv_paths_with_scale=[(input_csv, filename, scale)],
        output_path=output_path,
        samples_per_chart=samples_per_chart,
        n_augments=n_augments,
        seed=seed,
    )

def build_dataset(input_csv: str, output_folder: str):
    base_name = os.path.basename(input_csv)  # Trả về: "AUDUSD_15Min.csv"
    filename = os.path.splitext(base_name)[0]  # Trả về: "AUDUSD_15Min"

    scales = load_scale_factors("data/preprocessed/train/scale_factor.txt")
    scale = scales[filename]
    
    # 1. GRPO dataset — có augment
    print("=== 1) Build GRPO parquet (kèm augment) ===")
    build_grpo_dataset(input_csv, output_folder, filename, scale, n_augments=5, seed=42)
    
    # 2. Pretrain/SFT dataset
    print("=== 2) Build Pretrain/SFT parquet ===")
    build_pretrain_sft_dataset(input_csv, output_folder, filename, scale, samples_per_chart=10, n_augments=10, seed=42)

    print(f"Build completed for {input_csv}")
    
def build_train_dataset():
    input_csvs = "data/preprocessed/train/*.csv"
    output_folder = "data/dataset/"
    
    csv_paths = glob.glob(input_csvs)
    for csv_path in csv_paths:
        build_dataset(csv_path, output_folder)
        
def build_grpo_val_dataset():
    input_csvs = [
        "data/preprocessed/val/XAUUSD_M1_Val.csv",
        "data/preprocessed/val/GBPUSD_M1_Val.csv",
        "data/preprocessed/val/EURUSD_M1_Val.csv",
    ]
    output_folder = "data/dataset/"
    
    scales = [23.95, 26.89, 26.83]
    
    for csv_path, scale in zip(input_csvs, scales):
        base_name = os.path.basename(csv_path)  # Trả về: "AUDUSD_15Min.csv"
        filename = os.path.splitext(base_name)[0]  # Trả về: "AUDUSD_15Min"
        
        # 1. GRPO dataset — có augment
        print("=== 1) Build GRPO parquet (kèm augment) ===")
        build_grpo_dataset(csv_path, output_folder, filename, scale, stride=1, n_augments=0, seed=42)
        
        print(f"Build completed for {csv_path}")

if __name__ == "__main__":
    build_grpo_val_dataset()