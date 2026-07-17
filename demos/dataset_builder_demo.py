"""
Demo cho dataset_builder — chạy: python -m demos.dataset_builder_demo

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
from app.lang.parser import Parser


def make_synthetic_ohlc_csv(path: str, n_rows: int = 400, seed: int = 7) -> None:
    rng = random.Random(seed)
    price = 1900.0
    rows = []
    for _ in range(n_rows):
        o = price
        price += rng.uniform(-1.5, 1.5)
        c = price
        h = max(o, c) + rng.uniform(0, 1.0)
        l = min(o, c) - rng.uniform(0, 1.0)
        rows.append({"Open": o, "High": h, "Low": l, "Close": c})
        price = c

    df = pd.DataFrame(rows)
    # ATR_100 giả lập đơn giản (không cần đúng công thức thật — chỉ cần
    # dương, hợp lý, để ChartCodec.quantize_price chạy được; công thức
    # thật đã có sẵn ở app/preprocess/preprocess.py, không thuộc phạm vi
    # test module này).
    tr = (df["High"] - df["Low"]).rolling(14, min_periods=1).mean()
    df["ATR_100"] = tr.bfill()
    df.to_csv(path, index=False)


def run() -> None:
    csv_path = "./tmp/synthetic_xauusd_m15.csv"
    make_synthetic_ohlc_csv(csv_path, n_rows=400)

    # Minh hoạ scale_factor.txt do Preprocess.preprocess() ghi ra ở bước 1
    # (tách riêng, độc lập) — load_scale_factors() đọc lại thay vì gõ tay số.
    scale_factor_path = "./tmp/scale_factor.txt"
    with open(scale_factor_path, "w", encoding="utf-8") as f:
        f.write("synthetic_xauusd_m15: 24.74\n")
    scales = load_scale_factors(scale_factor_path)
    xauusd_scale = scales["synthetic_xauusd_m15"]
    print(f"Đọc scale từ scale_factor.txt: {xauusd_scale}\n")

    # ------------------------------------------------------------
    # 1) GRPO dataset — có augment (n_augments=2)
    # ------------------------------------------------------------
    print("=== 1) Build GRPO parquet (kèm augment) ===")
    grpo_df = build_grpo_parquet(
        csv_paths_with_scale=[(csv_path, "XAUUSD_M15", xauusd_scale)],
        output_path="./tmp/grpo_dataset.parquet",
        n_augments=2,
        seed=42,
    )
    print(grpo_df[["symbol", "window_id"]].head(10))

    assert set(grpo_df.columns) == {"prompt", "future_bins", "symbol", "window_id"}
    assert grpo_df["window_id"].is_unique, "window_id phải duy nhất (dùng để dedup/chống leakage)"

    for _, row in grpo_df.iterrows():
        assert len(row["future_bins"]) == 50, "future_bins phải đủ 50 nến"
        for o, h, l, c in row["future_bins"]:
            assert all(0 <= v <= 1023 for v in (o, h, l, c)), "future_bins phải nằm trong [0,1023]"
        assert row["prompt"].startswith("<chart>") and row["prompt"].endswith("</chart>")

    n_base = grpo_df["window_id"].apply(lambda w: "_aug" not in w).sum()
    n_aug = grpo_df["window_id"].apply(lambda w: "_aug" in w).sum()
    print(f"  window thật: {n_base}, window augment: {n_aug}")
    assert n_aug > 0, "Augment phải sinh thêm ít nhất vài window (n_augments=2 > 0)"

    # Đọc lại từ parquet (round-trip thật, không chỉ dùng DataFrame trong bộ nhớ)
    reloaded = pd.read_parquet("./tmp/grpo_dataset.parquet")
    assert len(reloaded) == len(grpo_df)
    assert list(reloaded.iloc[0]["future_bins"][0]) == list(grpo_df.iloc[0]["future_bins"][0])
    print("  -> Round-trip parquet OK.\n")

    # ------------------------------------------------------------
    # 2) Pretrain/SFT dataset — verify MỌI completion đều well-formed
    # ------------------------------------------------------------
    print("=== 2) Build pretrain/SFT parquet (double-check well-formed) ===")
    sft_df = build_pretrain_sft_parquet(
        csv_paths_with_scale=[(csv_path, "XAUUSD_M15", xauusd_scale)],
        output_path="./tmp/sft_dataset.parquet",
        samples_per_chart=5,
        n_augments=1,
        seed=42,
    )
    print(f"  Tổng {len(sft_df)} row pretrain/SFT.")

    assert set(sft_df.columns) == {"prompt", "completion"}
    fail = 0
    for _, row in sft_df.iterrows():
        full_text = row["prompt"] + " " + row["completion"]
        result = Parser.from_text(full_text).parse()
        if not result.is_well_formed():
            fail += 1
    print(f"  well-formed: {len(sft_df) - fail}/{len(sft_df)}")
    assert fail == 0, "Có completion KHÔNG well-formed — lỗi ở generator hoặc bridge format!"

    print("\nTất cả assertion PASS.")


if __name__ == "__main__":
    run()