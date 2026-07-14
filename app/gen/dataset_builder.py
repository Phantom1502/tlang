"""
app/gen/dataset_builder.py — Pipeline raw OHLC (đã preprocess, có cột ATR_100)
-> parquet dataset hoàn thiện, cho cả GRPO (schema mục 7.3) và pretrain/SFT
(schema mục 7.2).

KHÔNG import lại app/preprocess/candle.py, app/gen/base_gen.py,
app/gen/rawchart_gen.py, app/utils/parquet_writer.py — đây là source cũ
từ dự án trước (học 100 nến, input 50 nến dự đoán giá), chỉ MƯỢN Ý TƯỞNG
2 chỗ:
  1. Cách windows hoá + encode bin qua ChartCodec (module thật, dùng lại
     trực tiếp — không phải "source cũ" theo nghĩa cần viết lại).
  2. Data augmentation kiểu "shift đều toàn bộ window theo 1 delta bin
     ngẫu nhiên" (ý tưởng từ RawChartGenerator cũ) — viết lại sạch ở đây
     (`augment_shift`), không phụ thuộc CandleParser cũ.

Cầu nối quan trọng nhất trong module này: `ChartCodec.encode_window` xuất
text KHÔNG có dấu ngoặc quanh từng field ("O_434 H_543..."), nhưng grammar
hiện tại (app/lang/lexer.py) cần dạng atomic CÓ ngoặc ("<O_434>"). Hàm
`parse_window_text` + `render_chart_block` là 2 nửa của bước convert này.
"""
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from app.gen.generator import generate_dataset
from app.preprocess.chartcodec import ChartCodec

Candle = Tuple[int, int, int, int]   # (o, h, l, c)

WINDOW_SIZE = 100     # 100 nến/window: 50 đầu = input, 50 sau = future (đã chốt spec mục 7.1)
INPUT_CANDLES = 50
N_BINS = 1024

# Regex tự viết mới (KHÔNG import candle.py cũ) — chỉ cần khớp đúng output
# thô của ChartCodec.encode_window: "<chart> O_434 H_543 L_543 C_543 ... </chart>"
_TOKEN_RE = re.compile(r"([OHLC])_(\d+)")


# =====================================================================
# Đọc scale_factor.txt do Preprocess.preprocess() ghi ra (bước 1, tách
# riêng độc lập trước dataset_builder) — tránh phải gõ tay/copy nhầm số
# FINAL_SCALE_FACTOR vào csv_paths_with_scale.
#
# Format mỗi dòng trong file (xem app/preprocess/preprocess.py):
#     "{filename}: {FINAL_SCALE_FACTOR:.2f}\n"
# vd: "XAUUSD_15Min: 24.74"
# =====================================================================
def load_scale_factors(scale_factor_path: str) -> Dict[str, float]:
    """Trả về dict {filename (không đuôi .csv) -> scale}. Nếu 1 filename
    xuất hiện nhiều lần trong file (chạy Preprocess.preprocess() nhiều
    lần, mode append 'a'), LẤY DÒNG CUỐI CÙNG (mới nhất)."""
    scales: Dict[str, float] = {}
    with open(scale_factor_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, value = line.split(":", 1)
            scales[name.strip()] = float(value.strip())
    return scales


# =====================================================================
# Convert text thô của ChartCodec (không ngoặc) <-> list Candle (int)
# =====================================================================
def parse_window_text(text: str) -> List[Candle]:
    """'<chart> O_434 H_543 L_543 C_543 O_... </chart>' -> [(o,h,l,c), ...].
    Bỏ qua an toàn 2 tag "<chart>"/"</chart>" vì regex chỉ khớp phần O/H/L/C."""
    buckets: Dict[str, List[int]] = {"O": [], "H": [], "L": [], "C": []}
    for letter, num in _TOKEN_RE.findall(text):
        buckets[letter].append(int(num))

    n = len(buckets["O"])
    if not all(len(buckets[k]) == n for k in "HLC"):
        raise ValueError(
            f"Số token O/H/L/C không khớp nhau: "
            f"O={len(buckets['O'])} H={len(buckets['H'])} L={len(buckets['L'])} C={len(buckets['C'])}"
        )
    return list(zip(buckets["O"], buckets["H"], buckets["L"], buckets["C"]))


def render_chart_block(candles: Sequence[Candle]) -> str:
    """[(o,h,l,c), ...] -> '<chart> <O_x> <H_x> <L_x> <C_x> ... </chart>'
    ĐÚNG format atomic hiện tại của grammar (app/lang/lexer.py CANDLE_O/H/L/C:
    r"<O_\\d+>" ...) — khác hẳn format thô không ngoặc của ChartCodec."""
    parts = ["<chart>"]
    for o, h, l, c in candles:
        parts.extend([f"<O_{o}>", f"<H_{h}>", f"<L_{l}>", f"<C_{c}>"])
    parts.append("</chart>")
    return " ".join(parts)


# =====================================================================
# Data augmentation — dịch chuyển ĐỀU (uniform bin-shift) toàn bộ candle
# trong 1 nhóm theo cùng 1 delta ngẫu nhiên, giữ trong biên [0, N_BINS-1].
#
# Ý tưởng mượn từ RawChartGenerator cũ (min()/max() để tính biên độ dịch
# an toàn), viết lại độc lập ở đây. Mục đích: 1 window thật chỉ chiếm 1
# vị trí bin tuyệt đối cố định — shift đều tạo thêm biến thể KHÔNG đổi
# hình dạng tương đối (trend/zone/khoảng cách giữa các nến giữ nguyên)
# nhưng đổi toạ độ tuyệt đối, giúp model không ghi nhớ 1 vùng bin cụ thể
# mà học đúng quan hệ tương đối (điều thực sự cần cho suy luận trading).
# =====================================================================
def augment_shift(
    candles: Sequence[Candle],
    rng: random.Random,
    n_bins: int = N_BINS,
) -> Optional[List[Candle]]:
    """Trả None nếu window đã chiếm hết biên độ bin (không còn chỗ dịch)."""
    lows = [c[2] for c in candles]
    highs = [c[1] for c in candles]
    min_low, max_high = min(lows), max(highs)

    shift_min = -min_low
    shift_max = (n_bins - 1) - max_high
    if shift_min > shift_max:
        return None

    choices = [d for d in range(shift_min, shift_max + 1) if d != 0]
    if not choices:
        return None

    delta = rng.choice(choices)
    return [(o + delta, h + delta, l + delta, c + delta) for o, h, l, c in candles]


# =====================================================================
# GRPO dataset — schema spec mục 7.3: {"prompt","future_bins","symbol","window_id"}
# =====================================================================
def build_grpo_rows(
    encoded_df: pd.DataFrame,
    symbol: str,
    n_augments: int = 0,
    seed: Optional[int] = None,
) -> List[dict]:
    """
    encoded_df: output của `ChartCodec.encode_df(df, window_size=100, stride=50)`
    — mỗi row có cột "text" (100 nến thô, dạng "<chart> O_x H_x ... </chart>").

    n_augments: số biến thể shift-đều thêm MỖI window thật (0 = tắt augment).
    Augment áp dụng trên CẢ 100 nến cùng lúc (input + future cùng 1 delta)
    để giữ nhất quán nội bộ 1 window — không lệch giữa 2 phần.
    """
    rng = random.Random(seed)
    rows: List[dict] = []

    for idx, row in encoded_df.iterrows():
        candles_100 = parse_window_text(row["text"])
        if len(candles_100) != WINDOW_SIZE:
            continue  # window thiếu/lỗi nến — bỏ qua, không nên xảy ra nếu ChartCodec chạy đúng

        variants: List[Tuple[str, List[Candle]]] = [(f"{symbol}_{idx}", candles_100)]
        for k in range(n_augments):
            shifted = augment_shift(candles_100, rng)
            if shifted is not None:
                variants.append((f"{symbol}_{idx}_aug{k}", shifted))

        for window_id, candles in variants:
            input_candles = candles[:INPUT_CANDLES]
            future_candles = candles[INPUT_CANDLES:]
            rows.append({
                "prompt": render_chart_block(input_candles),
                "future_bins": [list(c) for c in future_candles],
                "symbol": symbol,
                "window_id": window_id,
            })

    return rows


def build_grpo_parquet(
    csv_paths_with_scale: Sequence[Tuple[str, str, float]],
    output_path: str,
    window_size: int = WINDOW_SIZE,
    stride: int = INPUT_CANDLES,
    n_augments: int = 0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Pipeline đầy đủ: CSV đã preprocess (có cột ATR_100 — xem
    `app/preprocess/preprocess.py:Preprocess.preprocess`) -> ChartCodec ->
    parse lại thành candles -> augment (tuỳ chọn) -> ghi ra 1 file parquet
    đúng schema GRPO.

    csv_paths_with_scale: mỗi phần tử (đường dẫn CSV, tên symbol, scale
    constant — tra ở app/preprocess/chartcodec.py, vd XAUUSD_M15_SCALE,
    hoặc đọc trực tiếp từ scale_factor.txt qua `load_scale_factors()`
    thay vì gõ tay).
    """
    all_rows: List[dict] = []

    for csv_path, symbol, scale in csv_paths_with_scale:
        df = pd.read_csv(csv_path)
        codec = ChartCodec(scale=scale)
        encoded_df = codec.encode_df(df, window_size=window_size, stride=stride)
        rows = build_grpo_rows(encoded_df, symbol=symbol, n_augments=n_augments, seed=seed)
        all_rows.extend(rows)
        print(f"[{symbol}] {len(encoded_df)} window thật -> {len(rows)} row GRPO (kể cả augment)")

    out_df = pd.DataFrame(all_rows)
    out_df.to_parquet(output_path, index=False)
    print(f"Đã ghi {len(out_df)} row vào {output_path}")
    return out_df


# =====================================================================
# Pretrain/SFT dataset — schema spec mục 7.2: {"prompt","completion"}
# Chỉ cần 50 nến INPUT (không cần future) -> ghép với app/gen/generator.py
# =====================================================================
def build_pretrain_sft_rows(
    encoded_df: pd.DataFrame,
    symbol: str,
    samples_per_chart: int = 4,
    n_augments: int = 0,
    seed: Optional[int] = None,
) -> List[dict]:
    rng = random.Random(seed)
    charts: List[List[Candle]] = []

    for idx, row in encoded_df.iterrows():
        candles_100 = parse_window_text(row["text"])
        if len(candles_100) != WINDOW_SIZE:
            continue
        input_candles = candles_100[:INPUT_CANDLES]
        charts.append(input_candles)

        # Augment chỉ trên 50 nến input (không cần biết future ở đây) —
        # biên độ dịch tính riêng trên input, KHÔNG bị future giới hạn.
        for _ in range(n_augments):
            shifted = augment_shift(input_candles, rng)
            if shifted is not None:
                charts.append(shifted)

    samples = generate_dataset(charts, samples_per_chart=samples_per_chart, seed=seed)
    return [{"prompt": s.prompt, "completion": s.completion} for s in samples]


def build_pretrain_sft_parquet(
    csv_paths_with_scale: Sequence[Tuple[str, str, float]],
    output_path: str,
    samples_per_chart: int = 4,
    window_size: int = WINDOW_SIZE,
    stride: int = INPUT_CANDLES,
    n_augments: int = 0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    all_rows: List[dict] = []

    for csv_path, symbol, scale in csv_paths_with_scale:
        df = pd.read_csv(csv_path)
        codec = ChartCodec(scale=scale)
        encoded_df = codec.encode_df(df, window_size=window_size, stride=stride)
        rows = build_pretrain_sft_rows(
            encoded_df, symbol=symbol, samples_per_chart=samples_per_chart,
            n_augments=n_augments, seed=seed,
        )
        all_rows.extend(rows)
        print(f"[{symbol}] {len(encoded_df)} window thật -> {len(rows)} row pretrain/SFT")

    out_df = pd.DataFrame(all_rows)
    out_df.to_parquet(output_path, index=False)
    print(f"Đã ghi {len(out_df)} row vào {output_path}")
    return out_df