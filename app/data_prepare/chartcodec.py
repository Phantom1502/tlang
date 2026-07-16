import re
import numpy as np
import pandas as pd
import os

N_BINS = 1024
_TOKEN_RE = re.compile(r"([OHLC])_(\d+)")

# Các loại scale factor có thể dùng để chuẩn hoá giá trong cửa sổ OHLC
XAUUSD_M1_SCALE     = 23.95         # Window Range = M1_SCALE * ATR
XAUUSD_M5_SCALE     = 27.38         # Window Range = M5_SCALE * ATR
XAUUSD_M15_SCALE    = 24.74         # Window Range = M15_SCALE * ATR
XAUUSD_H1_SCALE     = 22.81         # Window Range = H1_SCALE * ATR
XAUUSD_D1_SCALE     = 17.22         # Window Range = D1_SCALE * ATR
AUDUSD_M1_SCALE     = 27.19         # Window Range = M1_SCALE * ATR
AUDUSD_M5_SCALE     = 27.24      # Window Range = M5_SCALE * ATR
AUDUSD_M15_SCALE    = 23.73     # Window Range = M15_SCALE * ATR
AUDUSD_H1_SCALE     = 20.18     # Window Range = H1_SCALE * ATR
EURUSD_M1_SCALE     = 26.83 
EURUSD_M5_SCALE     = 28.12 
EURUSD_M15_SCALE    = 25.50 
EURUSD_H1_SCALE     = 22.96 
GBPUSD_M1_SCALE     = 26.89 
GBPUSD_M5_SCALE     = 26.50 
GBPUSD_M15_SCALE    = 25.32 
GBPUSD_H1_SCALE     = 22.15 
US500_M1_SCALE      = 30.20 
US500_M5_SCALE      = 32.12 
US500_M15_SCALE     = 27.86 
US500_H1_SCALE      = 26.47 # Not set yet

class ChartCodec:
    def __init__(self, scale: float, n_bins: int = N_BINS):
        self.scale = scale
        self.n_bins = n_bins
        
    def quantize_price(self, price, anchor_open, anchor_atr) -> int:
        if anchor_atr <= 0 or np.isnan(anchor_atr):
            raise ValueError("anchor_atr phải > 0")
        
        norm = (price - anchor_open) / (self.scale * anchor_atr)
        norm = np.clip(norm, -1.0, 1.0)
        bin_idx = int(round((norm + 1.0) / 2.0 * (self.n_bins - 1)))
        return bin_idx
    
    def dequantize_bin(self, bin_idx, anchor_open, anchor_atr) -> float:
        norm = (bin_idx / (self.n_bins - 1)) * 2.0 - 1.0
        price = anchor_open + norm * self.scale * anchor_atr
        return price
    
    def encode_window(self, window_df: pd.DataFrame, anchor_open, anchor_atr) -> str:
        parts = ["<chart>"]
        for _, row in window_df.iterrows():
            o = self.quantize_price(row['Open'], anchor_open, anchor_atr)
            h = self.quantize_price(row['High'], anchor_open, anchor_atr)
            l = self.quantize_price(row['Low'], anchor_open, anchor_atr)
            c = self.quantize_price(row['Close'], anchor_open, anchor_atr)
            parts.extend([f"O_{o}", f"H_{h}", f"L_{l}", f"C_{c}"])
        parts.append("</chart>")
        return " ".join(parts)
    
    def decode_window(self, text: str, anchor_open, anchor_atr) -> str:
        buckets = {"O": [], "H": [], "L": [], "C": []}
        for letter, num in _TOKEN_RE.findall(text):
            buckets[letter].append(int(num))
        
        n_candles = len(buckets["O"])
        if not all(len(buckets[k]) == n_candles for k in "HLC"):
            raise ValueError(
                f"Số token O/H/L/C không khớp nhau: "
                f"O={len(buckets['O'])} H={len(buckets['H'])} "
                f"L={len(buckets['L'])} C={len(buckets['C'])} "
                f"— text có thể bị model sinh lỗi/thiếu token."
            )
        
        rows = []
        for i in range(n_candles):
            rows.append({
                "Open":  self.dequantize_bin(buckets["O"][i], anchor_open, anchor_atr),
                "High":  self.dequantize_bin(buckets["H"][i], anchor_open, anchor_atr),
                "Low":   self.dequantize_bin(buckets["L"][i], anchor_open, anchor_atr),
                "Close": self.dequantize_bin(buckets["C"][i], anchor_open, anchor_atr),
            })
        
        return " ".join([f"O_{row['Open']} H_{row['High']} L_{row['Low']} C_{row['Close']}" for row in rows])
    
    def encode_df(self, df: pd.DataFrame, window_size: int, stride: int = 50) -> pd.DataFrame:
        df = df.reset_index(drop=True).copy()
        
        records = []
        last_start = len(df) - window_size
        # Duyệt qua các cửa sổ
        for i in range(0, last_start + 1, stride):
            anchor_open = df.loc[i, "Open"]
            anchor_atr = df.loc[i, "ATR_100"]
            if anchor_atr <= 0 or np.isnan(anchor_atr):
                continue
            
            window = df.iloc[i:i + window_size]
            text = self.encode_window(window, anchor_open, anchor_atr)
            records.append({
                "anchor_open": anchor_open,
                "anchor_atr": anchor_atr,
                "text": text,
            })

        return pd.DataFrame(records)

def encode_df(csv_path: str, output_path: str, window_size: int, scale: float, stride=50):
    codec = ChartCodec(scale=scale, n_bins=N_BINS)
    df = pd.read_csv(csv_path)
    encoded_df = codec.encode_df(df, window_size=window_size, stride=stride)
    
    filename = csv_path.split('/')[-1].split('.')[0]
    output_file = f"{output_path}/{filename}_encoded.csv"
    
    # Đảm bảo thư mục output tồn tại
    os.makedirs(output_path, exist_ok=True)
    encoded_df.to_csv(output_file, index=False)
    
if __name__ == "__main__":
    # Test the ChartCodec class
    output_path = "data/encoded"
    window_size = 100
    stride = window_size // 20
    
    inputs_datas = [
        {"csv_path": "data/preprocessed/AUDUSD_1Min_preprocessed.csv", "scale": AUDUSD_M1_SCALE},
        {"csv_path": "data/preprocessed/AUDUSD_5Min_preprocessed.csv", "scale": AUDUSD_M5_SCALE},
        {"csv_path": "data/preprocessed/AUDUSD_15Min_preprocessed.csv", "scale": AUDUSD_M15_SCALE},
        {"csv_path": "data/preprocessed/AUDUSD_H1_preprocessed.csv", "scale": AUDUSD_H1_SCALE},
        {"csv_path": "data/preprocessed/EURUSD_1Min_preprocessed.csv", "scale": EURUSD_M1_SCALE},
        {"csv_path": "data/preprocessed/EURUSD_5Min_preprocessed.csv", "scale": EURUSD_M5_SCALE},
        {"csv_path": "data/preprocessed/EURUSD_15Min_preprocessed.csv", "scale": EURUSD_M15_SCALE},
        {"csv_path": "data/preprocessed/EURUSD_H1_preprocessed.csv", "scale": EURUSD_H1_SCALE},
        {"csv_path": "data/preprocessed/GBPUSD_1Min_preprocessed.csv", "scale": GBPUSD_M1_SCALE},
        {"csv_path": "data/preprocessed/GBPUSD_5Min_preprocessed.csv", "scale": GBPUSD_M5_SCALE},
        {"csv_path": "data/preprocessed/GBPUSD_15Min_preprocessed.csv", "scale": GBPUSD_M15_SCALE},
        {"csv_path": "data/preprocessed/GBPUSD_H1_preprocessed.csv", "scale": GBPUSD_H1_SCALE},
        {"csv_path": "data/preprocessed/US500_1Min_preprocessed.csv", "scale": US500_M1_SCALE},
        {"csv_path": "data/preprocessed/US500_5Min_preprocessed.csv", "scale": US500_M5_SCALE},
        {"csv_path": "data/preprocessed/US500_15Min_preprocessed.csv", "scale": US500_M15_SCALE},
        {"csv_path": "data/preprocessed/US500_H1_preprocessed.csv", "scale": US500_H1_SCALE},
        {"csv_path": "data/preprocessed/XAUUSD_1Min_preprocessed.csv", "scale": XAUUSD_M1_SCALE},
        {"csv_path": "data/preprocessed/XAUUSD_5Min_preprocessed.csv", "scale": XAUUSD_M5_SCALE},
        {"csv_path": "data/preprocessed/XAUUSD_15Min_preprocessed.csv", "scale": XAUUSD_M15_SCALE},
        {"csv_path": "data/preprocessed/XAUUSD_H1_preprocessed.csv", "scale": XAUUSD_H1_SCALE},
    ]
    
    for input_data in inputs_datas:
        csv_path = input_data["csv_path"]
        scale = input_data["scale"]
        print(f"Processing {csv_path} with scale {scale}")
        encode_df(csv_path, output_path, window_size, scale, stride)