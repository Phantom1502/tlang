import pandas as pd
import numpy as np
import os

class Preprocess:
    @staticmethod
    def calculate_atr(
        df: pd.DataFrame, 
        period=100
    ):
        """Tính chỉ báo ATR 100 chuẩn kỹ thuật"""
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift(1))
        low_cp = np.abs(df['Low'] - df['Close'].shift(1))
        
        tr = np.max(np.vstack((high_low, high_cp, low_cp)), axis=0)
        # Dùng trung bình động lũy thừa (EMA) để tính ATR cho mượt
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        return atr
    
    @staticmethod
    def preprocess(
        csv_path: str, 
        output_path: str,
        period=100
    ):
        """Tính chỉ báo ATR 100 chuẩn kỹ thuật"""
        df = pd.read_csv(csv_path)
        
        # 1. Tính ATR 100 cho toàn bộ tập dữ liệu
        df['ATR_100'] = Preprocess.calculate_atr(df, period)
        df = df.dropna().reset_index(drop=True)
        
        all_max_ratios = []
        # 2. Quét cửa sổ trượt cuốn chiếu toàn bộ lịch sử
        for i in range(1, len(df)):
            # Trích xuất 100 nến quá khứ tính đến thời điểm i
            window = df.iloc[max(0, i - period + 1): i + 1]
            
            open_i = df.loc[i, 'Open']    # Giá neo số 0 hiện tại
            atr_i = df.loc[i, 'ATR_100']  # ATR tại thời điểm i làm thước đo
            
            # Tính khoảng cách thô tuyệt đối của tất cả 400 điểm trong vùng so với open_i
            ohlc_raw = window[['Open', 'High', 'Low', 'Close']].values
            max_absolute_distance = np.max(np.abs(ohlc_raw - open_i))
            
            # Tính tỷ lệ: Cửa sổ này dạt xa gấp mấy lần ATR_i?
            ratio = max_absolute_distance / atr_i
            all_max_ratios.append(ratio)
            
        # 3. Tìm hằng số SCALE bao trùm tuyệt đối
        # Lấy bách phân vị 99.9% để loại bỏ nhiễu cực đoan của lỗi dữ liệu nếu có
        FINAL_SCALE_FACTOR = np.percentile(all_max_ratios, 99.9)
        
        # 4. Save kết quả ra file CSV mới
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(output_path, exist_ok=True)
        
        base_name = os.path.basename(csv_path)  # Trả về: "AUDUSD_15Min.csv"
        filename = os.path.splitext(base_name)[0]  # Trả về: "AUDUSD_15Min"
        output_file = os.path.join(output_path, f"{filename}_preprocessed.csv")
        
        df.to_csv(output_file, index=False)
        
        # 5. Open and append the scale factor to a text file
        scale_factor_file = os.path.join(output_path, "scale_factor.txt")
        with open(scale_factor_file, 'a') as f:
            f.write(f"{filename}: {FINAL_SCALE_FACTOR:.2f}\n")
            
if __name__ == "__main__":
    csv_paths = [
        "data/raw/EURUSD_M1_Val.csv", 
        "data/raw/GBPUSD_M1_Val.csv",
        "data/raw/XAUUSD_M1_Val.csv",
    ]
    output_path = "data/preprocessed"
    for csv_path in csv_paths:
        print(f"Processing {csv_path}")
        Preprocess.preprocess(csv_path, output_path, period=100)