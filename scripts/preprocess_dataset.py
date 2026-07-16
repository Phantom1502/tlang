from app.data_prepare.preprocess import Preprocess
import glob

def preprocess_data(csv_folder, output_folder, period=100):
    # duyệt toàn bộ file
    csv_paths = glob.glob(csv_folder)
    for csv_path in csv_paths:
        print(f"Processing {csv_path}")
        Preprocess.preprocess(csv_path, output_folder, period=period)

if __name__ == "__main__":
    atr_period = 100
    train_raw_folder = "data/raw/train/*.csv"
    train_output_folder = "data/preprocessed/train/"
    preprocess_data(train_raw_folder, train_output_folder, period=atr_period)
    
    val_raw_folder = "data/raw/val/*.csv"
    val_output_folder = "data/preprocessed/val/"
    preprocess_data(val_raw_folder, val_output_folder, period=atr_period)