import pandas as pd

val_file = r"data\dataset\XAUUSD_M1_Val_grpo_dataset.parquet"

df = pd.read_parquet(val_file)
print(df.shape)
print(df.columns)

from datasets import load_dataset
ds = load_dataset("parquet", data_files="data/dataset/XAUUSD_M1_Val_grpo_dataset.parquet", split="train")
print(ds[6]['prompt'])
