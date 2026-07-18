import pandas as pd

val_file = "data/dataset/grpo/val.parquet"

df = pd.read_parquet(val_file)
print(df.shape)