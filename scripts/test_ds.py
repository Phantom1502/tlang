import sys
import pandas as pd
from datasets import load_dataset
from app.tokenizer.hub import load_tokenizer

tok = load_tokenizer()
dataset = load_dataset("sullivan1502/tlang-pretrain-ids", split="val")

# In thông tin tổng quan của dataset (nó sẽ tự nhận diện cả 2 split train và val)
print(dataset)

# Thử truy cập nhanh vào tập validation
print("\nSố lượng mẫu trong tập validation:", len(dataset["val"]))

# load 1 record, encode, decode, print
record = dataset[0]
#print(record["prompt"])
ids = tok.encode(record["completion"])
print(ids)
decoded = tok.decode(ids, skip_special_tokens=True)
print(decoded)