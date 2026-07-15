import sys
import pandas as pd
from datasets import load_dataset
from app.tokenizer.hub import load_tokenizer

tok = load_tokenizer()
dataset = load_dataset("sullivan1502/tlang-pretrain", split="val")

# load 1 record, encode, decode, print
record = dataset[0]
#print(record["prompt"])
ids = tok.encode(record["completion"])
print(ids)
decoded = tok.decode(ids, skip_special_tokens=True)
print(decoded)