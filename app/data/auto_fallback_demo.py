"""
Demo/test cho dataset_mode="auto" (fallback default->raw THẬT) — chạy:
    python -m app.data.auto_fallback_demo

Vì sandbox không gọi được huggingface.co, mock `datasets.load_dataset` để
giả lập 2 tình huống:
1. Config "default" (ids/, đã tokenize) tồn tại -> dùng thẳng, collator
   phải là DataCollatorForPreTokenizedCoT.
2. Config "default" KHÔNG tồn tại (raise lỗi) -> fallback config "raw",
   collator phải là DataCollatorForCoT, tokenize on-the-fly vẫn đúng.
"""
from unittest.mock import patch

from datasets import Dataset

from app.data.collator import DataCollatorForCoT, DataCollatorForPreTokenizedCoT
from app.data.data_module import DataArguments, make_data_module
from app.gen.generator import generate_dataset
from app.tokenizer.hub import load_tokenizer
import random


def make_synthetic_chart(rng, n=50, base=500, spread=30):
    candles = []
    price = base
    for _ in range(n):
        price += rng.randint(-3, 3)
        price = max(spread, min(1023 - spread, price))
        o = price
        c = price + rng.randint(-5, 5)
        h = max(o, c) + rng.randint(0, 5)
        l = min(o, c) - rng.randint(0, 5)
        candles.append((max(0, o), min(1023, h), max(0, l), max(0, min(1023, c))))
        price = c
    return candles


def run() -> None:
    tok = load_tokenizer()  # fallback local (sandbox không có mạng tới Hub thật)

    rng = random.Random(5)
    charts = [make_synthetic_chart(rng) for _ in range(3)]
    samples = generate_dataset(charts, samples_per_chart=4, seed=1)
    raw_examples = [{"prompt": s.prompt, "completion": s.completion} for s in samples]

    # ------------------------------------------------------------
    # 1) Scenario A: config "default" (ids) tồn tại -> pre_tokenized
    # ------------------------------------------------------------
    print("=== 1) Scenario A: config 'default' (ids/) tồn tại ===")
    pretokenized_rows = []
    for ex in raw_examples:
        prompt_ids = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        full_ids = tok(ex["prompt"] + " " + ex["completion"], add_special_tokens=True)["input_ids"]
        n_mask = 1 + len(prompt_ids)
        labels = [-100] * n_mask + full_ids[n_mask:]
        pretokenized_rows.append({"input_ids": full_ids, "labels": labels})

    default_train_ds = Dataset.from_list(pretokenized_rows)

    def fake_load_dataset_A(repo_id, name=None, split=None, **kwargs):
        if name == "default" and split == "train":
            return default_train_ds
        raise FileNotFoundError(f"[mock] không có split={split} cho config={name}")

    with patch("datasets.load_dataset", side_effect=fake_load_dataset_A):
        data_args = DataArguments(dataset_name="fake/repo", dataset_mode="auto")
        module = make_data_module(tok, data_args, is_pretrain=True)

    assert isinstance(module["data_collator"], DataCollatorForPreTokenizedCoT), \
        "Scenario A phải chọn DataCollatorForPreTokenizedCoT"
    assert module["eval_dataset"] is None  # validation split raise -> None, không crash
    batch = module["data_collator"]([module["train_dataset"][i] for i in range(3)])
    print(f"  collator = {type(module['data_collator']).__name__}")
    print(f"  batch input_ids.shape = {tuple(batch['input_ids'].shape)}")
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    print("  -> PASS: auto chọn đúng pre_tokenized, collate batch OK.\n")

    # ------------------------------------------------------------
    # 2) Scenario B: config "default" KHÔNG tồn tại -> fallback "raw"
    # ------------------------------------------------------------
    print("=== 2) Scenario B: config 'default' KHÔNG tồn tại -> fallback 'raw' ===")
    raw_train_ds = Dataset.from_list(raw_examples)

    def fake_load_dataset_B(repo_id, name=None, split=None, **kwargs):
        if name == "default":
            raise RuntimeError("[mock] config 'default' không tồn tại trong repo (chưa push ids/)")
        if name == "raw" and split == "train":
            return raw_train_ds
        raise FileNotFoundError(f"[mock] không có split={split} cho config={name}")

    with patch("datasets.load_dataset", side_effect=fake_load_dataset_B):
        data_args = DataArguments(dataset_name="fake/repo", dataset_mode="auto")
        module = make_data_module(tok, data_args, is_pretrain=True)

    assert isinstance(module["data_collator"], DataCollatorForCoT), \
        "Scenario B phải fallback về DataCollatorForCoT"
    batch = module["data_collator"]([module["train_dataset"][i] for i in range(3)])
    print(f"  collator = {type(module['data_collator']).__name__}")
    print(f"  batch input_ids.shape = {tuple(batch['input_ids'].shape)}")
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    print("  -> PASS: fallback đúng về raw, tokenize on-the-fly vẫn hoạt động.\n")

    print("Tất cả assertion PASS.")


if __name__ == "__main__":
    run()