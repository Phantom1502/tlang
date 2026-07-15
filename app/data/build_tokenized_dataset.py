"""
app/data/build_tokenized_dataset.py — Map dataset raw text sang input_ids,
rồi push lên cùng repo nhưng nằm ở subset 'ids'.

Hỗ trợ tham số --split để chạy riêng lẻ từng tập dữ liệu (train/val), tránh
chạy lại từ đầu.

=== CHANGELOG (fix cho scale ~20GB) ===

1. batched=True cho .map() — bản cũ gọi tokenizer 1 example/lần, bỏ phí hoàn
   toàn khả năng batch-encode của Rust backend (`tokenizers`), chậm hơn
   batched có thể 10-50 lần ở scale hàng chục triệu dòng. Giờ tokenize theo
   batch_size (mặc định 1000) — encode 1 lần cho cả batch.

2. TOKENIZERS_PARALLELISM=false trước khi import bất kỳ gì liên quan
   tokenizer — tránh xung đột giữa đa luồng nội bộ của Rust tokenizer và
   multiprocessing fork của `datasets.map(num_proc=...)`. Không set biến này
   thì thư viện tự tắt song song ở tiến trình con kèm cảnh báo, khiến
   num_proc>1 chỉ tốn overhead fork mà không có lợi ích gì.

3. Log số sample bị mask HẾT (labels toàn -100, không đóng góp loss) — xảy
   ra khi prompt dài gần chạm max_length khiến n_mask == len(full_ids) sau
   khi cắt. Trước đây âm thầm bỏ qua, giờ in cảnh báo rõ số lượng.

4. CẢNH BÁO CHƯA FIX ĐƯỢC BẰNG CODE — hành vi push_to_hub() khi chạy riêng
   lẻ --split (chỉ có 1 split trong DatasetDict) có ghi đè cấu hình
   "configs" YAML của các split khác trong cùng config_name hay không PHỤ
   THUỘC VERSION `datasets`, KHÔNG được đảm bảo. Trước khi push thật 20GB,
   BẮT BUỘC test trên 1 repo nhỏ: push val trước, push train sau (2 lệnh
   riêng biệt), rồi load_dataset(repo_id, name="ids") xác nhận CẢ 2 split
   còn nguyên trên Hub. Đừng tin giả định này khi chưa tự kiểm chứng.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
from typing import Any, Dict, List

# PHẢI set TRƯỚC khi import bất kỳ thứ gì từ transformers/tokenizers —
# tránh race giữa đa luồng nội bộ Rust tokenizer và multiprocessing fork
# của datasets.map(num_proc=...). Đặt ở đây (module-level, đầu file) để
# áp dụng ngay cả khi file này bị import (không chỉ khi chạy __main__).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger("build_tokenized_dataset")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

LABEL_PAD_ID = -100


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", required=True, help="Repo dataset trên Hub (chứa cả subset 'raw' và 'ids')")
    p.add_argument("--kind", choices=["pretrain_sft", "grpo"], required=True)
    p.add_argument("--split", default=None, help="Chỉ định split cụ thể để xử lý (vd: train, val). Nếu để None sẽ chạy tất cả.")
    p.add_argument("--tokenizer_repo", default=None, help="Mặc định DEFAULT_TOKENIZER_REPO trong app/tokenizer/hub.py")
    p.add_argument("--max_length", type=int, default=512, help="Khớp MAX_POSITION_EMBEDDINGS")
    p.add_argument("--num_proc", type=int, default=4)
    p.add_argument(
        "--batch_size", type=int, default=1000,
        help="Số example/batch cho .map(batched=True) — batch lớn hơn giảm overhead gọi tokenizer, "
             "nhưng tốn RAM hơn mỗi worker. 1000 là điểm khởi đầu hợp lý cho seq_len ~230 token/sample.",
    )
    p.add_argument("--private", action="store_true")
    p.add_argument(
        "--dry_run", action="store_true",
        help="Chỉ chạy .map() + in thống kê seq_len, KHÔNG push lên Hub — luôn chạy trước khi push thật",
    )
    p.add_argument(
        "--verify_samples", type=int, default=2000,
        help="Số row lấy NGẪU NHIÊN mỗi split để verify độc lập (decode round-trip) trước khi push. "
             "0 = tắt hẳn (không khuyến khích). Verify chạy CẢ khi --dry_run.",
    )
    p.add_argument(
        "--skip_verify", action="store_true",
        help="Bỏ qua hoàn toàn bước verify — CHỈ dùng nếu đã verify kỹ ở lần chạy trước với "
             "đúng version code/tokenizer này. Mặc định KHÔNG bật (fail-closed).",
    )
    return p


# =====================================================================
# pretrain_sft — batched: nhận dict-of-list, trả dict-of-list.
# Cùng rule mask với app.data.collator.DataCollatorForCoT (1 nguồn sự thật
# duy nhất về logic mask: <bos>+prompt -> -100), viết lại dạng batched ở
# đây vì đây là nơi thật sự cần hiệu năng (dataset ~20GB).
# =====================================================================
def _tokenize_and_mask_batch(
    batch: Dict[str, List[Any]], tokenizer, max_length: int
) -> Dict[str, List[Any]]:
    prompts: List[str] = batch["prompt"]
    completions: List[str] = batch["completion"]
    fulls = [p + " " + c for p, c in zip(prompts, completions)]

    # Encode CẢ BATCH trong 1 lệnh — đây là chỗ khác biệt hiệu năng lớn nhất
    # so với bản cũ (gọi tokenizer() riêng từng example).
    prompt_ids_batch = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    full_ids_batch = tokenizer(fulls, add_special_tokens=True)["input_ids"]

    out_input_ids: List[List[int]] = []
    out_labels: List[List[int]] = []
    n_fully_masked = 0

    for prompt_ids, full_ids in zip(prompt_ids_batch, full_ids_batch):
        n_mask = min(1 + len(prompt_ids), len(full_ids))
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            n_mask = min(n_mask, len(full_ids))

        labels = [LABEL_PAD_ID] * n_mask + full_ids[n_mask:]
        if n_mask >= len(full_ids):
            n_fully_masked += 1

        out_input_ids.append(full_ids)
        out_labels.append(labels)

    if n_fully_masked:
        logger.warning(
            f"{n_fully_masked}/{len(fulls)} sample trong batch này bị mask HẾT "
            f"(labels toàn -100, không đóng góp loss) — do prompt dài gần chạm "
            f"max_length={max_length} và bị cắt mất phần completion. Kiểm tra lại "
            f"phân phối seq_len nếu số này đáng kể."
        )

    return {"input_ids": out_input_ids, "labels": out_labels}


# =====================================================================
# grpo — batched: chỉ cần tokenize prompt (không có completion/mask).
# =====================================================================
def _tokenize_grpo_prompt_batch(
    batch: Dict[str, List[Any]], tokenizer, max_length: int
) -> Dict[str, List[Any]]:
    prompts: List[str] = batch["prompt"]
    prompt_ids_batch = tokenizer(prompts, add_special_tokens=False)["input_ids"]

    out_ids: List[List[int]] = []
    out_lens: List[int] = []
    n_truncated = 0

    for ids in prompt_ids_batch:
        if len(ids) > max_length:
            ids = ids[:max_length]
            n_truncated += 1
        out_ids.append(ids)
        out_lens.append(len(ids))

    if n_truncated:
        logger.warning(f"{n_truncated}/{len(prompts)} prompt trong batch này bị cắt ở max_length={max_length}.")

    return {"prompt_input_ids": out_ids, "prompt_length": out_lens}


# Giữ lại 2 hàm non-batched cũ dưới tên khác (không xoá) — để tương thích
# ngược cho bất kỳ chỗ nào khác còn import trực tiếp (vd
# app/data/build_tokenized_dataset_demo.py đang import _tokenize_grpo_prompt
# theo tên cũ). Chỉ dùng cho demo/1-off, KHÔNG dùng trong pipeline .map()
# thật ở dưới — pipeline .map() thật đã chuyển sang bản batched ở trên.
def _tokenize_grpo_prompt(example: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, Any]:
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    if len(prompt_ids) > max_length:
        prompt_ids = prompt_ids[:max_length]
    return {"prompt_input_ids": prompt_ids, "prompt_length": len(prompt_ids)}


# =====================================================================
# VERIFY — bước bắt buộc trước khi push. Thiết kế CỐ TÌNH ĐỘC LẬP với
# logic mask ở _tokenize_and_mask_batch: không tái sử dụng công thức
# n_mask/labels ở trên, mà chỉ dựa vào 1 sự thật cuối cùng duy nhất —
# decode(các token KHÔNG bị mask) phải khớp CHÍNH XÁC với completion gốc.
# Nếu logic mask có bug (off-by-one, lệch do batching, lệch do truncation),
# check kiểu "so công thức với chính nó" sẽ không bắt được vì cả 2 nơi
# cùng sai giống nhau — chỉ có so với TEXT GỐC mới đáng tin.
#
# Bất kỳ mismatch nào -> raise ngay, KHÔNG cho push. Đây là "fail-closed":
# lỗi phải chặn đứng pipeline, không được để lọt qua thành 1 dataset
# tokenize sai âm thầm rồi tốn cả round train mới phát hiện.
# =====================================================================
def _verify_pretrain_sft_split(
    raw_split, mapped_split, tokenizer, n_samples: int, max_length: int = 512, seed: int = 0
) -> None:
    n = len(raw_split)
    if n == 0 or n_samples <= 0:
        logger.warning(f"[VERIFY] bỏ qua verify (n={n}, n_samples={n_samples}).")
        return
    if len(mapped_split) != n:
        raise RuntimeError(
            f"[VERIFY FAIL] Số row mapped ({len(mapped_split)}) khác số row raw ({n}) — "
            f".map() làm mất/thêm row, KHÔNG được push."
        )

    rng = random.Random(seed)
    sample_idx = rng.sample(range(n), min(n_samples, n))

    mismatches: List[Dict[str, Any]] = []
    structure_errors: List[int] = []
    n_truncated_ok = 0

    for i in sample_idx:
        row = mapped_split[i]
        input_ids, labels = row["input_ids"], row["labels"]

        if len(input_ids) != len(labels):
            structure_errors.append(i)
            continue

        # Mask phải là 1 KHỐI LIỀN MẠCH ở đầu (bos + prompt), không được
        # xen kẽ -100 ở giữa/cuối — nếu xen kẽ nghĩa là boundary tính sai.
        seen_unmasked = False
        for lab in labels:
            if lab == LABEL_PAD_ID:
                if seen_unmasked:
                    structure_errors.append(i)
                    break
            else:
                seen_unmasked = True

        unmasked_ids = [tid for tid, lab in zip(input_ids, labels) if lab != LABEL_PAD_ID]
        decoded_completion = tokenizer.decode(unmasked_ids, skip_special_tokens=True).strip()
        expected_completion = raw_split[i]["completion"].strip()

        if decoded_completion == expected_completion:
            continue

        # Case hợp lệ: sample bị CẮT ở max_length (input_ids chạm đúng
        # max_length) -> decode chỉ còn 1 PREFIX của completion gốc, đây
        # KHÔNG phải bug, chỉ fail nếu decode không phải prefix thật sự.
        if len(input_ids) >= max_length and expected_completion.startswith(decoded_completion) and decoded_completion:
            n_truncated_ok += 1
            continue

        mismatches.append({
            "index": i,
            "expected": expected_completion[:80],
            "decoded": decoded_completion[:80],
        })

    if structure_errors:
        raise RuntimeError(
            f"[VERIFY FAIL] {len(structure_errors)}/{len(sample_idx)} sample có cấu trúc mask SAI "
            f"(len mismatch hoặc -100 xen kẽ thay vì là 1 khối liền ở đầu) — index ví dụ: "
            f"{structure_errors[:10]}. KHÔNG push lên Hub."
        )

    if mismatches:
        preview = mismatches[:5]
        raise RuntimeError(
            f"[VERIFY FAIL] {len(mismatches)}/{len(sample_idx)} sample decode lại KHÔNG khớp "
            f"completion gốc — mask SAI, KHÔNG được push. Ví dụ:\n"
            + "\n".join(
                f"  idx={m['index']}\n    gốc   : {m['expected']!r}\n    decode: {m['decoded']!r}"
                for m in preview
            )
        )

    trunc_note = f", {n_truncated_ok} sample bị cắt hợp lệ ở max_length (decode = prefix gốc)" if n_truncated_ok else ""
    logger.info(
        f"[VERIFY] pretrain_sft: {len(sample_idx)}/{n} sample ngẫu nhiên — decode round-trip khớp "
        f"100%{trunc_note}, cấu trúc mask hợp lệ (khối liền mạch ở đầu). AN TOÀN để push."
    )


def _verify_grpo_split(raw_split, mapped_split, tokenizer, n_samples: int, seed: int = 0) -> None:
    n = len(raw_split)
    if n == 0 or n_samples <= 0:
        logger.warning(f"[VERIFY] bỏ qua verify (n={n}, n_samples={n_samples}).")
        return
    if len(mapped_split) != n:
        raise RuntimeError(
            f"[VERIFY FAIL] Số row mapped ({len(mapped_split)}) khác số row raw ({n}) — "
            f".map() làm mất/thêm row, KHÔNG được push."
        )

    rng = random.Random(seed)
    sample_idx = rng.sample(range(n), min(n_samples, n))
    mismatches: List[int] = []

    for i in sample_idx:
        prompt_ids = mapped_split[i]["prompt_input_ids"]
        decoded = tokenizer.decode(prompt_ids, skip_special_tokens=True).strip()
        expected = raw_split[i]["prompt"].strip()
        # Nếu prompt bị cắt ở max_length, decoded sẽ là PREFIX của expected — chấp
        # nhận trường hợp này, chỉ fail khi KHÔNG phải quan hệ prefix (lỗi thật sự).
        if decoded != expected and not expected.startswith(decoded):
            mismatches.append(i)

    if mismatches:
        raise RuntimeError(
            f"[VERIFY FAIL] {len(mismatches)}/{len(sample_idx)} sample GRPO decode prompt "
            f"KHÔNG khớp gốc — index ví dụ: {mismatches[:10]}. KHÔNG push lên Hub."
        )

    logger.info(f"[VERIFY] grpo: {len(sample_idx)}/{n} sample ngẫu nhiên — decode prompt khớp 100%. AN TOÀN để push.")


def main() -> None:
    args = build_arg_parser().parse_args()

    from app.tokenizer.hub import load_tokenizer
    tok = load_tokenizer(repo_id=args.tokenizer_repo, allow_local_fallback=False)
    logger.info(f"tokenizer vocab_size = {tok.vocab_size}")

    from datasets import load_dataset, DatasetDict

    if args.split:
        logger.info(f"Đang tải riêng lẻ split '{args.split}' từ subset 'raw'...")
        single_ds = load_dataset(args.repo_id, name="raw", split=args.split)
        raw = DatasetDict({args.split: single_ds})
    else:
        logger.info("Đang tải TOÀN BỘ các split từ subset 'raw'...")
        raw = load_dataset(args.repo_id, name="raw")
        if not isinstance(raw, DatasetDict):
            raw = DatasetDict({"train": raw})

    logger.info(f"Loaded {args.repo_id} (subset 'raw') | Các split sẽ xử lý: {list(raw.keys())}")

    if args.split is not None:
        logger.warning(
            "Đang push riêng lẻ 1 split (--split được chỉ định). Hành vi push_to_hub() với "
            "DatasetDict chỉ có 1 split đối với việc GIỮ NGUYÊN các split khác trong cùng "
            "config_name='ids' PHỤ THUỘC VERSION `datasets`, chưa được đảm bảo bởi code này. "
            "Nếu đây là lần đầu chạy trên repo thật, hãy verify bằng tay: sau khi push xong, "
            "load_dataset(repo_id, name='ids') và kiểm tra CẢ split vừa push LẪN các split "
            "khác đã push trước đó còn nguyên."
        )

    if args.kind == "pretrain_sft":
        def _map_fn(batch):
            return _tokenize_and_mask_batch(batch, tok, args.max_length)

        mapped = raw.map(
            _map_fn,
            batched=True,
            batch_size=args.batch_size,
            remove_columns=next(iter(raw.values())).column_names,
            num_proc=args.num_proc,
            desc=f"Tokenize+mask (pretrain_sft, split={args.split or 'all'})",
        )

        for split_name, dataset in mapped.items():
            lens = [len(x) for x in dataset["input_ids"]]
            avg_len = sum(lens) / len(lens) if lens else 0
            logger.info(
                f"[{split_name}] seq_len: min={min(lens) if lens else 0} "
                f"max={max(lens) if lens else 0} avg={avg_len:.1f}"
            )
            n_truncated = sum(1 for l in lens if l >= args.max_length)
            if n_truncated:
                logger.warning(f"[{split_name}] {n_truncated}/{len(lens)} sample chạm max_length.")

    else:  # grpo
        def _map_fn(batch):
            return _tokenize_grpo_prompt_batch(batch, tok, args.max_length)

        mapped = raw.map(
            _map_fn,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            desc=f"Tokenize prompt (grpo, split={args.split or 'all'})",
        )

        for split_name, dataset in mapped.items():
            lens = dataset["prompt_length"]
            avg_len = sum(lens) / len(lens) if lens else 0
            logger.info(
                f"[{split_name}] prompt_len: min={min(lens) if lens else 0} "
                f"max={max(lens) if lens else 0} avg={avg_len:.1f}"
            )

    for split_name, dataset in mapped.items():
        logger.info(f"Kết quả split '{split_name}': {len(dataset)} row, columns={dataset.column_names}")

    # ------------------------------------------------------------
    # VERIFY — bắt buộc trước khi push (mặc định BẬT). Chạy CẢ khi
    # --dry_run, vì dry_run chính là lúc nên phát hiện lỗi sớm nhất, trước
    # khi tốn thời gian/tiền chạy .map() thật trên toàn bộ 20GB rồi push.
    # ------------------------------------------------------------
    if args.skip_verify:
        logger.warning(
            "[VERIFY] --skip_verify BẬT — bỏ qua hoàn toàn bước kiểm tra mask/prompt. "
            "CHỈ nên dùng nếu đã verify kỹ ở lần chạy trước với đúng version code/tokenizer này."
        )
    else:
        for split_name in mapped.keys():
            if args.kind == "pretrain_sft":
                _verify_pretrain_sft_split(
                    raw[split_name], mapped[split_name], tok,
                    n_samples=args.verify_samples, max_length=args.max_length,
                )
            else:
                _verify_grpo_split(
                    raw[split_name], mapped[split_name], tok, n_samples=args.verify_samples,
                )

    if args.dry_run:
        logger.info("[DRY RUN] Verify xong, không push lên Hub.")
        return

    mapped.push_to_hub(args.repo_id, config_name="ids", private=args.private)
    logger.info(f"Đã cập nhật thành công dữ liệu tokenize lên subset 'ids' của repo: {args.repo_id}")


if __name__ == "__main__":
    main()