"""
app/gen/push_dataset_to_hub.py — Upload dataset (output của
app/gen/dataset_builder.py) lên 1 dataset repo trên HF Hub.

v3.1 — Upload thẳng vào ROOT của repo (không tạo thư mục con "data/").
`datasets.load_dataset()` tự nhận diện split qua TỪ KHÓA trong tên file
(vd "train-xxx.parquet", "validation-xxx.parquet"), không bắt buộc phải
nằm trong 1 thư mục con cụ thể — nên bỏ "data/{split}/" cho gọn, chỉ cần
prefix đúng tên split vào tên file khi upload.

Lưu ý: nếu repo đã có sẵn thư mục "data/" từ lần chạy trước (bản cũ dùng
`DatasetDict.push_to_hub()`, hành vi mặc định của thư viện `datasets` là
tự tạo cấu trúc "data/<split>-*.parquet") — nên dọn thư mục "data/" cũ đi
(xoá qua giao diện Hub hoặc `HfApi().delete_folder()`) trước khi upload
tiếp theo cấu trúc root này, tránh 2 cấu trúc lẫn lộn khó kiểm soát.

v3 — BỎ HOÀN TOÀN cơ chế --append tải-về-gộp-lại-push (v2) vì:
  - Mỗi file parquet được build ĐỘC LẬP theo symbol/khối thời gian, KHÔNG
    bao giờ tạo lại/update (đã xác nhận với người dùng) -> không có gì
    cần merge/dedup trong bộ nhớ cả.
  - Tải cả dataset cũ về + to_pandas() + push_to_hub() bắt datasets phải
    pickle (qua dill) toàn bộ object để tính fingerprint cache -> với vài
    GB dữ liệu thật, RAM cần gấp nhiều lần dung lượng file -> MemoryError
    (đã gặp thực tế). Thiết kế v2 sai hướng, bỏ đi.

Cách làm (v3): mỗi lần gọi script này chỉ UPLOAD THẲNG file (hoặc vài
file) MỚI lên ROOT của repo bằng huggingface_hub.HfApi().upload_file() —
không tải gì về, không parse nội dung, không pickle object dataset nào
cả. Tên file được prefix "train-"/"validation-" (nếu chưa có sẵn từ khóa
đó) để datasets.load_dataset(repo_id) tự nhận diện đúng split.

Trả giá: KHÔNG có bước dedup/check-overlap tự động nữa (v2 có nhưng phải
tải dữ liệu cũ về mới check được) — chấp nhận được vì mỗi file độc lập,
không tái tạo. Script chỉ còn 1 lớp phòng vệ NHẸ: cảnh báo (không chặn)
nếu tên file trùng với file đã có trên repo (dùng list_repo_files, không
tải nội dung) — tránh vô tình ghi đè nhầm.

Yêu cầu trước khi chạy:
    huggingface-cli login
    # hoặc: export HF_TOKEN=hf_xxx

Usage:
    # Upload 1 hoặc nhiều file train
    python -m app.gen.push_dataset_to_hub \
        --parquet_paths AUDUSD_15Min_pretrain_sft.parquet \
        --repo_id <org>/trading-llm-pretrain

    # Upload thêm 1 file val (hold-out riêng, khác thời gian/symbol)
    python -m app.gen.push_dataset_to_hub \
        --holdout_parquet_paths XAUUSD_15Min_pretrain_sft_VAL.parquet \
        --repo_id <org>/trading-llm-pretrain

    # Về sau thêm symbol mới -> gọi lại y hệt, KHÔNG cần --append gì cả,
    # file mới tự cộng dồn vào repo, file cũ không bị động tới:
    python -m app.gen.push_dataset_to_hub \
        --parquet_paths GBPUSD_15Min_pretrain_sft.parquet \
        --repo_id <org>/trading-llm-pretrain

    # Xem trước, KHÔNG upload thật
    python -m app.gen.push_dataset_to_hub --parquet_paths ... --repo_id ... --dry_run
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

PRETRAIN_SFT_COLUMNS = {"prompt", "completion"}
GRPO_COLUMNS = {"prompt", "future_bins", "symbol", "window_id"}


def _validate_schema_light(path: str) -> str:
    """Chỉ đọc metadata schema của parquet (KHÔNG load toàn bộ dữ liệu vào
    RAM) để xác nhận đúng pretrain/sft hay grpo trước khi upload."""
    import pyarrow.parquet as pq

    cols = set(pq.ParquetFile(path).schema.names)
    if cols == PRETRAIN_SFT_COLUMNS:
        return "pretrain_sft"
    if cols == GRPO_COLUMNS:
        return "grpo"
    raise ValueError(
        f"{path}: schema không khớp pretrain/sft ({PRETRAIN_SFT_COLUMNS}) hay grpo "
        f"({GRPO_COLUMNS}) — cột thực tế: {cols}."
    )


def _upload_files(
    paths: Sequence[str],
    repo_id: str,
    split_dir: str,   # "train" | "validation"
    private: bool,
    commit_message: str,
    token: Optional[str],
    dry_run: bool,
) -> None:
    """Upload thẳng vào ROOT repo — path_in_repo KHÔNG có thư mục con
    "data/", chỉ prefix tên split vào tên file (xem docstring module)."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    if not dry_run:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
        try:
            existing_files = set(api.list_repo_files(repo_id, repo_type="dataset"))
        except Exception:
            existing_files = set()
    else:
        existing_files = set()

    for path in paths:
        schema = _validate_schema_light(path)
        basename = os.path.basename(path)
        # Prefix tên split vào tên file (thay vì đặt trong thư mục con) —
        # datasets.load_dataset() nhận diện split qua từ khóa "train"/
        # "validation" xuất hiện trong tên file, không cần thư mục riêng.
        remote_path = basename if split_dir in basename else f"{split_dir}-{basename}"

        if remote_path in existing_files:
            print(f"  [CẢNH BÁO] {remote_path} đã tồn tại trên repo — upload sẽ GHI ĐÈ file này. "
                  f"Nếu không cố ý, đổi tên file local trước khi upload lại.")

        if dry_run:
            print(f"[DRY RUN] sẽ upload {path} (schema={schema}) -> {repo_id}:{remote_path}")
            continue

        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=commit_message,
        )
        print(f"  Đã upload {path} (schema={schema}) -> {repo_id}:{remote_path}")


def push_dataset(
    parquet_paths: Optional[Sequence[str]],
    repo_id: str,
    holdout_parquet_paths: Optional[Sequence[str]] = None,
    private: bool = False,
    commit_message: str = "Add data file(s)",
    token: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    if not parquet_paths and not holdout_parquet_paths:
        raise ValueError("Cần truyền ít nhất --parquet_paths hoặc --holdout_parquet_paths.")

    if parquet_paths:
        print(f"[push_dataset_to_hub] upload {len(parquet_paths)} file -> split train")
        _upload_files(parquet_paths, repo_id, "train", private, commit_message, token, dry_run)

    if holdout_parquet_paths:
        print(f"[push_dataset_to_hub] upload {len(holdout_parquet_paths)} file -> split validation (hold-out)")
        _upload_files(holdout_parquet_paths, repo_id, "validation", private, commit_message, token, dry_run)

    if not dry_run:
        print(f"\nXong. Kiểm tra lại bằng: datasets.load_dataset({repo_id!r})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet_paths", nargs="+", default=None, help="File parquet đưa vào split train")
    p.add_argument(
        "--holdout_parquet_paths", nargs="+", default=None,
        help="File parquet RIÊNG BIỆT đưa vào split validation (hold-out, khác thời gian/symbol)",
    )
    p.add_argument("--repo_id", required=True, help="vd: my-org/trading-llm-pretrain")
    p.add_argument("--private", action="store_true")
    p.add_argument("--commit_message", default="Add data file(s)")
    p.add_argument("--token", default=None, help="HF token — mặc định dùng cached login / biến môi trường HF_TOKEN")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    push_dataset(
        parquet_paths=args.parquet_paths,
        repo_id=args.repo_id,
        holdout_parquet_paths=args.holdout_parquet_paths,
        private=args.private,
        commit_message=args.commit_message,
        token=args.token,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()