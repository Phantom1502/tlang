from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

# =====================================================================
# task_dataset.py — Bọc 1 dataset GRPO gốc (schema prompt/future_bins/
# symbol/window_id) để mỗi row gốc sinh ra ĐÚNG 2 mẫu (task=zone,
# task=action) khi load, KHÔNG vật chất hoá bản sao nào trong bộ nhớ/đĩa
# — __getitem__ tự suy ra base_idx + task_id runtime mỗi lần được gọi.
#
# CẢNH BÁO CHƯA GIẢI QUYẾT (đọc trước khi dùng với GRPOTrainer thật):
# 2 bản (zone/action) của CÙNG 1 window có `prompt` GIỐNG HỆT NHAU (chart
# text không đổi theo task — bắt buộc, vì input model nhìn thấy phải
# nguyên vẹn). GRPOTrainer (trl) tính advantage bằng cách gom các
# completion trong batch theo prompt để chia nhóm num_generations. NẾU
# trl gom theo prompt-string thô (chưa verify hành vi thật của
# trl==1.8.0), 2 completion set của 2 task (sinh độc lập, chấm 2 công
# thức reward khác thang hoàn toàn) có nguy cơ bị trộn vào CÙNG 1 nhóm để
# trừ baseline — phá vỡ đúng mục tiêu tách task. BẮT BUỘC viết 1 test nhỏ
# (in group-key nội bộ mà GRPOTrainer dùng, hoặc đọc source trl 1.8.0 tại
# đúng đoạn compute advantage) TRƯỚC KHI tin tưởng thiết kế duplicate-row
# này an toàn tuyệt đối trong production. Nếu xác nhận có rủi ro thật,
# phương án dự phòng: bỏ duplicate, mỗi sample gốc chỉ sinh 1 dòng, reward
# = blend 0.5*reward_zone + 0.5*reward_action (đổi 1 hàm ở reward_func_v2,
# không cần đổi lại toàn bộ 4 tầng đã tách).
# =====================================================================

TASK_ZONE = "zone"
TASK_ACTION = "action"
TASKS: tuple = (TASK_ZONE, TASK_ACTION)


class TaskExpandedGRPODataset:
    """
    __len__ = 2 * len(base). __getitem__(idx) suy base_idx = idx // n_tasks,
    task_idx = idx % n_tasks -> xen kẽ zone/action theo idx liên tiếp (0,1,2,3,..
    -> base 0/zone, base 0/action, base 1/zone, base 1/action, ...), không
    phải nửa đầu/nửa sau mảng — chỉ để dễ debug bằng mắt, RandomSampler của
    Trainer mới là thứ thật sự đảm bảo trộn đều giữa các batch.

    `base` chỉ cần hỗ trợ __len__/__getitem__ trả về dict — dùng thẳng
    được với `datasets.Dataset` (HF) sau `load_dataset(...)`.
    """

    def __init__(self, base_dataset: Sequence[Dict[str, Any]], tasks: tuple = TASKS):
        self.base = base_dataset
        self.tasks = tasks
        self.n_tasks = len(tasks)

    def __len__(self) -> int:
        return len(self.base) * self.n_tasks

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx, task_idx = divmod(idx, self.n_tasks)
        row = dict(self.base[base_idx])
        task_id = self.tasks[task_idx]
        row["task_id"] = task_id
        if "window_id" in row:
            row["window_id"] = f"{row['window_id']}_{task_id}"
        return row


def add_task_id_columns(dataset, tasks: tuple = TASKS, shuffle_seed: Optional[int] = None):
    """
    Bản THẬT SỰ VẬT CHẤT HOÁ (khác TaskExpandedGRPODataset ở trên) — dùng
    cho GRPOTrainer, vì Trainer/GRPOTrainer của trl dựa vào API của
    `datasets.Dataset` thật (column_names, .map, .remove_columns,
    integration với DataLoader) ở nhiều chỗ nội bộ (đặc biệt khi
    remove_unused_columns liên quan). Một object Python tự chế
    (TaskExpandedGRPODataset) DÙ hỗ trợ __len__/__getitem__ vẫn CÓ RỦI RO
    không tương thích với những chỗ Trainer gọi thẳng API kiểu HF Dataset
    — CHƯA verify với đúng version trl đang dùng, nên train_grpo_v2.py
    dùng hàm NÀY (trả về datasets.Dataset thật) làm đường đi an toàn mặc
    định, thay vì TaskExpandedGRPODataset.

    Input: 1 `datasets.Dataset` (HF), 1 split, schema GRPO gốc
    (prompt/future_bins/symbol/window_id).
    Output: `datasets.Dataset` gấp đôi số row, có thêm cột `task_id`,
    `window_id` đã hậu tố theo task, đã shuffle lại (nếu shuffle_seed
    truyền vào) để 2 nửa zone/action không nằm liền khối — tránh 1 batch
    tình cờ toàn zone hoặc toàn action nếu Trainer sampler không đủ ngẫu
    nhiên.
    """
    from datasets import concatenate_datasets

    def _make_tagger(task_id: str):
        def _tag(row: Dict[str, Any]) -> Dict[str, Any]:
            row = dict(row)
            row["task_id"] = task_id
            if "window_id" in row:
                row["window_id"] = f"{row['window_id']}_{task_id}"
            return row
        return _tag

    variants = [dataset.map(_make_tagger(task_id)) for task_id in tasks]
    combined = concatenate_datasets(variants)
    if shuffle_seed is not None:
        combined = combined.shuffle(seed=shuffle_seed)
    return combined