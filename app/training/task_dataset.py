from __future__ import annotations

from typing import Any, Dict, Sequence

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