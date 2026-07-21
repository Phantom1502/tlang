# Kinh nghiệm huấn luyện GRPO — Round 1

Ghi lại các bài học thực nghiệm khi tune `rounds/round1.json` và chạy round1 thật.
Mục tiêu: lần sau (round2, round3, ...) không phải dò lại từ đầu.

---

## 1. Tối ưu tham số buff (`buff_init`, `hold_buff_init`)

### Cách tìm giá trị init

Không đoán mò `buff_init`/`hold_buff_init` — dò bằng thực nghiệm:

1. Chốt trước tỉ lệ mong muốn (`target_action_ratio`, `target_hold_ratio`) — ví dụ
   action=20%, hold=10%.
2. Chạy 1 lần init với buff bắt đầu từ đáy (`buff_min`/`hold_buff_min`), để step
   update **lớn hơn bình thường** (chạy nhanh cho có tín hiệu), trong vài trăm step.
3. Quan sát `action_buffs`/`hold_buff` được in ra mỗi `save_steps`
   (`StatsPersistCallback.on_save` trong `train_grpo.py`) — chúng sẽ dao động rồi
   **hội tụ quanh 1 vùng giá trị** ứng với tỉ lệ target đã đặt.
4. Vùng hội tụ đó chính là `buff_init`/`hold_buff_init` nên dùng cho lần train thật.

Lý do làm vậy: buff là cơ chế bù trừ động (`update_buffs_from_stats`) — nếu tỉ lệ
action/hold thực tế thấp hơn target thì buff tăng, cao hơn thì buff giảm. Thay vì để
model phải tự leo từ 0 lên đúng mức trong hàng nghìn step thật (tốn step, và learning
rate GRPO rất nhỏ nên leo chậm), ta xác định trước điểm cân bằng rồi khởi động ngay
tại đó.

### Step update khi train thật — PHẢI nhỏ

Sau khi đã có `buff_init` tốt, **giảm mạnh** `buff_step`/`hold_buff_step` khi train
thật (khuyến nghị ~**0.005**, tức nhỏ hơn nhiều so với bước dò ban đầu).

Lý do: buff update mỗi `save_steps` dựa trên thống kê của đúng 1 chu kỳ ngắn (vài
chục/vài trăm rollout) — nhiễu (variance) giữa các chu kỳ khá lớn. Nếu `buff_step`
lớn, mỗi lần update buff nhảy mạnh theo nhiễu ngắn hạn → reward landscape đổi liên
tục giữa các chu kỳ → **model học trên 1 mục tiêu di chuyển không ổn định**, không
hội tụ. Step nhỏ giúp buff trôi dần theo xu hướng thật (trung bình dài hạn) thay vì
giật theo nhiễu từng chu kỳ.

Tóm lại: **dò `buff_init` bằng step lớn (để có tín hiệu nhanh), train thật bằng step
nhỏ (để ổn định)**.

---

## 2. Zone score — nên tắt hẳn (`zone_score_scale = 0`)

Ban đầu round1 dùng `zone_score_scale = 0.3` để thưởng thêm cho model khi vẽ zone
"chất lượng" (dựa trên `probe_zone_quality`). Sau khi quan sát, quyết định nên **tắt
hẳn** nhánh này (`zone_score_scale = 0`).

### Lý do

Zone là điều kiện **tiền đề** để action BUY/SELL được phép xảy ra (bảng E trong
`SemanticChecker`: zone_support + price_in_zone mới được BUY, tương tự cho SELL).
Nghĩa là:

- Model muốn có outcome tốt từ BUY/SELL → cần zone hợp lệ ở đúng vị trí trước.
- Nhưng chính hành động BUY/SELL có outcome tốt lại **quay lại củng cố** việc "vẽ
  zone kiểu này là đúng" — zone và outcome không độc lập, chúng tự khuếch đại lẫn
  nhau (feedback loop).

Hệ quả nếu vẫn thưởng riêng cho zone chất lượng: model có xu hướng đẩy zone theo
hướng **tối đa hoá zone_score** thay vì theo đúng ngữ nghĩa "vùng support/resistance
thật" — ví dụ nới zone rộng ra để tăng khả năng price_in_zone/touch, vì zone rộng dễ
"trúng" hơn zone hẹp chính xác. Không có tín hiệu nào trong reward ép zone phải hẹp
lại, nên nó chỉ có xu hướng nới rộng theo thời gian.

### Cách xử lý thay thế

Vì không thể dựa vào reward để ép zone "vừa phải", buộc phải **chặn cứng bằng
range cố định**: `zone_width_min_bins`/`zone_width_max_bins` trong `RoundConfig`
(đi qua `SemanticChecker._check_zone_width`) đóng vai trò ràng buộc cấu trúc, không
phải ràng buộc học được. Zone quality bonus tắt, nhưng range width vẫn giữ nguyên
để model không lợi dụng nới/co zone bừa bãi.

---

## 3. Checklist tham số cho round tiếp theo

Trước khi chạy 1 round mới, làm theo thứ tự:

1. Chốt `target_action_ratio`, `target_hold_ratio` mong muốn.
2. Dò `buff_init`/`hold_buff_init` bằng 1 lần chạy ngắn (vài trăm step), step update
   để lớn hơn bình thường, quan sát vùng hội tụ qua log `StatsPersistCallback`.
3. Set `buff_init`/`hold_buff_init` = giá trị hội tụ vừa tìm được.
4. Set `buff_step`/`hold_buff_step` nhỏ (~0.005) cho lần train thật.
5. `zone_score_scale = 0` — không thưởng riêng cho zone quality, chỉ dựa vào
   `zone_width_min_bins/max_bins` để chặn cấu trúc.
6. Chạy `RoundConfig.__post_init__` sẽ tự raise nếu `pass_gate2_bonus` (K) không đủ
   lớn so với worst-case reward — vẫn cần kiểm tra lại invariant này mỗi khi đổi
   `zone_score_scale`/`trade_fee_bins`/`sl_valid_bonus`.

---

## 4. Việc chưa làm / còn để ngỏ

- Chưa có cơ chế tự động hoá bước "dò buff_init" (hiện vẫn phải chạy tay + đọc log
  bằng mắt) — có thể cân nhắc viết script riêng đọc `*_stats_rank*.json` và gợi ý
  điểm hội tụ.
- Chưa đánh giá ảnh hưởng dài hạn của việc tắt hẳn `zone_score_scale` tới chất lượng
  zone thực tế (chỉ mới dựa trên suy luận lý thuyết + quan sát ngắn hạn) — cần theo
  dõi qua `eval_val.py` ở các checkpoint sau để xác nhận zone không suy biến theo
  hướng khác (vd luôn dùng đúng min/max width cho phép thay vì phân phối hợp lý).