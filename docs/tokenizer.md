# DOC: `app/tokenizer/` — Vocab / Build / Hub

Version: 1.0 (đi kèm `spec_trading_llm_v0.3.md` mục 3). Tool dùng CHUNG cho 2 nhánh khác:
`data_prepare` (build dataset "ids") và `training` (load lúc train/GRPO). Không phụ thuộc ngược vào
2 nhánh đó — chỉ phụ thuộc `app/lang/tokens.py`.

---

## 1. Quyết định: WordLevel, vocab đóng cố định — KHÔNG BPE

Ngôn ngữ think/action là 1 **grammar hình thức đóng** — toàn bộ token hợp lệ đã liệt kê tường minh
trong `app/lang/tokens.py` + bảng token spec của `app/lang/lexer.py`. Không có khái niệm "từ mới xuất
hiện trong corpus" — tiền đề mà BPE được thiết kế để giải quyết không tồn tại ở đây. Nghiêm trọng
hơn: BPE học merge theo tần suất thống kê, có thể tự ý gộp 2 digit token đứng cạnh nhau
(`"0"+"5"→"05"`), **phá vỡ trực tiếp** bất biến digit-decompose (current_price/zone/SL phải tách
từng chữ số thành 1 token riêng, dùng chung 10 token `0`-`9`, để model học so sánh/số học giữa các
field này dễ hơn).

Điều thay đổi thật khi scale dữ liệu lớn (~10B token): KHÔNG PHẢI thuật toán tokenize, mà là cách
implement (Rust backend `tokenizers`, không phải `PreTrainedTokenizer` thuần Python — xem mục 4).

---

## 2. `vocab_builder.py` — vocab tính ra từ hằng số, không "học"

```python
build_vocab() -> Dict[str, int]
```

Import trực tiếp `BIN_MIN, BIN_MAX, RR_MIN, RR_MAX` từ `app.lang.tokens` — KHÔNG tự định nghĩa lại.
9 nhóm token, id gán tuần tự SAU 4 special token (`<unk>=0, <bos>=1, <eos>=2, <pad>=3`):

| # | Nhóm | Số lượng |
|---|---|---|
| 1 | structural_tags | 6 |
| 2 | candle_O/H/L/C | 4×1024 = 4096 |
| 3 | trend | 3 |
| 4 | digit_field_tags | 7 |
| 5 | digit (dùng CHUNG cho current_price/zone/SL) | 10 |
| 6 | colon | 1 |
| 7 | flags | 2 |
| 8 | action_type | 7 |
| 9 | rr | 9 |

**Tổng = 4145 token** (số THẬT, tính bằng code — chạy `python -m app.tokenizer.vocab_builder` để in
lại bất cứ lúc nào, không cần tin vào con số ghi trong doc).

**Vocab contract**: thứ tự nhóm trong `_build_groups()` quyết định id tuyệt đối của từng token. Thêm
token mới (mở rộng `BIN_MAX`, thêm `action_type`...) CHỈ được thêm vào CUỐI danh sách nhóm hiện có —
không chèn giữa, không đổi thứ tự cũ. Đổi thứ tự = đổi id = mọi checkpoint model đã train (embedding
table học theo id cũ) không tương thích nữa.

---

## 3. `build_tokenizer.py` — WordLevel + WhitespaceSplit

```python
Tokenizer(WordLevel(vocab=build_vocab(), unk_token="<unk>"))
tokenizer.pre_tokenizer = WhitespaceSplit()   # KHÔNG dùng Whitespace mặc định (tách cả punctuation)
tokenizer.post_processor = TemplateProcessing(
    single="<bos> $A <eos>",
    pair="<bos> $A <eos> <bos> $B <eos>",
)
```

`WhitespaceSplit` (không phải `Whitespace`) — chỉ tách theo khoảng trắng ASCII, KHÔNG tách theo
punctuation (quan trọng vì `<O_543>` chứa ký tự không phải `\w`, phải giữ nguyên vẹn). Điều kiện bắt
buộc đi kèm: generator (`app/data_prepare/generator.py`) phải in **mỗi token cách nhau đúng 1 khoảng
trắng**, kể cả từng digit rời.

Post-processor `TemplateProcessing` tự động bọc `<bos>...<eos>` quanh MỌI lần encode với
`add_special_tokens=True` (default). Điều này đúng cho pretrain/SFT (`full_text = prompt + " " +
completion`, encode 1 lần → `<bos>+chart+think+action+<eos>`), nhưng **SAI cho GRPO rollout** — xem
mục 6 (pitfall quan trọng nhất của cả file này).

---

## 4. `push_to_hub.py` / `hub.py` — tokenizer là artifact BẤT BIẾN

**Luồng chuẩn**: build+push 1 lần (hoặc khi vocab thật sự đổi), mọi nơi khác chỉ `load_tokenizer()`:

```bash
python -m app.tokenizer.push_to_hub --repo_id <org>/tlang-tokenizer --dry_run   # xem trước
python -m app.tokenizer.push_to_hub --repo_id <org>/tlang-tokenizer            # push thật
```

```python
from app.tokenizer.hub import load_tokenizer
tok = load_tokenizer()                          # DEFAULT_TOKENIZER_REPO
tok = load_tokenizer(repo_id=args.repo_id)       # convention thật đang dùng — xem dưới
tok = load_tokenizer(allow_local_fallback=False) # train thật: fail loud nếu lỗi mạng/config, KHÔNG fallback build local
```

**Convention thật đang áp dụng (khác 1 chi tiết so với thiết kế "1 `DEFAULT_TOKENIZER_REPO` cho mọi
nơi" ban đầu)**: mỗi **model checkpoint repo** (`{model}-pretrain`/`-sft`/`-grpo-round{N}`) đã được
add sẵn tokenizer TRƯỚC khi train (thao tác tay 1 lần khi tạo repo) — mọi script train load tokenizer
bằng `load_tokenizer(repo_id=args.repo_id, ...)`, tức là load TỪ CHÍNH model repo đang train, không
phải từ 1 tokenizer-repo tách biệt. Lý do: đảm bảo model repo tự chứa đủ (model + tokenizer) khi cần
inference sau này, không phải nhớ thêm 1 repo tokenizer riêng.

`load_tokenizer()` có fallback build local (`allow_local_fallback=True` mặc định) — CHỈ dùng cho
dev/test, in cảnh báo rõ ràng. Script train thật LUÔN gọi `allow_local_fallback=False` để lỗi
mạng/config lộ ra ngay, không âm thầm train bằng tokenizer build-lại-tại-chỗ có thể lệch checkpoint
cũ.

---

## 5. Nguyên tắc "ánh xạ trung thực, không tự vá lành"

Theo bài học đã ghi ở `docs/lang.md` mục 5: token không có trong vocab (bin ngoài `[0,1023]`, enum lạ
như `SIDEWAYS`) → map thẳng về `<unk>`, KHÔNG suy diễn lại thành token hợp lệ gần nhất. Chỉ 1 tầng
duy nhất — Parser (`app/lang/parser.py`, panic-mode recovery) — chịu trách nhiệm khoan dung lỗi cấu
trúc. Đã kiểm chứng bằng `demos/tokenizer_demo.py` (input rác, bin ngoài vocab) — không crash, `<unk>`
xuất hiện đúng chỗ, round-trip encode→decode trung thực 100% trên mẫu hợp lệ.

---

## 6. Pitfall quan trọng nhất — GRPO tokenizer quirk (bug thật đã gặp)

**Triệu chứng**: reward log phẳng lì bất thường trong lúc GRPO training, model có xu hướng sinh
completion cực ngắn.

**Nguyên nhân gốc**: `GRPOTrainer` (TRL) tokenize prompt lúc rollout bằng
`self.processing_class(text=prompts)["input_ids"]` — **không truyền `add_special_tokens`** (default
`True`). Post-processor của tokenizer (mục 3) bọc CẢ `<bos>` lẫn `<eos>` quanh prompt — model thấy 1
`<eos>` giữa `<chart>` và `<think>`, sai hoàn toàn so với convention lúc SFT (`<eos>` chỉ nên đứng
cuối CẢ sequence).

**Fix** (trong `train_grpo.py`, ngay sau `load_tokenizer()`):
```python
tok.add_eos_token = False   # tự rebuild post_processor thành "chỉ <bos>, không <eos>"
tok.add_bos_token = True
```

Không có cách nào khác chỉnh `add_special_tokens` từ ngoài vì TRL gọi cố định như trên — đây là
mutation BẮT BUỘC, không phải tuỳ chọn.

**Đánh đổi cần biết**: mutation này VĨNH VIỄN trên object `tok` (thay đổi `_tokenizer.post_processor`
tại chỗ). `Trainer` tự động lưu `processing_class` (= `tok` đã mutate) vào MỌI checkpoint theo
`save_steps`, nên `tokenizer.json` trong các checkpoint GRPO từ giờ khác bản canonical trên Hub
(thiếu eos-wrap). Không ảnh hưởng pipeline của chính hệ thống này (mọi lần chạy lại đều
`load_tokenizer()` thẳng từ Hub, không đọc lại tokenizer từ checkpoint local) — chỉ ảnh hưởng nếu ai
đó load riêng tokenizer từ 1 checkpoint GRPO cụ thể cho việc khác.

**Vá artifact cuối cùng**: trước khi push/save bản "final", load lại 1 bản tokenizer CANONICAL fresh
(KHÔNG dùng `tok` đã mutate) để lưu:
```python
canonical_tok = load_tokenizer(repo_id=args.repo_id, allow_local_fallback=False)
canonical_tok.save_pretrained(args.output_dir)   # KHÔNG dùng tok.save_pretrained(...)
```

---

## 7. Demo / kiểm chứng nhanh

```bash
python -m demos.tokenizer_demo
```

Kiểm tra: (1) round-trip trung thực trên mẫu hợp lệ (generator thật), (2) seq_len nằm trong ngân
sách `max_position_embeddings=512` (đo thật: min/max/avg ≈ 216/235/229 token), (3) completion rác
không crash, (4) digit-decompose tách đúng từng token riêng (không bị BPE-style gộp).