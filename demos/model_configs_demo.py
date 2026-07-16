"""
model_configs_demo.py — Kiểm chứng nhanh cho model_configs.py — chạy:
    python -m app.model.model_configs_demo

Kiểm tra:
1. Mọi preset build được `LlamaConfig` thật (transformers) không lỗi,
   field khớp đúng bảng mục 1.2 train_pipeline_v0.1.md.
2. GQA constraint (num_key_value_heads < num_attention_heads, chia hết)
   được ép ngay ở `ModelArgs.__post_init__` — thử 1 preset cố ý sai để
   xác nhận có raise, không âm thầm bỏ qua.
3. So sánh số tham số tính ra (dùng vocab_size THẬT từ tokenizer đã build)
   với bảng ước lượng cũ trong doc — bảng cũ ghi rõ là ước lượng "phần
   transformer block, CHƯA cộng embedding" và chưa biết vocab_size cuối
   cùng, nên số ở đây (tính đủ, có embedding, vocab thật) là số chính xác
   hơn, KHÔNG kỳ vọng khớp tuyệt đối với bảng cũ.
4. Thử khởi tạo model thật (`build_model`) NẾU môi trường có torch — bỏ
   qua có thông báo rõ ràng nếu không có (vd máy chỉ chuẩn bị data).
"""
from __future__ import annotations

from app.training.model.configs import (
    MAX_POSITION_EMBEDDINGS,
    MODEL_PRESETS,
    ModelArgs,
    build_llama_config,
    build_model,
    estimate_param_count,
)

# Bảng ước lượng cũ (mục 1.2 train_pipeline_v0.1.md) — CHỈ để đối chiếu tham
# khảo, không phải ground truth (doc cũ tự ghi rõ là ước lượng phần
# transformer block, chưa cộng embedding, viết trước khi biết vocab_size).
OLD_DOC_ESTIMATE_RANGE_M = {
    "tiny": (2, 3),
    "small": (12, 15),
    "base": (45, 55),
    "large": (110, 130),
}


def run() -> None:
    from app.tokenizer.hub import load_tokenizer
    tok = load_tokenizer()
    vocab_size = tok.vocab_size

    print(f"vocab_size (từ app.tokenizer.hub.load_tokenizer(), tức HF Hub) = {vocab_size}")
    print(f"max_position_embeddings = {MAX_POSITION_EMBEDDINGS}\n")

    # ------------------------------------------------------------
    # 1) Build LlamaConfig thật cho từng preset
    # ------------------------------------------------------------
    print("=== 1) Build LlamaConfig thật cho 4 preset ===")
    for name in MODEL_PRESETS:
        cfg = build_llama_config(name, vocab_size)
        assert cfg.vocab_size == vocab_size
        assert cfg.max_position_embeddings == MAX_POSITION_EMBEDDINGS
        assert cfg.pad_token_id == 3 and cfg.bos_token_id == 1 and cfg.eos_token_id == 2
        assert cfg.tie_word_embeddings is True
        print(f"  [{name}] hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
              f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
              f"inter={cfg.intermediate_size}  -> LlamaConfig OK")
    print("  -> PASS: mọi preset build LlamaConfig hợp lệ, field khớp quy ước.\n")

    # ------------------------------------------------------------
    # 2) GQA constraint phải raise khi vi phạm
    # ------------------------------------------------------------
    print("=== 2) GQA constraint: cố ý truyền kv_heads sai để xác nhận raise ===")
    try:
        ModelArgs(hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                   num_key_value_heads=4, intermediate_size=512)  # kv_heads == heads -> không phải GQA
        raise AssertionError("Lẽ ra phải raise ValueError khi num_key_value_heads == num_attention_heads")
    except ValueError as e:
        print(f"  Raise đúng như kỳ vọng: {e}")
    try:
        ModelArgs(hidden_size=130, num_hidden_layers=4, num_attention_heads=4,
                   num_key_value_heads=2, intermediate_size=512)  # 130 không chia hết cho 4
        raise AssertionError("Lẽ ra phải raise ValueError khi hidden_size không chia hết cho num_attention_heads")
    except ValueError as e:
        print(f"  Raise đúng như kỳ vọng: {e}")
    print("  -> PASS: constraint được ép cứng, không âm thầm bỏ qua.\n")

    # ------------------------------------------------------------
    # 3) So sánh param count
    # ------------------------------------------------------------
    print("=== 3) Số tham số (tính đủ, có embedding, vocab thật) vs bảng ước lượng cũ ===")
    print(f"{'preset':<8} {'params tính đủ':<18} {'bảng cũ (block-only, ước lượng)':<35}")
    for name in MODEL_PRESETS:
        n_params = estimate_param_count(name, vocab_size)
        lo, hi = OLD_DOC_ESTIMATE_RANGE_M[name]
        print(f"{name:<8} {n_params/1e6:<18.1f} ~{lo}-{hi}M (không cộng embedding, viết trước khi có vocab thật)")
    print(
        "\n  Lưu ý: số 'tính đủ' ở đây CHÍNH XÁC hơn bảng cũ (bảng cũ tự ghi là ước lượng\n"
        "  phần transformer block, chưa cộng embedding, viết trước khi tokenizer tồn tại).\n"
        "  Dùng số ở cột trái làm chuẩn từ giờ, không dùng lại bảng cũ trong doc gốc.\n"
    )

    # ------------------------------------------------------------
    # 4) Thử khởi tạo model thật nếu có torch
    # ------------------------------------------------------------
    print("=== 4) Khởi tạo model thật (cần torch) ===")
    try:
        model = build_model("tiny", vocab_size)
        real_params = sum(p.numel() for p in model.parameters())
        print(f"  Khởi tạo model 'tiny' thành công — số tham số thật = {real_params/1e6:.2f}M "
              f"(so với ước lượng {estimate_param_count('tiny', vocab_size)/1e6:.2f}M)")
    except ImportError as e:
        print(f"  [SKIP] {e}")

    print("\nHoàn tất.")


if __name__ == "__main__":
    run()