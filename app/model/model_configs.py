"""
model_configs.py — Preset kiến trúc model (Llama-style) cho Trading Reasoning
LLM, theo docs/train_pipeline_v0.1.md mục 1.

Thiết kế dạng dict-of-dataclass (`MODEL_PRESETS`), KHÔNG hard-code trong
script train — chọn preset qua CLI `--model_size {tiny,small,base,large}`
(xem `model_configs_demo.py` cho ví dụ CLI).

Vocab_size KHÔNG cố định ở đây — luôn lấy từ tokenizer thật đã build
(`app/tokenizer/build_tokenizer.py:build_fast_tokenizer().vocab_size`), để
đổi vocab (vd đổi BIN_MAX) không cần sửa file này.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# =====================================================================
# Special token id — khớp app/tokenizer/vocab_builder.py
# (SPECIAL_TOKENS_IN_ID_ORDER = [<unk>, <bos>, <eos>, <pad>] -> id 0,1,2,3)
# và đúng quy ước đã chốt trong train_pipeline_v0.1.md mục 1.1.
# =====================================================================
PAD_TOKEN_ID = 3
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2

# Chốt cứng theo spec — seq_len thực tế ~230-240 token/sample, xem
# docs/tokenizer_v0.1.md mục 7.3 (đo thật: min=216 max=235 avg=229).
MAX_POSITION_EMBEDDINGS = 512


@dataclass(frozen=True)
class ModelArgs:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int   # GQA — luôn < num_attention_heads (seq_len ngắn, không cần MHA đầy đủ)
    intermediate_size: int

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) phải chia hết cho "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) phải chia hết cho "
                f"num_key_value_heads ({self.num_key_value_heads}) — điều kiện GQA"
            )
        if self.num_key_value_heads >= self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) phải NHỎ HƠN "
                f"num_attention_heads ({self.num_attention_heads}) — nếu không thì là MHA "
                f"thường, không phải GQA (xem lý do dùng GQA ở docstring module)"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# =====================================================================
# 4 preset — bảng mục 1.2 train_pipeline_v0.1.md. GQA áp dụng mọi preset
# vì seq_len ngắn (<=512) không cần MHA đầy đủ.
# =====================================================================
MODEL_PRESETS: Dict[str, ModelArgs] = {
    "tiny":  ModelArgs(hidden_size=128, num_hidden_layers=4,  num_attention_heads=4,  num_key_value_heads=2, intermediate_size=512),
    "small": ModelArgs(hidden_size=256, num_hidden_layers=6,  num_attention_heads=8,  num_key_value_heads=4, intermediate_size=1024),
    "base":  ModelArgs(hidden_size=512, num_hidden_layers=8,  num_attention_heads=8,  num_key_value_heads=4, intermediate_size=2048),
    "large": ModelArgs(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4, intermediate_size=3072),
}


def resolve_model_args(model_size_or_args) -> ModelArgs:
    """Cho phép truyền tên preset (str) hoặc 1 ModelArgs tuỳ chỉnh trực tiếp —
    tiện cho việc thử nghiệm kiến trúc mới mà không cần sửa MODEL_PRESETS."""
    if isinstance(model_size_or_args, ModelArgs):
        return model_size_or_args
    if model_size_or_args not in MODEL_PRESETS:
        raise ValueError(
            f"Không có preset {model_size_or_args!r}. Chọn 1 trong "
            f"{list(MODEL_PRESETS)} hoặc truyền thẳng 1 ModelArgs tuỳ chỉnh."
        )
    return MODEL_PRESETS[model_size_or_args]


# =====================================================================
# build_llama_config — KHÔNG cần torch (transformers config class là thuần
# Python/JSON), chỉ cần torch khi thật sự khởi tạo model (build_model bên
# dưới). Tách riêng 2 hàm để CI / script chuẩn bị data có thể import
# module này mà không phải cài torch.
# =====================================================================
def build_llama_config(
    model_size_or_args,
    vocab_size: int,
    max_position_embeddings: int = MAX_POSITION_EMBEDDINGS,
    tie_word_embeddings: bool = True,
):
    from transformers import LlamaConfig  # import trễ — tránh phụ thuộc cứng lúc chỉ cần ModelArgs

    args = resolve_model_args(model_size_or_args)

    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=PAD_TOKEN_ID,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        tie_word_embeddings=tie_word_embeddings,
    )


def build_model(
    model_size_or_args,
    vocab_size: int,
    max_position_embeddings: int = MAX_POSITION_EMBEDDINGS,
    attn_implementation: str = "sdpa",
):
    """
    Khởi tạo model from-scratch (KHÔNG fine-tune checkpoint có sẵn — đúng
    quyết định trong train_pipeline_v0.1.md). Cần torch — import trễ, báo
    lỗi rõ ràng nếu môi trường chưa cài (vd máy chỉ chuẩn bị data/tokenizer).
    """
    try:
        from transformers import LlamaForCausalLM
    except ImportError as e:
        raise ImportError(
            "build_model() cần torch (transformers models). Cài bằng: "
            "pip install torch --break-system-packages. Nếu chỉ cần config "
            "(không khởi tạo model), dùng build_llama_config() thay thế."
        ) from e

    config = build_llama_config(model_size_or_args, vocab_size, max_position_embeddings)
    return LlamaForCausalLM._from_config(config, attn_implementation=attn_implementation)


# =====================================================================
# Ước lượng số tham số — THUẦN PYTHON, không cần torch. Dùng để đối
# chiếu nhanh với bảng "params (~)" ở train_pipeline_v0.1.md mục 1.2
# trước khi thật sự khởi tạo model (hữu ích trên máy chưa cài torch).
#
# Công thức chuẩn kiến trúc Llama (attention GQA + SwiGLU MLP + RMSNorm),
# KHÔNG tính embedding riêng cho lm_head vì tie_word_embeddings=True (đề
# bài chốt cứng mọi size — mục 1.1).
# =====================================================================
def estimate_param_count(model_size_or_args, vocab_size: int) -> int:
    args = resolve_model_args(model_size_or_args)
    h = args.hidden_size
    head_dim = args.head_dim
    kv_dim = args.num_key_value_heads * head_dim
    inter = args.intermediate_size

    # Attention: q_proj (h*h) + k_proj (h*kv_dim) + v_proj (h*kv_dim) + o_proj (h*h)
    attn_params_per_layer = 2 * h * h + 2 * h * kv_dim

    # MLP SwiGLU: gate_proj (h*inter) + up_proj (h*inter) + down_proj (inter*h)
    mlp_params_per_layer = 3 * h * inter

    # RMSNorm (input_layernorm + post_attention_layernorm), mỗi cái h tham số — nhỏ, cộng cho đủ.
    norm_params_per_layer = 2 * h

    per_layer = attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer
    total = args.num_hidden_layers * per_layer

    total += h * vocab_size          # embedding table (tied với lm_head, chỉ tính 1 lần)
    total += h                        # final RMSNorm

    return total


if __name__ == "__main__":
    # vocab_size mặc định dùng số THẬT đã build ở app/tokenizer/vocab_builder.py
    # (4145) — import trễ để không bắt buộc phải có sẵn lúc chỉ xem preset.
    try:
        from app.tokenizer.vocab_builder import build_vocab
        vocab_size = len(build_vocab())
    except Exception:
        vocab_size = 4145  # fallback nếu chạy độc lập không có app.tokenizer

    print(f"vocab_size = {vocab_size}\n")
    print(f"{'preset':<8} {'hidden':<8} {'layers':<8} {'heads':<8} {'kv_heads':<10} {'inter':<8} {'params(~)':<12}")
    for name, args in MODEL_PRESETS.items():
        n_params = estimate_param_count(args, vocab_size)
        print(
            f"{name:<8} {args.hidden_size:<8} {args.num_hidden_layers:<8} "
            f"{args.num_attention_heads:<8} {args.num_key_value_heads:<10} "
            f"{args.intermediate_size:<8} {n_params/1e6:.1f}M"
        )