from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="sullivan1502/base-grpo-test",
    filename="round1_stats_rank0.json",
)
print(path)  # vd: ~/.cache/huggingface/hub/models--sullivan1502--base-grpo-test/snapshots/.../round1_stats_rank0.json