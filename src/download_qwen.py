from huggingface_hub import snapshot_download

# Qwen3-4B là tên đúng trên HuggingFace (không phải Qwen3.5-4B)
MODEL_NAME = "Qwen/Qwen3-4B"
LOCAL_DIR = "/content/naive_rag/models/qwen3-4b"

snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False
)

print(f"Đã tải model về: {LOCAL_DIR}")