from huggingface_hub import snapshot_download

local_dir = "./models/bge-reranker-base"
snapshot_download(
    repo_id="BAAI/bge-reranker-base",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)