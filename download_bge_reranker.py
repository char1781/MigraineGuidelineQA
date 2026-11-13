# from huggingface_hub import snapshot_download

# local_dir = "./models/bge-reranker-base"
# snapshot_download(
#     repo_id="BAAI/bge-reranker-base",
#     local_dir=local_dir,
#     local_dir_use_symlinks=False,
#     resume_download=True
# )

import os
# 放在最顶部，镜像加速
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

local_dir = "./models/bge-reranker-v2-m3"
snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)