"""
src/config.py
职责：全局配置，包括LLM和Embedding模型设置

这个文件会在任何模块导入时自动执行，确保Settings正确配置
使用方法：在需要使用LlamaIndex的模块顶部 import src.config
"""

import os
from pathlib import Path

import nltk
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# 加载环境变量
load_dotenv()

# 让所有 LlamaIndex / NLTK 依赖都使用本地 ~/.nltk_data，避免权限问题导致的 stopwords 缺失
nltk_data_dir = Path.home() / "nltk_data"
nltk_data_dir.mkdir(parents=True, exist_ok=True)
os.chmod(nltk_data_dir, 0o700)
os.environ.setdefault("NLTK_DATA", str(nltk_data_dir))
if str(nltk_data_dir) not in nltk.data.path:
    nltk.data.path.insert(0, str(nltk_data_dir))

for resource_name, finder_name in [
    ("stopwords", "corpora/stopwords"),
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
]:
    try:
        nltk.data.find(finder_name)
    except LookupError:
        nltk.download(resource_name, download_dir=str(nltk_data_dir), quiet=True)

# ═══════════════════════════════════════════════
# 全局Embedding模型配置
# ═══════════════════════════════════════════════
Settings.embed_model = OllamaEmbedding(
    model_name="bge-m3",
    base_url="http://localhost:11434",
    request_timeout=120.0,
)

# ═══════════════════════════════════════════════
# 全局LLM模型配置
# ═══════════════════════════════════════════════
Settings.llm = Ollama(
    model="qwen2.5:14b-instruct-q4_K_M",
    base_url="http://localhost:11434",
    request_timeout=120.0,
    temperature=0,
)
# ═══════════════════════════════════════════════
# 其他全局配置
# ═══════════════════════════════════════════════
# 设置chunk大小（如果需要覆盖默认值）
# Settings.chunk_size = 512
# Settings.chunk_overlap = 50

# 设置context window（如果需要）
# Settings.context_window = 4096

print("[Config] ✅ 全局Settings已配置")
print(f"  Embedding: {Settings.embed_model.__class__.__name__}")
print(f"  LLM: {Settings.llm.__class__.__name__}")
