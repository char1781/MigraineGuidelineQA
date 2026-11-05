"""
src/utils/graphrag_env.py
GraphRAG 集成的公共配置和索引状态检查。

只依赖 rag0 环境里已有的包（python-dotenv、pyarrow），不 import graphrag。
GraphRAG 装在独立的 graphrag311 环境里，rag0 通过 subprocess 调用它。

rag0/.env 需要的配置：
  GRAPHRAG_ROOT=/绝对路径/graphrag工作目录          # 含 settings.yaml、input/、output/
  GRAPHRAG_PYTHON=/绝对路径/graphrag311环境的python   # conda activate graphrag311 后执行 which python
  GRAPHRAG_OUTPUT_DIR=output        # 可选，与 settings.yaml 里的输出目录保持一致
  GRAPHRAG_QUERY_TIMEOUT=300        # 可选，单次查询超时（秒）
  GRAPHRAG_QUERY_EXTRA=             # 可选，附加给 query 的参数，如 "--community-level 2"
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()


def get_root() -> Path:
    root = os.getenv("GRAPHRAG_ROOT")
    if not root:
        raise RuntimeError("未设置 GRAPHRAG_ROOT，请在 rag0/.env 中配置 GraphRAG 工作目录")
    p = Path(root).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"GRAPHRAG_ROOT 不存在: {p}")
    return p


def get_python() -> str:
    py = os.getenv("GRAPHRAG_PYTHON")
    if not py:
        raise RuntimeError(
            "未设置 GRAPHRAG_PYTHON。请执行 conda activate graphrag311 && which python，"
            "把结果写进 rag0/.env"
        )
    p = Path(py).expanduser()  # 不要 resolve()，避免符号链接丢失环境信息
    if not p.exists():
        raise RuntimeError(f"GRAPHRAG_PYTHON 不存在: {p}")
    return str(p)


def get_output_dir(root: Optional[Path] = None) -> Path:
    root = Path(root) if root else get_root()
    return root / os.getenv("GRAPHRAG_OUTPUT_DIR", "output")


def _latest(root: Optional[Path], filename: str) -> Optional[Path]:
    """在输出目录下找最新的某个 parquet（兼容新旧版本的目录结构）"""
    out = get_output_dir(root)
    if not out.exists():
        return None
    files = list(out.rglob(filename))
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def index_ready(root: Optional[Path] = None) -> bool:
    """索引是否可用于查询（至少有实体和关系两张表）"""
    return all(_latest(root, f) is not None for f in ("entities.parquet", "relationships.parquet"))


def index_stats(root: Optional[Path] = None) -> Dict:
    """读取实体/关系/社区数量（只读 parquet 元数据，很快）"""
    stats = {"entities": 0, "relationships": 0, "communities": 0, "updated_at": None}
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return stats

    newest = 0.0
    for key, fname in (
        ("entities", "entities.parquet"),
        ("relationships", "relationships.parquet"),
        ("communities", "communities.parquet"),
    ):
        p = _latest(root, fname)
        if p is None:
            continue
        try:
            stats[key] = pq.ParquetFile(p).metadata.num_rows
        except Exception:
            pass
        newest = max(newest, p.stat().st_mtime)
    if newest:
        stats["updated_at"] = datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M")
    return stats
