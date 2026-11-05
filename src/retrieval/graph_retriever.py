"""
src/retrieval/graph_retriever.py
职责：GraphRAG 图谱检索。通过 subprocess 调用 graphrag311 环境里的
      `python -m graphrag query --root <root> --method local|global --query "..."`

为什么用 subprocess 而不是直接 import graphrag：
- rag0 环境（langchain 1.x）和 graphrag311 环境的依赖版本互相冲突，隔离最稳
- 代价：每次查询都要启动一次 Python 并加载索引，比常驻服务慢几秒

保持与原占位实现一致的接口（agent 侧不用改）：
  GraphRetriever.search(query, top_k) -> List[Dict]
  GraphRetriever.format_context(results) -> str

测试：
  python -m src.retrieval.graph_retriever "癫痫共病偏头痛如何用药"
"""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    _project_root = Path(__file__).parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

import logging
import os
import re
import shlex
import subprocess
from typing import Dict, List, Optional

from src.utils.graphrag_env import get_python, get_root, index_ready

log = logging.getLogger(__name__)

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_LOG_LINE = re.compile(r"^\s*(SUCCESS|INFO|WARNING|DEBUG)\b[:\s]")


class GraphRetriever:
    """GraphRAG 检索器"""

    # 含这些词的问题走 global（整体性、总结性），其余走 local（围绕具体实体）
    GLOBAL_HINTS = ("总结", "概述", "综述", "概括", "整体", "总体", "所有", "全部", "整个指南", "主要观点")

    def __init__(
        self,
        root: Optional[str] = None,
        python: Optional[str] = None,
        timeout: Optional[int] = None,
        default_method: str = "local",
    ):
        self.root = Path(root).expanduser().resolve() if root else get_root()
        self.python = python or get_python()
        self.timeout = int(timeout or os.getenv("GRAPHRAG_QUERY_TIMEOUT", "300"))
        self.default_method = default_method

    # ── 辅助 ──────────────────────────────────────────────
    def is_ready(self) -> bool:
        return index_ready(self.root)

    def choose_method(self, query: str) -> str:
        return "global" if any(k in query for k in self.GLOBAL_HINTS) else self.default_method

    @staticmethod
    def _clean(stdout: str) -> str:
        text = _ANSI.sub("", stdout)
        lines = [ln for ln in text.splitlines() if not _LOG_LINE.match(ln)]
        return "\n".join(lines).strip()

    @staticmethod
    def _error(message: str) -> List[Dict]:
        return [{"type": "graph_error", "message": message, "entities": [], "relations": [], "score": 0.0}]

    # ── 检索 ──────────────────────────────────────────────
    def search(self, query: str, top_k: int = 5, method: Optional[str] = None) -> List[Dict]:
        """
        执行 GraphRAG 查询。top_k 仅为兼容旧接口，GraphRAG 自己决定检索范围。
        返回单元素列表：{"type": "graph_answer", "method", "message"(答案正文), ...}
        失败时返回 {"type": "graph_error", ...}，不抛异常，避免拖垮整个 agent。
        """
        if not self.is_ready():
            try:
                from src.ingestion.graph_builder import GraphBuilder
                gb = GraphBuilder(root=str(self.root), python=self.python)
                if gb.list_documents():
                    started = gb.start_index()
                    if started.get("state") == "running":
                        return [{
                            "type": "graph_status",
                            "method": method or self.choose_method(query),
                            "message": "图谱索引正在后台构建中，请稍后再试；已自动启动 GraphRAG 索引。",
                            "entities": [],
                            "relations": [],
                            "score": 0.0,
                        }]
            except Exception as e:
                log.warning(f"[GraphRetriever] 自动触发索引失败: {e}")
            return self._error("图谱索引尚未构建完成（或正在构建中），已自动启动后台索引，请稍后再试，或改用 ES/混合检索。")

        method = method or self.choose_method(query)
        cmd = [self.python, "-m", "graphrag", "query", "--root", str(self.root), "--method", method, "--query", query]
        extra = os.getenv("GRAPHRAG_QUERY_EXTRA", "").strip()
        if extra:
            cmd[-2:-2] = shlex.split(extra)  # 插在 --query 之前

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONIOENCODING"] = "utf-8"

        log.info(f"[GraphRetriever] method={method} query={query[:40]}")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.root),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return self._error(f"图谱查询超时（>{self.timeout}s），请检查 Ollama 是否在运行。")
        except Exception as e:
            return self._error(f"图谱查询无法启动：{e}")

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-500:]
            log.error(f"[GraphRetriever] 查询失败: {tail}")
            return self._error(f"图谱查询失败：{tail}")

        answer = self._clean(proc.stdout)
        if not answer:
            return self._error("图谱查询没有返回内容。")

        return [{"type": "graph_answer", "method": method, "message": answer, "entities": [], "relations": [], "score": 1.0}]

    def format_context(self, results: List[Dict]) -> str:
        """格式化为给 agent / 前端的文本"""
        if not results:
            return "未找到图谱关联信息"
        parts = []
        for i, r in enumerate(results, 1):
            if r.get("type") == "graph_answer":
                parts.append(f"【图谱检索·{r.get('method', '')}】\n{r['message']}")
            else:
                parts.append(f"【图谱检索不可用】{r.get('message', '')}")
        return "\n\n".join(parts)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "偏头痛急性期有哪些治疗药物？"
    r = GraphRetriever()
    print(f"root={r.root}\nready={r.is_ready()}\nmethod={r.choose_method(q)}\n")
    print(r.format_context(r.search(q)))
