"""
src/ingestion/graph_builder.py
职责：把 rag0 已经解析好的 md 交给 GraphRAG 建图谱（不再调用 MinerU）。

做法：
1. add_markdown(): 把 md 复制到 GraphRAG 工作目录的 input/
2. start_index(): 用 graphrag311 环境的 python 在后台执行 `python -m graphrag index --root <root>`
3. get_status(): 读取状态文件（Streamlit 每次 rerun 都会丢变量，所以状态落盘）

注意：GRAPHRAG_ROOT 必须是 "md 输入模式" 的工作目录（settings.yaml 里 file_type: text），
不能是仍然配置为 pdf 输入的 graphrag-main，否则会混入 MinerU 流程。

命令行测试：
  python -m src.ingestion.graph_builder status
  python -m src.ingestion.graph_builder add /path/x.md --name 中国偏头痛诊治指南 --year 2022
  python -m src.ingestion.graph_builder start
  python -m src.ingestion.graph_builder tail -n 40
"""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    _project_root = Path(__file__).parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

from src.utils.graphrag_env import (
    get_output_dir,
    get_python,
    get_root,
    index_ready,
    index_stats,
)

log = logging.getLogger(__name__)
_TS = "%Y-%m-%d %H:%M:%S"


def _now() -> str:
    return datetime.now().strftime(_TS)


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        import psutil

        try:
            p = psutil.Process(pid)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


class GraphBuilder:
    """GraphRAG 图谱构建器（后台进程 + 状态文件）"""

    def __init__(self, root: Optional[str] = None, python: Optional[str] = None):
        self.root = Path(root).expanduser().resolve() if root else get_root()
        self.python = python or get_python()

        self.input_dir = self.root / "input"
        self.state_dir = self.root / ".rag0_state"
        self.status_file = self.state_dir / "index_status.json"
        self.exit_file = self.state_dir / "index_exit_code"
        self.log_file = self.state_dir / "index.log"

        self._guard_input_mode()
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # ── 安全检查 ──────────────────────────────────────────
    def _guard_input_mode(self):
        settings = self.root / "settings.yaml"
        if settings.exists():
            text = settings.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"^\s*file_type:\s*pdf\b", text, flags=re.M):
                raise RuntimeError(
                    f"{settings} 仍是 pdf 输入模式（file_type: pdf）。"
                    "请为 rag0 单独建一个 md 输入的 GraphRAG 工作目录，"
                    "把 file_type 改为 text、file_pattern 改为 .*\\.md$"
                )

    # ── 文档管理 ──────────────────────────────────────────
    @staticmethod
    def doc_filename(guideline_name: str, year=None) -> str:
        name = re.sub(r'[\\/:*?"<>|\s]+', "_", str(guideline_name)).strip("_") or "doc"
        return f"{name}_{year}.md" if year else f"{name}.md"

    def list_documents(self) -> List[Path]:
        return sorted(self.input_dir.glob("*.md"))

    def add_markdown(self, md_path, guideline_name: Optional[str] = None, year=None) -> Path:
        """复制 md 到 GraphRAG input/，避免覆盖历史文档，并支持增量索引。"""
        md_path = Path(md_path)
        if not md_path.exists():
            raise FileNotFoundError(f"md 文件不存在: {md_path}")

        base_target = self.input_dir / self.doc_filename(guideline_name or md_path.stem, year)
        target = base_target
        if target.exists():
            if target.read_bytes() == md_path.read_bytes():
                log.info(f"[GraphBuilder] input 中已存在相同文档，跳过复制: {target.name}")
                return target
            stem = target.stem
            suffix = 2
            while True:
                candidate = self.input_dir / f"{stem}_{suffix}{target.suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                suffix += 1

        shutil.copyfile(md_path, target)
        log.info(f"[GraphBuilder] 已加入 input: {target.name}")
        return target

    def remove_markdown(self, guideline_name: str, year=None, rebuild: bool = True) -> Dict:
        """删除某篇指南。GraphRAG 不支持按文档删除，需要用剩余文档整体重建"""
        target = self.input_dir / self.doc_filename(guideline_name, year)
        if not target.exists():
            return {"removed": False, "message": f"input 中没有 {target.name}"}
        target.unlink()

        if not self.list_documents():
            out = get_output_dir(self.root)
            if out.exists():
                shutil.rmtree(out)
            return {"removed": True, "message": "已无文档，图谱索引已清空"}
        if rebuild:
            self.start_index()
            return {"removed": True, "message": "已删除，已触发重建（或排队等待重建）"}
        return {"removed": True, "message": "已删除，需手动重建图谱"}

    # ── 状态文件 ──────────────────────────────────────────
    def _read(self) -> Dict:
        if self.status_file.exists():
            try:
                return json.loads(self.status_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"state": "idle", "dirty": False}

    def _write(self, st: Dict):
        self.status_file.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

    def _refresh(self) -> Dict:
        """根据退出码文件 / 进程存活情况更新状态"""
        st = self._read()
        if st.get("state") != "running":
            return st

        if self.exit_file.exists():  # 先看退出码文件，避免僵尸进程被误判为仍在运行
            try:
                code = int(self.exit_file.read_text().strip())
            except ValueError:
                code = -1
            st["exit_code"] = code
            st["state"] = "done" if code == 0 else "failed"
            st["finished_at"] = _now()
            self._write(st)
        elif not _pid_alive(st.get("pid")):
            st["state"] = "failed"
            st["error"] = "索引进程意外退出（可能被中断），请查看日志"
            st["finished_at"] = _now()
            self._write(st)
        return st

    # ── 构建 ──────────────────────────────────────────────
    def _launch(self) -> Dict:
        if self.exit_file.exists():
            self.exit_file.unlink()

        out_dir = get_output_dir(self.root)
        has_documents_storage = (out_dir / "documents.parquet").exists()
        has_any_parquet = out_dir.exists() and any(out_dir.rglob("*.parquet"))

        # GraphRAG 的 standard-update 要求 previous_storage 中必须存在 documents.parquet。
        # 如果 output 里有别的 parquet，但缺少 documents.parquet，说明上次索引状态残缺，
        # 不能继续增量更新；必须先清理 output，再按首次标准索引重建。这样不会覆盖 input。
        if has_any_parquet and not has_documents_storage:
            log.warning(
                "[GraphBuilder] output 中存在 parquet，但缺少 documents.parquet；上次索引状态残缺，"
                "先清理 output 并回退到 standard 首次重建，不覆盖 input 中的历史文档。"
            )
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            has_documents_storage = False

        # 额外保护：如果 output 里已经有 documents.parquet，则才允许走 standard-update；
        # 否则必须强制 standard，保证不会触发 `Could not find documents.parquet in storage!`。 
        method = "standard-update" if has_documents_storage else "standard"
        if has_documents_storage:
            log.info("[GraphBuilder] 使用增量更新模式: standard-update")
        else:
            log.info("[GraphBuilder] 使用标准重建模式: standard")

        cmd = (
            f"{shlex.quote(self.python)} -m graphrag index --root {shlex.quote(str(self.root))}"
            f" --method {method}"
            f" > {shlex.quote(str(self.log_file))} 2>&1; "
            f"status=$?; echo $status > {shlex.quote(str(self.exit_file))}; exit $status"
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            cwd=str(self.root),
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # 脱离 Streamlit 进程，页面刷新/重启不影响索引
        )
        st = {
            "state": "running",
            "pid": proc.pid,
            "started_at": _now(),
            "dirty": False,
            "docs": len(self.list_documents()),
        }
        self._write(st)
        log.info(f"[GraphBuilder] 索引已在后台启动 pid={proc.pid}")
        return st

    def start_index(self) -> Dict:
        """启动后台索引。已有任务在跑时，标记 dirty，等它结束后自动再跑一次"""
        st = self._refresh()
        if st.get("state") == "running":
            st["dirty"] = True
            self._write(st)
            return {**st, "message": "已有索引任务在运行，结束后将自动再跑一次"}
        if not self.list_documents():
            return {"state": "idle", "message": "input/ 目录里没有 md 文件"}
        return self._launch()

    def get_status(self) -> Dict:
        st = self._refresh()
        if st.get("state") in ("done", "failed") and st.get("dirty"):
            st = self._launch()
        return {
            **st,
            "index_ready": index_ready(self.root) and st.get("state") != "running",
            "documents": [p.name for p in self.list_documents()],
            "stats": index_stats(self.root),
        }

    def is_building(self) -> bool:
        return self.get_status().get("state") == "running"

    def tail_log(self, n: int = 30) -> str:
        if not self.log_file.exists():
            return "(暂无日志)"
        lines = self.log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n:])


# ── 命令行测试 ──────────────────────────────────────────
def _main():
    import argparse

    ap = argparse.ArgumentParser(description="GraphRAG 构建器命令行测试")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("start")
    p_add = sub.add_parser("add")
    p_add.add_argument("md")
    p_add.add_argument("--name")
    p_add.add_argument("--year")
    p_tail = sub.add_parser("tail")
    p_tail.add_argument("-n", type=int, default=30)
    p_rm = sub.add_parser("remove")
    p_rm.add_argument("name")
    p_rm.add_argument("--year")
    args = ap.parse_args()

    gb = GraphBuilder()
    if args.cmd == "status":
        print(json.dumps(gb.get_status(), ensure_ascii=False, indent=2))
    elif args.cmd == "start":
        print(json.dumps(gb.start_index(), ensure_ascii=False, indent=2))
    elif args.cmd == "add":
        print("已复制到:", gb.add_markdown(args.md, args.name, args.year))
    elif args.cmd == "tail":
        print(gb.tail_log(args.n))
    elif args.cmd == "remove":
        print(json.dumps(gb.remove_markdown(args.name, args.year), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
