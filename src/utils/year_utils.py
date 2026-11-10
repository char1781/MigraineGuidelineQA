# src/utils/year_utils.py


"""指南年份提取工具。

目的：让 ES 文档、KnowledgeBaseMetadata、GraphRAG 文件名三处的年份
使用同一套提取逻辑，避免出现 "ES 写 1111 / 图谱写 2024" 的不一致。

统一约定：
- 无法确定年份时返回 None，不再使用 1111 之类的哨兵值。
- 年份合理范围可通过 MIN_YEAR / MAX_YEAR 调整。

对外接口：
- extract_year(text) -> Optional[int]                      从任意文本提取
- extract_year_from_md(md_path, max_lines=30) -> Optional[int]  从 MD 头部提取
- extract_year_from_filename(filename) -> Optional[int]     从文件名提取
- resolve_guideline_year(raw_year, md_path, filename) -> Optional[int]
                                                            综合优先级解析
"""

import re
from pathlib import Path
from typing import Optional

# ── 可调参数 ─────────────────────────────────────────────────────
MIN_YEAR = 1900
MAX_YEAR = 2100

# ── 预编译正则 ──────────────────────────────────────────────────
# 严格匹配：xxxx年版 / xxxx年发布 / xxxx版 / xxxx修订
_STRICT_YEAR_RE = re.compile(
    r"(?:^|[^\d])"
    r"((?:19|20)\d{2})"
    r"(?=\s*年?\s*[版发修])"
)

# 括号中的年份：(2024) （2023）
_BRACKET_YEAR_RE = re.compile(r"[（(]\s*((?:19|20)\d{2})\s*[)）]")

# 宽松匹配：任意独立 4 位年份（前后非数字）
_LOOSE_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def _valid_year(y: int) -> bool:
    return MIN_YEAR <= y <= MAX_YEAR


def extract_year(text: Optional[str]) -> Optional[int]:
    """从任意文本中提取 4 位年份；找不到或超出范围返回 None。

    提取优先级：
    1. "xxxx年版" / "xxxx年发布" / "xxxx版" / "xxxx修订" 等版本表达
    2. 括号中的 4 位年份，如 (2024)、（2023）
    3. 裸 4 位年份（前后非数字）

    Examples:
        >>> extract_year("中国偏头痛诊治指南2022版")
        2022
        >>> extract_year("Migraine Guidelines (2024)")
        2024
        >>> extract_year("推荐布洛芬 400mg，每日 3 次")
        None
    """
    if not text:
        return None
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return None

    # 1. 严格版本表达
    m = _STRICT_YEAR_RE.search(text)
    if m:
        y = int(m.group(1))
        if _valid_year(y):
            return y

    # 2. 括号年份
    m = _BRACKET_YEAR_RE.search(text)
    if m:
        y = int(m.group(1))
        if _valid_year(y):
            return y

    # 3. 裸 4 位年份
    m = _LOOSE_YEAR_RE.search(text)
    if m:
        y = int(m.group(1))
        if _valid_year(y):
            return y

    return None


def extract_year_from_md(md_path, max_lines: int = 30) -> Optional[int]:
    """从 MD 文件头部（标题 / 前若干行）提取年份。

    优先只看第一个非空标题行，未命中再在头部全文中兜底。
    读文件失败、文件不存在、内容为空均返回 None。
    """
    if not md_path:
        return None
    p = Path(md_path)
    if not p.is_file():
        return None

    head_lines = []
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                head_lines.append(line)
    except OSError:
        return None

    if not head_lines:
        return None

    # 1. 优先第一个非空标题行
    for line in head_lines:
        stripped = line.strip().lstrip("#").strip()
        if not stripped:
            continue
        y = extract_year(stripped)
        if y is not None:
            return y
        break  # 只看第一个非空标题行，避免被正文随机年份误导

    # 2. 头部全文兜底
    return extract_year("".join(head_lines))


def extract_year_from_filename(filename: Optional[str]) -> Optional[int]:
    """从文件名（可含路径）提取年份，如 '偏头痛指南2024.pdf' → 2024。"""
    if not filename:
        return None
    try:
        stem = Path(filename).stem
    except Exception:
        stem = str(filename)
    return extract_year(stem)


def resolve_guideline_year(
    raw_year,
    md_path=None,
    filename=None,
) -> Optional[int]:
    """综合多个来源确定指南年份。

    优先级（从高到低）：
    1. raw_year（抽取器给出的值，排除 None / 1111 / 0 等哨兵）
    2. MD 文件标题 / 正文
    3. 文件名
    4. None（明确未知）

    Args:
        raw_year: 抽取器返回的年份，可能是 None / 1111 / 0 / 合法 int / 数字字符串
        md_path:  MD 文件路径
        filename: 原始文件名（可含路径）

    Returns:
        Optional[int]: 有效年份或 None
    """
    # 1. 抽取器给的（排除哨兵值）
    if raw_year not in (None, 1111, 0, "", "0", "1111"):
        try:
            y = int(raw_year)
            if _valid_year(y):
                return y
        except (TypeError, ValueError):
            pass

    # 2. MD 标题/正文
    y = extract_year_from_md(md_path)
    if y is not None:
        return y

    # 3. 文件名
    y = extract_year_from_filename(filename)
    if y is not None:
        return y

    # 4. 未知
    return None


__all__ = [
    "extract_year",
    "extract_year_from_md",
    "extract_year_from_filename",
    "resolve_guideline_year",
    "MIN_YEAR",
    "MAX_YEAR",
]