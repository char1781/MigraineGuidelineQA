"""
rebuild_chroma_for_eval.py

仅重建 Chroma 语义向量库，不触发 ES 入库、不触发 GraphRAG、不写审核记录。
用途：对比表格清洗前后的语义检索效果差异。

使用：
    rm -rf data/chroma_db
    python rebuild_chroma_for_eval.py        # 自动扫描 data/processed/*.md
    python rebuild_chroma_for_eval.py --file data/processed/full.md
"""

import sys, shutil, os, json, re
from pathlib import Path

# ── 路径处理 ──
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("CHROMA_PATH", "data/chroma_db")
os.environ.setdefault("CHROMA_COLLECTION", "medical_guidelines")


def convert_content_lists(proc: Path):
    """
    把 MinerU 的 *_content_list.json 转成 md。
    之前库里只有月经性偏头痛一篇（且 full.md 是它的重复），
    其余 7 篇指南只有 content_list.json，从未进入向量库。
    """
    for cl in sorted(proc.glob("*_content_list.json")):
        if cl.stem.endswith("_v2"):      # v1 是扁平结构，v2 是分页嵌套，用 v1
            continue
        uuid = cl.name.split("_content_list")[0]
        if any(proc.glob(f"{uuid}_*.md")):
            continue                      # 已有 md 就跳过

        blocks = json.loads(cl.read_text(encoding="utf-8"))
        title = next((b.get("text") for b in blocks
                      if b.get("type") == "text" and b.get("text_level") == 1), None)
        if not title or len(title) > 40:
            title = next((b.get("text") for b in blocks
                          if b.get("type") == "text" and b.get("text")), uuid)
        title = re.sub(r'[*/\s]+', '', title)[:40]

        lines = [f"# {title}", ""]
        for b in blocks:
            t = b.get("type")
            if t == "text":
                text = (b.get("text") or "").strip()
                if not text:
                    continue
                lvl = b.get("text_level") or 0
                lines.append(("## " + text) if lvl == 2 else text)
                lines.append("")
            elif t == "table":
                body = b.get("table_body") or ""
                if not body:
                    continue
                cap = " ".join(b.get("table_caption") or [])
                if cap:
                    lines.append(f"**{cap}**")
                lines.append(body)
                lines.append("")
            elif t == "equation" and b.get("text"):
                lines.append(b["text"])
                lines.append("")

        md = proc / f"{uuid}_{title}.md"
        md.write_text("\n".join(lines), encoding="utf-8")
        print(f"   ✅ 生成 {md.name}（{len(''.join(lines))} 字符）")


def main():
    from src.retrieval.vector_retriever import VectorRetriever

    proc = project_root / "data" / "processed"

    # 0. 补齐缺失的 md
    print("📝 检查 content_list.json → md ...")
    convert_content_lists(proc)

    # 1. 找到要索引的 md 文件
    #    排除重复项：
    #    - full.md 与 c39d45c7…_月经性…（2025版）.md 内容相同，且文件名无信息量
    #    - 1dc7f19e…_月经性….md 文件名错标，正文其实是《中国偏头痛中西医结合防治指南（2022年）》，
    #      与 b21a7b71…_中国偏头痛中西医结合防治指南（2022年）.md 重复
    SKIP = {
        "full.md",
        "1dc7f19edbb74a588b5458541a73a1e8_月经性偏头痛诊断和治疗中国专家共识2025版.md",
    }
    md_files = []
    if len(sys.argv) > 2 and sys.argv[1] == "--file":
        p = Path(sys.argv[2])
        if p.exists():
            md_files.append(p)
    else:
        md_files = sorted(f for f in proc.glob("*.md") if f.name not in SKIP)

    if not md_files:
        print("❌ 没找到 md 文件")
        return 1

    print(f"📄 将索引 {len(md_files)} 个 md 文件：")
    for f in md_files:
        print(f"   - {f.name}")

    # 2. 清掉旧库（如果还在）
    chroma_path = os.getenv("CHROMA_PATH", "data/chroma_db")
    if Path(chroma_path).exists():
        print(f"\n🧹 清除旧 Chroma 库: {chroma_path}")
        shutil.rmtree(chroma_path)

    # 3. 逐个构建并合并到同一个 collection
    print(f"\n🔨 开始构建 Chroma 索引（使用新版表格清洗逻辑）...")
    for i, md_file in enumerate(md_files, 1):
        print(f"\n── [{i}/{len(md_files)}] {md_file.name} ──")
        try:
            vr = VectorRetriever(
                md_file_path=str(md_file),
                chroma_path=chroma_path,
                collection_name=os.getenv("CHROMA_COLLECTION", "medical_guidelines"),
            )
            print(f"   ✅ nodes: {len(vr.nodes)}")
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            import traceback; traceback.print_exc()

    print(f"\n✅ 完成！索引路径: {chroma_path}")
    print(f"\n测试查询示例：")
    vr = VectorRetriever(chroma_path=chroma_path, collection_name=os.getenv("CHROMA_COLLECTION", "medical_guidelines"))
    for q in [
        "月经性偏头痛的诊断标准",
        "曲坦类药物有哪些禁忌",
        "预防偏头痛有哪些方法",
    ]:
        results = vr.search(q, top_k=2)
        print(f"\n  Q: {q}")
        for r in results:
            text = r.text if hasattr(r, 'text') else str(r)
            meta = r.metadata if hasattr(r, 'metadata') else {}
            fname = meta.get('file_name', '?')
            print(f"    → [{fname}] {text[:120].replace(chr(10),' ')}...")


if __name__ == "__main__":
    sys.exit(main() or 0)
