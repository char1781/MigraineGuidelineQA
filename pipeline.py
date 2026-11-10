"""
pipeline.py - 一键入库的统一入口（更新：增加审核环节）

新流程：
1. 用户上传PDF → MinerU解析 → 得到JSON和MD
2. JSON → ES提取 → 生成审核数据（保存到 data/review/）
3. 用户在Streamlit审核页面确认/修改/拒绝每条记录
4. 审核通过的记录 → 写入ES数据库
5. MD文件 → ChromaDB向量索引
6. MD文件 → GraphRAG抽取 

运行方式：
# 方式1：命令行（用于手动测试）
python pipeline.py --file data/raw/指南.pdf

# 方式2：通过Streamlit前端上传（推荐）
# 前端会自动调用 run_pipeline_upload()
"""

import argparse
from pathlib import Path
from typing import Dict, Optional
import os
import uuid
from dotenv import load_dotenv
from src.utils.year_utils import extract_year, extract_year_from_md, resolve_guideline_year

load_dotenv()


def _trigger_graph_index_for_md(md_path: str, guideline_name: Optional[str] = None, year: Optional[int] = None) -> Dict:
    """Always preserve GraphRAG history and trigger index in the correct incremental mode."""
    from src.ingestion.graph_builder import GraphBuilder

    md_file = Path(md_path)
    if not md_file.exists():
        return {"status": "skipped", "reason": f"MD文件不存在: {md_path}"}

    final_name = guideline_name or md_file.stem or "未知指南"

    # 年份兜底：优先外部传的有效值，否则从 MD 标题/正文提取，再退回 guideline_name
    if year in (None, 1111, 0):
            year = extract_year_from_md(md_file)
            if year is None and final_name:
                year = extract_year(final_name)
    
    gb = GraphBuilder()

    target = gb.add_markdown(
        md_file,
        guideline_name=final_name,
        year=year,
    )
    status = gb.start_index()
    return {
        "status": "ok",
        "target": str(target),
        "graph_status": status,
    }


def run_pipeline_upload(
    pdf_bytes: bytes,
    filename: str,
    skip_review: bool = False,
) -> Dict:
    """
    从Streamlit上传的入口（接收bytes）

    Args:
        pdf_bytes: PDF文件的字节内容
        filename: 文件名（浏览器上传时传来的原始名，仅用于提取指南名/审核ID，
                  不直接参与写磁盘，避免路径穿越）
        skip_review: 是否跳过审核（测试用）

    Returns:
        {
            'status': 'success' | 'pending_review' | 'no_records',
            'review_id': 'xxx',
            'json_path': 'data/processed/xxx.json',
            'md_path': 'data/processed/xxx.md',
            ...
        }
    """
    import uuid
    import os
    from src.ingestion.mineru_client import MinerUClient
    from src.ingestion.es_review_editor import ESReviewEditor
    from src.ingestion.es_indexer import MedicalGuidelineExtractor
    from src.utils.kb_metadata import KnowledgeBaseMetadata

    def _safe_guideline_name(source_name: str, fallback_name: Optional[str] = None) -> str:
        value = (fallback_name or source_name or "未知指南").strip()
        if value == "未知指南":
            value = Path(source_name).stem if source_name else "未知指南"
        return value or "未知指南"

    # ── 步骤1：保存上传的PDF到临时目录（防路径穿越 + 防并发覆盖）──
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 只取文件名最后一段，去掉任何 ../ 或目录部分；中文保留
    safe_name = os.path.basename(filename.replace("\\", "/")).strip()
    if not safe_name or safe_name in (".", ".."):
        safe_name = "upload.pdf"
    temp_pdf = temp_dir / f"{uuid.uuid4().hex}_{safe_name}"
    temp_pdf.write_bytes(pdf_bytes)

    print(f"[Pipeline] 1/6 已保存上传文件: {temp_pdf}")

    json_path = ""
    md_path = ""
    review_data: Dict = {}
    review_id = ""

    try:
        # ── 步骤2：调用MinerU解析 ──
        print(f"[Pipeline] 2/6 调用MinerU API解析...")
        client = MinerUClient()
        mineru_result = client.upload_and_wait(str(temp_pdf), timeout=300)

        json_path = mineru_result["json_path"]
        md_path = mineru_result["md_path"]

        print(f"[Pipeline] ✅ MinerU解析完成")
        print(f"  JSON: {json_path}")
        print(f"  MD:   {md_path}")

        # ── 步骤2.5：触发 GraphRAG 索引（解析阶段一次，审核后不再重复）──
        graph_index_status = "unknown"
        graph_index_reason = ""
        try:
            graph_result = _trigger_graph_index_for_md(
                md_path,
                guideline_name=Path(filename).stem,
                year=None,   # 函数内部会从 MD 标题/正文/文件名提取
            )
            graph_index_status = graph_result.get("status", "unknown")
            print(f"[Pipeline] GraphRAG input 已更新: {graph_result}")
        except Exception as e:
            graph_index_status = "error"
            graph_index_reason = str(e)
            print(f"[Pipeline] ⚠️ GraphRAG 触发失败: {e}")

        # ── 步骤3：ES提取 → 生成审核数据 ──
        print(f"[Pipeline] 3/6 ES提取...")
        editor = ESReviewEditor()
        review_data = editor.extract_to_reviewable(json_path, MedicalGuidelineExtractor)

        guideline_name = _safe_guideline_name(filename, review_data.get("guideline_name"))
        review_data["guideline_name"] = guideline_name

        # 年份统一解析：抽取值 > MD 标题 > 文件名 > None
        resolved_year = resolve_guideline_year(
            review_data.get("guideline_year"),
            md_path,
            filename,
        )
        review_data["guideline_year"] = resolved_year

        # 把路径和图谱状态写进 review_data，后续审核完直接用，不再猜路径
        review_data["md_path"] = md_path
        review_data["json_path"] = json_path
        review_data["graph_index_status"] = graph_index_status
        if graph_index_reason:
            review_data["graph_index_reason"] = graph_index_reason

        review_id = editor.make_review_id(filename, guideline_name)
        saved_review_path = editor.save_reviewed_data(review_data, review_id)

        # ── 知识库元数据（上传即入库，不依赖审核）──
        try:
            kb_meta = KnowledgeBaseMetadata()
            # 若 add_guideline 签名是 int，这里把 None 兜底为当前年份；
            # 若已改为 Optional[int]，可去掉 or 兜底
            kb_year = resolved_year if resolved_year is not None else 0
            kb_meta.add_guideline(guideline_name, kb_year)
            if resolved_year is None:
                print(f"[Pipeline] ⚠️ 无法确定年份，知识库记为未知: {guideline_name}")
            else:
                print(f"[Pipeline] ✅ 知识库元数据已更新: {guideline_name} ({resolved_year})")
        except Exception as e:
            print(f"[Pipeline] ⚠️ 更新知识库元数据失败: {e}")

        print(f"[Pipeline] ✅ ES提取完成，共 {review_data['extracted_count']} 条记录")
        print(f"  审核数据已保存: {saved_review_path}")

        # ── 无记录：跳过审核入口 ──
        if review_data.get("extracted_count", 0) <= 0:
            print("[Pipeline] ⚠️ 没有抽取到有效药物剂量记录，跳过审核入口，等待后续修正提取器。")
            return {
                "status": "no_records",
                "review_id": review_id,
                "json_path": json_path,
                "md_path": md_path,
                "extracted_count": 0,
                "guideline_name": review_data.get("guideline_name", "未知指南"),
                "warning": review_data.get("warning"),
            }

        # ── 测试模式：自动批准 ──
        if skip_review:
            print(f"[Pipeline] 🚧 测试模式：自动批准所有记录")
            for rec in review_data["records"]:
                editor.approve_record(review_data, rec["id"])
            editor.save_reviewed_data(review_data, review_id)
            return _continue_pipeline_after_review(review_id, json_path, md_path)

        # ── 正常模式：等待审核 ──
        return {
            "status": "pending_review",
            "review_id": review_id,
            "json_path": json_path,
            "md_path": md_path,
            "extracted_count": review_data["extracted_count"],
            "guideline_name": review_data["guideline_name"],
        }

    finally:
        # 无论成功失败都清理临时文件
        try:
            if temp_pdf.exists():
                temp_pdf.unlink()
                print(f"[Pipeline] 已清理临时文件: {temp_pdf}")
        except Exception as e:
            print(f"[Pipeline] ⚠️ 清理临时文件失败: {e}")


def run_pipeline_after_review(review_id: str) -> Dict:
    """
    审核完成后，继续后续流程
    
    Args:
        review_id: 审核数据ID
    
    Returns:
        {
            'status': 'success',
            'es_indexed': 50,
            'chroma_indexed': True,
            'graph_extracted': True,
        }
    """
    from src.ingestion.es_review_editor import ESReviewEditor
    
    editor = ESReviewEditor()
    review_data = editor.load_reviewed_data(review_id)

    def _resolve_md_path(review_data: Dict) -> str:
        # ★ 修复：直接拿上传时保存的绝对路径，绝不瞎猜
        saved_path = review_data.get("md_path")
        if saved_path and Path(saved_path).exists():
            return str(saved_path)

        # 老数据兼容：只在当前目录确实只有一份 md 时才安全兜底
        md_files = sorted(Path("data/processed").glob("*.md"))
        if len(md_files) == 1:
            return str(md_files[0])
        
        # 如果老数据没有 md_path，兜底报错，防止乱建索引
        raise FileNotFoundError(f"未在审核数据中找到有效的 md_path 记录，无法构建 Chroma 索引。当前记录: {saved_path}，当前 data/processed/*.md 有 {len(md_files)} 份")

    # def _resolve_md_path(review_data: Dict) -> str:
    #     candidates = []
    #     guideline_name = review_data.get("guideline_name") or "未知指南"
    #     if guideline_name == "未知指南":
    #         guideline_name = Path(review_data.get("source_file") or "guide").stem or "未知指南"
    #     if guideline_name:
    #         candidates.extend([
    #             f"data/processed/{guideline_name}.md",
    #             f"data/processed/{guideline_name.replace(' ', '_')}.md",
    #         ])

    #     src_name = review_data.get("source_file")
    #     if src_name:
    #         stem = Path(src_name).stem
    #         candidates.extend([
    #             f"data/processed/{stem}.md",
    #             f"data/processed/{stem}_full.md",
    #             f"data/processed/{stem}_mineru.md",
    #         ])

    #     candidates.extend([
    #         "data/processed/full.md",
    #         "data/processed/*.md",
    #     ])

    #     for cand in candidates:
    #         if cand.endswith("*.md"):
    #             matches = sorted(Path("data/processed").glob("*.md"))
    #             if matches:
    #                 return str(matches[0])
    #             continue
    #         p = Path(cand)
    #         if p.exists():
    #             return str(p)

    #     matches = sorted(Path("data/processed").rglob("*.md"))
    #     if matches:
    #         return str(matches[0])
    #     raise FileNotFoundError(
    #         "未找到可用于 GraphRAG/Chroma 的 MD 文件；需要先确认 MinerU 产物为 full.md 或其它 .md 文件。"
    #     )

    md_path = _resolve_md_path(review_data)
    return _continue_pipeline_after_review(review_id, "", md_path)




def _continue_pipeline_after_review(
    review_id: str,
    json_path: str,
    md_path: str,
) -> Dict:
    """审核后的后续流程（内部函数）"""
    from src.ingestion.es_review_editor import ESReviewEditor
    from elasticsearch import Elasticsearch
    from src.indexing.chroma_indexer import build_chroma_index
    from src.ingestion.graph_builder import GraphBuilder

    editor = ESReviewEditor()
    review_data = editor.load_reviewed_data(review_id)

    # 步骤4：导出审核通过的记录 → 写入ES
    print(f"[Pipeline] 4/6 写入ES数据库...")
    es_docs = editor.export_to_es_docs(review_data)

    es_host = os.getenv("ES_HOST", "http://localhost:9200")
    es_index = os.getenv("ES_INDEX", "medical_guidelines")
    es_client = Elasticsearch([es_host])

    indexed_count = 0
    for doc in es_docs:
        es_client.index(index=es_index, document=doc)
        indexed_count += 1

    print(f"[Pipeline] ✅ ES入库完成，写入 {indexed_count} 条文档")

    # 步骤5：MD → ChromaDB
    print(f"[Pipeline] 5/6 构建ChromaDB向量索引...")
    if not Path(md_path).exists():
        print(f"[Pipeline] ⚠️ MD文件不存在，跳过向量索引: {md_path}")
        chroma_success = False
    else:
        try:
            build_chroma_index(md_path)
            chroma_success = True
            print(f"[Pipeline] ✅ ChromaDB索引构建完成")
        except Exception as e:
            print(f"[Pipeline] ❌ ChromaDB索引失败: {e}")
            chroma_success = False

    # # 步骤6：启动 GraphRAG 索引（后台任务）
    # print(f"[Pipeline] 6/6 GraphRAG实体抽取（可选，耗时较长）...")
    # graph_success = False
    # try:
    #     guideline_name = review_data.get("guideline_name") or Path(md_path).stem or "未知指南"
    #     if guideline_name == "未知指南":
    #         guideline_name = Path(review_data.get("source_file") or md_path).stem

    #     result = _trigger_graph_index_for_md(
    #         md_path,
    #         guideline_name=guideline_name,
    #         year=review_data.get("guideline_year") if review_data.get("guideline_year") not in (None, 1111, 0) else None,
    #     )
    #     graph_success = result.get("status") == "ok"
    #     print(f"[Pipeline] GraphRAG已复制MD到输入目录: {result.get('target')}")
    #     print(f"[Pipeline] ✅ GraphRAG 索引已启动: {result.get('graph_status')}")
    # except Exception as e:
    #     print(f"[Pipeline] ⚠️ GraphRAG 跳过: {e}")

    # 步骤6：GraphRAG 已在解析阶段触发，这里跳过
    print(f"[Pipeline] 6/6 GraphRAG 已在解析阶段触发，跳过")


    print(f"\n{'='*60}")
    print(f"[Pipeline] ✅ 全部流程完成！")
    print(f"{'='*60}")

    return {
        "status": "success",
        "es_indexed": indexed_count,
        "chroma_indexed": chroma_success,
        "graph_extracted": True,
    }


# ── 命令行入口 ──
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="偏头痛指南入库Pipeline")
    parser.add_argument("--file", required=True, help="PDF文件路径")
    parser.add_argument("--skip-review", action="store_true", help="跳过审核（测试用）")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.file)
    if not pdf_path.exists():
        print(f"❌ 文件不存在: {pdf_path}")
        exit(1)
    
    # 读取文件
    pdf_bytes = pdf_path.read_bytes()
    
    # 执行pipeline
    result = run_pipeline_upload(
        pdf_bytes=pdf_bytes,
        filename=pdf_path.name,
        skip_review=args.skip_review,
    )
    
    print(f"\n📊 结果:")
    print(f"  状态: {result['status']}")
    
    if result["status"] == "pending_review":
        print(f"  ⏳ 等待人工审核")
        print(f"  审核ID: {result['review_id']}")
        print(f"  记录数: {result['extracted_count']}")
        print(f"\n请在Streamlit前端的 '审核中心' 页面完成审核")
    elif result["status"] == "no_records":
        print(f"  ⚠️ 未抽取到记录，已跳过审核")
        print(f"  审核ID: {result.get('review_id')}")
        print(f"  MD: {result.get('md_path')}")
    else:
        print(f"  ES入库: {result['es_indexed']} 条")
        print(f"  ChromaDB: {'✅' if result['chroma_indexed'] else '❌'}")
        print(f"  GraphRAG: {'✅' if result['graph_extracted'] else '⏭️ 跳过'}")
