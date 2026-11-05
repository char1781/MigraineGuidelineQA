"""
pipeline.py - 一键入库的统一入口（更新：增加审核环节）

新流程：
1. 用户上传PDF → MinerU解析 → 得到JSON和MD
2. JSON → ES提取 → 生成审核数据（保存到 data/review/）
3. 用户在Streamlit审核页面确认/修改/拒绝每条记录
4. 审核通过的记录 → 写入ES数据库
5. MD文件 → ChromaDB向量索引
6. MD文件 → GraphRAG抽取 → Neo4j

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
from dotenv import load_dotenv

load_dotenv()


def run_pipeline_upload(
    pdf_bytes: bytes,
    filename: str,
    skip_review: bool = False,
) -> Dict:
    """
    从Streamlit上传的入口（接收bytes）
    
    Args:
        pdf_bytes: PDF文件的字节内容
        filename: 文件名
        skip_review: 是否跳过审核（测试用）
    
    Returns:
        {
            'status': 'success' | 'pending_review',
            'review_id': 'xxx',  # 需要人工审核
            'json_path': 'data/processed/xxx.json',
            'md_path': 'data/processed/xxx.md',
        }
    """
    from src.ingestion.mineru_client import MinerUClient
    from src.ingestion.es_review_editor import ESReviewEditor
    from src.ingestion.es_indexer import MedicalGuidelineExtractor
    
    # 步骤1：保存上传的PDF到临时目录
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_pdf = temp_dir / filename
    temp_pdf.write_bytes(pdf_bytes)
    
    print(f"[Pipeline] 1/6 已保存上传文件: {temp_pdf}")
    
    # 步骤2：调用MinerU解析
    print(f"[Pipeline] 2/6 调用MinerU API解析...")
    client = MinerUClient()
    result = client.upload_and_wait(str(temp_pdf), timeout=300)
    
    json_path = result["json_path"]
    md_path = result["md_path"]
    
    print(f"[Pipeline] ✅ MinerU解析完成")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    
    # 步骤3：ES提取 → 生成审核数据
    print(f"[Pipeline] 3/6 ES提取...")
    editor = ESReviewEditor()
    review_data = editor.extract_to_reviewable(json_path, MedicalGuidelineExtractor)
    
    # 生成review_id（基于文件名）
    review_id = Path(filename).stem
    saved_review_path = editor.save_reviewed_data(review_data, review_id)
    
    print(f"[Pipeline] ✅ ES提取完成，共 {review_data['extracted_count']} 条记录")
    print(f"  审核数据已保存: {saved_review_path}")
    
    # 清理临时文件
    temp_pdf.unlink()
    
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
    
    if skip_review:
        # 测试模式：自动批准所有记录
        print(f"[Pipeline] 🚧 测试模式：自动批准所有记录")
        for rec in review_data["records"]:
            editor.approve_record(review_data, rec["id"])
        editor.save_reviewed_data(review_data, review_id)
        
        # 直接执行后续流程
        return _continue_pipeline_after_review(review_id, json_path, md_path)
    
    # 正常模式：返回pending_review状态，等待用户在前端审核
    return {
        "status": "pending_review",
        "review_id": review_id,
        "json_path": json_path,
        "md_path": md_path,
        "extracted_count": review_data["extracted_count"],
        "guideline_name": review_data["guideline_name"],
    }


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
        candidates = []
        guideline_name = review_data.get("guideline_name")
        if guideline_name:
            candidates.extend([
                f"data/processed/{guideline_name}.md",
                f"data/processed/{guideline_name.replace(' ', '_')}.md",
            ])

        src_name = review_data.get("source_file")
        if src_name:
            stem = Path(src_name).stem
            candidates.extend([
                f"data/processed/{stem}.md",
                f"data/processed/{stem}_full.md",
                f"data/processed/{stem}_mineru.md",
            ])

        candidates.extend([
            "data/processed/full.md",
            "data/processed/*.md",
        ])

        for cand in candidates:
            if cand.endswith("*.md"):
                matches = sorted(Path("data/processed").glob("*.md"))
                if matches:
                    return str(matches[0])
                continue
            p = Path(cand)
            if p.exists():
                return str(p)

        matches = sorted(Path("data/processed").rglob("*.md"))
        if matches:
            return str(matches[0])
        raise FileNotFoundError(
            "未找到可用于 GraphRAG/Chroma 的 MD 文件；需要先确认 MinerU 产物为 full.md 或其它 .md 文件。"
        )

    md_path = _resolve_md_path(review_data)
    return _continue_pipeline_after_review(review_id, "", md_path)


# def _continue_pipeline_after_review(
#     review_id: str,
#     json_path: str,
#     md_path: str,
# ) -> Dict:
#     """审核后的后续流程（内部函数）"""
#     from src.ingestion.es_review_editor import ESReviewEditor
#     from elasticsearch import Elasticsearch
#     from src.indexing.chroma_indexer import build_chroma_index
#     # from src.ingestion.graph_builder import build_graph
    
#     editor = ESReviewEditor()
#     review_data = editor.load_reviewed_data(review_id)
    
#     # 步骤4：导出审核通过的记录 → 写入ES
#     print(f"[Pipeline] 4/6 写入ES数据库...")
#     es_docs = editor.export_to_es_docs(review_data)
    
#     es_host = os.getenv("ES_HOST", "http://localhost:9200")
#     es_index = os.getenv("ES_INDEX", "medical_guidelines")
#     es_client = Elasticsearch([es_host])
    
#     indexed_count = 0
#     for doc in es_docs:
#         es_client.index(index=es_index, document=doc)
#         indexed_count += 1
    
#     print(f"[Pipeline] ✅ ES入库完成，写入 {indexed_count} 条文档")
    
#     # 步骤5：MD → ChromaDB
#     print(f"[Pipeline] 5/6 构建ChromaDB向量索引...")
#     if not Path(md_path).exists():
#         print(f"[Pipeline] ⚠️ MD文件不存在，跳过向量索引: {md_path}")
#         chroma_success = False
#     else:
#         try:
#             build_chroma_index(md_path)
#             chroma_success = True
#             print(f"[Pipeline] ✅ ChromaDB索引构建完成")
#         except Exception as e:
#             print(f"[Pipeline] ❌ ChromaDB索引失败: {e}")
#             chroma_success = False
    
#     # 步骤6：GraphRAG（可选，耗时长）
#     print(f"[Pipeline] 6/6 GraphRAG实体抽取（可选，耗时较长）...")
#     graph_success = False
#     try:
#         build_graph(md_path)
#         graph_success = True
#         print(f"[Pipeline] ✅ GraphRAG抽取完成")
#     except Exception as e:
#         print(f"[Pipeline] ⚠️ GraphRAG跳过: {e}")
    
#     print(f"\n{'='*60}")
#     print(f"[Pipeline] ✅ 全部流程完成！")
#     print(f"{'='*60}")
    
#     return {
#         "status": "success",
#         "es_indexed": indexed_count,
#         "chroma_indexed": chroma_success,
#         "graph_extracted": graph_success,
#     }

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

    # 步骤6：启动 GraphRAG 索引（后台任务）
    print(f"[Pipeline] 6/6 GraphRAG实体抽取（可选，耗时较长）...")
    graph_success = False
    try:
        if not Path(md_path).exists():
            raise FileNotFoundError(f"MD文件不存在: {md_path}")

        gb = GraphBuilder()
        target = gb.add_markdown(
            md_path,
            guideline_name=review_data.get("guideline_name") or Path(md_path).stem,
            year=review_data.get("guideline_year") if review_data.get("guideline_year") not in (None, 1111) else None,
        )
        print(f"[Pipeline] GraphRAG已复制MD到输入目录: {target}")
        status = gb.start_index()
        graph_success = bool(status.get("state") == "running" or status.get("state") == "idle")
        print(f"[Pipeline] ✅ GraphRAG 索引已启动: {status}")
    except Exception as e:
        print(f"[Pipeline] ⚠️ GraphRAG 跳过: {e}")

    print(f"\n{'='*60}")
    print(f"[Pipeline] ✅ 全部流程完成！")
    print(f"{'='*60}")

    return {
        "status": "success",
        "es_indexed": indexed_count,
        "chroma_indexed": chroma_success,
        "graph_extracted": graph_success,
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
    else:
        print(f"  ES入库: {result['es_indexed']} 条")
        print(f"  ChromaDB: {'✅' if result['chroma_indexed'] else '❌'}")
        print(f"  GraphRAG: {'✅' if result['graph_extracted'] else '⏭️ 跳过'}")
