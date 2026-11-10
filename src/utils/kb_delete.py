"""
src/utils/kb_delete.py
职责：级联删除知识库数据（ES、Chroma、GraphRAG、元数据）

确保删除指南时彻底清理，不留脏数据。
"""

import sys
from pathlib import Path

if __package__ is None or __package__ == '':
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import os
import chromadb
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()


def delete_guideline_from_es(guideline_name: str, es_host: str = None) -> int:
    """从 ES 中精确删除指定指南的所有文档"""
    es_host = es_host or os.getenv("ES_HOST", "http://localhost:9200")
    es_index = os.getenv("ES_INDEX", "medical_guidelines")
    es_client = Elasticsearch([es_host])
    
    # 使用 term 精确匹配，防止 match 分词误删
    query = {
        "query": {
            "term": {
                "source.guideline.keyword": guideline_name
            }
        }
    }
    
    try:
        count_result = es_client.count(index=es_index, body=query)
        doc_count = count_result['count']
        
        if doc_count > 0:
            es_client.delete_by_query(
                index=es_index,
                body=query,
                conflicts='proceed'
            )
        return doc_count
    except Exception as e:
        print(f"[ES Delete] 删除失败: {e}")
        return 0


def delete_guideline_from_chroma(guideline_name: str, md_file: str = None, chroma_path: str = "data/chroma_db") -> int:
    """
    从 ChromaDB 中删除指定指南的向量。
    优先使用 md_file 精确匹配；如果命中 0 条（可能因重复上传导致残留），则回退到模糊匹配兜底。
    """
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection("medical_guidelines")
        
        all_data = collection.get(include=['metadatas'])
        ids_to_delete = []
        
        if all_data and all_data['ids']:
            # 第一轮：尝试精确匹配最新的 md_file
            if md_file:
                for idx, meta in enumerate(all_data['metadatas']):
                    if meta.get('file_name', '') == md_file:
                        ids_to_delete.append(all_data['ids'][idx])
                        
            # 第二轮：如果精确匹配没找到（或者没传 md_file），触发模糊匹配兜底
            if not ids_to_delete:
                print(f"[Chroma Delete] 精确匹配失败，触发模糊匹配清理残留...")
                for idx, meta in enumerate(all_data['metadatas']):
                    fname = meta.get('file_name', '')
                    if guideline_name in fname:
                        ids_to_delete.append(all_data['ids'][idx])
                    
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"[Chroma Delete] 成功清理 {len(ids_to_delete)} 条向量")
            return len(ids_to_delete)
            
        return 0
    except Exception as e:
        print(f"[Chroma Delete] 清理失败: {e}")
        return 0



def delete_guideline_complete(guideline_name: str, guideline_year: int) -> dict:
    """
    完整删除一个指南（ES + ChromaDB + GraphRAG + 元数据）
    """
    from src.utils.kb_metadata import KnowledgeBaseMetadata
    
    print(f"\n=== 开始级联删除指南: {guideline_name} ({guideline_year}) ===")
    
    # ★ 修复：先去元数据里查出真实的 md_file 名字
    kb_meta = KnowledgeBaseMetadata()
    md_file_to_delete = None
    for kb in kb_meta.get_all():
        if kb.get('name') == guideline_name and kb.get('year') == guideline_year:
            md_file_to_delete = kb.get('md_file')
            break
            
    # 1. 删除 ES 文档
    es_deleted = delete_guideline_from_es(guideline_name)
    
    # 2. 删除 ChromaDB 向量（传入精确的 md_file）
    chroma_deleted = delete_guideline_from_chroma(guideline_name, md_file=md_file_to_delete)
    
    # 3. 级联清理 GraphRAG
    graph_deleted = False
    try:
        from src.ingestion.graph_builder import GraphBuilder
        gb = GraphBuilder()
        gb.remove_markdown(guideline_name, guideline_year)
        graph_deleted = True
        print(f"[GraphRAG Delete] 已清理图谱输入文件")
    except Exception as e:
        print(f"[GraphRAG Delete] 清理图谱文件失败: {e}")
    
    # 4. 删除元数据
    metadata_deleted = kb_meta.remove_guideline(guideline_name, guideline_year)
    
    result = {
        "es_deleted": es_deleted,
        "chroma_deleted": chroma_deleted,
        "graph_deleted": graph_deleted,
        "metadata_deleted": metadata_deleted,
    }
    return result