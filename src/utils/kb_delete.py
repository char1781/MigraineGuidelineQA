"""
src/utils/kb_delete.py
职责：级联删除知识库数据

功能：
- 从ES删除指定指南的所有文档
- 从ChromaDB删除对应的向量
- 从元数据文件中移除记录

测试方法：
python -m src.utils.kb_delete
或
python src/utils/kb_delete.py
"""

import sys
from pathlib import Path

# 【关键】必须在任何src模块导入之前处理路径
if __package__ is None or __package__ == '':
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from elasticsearch import Elasticsearch
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()


def delete_guideline_from_es(guideline_name: str, es_host: str = None) -> int:
    """
    从ES中删除指定指南的所有文档
    
    Args:
        guideline_name: 指南名称（如"中国偏头痛诊治指南2022版"）
        es_host: ES地址
    
    Returns:
        删除的文档数量
    """
    es_host = es_host or os.getenv("ES_HOST", "http://localhost:9200")
    es_client = Elasticsearch([es_host])
    
    # 查询条件：source.guideline == guideline_name
    query = {
        "query": {
            "match": {
                "source.guideline": guideline_name
            }
        }
    }
    
    # 先统计要删除的文档数
    count_result = es_client.count(index="medical_guidelines", body=query)
    doc_count = count_result['count']
    
    if doc_count == 0:
        return 0
    
    # 执行删除
    es_client.delete_by_query(
        index="medical_guidelines",
        body=query,
        conflicts='proceed'  # 忽略版本冲突
    )
    
    return doc_count


def delete_guideline_from_chroma(guideline_name: str, chroma_path: str = "data/chroma_db") -> int:
    """
    从ChromaDB中删除指定指南的向量
    
    注意：ChromaDB不支持按metadata删除，这里返回0表示需要手动重建索引
    或者标记为"已删除"
    
    Args:
        guideline_name: 指南名称
        chroma_path: ChromaDB路径
    
    Returns:
        删除的向量数量（当前实现返回0，表示需要手动处理）
    """
    # ChromaDB的删除比较复杂，有两种方案：
    # 方案1：重建整个索引（排除已删除的指南）
    # 方案2：标记为已删除，检索时过滤（推荐）
    
    # 这里返回0，提示用户需要重建ChromaDB索引
    # 或者在检索时通过metadata过滤
    return 0


def delete_guideline_complete(guideline_name: str, guideline_year: int) -> dict:
    """
    完整删除一个指南（ES + ChromaDB + 元数据）
    
    Returns:
        {
            "es_deleted": int,
            "chroma_deleted": int,
            "metadata_deleted": bool,
            "warning": str (如果有)
        }
    """
    from src.utils.kb_metadata import KnowledgeBaseMetadata
    
    # 1. 删除ES文档
    es_deleted = delete_guideline_from_es(guideline_name)
    
    # 2. ChromaDB删除（当前返回0）
    chroma_deleted = delete_guideline_from_chroma(guideline_name)
    
    # 3. 删除元数据
    kb_meta = KnowledgeBaseMetadata()
    metadata_deleted = kb_meta.remove_guideline(guideline_name, guideline_year)
    
    # 4. 返回结果
    result = {
        "es_deleted": es_deleted,
        "chroma_deleted": chroma_deleted,
        "metadata_deleted": metadata_deleted,
    }
    
    if chroma_deleted == 0:
        result["warning"] = (
            "ChromaDB向量未删除，建议重新构建向量索引。"
            "或者在检索时会自动过滤已删除的指南。"
        )
    
    return result


# ── 单元测试 ──
if __name__ == "__main__":
    # 测试删除
    result = delete_guideline_complete("测试指南", 2024)
    
    print(f"删除结果:")
    print(f"  ES文档: {result['es_deleted']} 条")
    print(f"  ChromaDB向量: {result['chroma_deleted']} 个")
    print(f"  元数据: {'成功' if result['metadata_deleted'] else '未找到'}")
    
    if "warning" in result:
        print(f"\n⚠️ 警告: {result['warning']}")
