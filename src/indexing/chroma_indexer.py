"""
src/indexing/chroma_indexer.py
职责：从MD文件构建ChromaDB向量索引

改动说明：
- 从原代码的inline逻辑提取为独立函数
- 增加了持久化配置，避免每次重建
- 其他逻辑保留原代码
- 【新增】导入src.config配置全局Embedding和LLM模型
- 【新增】兼容直接运行和模块导入两种方式

测试方法：
# 方式1：作为模块运行（推荐）
python -m src.indexing.chroma_indexer

# 方式2：使用测试脚本
python test_module.py src.indexing.chroma_indexer

# 方式3：直接运行
python src/indexing/chroma_indexer.py
"""

import sys
from pathlib import Path

# 【关键】必须在任何src模块导入之前处理路径
# 检测是否直接运行（通过检查 __package__ 是否为None）
if __package__ is None or __package__ == '':
    # 直接运行此文件，需要添加项目根目录到sys.path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# 【新增】导入全局配置
import src.config

from src.retrieval.vector_retriever import VectorRetriever


def build_chroma_index(
    md_file_path: str,
    chroma_path: str = "data/chroma_db",
    collection_name: str = "medical_guidelines",
) -> VectorRetriever:
    """
    从MD文件构建ChromaDB向量索引（原代码逻辑封装）
    
    Args:
        md_file_path: MD文件路径
        chroma_path: ChromaDB持久化路径
        collection_name: 集合名称
    
    Returns:
        VectorRetriever实例
    
    改动说明：
    - 这是对原代码中索引构建部分的封装
    - 逻辑完全保留，只是提取为独立函数便于pipeline调用
    """
    md_path = Path(md_file_path)
    if not md_path.exists():
        raise FileNotFoundError(f"MD文件不存在: {md_file_path}")
    
    print(f"[ChromaIndexer] 开始构建索引: {md_file_path}")
    
    # 使用VectorRetriever构建索引（原代码逻辑）
    retriever = VectorRetriever(
        md_file_path=str(md_path),
        chroma_path=chroma_path,
        collection_name=collection_name,
    )
    
    print(f"[ChromaIndexer] ✅ 索引构建完成")
    print(f"  持久化路径: {chroma_path}")
    print(f"  集合名称: {collection_name}")
    
    return retriever


# ── 单元测试 ──
if __name__ == "__main__":
    test_md = "data/processed/中国偏头痛诊治指南2022版.md"
    
    if not Path(test_md).exists():
        print(f"❌ 测试文件不存在: {test_md}")
    else:
        try:
            retriever = build_chroma_index(test_md)
            
            # 测试检索
            results = retriever.search("偏头痛急性期治疗", top_k=3)
            print(f"\n测试检索，找到 {len(results)} 个结果")
            
            print("\n✅ 测试通过")
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
