"""
src/retrieval/reranker.py
职责：对检索结果进行精排（Reranking）

使用场景：
- 混合检索后，对Top-K结果重新排序
- 提高Top-3结果的精度（用户最关注前几条）

推荐模型：
- bge-reranker-v2-m3 (中文效果好)
- cohere-rerank (商业API)

测试方法：
python -m src.retrieval.reranker
或
python src/retrieval/reranker.py
"""

import sys
from pathlib import Path

# 【关键】必须在任何src模块导入之前处理路径
if __package__ is None or __package__ == '':
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from typing import List
import os


class Reranker:
    """
    精排器（占位实现）
    
    待集成：
    - FlagEmbedding的bge-reranker-v2-m3
    - 或Cohere Rerank API
    """
    
    def __init__(self, model_name: str = "bge-reranker-v2-m3"):
        self.model_name = model_name
        self.model = None  # TODO: 加载模型
    
    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[int]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回Top-K个最相关的文档索引
        
        Returns:
            重排序后的文档索引列表 [2, 0, 5, ...]
        """
        # TODO: 实现真实Reranker
        # 1. 使用bge-reranker计算 query-doc 相关性分数
        # 2. 按分数排序
        # 3. 返回Top-K索引
        
        # 占位实现：返回原始顺序
        return list(range(len(documents)))[:top_k]


# 集成到VectorRetriever的示例：
# results = retriever.search(query, top_k=10)  # 先召回10个
# reranker = Reranker()
# reranked_indices = reranker.rerank(query, [r.text for r in results], top_k=3)
# final_results = [results[i] for i in reranked_indices]  # 返回精排后的Top-3
