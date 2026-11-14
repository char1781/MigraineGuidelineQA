"""
src/agent/tools.py
职责：定义所有Agent可用的工具

测试方法：
# 方式1：作为模块运行（推荐）
python -m src.agent.tools

# 方式2：使用测试脚本
python test_module.py src.agent.tools

# 方式3：直接运行
python src/agent/tools.py
"""

import sys
from pathlib import Path

# 【关键】必须在任何src模块导入之前处理路径
if __package__ is None or __package__ == '':
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import os
from typing import List

# 导入全局配置
import src.config

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()


class KnowledgeBaseArgs(BaseModel):
    """知识库检索参数"""
    query: str = Field(description="用户的问题")


def query_knowledge_base(query: str) -> str:
    """
    基于偏头痛医疗指南知识库的智能检索工具
    """
    from src.retrieval import detect_query_type, ESRetriever, VectorRetriever
    
    # 检查是否偏头痛相关
    
    query_lower = query.lower()

    disease_keywords = ['偏头痛', '头痛', 'migraine', '偏头疼']
    drug_keywords = ['布洛芬', '萘普生', '阿司匹林', '对乙酰氨基酚', '扑热息痛',
            '利扎曲普坦', '佐米曲普坦', '舒马曲普坦', '那拉曲普坦',
            '丙戊酸钠', '美托洛尔', '阿米替林', '氟桂利嗪', '拉莫三嗪',
            '曲坦', '药物', '治疗', '剂量', '用法', '禁忌', '不良反应']  

    is_relevant = any(k in query_lower for k in disease_keywords + drug_keywords)
    
    if not is_relevant:
        return (
            "❌ 此问题与偏头痛知识库无关。\n"
            "建议：使用网络搜索工具 (tavily_search) 查找答案。"
        )
    
    # 使用智能检索路由
    try:
        # 支持通过环境变量切换意图识别策略
        import os
        strategy = os.getenv("INTENT_DETECTION_STRATEGY", "rule")
        # strategy可选值: "rule"(默认,快速), "llm"(准确), "hybrid"(最准确)
        
        from src.retrieval.router import auto_detect_with_strategy
        query_type = auto_detect_with_strategy(query, strategy=strategy)
        
        if query_type == "es":
            # ES精确检索
            from src.retrieval.es_retriever import format_es_doc, clean_guideline_name, es_needs_fallback
            retriever = ESRetriever()
            results = retriever.search(query, top_k=3)
            if not results:
                return "⚠️ 知识库中未找到相关信息，建议使用网络搜索工具。"

            # ES 只覆盖治疗表格；阶段/禁忌/特殊人群等信息不足时降级到语义检索
            if es_needs_fallback(query, results):
                try:
                    from src.retrieval.vector_retriever import VectorRetriever
                    vec = VectorRetriever()
                    context = vec.format_context(vec.search(query, top_k=3))
                    return "ℹ️ 结构化知识库未覆盖该问题，已切换为指南全文语义检索：\n\n" + context
                except Exception as e:
                    print(f"[ES→Hybrid] 降级失败: {e}")

            # 按提问意图裁剪字段（剂量类必带证据等级/推荐强度/指南来源）
            docs_content = []
            seen = set()
            sources = []
            for result in results:
                doc = result['source']
                body = format_es_doc(doc, query)
                if body in seen:  # 同药不同剂型可能召回内容相同的行，避免重复占位
                    continue
                seen.add(body)
                docs_content.append(f"【参考{len(docs_content) + 1}】\n{body}")

                source_info = doc.get('source', {})
                guideline = clean_guideline_name(source_info.get('guideline', '未知指南'))
                year = source_info.get('year', '未知年份')
                sources.append(f"{guideline} ({year}年)")

            # 去重来源
            unique_sources = list(dict.fromkeys(sources))
            sources_text = "\n".join([f"  - {s}" for s in unique_sources])

            formatted = "\n\n".join(docs_content)
            return f"📚 知识库检索结果（来自ES数据库）：\n\n{formatted}\n\n📖 信息来源:\n{sources_text}"
        
        else:
            # BM25+向量混合检索
            try:
                # 方案1：尝试从已有的ChromaDB加载（推荐，不需要指定MD文件）
                retriever = VectorRetriever()  # 从ChromaDB加载
                nodes = retriever.search(query, top_k=3)
                if not nodes:
                    return "⚠️ 知识库中未找到相关信息，建议使用网络搜索工具。"
                
                # 格式化并返回
                formatted = retriever.format_context(nodes)
                return f"📚 知识库检索结果：\n\n{formatted}"
            except Exception as e:
                # 如果ChromaDB加载失败，尝试查找MD文件
                import glob
                md_files = glob.glob("data/processed/*.md")
                
                if not md_files:
                    return "❌ 未找到任何已入库的指南文件，请先上传PDF并完成入库。"
                
                # 使用第一个找到的MD文件（或者可以遍历所有文件）
                md_path = md_files[0]
                
                try:
                    retriever = VectorRetriever(md_file_path=md_path)
                    nodes = retriever.search(query, top_k=3)
                    formatted = retriever.format_context(nodes)
                    return f"📚 知识库检索结果：\n\n{formatted}"
                except Exception as e2:
                    return f"❌ 检索出错: {str(e2)}\n建议使用网络搜索工具。"
    
    except Exception as e:
        return f"❌ 检索出错: {str(e)}\n建议使用网络搜索工具。"


def get_tools() -> List:
    """
    获取所有工具列表
    
    改动说明：
    - 封装为函数，便于agent_executor调用
    - 逻辑完全保留原代码
    
    Returns:
        工具列表
    """
    # 知识库检索工具
    knowledge_base_tool = StructuredTool.from_function(
        func=query_knowledge_base,
        name="knowledge_base_search",
        description=(
            "🎯 偏头痛医疗指南知识库检索工具\n"
            "适用范围：\n"
            "- 药物剂量、用法、频次查询（如：利扎曲普坦的推荐剂量）\n"
            "- 治疗方案、证据等级查询（如：偏头痛急性期治疗方法）\n"
            "- 不良反应、禁忌症查询（如：佐米曲普坦的不良反应）\n\n"
            "⚠️ 限制：\n"
            "- 仅包含2025年之前的指南文档\n"
            "- 不包含最新的2026年指南\n\n"
            "💡 提示：如需查询2026年最新指南或共识，请使用 tavily_search 工具"
        ),
        args_schema=KnowledgeBaseArgs,
        return_direct=False,
    )
    
    # 网络搜索工具
    web_search = TavilySearch(
        max_results=2,
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
    )
    
    # 图谱检索工具（真实实现）
    def graph_search_real(query: str) -> str:
        """通过 GraphRAG 调用图谱查询，返回跨文档关系结果。"""
        from src.retrieval.graph_retriever import GraphRetriever

        retriever = GraphRetriever()

        try:
            retriever.timeout = 300 
            results = retriever.search(query, top_k=3)
            
            # 如果 GraphRAG 内部报错或正在建库，自动抛出异常触发降级
            if results and results[0].get("type") in ("graph_error", "graph_status"):
                print(f"[Tool-Graph] 检索不可用: {results[0].get('message')}，触发降级")
                raise Exception(results[0].get("message"))
                
            return retriever.format_context(results)
            
        except Exception as e:
            print(f"[Tool-Graph] 发生异常: {e}，熔断降级到 Hybrid 检索")
            from src.retrieval.vector_retriever import VectorRetriever
            
            # 兜底降级为向量混合检索
            retriever_h = VectorRetriever()
            nodes = retriever_h.search(query, top_k=3)
            
            answer = "⚠️ 跨文档图谱服务响应超时或正在构建中，已自动降级为本地文档语义检索。\n\n"
            answer += retriever_h.format_context(nodes)
            return answer

    graph_tool = StructuredTool.from_function(
        func=graph_search_real,
        name="graph_search",
        description="跨文档知识图谱检索工具，用于偏头痛共病、跨指南关联、相互作用等问题。",
        args_schema=KnowledgeBaseArgs,
    )

    return [knowledge_base_tool, web_search, graph_tool]


# ── 单元测试 ──
if __name__ == "__main__":
    print("=== 工具定义测试 ===\n")
    
    tools = get_tools()
    print(f"可用工具数量: {len(tools)}")
    
    for tool in tools:
        print(f"\n工具名称: {tool.name}")
        print(f"描述: {tool.description[:100]}...")
    
    # 测试知识库检索工具
    print("\n=== 测试知识库检索 ===")
    result = query_knowledge_base("利扎曲普坦的推荐剂量是多少？")
    print(result[:300])
    
    print("\n✅ 测试完成")
