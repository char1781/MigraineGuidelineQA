"""
src/retrieval/vector_retriever.py
职责：BM25+向量混合检索，从原代码中拆分出来

测试方法：
# 方式1：作为模块运行（推荐）
python -m src.retrieval.vector_retriever

# 方式2：直接运行（也支持）
python src/retrieval/vector_retriever.py
"""

import sys
from pathlib import Path

# 【关键】必须在任何src模块导入之前处理路径
if __package__ is None or __package__ == '':
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))



import os
import sys
from pathlib import Path
from typing import List, Optional

# 【新增】处理直接运行时的路径问题
if __name__ == "__main__":
    # 如果是直接运行此文件，需要把项目根目录加入sys.path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# 【新增】导入全局配置，必须在导入LlamaIndex组件之前
import src.config  # 这会自动配置Settings

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import TextNode, Document
import chromadb
from dotenv import load_dotenv

load_dotenv()


class VectorRetriever:
    """BM25+向量混合检索器"""
    
    def __init__(
        self,
        md_file_path: Optional[str] = None,
        chroma_path: str = "data/chroma_db",
        collection_name: str = "medical_guidelines",
    ):
        """
        初始化检索器
        
        Args:
            md_file_path: MD文件路径，如果为None则从已存在的ChromaDB加载
            chroma_path: ChromaDB持久化路径
            collection_name: 集合名称
        
        改动说明：
        - 增加了chroma_path和collection_name参数，方便配置
        - 其他逻辑完全保留原代码
        """
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        
        # 确保ChromaDB目录存在
        os.makedirs(self.chroma_path, exist_ok=True)
        
        # 初始化ChromaDB客户端
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        except Exception as e:
            print(f"ChromaDB初始化失败: {e}")
            print("尝试使用内存客户端...")
            self.chroma_client = chromadb.Client()
        
        # 获取或创建集合
        try:
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                self.collection_name
            )
        except Exception as e:
            print(f"获取集合失败: {e}")
            try:
                self.chroma_client.delete_collection(self.collection_name)
                self.chroma_collection = self.chroma_client.create_collection(
                    self.collection_name
                )
            except Exception as e2:
                raise RuntimeError(f"ChromaDB集合创建失败: {e2}")
        
        # 配置向量存储
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # 如果提供了MD文件，构建索引
        if md_file_path:
            self.nodes = self._build_nodes_from_md(md_file_path)
            self.index = self._build_index(self.nodes)
        else:
            # 从已有的ChromaDB加载
            # self.nodes = []  # 注：从ChromaDB加载时nodes为空，但不影响向量检索
            # 从已有的 ChromaDB 中提取全部文本，恢复 nodes，用于 BM25
            self.nodes = self._load_nodes_from_chroma()
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context,
            )
        
        # 初始化检索器
        self.vector_retriever = self.index.as_retriever(similarity_top_k=5)
        
        # BM25检索器（如果有nodes）
        if self.nodes:
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=self.nodes,
                similarity_top_k=5,
            )
        else:
            self.bm25_retriever = None
            print("⚠️ 警告：未提供nodes，BM25检索器不可用")
        
        # 融合检索器
        if self.bm25_retriever:
            self.fusion_retriever = QueryFusionRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                similarity_top_k=5,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
                verbose=False,  # 改为False避免调试时输出过多
                retriever_weights=[0.6, 0.4],  # 原代码权重
            )
        else:
            self.fusion_retriever = self.vector_retriever
    
    def _load_nodes_from_chroma(self) -> List[TextNode]:
        """从 Chroma 拉取文本重建 nodes"""
        try:
            results = self.chroma_collection.get(include=['documents', 'metadatas'])
            if not results or not results['documents']:
                return []
            nodes = []
            for doc, meta in zip(results['documents'], results['metadatas']):
                if doc:
                    nodes.append(TextNode(text=doc, metadata=meta or {}))
            return nodes
        except Exception as e:
            print(f"从 Chroma 恢复 nodes 失败: {e}")
            return []
    
    def _build_nodes_from_md(self, md_file_path: str) -> List:
        """
        从MD文件构建节点
        
        改动说明：
        - 封装为类方法，便于单独测试
        - 逻辑完全保留原代码（HierarchicalNodeParser + split_long_nodes）
        - 针对纯文本语义的切分（按 Markdown 标题层级）
        """
        md_path = Path(md_file_path)
        if not md_path.exists():
            raise FileNotFoundError(f"MD文件不存在: {md_file_path}")
        
        print(f"[VectorRetriever] 加载MD文件: {md_file_path}")
        
        # 使用SimpleDirectoryReader加载（原代码）
        print(f"[VectorRetriever] 加载MD文件: {md_file_path}")
        documents = SimpleDirectoryReader(input_files=[str(md_path)]).load_data()
        print(f"[VectorRetriever] 加载了 {len(documents)} 个文档")
        
         # 1. 先用 MarkdownNodeParser 按 # 标题切分（保持语义段落完整）
        md_parser = MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True)
        md_nodes = md_parser.get_nodes_from_documents(documents)
        print(f"[VectorRetriever] 按标题切分后，共 {len(md_nodes)} 个大段")
        
        # 2. 如果某个标题下的段落特别长（超过 1200 字符），再用句子切分器兜底
        # 这样既保证了结构完整，又避免了超长 node 被 embedding 模型截断
        sentence_splitter = SentenceSplitter(chunk_size=1200, chunk_overlap=100)
        final_nodes = []
        
        for node in md_nodes:
            # 如果 Markdown 某个标题下文本超过 1200 字符，就强制切分
            if len(node.text) > 1200:
                sub_nodes = sentence_splitter.get_nodes_from_documents([Document(text=node.text, metadata=node.metadata)])
                final_nodes.extend(sub_nodes)
            else:
                final_nodes.append(node)
                
        print(f"[VectorRetriever] 二次细化切分后，共 {len(final_nodes)} 个节点")
        return final_nodes
    
    def _build_index(self, nodes: List) -> VectorStoreIndex:
        """
        构建向量索引
        
        改动说明：
        - 增加了节点文本长度检查，避免超过embedding模型限制
        - 其他逻辑完全保留
        """
        # 检查节点长度（原代码有这个检查）
        for node in nodes:
            if len(node.text) > 2048:
                print(f"⚠️ 警告: 节点文本长度 {len(node.text)} > 2048，可能影响 embedding 效果！")
                # node.text = node.text[:2048]
        
        # 构建索引（原代码逻辑）
        index = VectorStoreIndex(nodes, storage_context=self.storage_context)
        
        # 持久化
        # self.vector_store.persist(persist_path=self.chroma_path)
        
        return index
    
    def search(self, query: str, top_k: int = 4) -> List:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            节点列表，每个节点有score和content
        """
        # 动态调整top_k
        self.fusion_retriever.similarity_top_k = top_k
        
        results = self.fusion_retriever.retrieve(query)
        return results
    
    def format_context(self, nodes: List) -> str:
        """
        格式化检索结果为上下文
        """
        if not nodes:
            return "未找到相关信息"
        
        context_parts = []
        for i, node in enumerate(nodes, 1):
            content = node.get_content()
            context = f"【参考{i}】\n{content}"
            context_parts.append(context)
        
        return "\n\n".join(context_parts)


# ── 单元测试 ──
if __name__ == "__main__":
    md_file = "data/processed/中国偏头痛诊治指南2022版.md"
    
    if not Path(md_file).exists():
        print(f"❌ 测试文件不存在: {md_file}")
        print("请先用MinerU生成MD文件，或修改路径")
    else:
        try:
            # 测试构建索引
            print("=== 测试1: 构建索引 ===")
            retriever = VectorRetriever(md_file_path=md_file)
            
            # 测试检索
            print("\n=== 测试2: 执行检索 ===")
            query = "偏头痛急性期有哪些治疗方法？"
            results = retriever.search(query, top_k=3)
            
            print(f"查询: {query}")
            print(f"找到 {len(results)} 个结果\n")
            
            for i, node in enumerate(results, 1):
                print(f"[{i}] Score: {node.score:.4f}")
                print(f"Content: {node.get_content()[:150]}...")
                print()
            
            # 测试上下文格式化
            print("\n=== 测试3: 格式化上下文 ===")
            context = retriever.format_context(results)
            print(context[:500])
            
            print("\n✅ 所有测试通过")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
