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
import re
from dotenv import load_dotenv

load_dotenv()

_reranker_model = None

def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        from FlagEmbedding import FlagReranker
        print("正在加载 BGE-Reranker 模型...")
        _reranker_model = FlagReranker('./models/bge-reranker-base', use_fp16=True)
    return _reranker_model


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
        self.vector_retriever = self.index.as_retriever(similarity_top_k=10)
        
        # BM25检索器（如果有nodes）
        if self.nodes:
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=self.nodes,
                similarity_top_k=10,
            )
        else:
            self.bm25_retriever = None
            print("⚠️ 警告：未提供nodes，BM25检索器不可用")
        
        # 融合检索器
        if self.bm25_retriever:
            self.fusion_retriever = QueryFusionRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                similarity_top_k=20,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
                verbose=False,  # 改为False避免调试时输出过多
                retriever_weights=[0.4, 0.6],  # 原代码权重
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
    
    # def _build_nodes_from_md(self, md_file_path: str) -> List:
    #     """
    #     从MD文件构建节点
        
    #     改动说明：
    #     - 封装为类方法，便于单独测试
    #     - 逻辑完全保留原代码（HierarchicalNodeParser + split_long_nodes）
    #     - 针对纯文本语义的切分（按 Markdown 标题层级）
    #     """
    #     md_path = Path(md_file_path)
    #     if not md_path.exists():
    #         raise FileNotFoundError(f"MD文件不存在: {md_file_path}")
        
    #     print(f"[VectorRetriever] 加载MD文件: {md_file_path}")

    #     # ★ 1. 读取并清洗表格
    #     raw_text = md_path.read_text(encoding="utf-8")
    #     # 匹配连续的以 | 开头的行（Markdown 表格）
    #     table_pattern = re.compile(r'((?:^\|.*\|$\n?)+)', re.MULTILINE)
    #     cleaned_text = table_pattern.sub('\n[此处原为表格，数据已提取至 ES 结构化数据库，请使用精确查询]\n', raw_text)
        
    #     # 写入临时文件或直接转 Document
    #     from llama_index.core import Document
    #     documents = [Document(text=cleaned_text, metadata={"file_name": str(md_path)})]
        
    #     # # 使用SimpleDirectoryReader加载（原代码）
    #     # print(f"[VectorRetriever] 加载MD文件: {md_file_path}")
    #     # documents = SimpleDirectoryReader(input_files=[str(md_path)]).load_data()
    #     # print(f"[VectorRetriever] 加载了 {len(documents)} 个文档")
        
    #      # 1. 先用 MarkdownNodeParser 按 # 标题切分（保持语义段落完整）
    #     md_parser = MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True)
    #     md_nodes = md_parser.get_nodes_from_documents(documents)
    #     print(f"[VectorRetriever] 按标题切分后，共 {len(md_nodes)} 个大段")
        
    #     # 2. 如果某个标题下的段落特别长（超过 1200 字符），再用句子切分器兜底
    #     # 这样既保证了结构完整，又避免了超长 node 被 embedding 模型截断
    #     sentence_splitter = SentenceSplitter(chunk_size=1200, chunk_overlap=100)
    #     final_nodes = []
        
    #     for node in md_nodes:
    #         # 如果 Markdown 某个标题下文本超过 1200 字符，就强制切分
    #         if len(node.text) > 1200:
    #             sub_nodes = sentence_splitter.get_nodes_from_documents([Document(text=node.text, metadata=node.metadata)])
    #             final_nodes.extend(sub_nodes)
    #         else:
    #             final_nodes.append(node)
                
    #     print(f"[VectorRetriever] 二次细化切分后，共 {len(final_nodes)} 个节点")
    #     return final_nodes

    def _clean_text_for_embedding(self, raw_text: str) -> str:
        """
        文本预处理：移除表格标签噪声、图片占位、参考文献符号，
        保留有语义价值的正文内容。
        """
        text = raw_text

        # ① HTML <table>：诊断标准类展平成行，噪声网格表直接删除
        html_table_pat = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)

        def _html_table_handler(m: re.Match) -> str:
            block = m.group(0)
            # 提取 <tr> 数和 <td> 数
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', block, re.DOTALL | re.IGNORECASE)
            td_counts = [len(re.findall(r'<td[^>]*>', r, re.IGNORECASE)) for r in rows]
            total_cols = sum(td_counts)
            avg_cols = total_cols / max(1, len(rows))

            # 噪声判定：行多+列多（典型日历/网格图，如 31 天月历）
            if len(rows) > 15 and avg_cols > 5:
                return ""  # 直接丢弃

            # 诊断标准类：有实质内容的表格，展平成行
            cells = re.findall(r'<td[^>]*>(.*?)</td>', block, re.DOTALL | re.IGNORECASE)
            cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]  # 去残留标签
            cells = [c for c in cells if c]
            if not cells:
                return ""

            # 如果是"诊断标准"这类有标题行的表，保留标题+条目
            header = cells[0] if cells else ""
            body = cells[1:]
            lines = [header]
            for c in body:
                # 把连续的"附 1.1.1"这类小节标题单独成行
                if re.match(r'^(附|表|[一二三四五六七八九十]+、)', c):
                    lines.append(f"\n{c}")
                else:
                    lines.append(c)
            return "\n".join(lines) + "\n"

        text = html_table_pat.sub(_html_table_handler, text)

        # ② Markdown 管道表格 — 全部替换（剂量等已进 ES）
        md_table_pat = re.compile(r'((?:^\|.*\|$\n?)+)', re.MULTILINE)
        text = md_table_pat.sub('', text)

        # ③ 图片占位：![]() 和 <img src=...>
        text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)
        text = re.sub(r'<img[^>]*>', '', text, flags=re.IGNORECASE)

        # ④ LaTeX 参考文献：$^{[35]}$ 这类 — 删掉参考文献标记，保留正文
        text = re.sub(r'\$\^\{?\[?\d+\]?\}?\$?', '', text)

        # ⑤ 清理所有残余 HTML 标签（不限定标签名，全部干掉）
        text = re.sub(r'<[^>]+>', '', text)

        # ⑥ 方括号参考文献引用 [35] [4,2023] 等
        text = re.sub(r'\[\d{1,3}\]', '', text)
        text = re.sub(r'\[\d{1,3},\d{1,3}\]', '', text)

        # ⑦ 清理残余 HTML 标签（<br/> <p> <span> 等）
        text = re.sub(r'</?(?:br|p|span|div|b|i|em|strong|u|sup|sub)[^>]*>', '', text, flags=re.IGNORECASE)

        # ⑧ 合并多余空行（连续 3 行以上空行 → 2 行）
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    def _build_nodes_from_md(self, md_file_path: str) -> List:
        """针对纯文本语义的切分（按 Markdown 标题层级）"""
        md_path = Path(md_file_path)
        if not md_path.exists():
            raise FileNotFoundError(f"MD文件不存在: {md_file_path}")

        print(f"[VectorRetriever] 加载MD文件: {md_file_path}")
        raw_text = md_path.read_text(encoding="utf-8")
        cleaned_text = self._clean_text_for_embedding(raw_text)
        print(f"[VectorRetriever] 清洗后文本长度: {len(raw_text)} → {len(cleaned_text)}")

        from llama_index.core import Document
        documents = [Document(text=cleaned_text, metadata={"file_name": str(md_path)})]
        
        # 2. 按标题切分
        from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
        md_parser = MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True)
        md_nodes = md_parser.get_nodes_from_documents(documents)
        print(f"[VectorRetriever] 按标题切分后，共 {len(md_nodes)} 个大段")
        
        # 3. 二次细化切分
        sentence_splitter = SentenceSplitter(chunk_size=1200, chunk_overlap=100)
        final_nodes = []
        
        for node in md_nodes:
            if len(node.text) > 1200:
                sub_nodes = sentence_splitter.get_nodes_from_documents(
                    [Document(text=node.text, metadata=node.metadata)]
                )
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
    
    # def search(self, query: str, top_k: int = 4) -> List:
    #     """
    #     执行混合检索
        
    #     Args:
    #         query: 查询文本
    #         top_k: 返回结果数量
        
    #     Returns:
    #         节点列表，每个节点有score和content
    #     """
    #     # 动态调整top_k
    #     self.fusion_retriever.similarity_top_k = top_k
        
    #     results = self.fusion_retriever.retrieve(query)
    #     return results

    def _expand_context(self, node) -> str:
        """从原文件中提取当前块前后的上下文（Parent-Child 变种）"""
        try:
            file_name = node.metadata.get("file_name") or node.metadata.get("file_path")
            if not file_name or not Path(file_name).exists():
                return node.get_content()
                
            full_text = Path(file_name).read_text(encoding="utf-8")
            snippet = node.get_content().strip()
            
            # 方案 A：用文本匹配找位置（最稳，兼容性最好）
            if snippet:
                idx = full_text.find(snippet)
                if idx != -1:
                    s = max(0, idx - 500)
                    e = min(len(full_text), idx + len(snippet) + 500)
                    return full_text[s:e]
            
            # 方案 B：如果文本没匹配上，退化为原样返回
            return node.get_content()
        except Exception as e:
            print(f"上下文扩展失败: {e}")
            return node.get_content()
    
    # def search(self, query: str, top_k: int = 5) -> List:
    #     # 1. 初始召回 15 条
    #     self.fusion_retriever.similarity_top_k = 15
    #     results = self.fusion_retriever.retrieve(query)
        
    #     # 2. 用 Reranker 进行精排
    #     if not results:
    #         return []
            
    #     try:
    #         reranker = _get_reranker()
    #         # 构造 pairs
    #         pairs = [[query, node.get_content()] for node in results]
    #         scores = reranker.compute_score(pairs)
            
    #         # 将分数绑定回 node 并重排
    #         scored_nodes = list(zip(scores, results))
    #         scored_nodes.sort(key=lambda x: x[0], reverse=True)
            
    #         # 取前 top_k 返回
    #     #     return [node for score, node in scored_nodes[:top_k]]
    #     # except Exception as e:
    #     #     print(f"Reranker 排序失败，退回原顺序: {e}")
    #     #     return results[:top_k]
    #         results = [node for score, node in scored_nodes]
    #     except Exception as e:
    #         print(f"Reranker 排序失败: {e}")

    #     # 关键改动：对选出的 Top-K 节点进行上下文扩展，并覆盖原 text
    #     final_results = results[:top_k]
    #     for node in final_results:
    #         expanded_text = self._expand_context(node)
    #         # 覆盖原始内容，这样 format_context 拿到的就是大段文本
    #         node.set_content(expanded_text)
            
    #     return final_results

    def search(self, query: str, top_k: int = 5) -> List:
        self.fusion_retriever.similarity_top_k = 20
        results = self.fusion_retriever.retrieve(query)
        
        if not results:
            return []
            
        # Reranker 精排
        try:
            reranker = _get_reranker()
            pairs = [[query, node.get_content()] for node in results]
            scores = reranker.compute_score(pairs)
            
            scored_nodes = list(zip(scores, results))
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            results = [node for score, node in scored_nodes]
        except Exception as e:
            print(f"Reranker 排序失败: {e}")

        # ★ 关键修复：正确处理 NodeWithScore 的内部 text
        import copy
        from llama_index.core.schema import NodeWithScore
        
        final_results = []
        for node in results[:top_k]:
            expanded_text = self._expand_context(node)
            
            # 1. 深拷贝底层的 TextNode
            new_text_node = copy.deepcopy(node.node)
            # 2. 覆盖底层 TextNode 的 text
            new_text_node.text = expanded_text
            # 3. 重新包装成 NodeWithScore
            new_node = NodeWithScore(node=new_text_node, score=node.score)
            
            final_results.append(new_node)
            
        return final_results
    
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
