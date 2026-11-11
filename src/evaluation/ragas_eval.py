"""
src/evaluation/ragas_eval.py
职责：使用 RAGAS 评估 RAG 系统，全程使用本地 Ollama 模型（qwen2.5 + bge-m3）
支持多路路由（ES, Hybrid, Graph, Web），并跑 100 条多维度测试集。
"""

import sys
from pathlib import Path
import math

if __package__ is None or __package__ == "":
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import os
import json
import warnings
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="ragas")

# ───────────────────────────────────────────────────────────
# 本地模型封装（Ollama 兼容 OpenAI API）
# ───────────────────────────────────────────────────────────
def _ollama_base() -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    return base + "/v1"


def _build_local_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("RAGAS_LLM_MODEL", "qwen2.5:14b-instruct-q4_K_M"),
        base_url=_ollama_base(),
        api_key="ollama",
        temperature=0,
        timeout=600,      # 10 分钟超时
        max_retries=3,
    )


def _build_local_emb():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=os.getenv("RAGAS_EMB_MODEL", "bge-m3"),
        base_url=_ollama_base(),
        api_key="ollama",
        check_embedding_ctx_length=False,
    )


# ───────────────────────────────────────────────────────────
# 用真实检索 + LLM 生成填充 contexts / answer
# ───────────────────────────────────────────────────────────
def _run_retrieval_and_answer(
    dataset: List[Dict],
    top_k: int = 5, 
) -> List[Dict]:
    from src.retrieval.router import detect_query_type

    _retrievers = {}

    def get_retriever(t: str):
        if t in _retrievers:
            return _retrievers[t]
        try:
            if t == "es":
                from src.retrieval.es_retriever import ESRetriever
                _retrievers[t] = ESRetriever()
            elif t == "hybrid":
                from src.retrieval.vector_retriever import VectorRetriever
                _retrievers[t] = VectorRetriever()
            elif t == "graph":
                from src.retrieval.graph_retriever import GraphRetriever
                _retrievers[t] = GraphRetriever()
            else:
                _retrievers[t] = None
        except Exception as e:
            print(f"⚠️ 初始化 {t} 检索器失败: {e}")
            _retrievers[t] = None
        return _retrievers[t]

    llm = _build_local_llm()
    temp_save_path = Path("data/eval/eval_dataset_100_filled.json")

    for idx, item in enumerate(dataset, 1):
        q = item.get("query") or item.get("question") or ""
        if not q:
            continue

        if item.get("contexts") and item.get("answer"):
            print(f"[{idx}/{len(dataset)}] 已填充，跳过: {q[:40]}")
            continue

        retrieval_type = (
            item.get("expected_route") 
            or item.get("metadata", {}).get("expected_retrieval")
            or detect_query_type(q)
        )

        print(f"[{idx}/{len(dataset)}] {retrieval_type} 检索: {q[:50]}")

        contexts = []
        answer = ""

        try:
            # ★ 修复：Web 和本地检索彻底分开，缩进拉平！
            if retrieval_type == "web":
                from langchain_tavily import TavilySearch
                searcher = TavilySearch(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))
                res = searcher.invoke(q)
                
                if isinstance(res, dict) and "results" in res:
                    for r in res["results"]:
                        if isinstance(r, dict) and r.get("content"):
                            contexts.append(r["content"][:1200])
                elif isinstance(res, list):
                    for r in res:
                        if isinstance(r, dict) and r.get("content"):
                            contexts.append(r["content"][:1200])
                else:
                    contexts = [str(res)[:1200]]

            else:
                r = get_retriever(retrieval_type)
                if r is not None:
                    raw_results = r.search(q, top_k=top_k)
                    
                    if retrieval_type == "es":
                        contexts = [
                            x["source"].get("traceability", {}).get("raw_text_segment", "")
                            for x in raw_results
                        ]
                    elif retrieval_type == "graph":
                        contexts = [x.get("message", "") for x in raw_results]
                    else:  # hybrid
                        contexts = [n.get_content() for n in raw_results]

            # 过滤空值
            contexts = [c.strip() for c in contexts if c and len(c.strip()) > 10]
            item["contexts"] = contexts[:top_k]

            # 新的 Prompt：放宽限制，引导提取
            if contexts:
                ctx_text = "\n\n---\n\n".join(contexts)
                prompt = (
                    "你是一个专业的医疗问答助手。请根据下面提供的《指南原文片段》回答用户的问题。\n"
                    "注意：\n"
                    "1. 请尽力从提供的文本中提取关键信息来回答问题，不要轻易放弃。\n"
                    "2. 如果文本中明确包含所需的数据（如剂量、药物名、分期、推荐等级），请直接提取并综合回答。\n"
                    "3. 只有当所有文本片段中**完全**没有任何相关内容时，才回答“根据当前资料无法确定”。\n"
                    "4. 请使用自然流畅的语言回答，提取核心事实，忽略无关的不良反应或禁忌症长文本。\n\n"
                    f"【指南原文片段】\n{ctx_text}\n\n"
                    f"【用户问题】\n{q}\n\n"
                    "【回答】"
                )
                try:
                    item["answer"] = llm.invoke(prompt).content.strip()
                except Exception as e:
                    print(f"    ⚠️ 生成回答失败: {e}")
                    item["answer"] = ""
            else:
                item["answer"] = ""

        except Exception as e:
            print(f"    ⚠️ 检索/生成失败: {e}")
            item["contexts"] = []
            item["answer"] = ""

        # 断点续传
        if idx % 5 == 0 or idx == len(dataset):
            temp_save_path.parent.mkdir(parents=True, exist_ok=True)
            temp_save_path.write_text(
                json.dumps(dataset, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"   💾 已自动保存前 {idx} 条回填数据至 {temp_save_path}")

    return dataset




# ───────────────────────────────────────────────────────────
# 主评估类
# ───────────────────────────────────────────────────────────
class RAGASEvaluator:
    METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def __init__(self, eval_dataset_path: str = "data/eval/e2e_dataset_100.json"):
        self.eval_dataset_path = Path(eval_dataset_path)
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> List[Dict]:
        if not self.eval_dataset_path.exists():
            print(f"⚠️ 数据集不存在: {self.eval_dataset_path}")
            return []

        try:
            data = json.loads(self.eval_dataset_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"❌ 数据集 JSON 解析失败: {e}")
            return []

        if not isinstance(data, list):
            return []

        for item in data:
            if "query" in item and "question" not in item:
                item["question"] = item["query"]
            if "ground_truth" not in item:
                item["ground_truth"] = ""

        return data

    def evaluate(
        self,
        fill_missing: bool = True,
        sample_limit: Optional[int] = None,
        save_result: bool = True,
    ) -> Dict:
        if not self.dataset:
            return {"error": "数据集为空或全部无效"}

        print(f"\n{'='*60}")
        print(f"RAGAS 评估（本地 Ollama）")
        print(f"{'='*60}")
        print(f"数据集: {self.eval_dataset_path}（{len(self.dataset)} 条）")

        if fill_missing:
            print("\n[步骤1] 填充 contexts / answer...")
            self.dataset = _run_retrieval_and_answer(self.dataset)
            filled_path = self.eval_dataset_path.parent / "eval_dataset_filled.json"
            filled_path.write_text(
                json.dumps(self.dataset, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"✅ 回填数据集已保存: {filled_path}")

        valid = [
            x for x in self.dataset
            if x.get("contexts") and x.get("answer") and x.get("ground_truth")
        ]
        if sample_limit:
            valid = valid[:sample_limit]

        print(f"\n[步骤2] 有效样本: {len(valid)} / {len(self.dataset)}")
        if not valid:
            return {"error": "没有有效样本（contexts/answer 全空）"}

        print("\n[步骤3] 加载本地模型...")
        try:
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            _llm = LangchainLLMWrapper(_build_local_llm())
            _emb = LangchainEmbeddingsWrapper(_build_local_emb())
        except ImportError as e:
            return {"error": f"RAGAS 未安装或缺少依赖: {e}。"}

        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
        except ImportError as e:
            return {"error": f"RAGAS 导入失败: {e}"}

        try:
            from ragas import EvaluationDataset
            from ragas.dataset_schema import SingleTurnSample
            samples = [
                SingleTurnSample(
                    user_input=x["question"],
                    response=x["answer"],
                    retrieved_contexts=x["contexts"],
                    reference=x["ground_truth"],
                )
                for x in valid
            ]
            eval_ds = EvaluationDataset(samples=samples)
            ragas_version = "0.2.x"
            print(f"  使用 RAGAS {ragas_version} 数据格式")
        except ImportError:
            from datasets import Dataset
            eval_ds = Dataset.from_dict(
                {
                    "question": [x["question"] for x in valid],
                    "answer": [x["answer"] for x in valid],
                    "contexts": [x["contexts"] for x in valid],
                    "ground_truth": [x["ground_truth"] for x in valid],
                }
            )
            ragas_version = "0.1.x"
            print(f"  使用 RAGAS {ragas_version} 数据格式")

        print(f"\n[步骤4] 运行 RAGAS 评估（{len(valid)} 条样本）...")
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        try:
            from ragas import RunConfig
            run_config = RunConfig(max_workers=1, timeout=600, max_retries=3)
        except ImportError:
            run_config = None

        try:
            result = ragas_evaluate(
                dataset=eval_ds,
                metrics=metrics,
                llm=_llm,
                embeddings=_emb,
                run_config=run_config,
                raise_exceptions=False,
                show_progress=True,
            )
        except TypeError:
            for m in metrics:
                try: m.llm = _llm
                except: pass
            try: answer_relevancy.embeddings = _emb
            except: pass
            result = ragas_evaluate(
                dataset=eval_ds,
                metrics=metrics,
                raise_exceptions=False,
            )

        scores = {}
        try:
            df = result.to_pandas()
            for name in self.METRIC_NAMES:
                if name in df.columns:
                    col_mean = df[name].mean()
                    scores[name] = float(col_mean) if not math.isnan(col_mean) else float('nan')
            
            per_sample_path = self.eval_dataset_path.parent / "eval_per_sample.csv"
            df.to_csv(per_sample_path, index=False, encoding="utf-8")
            print(f"✅ 逐条分数已保存: {per_sample_path}")
        except Exception as e:
            print(f"⚠️ to_pandas 失败: {e}")
            try:
                scores = {k: float(v) for k, v in dict(result).items() if isinstance(v, (int, float))}
            except Exception:
                scores = {}

        valid_scores = [v for v in scores.values() if isinstance(v, (int, float)) and not math.isnan(v)]
        scores["overall_score"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        scores["_n_samples"] = len(valid)
        scores["_ragas_version"] = ragas_version

        if save_result:
            result_path = self.eval_dataset_path.parent / "eval_result.json"
            clean_scores = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in scores.items()}
            result_path.write_text(
                json.dumps(clean_scores, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"✅ 汇总分数已保存: {result_path}")

        print(f"\n{'='*60}")
        print(f"RAGAS 评估结果")
        print(f"{'='*60}")
        for name in self.METRIC_NAMES:
            v = scores.get(name)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                bar = "█" * int(v * 20)
                print(f"  {name:20s} {v:.3f}  {bar}")
            else:
                print(f"  {name:20s} N/A (计算超时或失败)")
                
        overall = scores.get("overall_score", 0.0)
        print(f"  {'overall_score':20s} {overall:.3f}")
        print(f"  {'样本数':20s} {scores.get('_n_samples', 0)}")
        print(f"{'='*60}\n")

        return scores


if __name__ == "__main__":
    evaluator = RAGASEvaluator(eval_dataset_path="data/eval/e2e_dataset_100.json")
    if not evaluator.dataset:
        print("❌ 未找到 100 条数据集，请检查路径。")
        sys.exit(1)

    result = evaluator.evaluate(fill_missing=True, sample_limit=None)
    if "error" in result:
        print(f"\n❌ 评估失败: {result['error']}")
        sys.exit(1)
