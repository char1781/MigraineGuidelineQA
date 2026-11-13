"""
src/evaluation/ragas_eval.py
职责：使用 RAGAS 评估 RAG 系统，本地 Ollama（模型硬编码）
支持多路路由（ES, Hybrid, Graph, Web）+ Rerank 精排。

=============================== 主要配置（只改这里） ===============================
生成/judge 模型硬编码，不读 RAGAS_LLM_MODEL / RAGAS_JUDGE_MODEL 环境变量。
本地只有 14b 时，GEN_MODEL == JUDGE_MODEL == 14b。
=============================================================================
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
import time
import warnings
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="ragas")

# ───────────────────────────────────────────────────────────
# ★ 模型硬编码（不再读 RAGAS_*_MODEL 环境变量）
# ───────────────────────────────────────────────────────────
OLLAMA_HOST  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
GEN_MODEL    = "qwen2.5:14b-instruct-q4_K_M"     # 生成
JUDGE_MODEL  = "qwen2.5:14b-instruct-q4_K_M"     # 评判（有 32b 时改这里）
EMB_MODEL    = "bge-m3"                           # 嵌入

# ───────────────────────────────────────────────────────────
# ★ Rerank 配置
# ───────────────────────────────────────────────────────────
RERANK_MODEL  = "BAAI/bge-reranker-base"   # 轻量，CPU 可跑；显存够可换 v2-m3
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")   # cpu / cuda
RECALL_K      = 20                          # 粗排召回数（es/hybrid）
FINAL_K       = 5                           # 精排后保留数

# 输出路径
EVAL_DIR        = Path("data/eval")
FAILED_PATH     = EVAL_DIR / "failed_samples.json"
FILLED_PATH     = EVAL_DIR / "eval_dataset_filled.json"
PER_SAMPLE_PATH = EVAL_DIR / "eval_per_sample.csv"
CONFUSION_PATH  = EVAL_DIR / "route_confusion_matrix.csv"
RESULT_PATH     = EVAL_DIR / "eval_result.json"


# ═══════════════════════════════════════════════════════════
# 1. JSON 容错：剥 ```json 壳 + 截取最外层 {}
# ═══════════════════════════════════════════════════════════
def _clean_json_content(text: str) -> str:
    if not isinstance(text, str):
        return text
    t = text.strip()
    if not t:
        return t
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if t and t[0] not in "{[":
        s_candidates = [i for i in (t.find("{"), t.find("[")) if i != -1]
        if s_candidates:
            s = min(s_candidates)
            e = max(t.rfind("}"), t.rfind("]"))
            if e > s:
                t = t[s:e + 1]
    return t


# ═══════════════════════════════════════════════════════════
# 2. LLM 构建（带 JSON 容错）
# ═══════════════════════════════════════════════════════════
def _build_llm(model: str, timeout: int = 600):
    from langchain_openai import ChatOpenAI

    class _JsonSafeChatOpenAI(ChatOpenAI):
        def _postprocess_result(self, result):
            try:
                for gen_list in result.generations:
                    for gen in gen_list:
                        msg = getattr(gen, "message", None)
                        if msg is None:
                            continue
                        c = msg.content
                        if isinstance(c, str):
                            cleaned = _clean_json_content(c)
                            if cleaned != c:
                                try:
                                    msg.content = cleaned
                                except Exception:
                                    pass
                                try:
                                    gen.text = cleaned
                                except Exception:
                                    pass
            except Exception:
                pass
            return result

        def _generate(self, *args, **kwargs):
            return self._postprocess_result(super()._generate(*args, **kwargs))

        async def _agenerate(self, *args, **kwargs):
            return self._postprocess_result(await super()._agenerate(*args, **kwargs))

    print(f"   [LLM] model={model}  base={OLLAMA_HOST}/v1")
    return _JsonSafeChatOpenAI(
        model=model,
        base_url=OLLAMA_HOST + "/v1",
        api_key="ollama",
        temperature=0,
        timeout=timeout,
        max_retries=3,
    )


def _build_local_llm():
    return _build_llm(GEN_MODEL)


def _build_judge_llm():
    return _build_llm(JUDGE_MODEL)


def _build_local_emb():
    from langchain_openai import OpenAIEmbeddings
    print(f"   [EMB] model={EMB_MODEL}")
    return OpenAIEmbeddings(
        model=EMB_MODEL,
        base_url=OLLAMA_HOST + "/v1",
        api_key="ollama",
        check_embedding_ctx_length=False,
    )


# ═══════════════════════════════════════════════════════════
# 3. Reranker（懒加载 + 优雅降级）
# ═══════════════════════════════════════════════════════════
_RERANKER = None
_RERANK_AVAILABLE = True


def _get_reranker():
    global _RERANKER, _RERANK_AVAILABLE
    if not _RERANK_AVAILABLE:
        return None
    if _RERANKER is not None:
        return _RERANKER
    try:
        from sentence_transformers import CrossEncoder
        print(f"   [RERANK] loading {RERANK_MODEL} on {RERANK_DEVICE} ...")
        _RERANKER = CrossEncoder(RERANK_MODEL, max_length=512, device=RERANK_DEVICE)
        print(f"   [RERANK] ready")
    except Exception as e:
        print(f"   ⚠️ reranker 加载失败，将跳过重排: {type(e).__name__}: {e}")
        _RERANK_AVAILABLE = False
        _RERANKER = None
    return _RERANKER


def _rerank(query: str, contexts: List[str], top_k: int = FINAL_K,
            min_score: float = 0.0) -> List[str]:
    """cross-encoder 精排；失败时回退原顺序前 top_k"""
    if not contexts:
        return contexts
    model = _get_reranker()
    if model is None:
        return contexts[:top_k]
    try:
        pairs = [(query, c) for c in contexts]
        scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
        ranked = sorted(zip(scores, contexts), key=lambda x: -float(x[0]))
        out = [c for s, c in ranked if float(s) >= min_score][:top_k]
        return out if out else [c for _, c in ranked[:top_k]]
    except Exception as e:
        print(f"   ⚠️ rerank 打分失败，回退原顺序: {e}")
        return contexts[:top_k]


# ═══════════════════════════════════════════════════════════
# 4. Retriever 缓存
# ═══════════════════════════════════════════════════════════
_RETRIEVER_CACHE: Dict[str, object] = {}


def _get_retriever(route: str):
    if route in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[route]
    try:
        if route == "es":
            from src.retrieval.es_retriever import ESRetriever
            _RETRIEVER_CACHE[route] = ESRetriever()
        elif route == "hybrid":
            from src.retrieval.vector_retriever import VectorRetriever
            _RETRIEVER_CACHE[route] = VectorRetriever()
        elif route == "graph":
            from src.retrieval.graph_retriever import GraphRetriever
            _RETRIEVER_CACHE[route] = GraphRetriever()
        else:
            _RETRIEVER_CACHE[route] = None
    except Exception as e:
        print(f"   ⚠️ 初始化 {route} 检索器失败: {e}")
        _RETRIEVER_CACHE[route] = None
    return _RETRIEVER_CACHE[route]


# ═══════════════════════════════════════════════════════════
# 5. 三通道 context 统一抽取
# ═══════════════════════════════════════════════════════════
def _extract_contexts(route: str, raw_results) -> List[str]:
    if not raw_results:
        return []
    out: List[str] = []
    for x in raw_results:
        try:
            if route == "es":
                seg = (x.get("source") or {}).get("traceability", {}).get("raw_text_segment", "")
                out.append(seg or "")

            elif route == "graph":
                msg = x.get("message", "") or ""
                src = (
                    x.get("source_text")
                    or x.get("raw_text")
                    or (x.get("source") or {}).get("traceability", {}).get("raw_text_segment", "")
                    or ""
                )
                out.append(f"{msg}\n{src}".strip() if src else msg)

            else:  # hybrid
                if hasattr(x, "get_content"):
                    text = x.get_content()
                elif isinstance(x, dict):
                    text = x.get("page_content") or x.get("content") or str(x)
                else:
                    text = str(x)
                # ★ 去掉表格式标题行（"药物: XXX | 推荐强度: ..."），避免稀释 rerank 打分
                if text:
                    lines = text.split("\n")
                    if lines and ("|" in lines[0] or lines[0].startswith("药物")):
                        text = "\n".join(lines[1:]).strip() or text
                out.append(text)
        except Exception as e:
            print(f"   ⚠️ context 抽取异常({route}): {e}")
            out.append("")
    return out


# ═══════════════════════════════════════════════════════════
# 6. Web 检索（超时 + 重试）
# ═══════════════════════════════════════════════════════════
def _web_search(query: str, max_results: int = 3, timeout: int = 15, retries: int = 2) -> Tuple[List[str], str]:
    last_err = ""
    for attempt in range(retries + 1):
        try:
            from langchain_tavily import TavilySearch
            searcher = TavilySearch(
                max_results=max_results,
                tavily_api_key=os.getenv("TAVILY_API_KEY"),
                timeout=timeout,
            )
            res = searcher.invoke(query)
            out: List[str] = []
            if isinstance(res, dict) and "results" in res:
                for r in res["results"]:
                    if isinstance(r, dict) and r.get("content"):
                        out.append(r["content"][:1200])
            elif isinstance(res, list):
                for r in res:
                    if isinstance(r, dict) and r.get("content"):
                        out.append(r["content"][:1200])
            elif res:
                out.append(str(res)[:1200])
            return out, ""
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                print(f"   ⚠️ Web 检索失败（第 {attempt + 1} 次），2s 后重试: {last_err}")
                time.sleep(2)
    return [], last_err


# ═══════════════════════════════════════════════════════════
# 7. 生成 prompt（收紧拒答）
# ═══════════════════════════════════════════════════════════
_ANSWER_PROMPT = (
    "你是医疗问答助手。请**严格依据**下方《指南原文片段》回答用户问题。\n"
    "规则：\n"
    "1) 若片段中包含问题所需的实体或数值（药名、剂量、疗程、分期、推荐等级等），直接提取并简明作答。\n"
    "2) 若片段**未提及**问题的核心实体，或**未包含**所需数值，必须明确回答"
    "“资料中未提及【X】”并说明缺失内容；禁止用片段外的知识补全。\n"
    "3) 不要复述无关的不良反应/禁忌症长文本。\n"
    "4) 回答尽量简短，不超过 5 句话。\n\n"
    "【指南原文片段】\n{ctx}\n\n"
    "【用户问题】\n{q}\n\n"
    "【回答】"
)


def _is_filled(item: Dict) -> bool:
    ans = (item.get("answer") or "").strip()
    if not item.get("contexts") or not ans:
        return False
    if "资料中未提及" in ans and "建议" not in ans:
        return False
    return True


# ═══════════════════════════════════════════════════════════
# 8. 回填 contexts / answer（含 rerank）
# ═══════════════════════════════════════════════════════════
def _run_retrieval_and_answer(dataset: List[Dict], top_k: int = FINAL_K) -> List[Dict]:
    from src.retrieval.router import detect_query_type

    llm = _build_local_llm()
    failed: List[Dict] = []
    FILLED_PATH.parent.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(dataset, 1):
        q = item.get("query") or item.get("question") or ""
        if not q:
            continue

        if _is_filled(item):
            print(f"[{idx}/{len(dataset)}] 已填充，跳过: {q[:40]}")
            continue

        # expected / actual 分离
        expected = item.get("expected_route") or item.get("metadata", {}).get("expected_retrieval") or "unknown"
        try:
            actual = detect_query_type(q)
        except Exception as e:
            print(f"   ⚠️ 路由分类失败，回退 expected: {e}")
            actual = expected
        item["expected_route"] = expected
        item["actual_route"] = actual

        print(f"[{idx}/{len(dataset)}] {expected} → {actual} | {q[:50]}")

        contexts: List[str] = []
        err: str = ""

        try:
            if actual == "web":
                # Web：Tavily 已做相关性排序，不 rerank
                contexts, err = _web_search(q, max_results=top_k)

            elif actual == "graph":
                # ★ Graph 不 rerank：message 是推理摘要而非原文，cross-encoder 打分不可靠
                r = _get_retriever(actual)
                if r is None:
                    err = f"retriever_init_failed:{actual}"
                else:
                    raw = r.search(q, top_k=top_k)
                    contexts = _extract_contexts(actual, raw)

            else:
                # ★ ES / Hybrid：粗排 RECALL_K → 精排 FINAL_K
                r = _get_retriever(actual)
                if r is None:
                    err = f"retriever_init_failed:{actual}"
                else:
                    raw = r.search(q, top_k=RECALL_K)
                    contexts = _extract_contexts(actual, raw)
                    contexts = [c.strip() for c in contexts if c and len(c.strip()) > 10]
                    n_before = len(contexts)
                    contexts = _rerank(q, contexts, top_k=FINAL_K)
                    print(f"      rerank: {n_before} → {len(contexts)}")
        except Exception as e:
            err = f"retrieval_exception:{type(e).__name__}: {e}"
            print(f"   ⚠️ 检索异常: {err}")

        contexts = [c.strip() for c in contexts if c and len(c.strip()) > 10]
        item["contexts"] = contexts[:top_k]

        if item["contexts"]:
            ctx_text = "\n\n---\n\n".join(item["contexts"])
            try:
                item["answer"] = llm.invoke(_ANSWER_PROMPT.format(ctx=ctx_text, q=q)).content.strip()
            except Exception as e:
                err = err or f"llm_exception:{type(e).__name__}: {e}"
                item["answer"] = ""
        else:
            item["answer"] = ""
            err = err or "no_context"

        if not item["contexts"] or not item["answer"]:
            failed.append({
                "id": item.get("id"),
                "query": q,
                "expected_route": expected,
                "actual_route": actual,
                "reason": err or ("empty_context" if not item["contexts"] else "empty_answer"),
            })

        if idx % 5 == 0 or idx == len(dataset):
            FILLED_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"   💾 已保存前 {idx} 条至 {FILLED_PATH}")

    if failed:
        FAILED_PATH.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n⚠️ {len(failed)} 条失败样本已写入 {FAILED_PATH}")
    else:
        print("\n✅ 无失败样本")

    return dataset


# ═══════════════════════════════════════════════════════════
# 9. 主评估类
# ═══════════════════════════════════════════════════════════
class RAGASEvaluator:
    METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    METRIC_ALIAS = {
        "faithfulness":      ["faithfulness", "faithfullness"],
        "answer_relevancy":  ["answer_relevancy", "answer_relevancy_score", "answer_relevance"],
        "context_precision": ["context_precision", "context_precision_score"],
        "context_recall":    ["context_recall", "context_recall_score"],
    }

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

        skipped = 0
        for item in data:
            if "query" in item and "question" not in item:
                item["question"] = item["query"]
            if "ground_truth" not in item:
                item["ground_truth"] = ""
            if not item.get("ground_truth") and item.get("answer_status") != "pending":
                skipped += 1
        if skipped:
            print(f"⚠️ {skipped} 条样本缺 ground_truth（无法评测 recall），建议先补齐")
        return data

    def _run_ragas(self, samples_data: List[Dict], debug_one: bool = False) -> Tuple[object, str]:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        _judge = LangchainLLMWrapper(_build_judge_llm())
        _emb = LangchainEmbeddingsWrapper(_build_local_emb())

        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy, context_precision, context_recall,
        )

        try:
            from ragas import EvaluationDataset
            from ragas.dataset_schema import SingleTurnSample
            samples = [
                SingleTurnSample(
                    user_input=x["question"], response=x["answer"],
                    retrieved_contexts=x["contexts"], reference=x["ground_truth"],
                ) for x in samples_data
            ]
            eval_ds = EvaluationDataset(samples=samples)
            ragas_version = "0.2.x"
        except ImportError:
            from datasets import Dataset
            eval_ds = Dataset.from_dict({
                "question": [x["question"] for x in samples_data],
                "answer":   [x["answer"]   for x in samples_data],
                "contexts": [x["contexts"] for x in samples_data],
                "ground_truth": [x["ground_truth"] for x in samples_data],
            })
            ragas_version = "0.1.x"

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        try:
            from ragas import RunConfig
            run_config = RunConfig(max_workers=1, timeout=600, max_retries=3)
        except ImportError:
            run_config = None

        if debug_one:
            print("\n[DEBUG] 只跑 1 条，raise_exceptions=True，看真实异常 ↓↓↓")
            ds1 = eval_ds
            try:
                from ragas import EvaluationDataset
                ds1 = EvaluationDataset(samples=[eval_ds.samples[0]])
            except Exception:
                pass
            try:
                ragas_evaluate(
                    dataset=ds1, metrics=metrics, llm=_judge, embeddings=_emb,
                    run_config=run_config, raise_exceptions=True, show_progress=False,
                )
            except Exception as e:
                import traceback
                print(f"\n❌ 真实异常类型: {type(e).__name__}")
                print(f"❌ 真实异常信息: {e}")
                traceback.print_exc()
                raise SystemExit(1)

        result = ragas_evaluate(
            dataset=eval_ds, metrics=metrics, llm=_judge, embeddings=_emb,
            run_config=run_config, raise_exceptions=False, show_progress=True,
        )
        return result, ragas_version

    def evaluate(
        self,
        fill_missing: bool = True,
        sample_limit: Optional[int] = None,
        save_result: bool = True,
        debug_one: bool = False,
    ) -> Dict:
        if not self.dataset:
            return {"error": "数据集为空或全部无效"}

        print(f"\n{'='*64}")
        print(f"RAGAS 评估（本地 Ollama + Rerank）")
        print(f"{'='*64}")
        print(f"数据集: {self.eval_dataset_path}（{len(self.dataset)} 条）")
        print(f"生成模型: {GEN_MODEL}")
        print(f"评判模型: {JUDGE_MODEL}")
        print(f"嵌入模型: {EMB_MODEL}")
        print(f"Rerank  : {RERANK_MODEL}（device={RERANK_DEVICE}, RECALL_K={RECALL_K} → FINAL_K={FINAL_K}）")

        if fill_missing:
            print("\n[步骤1] 填充 contexts / answer ...")
            self.dataset = _run_retrieval_and_answer(self.dataset)

        eval_rows = []
        for x in self.dataset:
            if not x.get("ground_truth"):
                continue
            eval_rows.append({
                **x,
                "contexts": x.get("contexts") or [""],
                "answer":   x.get("answer") or "",
            })
        if sample_limit:
            eval_rows = eval_rows[:sample_limit]

        total = len(self.dataset)
        evaluated = len(eval_rows)
        n_empty_ctx = sum(1 for x in eval_rows if not x["contexts"][0])
        n_empty_ans = sum(1 for x in eval_rows if not x["answer"])

        print(f"\n[步骤2] 进入评估样本: {evaluated} / {total}")
        print(f"   其中: 空 context={n_empty_ctx}  空 answer={n_empty_ans}")
        if evaluated == 0:
            return {"error": "没有可评估样本（全部缺 ground_truth）"}

        print("\n[步骤3] 加载 LLM/Judge/Embedding ...")
        try:
            result, ragas_version = self._run_ragas(eval_rows, debug_one=debug_one)
        except ImportError as e:
            return {"error": f"RAGAS 依赖缺失: {e}"}

        scores: Dict[str, float] = {}
        try:
            df = result.to_pandas()
        except Exception as e:
            print(f"⚠️ to_pandas 失败: {e}")
            df = None
            try:
                scores = {k: float(v) for k, v in dict(result).items() if isinstance(v, (int, float))}
            except Exception:
                scores = {}

        if df is not None:
            import pandas as pd
            meta = {x["question"]: x for x in eval_rows}
            df["id"] = df["user_input"].map(lambda q: meta.get(q, {}).get("id"))
            df["expected_route"] = df["user_input"].map(lambda q: meta.get(q, {}).get("expected_route"))
            df["actual_route"] = df["user_input"].map(lambda q: meta.get(q, {}).get("actual_route"))
            df["key_constraints"] = df["user_input"].map(
                lambda q: ",".join(meta.get(q, {}).get("key_constraints") or [])
            )

            print("\n[DEBUG] df.columns =", df.columns.tolist())

            for metric, candidates in self.METRIC_ALIAS.items():
                col = next((c for c in candidates if c in df.columns), None)
                if col is None:
                    print(f"  ⚠️ {metric}: 列不存在（候选 {candidates}）")
                    continue
                s = pd.to_numeric(df[col], errors="coerce")
                n_nan = int(s.isna().sum())
                mean_v = s.mean()
                print(f"  [{metric}] col={col}  NaN={n_nan}/{len(s)}  mean={mean_v}")
                scores[metric] = float(mean_v) if not math.isnan(mean_v) else float("nan")

            PER_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(PER_SAMPLE_PATH, index=False, encoding="utf-8")
            print(f"✅ 逐条分数: {PER_SAMPLE_PATH}")

            try:
                cm = pd.crosstab(
                    df["expected_route"].fillna("unknown"),
                    df["actual_route"].fillna("unknown"),
                    margins=True,
                )
                cm.to_csv(CONFUSION_PATH, encoding="utf-8")
                print("\n【路由混淆矩阵 expected × actual】")
                print(cm.to_string())
                print(f"✅ 混淆矩阵: {CONFUSION_PATH}")
            except Exception as e:
                print(f"⚠️ 混淆矩阵生成失败: {e}")

            print("\n【按 expected_route 分桶】")
            for r in df["expected_route"].fillna("unknown").unique():
                sub = df[df["expected_route"] == r]
                parts = []
                for m in self.METRIC_NAMES:
                    col = next((c for c in self.METRIC_ALIAS[m] if c in sub.columns), None)
                    if col is not None:
                        v = pd.to_numeric(sub[col], errors="coerce").mean()
                        parts.append(f"{m}={v:.3f}" if not math.isnan(v) else f"{m}=N/A")
                print(f"  {r:8s} n={len(sub):3d}  " + "  ".join(parts))

        valid_scores = [v for v in scores.values() if isinstance(v, (int, float)) and not math.isnan(v)]
        scores["overall_score"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        scores["_n_samples"] = evaluated
        scores["_n_total"] = total
        scores["_n_empty_ctx"] = n_empty_ctx
        scores["_n_empty_ans"] = n_empty_ans
        scores["_ragas_version"] = ragas_version

        if save_result:
            clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in scores.items()}
            RESULT_PATH.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n✅ 汇总分数: {RESULT_PATH}")

        print(f"\n{'='*64}")
        print(f"RAGAS 评估结果")
        print(f"{'='*64}")
        for name in self.METRIC_NAMES:
            v = scores.get(name)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                print(f"  {name:20s} {v:.3f}  {'█' * int(v * 20)}")
            else:
                print(f"  {name:20s} N/A")
        print(f"  {'overall_score':20s} {scores.get('overall_score', 0.0):.3f}")
        print(f"  {'有效样本':20s} {evaluated} / {total}")
        print(f"  {'空context / 空answer':20s} {n_empty_ctx} / {n_empty_ans}")
        print(f"{'='*64}\n")

        return scores


# ═══════════════════════════════════════════════════════════
# 10. 入口
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    evaluator = RAGASEvaluator(eval_dataset_path="data/eval/e2e_dataset_100_new.json")
    if not evaluator.dataset:
        print("❌ 未找到数据集，请检查路径。")
        sys.exit(1)

    # 首次跑全量；如果全 NaN，把 debug_one=True 再跑一次
    result = evaluator.evaluate(fill_missing=True, sample_limit=None, debug_one=False)
    if "error" in result:
        print(f"\n❌ 评估失败: {result['error']}")
        sys.exit(1)