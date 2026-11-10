"""
test_e2e.py - 端到端（E2E）系统集成评测脚本
职责：模拟用户真实查询，调用路由、检索、生成链路，并使用 LLM-as-a-Judge 打分。
运行：python test_e2e.py
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 确保能导入 src 里的模块
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

# 初始化裁判 LLM（使用本地的 Qwen，也可以换成 API）
judge_llm = ChatOpenAI(
    model="qwen2.5:14b-instruct-q4_K_M",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0, # 裁判必须严谨，温度设为0
)

# ==========================================
# 1. 构造覆盖多路检索的测试集
# ==========================================
TEST_DATASET = [
    {
        "query": "利扎曲普坦的推荐剂量是多少？",
        "expected_route": "es",
        "ground_truth": "5-10mg，24小时内最大剂量不超过30mg。",
    },
    {
        "query": "偏头痛急性期有哪些治疗方法？",
        "expected_route": "hybrid",
        "ground_truth": "特异性治疗（曲普坦类、地坦类、吉泮类）和非特异性治疗（NSAIDs、对乙酰氨基酚）。",
    },
    {
        "query": "癫痫共病偏头痛如何用药？",
        "expected_route": "graph",
        "ground_truth": "丙戊酸钠兼具抗癫痫和偏头痛预防双重适应症。",
    },
    {
        "query": "2026年最新的偏头痛指南有什么更新？",
        "expected_route": "web",
        "ground_truth": "2026年发布了《中国前庭性偏头痛诊治指南2026版》等。",
    },
]

# ==========================================
# 2. 模拟 App.py 里的真实路由与生成逻辑
# ==========================================
def run_system_pipeline(query: str) -> dict:
    """跑通从路由到最终生成的全链路，返回：生成的答案、实际走的路由"""
    from src.retrieval.router import auto_detect_with_strategy
    
    # 路由分发
    strategy = os.getenv("INTENT_DETECTION_STRATEGY", "hybrid")
    actual_route = auto_detect_with_strategy(query, strategy=strategy)
    
    context = ""
    
    # 走对应的检索器
    if actual_route == "es":
        from src.retrieval.es_retriever import ESRetriever
        retriever = ESRetriever()
        results = retriever.search(query, top_k=3)
        context = retriever.format_context(results)
        
    elif actual_route == "hybrid":
        from src.retrieval.vector_retriever import VectorRetriever
        retriever = VectorRetriever()
        nodes = retriever.search(query, top_k=3)
        context = retriever.format_context(nodes)
        
    elif actual_route == "graph":
        from src.retrieval.graph_retriever import GraphRetriever
        retriever = GraphRetriever()
        retriever.timeout = 300
        try:
            results = retriever.search(query, top_k=3)
            if results and results[0].get("type") in ("graph_error", "graph_status"):
                raise Exception(results[0].get("message"))
            context = retriever.format_context(results)
        except Exception as e:
            # E2E测试中如果图谱挂了，简单降级记录
            context = f"[图谱服务不可用，降级报错]: {e}"
            
    elif actual_route == "web":
        from langchain_tavily import TavilySearch
        searcher = TavilySearch(max_results=2, tavily_api_key=os.getenv("TAVILY_API_KEY"))
        try:
            result = searcher.invoke(query)
            if isinstance(result, list):
                context = "\n\n".join(item.get("content", "") for item in result[:3])
            else:
                context = str(result)
        except Exception as e:
            context = f"[网络搜索失败]: {e}"

    # 最终生成回答（使用本地大模型）
    generator_prompt = f"""请严格根据以下资料回答用户问题，不要编造。
资料：
{context}

问题：{query}
回答："""
    
    try:
        final_answer = judge_llm.invoke(generator_prompt).content.strip()
    except Exception as e:
        final_answer = f"[生成回答失败]: {e}"

    return {
        "actual_route": actual_route,
        "final_answer": final_answer
    }

# ==========================================
# 3. LLM-as-a-Judge 裁判打分函数
# ==========================================
def evaluate_with_judge(query: str, ground_truth: str, generated_answer: str) -> dict:
    """让大模型自己当裁判，判断生成的答案是否包含标准答案的核心事实"""
    prompt = f"""你是一个严格的医疗质控裁判。请判断【系统回答】是否在核心医学事实上与【标准答案】一致。
注意：系统回答只要包含了标准答案中的核心药物或结论，就算 PASS。不需要字面完全一样。

问题：{query}
标准答案：{ground_truth}
系统回答：{generated_answer}

请严格按照以下格式输出：
结果：PASS 或 FAIL
理由：简单一句话解释
"""
    try:
        response = judge_llm.invoke(prompt).content.strip()
        is_pass = "结果：PASS" in response or "结果: PASS" in response or "PASS" in response.split("\n")[0]
        return {
            "is_pass": is_pass,
            "judge_reason": response.replace("\n", " ")
        }
    except Exception as e:
        return {"is_pass": False, "judge_reason": f"裁判评估失败: {e}"}

# ==========================================
# 4. 主流程执行
# ==========================================
def main():
    print(f"\n{'='*60}")
    print(f"🚀 启动 RAG 系统端到端 (E2E) 综合评测")
    print(f"{'='*60}\n")
    
    results = []
    pass_count = 0
    route_correct_count = 0
    
    for i, item in enumerate(TEST_DATASET, 1):
        query = item["query"]
        print(f"[{i}/{len(TEST_DATASET)}] 测试问题: {query}")
        
        start_time = time.time()
        
        # 1. 跑完整系统
        sys_result = run_system_pipeline(query)
        actual_route = sys_result["actual_route"]
        final_answer = sys_result["final_answer"]
        
        # 2. 判断路由是否命中
        route_correct = (actual_route == item["expected_route"])
        if route_correct:
            route_correct_count += 1
            
        print(f"  ├─ 路由判定: 期望[{item['expected_route']}] -> 实际[{actual_route}] {'✅' if route_correct else '❌'}")
        print(f"  ├─ 系统回答: {final_answer[:60]}...")
        
        # 3. 裁判打分
        judge_result = evaluate_with_judge(query, item["ground_truth"], final_answer)
        if judge_result["is_pass"]:
            pass_count += 1
            
        print(f"  └─ 裁判判定: {'✅ PASS' if judge_result['is_pass'] else '❌ FAIL'} | {judge_result['judge_reason']}")
        print(f"     (耗时 {time.time() - start_time:.1f}s)\n")
        
        results.append({
            "query": query,
            "expected_route": item["expected_route"],
            "actual_route": actual_route,
            "final_answer": final_answer,
            "ground_truth": item["ground_truth"],
            "route_correct": route_correct,
            "e2e_pass": judge_result["is_pass"],
            "judge_reason": judge_result["judge_reason"]
        })

    # 统计结果
    total = len(TEST_DATASET)
    route_acc = (route_correct_count / total) * 100
    e2e_acc = (pass_count / total) * 100
    
    print(f"{'='*60}")
    print(f"📊 最终评测报告")
    print(f"{'='*60}")
    print(f"  ➤ 路由命中率 (Router Accuracy): {route_acc:.1f}%  ({route_correct_count}/{total})")
    print(f"  ➤ 系统准确率 (E2E Accuracy):    {e2e_acc:.1f}%  ({pass_count}/{total})")
    print(f"{'='*60}\n")
    
    # 保存结果到 JSON
    output_path = Path("data/eval/e2e_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "metrics": {"route_accuracy": route_acc, "e2e_accuracy": e2e_acc},
        "details": results
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"✅ 详细评测报告已保存至: {output_path}")

if __name__ == "__main__":
    main()