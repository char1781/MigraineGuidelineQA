import json
import time
from pathlib import Path
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from src.retrieval.router import auto_detect_with_strategy

# 初始化裁判大模型（建议用 qwen 7b 或 14b）
judge_llm = ChatOpenAI(
    model="qwen2.5:14b-instruct-q4_K_M",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0,
    timeout=120
)

def evaluate_with_judge(query, ground_truth, generated_answer):
    """裁判打分函数"""
    prompt = f"""你是一个严格的医疗质控裁判。请判断【系统回答】是否在核心医学事实上与【标准答案】一致。
注意：系统回答只要包含了标准答案中的核心药物、剂量或结论，就算 PASS。不需要字面完全一样。

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
        return is_pass
    except Exception as e:
        print(f"裁判评估失败: {e}")
        return False

def main():
    dataset_path = Path("data/eval/eval_dataset_100_filled.json")
    if not dataset_path.exists():
        print(f"❌ 找不到数据集: {dataset_path}")
        return

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    print(f"✅ 成功加载 {len(dataset)} 条数据，开始端到端评测...")

    route_pass_count = 0
    e2e_pass_count = 0
    valid_count = 0

    for idx, item in enumerate(tqdm(dataset, desc="E2E Judging")):
        q = item.get("question") or item.get("query")
        gt = item.get("ground_truth")
        answer = item.get("answer")
        
        # 1. 路由命中率测试
        expected_route = item.get("expected_route", "unknown")
        actual_route = auto_detect_with_strategy(q, strategy="hybrid")
        if expected_route == actual_route:
            route_pass_count += 1
            
        # 2. 裁判大模型测试
        # 跳过没有回答或无效回答的样本
        if not answer or "根据当前资料无法确定" in answer:
            continue
            
        valid_count += 1
        is_pass = evaluate_with_judge(q, gt, answer)
        if is_pass:
            e2e_pass_count += 1

    # 计算最终指标
    total = len(dataset)
    route_acc = (route_pass_count / total) * 100
    e2e_acc = (e2e_pass_count / valid_count) * 100 if valid_count > 0 else 0

    print(f"\n{'='*50}")
    print(f"📊 100条全量端到端评测报告")
    print(f"{'='*50}")
    print(f"  ➤ 路由命中率: {route_acc:.1f}% ({route_pass_count}/{total})")
    print(f"  ➤ 有效回答数: {valid_count}/{total} (排除了'无法确定'的样本)")
    print(f"  ➤ 事实回答准确率: {e2e_acc:.1f}% ({e2e_pass_count}/{valid_count})")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()