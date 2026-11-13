import pandas as pd
import json

# 1. 读取评测数据
df = pd.read_csv("data/eval/eval_per_sample.csv")
dataset = json.loads(open("data/eval/eval_dataset_100_filled_v3.json", "r", encoding="utf-8").read())

# 2. 把路由类型映射回 DataFrame
# 因为原始 CSV 里可能没有路由字段，我们从 JSON 里按 user_input 匹配
route_map = {item["query"]: item["expected_route"] for item in dataset}
df["route"] = df["user_input"].map(route_map)

# 3. 按路由分组计算平均分
metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
summary = df.groupby("route")[metrics].mean().round(3)
counts = df.groupby("route").size()

print("\n" + "="*50)
print("📊 按路由通道拆分的 RAGAS 评测报告")
print("="*50)

for route in ["es", "hybrid", "graph", "web"]:
    if route in summary.index:
        print(f"\n▶ 【{route.upper()} 通道】 (样本数: {counts[route]} 条)")
        for m in metrics:
            score = summary.loc[route, m]
            if pd.isna(score):
                print(f"   {m:20s} N/A    (该通道全部样本计算失败，检查判分模型上下文长度)")
                continue
            bar = "█" * int(score * 20)
            print(f"   {m:20s} {score:.3f}  {bar}")
        valid = df[df["route"] == route][metrics].notna().all(axis=1).sum()
        print(f"   有效样本(四项均有值): {valid}/{counts[route]}")