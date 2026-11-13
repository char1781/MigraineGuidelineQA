import pandas as pd
import json

# ═══════════════════════════════════════════════════════════
# 1. 读数据
# ═══════════════════════════════════════════════════════════
CSV_PATH = "data/eval/eval_per_sample.csv"
JSON_PATH = "data/eval/eval_dataset_filled.json"   # ★ 改成实际产物名

df = pd.read_csv(CSV_PATH)

# ★ 关键：CSV 里已有 expected_route / actual_route，不再从 JSON 反查
# 如果 CSV 是旧版本没有这两列，才走 fallback
if "expected_route" not in df.columns or "actual_route" not in df.columns:
    print("⚠️ CSV 缺少 route 列，从 JSON 反查（不推荐，query 字符串差异会静默失配）")
    dataset = json.loads(open(JSON_PATH, "r", encoding="utf-8").read())
    route_map = {item["query"]: item for item in dataset}
    df["expected_route"] = df["user_input"].map(lambda q: route_map.get(q, {}).get("expected_route", "unknown"))
    df["actual_route"]   = df["user_input"].map(lambda q: route_map.get(q, {}).get("actual_route",   "unknown"))

# ★ 把 answerable_in_kb 从 JSON 合并进来（按 id 更稳，不按 query）
try:
    dataset = json.loads(open(JSON_PATH, "r", encoding="utf-8").read())
    kb_map = {item["id"]: item.get("answerable_in_kb", "unknown") for item in dataset if "id" in item}
    df["answerable_in_kb"] = df["id"].map(kb_map).fillna("unknown").astype(str)
except FileNotFoundError:
    print(f"⚠️ {JSON_PATH} 不存在，跳过 answerable_in_kb 分组")
    df["answerable_in_kb"] = "unknown"

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# ★ 强制转数值，把字符串 "NaN" 也吃掉
for m in METRICS:
    if m in df.columns:
        df[m] = pd.to_numeric(df[m], errors="coerce")


# ═══════════════════════════════════════════════════════════
# 2. 工具函数
# ═══════════════════════════════════════════════════════════
def _print_bucket(title: str, sub: pd.DataFrame, show_nan: bool = True):
    n = len(sub)
    if n == 0:
        return
    print(f"\n▶ 【{title}】 (样本数: {n} 条)")
    for m in METRICS:
        if m not in sub.columns:
            continue
        s = sub[m]
        v = s.mean()
        n_nan = int(s.isna().sum())
        if pd.isna(v):
            print(f"   {m:20s} N/A   (NaN={n_nan}/{n})")
        else:
            bar = "█" * int(v * 20)
            nan_note = f"  (NaN={n_nan}/{n})" if show_nan and n_nan > 0 else ""
            print(f"   {m:20s} {v:.3f}  {bar}{nan_note}")


# ═══════════════════════════════════════════════════════════
# 3. 总览
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 RAGAS 评测报告（含 expected / actual / answerable 三维诊断）")
print("=" * 60)
_print_bucket("全量", df)


# ═══════════════════════════════════════════════════════════
# 4. ★ 按 actual_route 分桶（这才是系统真实表现的视图）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 按 actual_route 分桶（系统实际走的路）")
print("=" * 60)
for route in ["es", "hybrid", "graph", "web", "unknown"]:
    sub = df[df["actual_route"] == route]
    _print_bucket(route.upper(), sub)


# ═══════════════════════════════════════════════════════════
# 5. 按 expected_route 分桶（你原来用的视图，保留对比）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 按 expected_route 分桶（系统该走的路）")
print("=" * 60)
for route in ["es", "hybrid", "graph", "web", "unknown"]:
    sub = df[df["expected_route"] == route]
    _print_bucket(route.upper(), sub)


# ═══════════════════════════════════════════════════════════
# 6. ★ 按 answerable_in_kb 分桶（区分"KB 没有"vs"检索差"）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 按 answerable_in_kb 分桶")
print("=" * 60)
for flag in ["true", "false", "needs_check", "unknown"]:
    sub = df[df["answerable_in_kb"].str.lower() == flag]
    if len(sub) > 0:
        _print_bucket(f"answerable_in_kb={flag}", sub)


# ═══════════════════════════════════════════════════════════
# 7. ★★ 交叉表：expected × actual × answerable，定位问题根源
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 交叉诊断表")
print("=" * 60)

# 7.1 路由混淆矩阵
print("\n【expected × actual 混淆矩阵】")
cm = pd.crosstab(df["expected_route"].fillna("unknown"),
                 df["actual_route"].fillna("unknown"),
                 margins=True)
print(cm.to_string())
acc = sum(df["expected_route"] == df["actual_route"]) / len(df)
print(f"\n路由准确率: {acc:.1%}")

# 7.2 route × answerable 的 recall 交叉
print("\n【actual_route × answerable_in_kb  的 context_recall 均值】")
try:
    pivot = df.pivot_table(
        index="actual_route",
        columns="answerable_in_kb",
        values="context_recall",
        aggfunc="mean",
    ).round(3)
    print(pivot.to_string())
except Exception as e:
    print(f"⚠️ 交叉表生成失败: {e}")

# 7.3 route × answerable 的样本数
print("\n【actual_route × answerable_in_kb  的样本数】")
try:
    count_pivot = df.pivot_table(
        index="actual_route",
        columns="answerable_in_kb",
        values="context_recall",
        aggfunc="count",
    ).fillna(0).astype(int)
    print(count_pivot.to_string())
except Exception as e:
    print(f"⚠️ 交叉表生成失败: {e}")

# 7.4 只对 answerable_in_kb=true 的题重算 route 分桶（排除 KB 覆盖缺口）
print("\n【仅 answerable_in_kb=true 的题，按 actual_route 分桶】")
df_kb = df[df["answerable_in_kb"].str.lower() == "true"]
for route in ["es", "hybrid", "graph", "web"]:
    sub = df_kb[df_kb["actual_route"] == route]
    _print_bucket(f"{route.upper()} (KB 有答案)", sub)


# ═══════════════════════════════════════════════════════════
# 8. 落盘：给后续调优用的明细
# ═══════════════════════════════════════════════════════════
diag_cols = ["id", "expected_route", "actual_route", "answerable_in_kb"] + METRICS
diag_cols = [c for c in diag_cols if c in df.columns]
df[diag_cols].to_csv("data/eval/diag_by_route.csv", index=False, encoding="utf-8")
print(f"\n✅ 诊断明细已保存: data/eval/diag_by_route.csv")