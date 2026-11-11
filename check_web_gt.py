import os
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

queries = [
    # "2026年新发布的偏头痛指南是什么？",
    # "最新的CGRP靶向药物在国内上市了吗？",
    "2026年有什么偏头痛新共识？",
    # "偏头痛非药物治疗的最新进展有哪些？",
    "2026年偏头痛生物标志物研究有什么新突破？",
    "2026年世界头痛大会有什么关于偏头痛的重要报告？",
    # "国产偏头痛新药最新获批情况？",
    # "2026年最新的偏头痛基因治疗研究进展？",
    # "最新的偏头痛AI辅助诊断产品有哪些？",
    # "2026年医保目录中纳入了哪些偏头痛新药？"
]

searcher = TavilySearch(max_results=2, tavily_api_key=os.getenv("TAVILY_API_KEY"))

for i, q in enumerate(queries, 1):
    print(f"\n{'='*50}")
    print(f"问题 {i}: {q}")
    try:
        res = searcher.invoke(q)
        
        # ★ 修复：兼容 Dict 格式
        items = []
        if isinstance(res, dict) and "results" in res:
            items = res["results"]
        elif isinstance(res, list):
            items = res
            
        if items:
            for j, item in enumerate(items[:2]):
                print(f"  [网页 {j+1}]: {item.get('title', '无标题')}")
                content = item.get('content', '').replace('\n', ' ')
                # 打印前 1200 字足够提炼了
                print(f"  摘要: {content[:1200]}...")
        else:
            print("  未找到网页结果")
            
    except Exception as e:
        print(f"  搜索失败: {e}")