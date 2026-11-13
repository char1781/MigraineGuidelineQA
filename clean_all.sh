#!/bin/bash

# ====================================================
# 一键清理医疗 RAG 系统的所有缓存和测试数据
# 运行，先给脚本添加执行权限，然后运行脚本
# chmod +x clean_all.sh
# ./clean_all.sh
# ====================================================

echo "🚀 开始清理系统环境..."

# 1. 清空 Elasticsearch
echo "⏳ 正在清空 ES 索引..."
python - <<EOF
from elasticsearch import Elasticsearch
try:
    es = Elasticsearch(['http://localhost:9200'])
    if es.indices.exists(index='medical_guidelines'):
        es.indices.delete(index='medical_guidelines')
        print("  ✅ ES 索引已彻底清空")
    else:
        print("  ℹ️ ES 索引不存在，无需清理")
except Exception as e:
    print(f"  ⚠️ ES 清理失败 (请检查 ES 是否启动): {e}")
EOF

# 2. 清空本地文件产物
echo "⏳ 正在清空本地文件产物..."
rm -rf data/chroma_db/*
rm -rf data/processed/*
rm -rf data/review/*
rm -f data/kb_metadata.json
echo "  ✅ Chroma、MinerU 产物、审核记录及元数据已清空"

# 3. 清空 GraphRAG 的输入和输出
echo "⏳ 正在清空 GraphRAG 目录..."
GRAPHRAG_DIR=~/Downloads/graphrag-main

if [ -d "$GRAPHRAG_DIR" ]; then
    rm -rf ${GRAPHRAG_DIR}/input/*
    rm -rf ${GRAPRAG_DIR}/output/*
    echo "  ✅ GraphRAG input 和 output 已清空"
else
    echo "  ⚠️ 未找到 GraphRAG 目录 ($GRAPHRAG_DIR)，请确认路径是否正确"
fi

# # 4. 清空 RAGAS 评测缓存
# echo "⏳ 正在清空 RAGAS 评测缓存..."
# rm -f data/eval/eval_dataset_filled.json
# rm -f data/eval/eval_result.json
# rm -f data/eval/eval_per_sample.csv
# echo "  ✅ 评测缓存已清空"

echo "===================================================="
echo "🎉 系统环境重置完成！可以重新启动 Streamlit 上传数据了。"
echo "===================================================="