"""
src/evaluation/ragas_eval.py
职责：使用RAGAS框架评估RAG系统

评估指标：
1. Answer Relevancy（回答相关性）：回答是否切中问题
2. Faithfulness（忠实性）：回答是否忠于检索文档（医疗场景最关键）
3. Context Precision（上下文精确度）：检索到的文档是否相关
4. Context Recall（上下文召回率）：是否召回了所有相关信息

数据集要求：
- 20-30条标注问答对
- 包含：question, answer, contexts, ground_truth

测试方法：
python -m src.evaluation.ragas_eval
或
python src/evaluation/ragas_eval.py
"""

import sys
from pathlib import Path

# 【关键】必须在任何src模块导入之前处理路径
if __package__ is None or __package__ == '':
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from typing import List, Dict
import json


class RAGASEvaluator:
    """RAGAS评估器（占位实现）"""
    
    def __init__(self, eval_dataset_path: str = "data/eval/eval_dataset.json"):
        self.eval_dataset_path = Path(eval_dataset_path)
        self.dataset = self._load_dataset()
    
    def _load_dataset(self) -> List[Dict]:
        """
        加载评估数据集
        
        格式：
        [
            {
                "question": "利扎曲普坦的推荐剂量是多少？",
                "ground_truth": "5-10mg，最大日剂量30mg",
                "contexts": ["参考文档1", "参考文档2"],  # 真实检索结果
                "answer": "根据指南..."  # 系统实际回答
            }
        ]
        """
        if not self.eval_dataset_path.exists():
            return []
        
        return json.loads(self.eval_dataset_path.read_text(encoding='utf-8'))
    
    def evaluate(self) -> Dict[str, float]:
        """
        运行评估
        
        Returns:
            {
                "answer_relevancy": 0.85,
                "faithfulness": 0.92,
                "context_precision": 0.78,
                "context_recall": 0.81,
                "overall_score": 0.84
            }
        """
        # TODO: 集成RAGAS库
        # from ragas import evaluate
        # from ragas.metrics import answer_relevancy, faithfulness, ...
        
        # results = evaluate(
        #     dataset=self.dataset,
        #     metrics=[answer_relevancy, faithfulness, context_precision, context_recall]
        # )
        
        # 占位返回
        return {
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "overall_score": 0.0,
            "note": "需要标注评估数据集并集成RAGAS库"
        }
    
    def create_eval_template(self) -> str:
        """
        创建评估数据集模板
        
        Returns:
            JSON模板路径
        """
        template = [
            {
                "question": "利扎曲普坦的推荐剂量是多少？",
                "ground_truth": "5-10mg，每次口服，2小时无效可追加，最大日剂量30mg",
                "contexts": [],  # 运行检索后填入
                "answer": "",     # 运行Agent后填入
                "metadata": {
                    "category": "剂量查询",
                    "expected_retrieval": "es"
                }
            },
            {
                "question": "偏头痛急性期有哪些治疗方法？",
                "ground_truth": "特异性：曲普坦类、地坦类、吉泮类；非特异性：NSAIDs、对乙酰氨基酚",
                "contexts": [],
                "answer": "",
                "metadata": {
                    "category": "治疗方案",
                    "expected_retrieval": "hybrid"
                }
            },
            # ... 继续添加到20-30条
        ]
        
        output_path = self.eval_dataset_path.parent / "eval_template.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(template, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        
        return str(output_path)


# ── 使用示例 ──
if __name__ == "__main__":
    evaluator = RAGASEvaluator()
    
    # 创建评估数据集模板
    template_path = evaluator.create_eval_template()
    print(f"✅ 评估数据集模板已创建: {template_path}")
    print("\n下一步:")
    print("1. 填充 contexts 和 answer 字段（运行检索和Agent）")
    print("2. 安装RAGAS: pip install ragas")
    print("3. 运行评估: evaluator.evaluate()")
