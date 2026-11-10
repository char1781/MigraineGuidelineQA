"""
src/retrieval/es_retriever.py
职责：ES结构化检索，从原代码中拆分出来，逻辑不变

测试方法：
# 方式1：作为模块运行（推荐）
python -m src.retrieval.es_retriever

# 方式2：使用测试脚本
python test_module.py src.retrieval.es_retriever
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
from elasticsearch import Elasticsearch
from typing import List, Dict, Optional
from dotenv import load_dotenv

# 【新增】处理直接运行时的路径问题
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

load_dotenv()


class ESRetriever:
    """ES精确检索器（原代码逻辑完全保留）"""
    
    def __init__(
        self,
        es_host: Optional[str] = None,
        index_name: str = "medical_guidelines",
    ):
        self.es_host = es_host or os.getenv("ES_HOST", "http://localhost:9200")
        self.index_name = index_name
        self.es_client = Elasticsearch([self.es_host])
        
        # 测试连接
        if not self.es_client.ping():
            raise ConnectionError(f"无法连接到Elasticsearch: {self.es_host}")
    
    def search(
        self,
        query: str,
        disease: str = None,
        drug_name: str = None,
        phase: str = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        ES结构化检索（原search_es函数逻辑）
        
        Args:
            query: 查询文本
            disease: 疾病名称过滤
            drug_name: 药物名称过滤
            phase: 治疗阶段过滤
            top_k: 返回结果数量
        
        Returns:
            [{'score': float, 'source': dict}, ...]
        """
        must_clauses = []
        
        # 自由文本查询（多字段匹配）
        if query:
            must_clauses.append({
                "multi_match": {
                    "query": query,
                    "fields": [
                        "intervention.name^3",  # 药物名权重最高
                        "disease^2",
                        "traceability.raw_text_segment",
                        "intervention.frequency",
                        "intervention.modality_specific.drug_specific.dose.recommended.text_original"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO"  # 注：对中文效果有限，保留原逻辑
                }
            })
        
        # 精确过滤
        if disease:
            must_clauses.append({"match": {"disease": disease}})
        
        if drug_name:
            must_clauses.append({
                "match": {
                    "intervention.name": {
                        "query": drug_name,
                        "fuzziness": "AUTO"
                    }
                }
            })
        
        if phase:
            must_clauses.append({"term": {"phase": phase}})
        
        # 构建完整查询
        es_query = {
            "query": {
                "bool": {
                    "must": must_clauses
                }
            },
            "size": top_k,
            "_source": [
                "rule_id",
                "disease",
                "phase",
                "intervention.name",
                "intervention.modality_specific.drug_specific.dose",
                "intervention.frequency",
                "intervention.route",
                "evidence.evidence_level",
                "evidence.recommendation_grade",
                "safety.adverse_effects",
                "source.guideline",
                "source.year",
                "traceability.raw_text_segment"
            ]
        }
        
        # 执行查询
        response = self.es_client.search(index=self.index_name, body=es_query)
        
        # 格式化结果（原逻辑）
        results = []
        for hit in response['hits']['hits']:
            results.append({
                "score": hit['_score'],
                "source": hit['_source']
            })
        
        return results
    
    def format_context(self, results: List[Dict]) -> str:
        """
        格式化ES检索结果为上下文（原format_es_context函数）
        
        Returns:
            多段参考文本的拼接
        """
        if not results:
            return "未找到相关信息"
        
        context_parts = []
        for i, result in enumerate(results, 1):
            doc = result['source']
            
            intervention = doc.get('intervention', {})
            drug_name = intervention.get('name', 'N/A')
            
            # 基本信息
            disease = doc.get('disease', 'N/A')
            phase = doc.get('phase', 'N/A')
            
            # 剂量
            drug_spec = intervention.get('modality_specific', {}).get('drug_specific', {})
            dose = drug_spec.get('dose', {})
            
            recommended = dose.get('recommended', {})
            dose_text = recommended.get('text_original', 'N/A')
            dose_unit = recommended.get('unit', 'mg')
            if dose_text and dose_text != 'N/A':
                if not any(u in dose_text for u in ['mg', 'g', 'ml', 'l']):
                    dose_text = f"{dose_text}{dose_unit}"
            
            max_daily = dose.get('max_daily', {})
            max_text = max_daily.get('text_original', 'N/A')
            max_unit = max_daily.get('unit', 'mg')
            if max_text and max_text != 'N/A':
                if not any(u in max_text for u in ['mg', 'g', 'ml', 'l']):
                    max_text = f"{max_text}{max_unit}"
            
            frequency = intervention.get('frequency', 'N/A')
            route = intervention.get('route', {}).get('description', 'N/A')
            
            # 证据
            evidence = doc.get('evidence', {})
            evidence_level = evidence.get('evidence_level', 'N/A')
            recommendation_grade = evidence.get('recommendation_grade', 'N/A')
            
            # 安全性
            safety = doc.get('safety', {})
            adverse = safety.get('adverse_effects', {})
            common_ae = adverse.get('common', 'N/A')
            
            # 来源
            source = doc.get('source', {})
            guideline = source.get('guideline', 'N/A')
            year = source.get('year', 'N/A')
            
            context = f"""
                    【参考{i}】
                    药物：{drug_name}
                    疾病：{disease} | 阶段：{phase}
                    推荐剂量：{dose_text}
                    最大日剂量：{max_text}
                    给药频次：{frequency}
                    给药途径：{route}
                    证据等级：{evidence_level}
                    推荐强度：{recommendation_grade}
                    常见不良反应：{common_ae}
                    来源：{guideline} ({year}年)
                """
            context_parts.append(context.strip())
        
        return "\n\n".join(context_parts)
    
    def print_results(self, results: List[Dict], query: str):
        """
        打印ES检索结果（原print_es_results函数）
        用于调试
        """
        print("\n" + "=" * 80)
        print(f"🔍 ES检索查询: {query}")
        print("=" * 80)
        
        if not results:
            print("❌ 未找到相关结果")
            return
        
        print(f"✅ 找到 {len(results)} 条结果\n")
        
        for i, result in enumerate(results, 1):
            score = result['score']
            doc = result['source']
            
            print(f"{'─' * 80}")
            print(f"📄 结果 {i} (相关性分数: {score:.4f})")
            print(f"{'─' * 80}")
            
            # 基本信息
            print(f"💊 药物: {doc.get('intervention', {}).get('name', 'N/A')}")
            print(f"🏥 疾病: {doc.get('disease', 'N/A')}")
            print(f"⏱️  阶段: {doc.get('phase', 'N/A')}")
            
            # 剂量信息
            drug_spec = doc.get('intervention', {}).get('modality_specific', {}).get('drug_specific', {})
            dose = drug_spec.get('dose', {})
            
            recommended = dose.get('recommended', {})
            if recommended:
                dose_text = recommended.get('text_original', 'N/A')
                dose_unit = recommended.get('unit', 'mg')
                if dose_text:
                    if not any(u in dose_text for u in ['mg', 'g', 'ml', 'l']):
                        dose_text = f"{dose_text}{dose_unit}"
                    print(f"💉 推荐剂量: {dose_text}")
                else:
                    print(f"💉 推荐剂量: N/A")
            
            max_daily = dose.get('max_daily', {})
            if max_daily:
                max_text = max_daily.get('text_original', 'N/A')
                max_unit = max_daily.get('unit', 'mg')
                
                if max_text:
                    if not any(u in max_text for u in ['mg', 'g', 'ml', 'l']):
                        max_text = f"{max_text}{max_unit}"
                    print(f"⚠️  最大日剂量: {max_text}")
                else:
                    print(f"⚠️  最大日剂量: N/A")
            
            # 频次和途径
            frequency = doc.get('intervention', {}).get('frequency', 'N/A')
            route = doc.get('intervention', {}).get('route', {}).get('description', 'N/A')
            print(f"🔄 给药频次: {frequency}")
            print(f"💊 给药途径: {route}")
            
            # 证据等级
            evidence = doc.get('evidence', {})
            evidence_level = evidence.get('evidence_level', 'N/A')
            recommendation_grade = evidence.get('recommendation_grade', 'N/A')
            print(f"📊 证据等级: {evidence_level}")
            print(f"⭐ 推荐强度: {recommendation_grade}")
            
            # 不良反应
            safety = doc.get('safety', {})
            adverse = safety.get('adverse_effects', {})
            common_ae = adverse.get('common', 'N/A')
            if common_ae and common_ae != 'N/A':
                print(f"⚠️  常见不良反应: {common_ae}")
            
            # 来源
            source = doc.get('source', {})
            guideline = source.get('guideline', 'N/A')
            year = source.get('year', 'N/A')
            print(f"📚 来源: {guideline} ({year}年)")
            
            # 原始文本
            raw_text = doc.get('traceability', {}).get('raw_text_segment', '')
            if raw_text:
                print(f"\n📝 原文:")
                print(f"   {raw_text[:200]}..." if len(raw_text) > 200 else f"   {raw_text}")
            
            print()


# ── 单元测试 ──
if __name__ == "__main__":
    try:
        retriever = ESRetriever()
        
        # 测试查询
        test_query = "利扎曲普坦的推荐剂量是多少？"
        results = retriever.search(test_query, top_k=3)
        
        retriever.print_results(results, test_query)
        
        # 测试上下文格式化
        print("\n" + "=" * 80)
        print("格式化上下文:")
        print("=" * 80)
        context = retriever.format_context(results)
        print(context)
        
    except ConnectionError as e:
        print(f"❌ {e}")
        print("请先启动Elasticsearch服务")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
