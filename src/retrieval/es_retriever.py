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
import re
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


# ───────────────────────────────────────────────────────────
# 按提问意图裁剪 ES 字段
# 召回阶段返回完整 _source（不裁字段，避免丢检索信号），
# 只在拼给用户/LLM 的文本时按意图取舍。
# ───────────────────────────────────────────────────────────
DOSE_KEYWORDS = [
    "剂量", "用量", "用法", "怎么吃", "如何服用", "服用", "口服", "给药",
    "频次", "一天几次", "每日", "每次", "单次", "最大", "上限", "mg", "毫克",
]
ADVERSE_KEYWORDS = [
    "不良反应", "副作用", "不良事件", "安全性", "安全吗", "耐受",
    "常见吗", "是否常见", "便秘", "恶心", "呕吐", "头晕", "嗜睡",
    "乏力", "体重增加", "口干", "心悸", "皮疹",
]
CONTRA_KEYWORDS = [
    "禁忌", "禁用", "不能用", "慎用", "不宜", "注意事项",
    "能用", "能吃", "可以吃", "可以服用", "能否使用", "可以用吗", "能不能用",
    "可以使用", "可以使用吗", "是否能用", "能否服用",
]
EVIDENCE_KEYWORDS = ["证据等级", "证据级别", "推荐强度", "推荐等级", "推荐级别", "循证", "grade"]


def detect_es_intents(query: str) -> List[str]:
    """识别提问关心的字段，可同时命中多个意图（如"剂量和副作用"）"""
    q = (query or "").lower()
    intents = []
    if any(k in q for k in DOSE_KEYWORDS):
        intents.append("dose")
    if any(k in q for k in ADVERSE_KEYWORDS):
        intents.append("adverse")
    if any(k in q for k in CONTRA_KEYWORDS):
        intents.append("contraindication")
    if any(k in q for k in EVIDENCE_KEYWORDS):
        intents.append("evidence")
    return intents


def _doc_contra_text(doc: dict) -> str:
    contra = (doc.get("safety") or {}).get("contraindications") or []
    if isinstance(contra, list):
        contra = "、".join(str(c) for c in contra if c)
    return str(contra or "") + " " + str(doc.get("traceability", {}).get("raw_text_segment") or "")


POPULATION_HINTS = ["孕", "妊娠", "哺乳", "儿童", "青少年", "老年"]


def es_needs_fallback(query: str, results: List[Dict]) -> bool:
    """判断 ES 结果是否不足以回答问题，调用方应降级到语义检索。

    触发条件：
    1. 没有召回任何文档；
    2. 提问点明了阶段（急性期/预防期/孕期…），但结果里没有任何该阶段的文档；
    3. 问禁忌/不良反应/剂量，但结果里对应字段全为空；
    4. 问特殊人群（孕/哺乳/儿童/老年），但结果里找不到该人群相关信息
       （这类信息多在指南正文段落，ES 只抽治疗表格，天然缺失）。
    """
    if not results:
        return True

    docs = [r.get("source") or r for r in results]

    phase = detect_phase_filter(query)
    if phase and not any((d.get("phase") or "") == phase for d in docs):
        return True

    intents = detect_es_intents(query)
    if "contraindication" in intents and not any(_doc_contra_text(d).strip() for d in docs):
        return True
    if "adverse" in intents and not any(
        ((d.get("safety") or {}).get("adverse_effects") or {}).get("common") for d in docs
    ):
        return True
    if "dose" in intents and not any(
        (((d.get("intervention") or {}).get("modality_specific") or {})
         .get("drug_specific") or {}).get("dose", {}).get("recommended")
        for d in docs
    ):
        return True

    hints = [h for h in POPULATION_HINTS if h in (query or "")]
    if hints:
        # 问的是某个具体药物时，只在这味药自己的文档里找该人群信息，
        # 避免被结果中其他药物的"孕妇禁用"误判为已覆盖
        named = [
            d for d in docs
            if (d.get("intervention") or {}).get("name")
            and (d["intervention"]["name"].split("(")[0] in query)
        ]
        scope = named or docs
        if not any(all(h in _doc_contra_text(d) for h in hints) for d in scope):
            return True

    return False


def _fmt_dose_item(item, unit_hint: str = "mg") -> str:
    """剂量对象 → 可读文本（缺单位时补上）"""
    if not isinstance(item, dict):
        return ""
    text = item.get("text_original") or ""
    unit = item.get("unit") or unit_hint
    if not text:
        return ""
    if unit and not any(u in text for u in ["mg", "g", "ml", "l", "U", "片", "mmol"]):
        text = f"{text} {unit}"
    return text


def detect_phase_filter(query: str) -> Optional[str]:
    """从提问中解析治疗阶段，用于给 ES 查询加 phase 过滤条件。

    顺序按「越具体越优先」，避免"慢性偏头痛急性发作"被判成急性期。
    返回 None 表示用户没有指定阶段。
    """
    q = (query or "").lower()
    # "非急性期预防性治疗"这类否定说法，不能按字面判成急性期
    if any(k in q for k in ["非急性期", "非发作期", "不是急性期"]):
        return "预防期"
    phase_keywords = [
        ("慢性偏头痛", ["慢性偏头痛", "慢性头痛"]),
        ("月经性偏头痛", ["月经性偏头痛", "月经性"]),
        ("围经期", ["围经期", "经期偏头痛", "月经期偏头痛", "经期"]),
        ("妊娠期", ["妊娠期", "妊娠", "怀孕", "孕期"]),
        ("哺乳期", ["哺乳期", "哺乳", "喂奶"]),
        ("急性期", ["急性期", "急性发作", "急性治疗", "发作期", "发作时", "止痛", "acute"]),
        ("预防期", ["预防期", "预防性治疗", "预防治疗", "预防用药", "预防", "prophyla", "prevent"]),
    ]
    for phase, keywords in phase_keywords:
        if any(k in q for k in keywords):
            return phase
    return None


def clean_guideline_name(name: str) -> str:
    """去掉指南名前面的文件哈希/UUID 与 CN 前缀，便于直接展示给用户"""
    name = name or ""
    name = re.sub(r"^[\s0-9a-fA-F\-]*(?=[\u4e00-\u9fa5A-Za-z])", "", name)
    return name.replace("CN", "", 1).strip() if name.startswith("CN") else name.strip()


def _clean_level_value(value: str) -> str:
    """去掉「证据级别A」这类值里重复的字段名前缀，避免输出成「证据等级：证据级别A」"""
    value = (value or "").strip()
    return re.sub(r'^(证据级别|证据等级|证据质量|推荐强度|推荐等级|推荐级别)[:：\s]*', '', value).strip()


def format_es_doc(doc: dict, query: str = "", intents: Optional[List[str]] = None) -> str:
    """按提问意图输出 ES 文档字段，避免把全部字段一股脑返回。

    - 剂量类问题：剂量 + 证据等级 / 推荐强度 / 指南来源（必返回，缺失写"暂无"）
    - 不良反应 / 禁忌症：对应字段 + 指南来源（必返回）
    - 未识别出意图：给一份精简摘要（药物、疾病/阶段、剂量、证据、来源）
    """
    if intents is None:
        intents = detect_es_intents(query)

    intervention = doc.get("intervention") or {}
    drug_name = intervention.get("name") or ""
    disease = doc.get("disease") or ""
    phase = doc.get("phase") or ""

    drug_spec = (intervention.get("modality_specific") or {}).get("drug_specific") or {}
    dose = drug_spec.get("dose") or {}

    evidence = doc.get("evidence") or {}
    evidence_level = _clean_level_value(evidence.get("evidence_level") or "")
    reco_grade = _clean_level_value(evidence.get("recommendation_grade") or "")

    safety = doc.get("safety") or {}
    adverse = (safety.get("adverse_effects") or {}).get("common") or ""
    contra = safety.get("contraindications") or []
    if isinstance(contra, list):
        contra = "、".join(str(c) for c in contra if c)
    contra = contra or ""

    source = doc.get("source") or {}
    guideline = clean_guideline_name(source.get("guideline") or "")
    year = source.get("year") or ""
    source_text = f"{guideline} ({year}年)" if guideline and year else (guideline or "暂无")

    lines: List[str] = []
    if drug_name:
        lines.append(f"药物：{drug_name}")
    if disease or phase:
        lines.append(" | ".join(p for p in [
            f"疾病：{disease}" if disease else "",
            f"治疗阶段：{phase}" if phase else "",
        ] if p))

    want_dose = "dose" in intents
    want_adverse = "adverse" in intents
    want_contra = "contraindication" in intents
    want_evidence = "evidence" in intents
    is_summary = not intents  # 未识别意图 → 精简摘要

    if want_dose or is_summary:
        rec_dose = _fmt_dose_item(dose.get("recommended"))
        max_dose = _fmt_dose_item(dose.get("max_daily"))
        if rec_dose:
            lines.append(f"推荐剂量：{rec_dose}")
        elif want_dose:
            lines.append("推荐剂量：暂无")
        if max_dose:
            lines.append(f"最大日剂量：{max_dose}")
        frequency = intervention.get("frequency") or ""
        route = intervention.get("route")
        route_desc = route.get("description") if isinstance(route, dict) else (route or "")
        if frequency:
            lines.append(f"给药频次：{frequency}")
        if route_desc:
            lines.append(f"给药途径：{route_desc}")

    if want_adverse:
        lines.append(f"常见不良反应：{adverse or '暂无'}")

    if want_contra:
        lines.append(f"禁忌症：{contra or '暂无'}")

    # 剂量类问题必须带上证据等级与推荐强度
    if want_dose:
        lines.append(f"证据等级：{evidence_level or '暂无'}")
        lines.append(f"推荐强度：{reco_grade or '暂无'}")
    elif want_evidence or is_summary:
        if evidence_level or want_evidence:
            lines.append(f"证据等级：{evidence_level or '暂无'}")
        if reco_grade or want_evidence:
            lines.append(f"推荐强度：{reco_grade or '暂无'}")

    lines.append(f"指南来源：{source_text}")
    return "\n".join(lines)


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

        阶段策略：
        - 提问里出现"急性/预防/慢性/围经期…"→ 自动加 phase 过滤，只答该阶段
        - 过滤后无结果（数据里阶段为空）→ 回退为不限阶段，避免直接答不出来
        - 提问没提阶段 → 急性期与预防期各取一部分，保证两类剂量都能给出

        Args:
            query: 查询文本
            disease: 疾病名称过滤
            drug_name: 药物名称过滤
            phase: 治疗阶段过滤（显式传入优先；None 时从 query 自动解析）
            top_k: 返回结果数量

        Returns:
            [{'score': float, 'source': dict}, ...]
        """
        explicit_phase = phase
        detected_phase = detect_phase_filter(query)
        effective_phase = explicit_phase or detected_phase

        if effective_phase:
            results = self._search_raw(query, disease, drug_name, effective_phase, top_k)
            if results:
                return results
            # 该阶段没有命中，回退到不限阶段（如阶段字段为空的旧文档）
            results = self._search_raw(query, disease, drug_name, None, top_k)
            return results

        return self._search_by_both_phases(query, disease, drug_name, top_k)

    def _search_by_both_phases(
        self,
        query: str,
        disease: Optional[str],
        drug_name: Optional[str],
        top_k: int,
    ) -> List[Dict]:
        """未指定阶段：在统一排序结果里给急性期、预防期各留一半配额，
        避免某一阶段（或阶段为空的老文档）把结果占满。"""
        pool = self._search_raw(query, disease, drug_name, None, max(top_k * 3, 10))
        quota = max(1, (top_k + 1) // 2)
        used: Dict[str, int] = {}
        picked: List[Dict] = []
        overflow: List[Dict] = []

        for item in pool:
            phase = item["source"].get("phase") or ""
            if used.get(phase, 0) >= quota or len(picked) >= top_k:
                overflow.append(item)
                continue
            used[phase] = used.get(phase, 0) + 1
            picked.append(item)

        for item in overflow:
            if len(picked) >= top_k:
                break
            picked.append(item)

        return picked[:top_k]

    def _search_raw(
        self,
        query: str,
        disease: Optional[str],
        drug_name: Optional[str],
        phase: Optional[str],
        top_k: int,
    ) -> List[Dict]:
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
            # phase 是 text 字段（中文会被分词），term 查 text 永远匹配不到，
            # 必须用 keyword 子字段做精确过滤
            must_clauses.append({"term": {"phase.keyword": phase}})
        
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
                "safety.contraindications",
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
    
    def format_context(self, results: List[Dict], query: str = "") -> str:
        """
        格式化ES检索结果为上下文（原format_es_context函数）

        Args:
            results: search() 的返回值
            query: 用户问题，用于按意图裁剪字段；为空时输出精简摘要

        Returns:
            多段参考文本的拼接
        """
        if not results:
            return "未找到相关信息"

        intents = detect_es_intents(query) if query else []
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"【参考{i}】\n" + format_es_doc(result['source'], intents=intents))

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
