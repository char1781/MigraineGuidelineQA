"""
src/ingestion/es_review_editor.py
职责：将ES提取的结果转换为可编辑的格式，用于人工审核确认

运行方式：
# 方式1：作为模块运行
python -m src.ingestion.es_review_editor

# 方式2：直接运行
python src/ingestion/es_review_editor.py

核心函数：
- extract_to_reviewable(): 将JSON提取为可审核的结构化数据
- save_reviewed_data(): 保存审核后的数据
- load_reviewed_data(): 加载已审核的数据
"""

import sys
from pathlib import Path

# 【关键】必须在任何src模块导入之前处理路径
if __package__ is None or __package__ == '':
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import json
import re
import uuid
from typing import List, Dict, Optional
from datetime import datetime


class ESReviewEditor:
    """ES提取结果审核编辑器"""
    
    def __init__(self, review_dir: str = "data/review"):
        self.review_dir = Path(review_dir)
        self.review_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_to_reviewable(
        self,
        json_path: str,
        extractor_class,
    ) -> Dict:
        """
        使用ES提取器处理JSON，转换为可审核的结构
        
        Args:
            json_path: MinerU产出的JSON文件路径
            extractor_class: ES提取器类（传入class，不是实例）
        
        Returns:
            {
                'source_file': 'xxx.json',
                'guideline_name': '中国偏头痛诊治指南2022版',
                'guideline_year': 2022,
                'extracted_count': 50,
                'records': [
                    {
                        'id': 'rec_001',
                        'drug_name': '利扎曲普坦',
                        'phase': '急性期',
                        'dose_recommended': '5-10mg',
                        'dose_max_daily': '30mg',
                        'frequency': '每日1-2次',
                        'route': '口服',
                        'evidence_level': 'A',
                        'recommendation_grade': '1',
                        'adverse_effects': '头晕、恶心',
                        'contraindications': ['冠心病', '高血压'],
                        'source_table': '表3',
                        'source_row': 2,
                        'raw_text': '原始文本片段...',
                        'status': 'pending',  # pending/approved/modified/rejected
                        'reviewed_by': None,
                        'reviewed_at': None,
                        'modifications': {},
                    },
                    ...
                ]
            }
        """
        extractor = extractor_class(json_path)
        success = extractor.extract_data()
        
        if not success:
            raise RuntimeError(f"ES提取失败: {json_path}")
        
        # 解析为ES文档
        es_docs = extractor.parse_tables_to_elasticsearch()
        
        # 转换为审核格式
        records = []
        for idx, doc in enumerate(es_docs, start=1):
            record = self._es_doc_to_review_record(doc, idx)
            records.append(record)
        
        review_data = {
            "source_file": Path(json_path).name,
            "guideline_name": extractor.source_guideline or "未知指南",
            "guideline_year": extractor.guideline_year,
            "extracted_count": len(records),
            "records": records,
            "created_at": datetime.now().isoformat(),
            "review_status": "pending" if records else "empty",
            "warning": None if records else "没有从该PDF中抽取到有效的药物剂量记录，请检查MinerU JSON或表格结构。",
        }
        
        return review_data
    
    def _es_doc_to_review_record(self, es_doc: Dict, idx: int) -> Dict:
        """将ES文档转换为审核记录"""
        intervention = es_doc.get("intervention") or {}
        if not isinstance(intervention, dict):
            intervention = {}

        # 提取剂量信息（适配drug/tcm/physical三种类型）
        dose_info = {}
        modality = intervention.get("modality_specific") or {}
        if not isinstance(modality, dict):
            modality = {}

        if "drug_specific" in modality:
            drug_spec = modality["drug_specific"] or {}
            if isinstance(drug_spec, dict):
                dose = drug_spec.get("dose") or {}
                if isinstance(dose, dict):
                    rec = dose.get("recommended") or {}
                    max_d = dose.get("max_daily") or {}
                    if not isinstance(rec, dict):
                        rec = {}
                    if not isinstance(max_d, dict):
                        max_d = {}
                    dose_info = {
                        "recommended": rec.get("text_original", ""),
                        "max_daily": max_d.get("text_original", "") if max_d else "",
                        "unit": rec.get("unit", ""),
                        "min": rec.get("min"),
                        "max": rec.get("max"),
                        "max_unit": max_d.get("unit", "") if max_d else "",
                    }
        elif "tcm_specific" in modality:
            tcm_spec = modality["tcm_specific"] or {}
            if isinstance(tcm_spec, dict):
                dose_info = {
                    "recommended": intervention.get("frequency", ""),
                    "tcm_pattern": tcm_spec.get("tcm_pattern", ""),
                }
        elif "physical_specific" in modality:
            phys_spec = modality["physical_specific"] or {}
            if isinstance(phys_spec, dict):
                dose_info = {
                    "device": phys_spec.get("device", ""),
                    "target_site": ", ".join(phys_spec.get("target_site", []) or []),
                }

        # 提取不良反应和禁忌症
        safety = es_doc.get("safety") or {}
        if not isinstance(safety, dict):
            safety = {}
        adverse_effects = safety.get("adverse_effects") or {}
        if not isinstance(adverse_effects, dict):
            adverse_effects = {}
        adverse = adverse_effects.get("common", "")
        contraindications = safety.get("contraindications") or []

        # 提取来源信息
        source = es_doc.get("source") or {}
        if not isinstance(source, dict):
            source = {}
        traceability = es_doc.get("traceability") or {}
        if not isinstance(traceability, dict):
            traceability = {}

        return {
            "id": f"rec_{idx:03d}",
            "rule_id": es_doc.get("rule_id"),
            "drug_name": intervention.get("name", ""),
            "category": intervention.get("category", ""),
            "phase": es_doc.get("phase", ""),
            "dose_info": dose_info,
            "frequency": intervention.get("frequency", ""),
            "route": intervention.get("route", {}).get("description", ""),
            "evidence_level": es_doc.get("evidence", {}).get("evidence_level", ""),
            "recommendation_grade": es_doc.get("evidence", {}).get("recommendation_grade", ""),
            "adverse_effects": adverse,
            "contraindications": contraindications,
            "source_table": source.get("table_title", ""),
            "source_row": traceability.get("table_row_index"),
            "raw_text": traceability.get("raw_text_segment", "")[:300],
            "status": "pending",
            "reviewed_by": None,
            "reviewed_at": None,
            "modifications": {},
            "_original_doc": es_doc,  # 保留完整ES文档，方便重建
        }
    
    @staticmethod
    def make_review_id(source_file: str = "", guideline_name: Optional[str] = None) -> str:
        """生成唯一 review_id，避免文档名相同导致审核记录覆盖。"""
        file_stem = Path(source_file).stem if source_file else "review"
        name = (guideline_name or file_stem or "review").strip()
        safe_name = re.sub(r"[^A-Za-z0-9\u4e00-\u9fa5_-]+", "_", name)[:40] or "review"
        return f"{safe_name}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4().hex[:8]}"

    def save_reviewed_data(
        self,
        review_data: Dict,
        review_id: Optional[str] = None,
    ) -> str:
        """
        保存审核数据到本地JSON
        
        Args:
            review_data: 审核数据结构
            review_id: 自定义ID，不提供则自动生成
        
        Returns:
            保存的文件路径
        """
        if not review_id:
            review_id = self.make_review_id(
                review_data.get("source_file", ""),
                review_data.get("guideline_name"),
            )
        
        file_path = self.review_dir / f"{review_id}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(review_data, f, ensure_ascii=False, indent=2)
        
        return str(file_path)
    
    def load_reviewed_data(self, review_id: str) -> Dict:
        """加载审核数据"""
        file_path = self.review_dir / f"{review_id}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"审核数据不存在: {review_id}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def list_reviews(self) -> List[Dict]:
        reviews = []
        review_dir = Path("data/review")
        if not review_dir.exists():
            return reviews

        for file_path in review_dir.glob("*.json"):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                
                if data.get("review_status") == "completed":
                    continue
                    
                reviews.append({
                    # ★ 修复：直接取文件名（不带后缀），这就是真实的 ID
                    "review_id": file_path.stem, 
                    "guideline_name": data.get("guideline_name", "未知指南"),
                    "guideline_year": data.get("guideline_year", "未知"),
                    "extracted_count": data.get("extracted_count", 0),
                    "review_status": data.get("review_status", "pending"),
                    "md_path": data.get("md_path", ""),
                    "graph_doc_name": data.get("graph_doc_name", "")
                })
            except Exception as e:
                print(f"读取审核文件失败 {file_path}: {e}")
                
        return reviews
    
    def apply_modifications(
        self,
        review_data: Dict,
        record_id: str,
        modifications: Dict,
        reviewer: str = "user",
    ) -> Dict:
        """
        应用用户修改到某条记录
        
        Args:
            review_data: 审核数据
            record_id: 记录ID（如 'rec_001'）
            modifications: 修改内容，如 {'drug_name': '新名称', 'dose_recommended': '新剂量'}
            reviewer: 审核人
        
        Returns:
            更新后的review_data
        """
        for record in review_data["records"]:
            if record["id"] == record_id:
                record["modifications"] = modifications
                record["status"] = "modified"
                record["reviewed_by"] = reviewer
                record["reviewed_at"] = datetime.now().isoformat()
                
                # 将修改应用到显示字段
                for key, value in modifications.items():
                    if key in record:
                        record[key] = value
                
                break
        
        return review_data
    
    def approve_record(
        self,
        review_data: Dict,
        record_id: str,
        reviewer: str = "user",
    ) -> Dict:
        """批准某条记录（无修改）"""
        for record in review_data["records"]:
            if record["id"] == record_id:
                record["status"] = "approved"
                record["reviewed_by"] = reviewer
                record["reviewed_at"] = datetime.now().isoformat()
                break
        
        return review_data
    
    def reject_record(
        self,
        review_data: Dict,
        record_id: str,
        reason: str = "",
        reviewer: str = "user",
    ) -> Dict:
        """拒绝某条记录"""
        for record in review_data["records"]:
            if record["id"] == record_id:
                record["status"] = "rejected"
                record["reviewed_by"] = reviewer
                record["reviewed_at"] = datetime.now().isoformat()
                record["modifications"]["reject_reason"] = reason
                break
        
        return review_data
    
    def export_to_es_docs(self, review_data: Dict) -> List[Dict]:
        """
        将审核通过的记录导出为ES文档格式
        只导出 status = 'approved' 或 'modified' 的记录
        """
        import copy
        es_docs = []
        
        for rec in review_data.get("records", []):
            if rec.get("status") not in ("approved", "modified"):
                continue
                
            original_doc = rec.get("_original_doc")
            if not original_doc:
                continue
                
            # ★ 修复1：使用深拷贝，防止污染原内存数据
            doc = copy.deepcopy(original_doc)
            
            # ★ 修复2：应用用户在前端做的修改（药名、剂量、不良反应等）
            if rec.get("modifications"):
                doc = self._apply_modifications_to_es_doc(doc, rec["modifications"])
                
            # ★ 修复3：统一覆写来源指南名和年份
            src = doc.setdefault("source", {})
            src["guideline"] = review_data.get("guideline_name") or src.get("guideline")
            year = review_data.get("guideline_year")
            if year:
                src["year"] = year
                src["version"] = f"{year}版"
                
            es_docs.append(doc)
            
        return es_docs
    
    # def _apply_modifications_to_es_doc(
    #     self,
    #     es_doc: Dict,
    #     modifications: Dict,
    # ) -> Dict:
    #     """将用户修改应用到ES文档结构"""
    #     # 映射审核字段到ES文档路径
    #     field_mapping = {
    #         "drug_name": ["intervention", "name"],
    #         "frequency": ["intervention", "frequency"],
    #         "route": ["intervention", "route", "description"],
    #         "evidence_level": ["evidence", "evidence_level"],
    #         "recommendation_grade": ["evidence", "recommendation_grade"],
    #         "adverse_effects": ["safety", "adverse_effects", "common"],
    #     }
        
    #     for field, path in field_mapping.items():
    #         if field in modifications:
    #             # 按路径更新嵌套字典
    #             target = es_doc
    #             for key in path[:-1]:
    #                 target = target.setdefault(key, {})
    #             target[path[-1]] = modifications[field]
        
    #     # 特殊处理：剂量信息
    #     if "dose_info" in modifications:
    #         dose_info = modifications["dose_info"]
    #         drug_spec = es_doc.setdefault("intervention", {}) \
    #                           .setdefault("modality_specific", {}) \
    #                           .setdefault("drug_specific", {}) \
    #                           .setdefault("dose", {})
            
    #         if "recommended" in dose_info:
    #             drug_spec.setdefault("recommended", {})["text_original"] = dose_info["recommended"]
    #         if "max_daily" in dose_info:
    #             drug_spec.setdefault("max_daily", {})["text_original"] = dose_info["max_daily"]
        
    #     return es_doc

    def _apply_modifications_to_es_doc(
        self,
        es_doc: Dict,
        modifications: Dict,
    ) -> Dict:
        """将用户修改应用到ES文档结构"""
        # 字段路径映射
        field_mapping = {
            "drug_name": ["intervention", "name"],
            "frequency": ["intervention", "frequency"],
            "route": ["intervention", "route", "description"],
            "evidence_level": ["evidence", "evidence_level"],
            "recommendation_grade": ["evidence", "recommendation_grade"],
            "adverse_effects": ["safety", "adverse_effects", "common"],
            "contraindications": ["safety", "contraindications"],
            "source_guideline": ["source", "guideline"],
            "phase": ["phase"],  # ★ 新增：治疗阶段
        }
        
        for field, path in field_mapping.items():
            if field in modifications:
                # 按路径更新嵌套字典
                target = es_doc
                for key in path[:-1]:
                    target = target.setdefault(key, {})
                target[path[-1]] = modifications[field]
        
        # ★ 特殊处理：剂量信息（包含单位更新）
        if "dose_info" in modifications:
            dose_info = modifications["dose_info"]
            drug_spec = es_doc.setdefault("intervention", {}) \
                              .setdefault("modality_specific", {}) \
                              .setdefault("drug_specific", {}) \
                              .setdefault("dose", {})
            
            new_unit = dose_info.get("unit", "")
            
            if "recommended" in dose_info:
                target_rec = drug_spec.setdefault("recommended", {})
                raw_text = dose_info["recommended"]
                target_rec["text_original"] = raw_text
                
                # 重新解析数字并把单位写进去
                parsed = self._reparse_dose_text(raw_text, new_unit)
                target_rec.update(parsed)
                
            if "max_daily" in dose_info:
                target_max = drug_spec.setdefault("max_daily", {})
                raw_text = dose_info["max_daily"]
                target_max["text_original"] = raw_text
                
                parsed = self._reparse_dose_text(raw_text, new_unit)
                target_max.update(parsed)
        
        return es_doc

    def _reparse_dose_text(self, text: str, unit: str) -> dict:
        """辅助函数：解析审核后的剂量文本和单位"""
        import re
        if not text:
            return {}
            
        clean_text = str(text).replace('～', '-').replace('~', '-')
        numeric_part = re.sub(r'[a-zA-Z%]+', '', clean_text).strip()
        
        res = {"unit": unit}
        if '-' in numeric_part:
            parts = numeric_part.split('-')
            if len(parts) == 2:
                try:
                    res["min"] = float(parts[0].strip())
                    res["max"] = float(parts[1].strip())
                    res["type"] = "range"
                    return res
                except ValueError:
                    pass
        try:
            res["value"] = float(numeric_part)
            res["type"] = "single"
            return res
        except ValueError:
            pass
            
        res["type"] = "unparsed"
        return res


# ── 单元测试 ──
if __name__ == "__main__":
    """已入库数据的修正工具：改完审核文件后重新写回 ES。

    用法：
      # 仅覆盖有改动的记录（按 rule_id upsert，不会产生重复文档）
      python -m src.ingestion.es_review_editor --review-id <review_id>

      # 重新抽取过（rule_id 全变）时，先删该指南旧文档再写入
      python -m src.ingestion.es_review_editor --review-id <review_id> \
          --delete-guideline 中国偏头痛诊断与治疗指南
    """
    import argparse
    import os

    import src.config  # noqa: F401  确保 .env 已加载
    from elasticsearch import Elasticsearch

    parser = argparse.ArgumentParser(description="把审核文件重新写入 ES")
    parser.add_argument("--review-id", required=True, help="data/review 下的 review_id（不含 .json）")
    parser.add_argument("--delete-guideline", default="", help="写入前先删除该关键字匹配的旧文档")
    args = parser.parse_args()

    editor = ESReviewEditor()
    review_data = editor.load_reviewed_data(args.review_id)

    es = Elasticsearch([os.getenv("ES_HOST", "http://localhost:9200")])
    index = os.getenv("ES_INDEX", "medical_guidelines")

    if args.delete_guideline:
        res = es.delete_by_query(
            index=index,
            body={"query": {"wildcard": {"source.guideline.keyword": f"*{args.delete_guideline}*"}}},
            refresh=True,
        )
        print(f"🗑️  删除旧文档 {res['deleted']} 条（匹配: *{args.delete_guideline}*）")

    es_docs = editor.export_to_es_docs(review_data)
    for doc in es_docs:
        es.index(index=index, id=doc.get("rule_id"), document=doc)
    if es_docs:
        es.indices.refresh(index=index)

    print(f"✅ 已写入 {len(es_docs)} 条文档（按 rule_id 覆盖）")
