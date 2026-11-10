"""
src/utils/kb_metadata.py
职责：管理知识库元数据的持久化

功能：
- 记录已入库的指南列表（指南名称、年份、入库时间）
- 持久化到JSON文件，重启后自动加载
- 支持增量添加和删除

测试方法：
python -m src.utils.kb_metadata
或
python src/utils/kb_metadata.py
"""

import sys
from pathlib import Path

# 【关键】必须在任何src模块导入之前处理路径
if __package__ is None or __package__ == '':
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import json
from datetime import datetime
from typing import List, Dict, Optional


class KnowledgeBaseMetadata:
    """知识库元数据管理器"""
    
    def __init__(self, metadata_file: str = "data/kb_metadata.json"):
        self.metadata_file = Path(metadata_file)
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        self.kb_list = self._load()
    
    def _load(self) -> List[Dict]:
        """加载已入库的指南列表"""
        if self.metadata_file.exists():
            try:
                return json.loads(self.metadata_file.read_text(encoding='utf-8'))
            except Exception:
                return []
        return []
    
    def _save(self):
        """保存到文件"""
        self.metadata_file.write_text(
            json.dumps(self.kb_list, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
    
    def add_guideline(self, name: str, year: int, **extra_info) -> bool:
        """
        添加一个指南
        
        Args:
            name: 指南名称
            year: 发布年份
            **extra_info: 额外信息（如文件路径、MD5等）
        
        Returns:
            True if added, False if already exists
        """
        # 检查是否已存在（去重）
        for kb in self.kb_list:
            if kb['name'] == name and kb['year'] == year:
                # 允许覆盖更新最新的 md_file 和 added_at 时间
                kb.update(extra_info)
                kb["added_at"] = datetime.now().isoformat()
                self._save()
                return False  # 返回 False 表示已存在，但数据已刷新
        
        # 添加新指南
        self.kb_list.append({
            "name": name,
            "year": year,
            "status": "ok",
            "added_at": datetime.now().isoformat(),
            **extra_info
        })
        self._save()
        return True
    
    def remove_guideline(self, name: str, year: int) -> bool:
        """
        删除一个指南
        
        Returns:
            True if removed, False if not found
        """
        original_len = len(self.kb_list)
        self.kb_list = [
            kb for kb in self.kb_list 
            if not (kb['name'] == name and kb['year'] == year)
        ]
        
        if len(self.kb_list) < original_len:
            self._save()
            return True
        return False
    
    def get_all(self) -> List[Dict]:
        """获取所有已入库的指南"""
        return self.kb_list
    
    def exists(self, name: str, year: int) -> bool:
        """检查指南是否已入库"""
        return any(kb['name'] == name and kb['year'] == year for kb in self.kb_list)


# ── 单元测试 ──
if __name__ == "__main__":
    kb_meta = KnowledgeBaseMetadata("data/kb_metadata.json")
    
    # 测试添加
    added = kb_meta.add_guideline("中国偏头痛诊治指南2022版", 2022)
    print(f"添加指南: {'成功' if added else '已存在'}")
    
    # 测试重复添加（应该被去重）
    added2 = kb_meta.add_guideline("中国偏头痛诊治指南2022版", 2022)
    print(f"重复添加: {'成功（异常）' if added2 else '已去重（正常）'}")
    
    # 测试获取
    all_kb = kb_meta.get_all()
    print(f"\n当前知识库列表 ({len(all_kb)}个):")
    for kb in all_kb:
        print(f"  - {kb['name']} ({kb['year']}年)")
    
    print("\n✅ 测试完成")
