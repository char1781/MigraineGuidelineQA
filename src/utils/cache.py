"""
src/utils/cache.py
职责：查询缓存，节省API费用

改动说明：新增模块，原代码中没有缓存机制

测试方法：
python -c "
from src.utils.cache import QueryCache
cache = QueryCache()
cache.set('test', '回答')
print(cache.get('test'))
"
"""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime


class QueryCache:
    """查询缓存管理器"""
    
    def __init__(self, cache_file: str = "data/cache/query_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load()
    
    def _load(self) -> dict:
        """加载缓存"""
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}
    
    def _save(self):
        """保存缓存"""
        self.cache_file.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    
    def get(self, query: str) -> Optional[str]:
        """获取缓存的回答"""
        key = query.strip().lower()
        return self.cache.get(key, {}).get("answer")
    
    def set(self, query: str, answer: str):
        """设置缓存"""
        key = query.strip().lower()
        self.cache[key] = {
            "answer": answer,
            "cached_at": datetime.now().isoformat(),
        }
        self._save()
    
    def clear(self):
        """清空缓存"""
        self.cache = {}
        self._save()


# ── 单元测试 ──
if __name__ == "__main__":
    cache = QueryCache("data/cache/test_cache.json")
    
    # 测试设置和获取
    cache.set("测试问题", "测试回答")
    assert cache.get("测试问题") == "测试回答"
    
    # 测试大小写不敏感
    assert cache.get("测试问题") == cache.get("测试问题  ")
    
    print("✅ 缓存测试通过")
