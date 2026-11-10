import json
import uuid
import re
from typing import List, Dict, Optional, Tuple, Set
from bs4 import BeautifulSoup
import os
from dataclasses import dataclass
from enum import Enum
from src.utils.year_utils import extract_year


class TreatmentType(Enum):
    """治疗类型枚举"""
    DRUG = 'drug'
    TCM = 'tcm'
    PHYSICAL = 'physical'


class TreatmentPhase(Enum):
    """治疗阶段枚举"""
    ACUTE = '急性期'
    PREVENTION = '预防期'
    PREVENTION_CHRONIC = '预防期-慢性'
    PERIMENSTRUAL = '围经期'
    INTERMITTENT = '间歇期'
    PREGNANCY = '妊娠期'
    LACTATION = '哺乳期'
    UNKNOWN = '未知'


@dataclass
class DoseInfo:
    """剂量信息数据类"""
    recommended: Optional[Dict] = None
    max_daily: Optional[Dict] = None
    route: str = 'oral'
    frequency: str = ''
    constraints: List[str] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []


class MedicalGuidelineExtractor:
    """医疗指南提取器：支持表格和正文提取
    """

    MAX_TABLE_HTML_LENGTH = 200000
    MAX_TABLE_ROWS = 80
    MAX_TABLE_COLS = 20
    MAX_TABLE_CELLS = 1200
    
    # 配置常量
    DEFAULT_GUIDELINE_YEAR = 1111
    DEFAULT_POPULATION = "成人"
    DEFAULT_AGE_RANGE = ">=18"
    DEFAULT_LANGUAGE = "zh-CN"
    
    # 治疗相关关键词
    TREATMENT_TABLE_KEYWORDS_CN = [
        '药物', '治疗', '剂量', '推荐', '预防', '急性', '急性期',
        '处方', '非处方', '镇痛', 'cgrp', '特异性', '非特异性',
        '神经调控', '豪针刺法', '针灸', '辩经取穴', '非药物'
    ]
    
    TREATMENT_TABLE_KEYWORDS_EN = [
        'drug', 'treatment', 'dose', 'recommendation', 'prophylaxis',
        'acute', 'prescription', 'over-the-counter', 'cgrp',
        'pharmacological', 'analgesic', 'specific', 'non-specific',
        'neuromodulation', 'therapy', 'acupuncture'
    ]
    
    def __init__(self, json_file_path: str):
        """初始化提取器
        
        Args:
            json_file_path: JSON文件路径
        """
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"找不到文件: {json_file_path}")
        
        self.json_file_path = json_file_path
        self.source_guideline: Optional[str] = None
        self.guideline_year: int = self.DEFAULT_GUIDELINE_YEAR
        self.tables: List[Dict] = []
        self.text_blocks: List[Dict] = []
        self.last_title: Optional[Dict] = None
        self.processed_indices: Set[int] = set()
        
        import hashlib
        filename = os.path.basename(json_file_path)
        # 方案1：使用文件名（可读性好）
        self.pdf_id = filename.replace('.json', '').replace('MinerU_CN ', '')
        # 方案2：使用哈希（唯一性更强）from src.utils.year_utils import extract_year
        # self.pdf_id = hashlib.md5(filename.encode()).hexdigest()[:16]
        
    def extract_data(self) -> bool:
        """提取指南中的所有数据       
        Returns:
            bool: 是否成功提取
        """
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._traverse(data)

            if not self.source_guideline:
                self.source_guideline = self._fallback_guideline_name()
            if self.guideline_year == self.DEFAULT_GUIDELINE_YEAR:
                self.guideline_year = self._extract_year(self.source_guideline or '')
            if self.guideline_year == self.DEFAULT_GUIDELINE_YEAR:
                self._extract_year_from_filename()
            
            self.tables.sort(key=self._get_table_sort_key)
            
            return True
            
        except FileNotFoundError:
            print(f"错误: 找不到文件 {self.json_file_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"错误: JSON解析失败 - {e}")
            return False
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def _get_table_sort_key(table_item: Dict) -> int:
        """获取表格排序键
        Args:
            table_item: 表格项   
        Returns:
            float: 排序键（表格编号）from src.utils.year_utils import extract_year
        """
        caption = table_item.get('caption', '') or ''
        
        # 优先匹配中文表格编号：要么匹配「附表」，要么匹配「表」，二选一
        cn_pattern = re.search(r'(?:附表|表)\s*(\d+(?:\.\d+)?)', caption, re.UNICODE)
        
        if cn_pattern:
            return float(cn_pattern.group(1))
        
        # 其次匹配英文表格编号
        en_pattern = re.search(r'Table\s*(\d+(?:\.\d+)?)', caption, re.IGNORECASE | re.UNICODE)
        
        if en_pattern:
            return float(en_pattern.group(1))
        
        # 未找到编号则排到最后
        return 999999.0

    def _traverse(self, obj, parent_list=None):
        """递归遍历JSON结构
        """
        if isinstance(obj, dict):
            obj_type = obj.get('type')
            
            if isinstance(obj, dict):
                # 提取指南名称和年份
                if self.source_guideline is None and obj.get('type') == 'title':
                    extracted_text = self._extract_text_from_lines(obj.get('lines', []))
                    if self._is_main_guideline_title(extracted_text):
                        self.source_guideline = extracted_text
                        self.guideline_year = self._extract_year(self.source_guideline)
                
                # 处理标题
                if obj.get('type') == 'title' and self.source_guideline is not None:
                    self._process_title(obj)
                
                # 表格专用路径（不提取lines）
                if obj.get('type') == 'table':
                    self._handle_table(obj)
                
                # 文本专用路径（排除表格）
                # self._process_text_content(obj)
                elif obj_type != 'table':  # 明确排除表格
                    self._handle_text(obj)

                # 递归处理值
                for value in obj.values():
                    self._traverse(value, parent_list=obj if isinstance(obj, list) else parent_list)
           
        elif isinstance(obj, list):
            self._check_title_list_pattern(obj)
            for item in obj:
                self._traverse(item, parent_list=obj)
    
    def _handle_table(self, obj: Dict):
        """✅ 表格专用处理"""
        table_data = self._extract_table(obj)
        if table_data:
            self.tables.append(table_data)
            self.last_title = None
            
    def _handle_text(self, obj: Dict):
        """✅ 文本专用处理（排除表格）"""
        text_content = self._extract_text_from_lines(obj.get('lines', []))
        
        if not text_content or not text_content.strip():
            return
        
        # 过滤逻辑
        if self._is_valuable_text(text_content):
            self.text_blocks.append({
                'content': text_content,
                'bbox': obj.get('bbox')
            })
    
    def _is_valuable_text(self, text: str) -> bool:
        """✅ 判断文本是否有价值（独立函数便于调试）"""
        cleaned = text.strip()
        
        # 规则1：包含治疗关键词
        treatment_keywords = ['mg', '剂量', '推荐', '治疗', '预防', '曲坦', '萘普生', '布洛芬']
        if any(kw in cleaned for kw in treatment_keywords):
            return True
        
        # 规则2：过滤纯数字/符号
        if len(cleaned) < 3 and re.match(r'^[\d\W]+$', cleaned):
            return False
        
        # 规则3：最小长度要求
        return len(cleaned) >= 3        
    
    
    def _process_title(self, obj: Dict):
        """处理标题对象
        """
        title_text = self._extract_text_from_lines(obj.get('lines', []))
        title_stripped = title_text.strip() if title_text else ''
        
        if not title_stripped:
            return
        
        is_cn_table_title = title_stripped.startswith('表') or title_stripped.startswith('附表')
        is_en_table_title = title_stripped.lower().startswith('table')
        
        if is_cn_table_title or is_en_table_title:
            if is_cn_table_title or self.last_title is None:
                self.last_title = {
                    'text': title_text,
                    'index': obj.get('index')
                }
            
    def _is_main_guideline_title(self, text: str) -> bool:
        """判断是否为主指南标题
        """
        if not text or len(text) < 5:
            return False
        
        # 指南关键词
        guideline_keywords = [
            '治疗指南', '诊治指南', '防治指南', '专家共识', 
            '规范', '专家建议', '专家意见', '临床指南'
        ]
        
        # 疾病关键词
        disease_keywords = ['偏头痛', '头痛', 'migraine', 'Migraine']
        
        has_keyword = any(kw in text for kw in guideline_keywords)
        has_disease = any(disease in text for disease in disease_keywords)
        has_year = bool(re.search(r'\d{4}', text))
        
        # 必须有关键词，且有疾病名或年份
        return has_keyword and (has_disease or has_year)
    
    def _extract_text_from_lines(self, lines: List[Dict]) -> str:
        """从lines结构中提取文本
        """
        if not lines:
            return ''
        
        texts = []
        for line in lines:
            if not isinstance(line, dict):
                continue
            
            spans = line.get('spans', [])
            
            if not spans:
            # 如果没有spans，尝试直接获取content
                direct_content = line.get('content', '')
                if direct_content:
                    texts.append(direct_content)
                continue
            
            for span in spans:
                content = span.get('content', '')
                cleaned = content
                if content:
                    cleaned = cleaned.replace('\\%', '%').replace('\\\\', '\\')
                    cleaned = re.sub(r'\^\{([^}]+)\}', r'\1', cleaned)
                    cleaned = re.sub(r'_\{([^}]+)\}', r'\1', cleaned)
                    cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned)
                    
                    cleaned = re.sub(r'\\sim', '~', cleaned)
                    cleaned = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', cleaned)
                    cleaned = re.sub(r'\\mathrm', '', cleaned)
                    cleaned = cleaned.replace('~', '')
                                      
                    if cleaned.strip():
                        texts.append(cleaned)
        # result = ''.join(texts) if texts else ''
        # print(result)
        # return result
        return ''.join(texts) if texts else ''
    
    def _extract_table(self, table_obj: Dict) -> Optional[Dict]:
        """提取表格信息
        支持两种 MinerU JSON 结构：
        1) 老结构：blocks -> table_caption/table_body/table_footnote
        2) 新结构：type='table', content='<table>...</table>'
        """
        table_info = {
            'caption': '',
            'html': None,
            'footnotes': []
        }

        # 兼容新 MinerU 输出：表格对象直接带 content 字段
        html = table_obj.get('html') or table_obj.get('content')
        if html:
            table_info['html'] = html

        caption = table_obj.get('caption') or table_obj.get('title')
        if caption:
            table_info['caption'] = str(caption)

        blocks = table_obj.get('blocks', [])
        caption_parts = []

        for block in blocks:
            block_type = block.get('type')

            if block_type == 'table_caption':
                caption_text = self._extract_text_from_lines(block.get('lines', []))
                if caption_text:
                    caption_parts.append(caption_text)

            elif block_type == 'table_body':
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        if span.get('type') == 'table':
                            table_info['html'] = span.get('html') or table_info['html']

            elif block_type == 'table_footnote':
                footnote = self._extract_text_from_lines(block.get('lines', []))
                if footnote:
                    table_info['footnotes'].append(footnote)

        if caption_parts:
            table_info['caption'] = " / ".join(caption_parts)
        elif not table_info['caption'] and table_info['html']:
            table_info['caption'] = self._infer_caption_from_html(table_info['html'])

        # 只返回有HTML内容的表格
        return table_info if table_info['html'] else None
    
    # def _extract_text_block(self, text_obj: Dict) -> Optional[Dict]:
    #     """提取文本块
    #     """
    #     if 'content' in text_obj:
    #         text_content = text_obj['content']
    #     else:
    #         text_content = self._extract_text_from_lines(text_obj.get('lines', []))
        
    #     cleaned_content = text_content.strip()
        
    #     if not cleaned_content:
    #         return None
        
    #     treatment_keywords = ['mg', '剂量', '推荐', '治疗', '预防', '曲坦', '萘普生', '布洛芬']
    #     has_treatment_info = any(kw in cleaned_content for kw in treatment_keywords)
        
    #     if has_treatment_info:
    #         return {
    #             'content': text_content,
    #             'bbox': text_obj.get('bbox')
    #         }
        
    #     # 过滤纯数字和符号
    #     # if not cleaned_content or re.match(r'^[\d\W]+$', cleaned_content):
    #     #     return None
    #     if len(cleaned_content) < 3 and re.match(r'^[\d\W]+$', cleaned_content):
    #         return None
        
    #     return {
    #         'content': text_content,
    #         'bbox': text_obj.get('bbox')
    #     }
        
    def _check_title_list_pattern(self, obj_list: List):
        """检查标题列表模式并提取伪表格
        """
        if not isinstance(obj_list, list):
            return
        
        i = 0
        while i < len(obj_list):
            item = obj_list[i]
            
            if not isinstance(item, dict):
                i += 1
                continue
            
            # 检查是否为表格标题
            text = self._extract_text_from_lines(item.get('lines', []))
            if not self._is_table_title(text):
                i += 1
                continue
            
            # 处理表格标题
            title_index = item.get('index')
            if title_index in self.processed_indices:
                i += 1
                continue
            
            # 收集表格内容
            title_text = text
            list_content, english_title, next_pos = self._collect_table_content(
                obj_list, i + 1
            )
            
            # 创建伪表格
            if list_content:
                pseudo_table = self._create_pseudo_table(
                    title_text, english_title, list_content, title_index
                )
                self.tables.append(pseudo_table)
                self.processed_indices.add(title_index)
            
            i = next_pos
    
    @staticmethod
    def _is_table_title(text: Optional[str]) -> bool:
        """判断是否为表格标题
        """
        if not text:
            return False
        
        cleaned = text.strip()
        if not cleaned:
            return False
        
        is_chinese_title = (cleaned.startswith('表') or cleaned.startswith('附表')) and len(cleaned) < 50
        is_english_title = re.match(r'^Table\s*[\dA-Z]+\.?\d*', cleaned, re.IGNORECASE) is not None
        
        return is_chinese_title
    
    def _collect_table_content(
        self, 
        obj_list: List, 
        start_pos: int
    ) -> Tuple[List, Optional[str], int]:
        """收集表格内容直到下一个表格标题
        Returns:
            Tuple[List, Optional[str], int]: (内容列表, 英文标题, 下一个位置)
        """
        list_content = []
        english_title = None
        j = start_pos
        
        while j < len(obj_list):
            next_item = obj_list[j]
            
            if not isinstance(next_item, dict):
                break
            
            # 检查是否遇到下一个表格标题
            next_text = self._extract_text_from_lines(next_item.get('lines', []))
            if self._is_table_title(next_text):
                break
            
            # 检查是否为表格对象
            if next_item.get('type') == 'table':
                break
            
            # 收集内容
            next_type = next_item.get('type')
            if next_type == 'text':
                if next_text and 'Table' in next_text and j == start_pos:
                    english_title = next_text
                else:
                    list_content.append(next_item)
            elif next_type == 'list':
                list_content.append(next_item)
            
            j += 1
        
        return list_content, english_title, j
    
    @staticmethod
    def _create_pseudo_table(
        title_text: str,
        english_title: Optional[str],
        list_content: List,
        index: int
    ) -> Dict:
        """创建伪表格对象
        """
        caption = title_text
        if english_title:
            caption += f" / {english_title}"
        
        return {
            'caption': caption,
            'html': None,
            'list_content': list_content,
            'footnotes': [],
            'is_list_format': True,
            'index': index
        }
        
    def _fallback_guideline_name(self) -> str:
        """兜底指南名：优先使用文件名，再使用 PDF 标题中的年版信息。"""
        filename = os.path.basename(self.json_file_path)
        stem = os.path.splitext(filename)[0]
        if stem:
            stem = stem.replace('MinerU_CN ', '').replace('MinerU_', '')
            stem = re.sub(r'[_\-]+', ' ', stem).strip()
            if stem and stem != 'output':
                return stem
        return '未知指南'

    # def _extract_year(self, guideline_name: str) -> int:
    #     """从指南名称提取年份
    #     """
    #     if not guideline_name:
    #         return self.DEFAULT_GUIDELINE_YEAR
        
    #     # 模式1: 直接匹配4位数字
    #     match = re.search(r'(\d{4})', guideline_name)
    #     if match:
    #         year = int(match.group(1))
    #         # 验证年份合理性 (1990-2030)
    #         if 1990 <= year <= 2030:
    #             return year
        
    #     # 模式2: 匹配"年版"格式
    #     version_match = re.search(r'(\d{4})版', guideline_name)
    #     if version_match:
    #         return int(version_match.group(1))
        
    #     return self.DEFAULT_GUIDELINE_YEAR

    # def _extract_year_from_filename(self):
    #     """从文件名提取年份（兜底方案）
    #     """
    #     filename = os.path.basename(self.json_file_path)
    #     year_match = re.search(r'(\d{4})', filename)
        
    #     if year_match:
    #         year = int(year_match.group(1))
    #         if 1990 <= year <= 2030:
    #             self.guideline_year = year
    #             print(f"从文件名提取年份: {self.guideline_year}")
    #         else:
    #             print(f"文件名中的年份 {year} 不在合理范围内")
    #     else:
    #         print("未能从文件名提取年份")

    def _extract_year(self, guideline_name: str) -> int:
        """从指南名称提取年份（内部实现改为复用共享工具）"""
        year = extract_year(guideline_name)
        return year if year is not None else self.DEFAULT_GUIDELINE_YEAR

    def _extract_year_from_filename(self):
        filename = os.path.basename(self.json_file_path)
        year = extract_year(filename)   # 文件名里也可能是 "xxx2023.json"
        if year is not None:
            self.guideline_year = year
            print(f"从文件名提取年份: {self.guideline_year}")
        else:
            print("未能从文件名提取年份")
    
    def _validate_treatment(self, treatment: Dict) -> bool:
        """验证治疗方案是否有效，过滤误匹配
        """
        name = treatment.get('name', '').strip()
        
        # 规则1: 过滤非药物名称（使用正则模式）
        invalid_patterns = [
            r'^.*患病率.*$', r'^.*发病率.*$', 
            r'^.*约\d+%.*$', r'^每个.*$', r'^该研究.*$', r'^结果显示.*$',
            r'^.*患病率.*$', r'^.*发病率.*$', r'^.*减少.*$', r'^.*增加.*$',
            r'^.*约\d+%.*$', r'^.*每个.*$', r'^.*该研究.*$', r'^.*结果显示.*$',
            r'^男性.*$', r'^女性.*$', r'^我国.*$', r'^全球.*$',
            r'^.*比例.*$', r'^.*发生率.*$'  
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, name):
                return False
        
        # 规则2: 长度验证
        is_letter_drug = re.match(r'^(?=.*[a-zA-Z0-9])[a-zA-Z0-9\-]{2,}$', name)  # 匹配纯字母/数字/连字符（如CGRP、5-HT）
        if is_letter_drug:
            if len(name) > 40:  
                return False
        else:
            if len(name) < 2 or len(name) > 40:  
                return False
        
        # 规则3: 过滤纯数字/空格/百分号等
        if re.match(r'^[\d\s%~]+$', name):  
            return False
        
        # 规则4: 必须包含有效字符
        if not re.search(r'[\u4e00-\u9fa5a-zA-Z]', name) and not re.match(r'^\d+-\w+$', name):
            return False
        
        return True
    
    def parse_tables_to_elasticsearch(self, pdf_id=None):
        """将表格解析为Elasticsearch文档格式
            功能：
        - 支持rowspan/colspan（合并单元格）
        - 过滤分类标题行
        - 过滤空行"""
        
        pdf_id = pdf_id or self.pdf_id
        
        es_documents = []
        
        print(f"\n=== 开始处理表格 ===")
        print(f"总表格数: {len(self.tables)}")
        
        if len(self.tables) == 0:
            print("⚠️  警告：没有找到任何表格！")
    
        for table_idx, table in enumerate(self.tables):
            caption = table.get('caption') or ''
            html = table.get('html') or table.get('content') or ''

            if not html:
                print(f"  跳过空HTML表格[{table_idx+1}]: {caption[:50]}...")
                continue

            if len(html) > self.MAX_TABLE_HTML_LENGTH:
                print(f"  跳过超大表格[{table_idx+1}]：{len(html)} 字符，避免解析卡住")
                continue

            try:
                soup = BeautifulSoup(html, 'html.parser')
                rows = soup.find_all('tr')
                row_count = len(rows)
                cell_count = sum(len(r.find_all(['th', 'td'])) for r in rows)
                if row_count > self.MAX_TABLE_ROWS or cell_count > self.MAX_TABLE_CELLS:
                    print(
                        f"  跳过过大/异常表格[{table_idx+1}]：{row_count}行、{cell_count}格，避免卡住"
                    )
                    continue
            except Exception as exc:
                print(f"  跳过无法解析的表格[{table_idx+1}]：{exc}")
                continue

            if table.get('is_list_format'):
                print(f"  跳过列表格式表格[{table_idx+1}]: {caption[:50]}...")
                continue

            parsed_rows = self._parse_html_table_with_spans(html)
            headers = parsed_rows[0] if parsed_rows else []
            if not caption:
                caption = self._infer_caption_from_html(table['html'])
            if not caption and headers:
                caption = ' '.join([h for h in headers if h][:4])

            if not self._is_treatment_table(caption, headers):
                print(f"  跳过非治疗表格[{table_idx+1}]: {caption[:50]}...")
                continue

            table['caption'] = caption
            print(f"  处理表格[{table_idx+1}/{len(self.tables)}]: {caption[:50]}...")

            phase = self._extract_phase_from_caption(caption)
            
            if len(parsed_rows) < 2:
                print(f"    表格行数不足")
                continue
            
            headers = parsed_rows[0]
            print(f"    表头: {headers}")
            
            doc_count = 0
            for row_idx, cells in enumerate(parsed_rows[1:], start=1):
                # ✅ 过滤分类行
                if self._is_category_row(cells):
                    print(f"      跳过分类行: {cells[0]}")
                    continue
                
                # ✅ 过滤空行
                if all(not cell or cell.strip() == '' for cell in cells):
                    continue
                
                doc = self._parse_table_row_to_document(
                    headers, cells, caption, phase,
                    table.get('footnotes', []), pdf_id, row_idx
                )
                
                if doc:
                    es_documents.append(doc)
                    doc_count += 1
            
            print(f"    生成 {doc_count} 条文档")
        
        return es_documents
    
    def _infer_caption_from_html(self, html: str) -> str:
        """从 HTML 表头中推断表格标题，便于药物剂量类表格识别。"""
        if not html:
            return ''
        try:
            soup = BeautifulSoup(html, 'html.parser')
            rows = soup.find_all('tr')
            if not rows:
                return ''
            first_row = rows[0].find_all(['th', 'td'])
            cells = [cell.get_text(strip=True) for cell in first_row if cell.get_text(strip=True)]
            if not cells:
                return ''
            text = ' '.join(cells[:4])
            return text[:80]
        except Exception:
            return ''

    def _parse_html_table_with_spans(self, html: str) -> List[List[str]]:
        """解析HTML表格，正确处理rowspan和colspan

        对于极大或高度畸形的 HTML 表格，直接跳过，避免在 PDF 表格解析阶段卡住。
        """
        if not html or not str(html).strip():
            return []
        html_text = str(html)
        if len(html_text) > self.MAX_TABLE_HTML_LENGTH:
            print(f"[ESExtractor] 表格过大，跳过解析，长度={len(html_text)}")
            return []

        try:
            soup = BeautifulSoup(html_text, 'html.parser')
            rows = soup.find_all('tr')
        except Exception:
            return []

        if not rows:
            return []

        total_cells = sum(len(r.find_all(['th', 'td'])) for r in rows)
        if len(rows) > self.MAX_TABLE_ROWS or total_cells > self.MAX_TABLE_CELLS:
            print(
                f"[ESExtractor] 表格行列过多，跳过解析，rows={len(rows)} cells={total_cells}"
            )
            return []
        
        # 第1步：计算最大列数
        num_cols = 0
        for row in rows:
            cells = row.find_all(['th', 'td'])
            cols_in_row = sum(int(cell.get('colspan', 1)) for cell in cells)
            num_cols = max(num_cols, cols_in_row)
        
        # 第2步：初始化网格
        num_rows = len(rows)
        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        
        # 第3步：填充网格
        for row_idx, row in enumerate(rows):
            cells = row.find_all(['th', 'td'])
            col_idx = 0
            
            for cell in cells:
                # 跳过已被rowspan填充的列
                while col_idx < num_cols and grid[row_idx][col_idx] is not None:
                    col_idx += 1
                
                if col_idx >= num_cols:
                    break
                
                # 提取单元格信息
                cell_text = cell.get_text(strip=True)
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                
                # 填充当前单元格及其跨越的区域
                for r_offset in range(rowspan):
                    for c_offset in range(colspan):
                        target_row = row_idx + r_offset
                        target_col = col_idx + c_offset
                        
                        if target_row < num_rows and target_col < num_cols:
                            grid[target_row][target_col] = cell_text
                
                col_idx += colspan
        
        # 第4步：处理None值（理论上不应该有，但做防御性处理）
        for row in grid:
            for i in range(len(row)):
                if row[i] is None:
                    row[i] = ''
        
        return grid
    
    
    def _is_treatment_table(self, caption, headers=None):
        """判断是否为治疗相关表格"""
        # if not caption:
        #     return False
        full_text = caption.strip().lower()
        if headers:
            full_text += ' ' + ''.join(headers).strip().lower()
            
        full_text_clean = full_text.replace('‑', '-').replace(' ', '')
        
        if not full_text_clean:
            return False
        
        # caption_clean = caption.strip().lower().replace('‑', '-').replace(' ', '')
        
        keywords_cn = ['药物', '治疗', '剂量', '推荐', '预防', '急性', '急性期',
                    '处方', '非处方', '镇痛', 'cgrp', '特异性', '非特异性','用法',
                    '神经调控', '豪针刺法', '针灸', '辩经取穴', '非药物', '方案', '发作期',
                    '慢性偏头痛', '预防性', '偏头痛发作','镇痛药物','处方药','重叠推荐','防治',
                    '常用','前庭性偏头痛','禁忌症','不良反应']
        keywords_en = ['drug', 'treatment', 'dose', 'recommendation', 'prophylaxis',
                    'acute', 'prescription', 'over-the-counter', 'cgrp',
                    'pharmacological', 'analgesic', 'specific', 'non-specific',
                    'neuromodulation', 'therapy', 'acupuncture']

        is_match_cn = any(kw in full_text_clean for kw in keywords_cn)
        is_match_en = any(kw in full_text_clean for kw in keywords_en)
        if is_match_cn or is_match_en:
            return True

        exclude_keywords = [
            'grade', 'grading', '分级', '定义','测定量表', '失能量表', '测试量表', 
            '评分', '量表', '分类标准', '诊断标准','病史的获取', '治疗指征', '患者特征', 
            '关键因素', '解释与建议', '分类', '头痛的鉴别', '评定量表', '筛查量表', 
            'GRADE', '个体化', '评估问卷', '证据级别', '推荐等级', '推荐强度', '强度分级'
        ]
        if any(kw in full_text_clean for kw in exclude_keywords):
            return False

        return False
    
    def _extract_phase_from_caption(self, caption):
        """从表格标题提取治疗阶段
        三个阶段及其可能的叫法：
        1. 急性期：急性期、急性发作期、发作急性期、急性偏头痛发作
        2. 预防期：预防性治疗、缓解期、预防治疗
        3. 慢性偏头痛：慢性偏头痛（通常只用预防性治疗）
        """
        if not caption:
            return ''
        
        caption_clean = caption.strip().lower().replace('‑', '-').replace(' ', '')

        # 优先级1: 慢性偏头痛（最具体）
        chronic_keywords = ['慢性偏头痛', 'chronic migraine', 'chronicmigraine']
        if any(kw in caption_clean for kw in chronic_keywords):
            return '慢性偏头痛'
        
        # 优先级2: 急性期（包含各种变体）
        acute_keywords = [
            '急性期', '急性发作期', '发作急性期', 
            '急性偏头痛发作', '急性发作', '急性治疗',
            'acute', 'attack', 'acutephase', 'acuteattack', 'episodic'
        ]
        if any(kw in caption_clean for kw in acute_keywords):
            # 排除"急性"出现在"慢性"相关描述中的情况
            if '慢性' not in caption_clean and 'chronic' not in caption_clean:
                return '急性期'
        
        # 优先级3: 预防期
        preventive_keywords = [
            '预防期', '预防性治疗', '预防治疗', '缓解期',
            'prophylaxis', 'preventive', 'prevention', 'prophylactic'
        ]
        if any(kw in caption_clean for kw in preventive_keywords):
            # 如果同时提到慢性，归类为慢性偏头痛
            if '慢性' in caption_clean or 'chronic' in caption_clean:
                return '慢性偏头痛'
            return '预防期'
        
        return ''
    
    def _is_category_row(self, cells):
        """判断是否为分类标题行"""
        if not cells:
            return True
    
        if len(cells) == 1 or (len(cells) > 1 and all(not c for c in cells[1:])):
            return True
        
        categories = ['非特异性治疗', '特异性治疗', '非甾体抗炎药', '曲普坦类', '曲坦类',
                    '麦角胺', '地坦类', '吉泮类', '物理治疗', '中医治疗',
                    '乙酰苯胺类解热镇痛药', '含咖啡因的复合制剂', '麦角胺及其衍生物',
                    'β受体阻滞剂', '抗癫痫药', '抗惊厥药', '钙通道拮抗剂', '钙离子拮抗剂',
                    '抗抑郁药', 'CGRP', '其他药物', '止呕剂', '复合制剂', 'COX-2抑制剂']
        return any(cat in cells[0] for cat in categories)

    def _extract_physical_device_name(self, data_map, headers, cells):
        """提取物理治疗设备名称"""
        # 物理治疗设备关键词
        device_keywords = ['神经调控', '电刺激', '磁刺激', '三叉神经','眶上神经','经皮神经','经颅', '迷走神经', '枕神经', '神经电刺激','治疗方式', '物理治疗', '刺激方式', '联合刺激', '额']
        
        # 优先从"神经调控方法"列提取
        target_col_keywords = ['神经调控', '非药物', '方法', '治疗方式', '物理治疗', '刺激方式']
        for key in data_map:
            if any(kw in key for kw in target_col_keywords):
                value = data_map[key].strip()  # 先去空格
                if value and any(kw in value for kw in device_keywords):
                    # 移除参考文献标注（支持[12]、[34,56]、[78-90]格式）
                    cleaned = re.sub(r'\[\d+[,\d\-]*\]', '', value).strip()
                    return cleaned  # 返回清洗后的值
        
        # 从第一列提取
        if cells and len(cells) > 0:
            first_cell = cells[0].strip()
            if first_cell and any(kw in first_cell for kw in device_keywords):
                cleaned = re.sub(r'\[\d+[,\d\-]*\]', '', first_cell).strip()
                return cleaned
        
        return None
    
    def _parse_table_row_to_document(self, headers, cells, caption, phase, footnotes, pdf_id, row_idx):
        """将表格行解析为ES文档"""
        
        if headers:
            header_text = ''.join(headers).lower()
            # 说明性表格的特征表头
            explanatory_patterns = [
            # 基础说明类
            '关键因素', '解释', '建议', '解释与建议',
            # 分级/评分类
            'grade', 'grading', '分级', '评分', '评定', '筛查',
            # 量表/问卷类
            '测定量表', '失能量表', '测试量表', '量表', '评估问卷',
            # 标准/定义类
            '定义', '分类标准', '诊断标准', '分类',
            # 临床特征类
            '治疗指征', '患者特征', '病史的获取', '头痛的鉴别', '个体化'
            ]
        # 检查是否包含任意说明性关键词
            if any(pattern.lower() in header_text for pattern in explanatory_patterns):
                return None  
        
        data_map = {}
        for i, header in enumerate(headers):
            if i < len(cells):
                data_map[header] = cells[i]
        
        drug_name = self._extract_drug_name(data_map, headers, cells)
        
        if drug_name:
            # 排除纯数字/剂量格式
            if re.match(r'^[\d\s~～\-,，.]+(?:IU|mg|g|ml|U)?$', drug_name):
                print(f"    ⚠️  跳过：提取到的药物名是纯剂量 '{drug_name}'")
                return None
            
            # 排除包含"按需"、逗号的描述性文本
            if '按需' in drug_name or re.match(r'^\d+[,，]', drug_name):
                print(f"    ⚠️  跳过：提取到的药物名是描述文本 '{drug_name}'")
                return None
            
            if any(word in drug_name for word in ['相关', '问题', '经验', '建议', '解释', '如果', '患者']):
                return None
        
        treatment_type = self._determine_treatment_type(caption, drug_name or '')
        
        if treatment_type == 'physical':
            # 尝试从第一列提取设备名称
            physical_device = self._extract_physical_device_name(data_map, headers, cells)
            if physical_device:
                drug_name = physical_device

        if not drug_name or drug_name == '/':
            return None
        
        if not phase:
            phase = self._extract_phase_from_caption(caption)
            
        strength_value = data_map.get('推荐强度', '') or data_map.get('推荐等级', '') or data_map.get('推荐级别', '')
        level_value = data_map.get('证据等级', '') or data_map.get('证据级别', '')
        
        doc = {
            "rule_id": str(uuid.uuid4()),
            "disease": "偏头痛",
            "phase": phase,
            "treatment_type": treatment_type,
            "population": "成人",
            "language": "zh-CN",
            "applicability": {
                "age_range": ">=18",
                "pregnancy_lactation": self._extract_pregnancy_status(data_map),
                "comorbidities_excluded": []
            },
            
            "evidence": {
                "evidence_level": self._normalize_evidence_level(level_value) if level_value else "",
                "recommendation_grade": self._normalize_recommendation_grade(strength_value) if strength_value else "",
                "consistency": "有证据等级或推荐强度" if (strength_value or level_value) else ""
            },
            
            "safety": {
                "adverse_effects": {
                    "common": self._extract_adverse_effects(data_map),
                    "rare": "",
                    "severity": self._infer_severity(data_map)
                },
                "contraindications": self._extract_contraindications_list(data_map),
                "precautions": []
            },
            "source": {
                "guideline": self.source_guideline or '',
                "version": f"{self.guideline_year}版" if self.guideline_year else "",
                "year": self.guideline_year,
                "section": "",
                "table_title": caption,
                "pdf_id": pdf_id or "",
                # "page": None,
                "source_type": "table"
            },
            "traceability": {
                "table_row_index": row_idx,
                "footnote_refs": footnotes,
                "raw_text_segment": ' | '.join([f"{h}: {c}" for h, c in zip(headers, cells) if c])
            },
            "extraction_method": "table_parser"
        }
        
        if treatment_type == 'drug':
            dose_info = self._parse_dose_from_table(data_map)
            category = self._extract_category(caption, data_map)
            if not category:  
                category = self._infer_drug_category(drug_name)
            
            doc["intervention"] = {
                "name": drug_name,
                "category": self._extract_category(caption, data_map),
                "sub_category": "",
                "route": {
                    "primary": dose_info['route'],
                    "description": self._get_route_description(dose_info)
                },
                "frequency": dose_info['frequency'],
                "course": self._extract_course(data_map),
                "constraints": dose_info['constraints'],
                "modality_specific": {
                    "drug_specific": {
                        "dose": dose_info['dose'],
                        "dosing_logic": dose_info.get('dosing_logic', {})
                    }
                }
            }
        
        elif treatment_type == 'physical':
            doc["intervention"] = {
                "name": drug_name,
                "category": "物理疗法",
                "sub_category": "",
                "route": {
                    "primary": "external",
                    "description": self._extract_physical_description(data_map)
                },
                "frequency": self._extract_frequency_from_table(data_map),
                "course": self._extract_course(data_map),
                "constraints": [],
                "modality_specific": {
                    "physical_specific": {
                        "target_site": self._extract_target_site(drug_name),
                        "device": drug_name,
                        "parameters": {},
                        "duration_per_session": ""
                    }
                }
            }
        
        elif treatment_type == 'tcm':
            doc["intervention"] = {
                "name": drug_name,
                "category": "中医疗法",
                "sub_category": "",
                "route": {
                    "primary": "acupuncture",
                    "description": "中医"
                },
                "frequency": self._extract_frequency_from_table(data_map),
                "course": self._extract_course(data_map),
                "constraints": [],
                "modality_specific": {
                    "tcm_specific": {
                        "tcm_pattern": "",
                        "symptoms": [],
                        "acupoints": {
                            "primary": self._extract_acupoints(drug_name),
                            "secondary": [],
                            "method": "中医疗法"
                        },
                        "herbal_adjunct": ""
                    }
                }
            }
        
        return doc
    
    def _extract_drug_name(self, data_map, headers, cells):
        """提取药物/治疗方式名称 """
        drug_column_keywords = [
            '药物及治疗方式', '药物', '治疗方式', '药品名称', '药物名称',
            'drug name', 'medication', 'treatment', 'therapy', 'intervention',
            '推荐药物', '治疗药物'
        ]
        
        def is_valid_name(text):
            """综合验证函数 - 保留所有原有检查"""
            if not text or not isinstance(text, str):
                return False
            text = text.strip()
            
            # ✅ 第1层：纯数字/剂量/时间格式过滤
            dose_time_patterns = [
                r'^[\d\s~～\-,，.]+$',  # 纯数字符号（如"25~100"）
                r'^\d+[\s~～\-]\d+\s*(?:IU|mg|g|ml|U)?$',  # 剂量范围（如"155~195 IU"）
                r'^\d+[.,，]',  # 数字+逗号开头（如"75，按需服用"）
                r'^\d+天$',  # 纯数字+天（如"3天"）
                r'^\d+或\d+天$',  # 数字或数字+天（如"4或5天"）
                r'^\d+$',  # 纯数字（如"100"）
                r'^\d+[\u4e00-\u9fa5]',  # 数字+中文（如"5年"）
                r'^\d+[天周月年]',  # 数字+时间单位（如"7周"）
                r'^.*或更多$',  # xxx或更多（如"10mg或更多"）
                r'^.*以上$',  # xxx以上（如"5天以上"）
                r'^.*以下$',  # xxx以下（如"3天以下"）
            ]
            for pattern in dose_time_patterns:
                if re.match(pattern, text):
                    return False
            
            # ✅ 第2层：排除模式过滤
            exclude_patterns = [
                # 药物分类类（保留所有）
                r'^非甾体抗炎药', r'^曲坦类', r'^麦角胺类', r'^吉泮类', r'^地坦类',
                r'^抗癫痫药', r'^β受体阻滞剂', r'^钙离子拮抗剂', r'^钙通道拮抗剂', r'^抗抑郁药',
                r'^特异性治疗', r'^非特异性治疗', r'^含咖啡因', r'^CGRP', r'^CGRP单抗',
                # 治疗/分型类（保留所有）
                r'^神经调控方法', r'^证型', r'^治疗部位',
                # 失能/程度类（保留所有）
                r'^完全失能', r'^部分失能', r'^无失能', r'^严重失能',
                r'^轻度', r'^中度', r'^重度', r'^严重',
                # 动作类（保留所有）
                r'^需要', r'^应用', r'^考虑', r'^需要考虑', r'^需要应用', r'^考虑应用',
            ]
            for pattern in exclude_patterns:
                if re.match(pattern, text):
                    return False
            
            # ✅ 第3层：排除研究描述
            research_phrases = [
                '项', '研究', '试验', '发现', '显示', '结果', 
                '对照', '安慰剂', '荟萃', '分析', '小规模', '大规模',
                '补充', '服用', '注射首次', '之后', '个位点', '共注射'
            ]
            research_count = sum(1 for phrase in research_phrases if phrase in text)
            if research_count >= 2:
                return False
            
            # ✅ 第4层：排除特定短语
            invalid_exact_phrases = [
                '每日补充', '每天补充', '每天服用', '每日服用',
                '皮下注射首次', '之后每月给药', '个位点共注射',
                '项针对', '项安慰剂', '项对照', '需要应用', '需要考虑应用',
                '考虑应用', '或更多'
            ]
            if any(phrase in text for phrase in invalid_exact_phrases):
                return False
            
            # ✅ 第5层：排除频次词
            frequency_words = ['每日', '每次', '每周', '每月', '每年', '每天', '每小时', '每2天', '每3天']
            if text in frequency_words:
                return False
            
            # ✅ 第6层：排除过长描述性文本
            if len(text) > 15 and any(word in text for word in ['的', '为', '在', '与', '和', '或']):
                return False
            
            # ✅ 第7层：排除描述性模式
            invalid_patterns = [
                r'^(?:推荐|剂量|治疗|预防|围经期|短期)',
                r'(?:推荐|剂量|治疗|预防)$',
                r'推荐剂量为',
                r'的推荐',
            ]
            for pattern in invalid_patterns:
                if re.search(pattern, text):
                    return False
            
            # ✅ 第8层：排除参考文献标记（但这个会在clean_name中处理，这里只是预检查）
            if re.search(r'^\[\d+[,\d\-]*\]$', text):  # 纯参考文献
                return False
            
            # ✅ 第9层：排除特殊符号开头
            if text.startswith(('·', '•', '●', '○', '—', '-', '→', '①', '②', '③')):
                return False
            
            # ✅ 第10层：排除提示性文字
            alert_keywords = ['禁忌证', '注意事项', '不良反应', '使用前', '携带', '寻求', '选择', '了解', '告诉', '分享']
            if any(kw in text for kw in alert_keywords):
                return False
            
            # ✅ 第11层：排除剂量描述
            if re.match(r'^\d+[.,]?\d*\s*[\(（]?(?:口服|鼻腔|皮下|肌注|静脉)', text):
                return False
            if re.match(r'^不?[超过低于大于小于等于]\s*\d+', text):
                return False
            
            # ✅ 第12层：排除单字母
            if len(text) == 1 and text.isalpha():
                return False
            
            # ✅ 第13层：排除研究关键词
            research_keywords = ['对比', '研究表明', '约占比', '一项研究', '数据显示']
            if any(kw in text for kw in research_keywords):
                return False
            
            # ✅ 第14层：排除穴位组合
            if '、' in text and any(point in text for point in ['穴', '百会', '风池', '太阳', '率谷']):
                return False
            
            # ✅ 第15层：排除纯符号
            if re.match(r'^[\s\/\-\+\*\&]*$', text):
                return False
            
            # ✅ 第16层：排除症状关键词
            symptom_keywords = [
                '失能', '疼痛', '头痛', '恶心', '呕吐', 
                '畏光', '怕声', '发作', '缓解', '改善',
                '完全', '部分', '轻度', '中度', '重度',
                '严重', '无效', '有效', '复发', '残留', '需要', '应用'
            ]
            if any(kw in text for kw in symptom_keywords):
                return False
            
            # ✅ 第17层：包含"或"的短文本过滤
            if '或' in text and len(text) < 15:
                return False
            
            # ✅ 第18层：排除包含特定动词/形容词的文本
            invalid_verbs = [
                '增加了', '减少了', '降低了', '升高了', '治疗', '发作', '有效',
                '相关', '问题', '经验', '建议', '解释', '选择', '避免',
                '兼顾', '损坏', '加重', '告诉', '优先', '能', '会'
            ]
            if any(verb in text for verb in invalid_verbs):
                return False
            
            # ✅ 第19层：排除以"的"开头的文本
            if text.startswith('的'):
                return False
            
            # ✅ 第20层：排除以"年"结尾的统计描述
            if text.endswith('年') or text.endswith('年增加了'):
                return False
            
            # ✅ 第21层：排除包含"患者"、"问题"等描述性词汇
            descriptive_keywords = [
                '患者', '问题', '建议', '解释', '说明', '注意', '关注',
                '如果', '或避免', '能兼顾', '会加重', '有其他', '过去',
                '告诉医师', '与具体', '相关'
            ]
            if any(kw in text for kw in descriptive_keywords):
                return False
            
            return True
        
        def clean_name(raw_name):
            """清理药物名"""
            if not raw_name:
                return ''
            cleaned = re.sub(r'\[\d+(?:[,\-]\d+)*\]', '', raw_name)
            cleaned = re.sub(r'^[·•●○—\-→]+\s*', '', cleaned)
            cleaned = re.sub(r'[\&\*\#\^]+', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if not cleaned or re.match(r'^[\W_]+$', cleaned):
                return ''
            return cleaned
        
        def final_drug_validation(candidate_name):
            """最终药物名验证"""
            if not candidate_name or len(candidate_name) < 2:
                return False
            
            # 必须包含中文或字母
            if not re.search(r'[\u4e00-\u9fa5a-zA-Z]', candidate_name):
                return False
            
            # 药物名特征检查（至少满足一项）
            target_drugs = [
                # 非甾体抗炎药
                '布洛芬', '阿司匹林', '双氯芬酸', '萘普生', '酮洛芬', 
                '吲哚美辛', '塞来昔布', '安乃近', '安替比林', '托芬那酸',
                '阿司匹林赖氨酸盐', '赖安匹林',
                # 对乙酰氨基酚类
                '对乙酰氨基酚', '扑热息痛',
                # 曲普坦类
                '舒马曲普坦', '舒马普坦', '佐米曲普坦', '利扎曲普坦', 
                '利扎曲坦', '那拉曲普坦', '那拉曲坦', '阿莫曲坦', 
                '依来曲坦', '夫罗曲坦',
                # 麦角胺类
                '麦角胺', '双氢麦角胺', '酒石酸麦角胺', '麦角胺咖啡因',
                # 地坦类
                '拉米地坦',
                # 吉泮类
                '来那吉泮', '瑞美吉泮', 'rimegepant', '乌布吉泮', 
                '扎维吉泮', '阿托吉泮', 'atogepant', 'Telcagepant (MK0974)',
                # 钙离子拮抗剂
                '氟桂利嗪', '盐酸氟桂利嗪', '普瑞巴林',
                # 抗癫痫药
                '丙戊酸钠', '丙戊酸', '托吡酯', '加巴喷丁', '拉莫三嗪',
                '双丙戊酸钠', '丙戊酸盐', '丙戊酸镁', '左乙拉西坦',
                # β受体阻滞剂
                '美托洛尔', '普萘洛尔', '比索洛尔', '阿替洛尔',
                # 抗抑郁药
                '阿米替林', '文拉法辛', '去甲替林', '替扎尼定',
                # CGRP单抗
                '依瑞奈尤单抗', 'Erenumab', '瑞玛奈珠单抗', 'Fremanezumab',
                '加卡奈珠单抗', 'Galcanezumab', '艾普奈珠单抗', 'Eptinezumab',
                # 肉毒毒素类
                '肉毒素', '肉毒毒素', 'A型肉毒毒素', 'A型肉毒毒', '肉毒杆菌',
                # 血管紧张素受体拮抗剂(ARB)
                '坎地沙坦',
                # 血管紧张素转换酶抑制剂(ACEI)
                '赖诺普利',
                # 阿片类
                '吗啡', '可待因', '羟考酮', '洛芬待因', 
                '氨酚待因', '氨酚氢可酮', '布托啡诺',
                # 止吐药
                '甲氧氯普胺', '多潘立酮', '氯丙嗪', '异丙嗪',
                # 复合制剂
                '阿司匹林/咖啡因', '去痛片', '脑清片', '头痛粉',
                '酚咖片', '氨酚巴妥片', '新复方大青叶片', '散利痛',
                # 镁剂类
                '镁剂', '柠檬酸镁', '苹果酸镁', '甘氨酸镁', '氧化镁', '硫酸镁',
                # 其他西药
                '苯二氮䓬类', '巴比妥类镇静剂', '核黄素', '辅酶 Q10',
                # 中医方剂
                '天舒胶囊', '都梁软胶囊', '川芎茶调散', '散偏汤', '正天丸',
                '通天口服液', '川芎清脑颗粒', '天麻钩藤饮', '养血清脑颗粒',
                '丹珍头痛胶囊', '头痛宁胶囊', '通窍活血汤', '血府逐瘀汤', '通心络胶囊'
            ]
            cleaned_candidate = candidate_name.strip().lower()
            target_drugs_lower = [drug.lower() if isinstance(drug, str) else drug for drug in target_drugs]
            if cleaned_candidate in target_drugs_lower:
                return True
            
            drug_chars = [
                '林', '芬', '坦', '胺', '酸', '醇', '素', '西', '普', '他', '洛', '替', '嗪', '丁', '喷', '利', '米', '托', '巴']
            has_drug_char = any(char in candidate_name for char in drug_chars)
            has_category = self._infer_drug_category(candidate_name) != ''
            is_reasonable_length = 2 <= len(candidate_name) <= 30
            
            if not (has_drug_char or has_category or is_reasonable_length):
                return False
            
            return True
        
        # ===== 提取逻辑（按优先级） =====
        
        # 🔹 优先级1：从"药物"列提取
        normalized_data_map = {}
        if data_map and isinstance(data_map, dict):
            for k, v in data_map.items():
                normalized_key = k.strip().lower()
                normalized_data_map[normalized_key] = v
        
        for key in drug_column_keywords:
            norm_key = key.strip().lower()
            if norm_key in normalized_data_map:
                cell_value = normalized_data_map[norm_key]
                if is_valid_name(cell_value):
                    cleaned = clean_name(cell_value)
                    if cleaned and final_drug_validation(cleaned):
                        return cleaned
        
        # 🔹 优先级2：从cells[0]提取（第1列通常是药物名）
        if cells and len(cells) > 0:
            first_cell = cells[0]
            if is_valid_name(first_cell):
                cleaned = clean_name(first_cell)
                if cleaned and final_drug_validation(cleaned):
                    return cleaned
        
        # 🔹 优先级3：从cells[1]提取（有些表格第2列是药物名）
        if cells and len(cells) > 1:
            second_cell = cells[1]
            if is_valid_name(second_cell):
                cleaned = clean_name(second_cell)
                if cleaned and final_drug_validation(cleaned):
                    return cleaned
        
        # 🔹 优先级4：遍历所有cells（兜底）
        for cell in cells:
            if is_valid_name(cell):
                cleaned = clean_name(cell)
                if cleaned and final_drug_validation(cleaned):
                    return cleaned
        
        return '' 

    def _determine_treatment_type(self, caption, drug_name):
        """判断治疗类型"""
        caption = str(caption).lower().strip() if caption else ""
        drug_name = str(drug_name).lower().strip() if drug_name else ""
        if '物理' in caption or '神经调控' in caption or '非药物治疗' in caption or '神经电刺激' in caption:
            return 'physical'
        elif '中医' in caption or '针灸' in drug_name or '取穴' in drug_name or '针刺' in caption or '毫针刺法' in caption:
            return 'tcm'
        else:
            return 'drug'

    def _extract_category(self, caption, data_map):
        """提取药物分类"""
        categories = ['非甾体抗炎药', 'COX-2抑制剂', '曲普坦类', '曲坦类', 
                    '麦角胺类', '地坦类', '吉泮类', '乙酰苯胺类', '复合制剂',
                    '钙离子拮抗剂', '钙通道拮抗剂', '抗癫痫药', '抗惊厥药',
                    'β受体阻滞剂', 'β-受体阻断剂', '抗抑郁药', 'CGRP', 
                    '肉毒毒素', '血管紧张素', '止呕剂', '其他药物']
        data_map = data_map or {}
        caption = str(caption).lower()
        data_vals = [str(v).lower() for v in data_map.values()]
        
        for cat in categories:
            cat_lower = cat.lower()
            if cat_lower in caption or any(cat_lower in v for v in data_vals):
                return cat
        
        return data_map.get('分类', '').strip()

    def _parse_dose_from_table(self, data_map):
        """解析表格中的剂量信息
            优先级：
        1. 优先从表头提取单位（大部分情况）
        2. 如果表头没有单位，_parse_dose_to_structured 会从单元格内容提取（兜底）
        支持的表头格式：
        - 推荐剂量(mg)、推荐剂量（mg）
        - 推荐剂量/mg
        - 推荐剂量 mg
        - 推荐剂量mg
        
        支持的复合单位：
        - mg/d, mg/次, IU/日, μg/kg 等
        """
        dose_text = ""
        max_dose_text = ""
        
        dose_unit_from_header = ""
        max_unit_from_header = ""
        
        for k, v in data_map.items():
            if v and any(keyword in k for keyword in ['推荐剂量', '每次剂量', '单次', '每日剂量']):
                dose_text = v
                
                # 匹配模式：
                # - 基础单位: mg, g, IU, μg 等
                # - 复合单位: mg/d, mg/次, IU/日, μg/kg 等
                # 例如：推荐剂量(mg/d) → mg/d
                #      推荐剂量(IU/日) → IU/日
                #      推荐剂量(mg) → mg
                
                dose_unit_from_header = self._extract_unit_from_header(k)
                
            elif v and any(keyword in k for keyword in ['最大剂量', '每日最大', '每日最大剂量']):
                max_dose_text = v
                max_unit_from_header = self._extract_unit_from_header(k)
        
        route = 'oral'
        
        for k, v in data_map.items():
            if v and ('药物' in k or '治疗' in k or '剂量' in k):
                if '口服' in v or '口崩' in v:
                    route = 'oral' if '口服' in v else 'oral_disintegrating'
                elif '鼻喷' in v or '鼻腔' in v:
                    route = 'nasal'
                elif '皮下' in v:
                    route = 'subcutaneous'
                elif '静脉' in v or '静滴' in v:
                    route = 'intravenous'
                elif '肌注' in v or '肌肉' in v or '肌内注射' in v:
                    route = 'intramuscular'
                elif '直肠' in v or '肛栓' in v:
                    route = 'rectal'
                break
        
        if not route or route == 'oral':
            combined_text = dose_text + ' ' + max_dose_text
            route = self._extract_route(combined_text)
            
        recommended_dose = self._parse_dose_to_structured(dose_text, dose_unit_from_header)
        max_dose = self._parse_dose_to_structured(max_dose_text, max_unit_from_header) if max_dose_text else None
        
        constraints = []
        combined_constraint_text = dose_text + ' ' + max_dose_text
        for k, v in data_map.items():
            if v and '注意' in k:
                combined_constraint_text += ' ' + v
        
        constraints.extend(self._extract_constraints(combined_constraint_text))
        
        dose_info = {
            'route': route,
            'dose': {
                'recommended': recommended_dose,
                'max_daily': max_dose,
                'adjustment': {}
            },
            'frequency': self._extract_frequency(dose_text + ' ' + max_dose_text),
            'constraints': constraints,
            'dosing_logic': {
                'interval_min_hours': self._extract_interval_hours(combined_constraint_text)
            }
        }

        return dose_info
    
    def _extract_unit_from_header(self, header_text: str) -> str:
        """从表头提取单位（兼容多种格式）
        
        支持格式：
        1. 括号: 推荐剂量(mg)、推荐剂量（mg）、推荐剂量(mg/d)
        2. 斜杠: 推荐剂量/mg、推荐剂量/mg/d
        3. 空格: 推荐剂量 mg、推荐剂量 mg/d
        
        Args:
            header_text: 表头文本
            
        Returns:
            str: 提取的单位（可能是复合单位，如 mg/d）
            
        Examples:
            "推荐剂量(mg/d)" → "mg/d"
            "推荐剂量/mg" → "mg"
            "推荐剂量 mg" → "mg"
            "推荐剂量" → ""
        """
        if not header_text:
            return ''
        
        # ===== 策略1：从括号中提取（优先级最高）=====
        # 兼容半角括号() 和 全角括号（）
        # 匹配: (mg), (mg/d), (IU/日), （mg）, （mg/次）
        bracket_pattern = r'[（(]([a-zA-Zμ%]+(?:/[a-zA-Z日次时周月年kg体重]+)?)[)）]'
        bracket_match = re.search(bracket_pattern, header_text)
        if bracket_match:
            return bracket_match.group(1)
        
        # ===== 策略2：从斜杠分隔中提取 =====
        # 匹配: 推荐剂量/mg, 推荐剂量/mg/d
        # 注意：要排除已经是单位一部分的斜杠（如 mg/d）
        
        # 先检查是否有复合单位的情况
        # 例如: 推荐剂量/mg/d → 应该提取 mg/d，而不是 mg
        slash_pattern = r'/([a-zA-Zμ%]+(?:/[a-zA-Z日次时周月年kg体重]+)?)(?:\s|$)'
        slash_match = re.search(slash_pattern, header_text)
        if slash_match:
            return slash_match.group(1)
        
        # ===== 策略3：从空格分隔中提取 =====
        # 匹配: 推荐剂量 mg, 每日剂量 mg/d
        space_pattern = r'\s+([a-zA-Zμ%]+(?:/[a-zA-Z日次时周月年kg体重]+)?)$'
        space_match = re.search(space_pattern, header_text)
        if space_match:
            return space_match.group(1)
        
        # ===== 策略4：兜底 - 检查是否包含常见单位词 =====
        # 如果表头中明确包含 "mg" 但前面的策略都没匹配到
        if 'mg' in header_text.lower():
            return 'mg'
        
        # 没有找到单位
        return ''

    def _extract_route(self, text):
        """提取给药途径"""
        routes = {
            '口崩': 'oral_disintegrating',
            '崩解': 'oral_disintegrating',
            '口服': 'oral',
            '肌注': 'intramuscular',
            '肌肉注射': 'intramuscular',
            '肌内注射': 'intramuscular',  
            '静脉': 'intravenous',
            '静滴': 'intravenous',
            '鼻喷': 'nasal',
            '鼻腔': 'nasal',
            '直肠': 'rectal',
            '肛栓': 'rectal',
            '皮下': 'subcutaneous',
            '皮下注射': 'subcutaneous'
        }
        
        for cn, en in routes.items():
            if cn in text:
                return en
        
        return 'oral'

    def _extract_frequency(self, text):
        """提取用药频次"""
        chinese_num_map = {
            '零': '0', '一': '1', '二': '2', '两': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
            '半': '0.5', '壹': '1', '贰': '2', '叁': '3', '肆': '4', '伍': '5',
            '陆': '6', '柒': '7', '捌': '8', '玖': '9', '拾': '10'
        }
        
        def process_times(times_str):
            """处理频次数字（兼容中文数字、小数、整数）"""
            if not times_str:
                return '1'
            
            # 第一步：替换中文数字为阿拉伯数字
            for cn, an in chinese_num_map.items():
                times_str = times_str.replace(cn, an)
            
            # 第二步：校验并转换为有效数字（兼容小数）
            try:
                # 转浮点数处理小数，再优化显示（如2.0→2，0.5保留）
                num = float(times_str)
                # 整数转成无小数点格式，小数保留原格式
                if num.is_integer():
                    return str(int(num))
                else:
                    return str(num)
            except ValueError:
                # 非数字则返回默认值1
                return '1'
        
        if '按需' in text:
            return '按需服用'
        elif '隔日' in text:
            return '隔日'
        elif '/月' in text or '每月' in text:
            match = re.search(r'(?:每月|/月)\D*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\D*次\D*(?:每月|/月)', text)
            if match:  # ✅ 添加空值检查
                times_str = match.group(1) or match.group(2)
                times = process_times(times_str)
                return f"每月{times}次"
            else:
                return '每月默认1次'  # ✅ 提供默认值
        elif '/季度' in text or '每季度' in text:
            match = re.search(r'(?:每季度|/季度)\D*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\D*次\D*(?:每季度|/季度)', text)
            if match:  # ✅ 添加空值检查
                times_str = match.group(1) or match.group(2)
                times = process_times(times_str)
                return f"每季度{times}次"
            else:
                return '每季度默认1次'  
        elif '每30天' in text:  
            match = re.search(r'每30天\D*(\d+(?:\.\d+)?)', text)
            if match:  # ✅ 添加空值检查
                times_str = match.group(1)
                times = process_times(times_str)
                return f"每30天{times}次"
            else:
                return '每30天默认1次' 
        elif re.search(r'每日\s*\d+\s*次', text):
            match = re.search(r'每日\s*(\d+(?:\.\d+)?)\s*次', text)
            if match:  # ✅ 新增空值检查
                times_str = match.group(1)
                times = process_times(times_str)
                return f"每日{times}次"
            return "每日默认1次"  
        elif 'bid' in text.lower():
            return '每日2次'
        elif 'tid' in text.lower():
            return '每日3次'
        elif 'qd' in text.lower() or '每日' in text:
            return '每日1次'
        
        return ''

    def _extract_constraints(self, text):
        """提取用药约束"""
        constraints = []
        
        if '24小时' in text or '24 h' in text or '24h' in text:
            match = re.search(r'24\s*[小时h]*\s*[内]?不?超过\s*(\d+\.?\d*)\s*(mg|片|次|IU)', text)  
            if match:
                constraints.append(f"24小时内不超过{match.group(1)}{match.group(2)}")
        
        if '30天' in text or '每月' in text:
            match = re.search(r'(?:30天|每月)[^。]{0,50}?(?:不?超过|超过)\s*(\d+)\s*次', text)
            if match:
                if '超过' in text:
                    constraints.append(f"每30天使用不建议超过{match.group(1)}次（超过则安全性未建立）")
                else:
                    constraints.append(f"每30天使用不超过{match.group(1)}次")
        
        if '至少间隔' in text:
            match = re.search(r'至少间隔\s*(\d+)\s*小时', text)
            if match:
                constraints.append(f"首剂后至少间隔{match.group(1)}小时可加服")
        
        if '总剂量' in text:
            match = re.search(r'总剂量[不能]*超过\s*(\d+\.?\d*)\s*(mg|IU)', text)
            if match:
                constraints.append(f"总剂量不超过{match.group(1)}{match.group(2)}")
        
        return constraints

    def _get_route_description(self, dose_info):
        """生成给药途径描述"""
        route_cn = {
            'oral': '口服',
            'oral_disintegrating': '口腔崩解',
            'intramuscular': '肌内注射',
            'intravenous': '静脉注射',
            'nasal': '鼻腔喷雾',
            'rectal': '直肠给药',
            'subcutaneous': '皮下注射'
        }
        
        route_text = route_cn.get(dose_info['route'], '未知途径')
        
        if dose_info['frequency']:
            return f"{dose_info['frequency']} {route_text}"
        return route_text

    def _extract_course(self, data_map):
        """提取疗程信息"""
        for k, v in data_map.items():
            if v and ('疗程' in k or '使用时间' in k or '治疗周期' in k):
                return v
        for k, v in data_map.items():
            if v and '注意' in k and '疗程' in v:
                match = re.search(r'总疗程不超过\s*(\d+)\s*个月', v)
                return match.group(0) if match else v 
        return ""

    def _extract_frequency_from_table(self, data_map):
        """从表格中提取频次"""
        for k, v in data_map.items():
            if v and ('频次' in k or '频率' in k):
                return v
        return ""

    def _extract_physical_description(self, data_map):
        """提取物理治疗描述"""
        desc_parts = []
        for k, v in data_map.items():
            if v and k in ['参数', '方法', '说明', '神经调控', '治疗方式', '治疗部位', '操作']:
                desc_parts.append(v)
        return ' | '.join(desc_parts) if desc_parts else ""

    def _extract_target_site(self, device_name):
        """从设备名称推断治疗部位"""
        site_keywords = {
            '头部': ['头', '颅', '经颅', '脑', '额', '眶上', '三叉', '滑车', '枕'],
            '颈部': ['颈'],
            '上臂': ['臂', '上肢', '远隔'],
            '迷走神经': ['迷走']
        }
    
        sites = []
        for site, keywords in site_keywords.items():
            if any(kw in device_name for kw in keywords):
                sites.append(site)
        
        return sites if sites else []

    def _extract_acupoints(self, acupoint_text):
        """提取穴位列表"""
        common_points = ['风池', '百会', '太冲', '太阳', '合谷', '印堂', '率谷',
                        '丝竹空', '阿是穴', '外关', '阳陵泉', '头维', '内庭',
                        '天柱', '昆仑', '后溪', '内关', '三阴交', '足三里']
        found_points = []
        for point in common_points:
            if point in acupoint_text:
                found_points.append(point)
        return found_points

    def _extract_adverse_effects(self, data_map):
        """提取不良反应
            处理逻辑：
        1. 优先从"不良反应"列提取
        - 如果有"不良反应："标记 → 提取标记部分
        - 如果没有标记但有"禁忌证："→ 排除禁忌证部分，返回剩余
        - 如果都没有标记 → 返回全部内容（整列都是不良反应）
        
        2. 如果没有"不良反应"列，从"注意事项"列提取
        - 如果有"不良反应："标记 → 提取标记部分
        - 如果没有标记 → 返回全部（无法区分）
        
        Examples:
            # 情况1：仅不良反应
            {'主要不良反应': '胃肠道反应、头晕'}
            → "胃肠道反应、头晕"
            
            # 情况2：混合（有标记）
            {'主要不良反应': '不良反应：头晕；禁忌证：冠心病'}
            → "头晕"
            
            # 情况3：混合（无标记，但有禁忌证标记）
            {'主要不良反应': '头晕、恶心；禁忌证：冠心病'}
            → "头晕、恶心"
            
            # 情况4：注意事项
            {'注意事项': '不良反应：头晕；禁忌证：冠心病'}
            → "头晕
            """
            # ===== 步骤1：优先从"不良反应"列提取 =====
        adverse_keys = ['不良反应', '主要不良反应', '常见不良反应', '副作用']
        
        for key in adverse_keys:
            if key in data_map and data_map[key]:
                text = str(data_map[key]).strip()
                
                if not text:
                    continue
                
                # 提取不良反应部分
                result = self._extract_adverse_from_text(text)
                return result if result else ''
        
        # ===== 步骤2：从"注意事项"列提取 =====
        if '注意事项' in data_map and data_map['注意事项']:
            text = str(data_map['注意事项']).strip()
        
            if text:
                result = self._extract_adverse_from_text(text)
                return result if result else ''
        
        return ''
    
    
    def _extract_adverse_from_text(self, text: str) -> str:
        """从文本中提取不良反应部分（保留标记）
        
             策略：
        1. 如果有"不良反应"标记 → 提取带标记的完整部分
        2. 如果有"禁忌/禁用/慎用/禁止"标记 → 提取这些标记之外的部分
        3. 如果都没有 → 返回全部文本（轻量清理后）
        
        核心原则：
        - 在"不良反应"列中，除了明确的禁忌内容外，其余都是不良反应
        - 不使用关键词判断（不良反应种类太多，无法穷举）
        - 保留所有不良反应相关标记
        
        Args:
            text: 原始文本
            
        Returns:
            str: 提取的不良反应部分
        """
        if not text:
            return ''
        
        # ===== 策略1：有明确的"不良反应"标记（多种格式） =====
        
        # 模式1a：不良反应：xxx → 兼容无冒号（不良反应xxx）
        match = re.search(r'((?:主要|常见)?不良反应[：:]?\s*[^；;。]+)', text)
        if match:
            adverse_text = match.group(1).strip()
            return adverse_text
        
        # 模式1b：不良反应 + 连接词 + 内容
        # 例如：不良反应主要有、不良反应包括、不良反应为
        match = re.search(r'((?:主要|常见)?不良反应(?:主要|常见)?(?:有|包括|为|是)\s*[^；;。禁忌]+)', text)
        if match:
            adverse_text = match.group(1).strip()
            return adverse_text
        
        # 模式1c：xxx不良反应（嵌入式）
        # 例如：胃肠道不良反应、心血管不良反应
        # 但要排除单独出现的"不良反应"标题
        if re.search(r'\S+不良反应', text) and not text.strip().startswith('不良反应'):
            if '禁忌' in text or '禁用' in text or '慎用' in text or '禁止' in text:
                adverse_text = self._extract_before_contraindications(text)
                if adverse_text:
                    return adverse_text
            else:
                return text
        
        # ===== 策略2：没有"不良反应："但有"禁忌证："标记 =====
        # 例如: "头晕、恶心；禁忌证：冠心病"
        # 或: "头晕、恶心；冠心病禁用"
        
        if re.search(r'禁忌|禁用|慎用|禁止', text) and '不良反应' not in text:
            adverse_text = self._extract_before_contraindications(text)
            if adverse_text:
                return adverse_text
        
        # ===== 策略3：没有任何标记 =====
        # 假设整个文本都是不良反应
        # 清理前缀后返回
        cleaned_text = text.strip()
        return cleaned_text
    
    def _extract_before_contraindications(self, text: str) -> str:
        """提取禁忌相关内容之外的部分（适配无冒号/禁忌在前/后，保留不良反应标记）
        
        匹配多种禁忌表达方式，无匹配时返回完整文本（避免丢失信息）：
        - 禁忌证：xxx
        - 禁忌xxx使用
        - xxx禁用
        - xxx慎用
        - 禁止xxx
        
        Args:
            text: 原始文本
            
        Returns:
            str: 禁忌内容之前的部分
        """
        if not text:
            return ''
        
        # 模式1：禁忌在后 → 提取禁忌之前的部分（兼容无冒号）
        patterns_after = [
            r'(.+?)[；;。、，]?\s*禁忌[症证][：:]?',      # 头晕；禁忌证冠心病
            r'(.+?)[；;。、，]?\s*禁忌\S*使用?',          # 头晕；禁忌使用
            r'(.+?)[；;。、，]?\s*\S+[者患]*禁用',       # 头晕；冠心病禁用
            r'(.+?)[；;。、，]?\s*\S+[者患]*慎用',       # 头晕；冠心病慎用
            r'(.+?)[；;。、，]?\s*禁止',                 # 头晕；禁止使用
        ]
        
        for pattern in patterns_after:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # 模式2：禁忌在前 → 提取禁忌之后的部分（兼容无冒号）
        patterns_before = [
            r'禁忌[症证][：:]?\s*[\s\S]+?[；;。、，]([\s\S]+)',  # 禁忌证冠心病；头晕
            r'\S+[者患]*禁用\s*[\s\S]+?[；;。、，]([\s\S]+)',   # 冠心病禁用；头晕
            r'\S+[者患]*慎用\s*[\s\S]+?[；;。、，]([\s\S]+)',   # 冠心病慎用；头晕
        ]
        
        for pattern in patterns_before:
            match = re.search(pattern, text)
            if match and match.group(1).strip():
                return match.group(1).strip()
        
        # 无匹配到禁忌位置 → 返回完整文本（核心优化）
        return text.strip()
     
    def _extract_contraindications_list(self, data_map):
        """提取禁忌症列表（保留所有禁忌症标记，最终版）
        
        处理逻辑：
        1. 优先从"禁忌症"列提取（整列都是禁忌症）
        2. 如果没有，从"不良反应"列提取（可能混合）
        - 如果有"禁忌证："标记 → 提取标记部分
        - 如果没有标记 → 不提取（避免误判）
        3. 如果还没有，从"注意事项"列提取
        - 如果有"禁忌证："标记 → 提取标记部分
        - 如果没有标记 → 不提取（避免误判）
        
        Examples:
            # 情况1：专门的禁忌症列
            {'主要禁忌证': '冠心病、严重肝损害'}
            → ['冠心病', '严重肝损害']
            
            # 情况2：不良反应列混合
            {'主要不良反应': '不良反应：头晕；禁忌证：冠心病、高血压'}
            → ['冠心病', '高血压']
            
            # 情况3：注意事项列
            {'注意事项': '头晕；禁忌证：冠心病'}
            → ['冠心病']
            
            # 情况4：无明确标记（不提取，避免误判）
            {'主要不良反应': '头晕、恶心'}
            → []
        """
        # ===== 步骤1：优先从"禁忌症"列提取 =====
        contra_keys = ['禁忌症', '禁忌证', '禁忌', '主要禁忌证', '主要禁忌症']
        
        for key in contra_keys:
            if key in data_map and data_map[key]:
                text = str(data_map[key]).strip()
                
                if not text:
                    continue
                
                # 整列都是禁忌症，清理前缀后分割
                marked_text = f"{key}：{text}" if not text.startswith(key) else text
                return self._extract_contraindication_from_text(marked_text)
        
        # ===== 步骤2：从"不良反应"列提取（可能混合）=====
        adverse_keys = ['不良反应', '主要不良反应', '常见不良反应', '副作用']
        
        for key in adverse_keys:
            if key in data_map and data_map[key]:
                text = data_map[key].strip()
                
                if not text:
                    continue
                
                # 只提取明确标记的禁忌证部分
                extracted = self._extract_contraindication_from_text(text)
                if extracted:
                    return self._extract_contraindication_from_text(extracted)
        
        # ===== 步骤3：从"注意事项"列提取 =====
        if '注意事项' in data_map and data_map['注意事项']:
            text = data_map['注意事项'].strip()
            
            if text:
                extracted = self._extract_contraindication_from_text(text)
                if extracted:
                    return self._extract_contraindication_from_text(extracted)
        
        return []


    def _extract_contraindication_from_text(self, text: str) -> str:
        """从文本中提取禁忌证部分（兼容无冒号，仅提取明确标记）
        
        Args:
            text: 原始文本
            
        Returns:
            str: 提取的禁忌证部分，如果没有明确标记返回空字符串
        """
        if not text:
            return ''
        
        # 匹配带标记的禁忌证（关键词在前，保留标记）
        match_key_first = re.search(r'((?:主要)?禁忌[症证][：:]?\s*[\s\S]*?)(?=；|。|不良反应|$)', text)
        if match_key_first:
            return match_key_first.group(1).strip()
        
         # 模式2：匹配xxx禁用/慎用（关键词在后，保留标记）
        match_key_last = re.search(r'([\s\S]*?(?:禁用|慎用|禁止使用))(?=；|。|不良反应|$)', text)
        if match_key_last:
            return match_key_last.group(1).strip()
        
        # 没有找到明确标记
        return ''


    def _clean_contraindication_prefix(self, text: str) -> str:
        """清理禁忌症文本的冗余前缀（仅去冗余标点/空格，保留所有禁忌症标记）
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        if not text:
            return ''
        
        prefix = ""
        prefix_match = re.search(r'^(?:主要)?禁忌[症证][：:]?', text)
        if prefix_match:
            prefix = prefix_match.group(0).strip()
            # 去掉前缀后清理文本
            text = text[len(prefix):].strip()
        
        # 步骤2：轻量清理文本（仅去冗余标点/空格，不分割）
        text = text.strip()
        text = re.sub(r'[；;，,、\s]+', '，', text)  # 统一分隔符为中文逗号
        text = re.sub(r'^[，。：:]+', '', text)     # 去掉开头冗余标点
        text = re.sub(r'[，。：:]+$', '', text)     # 去掉结尾冗余标点
        
        # 步骤3：补全标记前缀（如果有）
        if prefix:
            text = f"{prefix}{text}"
        
        # 步骤4：最终空值校验
        return text if text else ""

    
    def _infer_severity(self, data_map):
        """推断不良反应严重程度"""
        adverse = self._extract_adverse_effects(data_map).lower()
        if '严重' in adverse or '重度' in adverse or '危及生命' in adverse or '严重不良事件' in adverse:
            return '重度'
        elif '中度' in adverse:
            return '中度'
        elif adverse and '轻度' in adverse:
            return '轻度'
        elif '常见' in adverse or '主要' in adverse:
            return '常见/主要'
        elif adverse:
            return '常见/主要'
        return ''

    def _extract_pregnancy_status(self, data_map):
        """提取妊娠期用药状态"""
        contraindications_text = ' '.join(self._extract_contraindications_list(data_map))
        
        if '妊娠' in contraindications_text or '孕妇' in contraindications_text or '哺乳期' in contraindications_text or '哺乳' in contraindications_text:
            if '禁' in contraindications_text:
                return '禁用（妊娠/哺乳期）'
            elif '慎' in contraindications_text:
                return '慎用（妊娠/哺乳期）'
        
        adverse = self._extract_adverse_effects(data_map)
        if '妊娠' in adverse or '孕妇' in adverse or '哺乳期' in adverse or '哺乳' in adverse:
            if '禁' in adverse:
                return '禁用（妊娠/哺乳期）'
            elif '慎' in adverse:
                return '慎用（妊娠/哺乳期）'
            elif '可用' in adverse or '安全' in adverse:
                return '可用（妊娠/哺乳期）'
        
        return '无数据'
    
    
    def parse_text_blocks_to_elasticsearch(self, pdf_id: Optional[str] = None) -> List[Dict]:
        """将正文块解析为Elasticsearch文档
        """
        pdf_id = pdf_id or self.pdf_id
        es_documents = []
        total_blocks = len(self.text_blocks)
        
        print(f"\n=== 开始处理 {total_blocks} 个文本块 ===")
        if total_blocks == 0:
            print("⚠️  警告：没有找到任何文本块！")
            print("可能原因：")
            print("  1. JSON文件中没有文本内容")
            print("  2. 文本被过滤条件过滤掉了")
            print("  3. JSON结构不符合预期")
        
        for idx, text_block in enumerate(self.text_blocks, 1):
            content = text_block.get('content', '').strip()
            if not content:  # 跳过空文本块
                print(f"[文本块 {idx}/{total_blocks}] 空内容，跳过")
                continue
            
            treatments = self._extract_treatments_from_text(content)
            
            if treatments:
                # 过滤无效治疗方案
                valid_treatments = [t for t in treatments if self._validate_treatment(t)]
                
                if valid_treatments:
                    print(f"[文本块 {idx}/{total_blocks}] 提取到 {len(valid_treatments)} 个治疗方案: "
                          f"{[t['name'] for t in valid_treatments]}")
                
                treatments = valid_treatments
            
            # 转换为ES文档
            for treatment in treatments:
                doc = self._create_es_document(treatment, content, pdf_id)
                if doc:
                    es_documents.append(doc)
        
        return es_documents
    
    def _extract_treatments_from_text(self, text: str) -> List[Dict]:
        """从正文中提取治疗信息
        """
        # 已知中药方剂常量
        KNOWN_TCM = {
            '天舒胶囊', '都梁软胶囊', '川芎茶调散', '散偏汤', '正天丸',
            '通天口服液', '川芎清脑颗粒', '天麻钩藤饮', '养血清脑颗粒',
            '丹珍头痛胶囊', '头痛宁胶囊', '通窍活血汤', '血府逐瘀汤', '通心络胶囊'
        }
        
        treatments = []
        
        # 预处理文本
        cleaned_text = self._preprocess_text(text)
        
        # 模式1: 中药推荐意见格式
        treatments.extend(self._extract_tcm_recommendations(cleaned_text, KNOWN_TCM))
        
        # 模式2: 西药推荐意见格式（带剂量）
        treatments.extend(self._extract_drug_recommendations(text))
        
        # 模式3: 标准剂量格式
        treatments.extend(self._extract_standard_dosage(text))
        
        # 模式4: 中医证型格式
        treatments.extend(self._extract_tcm_syndromes(text, cleaned_text, KNOWN_TCM))
        
        # 模式5: 针灸推荐格式
        treatments.extend(self._extract_acupuncture(cleaned_text))
        
        # 去重
        return self._deduplicate_treatments(treatments)
    
    @staticmethod
    def _preprocess_text(text: str) -> str:
        """预处理文本
        """
        cleaned = text.strip()
        cleaned = cleaned.replace('～', '-').replace('~', '-')
        cleaned = re.sub(r'\s+', '', cleaned)
        # cleaned = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', cleaned)
        # cleaned = re.sub(r'(\d)\s+([a-zμ%gGmLml粒])', r'\1\2', cleaned)
        cleaned = cleaned.replace('：', ':').replace('（', '(').replace('）', ')')
        return cleaned
    
    def _extract_tcm_recommendations(self, cleaned_text: str, known_tcm: set) -> List[Dict]:
        """提取中药推荐意见
        """
        treatments = []
        
        # print(f"\n=== 调试 _extract_tcm_recommendations ===")
        # print(f"输入文本: {cleaned_text[:200]}")
        
        cleaned_text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', cleaned_text)
        cleaned_text = re.sub(r'\\textrm\{([^}]+)\}', r'\1', cleaned_text) 
        cleaned_text = cleaned_text.replace('\\sim', '~')
        cleaned_text = cleaned_text.replace('\\pm', '±') 
        cleaned_text = re.sub(r'[\u3000\s]+', ' ', cleaned_text).strip()
        
        pattern = (r'[①②③④⑤⑥⑦⑧⑨⑩]?(?:推荐|推荐意见|推荐建议)?\s*\d*[:：]?\s*'
               r'([\u4e00-\u9fa5a-zA-Z0-9_\-]+)[,，]?每日(\d+)(剂|次)[,，]'
               r'([^。]+?)[。]?[(（]([12][ABCD])[)）]')
        
        # print(f"清理LaTeX后: {cleaned_text[:200]}") 
        
        # cleaned_text = cleaned_text.replace('～', '-').replace('~', '-')  # 统一为短横线
        # print(f"替换波浪号后: {cleaned_text[:200]}")
        
        # cleaned_text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', cleaned_text)
        # print(f"规范化范围后: {cleaned_text[:200]}")
        
        # cleaned_text = re.sub(r'(?<=\D)\s+(?=\D)', '', cleaned_text)  # 只删除非数字间空格
        # cleaned_text = re.sub(r'(\d)\s+([gGmLml粒])', r'\1\2', cleaned_text) 
        # print(f"删除空格后: {cleaned_text[:200]}")
        
        # cleaned_text = re.sub(r'、', '', cleaned_text)  # 移除顿号
        # cleaned_text = re.sub(r'\s+', '', cleaned_text)  # 移除所有空格
        # cleaned_text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', cleaned_text)
        
        # print(f"增强清理后: {cleaned_text[:200]}")
       
        
        # 兼容多种推荐意见格式
        # pattern = (r'[①②③④⑤⑥⑦⑧⑨⑩]?(?:推荐|推荐意见|推荐建议)?\s*\d*[:：]?\s*'
        #            r'([\u4e00-\u9fa5a-zA-Z0-9_\-]+)[,，]?每日(\d+)(剂|次)[,，]'
        #            r'([^。]+?)[。]?[(（]([12][ABCD])[)）]')
        # pattern = (
        #     r'[①②③④⑤⑥⑦⑧⑨⑩]?'               # 序号
        #     r'(?:推荐意见|推荐)?'                   # 推荐意见前缀
        #     r'\s*\d*[:：]?'                        # 数字和冒号
        #     r'\s*([\u4e00-\u9fa5]{2,10})'          # 中药名（2-10个中文字符）
        #     r'每日(\d+)(剂|次)'                     # 每日X剂/次
        #     r'[^。]*?'                             # 中间内容（如分早晚2次口服）
        #     r'[(（]([12][ABCD])[)）]'              # 证据等级（1B、2C等）
        # )
        
        # for match in re.finditer(pattern, cleaned_text):
        #     prescription_name = match.group(1).strip()
        #     daily_dose = match.group(2).strip()
        #     freq_unit = match.group(3).strip()
        #     freq_desc = match.group(4).strip()
        #     grade = match.group(5).strip() if match.group(5) else ''
            
        for match in re.finditer(pattern, cleaned_text):
            prescription_name = match.group(1).strip()
            daily_dose = match.group(2).strip()
            freq_unit = match.group(3).strip()
            freq_desc = match.group(4).strip() if match.group(4) else ''
            grade = match.group(5).strip() if match.group(5) else ''
            
            # print(f"匹配到中药: {prescription_name}, 剂量: {daily_dose}{freq_unit}, 等级: {grade}")
            
            # 判断是否为已知中药
            if self._is_known_tcm(prescription_name, known_tcm):
                treatment = self._create_tcm_treatment(
                    prescription_name, daily_dose, freq_unit, freq_desc, grade, cleaned_text
                )
                treatments.append(treatment)
            else:
                # 按普通药物处理
                treatment = self._create_drug_treatment_from_recommendation(
                    prescription_name, daily_dose, freq_unit, freq_desc, grade, cleaned_text
                )
                treatments.append(treatment)
        
        return treatments
    
    @staticmethod
    def _is_known_tcm(name: str, known_tcm: set) -> bool:
        """判断是否为已知中药
        """
        # 检查是否在已知列表中
        if name in known_tcm:
            return True
        
        # 检查是否包含中药特征字
        tcm_markers = ['汤', '散', '丸', '胶囊', '软胶囊', '颗粒', '口服液', '饮']
        return any(marker in name for marker in tcm_markers)
    
    def _create_tcm_treatment(
        self,
        name: str,
        daily_dose: str,
        freq_unit: str,
        freq_desc: str,
        grade: str,
        context: str
    ) -> Dict:
        """创建中药治疗方案
        """
        dose_raw = f"每日{daily_dose}{freq_unit}，{freq_desc}"
        
        # 解析频次
        if freq_unit == '剂':
            frequency = f"{daily_dose}剂/日"
            # 提取分几次口服（如分早晚2次）
            split_times_match = re.search(r'分[^，。]+(\d+)次', freq_desc)
            if split_times_match:
                split_times = split_times_match.group(1).strip()
                frequency += f"（分{split_times}次口服）"
        else:
            # # 提取单次剂量（兼容5~10g、10mL、3粒等）
            # per_time_range_match = re.search(r'每次(\d+)-(\d+)\s*([gGmLml粒])', freq_desc)
            # if per_time_range_match:
            #     # 精准提取 5~10 + g → 拼接为 5~10g
            #     min_dose = per_time_range_match.group(1).strip()
            #     max_dose = per_time_range_match.group(2).strip()
            #     per_unit = per_time_range_match.group(3).strip()
            #     per_time = f"{min_dose}-{max_dose}{per_unit}"
            # else:
            #     per_time_single_match = re.search(r'每次(\d+\.?\d*)\s*([gGmLml粒])', freq_desc)
            #     if per_time_single_match:
            #         per_num = per_time_single_match.group(1).strip()
            #         per_unit = per_time_single_match.group(2).strip()
            #         per_time = f"{per_num}{per_unit}"  # ✅ "10g"
            #     else:
            #         per_time = ''
            per_time_match = re.search(r'每次([\d]+~[\d]+)\s*([gGmLml粒])', freq_desc)
            if not per_time_match:
                per_time_match = re.search(r'每次([\d~±\s]+[gGmLml粒])', freq_desc)
            
            if per_time_match:
                per_time = per_time_match.group(1).strip()
            else:
                per_time = ''
                
            frequency = f"{per_time}/次，{daily_dose}次/日" if per_time else f"{daily_dose}次/日"
            
        syndrome_treatments = self._extract_tcm_syndromes(context, context, set([name]))
        syndrome = ""
        method = ""
        for item in syndrome_treatments:
            if item['name'] == name:
                syndrome = item['syndrome']
                method = item['method']
                break
        
        return {
            'type': 'tcm',
            'name': name,
            'prescription': name,
            'dose_raw': dose_raw,
            'dose_parsed': {
                'recommended': frequency,
                'frequency': freq_desc
            },
            'recommendation_grade': grade,
            'evidence_level': grade[0] if grade else '',
            'phase': self._infer_phase(context),
            'syndrome': syndrome,  
            'method': method, 
            'context': context
        }
    
    def _create_drug_treatment_from_recommendation(
        self,
        name: str,
        daily_dose: str,
        freq_unit: str,
        freq_desc: str,
        grade: str,
        context: str
    ) -> Dict:
        """从推荐意见创建药物治疗方案
        """
        dose_raw = f"每日{daily_dose}{freq_unit}，{freq_desc}"
        
        return {
            'type': 'drug',
            'name': name,
            'dose_raw': dose_raw,
            'dose_parsed': self._parse_dose_from_text(dose_raw),
            'recommendation_grade': grade,
            'evidence_level': grade[0] if grade else '',
            'phase': self._infer_phase(context),
            'context': context
        }
    
    def _extract_drug_recommendations(self, text: str) -> List[Dict]:
        """提取西药推荐意见（带具体剂量mg）
        """
        treatments = []
        cleaned_text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
        cleaned_text = cleaned_text.replace('\\sim', '~')
        cleaned_text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', cleaned_text)
        cleaned_text = cleaned_text.replace('\\', '')
        
        cleaned_text = cleaned_text.replace('～', '-').replace('~', '-')
        cleaned_text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', cleaned_text)  # "5 - 10" → "5-10"
        
        def is_valid_drug_extraction(drug_name, context_before):
            """验证提取的药物名是否有效"""
            # 1. 基础验证
            if not drug_name or len(drug_name) < 2 or len(drug_name) > 30:
                return False
            
            # 2. 排除纯数字和剂量格式
            if re.match(r'^[\d\s~～\-,，.]+$', drug_name):
                return False
            
            # 3. 排除频次词和描述词
            invalid_words = [
                '每日', '每次', '每周', '每月', '每年', '每天', '每小时',
                '之后', '首次', '共注射', '个位点', '补充', '服用',
                '强度', '持续', '方面', '与', '显示', '发现', '研究', '试验'
            ]
            if any(word in drug_name for word in invalid_words):
                return False
            
            # 4. 检查前文上下文
            if context_before:
                research_triggers = [
                    '研究发现', '试验发现', '发现每', '显示每', '结果显示',
                    '对照研究', '小规模', '大规模', '另1项', '另一项',
                    '方面与', '持续时间'
                ]
                if any(trigger in context_before for trigger in research_triggers):
                    return False
            
            # 5. 必须包含药物特征
            drug_indicators = [
                '林', '芬', '坦', '胺', '酸', '醇', '素', '西', '普', '他',
                '洛', '替', '嗪', '丁', '喷', '利', '米', '托', '巴',
                '酚', '萘', '氯', '马', '佐', '那', '阿', '依', '夫',
                '坎', '拉', '麦', '角', '氟', '桂', '丙', '加', '莫',
                '美', '比', '肉', '镁', '核', '辅', '左'
            ]
            has_drug_char = any(char in drug_name for char in drug_indicators)
            has_category = self._infer_drug_category(drug_name) != ''
            
            if drug_name.startswith('的'):
                return False
            
            # 排除包含"治疗"、"发作"等动词的描述性文本
            if any(word in drug_name for word in ['治疗', '发作', '有效', '安全', '增加了', '减少了']):
                return False
            
            # 排除包含"都"、"且"等连词的长句
            if len(drug_name) > 10 and any(word in drug_name for word in ['都', '且', '但', '而']):
                return False
            
            if not (has_drug_char or has_category):
                return False
            
            return True
        
        # 匹配：药物名 + 剂量（含范围） + 单位 + 可选频次/给药方式/证据等级
        pattern1 = (
            # r'(?<![发现每天服用|发现每天补充|显示每天|结果显示每天|研究发现每天|试验发现每天])'
            r'([\u4e00-\u9fa5a-zA-Z]{2,20}?)\s{0,2}'
                #    r'(\d+\.?\d*\s*[~～-]\s*\d+\.?\d*|\d+\.?\d*)\s*'
            r'(\d+\.?\d*(?:\s*[~～-]\s*\d+\.?\d*)?)\s*'
            r'(mg|g|ml|mL|IU|U|μg|mcg|mmol|%)\s*'
            r'([，,；;]?\s*(?:每日|每次|每\d+[小时时日天]|/d|/日|/h|q\d+h?|'
            r'口服|皮下注射|静脉给药|直肠给药|肌内注射)*)?'
            r'[^。（]*?(?:[（(]\s*([IⅠⅡⅢⅣVVIXLCivⅰⅱⅲⅳⅴⅵⅶⅷⅸ\d]+[ABCDE]?)[）)])?')
        
        for match in re.finditer(pattern1, cleaned_text):
            start_pos = match.start()
            drug_name = match.group(1).strip()
            dose_num = match.group(2).strip().replace('～', '-').replace(' ', '')
            dose_unit = match.group(3).strip()
            dose_ext = match.group(4).strip() if match.group(4) else ''
            grade = match.group(5).strip() if match.group(5) else ''
            
            context_before = cleaned_text[max(0, start_pos-50):start_pos]
            if not is_valid_drug_extraction(drug_name, context_before):
                continue
            
            # research_triggers = [
            #     '研究发现', '试验发现', '发现每天', '发现每日', '显示每天', 
            #     '显示每日', '结果显示', '研究结果', '对照研究', '补充',
            #     '服用', '注射', '给药', '小规模', '大规模', '另1项', '另一项'
            # ]
            # if any(trigger in context_before for trigger in research_triggers):
            #     continue
            
            # invalid_names = [
            #     '每日', '每次', '每周', '每月', '每年', '每天', '每小时',
            #     '之后', '首次', '共注射', '个位点', '每日补充', '每天补充',
            #     '每天服用', '每日服用', '皮下注射首次'
            # ]
            # if drug_name in invalid_names or any(inv in drug_name for inv in ['推荐', '剂量', '治疗', '预防', '围经期', '研究', '试验', '发现']):
            #     continue
            
            # research_keywords = [
            #     '研究', '试验', '发现', '显示', '结果', '对照', 
            #     '安慰剂', '荟萃', '分析', '评价', '项', '小规模',
            #     '大规模', '另', '补充', '服用', '注射', '给药'
            # ]
            # if any(kw in drug_name for kw in research_keywords):
            #     continue
            
            # if drug_name in ['每日', '每次', '每周', '每月', '每年', '每天', '每小时']:
            #     continue
            
            # if any(word in drug_name for word in ['推荐', '剂量', '治疗', '预防', '短期', '围经期', '的推荐', '为']):
            #     continue
            
            # # 跳过百分比格式的非药物
            # if dose_unit == '%' and not self._is_likely_drug_name(drug_name):
            #     continue
            
            dose_raw = f"{dose_num} {dose_unit} {dose_ext}".strip()
            
            treatments.append({
                'type': 'drug',
                'name': drug_name,
                'dose_raw': dose_raw,
                'dose_parsed': self._parse_dose_from_text(dose_raw),
                'recommendation_grade': grade,
                'evidence_level': '',
                'phase': self._infer_phase(text),
                'context': text[:500]
            })
        
        # perimenstrual_pattern = r'围经期[^。]{0,30}?(?:推荐剂量|剂量)[为是]?\s*([\u4e00-\u9fa5a-zA-Z]{2,20})\s*(\d+\.?\d*)\s*(mg|g|ml)'
        perimenstrual_pattern = r'(?:围经期[^。]{0,30}?)?(?:推荐剂量|剂量)[为是：:]\s*'  # 前缀
        perimenstrual_pattern += r'([\u4e00-\u9fa5a-zA-Z]{2,15}?)\s{0,2}'  # ✅ 药物名（限制长度）
        perimenstrual_pattern += r'(\d+\.?\d*(?:\s*[~～-]\s*\d+\.?\d*)?)\s*'  # 剂量
        perimenstrual_pattern += r'(mg|g|ml|IU|μg)'  # 单位
        
        for match in re.finditer(perimenstrual_pattern, cleaned_text):
            drug_name = match.group(1).strip()
            dose_num = match.group(2).strip()
            dose_unit = match.group(3).strip()
            
            if not drug_name or len(drug_name) < 2:
                continue
            
            invalid_prefixes = ['围经期', '短期', '预防', '治疗', '的推荐', '推荐剂量为']
            should_skip = False
            for prefix in invalid_prefixes:
                if prefix in drug_name:
                    # 尝试移除前缀
                    drug_name = drug_name.replace(prefix, '').strip()
                    if len(drug_name) < 2:
                        should_skip = True
                        break
                    
            if should_skip or not self._is_likely_drug_name(drug_name):
                continue
            
            dose_raw = f"{dose_num} {dose_unit}"
            treatments.append({
                'type': 'drug',
                'name': drug_name,
                'dose_raw': dose_raw,
                'dose_parsed': self._parse_dose_from_text(dose_raw),
                'recommendation_grade': '',
                'evidence_level': '',
                'phase': '围经期预防',
                'context': text[:500]
            })
            
        return treatments
    
    def _extract_standard_dosage(self, text: str) -> List[Dict]:
        """提取标准剂量格式
        """
        treatments = []
        cleaned_text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
        cleaned_text = cleaned_text.replace('\\sim', '~')
        cleaned_text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', cleaned_text)
        cleaned_text = cleaned_text.replace('\\', '')
        
        # 匹配两种格式：
        # 1. 频次 + 药物名 + 剂量
        # 2. 药物名 + 剂量 + 频次
        pattern = (r'(?:(每日|每次|每\d+天)\s*(\d+)\s*(次|天|日))?\s*'
                   r'([\u4e00-\u9fa5a-zA-Z]{2,16})\s*'
                   r'(\d+\.?\d*\s*[~～-]\s*\d+\.?\d*|\d+\.?\d*)\s*'
                   r'(mg|g|ml|IU|mmol)|'
                   r'([\u4e00-\u9fa5a-zA-Z]{2,16})\s*'
                   r'(\d+\.?\d*\s*[~～-]\s*\d+\.?\d*|\d+\.?\d*)\s*'
                   r'(mg|g|ml|IU|mmol)\s*'
                   r'(每日|每次|每\d+天)\s*(\d+)\s*(次|天|日)')
        
        for match in re.finditer(pattern, cleaned_text):
            # 解析第一种格式
            if match.group(4):
                drug_name = match.group(4).strip()
                dose_num = match.group(5).strip().replace('～', '-').replace(' ', '')
                dose_unit = match.group(6).strip()
                freq_parts = [match.group(1), match.group(2), match.group(3)]
                freq = ' '.join([p for p in freq_parts if p]).strip()
            # 解析第二种格式
            else:
                drug_name = match.group(7).strip()
                dose_num = match.group(8).strip().replace('～', '-').replace(' ', '')
                dose_unit = match.group(9).strip()
                freq_parts = [match.group(10), match.group(11), match.group(12)]
                freq = ' '.join([p for p in freq_parts if p]).strip()
            
            # 验证是否为药物名
            if not self._is_likely_drug_name(drug_name):
                continue
            
            dose_raw = f"{dose_num} {dose_unit} {freq}".strip()
            
            treatments.append({
                'type': 'drug',
                'name': drug_name,
                'dose_raw': dose_raw,
                'dose_parsed': self._parse_dose_from_text(dose_raw),
                'recommendation_grade': '',
                'evidence_level': '',
                'phase': self._infer_phase(text),
                'context': text
            })
        
        return treatments
    
    def _extract_tcm_syndromes(self, text: str, cleaned_text: str, known_tcm: set) -> List[Dict]:
        """提取中医证型格式
        """
        treatments = []
        
        # 匹配证型和治法
        tcm_pattern = r'([^，。\s]{2,8}(?:证型|证|型))\s*[，。]?\s*(?:治法|治疗)?\s*[：:]?\s*([^。]{0,30})[。，]?'
        tcm_pattern = r'[（(]\d+[)）]\s*([\u4e00-\u9fa5]{2,8}证)\s*[，：:]\s*治法\s*[：:]\s*([^。]{0,30})'
        tcm_patterns = [
            r'([^，。\s]{2,8}(?:证型|证|型))\s*[，。]?\s*(?:治法|治疗)?\s*[：:]?\s*([^。]{0,30})[。，]?',
            r'[（(]\d+[)）]\s*([\u4e00-\u9fa5]{2,8}证)\s*[，：:]\s*治法\s*[：:]\s*([^。]{0,30})'
        ]
        
        # 构建中药名正则
        tcm_names_pattern = '|'.join(re.escape(name) for name in known_tcm)
        
        for match in re.finditer(tcm_pattern, text):
            syndrome = match.group(1).strip()
            method = match.group(2).strip()
            
            # 在后续文本中查找方剂名
            remaining_text = text[match.end():]
            
            remaining_clean = re.sub(r'(\d+~?\d*)\s+([gGmLml粒])', r'\1\2', remaining_text)
            remaining_clean = re.sub(r'[^\d~a-zA-Z\u4e00-\u9fa5]+', '', remaining_clean)
            remaining_clean = remaining_clean.replace('：', ':').replace('（', '(').replace('）', ')')
            
            prescription = self._find_prescription_in_text(
                remaining_clean, tcm_names_pattern
            )
            
            if prescription:
                treatments.append({
                    'type': 'tcm',
                    'name': prescription,
                    'syndrome': syndrome,
                    'method': method,
                    'prescription': prescription,
                    'dose_raw': '',
                    'recommendation_grade': '',
                    'evidence_level': '',
                    'phase': self._infer_phase(text),
                    'context': text
                })
        
        return treatments
    
    @staticmethod
    def _find_prescription_in_text(text: str, tcm_pattern: str) -> str:
        """在文本中查找方剂名
        """
        # 优先在推荐意见中查找
        rx_title_pattern = r'(?:推荐意见|推荐|建议)?\s*\d*\s*[:：]?\s*(' + tcm_pattern + r')[，,]'
        prescription_match = re.search(rx_title_pattern, text)
        
        if prescription_match:
            return prescription_match.group(1).strip()
        
        # 直接查找中药名
        prescription_match = re.search(r'(' + tcm_pattern + r')', text)
        return prescription_match.group(1).strip() if prescription_match else ''
    
    def _extract_acupuncture(self, cleaned_text: str) -> List[Dict]:
        """提取针灸推荐格式
        """
        treatments = []
        
        # 主要针灸模式
        acupuncture_pattern = r'''
            ([^。：\s]{2,8}(?:针|灸|疗法|压丸))  
            (?:\[\d+[,\d]*\])?                        
            (?:主要适用于([^。：]+?)型偏头痛)?        
            [。：]?\s*推荐(?:主穴|穴位|部位|耳穴)[:：]([^。]+?)  
            [。，]?\s*(?:推荐疗程及操作方法|操作方法|推荐疗程)[:：]?([^。]+)?  
        '''
        
        acu_pattern = re.compile(acupuncture_pattern, re.VERBOSE | re.DOTALL)
        
        for match in acu_pattern.finditer(cleaned_text):
            therapy_name = match.group(1).strip()
            syndrome = match.group(2).strip() if match.group(2) else ''
            acupoints = match.group(3).strip()
            operation = match.group(4).strip() if match.group(4) else ''
            
            treatment = self._create_acupuncture_treatment(
                therapy_name, syndrome, acupoints, operation, cleaned_text
            )
            treatments.append(treatment)
        
        # 间歇期针灸
        intermittent_treatment = self._extract_intermittent_acupuncture(cleaned_text)
        if intermittent_treatment:
            treatments.append(intermittent_treatment)
        
        return treatments
    
    def _create_acupuncture_treatment(
        self,
        therapy_name: str,
        syndrome: str,
        acupoints: str,
        operation: str,
        context: str
    ) -> Dict:
        """创建针灸治疗方案
        """
        # 拼接dose_raw
        if syndrome:
            dose_raw = f"适用证型：{syndrome}；推荐穴位：{acupoints}；操作/疗程：{operation}"
        else:
            dose_raw = f"推荐穴位：{acupoints}；操作/疗程：{operation}"
        
        return {
            'type': 'tcm',
            'name': therapy_name,
            'prescription': therapy_name,
            'dose_raw': dose_raw,
            'dose_parsed': {
                'frequency': operation,
                'acupoints': acupoints,
                'syndrome': syndrome
            },
            'recommendation_grade': '',
            'evidence_level': '',
            'phase': self._infer_phase(context),
            'context': context
        }
    
    def _extract_intermittent_acupuncture(self, cleaned_text: str) -> Optional[Dict]:
        """提取间歇期针灸
        """
        pattern = r'间歇期\s*([^。：\s]{2,8}(?:针|灸|疗法))(?:\[\d+[,\d]*\])?[:：]([^。]+?)推荐疗程及操作方法[:：]([^。]+?)'
        match = re.search(pattern, cleaned_text)
        
        if not match:
            return None
        
        therapy_name = f"{match.group(1)}（间歇期）"
        intermittent_points = match.group(2).strip()
        intermittent_operation = match.group(3).strip()
        
        dose_raw = f"间歇期{therapy_name}：推荐穴位：{intermittent_points}；疗程/操作：{intermittent_operation}"
        
        return {
            'type': 'tcm',
            'name': therapy_name,
            'prescription': therapy_name,
            'dose_raw': dose_raw,
            'dose_parsed': {
                'frequency': intermittent_operation,
                'acupoints': intermittent_points,
                'syndrome': ''
            },
            'recommendation_grade': '',
            'evidence_level': '',
            'phase': '间歇期',
            'context': cleaned_text
        }
    
    def _deduplicate_treatments(self, treatments: List[Dict]) -> List[Dict]:
        """去重治疗方案
        """
        seen = set()
        unique_treatments = []
        
        for t in treatments:
            # 根据治疗类型生成唯一键
            if t['type'] == 'drug':
                route = t['dose_parsed'].get('route', 'unknown')
                phase = t.get('phase', 'unknown')
                key = (
                    t['name'].lower(),
                    t.get('dose_raw', ''),
                    route,
                    phase
                )
            else:  # tcm 或 physical
                base_name = t.get('prescription', t.get('name', '')).lower()
                syndrome = t.get('syndrome', '')
                
                # 针灸的额外键
                if '针' in base_name or '灸' in base_name:
                    acupoints = t.get('dose_parsed', {}).get('acupoints', '')
                    key = (base_name, syndrome, acupoints)
                else:
                    key = (base_name, syndrome, '')
            
            # 验证键的有效性
            if key not in seen and any(str(k).strip() for k in key):
                seen.add(key)
                unique_treatments.append(t)
        
        return unique_treatments
    
    def _parse_dose_from_text(self, dose_text: str) -> Dict:
        """解析剂量文本 - 支持小数和多种单位
        """
        dose_info = {
            'recommended': '',
            'max_daily': '',
            'route': 'unknown',
            'frequency': '',
            'route_cn': '',
            'dose_range': False,
            'applicable_population': ''
        }
        
        if not dose_text:
            return dose_info
        
        # 预处理
        clean_text = dose_text.lower()
        
       # 步骤1：清理LaTeX特殊字符
        clean_text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', clean_text)
        clean_text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', clean_text)
        clean_text = clean_text.replace('\\', '')
        
        # 步骤2：统一波浪号为短横线（保留范围标记）
        clean_text = clean_text.replace('～', '-').replace('~', '-')
        
        # 步骤3：智能空格处理
        # 保留数字间的空格（如 "5 - 10" → "5-10"）
        # 删除其他位置的空格
        clean_text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', clean_text)  # "5 - 10" → "5-10"
        clean_text = re.sub(r'(\d)\s+([a-zμ%])', r'\1\2', clean_text)   # "10 g" → "10g"
        # 删除其他多余空格（但不影响已处理的范围）
        clean_text = re.sub(r'(?<=[^\d-])\s+(?=[^\d-])', '', clean_text)
        
        # 提取推荐剂量
        dose_info['recommended'] = self._extract_recommended_dose(clean_text)
        dose_info['dose_range'] = '-' in dose_info['recommended'] or '~' in dose_info['recommended']
        
        # 提取最大剂量
        dose_info['max_daily'] = self._extract_max_daily_dose(clean_text)
        
        # 提取给药途径
        route, route_cn = self._extract_route_from_text(clean_text)
        dose_info['route'] = route
        dose_info['route_cn'] = route_cn
        
        # 提取频次
        frequency = self._extract_frequency_from_text(clean_text)
        # 移除频次中的口服/route相关描述
        for route_word in ['口服', '冲服', '外用', '注射']:
            frequency = frequency.replace(route_word, '').strip().rstrip(',').rstrip('，')
        dose_info['frequency'] = frequency
        
        # 提取适用人群
        dose_info['applicable_population'] = self._extract_population(clean_text)
        
        return dose_info
    
    @staticmethod
    def _extract_recommended_dose(text: str) -> str:
        """提取推荐剂量
        """
        # 优先匹配范围剂量（数字-数字+单位）
        range_pattern = r'(\d+\.?\d*-\d+\.?\d*\s*(?:mg|片|g|ml|mL|μg|mcg|IU|U|mmol|%|剂)(?:/[日d天次h])?)'
        range_match = re.search(range_pattern, text)
        if range_match:
            dose = range_match.group(1).strip()
            # 标准化单位表示
            dose = dose.replace('/d', '/日').replace('/天', '/日').replace('/h', '/小时')
            return dose
        
        # 其次匹配单一剂量（数字+单位）
        single_pattern = r'(\d+\.?\d*\s*(?:mg|片|g|ml|mL|μg|mcg|IU|U|mmol|%|剂)(?:/[日d天次h])?)'
        single_match = re.search(single_pattern, text)
        if single_match:
            dose = single_match.group(1).strip()
            dose = dose.replace('/d', '/日').replace('/天', '/日').replace('/h', '/小时')
            return dose
        
        return ''
    
    @staticmethod
    def _extract_max_daily_dose(text: str) -> str:
        """提取最大剂量
        """
        max_pattern = r'(?:每日)?最大剂量[：:]?\s*(\d+\.?\d*\s*(?:mg|片|g|ml|mL|μg|mcg|IU|U|mmol|%))'
        match = re.search(max_pattern, text)
        return match.group(1).strip() if match else ''
    
    def _extract_route_from_text(self, text: str) -> Tuple[str, str]:
        """从文本提取给药途径
        
        Returns:
            Tuple[str, str]: (英文途径, 中文途径)
        """
        route_mapping = [
            ('肌注|肌肉注射|肌内注射', 'intramuscular', '肌内注射'),
            ('静脉滴注|静滴|注射液', 'intravenous', '静脉滴注'),
            ('静脉注射|iv|静脉推注|静脉', 'intravenous', '静脉注射'),
            ('静脉给药', 'intravenous', '静脉给药'),
            ('鼻喷|鼻腔|鼻喷剂', 'nasal', '鼻喷'),
            ('直肠|肛栓|肛门栓剂|直肠给药', 'rectal', '直肠给药'),
            ('皮下|皮下注射', 'subcutaneous', '皮下注射'),
            ('口崩|崩解|口崩片', 'oral_disintegrating', '口崩片'),
            ('口服|口服剂', 'oral', '口服'),
        ]
        
        for pattern, en_route, cn_route in route_mapping:
            if re.search(pattern, text):
                return en_route, cn_route
        
        return 'oral', '口服'  # 默认值
    
    def _extract_frequency_from_text(self, text: str) -> str:
        """从文本提取频次
        """
        text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        text = text.replace('\\', '')
        
        # 特殊频次（优先级高）
        special_freq = {
            '按需': '按需服用',
            '隔日': '隔日1次',
            '/月': '每月1次',
            '/季度': '每季度1次',
            '每季度': '每季度1次',
            '每30天': '每30天',
        }
        
        for pattern, freq in special_freq.items():
            if pattern in text:
                return freq
        
        # 复杂频次模式
        complex_patterns = [
            (r'每次\s*(\d+)', lambda m: f"每次{m.group(1)}"),
            (r'每(\d+)小时\s*(\d+)次', lambda m: f"每{m.group(1)}小时{m.group(2)}次"),
            (r'[qQ]\s*(\d+)[hH]', lambda m: f"每{m.group(1)}小时1次"),
            (r'每日\s*(\d+)\s*次', lambda m: f"每日{m.group(1)}次"),
            (r'每(\d+)天\s*(\d+)次', lambda m: f"每{m.group(1)}天{m.group(2)}次"),
            (r'月经前(\d+)天开始', lambda m: f"月经前{m.group(1)}天开始给药"),
        ]
        
        for pattern, formatter in complex_patterns:
            match = re.search(pattern, text)
            if match:
                return formatter(match)
        
        # 简单频次
        if '/h' in text:
            return '每小时1次'
        elif '分早晚' in text or '早晚' in text:
            return '每日2次(早晚)'
        elif 'bid' in text.lower():
            return '每日2次'
        elif 'tid' in text.lower():
            return '每日3次'
        elif 'qd' in text.lower() or '每日' in text:
            return '每日1次'
        
        return ''
    
    @staticmethod
    def _extract_population(text: str) -> str:
        """提取适用人群
        """
        # 儿童人群
        if re.search(r'6岁以上|体重\d+kg以上|儿童', text):
            age_match = re.search(r'(\d+)岁以上', text)
            weight_match = re.search(r'体重(\d+)kg以上', text)
            
            if age_match:
                return f"{age_match.group(1)}岁以上儿童"
            elif weight_match:
                return f"体重{weight_match.group(1)}kg以上儿童"
            else:
                return '儿童'
        
        # 特殊人群
        special_populations = {
            '成人': '成人',
            '妊娠期': '妊娠期',
            '孕期': '妊娠期',
            '哺乳期': '哺乳期',
            '泌乳期': '哺乳期',
            '喂奶期': '哺乳期',
        }
        
        for keyword, population in special_populations.items():
            if keyword in text:
                return population
        
        return ''
    
    def _infer_phase(self, text: str) -> str:
        """推断治疗阶段
        """
        clean_text = text.lower()
        
        if '围经期' in clean_text or '月经' in clean_text and '预防' in clean_text:
            return TreatmentPhase.PERIMENSTRUAL.value
        
        # 优先匹配特定阶段
        if '预防' in clean_text or '预防性' in clean_text:
            if '围经期' in clean_text:
                return TreatmentPhase.PERIMENSTRUAL.value
            elif '间歇期' in clean_text:
                return '间歇期预防'
            elif '妊娠期' in clean_text:
                return '妊娠期预防'
            elif '哺乳期' in clean_text:
                return '哺乳期预防'
            else:
                return TreatmentPhase.PREVENTION.value
        
        # 急性期
        if '急性' in clean_text or '发作期' in clean_text or '发作' in clean_text:
            return TreatmentPhase.ACUTE.value
        
        # 其他特殊阶段
        special_phases = {
            '围经期': TreatmentPhase.PERIMENSTRUAL.value,
            '间歇期': TreatmentPhase.INTERMITTENT.value,
            '妊娠期': TreatmentPhase.PREGNANCY.value,
            '哺乳期': TreatmentPhase.LACTATION.value,
        }
        
        for keyword, phase in special_phases.items():
            if keyword in clean_text:
                return phase
        
        return TreatmentPhase.UNKNOWN.value
    
    def _create_es_document(self, treatment: Dict, raw_text: str, pdf_id: Optional[str]) -> Optional[Dict]:
        """创建ES文档
        """
        try:
            treatment_type = treatment['type']
            phase = treatment['phase']
            
            # 创建基础文档
            base_doc = self._create_base_document(treatment, phase, pdf_id, raw_text)
            
            # 根据治疗类型添加intervention
            if treatment_type == 'drug':
                base_doc['intervention'] = self._create_drug_intervention(treatment)
            elif treatment_type == 'tcm':
                base_doc['intervention'] = self._create_tcm_intervention(treatment)
            elif treatment_type == 'physical':
                base_doc['intervention'] = self._create_physical_intervention(treatment)
            else:
                print(f"警告：未知治疗类型 {treatment_type}")
                return None
            
            return base_doc
            
        except Exception as e:
            print(f"创建ES文档时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_base_document(
        self,
        treatment: Dict,
        phase: str,
        pdf_id: Optional[str],
        raw_text: str
    ) -> Dict:
        """创建基础ES文档结构
        """
        
        # clean_raw_text = raw_text.replace('\\sim', '~').replace('\\mathrm{~', '').replace('}', '').replace('\\', '')
        clean_raw_text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', raw_text)  # \mathrm{mg} -> mg
        clean_raw_text = clean_raw_text.replace('\\sim', '~')  # \sim -> ~
        clean_raw_text = re.sub(r'\\[a-zA-Z]+\{[^\}~]*\}', '', clean_raw_text)  # 其他LaTeX命令，保留~
        clean_raw_text = clean_raw_text.replace('\\', '')  # 剩余的反斜杠
        clean_raw_text = re.sub(r'(?<=[^\d~])\s+(?=[^\d~])', ' ', clean_raw_text)  # 标准化非数字/~的空格
        clean_raw_text = re.sub(r'(?<=[\d~])\s+(?=[\d~])', '', clean_raw_text)  # 合并数字/~间的空格（5~ 10→5~10）
        
        evidence_str = ""
        evidence_match = re.search(r'[()（）](\d+[A-Z]+|I+[A-Z]+|II+[A-Z]+)[()（）]', clean_raw_text.upper())
        if evidence_match:
            evidence_str = evidence_match.group(1)
            
        rec_grade, evi_level = self._split_combined_grade(evidence_str)
         
        # evidence_level = self._normalize_evidence_level(treatment.get('evidence_level', ''))
        # recommendation_grade = self._normalize_recommendation_grade(treatment.get('recommendation_grade', ''))
        evidence_level = self._normalize_evidence_level(evi_level, from_combined=True) if evi_level else self._normalize_evidence_level(treatment.get('evidence_level', ''))
        recommendation_grade = self._normalize_recommendation_grade(rec_grade, from_combined=True) if rec_grade else self._normalize_recommendation_grade(treatment.get('recommendation_grade', ''))
        
        return {
            "rule_id": str(uuid.uuid4()),
            "disease": "偏头痛",
            "phase": phase,
            "treatment_type": treatment['type'],
            "population": self.DEFAULT_POPULATION,
            "language": self.DEFAULT_LANGUAGE,
            "applicability": {
                "age_range": self.DEFAULT_AGE_RANGE,
                "pregnancy_lactation": "无数据",
                "comorbidities_excluded": []
            },
            "evidence": {
                "evidence_level": evidence_level,
                "recommendation_grade": recommendation_grade,
                # "evidence_source": "专家共识",
                # "study_count": None,
                "consistency": "有证据等级或推荐强度" if (evidence_level or recommendation_grade) else ""
            },
            "safety": {
                "adverse_effects": {
                    "common": "",
                    "rare": "",
                    "severity": ""
                },
                "contraindications": [],
                "precautions": []
            },
            "source": {
                "guideline": self.source_guideline or '',
                "version": f"{self.guideline_year}版" if self.guideline_year else "",
                "year": self.guideline_year,
                "section": "",
                "table_title": "",
                "pdf_id": pdf_id or "",
                "page": None,
                "source_type": "text"
            },
            "traceability": {
                "table_row_index": None,
                "footnote_refs": [],
                "raw_text_segment": treatment.get('context', clean_raw_text)[:500]
            },
            "extraction_method": "regex"
        }
    
    def _create_drug_intervention(self, treatment: Dict) -> Dict:
        """创建药物干预信息
        """
        dose_parsed = treatment.get('dose_parsed', {})
        recommended_text = dose_parsed.get('recommended', '') or treatment.get('dose_raw', '')
        dose_obj = self._parse_dose_to_structured(recommended_text, '')
        
        max_daily_text = dose_parsed.get('max_daily', '')
        max_dose_obj = self._parse_dose_to_structured(max_daily_text) if max_daily_text else None
        
        route = dose_parsed.get('route', 'oral')
        frequency = dose_parsed.get('frequency', '')
        
        # 提取约束条件
        constraints = self._extract_constraints_from_context(treatment.get('context', ''))
        
        # 构建给药途径描述
        route_cn = self._route_to_chinese(route) or "口服"
        route_desc = f"{frequency} {route_cn}".strip() if frequency else route_cn
        
        # 提取间隔小时
        interval_hours = self._extract_interval_hours(treatment.get('context', ''))
        
        return {
            "name": treatment['name'],
            "category": self._infer_drug_category(treatment['name']),
            "sub_category": "",
            "route": {
                "primary": route,
                "description": route_desc
            },
            "frequency": frequency,
            "course": "",
            "constraints": constraints,
            "modality_specific": {
                "drug_specific": {
                    "dose": {
                        "recommended": dose_obj,
                        "max_daily": max_dose_obj,
                        "adjustment": {}
                    },
                    "dosing_logic": {
                        "interval_min_hours": interval_hours
                    }
                }
            }
        }
        
    def _create_tcm_intervention(self, treatment: Dict) -> Dict:
        """创建中医干预信息
        """
        prescription = treatment.get('prescription', '') or "未指定方剂"
        if '汤' in prescription or '胶囊' in prescription or '颗粒' in prescription or '口服液' in prescription:
            route_desc = "口服"
        else:
            route_desc = "待确认"
            
        method = treatment.get('method', '')
        if method and 5 < len(method) < 50:  # 治法长度应在5-50之间
            route_desc = f"治法：{method}；{route_desc}"
        
        syndrome = treatment.get('syndrome', '')
        valid_syndrome = ''
        
        if syndrome:
            # 必须包含"证"或"型"
            if '证' in syndrome or '型' in syndrome:
                # 长度合理（2-20字符）
                if 2 <= len(syndrome) <= 20:
                    # ✅ 新增：排除明显不是证型的文本
                    invalid_keywords = ['研究', 'RCT', '临床', '治疗', '证实', '疗效', '来证']
                    if not any(kw in syndrome for kw in invalid_keywords):
                        valid_syndrome = syndrome
        
        return {
            "name": prescription,
            "category": "中医方剂",
            "sub_category": "",
            "route": {
                "primary": "oral",
                "description": route_desc
            },
            "frequency": treatment.get('dose_raw', '分早晚2次'),
            "course": "",
            "constraints": [],
            "modality_specific": {
                "tcm_specific": {
                    "tcm_pattern": valid_syndrome,
                    "symptoms": [],
                    "acupoints": {
                        "primary": [],
                        "secondary": [],
                        "method": ""
                    },
                    "herbal_adjunct": prescription
                }
            }
        }
    
    def _create_physical_intervention(self, treatment: Dict) -> Dict:
        """创建物理治疗干预信息
        
        优化点：提取为独立方法（预留接口）
        """
        return {
            "name": treatment['name'],
            "category": "物理疗法",
            "sub_category": "",
            "route": {
                "primary": "external",
                "description": "物理治疗"
            },
            "frequency": "",
            "course": "",
            "constraints": [],
            "modality_specific": {
                "physical_specific": {
                    "target_site": [],
                    "device": treatment['name'],
                    "parameters": {},
                    "duration_per_session": ""
                }
            }
        }
    
    def _extract_constraints_from_context(self, context_text: str) -> List[str]:
        """从上下文提取约束条件
        
        优化点：
        1. 提取为独立方法
        2. 使用配置化的约束模式
        3. 改进正则表达式
        """
        constraints = []
        
        if not context_text:
            return constraints
        
        # 约束模式配置
        constraint_patterns = [
            # 24小时约束
            {
                'pattern': r'24\s*[小时h]*\s*[内]?[不]*超过\s*(\d+\.?\d*)\s*(mg|片|次|IU)',
                'template': "24小时内不超过{}{}"
            },
            # 30天约束
            {
                'pattern': r'(?:30天|每月)[^。]{0,50}?不?超过\s*(\d+)\s*次',
                'template': "每30天使用不超过{}次",
                'extra_check': lambda text: '超过' in text,
                'extra_template': "每30天使用不建议超过{}次（超过则安全性未建立）"
            },
            # 间隔约束
            {
                'pattern': r'至少间隔\s*(\d+)\s*小时',
                'template': "首剂后至少间隔{}小时可加服"
            },
            # 总剂量约束
            {
                'pattern': r'总剂量[不能]*超过\s*(\d+\.?\d*)\s*(mg|IU)',
                'template': "总剂量不超过{}{}"
            }
        ]
        
        for config in constraint_patterns:
            match = re.search(config['pattern'], context_text)
            if match:
                if 'extra_check' in config and config['extra_check'](context_text):
                    constraints.append(config['extra_template'].format(*match.groups()))
                else:
                    constraints.append(config['template'].format(*match.groups()))
        
        return constraints
    
    def _parse_dose_to_structured(self, dose_text, fallback_unit=''):
        """将剂量文本解析为结构化对象 - 支持小数和空格
            支持格式：
        - 单一剂量: 400, 400mg, 400mg/d
        - 范围剂量: 50-100, 50-100mg, 50-100mg/d
        """
        if not dose_text:
            return None

        clean_text = str(dose_text).strip()
        # 统一分隔符
        clean_text = clean_text.replace('～', '-').replace('—', '-').replace('~', '-')
        
        if 'IU' in clean_text:
            clean_text = re.sub(r'\s+', '', clean_text)
            
        compound_unit_pattern = r'([a-zA-Zμ%]+/[a-zA-Z日次时周月年kg体重]+)'
        compound_match = re.search(compound_unit_pattern, clean_text)
        
        unit = ''
        numeric_part = clean_text
        
        if compound_match:
            # 提取到完整的复合单位
            unit = compound_match.group(1)
            numeric_part = clean_text.replace(unit, '').strip()
        else:
            # 兜底：尝试提取简单单位（旧逻辑）
            simple_units = ['mg', 'g', 'ml', 'mL', 'μg', 'mcg', 'IU', 'U', 'mmol', '%', '片']
            for u in simple_units:
                if u in clean_text:
                    unit = u
                    numeric_part = clean_text.replace(u, '').strip()
                    break
            
        if not unit and fallback_unit:
            unit = fallback_unit
        
        # 修改：去除数字中的空格（如"1 000" -> "1000"）
        numeric_part = re.sub(r'(\d)\s+(\d)', r'\1\2', numeric_part)
        
        # ✅ 修改：解析范围剂量
        if '-' in numeric_part:
            parts = numeric_part.split('-')
            if len(parts) == 2:
                try:
                    min_val = float(parts[0].strip())
                    max_val = float(parts[1].strip())
                    # 整数转换
                    min_val = int(min_val) if min_val == int(min_val) else min_val
                    max_val = int(max_val) if max_val == int(max_val) else max_val
                    return {
                        "text_original": dose_text.strip(),
                        "min": min_val,
                        "max": max_val,
                        "unit": unit,  # ✅ 这里确保返回了unit
                        "type": "range"
                    }
                except ValueError:
                    pass
        
        # ✅ 修改：解析单一剂量
        try:
            val = float(numeric_part.strip())
            val = int(val) if val == int(val) else val
            return {
                "text_original": dose_text.strip(),
                "value": val,
                "unit": unit,  
                "type": "single"
            }
        except ValueError:
            pass
        
        # 无法解析
        return {
            "text_original": dose_text.strip(),
            "value": None,
            "unit": unit if unit else None,
            "min": None,
            "max": None,
            "type": "unparsed" 
        }
    
    @staticmethod
    def _normalize_numeric(value: float) -> int | float:
        """标准化数值（整数去掉小数点）
        """
        return int(value) if value == int(value) else value
    
    def _route_to_chinese(self, route: str) -> str:
        """将英文给药途径转为中文
        """
        route_map = {
            # 口服类
            'oral': '口服',
            'oral_disintegrating': '口腔崩解片',
            'oral_tablet': '口服片剂',
            
            # 注射类
            'intramuscular': '肌肉注射',
            'im': '肌肉注射',
            'subcutaneous': '皮下注射',
            'sc': '皮下注射',
            'intravenous': '静脉给药',
            'iv': '静脉给药',
            'intravenous_drip': '静脉滴注',
            'ivdrip': '静脉滴注',
            'intravenous_push': '静脉推注',
            'ivpush': '静脉推注',
            
            # 其他途径
            'nasal': '鼻喷',
            'nasal_spray': '鼻喷剂',
            'rectal': '直肠给药',
            'rectal_suppository': '直肠栓',
            'topical': '外用',
        }
        
        if not route or not isinstance(route, str):
            return '未知途径'
        
        return route_map.get(route.lower(), '未知途径')
    
    def _infer_drug_category(self, drug_name: str) -> str:
        """从药物名称推断分类 
        """
        # 药物分类关键词映射
        category_keywords = {
            '非甾体抗炎药': [
                '布洛芬', '阿司匹林', '双氯芬酸', '萘普生', '酮洛芬', 
                '吲哚美辛', '塞来昔布', '安乃近', '安替比林', '托芬那酸',
                '阿司匹林赖氨酸盐', '赖安匹林'
            ],
            '对乙酰氨基酚类': ['对乙酰氨基酚', '扑热息痛'],
            '曲普坦类': [
                '舒马曲普坦', '舒马普坦', '佐米曲普坦', '利扎曲普坦', 
                '利扎曲坦', '那拉曲普坦', '那拉曲坦', '阿莫曲坦', 
                '依来曲坦', '夫罗曲坦'
            ],
            '麦角胺类': ['麦角胺', '双氢麦角胺', '酒石酸麦角胺', '麦角胺咖啡因'],
            '地坦类': ['拉米地坦'],
            '吉泮类': [
                '来那吉泮', '瑞美吉泮', 'rimegepant', '乌布吉泮', 
                '扎维吉泮', '阿托吉泮', 'atogepant', 'Telcagepant (MK0974)'
            ],
            '钙离子拮抗剂': ['氟桂利嗪', '盐酸氟桂利嗪', '普瑞巴林'],
            '抗癫痫药': [
                '丙戊酸钠', '丙戊酸', '托吡酯', '加巴喷丁', '拉莫三嗪',
                '双丙戊酸钠', '丙戊酸盐', '丙戊酸镁', '左乙拉西坦'
            ],
            'β受体阻滞剂': ['美托洛尔', '普萘洛尔', '比索洛尔', '阿替洛尔'],
            '抗抑郁药': ['阿米替林', '文拉法辛', '去甲替林', '替扎尼定'],
            'CGRP单抗': [
                '依瑞奈尤单抗', 'Erenumab', '瑞玛奈珠单抗', 'Fremanezumab',
                '加卡奈珠单抗', 'Galcanezumab', '艾普奈珠单抗', 'Eptinezumab'
            ],
            '肉毒毒素类': ['肉毒素', '肉毒毒素', 'A型肉毒毒素', 'A型肉毒毒', '肉毒杆菌'],
            '血管紧张素受体拮抗剂(ARB)': ['坎地沙坦'],
            '血管紧张素转换酶抑制剂(ACEI)': ['赖诺普利'],
            '阿片类': [
                '吗啡', '可待因', '羟考酮', '洛芬待因', 
                '氨酚待因', '氨酚氢可酮', '布托啡诺'
            ],
            '止吐药': ['甲氧氯普胺', '多潘立酮', '氯丙嗪', '异丙嗪'],
            '复合制剂': [
                '阿司匹林/咖啡因', '去痛片', '脑清片', '头痛粉',
                '酚咖片', '氨酚巴妥片', '新复方大青叶片', '散利痛'
            ],
            '镁剂类': ['镁剂', '柠檬酸镁', '苹果酸镁', '甘氨酸镁', '氧化镁', '硫酸镁'],
            '其他西药': ['苯二氮䓬类', '巴比妥类镇静剂', '核黄素', '辅酶 Q10'],
            '中医方剂': [
                '天舒胶囊', '都梁软胶囊', '川芎茶调散', '散偏汤', '正天丸',
                '通天口服液', '川芎清脑颗粒', '天麻钩藤饮', '养血清脑颗粒',
                '丹珍头痛胶囊', '头痛宁胶囊', '通窍活血汤', '血府逐瘀汤', '通心络胶囊'
            ]
        }
        
        if not drug_name or not isinstance(drug_name, str):
            return ""
        
        # 精确匹配优先
        for category, keywords in category_keywords.items():
            if drug_name in keywords:
                return category
        
        # 模糊匹配
        for category, keywords in category_keywords.items():
            if any(keyword in drug_name for keyword in keywords):
                return category
        
        return ""
    
    def _extract_interval_hours(self, text: str) -> Optional[int | float]:
        """提取最小用药间隔
        """
        if not text:
            return None
        
        # 多种间隔表达模式
        patterns = [
            r'间隔\s*(\d+\.?\d*)\s*小?时?[hH]?',
            r'(\d+\.?\d*)\s*小?时?[hH]?.*?间隔',
            r'至少间隔\s*(\d+\.?\d*)\s*小?时?[hH]?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                val = float(match.group(1))
                # 验证合理性（0.5-72小时）
                if 0.5 <= val <= 72:
                    return self._normalize_numeric(val)
        
        return None
    
    def _split_combined_grade(self, text: str) -> tuple[str, str]:
        """拆分组合格式的推荐等级（如1B, 2A, IB等）
        
        Returns:
            (recommendation_grade, evidence_level) 元组
        """
        if not text:
            return '', ''
        
        cleaned = text.strip().upper()
        cleaned = re.sub(r'[()（）\s]', '', cleaned)
        
        # 全角罗马数字转半角
        roman_map = {'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV'}
        for full, half in roman_map.items():
            cleaned = cleaned.replace(full, half)
        
        # 匹配组合格式：数字/罗马数字 + 字母
        combo_match = re.match(r'^([1-2I]+)([A-D])$', cleaned)
        if combo_match:
            number = combo_match.group(1)  # 推荐强度部分
            letter = combo_match.group(2)  # 证据级别部分
            
            # 转换推荐强度
            if number == 'I' or number == '1':
                rec_grade = '1'  # 强推荐
            elif number == 'II' or number == '2':
                rec_grade = '2'  # 弱推荐
            else:
                rec_grade = number
                
            return rec_grade, letter
        
        return '', ''

    def _normalize_evidence_level(self, level_text: str, from_combined: bool = False) -> str:
        """规范化证据等级
        
        Args:
            level_text: 证据等级文本
            from_combined: 是否来自组合格式拆分（如果是，level_text已经是单个字母）
        """
        if not level_text:
            return ''
        
        cleaned = level_text.strip()
        
        # 如果是从组合格式拆分出来的，直接映射
        if from_combined:
            level_mapping = {
                'A': '证据级别A',
                'B': '证据级别B',
                'C': '证据级别C',
                'D': '证据级别D',
            }
            return level_mapping.get(cleaned, '')
        
        # 移除多余字符
        cleaned = re.sub(r'[()（）\s]', '', cleaned)
        
        # 全角罗马数字转半角
        roman_map = {'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV'}
        for full, half in roman_map.items():
            cleaned = cleaned.replace(full, half)
        
        # 检查是否是组合格式，如果是则返回空（应该用_split_combined_grade处理）
        combo_match = re.match(r'^([1-2I]+)([A-D])$', cleaned)
        if combo_match:
            return ''  # 组合格式应该在外部先拆分
            
        evidence_mapping = {
            # 字母格式
            'A': '证据级别A',
            'B': '证据级别B', 
            'C': '证据级别C',
            'D': '证据级别D',
            # 中文格式
            '高': '证据级别高',
            '中': '证据级别中',
            '低': '证据级别低',
            '极低': '证据级别极低',
            # 带括号的格式（已去除括号）
            '高A': '证据级别A',
            '中B': '证据级别B',
            '低C': '证据级别C',
            '极低D': '证据级别D',
        }
        
        if cleaned in evidence_mapping:
            return evidence_mapping[cleaned]
        
        if '证据' in cleaned:
            return cleaned
        
        # 排除推荐强度相关的文本
        if cleaned in ['1', '2', 'I', 'II', '强', '弱'] or '推荐' in cleaned or '强度' in cleaned:
            return ''
        
        return ''

    def _normalize_recommendation_grade(self, grade_text: str, from_combined: bool = False) -> str:
        """规范化推荐等级
        
        Args:
            grade_text: 推荐等级文本
            from_combined: 是否来自组合格式拆分（如果是，grade_text已经是单个数字）
        """
        if not grade_text:
            return ''
        
        cleaned = grade_text.strip().upper()
        
        # 如果是从组合格式拆分出来的，直接映射
        if from_combined:
            grade_mapping = {
                '1': '推荐强度1',  # 强推荐
                '2': '推荐强度2',  # 弱推荐
            }
            return grade_mapping.get(cleaned, '')
        
        cleaned = re.sub(r'[()（）\s]', '', cleaned)
        
        # 全角罗马数字转半角
        roman_map = {'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV'}
        for full, half in roman_map.items():
            cleaned = cleaned.replace(full, half)
        
        # 检查是否是组合格式，如果是则返回空（应该用_split_combined_grade处理）
        combo_match = re.match(r'^([1-2I]+)([A-D])$', cleaned)
        if combo_match:
            return ''  # 组合格式应该在外部先拆分
            
        grade_mapping = {
            # 罗马数字
            'I': '推荐强度1',   # 强推荐
            'II': '推荐强度2',  # 弱推荐
            # 阿拉伯数字
            '1': '推荐强度1',
            '2': '推荐强度2',
            # 中文格式
            '强': '推荐强度强',
            '弱': '推荐强度弱',
            '强推荐': '强推荐',
            '弱推荐': '弱推荐',
        }
        
        if cleaned in grade_mapping:
            return grade_mapping[cleaned]
            
        if '推荐' in cleaned or '强度' in cleaned:
            return cleaned
        
        # 排除证据级别相关的文本
        if cleaned in ['A', 'B', 'C', 'D'] or '证据' in cleaned:
            return ''
        
        return cleaned if cleaned else grade_text

    
    def _is_likely_drug_name(self, name: str) -> bool:
        """判断是否可能是药物名称
        """
        if not name or not isinstance(name, str):
            return False
        
        name_stripped = name.strip()
        if len(name_stripped) < 2 or len(name_stripped) > 40:
            return False
        
        if name_stripped in ['每日', '每次', '每周', '每月', '每年', '每天', '每小时']:
            return False
        
        # ✅ 新增：排除包含特定描述性词组的文本
        invalid_phrases = [
            '推荐剂量为', '的推荐剂量', '围经期短期', '预防治疗',
            '治疗的推荐', '推荐的', '剂量为'
        ]
        if any(phrase in name_stripped for phrase in invalid_phrases):
            return False
        
        # 排除词汇
        non_drug_words = [
            '推荐', '治疗', '急性', '预防', '患者', '临床', '研究',
            '证据', '指南', '方法', '效果', '作用', '不良', '禁忌',
            '注意', '说明', '剂量', '用法', '频次', '疗程', '每日',
            '口服', '静脉', '肌注', '鼻喷', '皮下', '直肠', '意见',
            '抑制剂', '激动剂', '拮抗剂', '受体', '类', '药物', '制剂',
            '解热镇痛药', '非甾体类抗炎药', '复合制剂', '阿片类', 
            '止呕剂', '止吐药', 'COX-2', 'NSAIDs', 'ACEI/ARB', 
            '5-羟色胺', 'CGRP', 'SSRIs', 'SNRI', 'NaSSA',
            '钙离子拮抗剂', '抗癫痫药物', '抗惊厥药', 'β受体阻滞剂', '抗抑郁药',
            '增加', '减少', '降低', '升高', '改善', '缓解', '头痛', '频率',
            '比例', '患者', '受试者', '对照组', '安慰剂', '发作', '严重程度'
        ]
        
        if any(non_drug in name for non_drug in non_drug_words):
            return False
        
        if not re.search(r'[\u4e00-\u9fa5a-zA-Z]', name):
            return False
        
        verb_prefixes = ['增加', '减少', '降低', '升高', '改善', '缓解', '提高', '下降']
        if any(name.startswith(verb) for verb in verb_prefixes):
            return False
        
        # 药物常见字符
        drug_indicators = [
            '林', '芬', '坦', '胺', '酸', '醇', '素', '西', '普', '他',
            '洛', '替', '嗪', '丁', '喷', '利', '米', '托', '巴',
            '酚', '洛芬', '萘', '氯', '马', '佐', '那', '阿',
            '依', '夫', '坎', '拉', '麦', '角', '氟', '桂', '丙',
            '加', '莫', '美', '比', '肉', '镁', '核', '辅', '左', 
            '柠', '苹', '甘', '氧', '硫', '布', '妥', '酶', '肽', 
            '抗体', '单抗'
        ]
        
        if any(indicator in name for indicator in drug_indicators):
            return True
        
        # 特殊药物名单（略，已在_infer_drug_category中定义）
        # 可以复用_infer_drug_category的逻辑
        category = self._infer_drug_category(name)
        if category:
            return True
        
        # 英文药物名
        if len(name) >= 5 and re.match(r'^[A-Za-z]+$', name):
            if name.endswith(('mab', 'pant', 'gepant')) or name in ['Erenumab', 'Fremanezumab']:
                return True
        
        return False


# ===== 主程序 =====
if __name__ == "__main__":
    # 修改为你的JSON文件路径
    json_file_path = "/home/Downloads/中国偏头痛中西医结合防治指南2022年.json"
    
    extractor = MedicalGuidelineExtractor(json_file_path)
    
    if extractor.extract_data():
        print(f"\n{'='*60}")
        print(f"指南名称: {extractor.source_guideline}")
        print(f"指南年份: {extractor.guideline_year}")
        print(f"找到 {len(extractor.tables)} 个表格")
        print(f"找到 {len(extractor.text_blocks)} 个文本块")
        print(f"{'='*60}\n")
        
        if len(extractor.tables) == 0 and len(extractor.text_blocks) == 0:
            print("\n⚠️  严重警告：没有提取到任何数据！")
            print("\n诊断建议：")
            print("1. 检查JSON文件结构是否符合预期")
            print("2. 使用以下代码查看JSON文件的前1000个字符：")
            print(f"   with open('{json_file_path}', 'r') as f: print(f.read(1000))")
            print("3. 检查JSON中是否有 'type': 'table' 或文本内容")
        
        # 解析表格
        print("开始解析表格...")
        table_docs = extractor.parse_tables_to_elasticsearch()
        print(f"\n✅ 从表格生成 {len(table_docs)} 条文档")
        
        # 打印表格摘要
        for i, table in enumerate(extractor.tables):
            print(f"\n表格{i+1}:")
            print(f"  标题: {table['caption']}")
            print(f"  是否有HTML: {'是' if table['html'] else '否'}")
            print(f"  是否为列表格式: {'是' if table.get('is_list_format', False) else '否'}")
            print(f"  脚注数量: {len(table['footnotes'])}")
            if table['caption']:
                print(f"  标题预览: {table['caption'][:50]}...")
        
        # 解析正文
        print("\n开始解析正文...")
        text_docs = extractor.parse_text_blocks_to_elasticsearch()
        print(f"\n✅ 从正文生成 {len(text_docs)} 条文档")
        
        # 合并所有文档
        all_docs = table_docs + text_docs
        print(f"\n{'='*60}")
        print(f"总共生成 {len(all_docs)} 条ES文档")
        print(f"{'='*60}\n")
        
        # 展示示例
        if table_docs:
            print("=== 表格提取示例（第1条）===")
            print(json.dumps(table_docs[0], ensure_ascii=False, indent=2))
            print()
        
        if text_docs:
            print("\n=== 正文提取示例（第1条）===")
            print(json.dumps(text_docs[0], ensure_ascii=False, indent=2))
            print()
            
        input_filename = os.path.basename(json_file_path)  # 获取文件名
        input_name = os.path.splitext(input_filename)[0]   # 去掉.json扩展名
        output_file = f'{input_name}_es_docs.json'         # 生成新文件名
        
        
        # output_file = 'all_elasticsearch_docs.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ 已保存到 {output_file}")
        print(f"{'='*60}")
        
        # 统计信息
        print(f"\n=== 统计信息 ===")
        print(f"表格文档: {len(table_docs)} 条")
        print(f"正文文档: {len(text_docs)} 条")
        
        # 按治疗类型统计
        drug_count = sum(1 for doc in all_docs if doc['treatment_type'] == 'drug')
        tcm_count = sum(1 for doc in all_docs if doc['treatment_type'] == 'tcm')
        physical_count = sum(1 for doc in all_docs if doc['treatment_type'] == 'physical')
        
        print(f"\n按治疗类型:")
        print(f"  - 药物治疗: {drug_count} 条")
        print(f"  - 中医治疗: {tcm_count} 条")
        print(f"  - 物理治疗: {physical_count} 条")
        
        # 按阶段统计
        acute_count = sum(1 for doc in all_docs if doc['phase'] == '急性期')
        prevention_count = sum(1 for doc in all_docs if doc['phase'] == '预防期')
        
        print(f"\n按治疗阶段:")
        print(f"  - 急性期: {acute_count} 条")
        print(f"  - 预防期: {prevention_count} 条")
        
        print(f"\n{'='*60}")
        print("处理完成！")
        print(f"{'='*60}\n")
    else:
        print("❌ 数据提取失败，请检查JSON文件路径和格式")