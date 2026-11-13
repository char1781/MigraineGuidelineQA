"""
app.py - Streamlit前端入口
职责：纯UI，不写业务逻辑，所有检索和Agent从src/模块import
运行：streamlit run app.py
Demo模式：在.env中设置 DEMO_MODE=true，不调用任何API
"""

import streamlit as st
from datetime import datetime
import time
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# ═══════════════════════════════════════════════
# 统一的指南名/年份清洗工具
# ═══════════════════════════════════════════════
_UUID_PREFIX_RE = re.compile(r'^[0-9a-f]{24,40}[\s_\-]+', re.IGNORECASE)
_YEAR_TAIL_RE = re.compile(r'[\s]*[（(]?\s*(?:19|20)\d{2}\s*年?\s*[)）]?\s*$')
_YEAR_IN_NAME_RE = re.compile(r'[（(]?\s*((?:19|20)\d{2})\s*年?\s*[)）]?')


def _clean_display_name(name) -> str:
    """显示用：去掉 uuid 前缀、尾部的 (xxxx年) 等噪声。"""
    if not name:
        return "未知指南"
    s = str(name).strip()
    s = _UUID_PREFIX_RE.sub('', s).strip()
    s = _YEAR_TAIL_RE.sub('', s).strip()
    return s or "未知指南"


def _format_year(year, fallback="年份未知") -> str:
    """显示用：None/0/1111 显示为'年份未知'，其余显示 '2024版'。"""
    if year in (None, 0, 1111, "", "0", "1111"):
        return fallback
    return f"{year}版"


# ───────────────────────────────────────────────
# 页面基础配置
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="偏头痛指南智能问答",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────
# 全局CSS
# ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg-deep:#f5f5f5; --bg-card:#ffffff; --bg-hover:#eeeeee; --border:#dddddd;
    --accent-blue:#388bfd; --accent-teal:#3fb950; --accent-amber:#d29922; --accent-red:#f85149;
    --text-main:#000000; --text-sub:#333333; --text-muted:#666666;
    --user-bubble:#e6f0ff; --ai-bubble:#f0f0f0;
    --tag-es:#e6f0ff; --tag-hybrid:#e6ffe6; --tag-graph:#f5e6ff; --tag-web:#fff0e6;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-main);
    font-family: 'Noto Serif SC', serif;
    font-size: 16px !important;
}
section[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border);
    padding: 20px 16px !important;
    font-size: 16px !important;
}
[data-testid="stHeader"] { display: none; }
.block-container { padding-top: 1rem !important; }

.top-bar { display:flex; align-items:center; gap:14px; padding:18px 24px;
    border-bottom:1px solid var(--border); background:var(--bg-card); border-radius:12px; margin-bottom:20px; }
.top-bar-icon { font-size:2rem; }
.top-bar-title { font-size:1.8rem !important; font-weight:700; color:var(--text-main); letter-spacing:0.02em; }
.top-bar-sub { font-size:1rem !important; color:var(--text-sub); font-family:'JetBrains Mono',monospace; margin-top:2px; }
.demo-badge { margin-left:auto; background:var(--accent-amber); color:#000; padding:4px 12px;
    border-radius:20px; font-size:0.75rem; font-weight:700; font-family:'JetBrains Mono',monospace; letter-spacing:0.05em; }

.msg-wrap { display:flex; gap:12px; margin:14px 0; animation:fadeUp 0.25s ease-out; }
.msg-wrap.user { flex-direction:row-reverse; }
@keyframes fadeUp { from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:translateY(0);} }
.msg-avatar { width:36px; height:36px; border-radius:50%; display:flex; align-items:center;
    justify-content:center; font-size:1rem; flex-shrink:0; margin-top:4px; }
.msg-avatar.user-av { background:var(--accent-blue); }
.msg-avatar.ai-av { background:var(--bg-hover); border:1px solid var(--border); }
.msg-body { max-width:76%; }
.msg-bubble { padding:13px 17px; border-radius:12px; line-height:1.7; font-size:1.1rem !important; word-break:break-word; }
.msg-bubble.user { background:var(--user-bubble); border:1px solid #264878; border-top-right-radius:3px; text-align:right; }
.msg-bubble.ai { background:var(--ai-bubble); border:1px solid var(--border); border-top-left-radius:3px; }
.tool-tags { margin-bottom:7px; display:flex; flex-wrap:wrap; gap:6px; }
.tool-tag { padding:3px 10px; border-radius:20px; font-size:0.85rem !important;
    font-family:'JetBrains Mono',monospace; font-weight:500; letter-spacing:0.04em; border:1px solid transparent; }
.tool-tag.es { background:var(--tag-es); color:#79c0ff; border-color:#1a5c8a; }
.tool-tag.hybrid { background:var(--tag-hybrid); color:#56d364; border-color:#1a5c3a; }
.tool-tag.graph { background:var(--tag-graph); color:#d2a8ff; border-color:#5a2d8a; }
.tool-tag.web { background:var(--tag-web); color:#ffa657; border-color:#7a4400; }
.msg-time { font-size:0.85rem !important; color:var(--text-muted); margin-top:5px;
    font-family:'JetBrains Mono',monospace; padding:0 4px; }
.msg-wrap.user .msg-time { text-align:right; }
.source-block { margin-top:10px; padding:10px 14px; background:var(--bg-deep);
    border-left:3px solid var(--accent-blue); border-radius:0 8px 8px 0; font-size:0.9rem !important;
    color:var(--text-sub); font-family:'JetBrains Mono',monospace; }
.source-block .src-title { color:var(--accent-blue); font-weight:600; margin-bottom:4px; }

.input-area { position:sticky; bottom:0; background:var(--bg-deep); padding:14px 0 8px; border-top:1px solid var(--border); }
[data-testid="stTextInput"] { height:60px !important; }
[data-testid="stTextInput"] > div { height:60px !important; margin:0 !important; }
[data-testid="stTextInput"] input {
    background:var(--bg-card) !important; border:1px solid var(--border) !important;
    border-radius:10px !important; color:var(--text-main) !important;
    font-family:'Noto Serif SC', serif !important; font-size:1.2rem !important;
    padding:0 16px !important; height:60px !important; min-height:60px !important;
    max-height:60px !important; line-height:60px !important; box-sizing:border-box !important;
    transition:border-color 0.2s;
}
[data-testid="stTextInput"] input::placeholder { line-height:60px !important; height:60px !important; }
[data-testid="stButton"] { height:60px !important; }
[data-testid="stButton"] > div { height:60px !important; margin:0 !important; }
[data-testid="stButton"] button {
    background:var(--accent-blue) !important; color:#fff !important; border:none !important;
    border-radius:10px !important; font-family:'JetBrains Mono',monospace !important;
    font-size:1.5rem !important; font-weight:600 !important; padding:0 !important;
    height:60px !important; min-height:60px !important; max-height:60px !important;
    line-height:60px !important; transition:opacity 0.2s, transform 0.15s !important;
    letter-spacing:0.03em; display:flex !important; align-items:center !important;
    justify-content:center !important;
}
.row-widget.stButton { margin-top:0 !important; padding-top:0 !important; }

.sidebar-section { margin:16px 0; padding:14px; background:var(--bg-deep);
    border:1px solid var(--border); border-radius:10px; }
.sidebar-section-title { font-size:1.1rem !important; font-family:'JetBrains Mono',monospace;
    color:var(--text-sub); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px; }

.kb-item { display:flex; align-items:center; justify-content:space-between; padding:8px 10px;
    border-radius:8px; margin:5px 0; background:var(--bg-hover); border:1px solid var(--border); font-size:0.8rem; }
.kb-item-name { color:var(--text-main); font-size:1.1rem !important; }
.kb-item-meta { color:var(--text-muted); font-size:0.95rem !important; font-family:'JetBrains Mono',monospace; }
.kb-status-ok { width:7px; height:7px; border-radius:50%; background:var(--accent-teal); flex-shrink:0; }

.stat-row { display:flex; gap:10px; margin:8px 0; }
.stat-box { flex:1; background:var(--bg-hover); border:1px solid var(--border); border-radius:8px; padding:10px 8px; text-align:center; }
.stat-num { font-size:2.2rem !important; font-weight:700; color:var(--accent-blue); }
.stat-lbl { font-size:1rem !important; color:var(--text-muted); font-family:'JetBrains Mono',monospace; margin-top:2px; }

.tool-status { display:flex; align-items:center; gap:8px; padding:8px 12px;
    border-radius:6px; margin:6px 0; font-size:1.1rem !important; background:var(--bg-hover); }
.dot-on { width:6px; height:6px; border-radius:50%; background:var(--accent-teal); }
.dot-off { width:6px; height:6px; border-radius:50%; background:var(--text-muted); }

[data-testid="stFileUploader"] {
    background:var(--bg-card) !important; border:1.5px dashed var(--border) !important;
    border-radius:10px !important; padding:10px !important;
}
[data-testid="stFileUploader"]:hover { border-color:var(--accent-blue) !important; }
[data-testid="stExpander"] { background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:8px !important; }
[data-testid="stProgress"] > div > div { background:var(--accent-blue) !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg-deep); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:4px; }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────
# Session State 初始化
# ───────────────────────────────────────────────
def _init_state():
    kb_list = []
    if not DEMO_MODE:
        try:
            from src.utils.kb_metadata import KnowledgeBaseMetadata
            kb_meta = KnowledgeBaseMetadata()
            kb_list = kb_meta.get_all()
        except Exception:
            kb_list = [
                {"name": "中国偏头痛诊治指南2022版", "year": 2022, "status": "ok"},
                {"name": "中国偏头痛急性期治疗指南2024", "year": 2024, "status": "ok"},
            ]
    else:
        kb_list = [
            {"name": "中国偏头痛诊治指南2022版", "year": 2022, "status": "ok"},
            {"name": "中国偏头痛急性期治疗指南2024", "year": 2024, "status": "ok"},
        ]

    defaults = {
        "messages": [],
        "thread_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "total_queries": 0,
        "es_queries": 0,
        "hybrid_queries": 0,
        "graph_queries": 0,
        "web_searches": 0,
        "kb_list": kb_list,
        "pipeline_running": False,
        "pipeline_log": [],
        "search_mode": "auto",
        "example_query": None,
        # ★ 修改点2：路由状态 / flash
        "current_page": "chat",
        "flash": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _refresh_kb_list():
    """从持久化元数据刷新知识库列表"""
    try:
        from src.utils.kb_metadata import KnowledgeBaseMetadata
        st.session_state.kb_list = KnowledgeBaseMetadata().get_all()
    except Exception:
        pass


# ───────────────────────────────────────────────
# 审核数量统计
# ───────────────────────────────────────────────
def _count_pending_reviews() -> int:
    if DEMO_MODE:
        return 1
    try:
        from src.ingestion.es_review_editor import ESReviewEditor
        editor = ESReviewEditor()
        reviews = editor.list_reviews()
        return len([r for r in reviews if r["review_status"] == "pending"])
    except Exception:
        return 0


# ═══════════════════════════════════════════════
# 审核页 —— 显示前清洗名字/年份
# ═══════════════════════════════════════════════
def _render_review_page():
    st.markdown('<div class="top-bar"><div class="top-bar-icon">📝</div><div><div class="top-bar-title">ES提取结果审核中心</div><div class="top-bar-sub">确认/修改/拒绝每条提取记录</div></div></div>', unsafe_allow_html=True)

    if st.button("← 返回聊天界面", key="btn_back_chat"):
        st.session_state.current_page = "chat"
        st.session_state.pop("current_review_id", None)
        st.rerun()

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if DEMO_MODE:
        _render_review_page_demo()
        return

    try:
        _refresh_kb_list()
        from src.ingestion.es_review_editor import ESReviewEditor
        editor = ESReviewEditor()
        reviews = editor.list_reviews()

        if not reviews:
            st.info("📭 暂无待审核指南，请先上传PDF；若刚上传失败，说明原始 PDF 没有抽取到有效药物剂量表。")
            return

        st.markdown("### 待审核指南列表")
        for review_meta in reviews:
            # ★ 清洗显示
            disp_name = _clean_display_name(review_meta.get("guideline_name"))
            disp_year = review_meta.get("guideline_year")
            year_txt = f"{disp_year}年" if disp_year not in (None, 0, 1111, "", "0", "1111") else "年份未知"
            with st.expander(
                f"📖 {disp_name} ({year_txt}) "
                f"- {review_meta['extracted_count']}条记录 "
                f"- {review_meta['review_status']}"
            ):
                if st.button("开始审核", key=f"start_{review_meta['review_id']}"):
                    st.session_state.current_review_id = review_meta['review_id']
                    st.rerun()

        if "current_review_id" in st.session_state:
            _render_review_detail(editor, st.session_state.current_review_id)

    except ImportError:
        st.error("审核模块未找到，请检查 src/ingestion/es_review_editor.py")


# def _render_review_page_demo():
#     st.markdown("### 📋 中国偏头痛诊治指南2022版 - 提取结果审核")
#     st.info("⚡ Demo模式：这是模拟数据，不调用真实API")

#     demo_records = [
#         {"id": "rec_001", "drug_name": "利扎曲普坦", "phase": "急性期",
#          "dose_recommended": "5-10mg", "dose_max_daily": "30mg/日",
#          "evidence_level": "A", "recommendation_grade": "1", "source_table": "表3"},
#         {"id": "rec_002", "drug_name": "佐米曲普坦", "phase": "急性期",
#          "dose_recommended": "2.5-5mg", "dose_max_daily": "10mg/日",
#          "evidence_level": "A", "recommendation_grade": "1", "source_table": "表3"},
#     ]

#     for rec in demo_records:
#         with st.container():
#             st.markdown(f"""
#             <div style="background:var(--bg-card); padding:14px; border-radius:10px; border:1px solid var(--border); margin:10px 0;">
#                 <div style="display:flex; justify-content:space-between; align-items:center;">
#                     <div>
#                         <div style="font-size:1rem; font-weight:600; color:var(--text-main);">💊 {rec['drug_name']}</div>
#                         <div style="font-size:0.8rem; color:var(--text-sub); margin-top:4px;">{rec['phase']} | {rec['source_table']}</div>
#                     </div>
#                     <div style="font-size:0.75rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace;">
#                         {rec['evidence_level']} / {rec['recommendation_grade']}
#                     </div>
#                 </div>
#                 <div style="margin-top:10px; padding-top:10px; border-top:1px solid var(--border);">
#                     <div style="font-size:0.85rem; color:var(--text-sub);">
#                         推荐剂量: <span style="color:var(--accent-blue); font-weight:600;">{rec['dose_recommended']}</span>
#                         &nbsp;|&nbsp; 最大剂量: <span style="color:var(--accent-red);">{rec['dose_max_daily']}</span>
#                     </div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 if st.button("✅ 批准", key=f"approve_{rec['id']}", use_container_width=True):
#                     st.success(f"已批准 {rec['drug_name']}")
#             with col2:
#                 if st.button("✏️ 修改", key=f"edit_{rec['id']}", use_container_width=True):
#                     st.info("Demo模式：修改功能仅演示")
#             with col3:
#                 if st.button("❌ 拒绝", key=f"reject_{rec['id']}", use_container_width=True):
#                     st.warning(f"已拒绝 {rec['drug_name']}")

#     st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

#     col_finish, col_cancel = st.columns(2)
#     with col_finish:
#         if st.button("🎉 完成审核，写入ES", key="finish_review_demo", use_container_width=True, type="primary"):
#             st.success("✅ Demo模式：审核已完成（模拟）")
#     with col_cancel:
#         if st.button("🚫 取消", key="cancel_review_demo", use_container_width=True):
#             st.session_state.current_page = "chat"
#             st.rerun()


# def _render_review_detail(editor, review_id: str):
#     review_data = editor.load_reviewed_data(review_id)

#     if review_data.get("extracted_count", 0) <= 0 or not review_data.get("records"):
#         st.warning("⚠️ 该指南没有抽取到有效的药物剂量记录，不能进入审核流程。请检查 PDF 是否含药物表格，或修正 MinerU/ES 提取器。")
#         if st.button("← 返回审核列表", key=f"back_review_list_{review_id}"):
#             st.session_state.pop("current_review_id", None)
#             st.rerun()
#         return

#     # ★ 清洗显示
#     disp_name = _clean_display_name(review_data.get('guideline_name'))
#     st.markdown(f"### 📋 {disp_name} - 详细审核")
#     st.markdown(f"提取了 {review_data['extracted_count']} 条记录，请逐条确认")

#     approved = len([r for r in review_data["records"] if r["status"] == "approved"])
#     modified = len([r for r in review_data["records"] if r["status"] == "modified"])
#     rejected = len([r for r in review_data["records"] if r["status"] == "rejected"])
#     pending = len([r for r in review_data["records"] if r["status"] == "pending"])
#     total = max(1, review_data["extracted_count"])

#     st.progress(
#         (approved + modified + rejected) / total,
#         text=f"进度: ✅{approved} | ✏️{modified} | ❌{rejected} | ⏳{pending}"
#     )
#     st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

#     for rec in review_data["records"]:
#         status_badge = {"pending": "⏳", "approved": "✅", "modified": "✏️", "rejected": "❌"}[rec["status"]]

#         with st.expander(f"{status_badge} {rec['drug_name']} - {rec['phase']}", expanded=(rec["status"] == "pending")):
#             col_info1, col_info2 = st.columns(2)
#             with col_info1:
#                 st.markdown(f"**药物名称:** {rec['drug_name']}")
#                 st.markdown(f"**治疗阶段:** {rec['phase']}")
#                 st.markdown(f"**分类:** {rec.get('category', 'N/A')}")
#             with col_info2:
#                 st.markdown(f"**证据等级:** {rec['evidence_level']}")
#                 st.markdown(f"**推荐强度:** {rec['recommendation_grade']}")
#                 st.markdown(f"**来源:** {rec['source_table']} 第{rec['source_row']}行")

#             st.markdown("---")

#             col_edit1, col_edit2 = st.columns(2)
#             with col_edit1:
#                 new_name = st.text_input("药物名称", value=rec['drug_name'], key=f"name_{rec['id']}")
#                 new_freq = st.text_input("给药频次", value=rec['frequency'], key=f"freq_{rec['id']}")
#             with col_edit2:
#                 dose_info = rec.get('dose_info', {})
#                 new_dose_rec = st.text_input("推荐剂量", value=dose_info.get('recommended', ''), key=f"dose_rec_{rec['id']}")
#                 new_dose_max = st.text_input("最大剂量", value=dose_info.get('max_daily', ''), key=f"dose_max_{rec['id']}")

#             new_adverse = st.text_area("不良反应", value=rec.get('adverse_effects', ''), key=f"adverse_{rec['id']}")
#             st.caption(f"原始文本片段: {rec.get('raw_text', '')[:200]}...")

#             col_btn1, col_btn2, col_btn3 = st.columns(3)
#             with col_btn1:
#                 if st.button("✅ 批准（无修改）", key=f"approve_{rec['id']}", use_container_width=True):
#                     editor.approve_record(review_data, rec['id'])
#                     editor.save_reviewed_data(review_data, review_id)
#                     st.rerun()
#             with col_btn2:
#                 if st.button("💾 保存修改", key=f"save_{rec['id']}", use_container_width=True):
#                     modifications = {}
#                     if new_name != rec['drug_name']:
#                         modifications['drug_name'] = new_name

#                     if new_phase != rec['phase']: modifications['phase'] = new_phase
#                     if new_unit != dose_info.get('unit',''): modifications['unit'] = new_unit


#                     if new_freq != rec['frequency']:
#                         modifications['frequency'] = new_freq

#                     if new_dose_rec or new_dose_max:
#                         modifications['dose_info'] = {'recommended': new_dose_rec, 'max_daily': new_dose_max,
#                                                         'unit': new_unit,}  # ← 关键：把单位也带上

#                     if new_evidence != rec.get('evidence_level',''): modifications['evidence_level'] = new_evidence
#                     if new_grade != rec.get('recommendation_grade',''): modifications['recommendation_grade'] = new_grade
                        
#                     if new_adverse != rec.get('adverse_effects', ''):
#                         modifications['adverse_effects'] = new_adverse
#                     editor.apply_modifications(review_data, rec['id'], modifications)
#                     editor.save_reviewed_data(review_data, review_id)
#                     st.rerun()
#             with col_btn3:
#                 if st.button("❌ 拒绝", key=f"reject_{rec['id']}", use_container_width=True):
#                     editor.reject_record(review_data, rec['id'], "用户在页面直接拒绝")
#                     # reason = st.text_input("拒绝理由", key=f"reason_{rec['id']}")
#                     # editor.reject_record(review_data, rec['id'], reason)
#                     editor.save_reviewed_data(review_data, review_id)
#                     st.rerun()

#     st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

def _render_review_detail(editor, review_id: str):
    review_data = editor.load_reviewed_data(review_id)

    if review_data.get("extracted_count", 0) <= 0 or not review_data.get("records"):
        st.warning("⚠️ 该指南没有抽取到有效的药物剂量记录，不能进入审核流程。请检查 PDF 是否含药物表格，或修正 MinerU/ES 提取器。")
        if st.button("← 返回审核列表", key=f"back_review_list_{review_id}"):
            st.session_state.pop("current_review_id", None)
            st.rerun()
        return

    guideline_name = _clean_display_name(review_data.get('guideline_name'))
    st.markdown(f"### 📋 {guideline_name} - 详细审核")
    st.markdown(f"提取了 {review_data['extracted_count']} 条记录，请逐条确认")

    approved = len([r for r in review_data["records"] if r["status"] == "approved"])
    modified = len([r for r in review_data["records"] if r["status"] == "modified"])
    rejected = len([r for r in review_data["records"] if r["status"] == "rejected"])
    pending = len([r for r in review_data["records"] if r["status"] == "pending"])
    total = max(1, review_data["extracted_count"])

    st.progress(
        (approved + modified + rejected) / total,
        text=f"进度: ✅{approved} | ✏️{modified} | ❌{rejected} | ⏳{pending}"
    )
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    PHASE_OPTIONS = ["急性期", "预防期", "慢性偏头痛", "围经期", "间歇期", "妊娠期", "哺乳期", "未知"]

    for rec in review_data["records"]:
        status_badge = {"pending": "⏳", "approved": "✅", "modified": "✏️", "rejected": "❌"}[rec["status"]]

        # ★ 核心修改：只有成功提取到阶段时，才拼接在标题上
        title_phase = f" - {rec['phase']}" if rec.get('phase') else ""
        
        with st.expander(
            f"{status_badge} {rec['drug_name']}{title_phase}",
            expanded=(rec["status"] == "pending"),
        ):
            dose_info = rec.get('dose_info', {}) or {}
            if not isinstance(dose_info, dict):
                dose_info = {}

            # 禁忌症：兼容列表/字符串
            contraindications = rec.get('contraindications', [])
            if isinstance(contraindications, (list, tuple)):
                contra_text = "、".join([str(x) for x in contraindications if x])
            else:
                contra_text = str(contraindications or "")

            # 指南来源：取原始文档 → 清洗
            raw_source = (rec.get('_original_doc', {}) or {}).get('source', {}).get('guideline') or guideline_name
            display_source = _clean_display_name(raw_source)

            # ── 第1排：药物名称 / 推荐剂量 / 剂量单位 ──
            col_row1_1, col_row1_2, col_row1_3 = st.columns(3)
            with col_row1_1:
                new_name = st.text_input("药物名称", value=rec['drug_name'], key=f"name_{rec['id']}")
            with col_row1_2:
                new_dose_rec = st.text_input("推荐剂量", value=dose_info.get('recommended', ''), key=f"dose_rec_{rec['id']}")
            with col_row1_3:
                new_unit = st.text_input("剂量单位", value=dose_info.get('unit', ''), key=f"unit_{rec['id']}")

            # 增加行间距
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

            # ── 第2排：治疗阶段 / 最大剂量 / 证据等级 ──
            col_row2_1, col_row2_2, col_row2_3 = st.columns(3)
            with col_row2_1:
                if rec.get('phase') in PHASE_OPTIONS:
                    phase_index = PHASE_OPTIONS.index(rec['phase'])
                else:
                    phase_index = 0  # 默认急性期
                new_phase = st.selectbox("治疗阶段", options=PHASE_OPTIONS, index=phase_index, key=f"phase_{rec['id']}")
            with col_row2_2:
                new_dose_max = st.text_input("最大剂量", value=dose_info.get('max_daily', ''), key=f"dose_max_{rec['id']}")
            with col_row2_3:
                new_evidence = st.text_input("证据等级", value=rec.get('evidence_level', ''), key=f"ev_{rec['id']}")

            # 增加行间距
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

            # ── 第3排：给药频次 / 推荐强度 / 指南来源 ──
            col_row3_1, col_row3_2, col_row3_3 = st.columns(3)
            with col_row3_1:
                new_freq = st.text_input("给药频次", value=rec.get('frequency', ''), key=f"freq_{rec['id']}")
            with col_row3_2:
                new_grade = st.text_input("推荐强度", value=rec.get('recommendation_grade', ''), key=f"gr_{rec['id']}")
            with col_row3_3:
                # 用 text_area 来容纳长文本，设置 height=68 使其与旁边的单行输入框视觉平齐
                new_source_guideline = st.text_area(
                    "指南来源", 
                    value=display_source, 
                    key=f"src_gl_{rec['id']}", 
                    height=68
                )

            # 增加行间距
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

            # ── 文本域区（独占整行） ──
            new_adverse = st.text_area("不良反应", value=rec.get('adverse_effects', ''), key=f"adverse_{rec['id']}", height=80)
            
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            
            new_contra = st.text_area("禁忌症", value=contra_text, key=f"contra_{rec['id']}", height=80)

            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            st.caption(f"原始文本片段: {rec.get('raw_text', '')[:200]}...")

            # ── 操作按钮 ──
            col_btn1, col_btn2, col_btn3 = st.columns(3)

            with col_btn1:
                if st.button("✅ 批准（无修改）", key=f"approve_{rec['id']}", use_container_width=True):
                    editor.approve_record(review_data, rec['id'])
                    editor.save_reviewed_data(review_data, review_id)
                    st.rerun()

            with col_btn2:
                if st.button("💾 保存修改", key=f"save_{rec['id']}", use_container_width=True):
                    modifications = {}

                    if new_name != rec['drug_name']:
                        modifications['drug_name'] = new_name
                    if new_phase != rec.get('phase'):
                        modifications['phase'] = new_phase
                    if new_freq != rec.get('frequency'):
                        modifications['frequency'] = new_freq
                    if new_evidence != rec.get('evidence_level'):
                        modifications['evidence_level'] = new_evidence
                    if new_grade != rec.get('recommendation_grade'):
                        modifications['recommendation_grade'] = new_grade
                    if new_adverse != rec.get('adverse_effects'):
                        modifications['adverse_effects'] = new_adverse

                    # 禁忌症：字符串 → 列表
                    new_contra_list = [x.strip() for x in re.split(r'[、,，;；\n]+', new_contra) if x.strip()]
                    old_contra_list = contraindications if isinstance(contraindications, list) else []
                    if new_contra_list != old_contra_list:
                        modifications['contraindications'] = new_contra_list

                    # 指南来源写回
                    if new_source_guideline != display_source:
                        modifications['source_guideline'] = new_source_guideline

                    # 剂量信息打包
                    new_dose_pack = {
                        'recommended': new_dose_rec,
                        'max_daily': new_dose_max,
                        'unit': new_unit,
                    }
                    old_dose_pack = {
                        'recommended': dose_info.get('recommended', ''),
                        'max_daily': dose_info.get('max_daily', ''),
                        'unit': dose_info.get('unit', ''),
                    }
                    if new_dose_pack != old_dose_pack:
                        modifications['dose_info'] = new_dose_pack

                    if modifications:
                        editor.apply_modifications(review_data, rec['id'], modifications)
                        editor.save_reviewed_data(review_data, review_id)
                        st.rerun()
                    else:
                        st.info("没有检测到修改内容")

            with col_btn3:
                if st.button("❌ 拒绝", key=f"reject_{rec['id']}", use_container_width=True):
                    editor.reject_record(review_data, rec['id'], "用户在页面直接拒绝")
                    editor.save_reviewed_data(review_data, review_id)
                    st.rerun()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if pending == 0:
        st.success("🎉 所有记录已审核完成！")
        col_final1, col_final2 = st.columns(2)
        with col_final1:
            if st.button("✅ 确认完成，写入ES数据库", key="finish_review", use_container_width=True, type="primary"):
                with st.spinner("正在写入ES..."):
                    try:
                        from pipeline import run_pipeline_after_review
                        result = run_pipeline_after_review(review_id)
                        review_data["review_status"] = "completed"
                        editor.save_reviewed_data(review_data, review_id)

                        st.session_state.current_page = "chat"
                        st.session_state.pop("current_review_id", None)
                        st.session_state.flash = {
                            "level": "success",
                            "msg": f"✅ 已写入 {result.get('es_indexed', 0)} 条文档到ES",
                        }
                        st.rerun()
                    except Exception as e:
                        st.session_state.flash = {"level": "error", "msg": f"写入ES失败: {e}"}
                        st.rerun()
        with col_final2:
            if st.button("🚫 取消", key="cancel_review", use_container_width=True):
                st.session_state.pop("current_review_id", None)
                st.session_state.current_page = "chat"
                st.rerun()


# ───────────────────────────────────────────────
# 懒加载真实模块
# ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_agent():
    if DEMO_MODE:
        return None
    try:
        from src.agent.agent_executor import build_agent
        return build_agent()
    except Exception as e:
        st.error(f"Agent初始化失败：{e}")
        return None

@st.cache_resource(show_spinner=False)
def _load_router():
    if DEMO_MODE:
        return None
    try:
        from src.retrieval.router import detect_query_type
        return detect_query_type
    except Exception:
        return None


# ───────────────────────────────────────────────
# Mock数据
# ───────────────────────────────────────────────
def _mock_answer(query: str) -> dict:
    time.sleep(0.4)
    answers = {
        "剂量": {"tool": "es", "answer": "根据《中国偏头痛诊治指南2022版》，**利扎曲普坦**推荐剂量为 **5mg 或 10mg**，每次口服，2小时内无效可追加一次，24小时内最大剂量 **30mg**。证据等级：A；推荐强度：1。\n\n⚠️ 冠心病、未控制高血压患者禁用。", "sources": ["中国偏头痛诊治指南2022版 · 表3 · 第2行"]},
        "治疗": {"tool": "hybrid", "answer": "偏头痛急性期治疗分为**特异性**和**非特异性**两类：\n\n**特异性**：曲普坦类（首选）、地坦类、吉泮类；\n**非特异性**：NSAIDs（布洛芬400-600mg、萘普生500mg）、对乙酰氨基酚1000mg；\n**止吐**：甲氧氯普胺可辅助用于恶心呕吐明显者。", "sources": ["中国偏头痛诊治指南2022版 · 第四章", "中国偏头痛急性期治疗指南2024 · 第二节"]},
        "最新": {"tool": "web", "answer": "（网络搜索结果）2026年最新发布的《中国前庭性偏头痛诊治指南2026版》已于2026年1月发布。\n\n注：此为模拟数据，仅用于界面演示。", "sources": ["Tavily网络搜索"]},
        "共病": {"tool": "graph", "answer": "【图谱检索结果】偏头痛与癫痫共病时，**丙戊酸钠**兼具抗癫痫和偏头痛预防双重适应症，是首选药物。\n\n注：跨文档关联来自《偏头痛诊治指南2022》和《癫痫共患偏头痛专家共识》。", "sources": ["GraphRAG 图谱 · 跨文档关联"]},
    }
    for kw, data in answers.items():
        if kw in query:
            return data
    return {"tool": "hybrid", "answer": f"【Demo模式】您询问的是：「{query}」\n\n这是模拟回答，用于前端调试，不消耗任何API额度。", "sources": ["Demo模式 · 无真实检索"]}


# ───────────────────────────────────────────────
# 真实Agent调用
# ───────────────────────────────────────────────
import json
import re


def _execute_tool_call(tool_name: str, arguments: dict) -> str:
    if not tool_name:
        return "无法识别工具调用"
    query = arguments.get("query") if isinstance(arguments, dict) else str(arguments)
    if tool_name == "knowledge_base_search":
        from src.agent.tools import query_knowledge_base
        return query_knowledge_base(query or "")
    if tool_name == "tavily_search":
        from langchain_tavily import TavilySearch
        searcher = TavilySearch(max_results=2, tavily_api_key=os.getenv("TAVILY_API_KEY"))
        try:
            result = searcher.invoke(query or "")
            if isinstance(result, list):
                return "\n\n".join(item.get("content") if isinstance(item, dict) and item.get("content") else str(item) for item in result[:3])
            return str(result)
        except Exception as exc:
            return f"网络搜索失败：{exc}"
    if tool_name == "graph_search":
        from src.retrieval.graph_retriever import GraphRetriever
        retriever = GraphRetriever()
        results = retriever.search(query or "", top_k=3)
        return retriever.format_context(results)
    return f"未实现的工具：{tool_name}"


def _parse_tool_call_text(text: str):
    if not isinstance(text, str) or not text.strip():
        return None
    match = re.search(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}', text, re.S)
    if not match:
        return None
    tool_name, args_json = match.groups()
    try:
        return tool_name, json.loads(args_json)
    except Exception:
        return None


def _real_answer(query: str) -> dict:
    mode = st.session_state.get("search_mode", "auto")
    from src.retrieval.router import auto_detect_with_strategy, rewrite_query_with_history
    
    # 1. 多轮对话 Query 重写
    if mode == "auto":
        history = st.session_state.messages[-4:] if st.session_state.messages else []
        query = rewrite_query_with_history(query, history)
        print(f"[Router] 重写后的 Query: {query}")
        
        # 2. 智能路由
        strategy = os.getenv("INTENT_DETECTION_STRATEGY", "hybrid")
        detected_mode = auto_detect_with_strategy(query, strategy=strategy)
        print(f"[Router] 路由决策: {detected_mode}")
        mode = detected_mode

    # === 抽离 LLM，统一用于所有分支的总结 ===
    llm = ChatOpenAI(model="qwen2.5:14b-instruct-q4_K_M", base_url="http://localhost:11434/v1", api_key="ollama", temperature=0)
    
    def _summarize(context: str) -> str:
        if not context or not context.strip():
            return "根据当前资料无法确定。"
        prompt = f"根据以下资料回答问题：\n{context}\n\n问题：{query}\n回答："
        return llm.invoke(prompt).content

    # --- 按照 mode 走不同的检索 ---
    # if mode == "es":
    #     from src.retrieval.es_retriever import ESRetriever
    #     retriever = ESRetriever()
    #     context = retriever.format_context(retriever.search(query, top_k=3))
    #     return {"tool": "es", "answer": _summarize(context), "sources": ["ES结构化检索"]}

    # if mode == "hybrid":
    #     from src.retrieval.vector_retriever import VectorRetriever
    #     try:
    #         retriever = VectorRetriever()
    #         context = retriever.format_context(retriever.search(query, top_k=3))
    #         return {"tool": "hybrid", "answer": _summarize(context), "sources": ["hybrid检索"]}
    #     except Exception as e:
    #         print(f"[Hybrid] 检索发生异常: {e}")
    #         # 如果检索崩溃，返回一个友好的错误提示，防止 Streamlit 红屏
    #         return {"tool": "error", "answer": f"⚠️ 语义检索服务暂时不可用，请稍后重试。", "sources": []}

    if mode == "es":
        from src.retrieval.es_retriever import ESRetriever
        retriever = ESRetriever()
        results = retriever.search(query, top_k=3)
        context = retriever.format_context(results)
        
        # ★ 提取真实的指南来源
        sources = []
        for r in results:
            doc = r.get('source', {})
            src_info = doc.get('source', {})
            guide_name = src_info.get('guideline', '未知指南')
            year = src_info.get('year')
            
            # 清洗一下名字（去 UUID）和年份
            clean_name = _clean_display_name(guide_name)
            year_str = f" ({year}版)" if year not in (None, 1111, 0) else ""
            
            source_text = f"{clean_name}{year_str}"
            if source_text not in sources:
                sources.append(source_text)
                
        return {"tool": "es", "answer": _summarize(context), "sources": sources or ["ES结构化检索"]}

    if mode == "hybrid":
        from src.retrieval.vector_retriever import VectorRetriever
        try:
            retriever = VectorRetriever()
            nodes = retriever.search(query, top_k=3)
            context = retriever.format_context(nodes)
            
            # ★ 提取真实的指南来源
            sources = []
            for n in nodes:
                # LlamaIndex 读取 MD 时，默认会把文件名存在 metadata["file_name"]
                file_name = n.metadata.get("file_name", "")
                if file_name:
                    clean_name = _clean_display_name(file_name)
                    if clean_name not in sources:
                        sources.append(clean_name)
                        
            return {"tool": "hybrid", "answer": _summarize(context), "sources": sources or ["hybrid检索"]}
        except Exception as e:
            print(f"[Hybrid] 检索发生异常: {e}")
            return {"tool": "error", "answer": f"⚠️ 语义检索服务暂时不可用，请稍后重试。", "sources": []}

        
    if mode == "graph":
        from src.retrieval.graph_retriever import GraphRetriever
        retriever = GraphRetriever()
        try:
            retriever.timeout = 300 
            results = retriever.search(query, top_k=3)
            if results and results[0].get("type") in ("graph_error", "graph_status"):
                raise Exception(results[0].get("message"))
            
            context = retriever.format_context(results)
            return {"tool": "graph", "answer": _summarize(context), "sources": ["GraphRAG 跨文档知识图谱"]}
        except Exception as e:
            print(f"[Graph] 发生异常: {e}，熔断降级到 Hybrid 检索")
            from src.retrieval.vector_retriever import VectorRetriever
            retriever_h = VectorRetriever()
            context_h = retriever_h.format_context(retriever_h.search(query, top_k=3))
            fallback_answer = "⚠️ 跨文档图谱服务响应超时/正在构建中，已为您自动降级为本地文档语义检索。\n\n" + _summarize(context_h)
            return {"tool": "hybrid", "answer": fallback_answer, "sources": ["GraphRAG降级 - 本地语义检索"]}

    # 显式的网络搜索分支
    if mode == "web":
        print(f"[Web] 调用 Tavily 搜索: {query}")
        try:
            searcher = TavilySearch(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))
            result = searcher.invoke(query)
            if isinstance(result, list):
                context = "\n\n".join(item.get("content", "") for item in result[:3])
            else:
                context = str(result)
            return {"tool": "web", "answer": _summarize(context), "sources": ["Tavily 网络搜索"]}
        except Exception as e:
            return {"tool": "error", "answer": f"网络搜索失败：{e}", "sources": []}

    # --- 剩余情况走 Agent 兜底（用于处理 auto 模式下的复杂调度） ---
    agent = _load_agent()
    if agent is None:
        return {"tool": "error", "answer": "Agent未能初始化，请检查src/agent/模块", "sources": []}

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    tool_used = None
    answer_text = ""

    try:
        for event in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            config=config, stream_mode="values",
        ):
            if "messages" in event:
                last = event["messages"][-1]
                if last.type == "ai":
                    if hasattr(last, "tool_calls") and last.tool_calls:
                        tool_used = last.tool_calls[0]["name"]
                        if last.tool_calls[0].get("args"):
                            answer_text = _execute_tool_call(tool_used, last.tool_calls[0]["args"])
                    elif last.content:
                        parsed = _parse_tool_call_text(str(last.content))
                        if parsed:
                            tool_name, arguments = parsed
                            tool_used = tool_name
                            answer_text = _execute_tool_call(tool_name, arguments)
                        else:
                            answer_text = last.content
    except Exception as e:
        return {"tool": "error", "answer": f"调用出错：{e}", "sources": []}

    return {"tool": tool_used or "hybrid", "answer": answer_text or "无法获取回答", "sources": []}


# ───────────────────────────────────────────────
# 工具标签渲染
# ───────────────────────────────────────────────
_TOOL_META = {
    "es": ("ES精确检索", "es"), "hybrid": ("语义混合检索", "hybrid"),
    "graph": ("图谱检索", "graph"), "web": ("网络搜索", "web"),
    "knowledge_base_search": ("知识库检索", "hybrid"),
    "tavily_search": ("网络搜索", "web"),
    "graph_search": ("图谱检索", "graph"),
    "error": ("错误", "es"),
}

def _tool_tag_html(tool_key: str) -> str:
    label, css = _TOOL_META.get(tool_key, ("未知工具", "hybrid"))
    return f'<span class="tool-tag {css}">{label}</span>'

def _render_message(msg: dict):
    role = msg["role"]
    content = msg["content"]
    ts = msg.get("time", "")
    tool = msg.get("tool")
    sources = msg.get("sources", [])

    if role == "user":
        # 用户消息（靠右）
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
            st.markdown(f'<div class="msg-time" style="text-align: right; margin-top: -10px;">{ts}</div>', unsafe_allow_html=True)
            
    else:
        # AI 消息（靠左）
        with st.chat_message("assistant", avatar="🧠"):
            # 工具标签
            if tool:
                tag_html = _tool_tag_html(tool)
                st.markdown(f'<div class="tool-tags">{tag_html}</div>', unsafe_allow_html=True)
            
            # 内容（原生 Markdown 渲染，列表、加粗完美生效）
            st.markdown(content)
            
            # 来源信息
            if sources:
                src_lines = "".join(f"<div>· {s}</div>" for s in sources)
                sources_html = f'<div class="source-block" style="margin-top: 10px;"><div class="src-title">📎 信息来源</div>{src_lines}</div>'
                st.markdown(sources_html, unsafe_allow_html=True)
                
            # 时间
            st.markdown(f'<div class="msg-time" style="margin-top: -5px;">{ts}</div>', unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Pipeline Demo
# ───────────────────────────────────────────────
def _run_pipeline_demo(filename: str):
    steps = [
        ("📤 上传至MinerU API...", 0.15), ("⏳ 等待MinerU解析...", 0.40),
        ("📥 下载JSON和MD文件...", 0.55), ("🗄️ ES结构化入库...", 0.70),
        ("🔍 构建ChromaDB向量索引...", 0.85), ("🕸️ GraphRAG实体抽取...", 0.95),
        ("✅ 入库完成！", 1.00),
    ]
    prog = st.progress(0, text="初始化...")
    log_area = st.empty()
    log_lines = []
    for text, pct in steps:
        time.sleep(0.6)
        prog.progress(pct, text=text)
        log_lines.append(f"`{datetime.now().strftime('%H:%M:%S')}` {text}")
        log_area.markdown("\n\n".join(log_lines))
    st.session_state.kb_list.append({"name": Path(filename).stem, "year": 2024, "status": "ok"})


# ═══════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════
with st.sidebar:

    st.markdown("""
    <div style="padding:14px 0 6px; text-align:center;">
        <div style="font-size:2rem;">🧠</div>
        <div style="font-size:1.4rem; font-weight:700; color:#555555; margin:4px 0;">偏头痛指南 RAG</div>
        <div style="font-size:1.1rem; color:#777777; font-family:'JetBrains Mono',monospace;">
            Es · Vector · BM25 · GraphRAG · TavilySearch
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── 1. PDF上传入口 ──
    st.markdown('<div class="sidebar-section-title">📁 上传指南PDF</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label="拖拽或点击上传", type=["pdf"], accept_multiple_files=False,
        label_visibility="collapsed", key="pdf_uploader",
    )
    if uploaded:
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"📄 {uploaded.name[:18]}...")
        with col_b:
            if st.button("🚀 开始入库", key="btn_ingest", use_container_width=True):
                with st.spinner("处理中..."):
                    if DEMO_MODE:
                        _run_pipeline_demo(uploaded.name)
                        st.session_state.flash = {
                            "level": "success",
                            "msg": f"✅ 《{uploaded.name}》已入库（Demo）",
                        }
                    else:
                        try:
                            from pipeline import run_pipeline_upload
                            result = run_pipeline_upload(
                                pdf_bytes=uploaded.read(),
                                filename=uploaded.name,
                                skip_review=False,
                            )
                            if result["status"] == "pending_review":
                                _refresh_kb_list()
                                st.session_state.current_page = "review"
                                st.session_state.flash = {
                                    "level": "success",
                                    "msg": f"✅ 已提取 {result['extracted_count']} 条记录，请逐条审核",
                                }
                            elif result["status"] == "no_records":
                                _refresh_kb_list()
                                st.session_state.current_page = "chat"
                                st.session_state.flash = {
                                    "level": "warning",
                                    "msg": f"⚠️ 未抽取到有效记录：{result.get('warning') or '请检查 PDF 是否包含药物治疗表格。'}",
                                }
                            else:
                                _refresh_kb_list()
                                st.session_state.current_page = "chat"
                                st.session_state.flash = {"level": "success", "msg": "✅ 入库完成！"}
                        except Exception as e:
                            st.session_state.current_page = "chat"
                            st.session_state.flash = {"level": "error", "msg": f"入库失败: {e}"}
                st.rerun()

    st.divider()

    # ── 2. 知识库管理 ──
    st.markdown('<div class="sidebar-section-title">📚 知识库</div>', unsafe_allow_html=True)
    if st.session_state.kb_list:
        for i, kb in enumerate(st.session_state.kb_list):
            # ★ 修改点4：显示前清洗名字/年份
            disp_name = _clean_display_name(kb.get('name'))
            disp_year = _format_year(kb.get('year'))
            short_name = disp_name if len(disp_name) <= 16 else disp_name[:14] + "…"

            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"""
                <div class="kb-item">
                    <div>
                        <div class="kb-item-name">📖 {short_name}</div>
                        <div class="kb-item-meta">{disp_year}</div>
                    </div>
                    <div class="kb-status-ok"></div>
                </div>""", unsafe_allow_html=True)
            with col2:

                if st.button("🗑", key=f"del_kb_{i}", help="从知识库移除"):
                    kb = st.session_state.kb_list[i]
                    if DEMO_MODE:
                        st.session_state.kb_list.pop(i)
                        st.session_state.flash = {"level": "success", "msg": f"✅ 已移除：{disp_name}（Demo）"}
                        st.rerun()
                    else:
                        from src.utils.kb_delete import delete_guideline_complete
                        with st.spinner(f"正在删除 {disp_name}..."):
                            try:
                                result = delete_guideline_complete(kb['name'], kb['year'])
                                
                                # ★ 构造极其详细的清理报告
                                detail_msg = f"ES清理 {result.get('es_deleted', 0)} 条 | Chroma清理 {result.get('chroma_deleted', 0)} 条"
                                if result.get("graph_deleted"):
                                    detail_msg += " | GraphRAG已清理"
                                else:
                                    detail_msg += " | ⚠️ 图谱未清理"
                                
                                # 保存到 flash 供页面顶部展示
                                st.session_state.flash = {
                                    "level": "success", 
                                    "msg": f"✅ 《{disp_name}》已彻底删除！详情：{detail_msg}"
                                }
                                
                                # ★ 尝试使用 st.toast 弹窗提示 (需 Streamlit 1.25+)
                                try:
                                    st.toast(f"🗑️ 已删除 {disp_name}，{detail_msg}", icon="✅")
                                except AttributeError:
                                    pass # 如果版本低不支持 toast，则跳过，靠 flash 提示
                                    
                            except Exception as e:
                                st.session_state.flash = {"level": "error", "msg": f"删除失败: {e}"}
                                
                        st.session_state.kb_list.pop(i)
                        st.rerun()
    else:
        st.caption("暂无指南，请上传PDF")

    st.divider()

    # ── 3. 检索模式 ──
    st.markdown('<div class="sidebar-section-title">🎯 检索模式</div>', unsafe_allow_html=True)
    mode = st.radio(
        label="检索模式", options=["auto", "es", "hybrid", "graph"],
        format_func=lambda x: {
            "auto": "🤖 自动路由（推荐）", "es": "🗄️  ES精确检索",
            "hybrid": "🔍 语义混合检索", "graph": "🕸️  图谱检索",
        }[x],
        index=0, label_visibility="collapsed", key="search_mode_radio",
    )
    st.session_state.search_mode = mode

    st.divider()

    # ── 4. 图谱探索入口 ──
    st.markdown('<div class="sidebar-section-title">🕸️ 图谱探索</div>', unsafe_allow_html=True)
    if st.button("打开知识图谱可视化", use_container_width=True, key="btn_graph"):
        st.session_state["show_graph"] = not st.session_state.get("show_graph", False)

    st.divider()

    # ── 5. 统计面板 ──
    st.markdown('<div class="sidebar-section-title">📊 会话统计</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-num">{st.session_state.total_queries}</div><div class="stat-lbl">总查询</div></div>
        <div class="stat-box"><div class="stat-num">{st.session_state.es_queries}</div><div class="stat-lbl">ES检索</div></div>
    </div>
    <div class="stat-row">
        <div class="stat-box"><div class="stat-num">{st.session_state.hybrid_queries}</div><div class="stat-lbl">混合</div></div>
        <div class="stat-box"><div class="stat-num">{st.session_state.graph_queries}</div><div class="stat-lbl">图谱</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── 6. 工具状态 ──
    st.markdown('<div class="sidebar-section-title">🔧 工具状态</div>', unsafe_allow_html=True)
    demo_dot = "dot-off" if not DEMO_MODE else "dot-on"
    st.markdown(f"""
    <div class="tool-status"><div class="dot-on"></div>knowledge_base_search</div>
    <div class="tool-status"><div class="dot-on"></div>tavily_search</div>
    <div class="tool-status"><div class="dot-on"></div>graph_search</div>
    <div class="tool-status"><div class="{demo_dot}"></div>
        {"⚡ DEMO MODE" if DEMO_MODE else "生产模式"}</div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── 7. 示例问题 ──
    st.markdown('<div class="sidebar-section-title">💡 示例问题</div>', unsafe_allow_html=True)
    examples = [
        "利扎曲普坦的推荐剂量是多少？",
        "偏头痛急性期有哪些治疗方法？",
        "佐米曲普坦有哪些禁忌症？",
        "癫痫共病偏头痛如何用药？",
        "2026年最新偏头痛指南有什么更新？",
    ]
    for ex in examples:
        if st.button(f"▸ {ex[:20]}", key=f"ex_{ex[:8]}", use_container_width=True):
            st.session_state.example_query = ex
            st.rerun()

    st.divider()

    # ── 8. 审核中心入口 ──
    st.markdown('<div class="sidebar-section-title">📝 审核中心</div>', unsafe_allow_html=True)
    pending_count = _count_pending_reviews()
    if pending_count > 0:
        st.markdown(f"""
        <div style="background:#4a2800; padding:10px; border-radius:8px; border:1px solid #7a4400;">
            <div style="color:#ffa657; font-size:0.8rem; font-weight:600;">
                🔔 {pending_count} 个指南待审核
            </div>
        </div>
        """, unsafe_allow_html=True)
    if st.button("进入审核中心", use_container_width=True, key="btn_to_review"):
        st.session_state.current_page = "review"
        st.rerun()

    st.divider()

    # ── 9. 对话管理 ──
    col_exp, col_clr = st.columns(2)
    with col_exp:
        if st.button("💾 导出", use_container_width=True, key="btn_export"):
            if st.session_state.messages:
                lines = []
                for m in st.session_state.messages:
                    role_str = "用户" if m["role"] == "user" else "AI"
                    lines.append(f"[{m.get('time','')}] {role_str}：{m['content']}\n")
                export_text = "\n".join(lines)
                st.download_button(
                    "📥 下载TXT", data=export_text,
                    file_name=f"对话_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain", key="btn_dl",
                )
    with col_clr:
        if st.button("🗑 清空", use_container_width=True, key="btn_clear"):
            st.session_state.messages = []
            st.session_state.thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.rerun()


# ═══════════════════════════════════════════════
# 主内容区
# ═══════════════════════════════════════════════

# flash 提示 + 路由提前
flash = st.session_state.pop("flash", None)
if flash:
    _fn = {"success": st.success, "warning": st.warning,
           "error": st.error, "info": st.info}.get(flash["level"], st.info)
    _fn(flash["msg"])

page = st.session_state.get("current_page", "chat")
if page == "review":
    _render_review_page()
    st.stop()

# ── 以下是聊天页 ──
demo_badge = '<span class="demo-badge">⚡ DEMO MODE</span>' if DEMO_MODE else ""
st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-icon">🧠</div>
    <div>
        <div class="top-bar-title">偏头痛医疗指南智能问答</div>
        <div class="top-bar-sub">
            ElasticSearch · BM25+Vector · GraphRAG · TavilySearch · Agentic RAG
            &nbsp;|&nbsp; 会话 {st.session_state.thread_id[-8:]}
        </div>
    </div>
    {demo_badge}
</div>
""", unsafe_allow_html=True)

if st.session_state.get("show_graph", False):
    with st.expander("🕸️ 知识图谱探索（偏头痛实体关系）", expanded=True):
        st.info("Demo模式下显示静态示意。")
        if DEMO_MODE:
            st.markdown("""
            ```
            偏头痛
            ├── [急性期治疗]
            │   ├── 曲普坦类 → 利扎曲普坦 (10mg, A级推荐)
            │   ├── 曲普坦类 → 佐米曲普坦 (2.5mg, A级推荐)
            │   └── NSAIDs  → 布洛芬 (400-600mg)
            ├── [预防性治疗]
            │   ├── β受体阻滞剂 → 美托洛尔
            │   └── 抗癫痫药 → 丙戊酸钠 ←→ [癫痫共病]
            └── [共病关系]
                ├── 癫痫 → 丙戊酸钠（双适应症）
                └── 抑郁 → 阿米替林（双适应症）
            ```
            """)
        else:
            st.warning("GraphRAG 正在初始化中")

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── 聊天消息列表 ──
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding:60px 0; color:#484f58;">
            <div style="font-size:3rem; margin-bottom:16px;">🩺</div>
            <div style="font-size:1.2rem; font-family:'JetBrains Mono',monospace;">
                支持剂量精准查询 · 不良反应 · 禁忌症 · 跨指南关联
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            _render_message(msg)

# ── 输入区 ──
st.markdown("<div class='input-area'>", unsafe_allow_html=True)

# 处理示例问题预填
if st.session_state.get("example_query"):
    st.session_state.chat_input = st.session_state.pop("example_query")

# ★ 使用 st.form 包裹，clear_on_submit=True 会在提交后自动清空输入框
with st.form(key="chat_form", clear_on_submit=True, border=False):
    col_input, col_send = st.columns([6, 1])
    with col_input:
        user_input = st.text_input(
            label="输入框",
            placeholder="输入问题，例如：利扎曲普坦的推荐剂量是多少？",
            label_visibility="collapsed",
            key="chat_input",
        )
    with col_send:
        # 在 form 中必须使用 form_submit_button
        send = st.form_submit_button("发送 →", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── 免责声明 ──
st.markdown("""
<div style="text-align:center; padding:10px 0 4px;
            font-size:0.9rem; color:#484f58; font-family:'JetBrains Mono',monospace;">
    ⚠️ 本系统仅供参考，不替代专业医生诊断建议
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# 消息处理逻辑
# ═══════════════════════════════════════════════
if send and user_input and user_input.strip():
    query = user_input.strip()
    now_str = datetime.now().strftime("%H:%M")

    # 记录用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "time": now_str,
    })
    st.session_state.total_queries += 1

    # 获取回答
    with st.spinner("思考中..."):
        if DEMO_MODE:
            result = _mock_answer(query)
        else:
            result = _real_answer(query)

    # 更新统计
    tool = result.get("tool", "hybrid")
    if tool in ("es",):
        st.session_state.es_queries += 1
    elif tool in ("graph", "graph_search"):
        st.session_state.graph_queries += 1
    elif tool in ("web", "tavily_search"):
        st.session_state.web_searches += 1
    else:
        st.session_state.hybrid_queries += 1

    # 记录AI回答
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "tool": tool,
        "sources": result.get("sources", []),
        "time": datetime.now().strftime("%H:%M"),
    })

    st.rerun()
    
# ======================
# 侧边栏宽度滑块（放最底部）
# ======================
with st.sidebar:
    st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px; color:#333333; text-align:center; margin-bottom:5px;'>侧边栏宽度</div>", unsafe_allow_html=True)
    sw = st.slider("", 280, 800, 680, 10, label_visibility="collapsed")

st.markdown(f"""
<style>
section[data-testid="stSidebar"] {{
    width: {sw}px !important;
    min-width: {sw}px !important;
    max-width: {sw}px !important;
}}
</style>
""", unsafe_allow_html=True)
