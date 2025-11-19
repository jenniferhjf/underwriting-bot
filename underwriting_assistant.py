"""
Underwriting Assistant - Professional RAG+CoT System
专业承保助手 - RAG+CoT系统

Updates (2025-11-19):
- 取消 Workspace 切换功能，只保留一个默认知识库（用户无感知）
- 保留左侧导航条：上方主题切换 + 中间多会话列表 + 下方知识库统计
- 每次聊天记录本地持久化，并支持像 GPT 一样在左侧新建/切换会话
"""

import streamlit as st
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any
import requests
import PyPDF2
from docx import Document
import base64
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

# DeepSeek API Configuration（你也可以改成用环境变量）
DEEPSEEK_API_KEY = "sk-99bba2ce117444e197270f17d303e74f"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# Directories
DATA_DIR = "data"
WORKSPACES_DIR = os.path.join(DATA_DIR, "workspaces")  # 内部用一个 default workspace
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
CHATS_DIR = os.path.join(DATA_DIR, "chats")

os.makedirs(WORKSPACES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(CHATS_DIR, exist_ok=True)

CHATS_INDEX_FILE = os.path.join(CHATS_DIR, "sessions.json")

SUPPORTED_FORMATS = {
    "pdf": "📄 PDF",
    "docx": "📝 Word",
    "doc": "📝 Word",
    "txt": "📃 Text",
    "xlsx": "📊 Excel",
    "xls": "📊 Excel",
    "png": "🖼️ Image",
    "jpg": "🖼️ Image",
    "jpeg": "🖼️ Image"
}

TAG_OPTIONS = {
    "equipment": ["Gas Turbine", "Steam Turbine", "Boiler", "Generator", "Compressor",
                  "Heat Exchanger", "Pump", "Transformer", "Motor", "Other"],
    "industry": ["Oil & Gas", "Power Generation", "Manufacturing", "Chemical",
                 "Mining", "Refining", "Marine", "Aviation", "Other"],
    "timeline": ["2025-Q4", "2025-Q3", "2025-Q2", "2025-Q1", "2024", "2023", "Earlier"]
}

DEFAULT_WORKSPACE_NAME = "default"  # 只用这一个知识库

# ============================================================================
# SYSTEM INSTRUCTION (CoT)
# ============================================================================

SYSTEM_INSTRUCTION = """Role: You are Mr. X's AI underwriting assistant

Task: Answer underwriting queries using retrieved cases

Process: Think step-by-step using this framework:

Step 1: Extract key tags from query
Step 2: Analyze retrieved precedents
Step 3: Check recency & applicability
Step 4: Identify decision patterns
Step 5: Recommend with rationale

Output: Provide decision + premium + sources"""

# ============================================================================
# UTILS
# ============================================================================

def call_deepseek_api(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2000) -> str:
    if not DEEPSEEK_API_KEY:
        return "❌ API Error: DEEPSEEK_API_KEY is not set."
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        resp = requests.post(f"{DEEPSEEK_API_BASE}/chat/completions",
                             headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API Error: {str(e)}"

def extract_text_from_pdf(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return f"Error extracting DOCX: {str(e)}"

def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading TXT: {str(e)}"

def extract_text_from_file(file_path: str, file_format: str) -> str:
    if file_format == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_format in ["docx", "doc"]:
        return extract_text_from_docx(file_path)
    elif file_format == "txt":
        return extract_text_from_txt(file_path)
    elif file_format in ["xlsx", "xls", "png", "jpg", "jpeg"]:
        # 这些类型右侧预览时处理，这里不抽取
        return ""
    return "Unsupported format for text extraction"

def generate_embedding(text: str) -> List[float]:
    text_hash = hashlib.md5((text or "").encode()).hexdigest()
    fake = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
    fake = fake + [0.0] * (1536 - len(fake))
    return fake[:1536]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a*b for a, b in zip(v1, v2))
    m1 = sum(a*a for a in v1) ** 0.5
    m2 = sum(b*b for b in v2) ** 0.5
    if m1 == 0 or m2 == 0:
        return 0.0
    return dot / (m1 * m2)

def file_to_data_uri(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

# ============================================================================
# AUTO ANNOTATION (LLM)
# ============================================================================

AUTO_ANNOTATE_SYSTEM = """You are an underwriting document auto-tagger.
Given raw extracted text and the filename, produce a STRICT JSON object with:
{
  "tags": {"equipment": string[], "industry": string[], "timeline": string[]},
  "decision": "Approved" | "Declined" | "Conditional" | "Pending",
  "premium": number,
  "risk_level": "Low" | "Medium" | "Medium-High" | "High" | "Critical",
  "case_summary": string,
  "key_insights": string
}
Rules:
- Return ONLY valid JSON. No commentary.
- If unavailable, use 'Other' / 'Earlier' conservatively.
"""

def auto_annotate_by_llm(extracted_text: str, filename: str) -> Dict[str, Any]:
    user_prompt = f"FILENAME: {filename}\nTEXT:\n{(extracted_text or '')[:4000]}"
    content = call_deepseek_api(
        messages=[
            {"role": "system", "content": AUTO_ANNOTATE_SYSTEM},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=700
    )
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
        data = json.loads(cleaned)
    except Exception:
        data = {
            "tags": {"equipment": ["Other"], "industry": ["Other"], "timeline": ["Earlier"]},
            "decision": "Pending",
            "premium": 0,
            "risk_level": "Medium",
            "case_summary": "Auto-tagging failed. Placeholder values used.",
            "key_insights": "Please re-run auto-tagging if needed."
        }
    data.setdefault("tags", {})
    data["tags"].setdefault("equipment", ["Other"])
    data["tags"].setdefault("industry", ["Other"])
    data["tags"].setdefault("timeline", ["Earlier"])
    data.setdefault("decision", "Pending")
    data.setdefault("premium", 0)
    data.setdefault("risk_level", "Medium")
    data.setdefault("case_summary", "")
    data.setdefault("key_insights", "")
    return data

# ============================================================================
# WORKSPACE（内部只用一个 default）
# ============================================================================

class Workspace:
    def __init__(self, name: str):
        self.name = name
        self.workspace_dir = os.path.join(WORKSPACES_DIR, name)
        self.documents_dir = os.path.join(self.workspace_dir, "documents")
        self.metadata_file = os.path.join(self.workspace_dir, "metadata.json")
        self.embeddings_file = os.path.join(self.workspace_dir, "embeddings.json")
        os.makedirs(self.documents_dir, exist_ok=True)
        self.metadata = self._load_metadata()
        self.embeddings = self._load_embeddings()

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_metadata(self):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_embeddings(self):
        with open(self.embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(self.embeddings, f, indent=2)

    def add_document(self, uploaded_file, tags: Dict[str, List[str]],
                     case_summary: str, key_insights: str,
                     decision: str, premium: int, risk_level: str,
                     extracted_text_preview: str = "") -> Dict[str, Any]:
        doc_id = f"DOC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(uploaded_file.name.encode()).hexdigest()[:6].upper()}"
        ext = uploaded_file.name.split('.')[-1].lower()
        filename = f"{doc_id}.{ext}"
        file_path = os.path.join(self.documents_dir, filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        full_text = f"{case_summary} {key_insights} {extracted_text_preview[:1000]}"
        embedding = generate_embedding(full_text)
        doc_meta = {
            "doc_id": doc_id,
            "filename": uploaded_file.name,
            "file_format": ext,
            "file_path": file_path,
            "file_size_kb": uploaded_file.size / 1024,
            "upload_date": datetime.now().isoformat(),
            "tags": tags,
            "decision": decision,
            "premium": premium,
            "risk_level": risk_level,
            "case_summary": case_summary,
            "key_insights": key_insights,
            "extracted_text_preview": extracted_text_preview[:500]
        }
        self.metadata.append(doc_meta)
        self.embeddings[doc_id] = embedding
        self._save_metadata()
        self._save_embeddings()
        return doc_meta

    def search_documents(self, query: str, top_k: int = 5):
        if not self.metadata:
            return []
        qv = generate_embedding(query)
        scored = []
        for doc in self.metadata:
            doc_id = doc["doc_id"]
            if doc_id in self.embeddings:
                sim = cosine_similarity(qv, self.embeddings[doc_id])
                ql = query.lower()
                for tag_list in doc["tags"].values():
                    for tag in tag_list:
                        if tag.lower() in ql:
                            sim += 0.1
                scored.append((sim, doc))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [d for _, d in scored[:top_k]]

    def delete_document(self, doc_id: str):
        self.metadata = [d for d in self.metadata if d["doc_id"] != doc_id]
        if doc_id in self.embeddings:
            del self.embeddings[doc_id]
        for fn in os.listdir(self.documents_dir):
            if fn.startswith(doc_id):
                os.remove(os.path.join(self.documents_dir, fn))
        self._save_metadata()
        self._save_embeddings()

    def get_stats(self):
        return {
            "total_documents": len(self.metadata),
            "total_size_mb": sum(d["file_size_kb"] for d in self.metadata) / 1024 if self.metadata else 0.0,
            "format_distribution": self._get_fmt_dist(),
            "decision_distribution": self._get_decision_dist()
        }

    def _get_fmt_dist(self):
        dist = {}
        for d in self.metadata:
            dist[d["file_format"]] = dist.get(d["file_format"], 0) + 1
        return dist

    def _get_decision_dist(self):
        dist = {}
        for d in self.metadata:
            dist[d["decision"]] = dist.get(d["decision"], 0) + 1
        return dist

# ============================================================================
# CHAT SESSION PERSISTENCE（多会话 + 本地持久化）
# ============================================================================

def _load_chat_sessions() -> List[Dict[str, Any]]:
    if os.path.exists(CHATS_INDEX_FILE):
        try:
            with open(CHATS_INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_chat_sessions(sessions: List[Dict[str, Any]]):
    with open(CHATS_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

def _chat_file_path(session_id: str) -> str:
    return os.path.join(CHATS_DIR, f"{session_id}.json")

def _load_session_messages(session_id: str) -> List[Dict[str, str]]:
    path = _chat_file_path(session_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_session_messages(session_id: str, messages: List[Dict[str, str]]):
    path = _chat_file_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def create_chat_session(title: str = "New chat") -> Dict[str, Any]:
    sessions = _load_chat_sessions()
    now = datetime.now()
    sid_raw = now.isoformat()
    sid = hashlib.md5(sid_raw.encode()).hexdigest()[:8].upper()
    session_id = f"{now.strftime('%Y%m%d%H%M%S')}-{sid}"
    session = {
        "id": session_id,
        "title": title,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat()
    }
    sessions.append(session)
    _save_chat_sessions(sessions)
    _save_session_messages(session_id, [])
    return session

def update_chat_session_meta(session_id: str, title: str | None = None):
    sessions = _load_chat_sessions()
    changed = False
    now = datetime.now().isoformat()
    for s in sessions:
        if s["id"] == session_id:
            s["updated_at"] = now
            if title and (s.get("title") in ["New chat", "", None]):
                s["title"] = title
            changed = True
            break
    if changed:
        _save_chat_sessions(sessions)

# ============================================================================
# CHAT (RAG + CoT)
# ============================================================================

def generate_cot_response(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    if not retrieved_docs:
        # 没有文档时也给一个兜底回答
        return call_deepseek_api(
            [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": f"Query: {query}\n\nNo retrieved cases available. Please answer based on general underwriting principles."}
            ]
        )

    docs_text = ""
    for doc in retrieved_docs:
        equipment = ", ".join(doc["tags"].get("equipment", []))
        industry = ", ".join(doc["tags"].get("industry", []))
        timeline = ", ".join(doc["tags"].get("timeline", []))
        docs_text += f"""
{'='*70}
DOCUMENT #{doc["doc_id"]}
{'='*70}
File: {doc["filename"]} ({doc["file_format"].upper()})
Tags: 🔧 {equipment} | 🏭 {industry} | 📅 {timeline}

Decision: {doc["decision"]}
Premium: ${doc["premium"]:,}
Risk Level: {doc["risk_level"]}

Case Summary:
{doc["case_summary"]}

Key Insights:
{doc["key_insights"]}

"""
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"""Query: "{query}"

Retrieved Cases:
{docs_text}

Please analyze using the 5-step CoT framework:
1. Extract key tags from query
2. Analyze retrieved precedents
3. Check recency & applicability
4. Identify decision patterns
5. Recommend with rationale

Provide: Decision + Premium Range + Sources"""}
    ]
    return call_deepseek_api(messages)

# ============================================================================
# UI: CSS
# ============================================================================

def inject_css(appearance: str):
    if appearance == "Dark":
        css = """
        <style>
        :root {
            --text-primary: #e5e7eb;
            --text-secondary: #cbd5e1;
            --muted: #9ca3af;
            --bg-app: #0b1220;
            --card-bg: #101826;
            --shadow: 0 1px 3px rgba(0,0,0,0.5);
            --brand: #93c5fd;
            --green: #86efac;
            --amber: #fde68a;
        }
        .stApp { background-color: var(--bg-app); color: var(--text-primary); }
        .main-header { font-size: 2rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.25rem; }
        .sub-header  { font-size: 1rem; color: var(--muted); margin-bottom: 1.0rem; }
        .tag-badge { display:inline-block; padding:0.25rem 0.75rem; margin:0.25rem 0.4rem 0.25rem 0; border-radius:1rem; font-size:0.875rem; font-weight:700; color:#0b1220; }
        .tag-equipment { background-color: #93c5fd; }
        .tag-industry  { background-color: #86efac; }
        .tag-timeline  { background-color: #fde68a; }
        .stChatMessage, .stMarkdown, p, li, label, span, div { color: var(--text-primary); }
        [data-testid="stMetricDelta"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--text-primary) !important; }
        #MainMenu, footer, header {visibility: hidden;}
        </style>
        """
    else:
        css = """
        <style>
        :root {
            --text-primary: #0f172a;
            --text-secondary: #374151;
            --muted: #6b7280;
            --bg-app: #f5f7fa;
            --card-bg: #ffffff;
            --shadow: 0 1px 3px rgba(0,0,0,0.1);
            --brand: #1e40af;
            --green: #166534;
            --amber: #92400e;
        }
        .stApp { background-color: var(--bg-app); color: var(--text-primary); }
        .main-header { font-size: 2rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.25rem; }
        .sub-header  { font-size: 1rem; color: var(--muted); margin-bottom: 1.0rem; }
        .tag-badge { display:inline-block; padding:0.25rem 0.75rem; margin:0.25rem 0.4rem 0.25rem 0; border-radius:1rem; font-size:0.875rem; font-weight:700; color: var(--text-primary); }
        .tag-equipment { background-color: #dbeafe; }
        .tag-industry  { background-color: #dcfce7; }
        .tag-timeline  { background-color: #fef3c7; }
        .stChatMessage, .stMarkdown, p, li, label, span, div { color: var(--text-primary); }
        [data-testid="stMetricDelta"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--text-primary) !important; }
        #MainMenu, footer, header {visibility: hidden;}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# MAIN
# ============================================================================

def main():
    st.set_page_config(
        page_title="Underwriting Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # === 侧边栏：主题 + 会话导航 + 知识库统计 ===
    workspace = Workspace(DEFAULT_WORKSPACE_NAME)
    stats = workspace.get_stats()

    with st.sidebar:
        # 主题
        st.markdown("### 🎨 Appearance")
        appearance = st.radio(
            "Theme",
            ["Light", "Dark"],
            horizontal=True,
            key="appearance_choice"
        )

        st.markdown("---")
        st.markdown("### 💬 Conversations")

        # 加载所有会话
        sessions = _load_chat_sessions()
        if not sessions:
            default_session = create_chat_session()
            sessions = _load_chat_sessions()

        sessions_sorted = sorted(
            sessions,
            key=lambda s: s.get("updated_at", ""),
            reverse=True
        )

        if "current_session_id" not in st.session_state:
            st.session_state.current_session_id = sessions_sorted[0]["id"]

        # 新建对话
        if st.button("➕ New Chat"):
            new_sess = create_chat_session()
            st.session_state.current_session_id = new_sess["id"]
            st.session_state.messages = []
            st.session_state.loaded_for = new_sess["id"]
            st.experimental_rerun()

        # 会话列表（像 GPT 左侧导航）
        label_to_id = {s["title"]: s["id"] for s in sessions_sorted}
        current_id = st.session_state.current_session_id
        current_label = next(
            (lbl for lbl, sid in label_to_id.items() if sid == current_id),
            list(label_to_id.keys())[0]
        )
        selected_label = st.radio(
            "Chats",
            options=list(label_to_id.keys()),
            index=list(label_to_id.keys()).index(current_label),
            key="chat_session_selector"
        )
        selected_id = label_to_id[selected_label]

        # 切换会话时加载 messages
        if (selected_id != current_id or
            "messages" not in st.session_state or
            st.session_state.get("loaded_for") != selected_id):
            st.session_state.current_session_id = selected_id
            st.session_state.messages = _load_session_messages(selected_id)
            st.session_state.loaded_for = selected_id

        # 知识库统计
        st.markdown("---")
        st.markdown("### 📊 Knowledge Base")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Documents", stats["total_documents"])
        with c2:
            st.metric("Size", f"{stats['total_size_mb']:.1f} MB")
        if stats["format_distribution"]:
            st.markdown("**Formats:**")
            for fmt, count in stats["format_distribution"].items():
                st.write(f"{SUPPORTED_FORMATS.get(fmt, fmt)}: {count}")

    # 应用主题 CSS
    inject_css(appearance)

    # 主标题
    st.markdown('<div class="main-header">🤖 Underwriting Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">RAG + CoT | Multimodal Extraction | Vector DB</div>', unsafe_allow_html=True)

    # Tabs 仍然在页面上方
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "📤 Upload (Auto-Tag)"])

    # === TAB 1: CHAT ===
    with tab1:
        st.markdown("### 💬 Chat with AI Assistant")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 展示历史消息
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt := st.chat_input("Ask about underwriting cases..."):
            # 用户消息
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 自动把第一条用户消息作为会话标题
            first_user_msg = next(
                (msg["content"] for msg in st.session_state.messages if msg["role"] == "user"),
                ""
            )
            short_title = (first_user_msg[:20] + "...") if len(first_user_msg) > 20 else first_user_msg
            if short_title:
                update_chat_session_meta(st.session_state.current_session_id, title=short_title)

            # Assistant 回复
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching knowledge base..."):
                    retrieved = workspace.search_documents(prompt, top_k=3)
                    resp = generate_cot_response(prompt, retrieved)
                    st.markdown(resp)
                    if retrieved:
                        with st.expander(f"📚 {len(retrieved)} Retrieved Documents"):
                            for d in retrieved:
                                st.markdown(f"**{d['doc_id']}** - {d['filename']}")
                                tags_html = ""
                                for t in d["tags"].get("equipment", []):
                                    tags_html += f'<span class="tag-badge tag-equipment">🔧 {t}</span>'
                                for t in d["tags"].get("industry", []):
                                    tags_html += f'<span class="tag-badge tag-industry">🏭 {t}</span>'
                                for t in d["tags"].get("timeline", []):
                                    tags_html += f'<span class="tag-badge tag-timeline">📅 {t}</span>'
                                st.markdown(tags_html, unsafe_allow_html=True)
                                st.markdown("---")

            # 保存助手消息 & 持久化本会话
            st.session_state.messages.append({"role": "assistant", "content": resp})
            _save_session_messages(st.session_state.current_session_id, st.session_state.messages)
            update_chat_session_meta(st.session_state.current_session_id)

    # === TAB 2: DOCUMENTS ===
    with tab2:
        st.markdown("### 📄 Knowledge Base")
        if not workspace.metadata:
            st.info("No documents yet. Upload in 'Upload (Auto-Tag)'.")
        else:
            left, right = st.columns([1, 2.2])

            with left:
                st.markdown("#### 📚 Knowledge Base Browser")
                q = st.text_input("Search title/tags...", key="kb_search")
                fe = st.multiselect("🔧 Equipment", TAG_OPTIONS["equipment"])
                fi = st.multiselect("🏭 Industry", TAG_OPTIONS["industry"])
                ft = st.multiselect("📅 Timeline", TAG_OPTIONS["timeline"])

                docs = workspace.metadata
                if q:
                    ql = q.lower()
                    docs = [
                        d for d in docs
                        if (ql in d["filename"].lower() or
                            any(ql in tag.lower() for v in d["tags"].values() for tag in v))
                    ]
                if fe:
                    docs = [d for d in docs if any(t in d["tags"].get("equipment", []) for t in fe)]
                if fi:
                    docs = [d for d in docs if any(t in d["tags"].get("industry", []) for t in fi)]
                if ft:
                    docs = [d for d in docs if any(t in d["tags"].get("timeline", []) for t in ft)]

                docs = sorted(docs, key=lambda d: d.get("upload_date", ""), reverse=True)

                options = {
                    f"{SUPPORTED_FORMATS.get(d['file_format'], '📎')} {d['filename']} [{d['doc_id']}]": d["doc_id"]
                    for d in docs
                }
                selected_doc = None
                if options:
                    selected_label_doc = st.radio("Documents", list(options.keys()), index=0, key="kb_selected")
                    selected_id_doc = options[selected_label_doc]
                    selected_doc = next((d for d in docs if d["doc_id"] == selected_id_doc), None)

                if selected_doc and st.button("🗑️ Delete Selected"):
                    workspace.delete_document(selected_doc["doc_id"])
                    st.success("Document deleted!")
                    st.experimental_rerun()

            with right:
                st.markdown("#### 👀 Preview Original")
                if not selected_doc:
                    st.info("Select a document on the left to preview.")
                else:
                    doc = selected_doc
                    st.markdown(
                        f"**{doc['filename']}**  \n"
                        f"ID: `{doc['doc_id']}` | Format: **{doc['file_format'].upper()}** | "
                        f"Size: {doc['file_size_kb']:.1f} KB"
                    )

                    # 下载按钮
                    with open(doc["file_path"], "rb") as f:
                        st.download_button("⬇️ Download file", f, file_name=doc["filename"], mime=None)

                    ext = doc["file_format"]
                    path = doc["file_path"]

                    if ext == "pdf":
                        try:
                            data_uri = file_to_data_uri(path, "application/pdf")
                            html = f'<iframe src="{data_uri}" width="100%" height="800px" style="border:none;"></iframe>'
                            st.components.v1.html(html, height=820, scrolling=True)
                        except Exception as e:
                            st.error(f"PDF preview failed: {e}")
                    elif ext in ["png", "jpg", "jpeg"]:
                        try:
                            st.image(path, use_column_width=True)
                        except Exception as e:
                            st.error(f"Image preview failed: {e}")
                    elif ext in ["docx", "doc"]:
                        text = extract_text_from_docx(path) if ext == "docx" else "(DOC preview not supported; please download)"
                        st.text_area("Extracted Text (preview)", value=text[:8000], height=400)
                    elif ext == "txt":
                        text = extract_text_from_txt(path)
                        st.text_area("Text File (preview)", value=text[:8000], height=400)
                    elif ext in ["xlsx", "xls"]:
                        try:
                            df = pd.read_excel(path)
                            st.dataframe(df.head(200), use_container_width=True)
                        except Exception as e:
                            st.error(f"Excel preview failed: {e}")
                    else:
                        st.info("Preview not supported for this file type. Please download to view.")

                    st.markdown("---")
                    st.markdown("**Auto Tags & Case Info**")
                    tags_html = ""
                    for t in doc["tags"].get("equipment", []):
                        tags_html += f'<span class="tag-badge tag-equipment">🔧 {t}</span>'
                    for t in doc["tags"].get("industry", []):
                        tags_html += f'<span class="tag-badge tag-industry">🏭 {t}</span>'
                    for t in doc["tags"].get("timeline", []):
                        tags_html += f'<span class="tag-badge tag-timeline">📅 {t}</span>'
                    st.markdown(tags_html, unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write(f"**Decision:** {doc['decision']}")
                    with c2:
                        st.write(f"**Premium:** ${doc['premium']:,}")
                    with c3:
                        st.write(f"**Risk:** {doc['risk_level']}")
                    st.write("**Case Summary:**")
                    st.info(doc["case_summary"])
                    st.write("**Key Insights:**")
                    st.write(doc["key_insights"])

    # === TAB 3: UPLOAD (Auto-Tag) ===
    with tab3:
        st.markdown("### 📤 Upload Document (Auto-Tag by Model)")
        st.caption("只需上传文件，系统会自动抽取文本并由模型进行标签与条款识别。")

        with st.form("upload_form_autotag"):
            uploaded_file = st.file_uploader(
                "Choose a document",
                type=list(SUPPORTED_FORMATS.keys()),
                help="Supported: PDF, Word, Excel, Text, Images"
            )
            submitted = st.form_submit_button("📤 Upload & Auto-Tag")

        if submitted:
            if not uploaded_file:
                st.error("Please upload a document")
            else:
                with st.spinner("Processing document & auto-tagging..."):
                    temp_id = f"TEMP-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(uploaded_file.name.encode()).hexdigest()[:6].upper()}"
                    ext = uploaded_file.name.split('.')[-1].lower()
                    temp_path = os.path.join(workspace.documents_dir, f"{temp_id}.{ext}")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    extracted_text = extract_text_from_file(temp_path, ext)
                    auto = auto_annotate_by_llm(extracted_text, uploaded_file.name)

                    doc = workspace.add_document(
                        uploaded_file=uploaded_file,
                        tags=auto["tags"],
                        case_summary=auto["case_summary"],
                        key_insights=auto["key_insights"],
                        decision=auto["decision"],
                        premium=int(auto.get("premium", 0) or 0),
                        risk_level=auto["risk_level"],
                        extracted_text_preview=extracted_text[:800]
                    )

                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception:
                        pass

                    st.success(f"✅ Document uploaded & auto-tagged: {doc['doc_id']}")
                    with st.expander("🔎 Auto-Tag Result"):
                        st.json(auto)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
