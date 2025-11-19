"""
Underwriting Assistant - Professional RAG+CoT System (No Workspace)
专业承保助手 - 无Workspace版本，支持多会话管理

Updates (2025-11-19):
- ❌ 移除Workspace功能 - 所有文档在统一知识库
- ✅ 多会话管理 - 像ChatGPT一样可以新建/切换/删除聊天
- ✅ 会话历史保存 - 每个会话独立保存聊天记录
- ✅ 侧边栏会话列表 - 快速切换不同对话
- ✅ 保留外观切换、自动标签、原件预览等功能
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

# DeepSeek API Configuration
DEEPSEEK_API_KEY = "sk-99bba2ce117444e197270f17d303e74f"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# Directories
DATA_DIR = "data"
KNOWLEDGE_BASE_DIR = os.path.join(DATA_DIR, "knowledge_base")  # 统一知识库
DOCUMENTS_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "documents")
METADATA_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "metadata.json")
EMBEDDINGS_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "embeddings.json")
CONVERSATIONS_DIR = os.path.join(DATA_DIR, "conversations")  # 会话目录

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

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
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"model": DEEPSEEK_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        resp = requests.post(f"{DEEPSEEK_API_BASE}/chat/completions", headers=headers, json=payload, timeout=60)
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
        return ""
    return "Unsupported format for text extraction"

def generate_embedding(text: str) -> List[float]:
    text_hash = hashlib.md5((text or "").encode()).hexdigest()
    fake = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
    fake = fake + [0.0] * (1536 - len(fake))
    return fake[:1536]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a*b for a,b in zip(v1, v2))
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
        messages=[{"role":"system","content":AUTO_ANNOTATE_SYSTEM},
                  {"role":"user","content":user_prompt}],
        temperature=0.2, max_tokens=700
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
            "tags":{"equipment":["Other"],"industry":["Other"],"timeline":["Earlier"]},
            "decision":"Pending","premium":0,"risk_level":"Medium",
            "case_summary":"Auto-tagging failed. Placeholder values used.",
            "key_insights":"Please re-run auto-tagging if needed."
        }
    data.setdefault("tags", {})
    data["tags"].setdefault("equipment", ["Other"])
    data["tags"].setdefault("industry", ["Other"])
    data["tags"].setdefault("timeline", ["Earlier"])
    data.setdefault("decision","Pending")
    data.setdefault("premium",0)
    data.setdefault("risk_level","Medium")
    data.setdefault("case_summary","")
    data.setdefault("key_insights","")
    return data

# ============================================================================
# KNOWLEDGE BASE (统一知识库，无Workspace)
# ============================================================================

class KnowledgeBase:
    def __init__(self):
        self.metadata = self._load_metadata()
        self.embeddings = self._load_embeddings()
    
    def _load_metadata(self):
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_metadata(self):
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def _load_embeddings(self):
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_embeddings(self):
        with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.embeddings, f, indent=2)
    
    def add_document(self, uploaded_file, tags: Dict[str, List[str]], 
                     case_summary: str, key_insights: str,
                     decision: str, premium: int, risk_level: str,
                     extracted_text_preview: str = "") -> Dict[str, Any]:
        doc_id = f"DOC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(uploaded_file.name.encode()).hexdigest()[:6].upper()}"
        ext = uploaded_file.name.split('.')[-1].lower()
        filename = f"{doc_id}.{ext}"
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        full_text = f"{case_summary} {key_insights} {extracted_text_preview[:1000]}"
        embedding = generate_embedding(full_text)
        
        doc_meta = {
            "doc_id": doc_id,
            "filename": uploaded_file.name,
            "file_format": ext,
            "file_path": file_path,
            "file_size_kb": uploaded_file.size/1024,
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
        
        for fn in os.listdir(DOCUMENTS_DIR):
            if fn.startswith(doc_id):
                os.remove(os.path.join(DOCUMENTS_DIR, fn))
        
        self._save_metadata()
        self._save_embeddings()
    
    def get_stats(self):
        return {
            "total_documents": len(self.metadata),
            "total_size_mb": sum(d["file_size_kb"] for d in self.metadata)/1024 if self.metadata else 0.0,
            "format_distribution": self._get_fmt_dist(),
            "decision_distribution": self._get_decision_dist()
        }
    
    def _get_fmt_dist(self):
        dist = {}
        for d in self.metadata:
            dist[d["file_format"]] = dist.get(d["file_format"], 0)+1
        return dist
    
    def _get_decision_dist(self):
        dist = {}
        for d in self.metadata:
            dist[d["decision"]] = dist.get(d["decision"], 0)+1
        return dist

# ============================================================================
# CONVERSATION MANAGEMENT (会话管理)
# ============================================================================

class ConversationManager:
    """管理多个聊天会话"""
    
    def __init__(self):
        self.conversations_file = os.path.join(CONVERSATIONS_DIR, "conversations.json")
        self.conversations = self._load_conversations()
    
    def _load_conversations(self) -> Dict[str, Dict]:
        if os.path.exists(self.conversations_file):
            with open(self.conversations_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_conversations(self):
        with open(self.conversations_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, ensure_ascii=False, indent=2)
    
    def create_conversation(self, title: str = None) -> str:
        """创建新会话"""
        conv_id = f"CONV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if not title:
            title = f"New Chat {datetime.now().strftime('%m/%d %H:%M')}"
        
        self.conversations[conv_id] = {
            "id": conv_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": []
        }
        self._save_conversations()
        return conv_id
    
    def get_conversation(self, conv_id: str) -> Dict:
        """获取会话"""
        return self.conversations.get(conv_id, None)
    
    def update_conversation_title(self, conv_id: str, title: str):
        """更新会话标题"""
        if conv_id in self.conversations:
            self.conversations[conv_id]["title"] = title
            self.conversations[conv_id]["updated_at"] = datetime.now().isoformat()
            self._save_conversations()
    
    def add_message(self, conv_id: str, role: str, content: str):
        """添加消息到会话"""
        if conv_id in self.conversations:
            self.conversations[conv_id]["messages"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            self.conversations[conv_id]["updated_at"] = datetime.now().isoformat()
            
            # 自动更新标题（使用第一条用户消息的前30个字符）
            if role == "user" and len(self.conversations[conv_id]["messages"]) == 1:
                auto_title = content[:30] + ("..." if len(content) > 30 else "")
                self.conversations[conv_id]["title"] = auto_title
            
            self._save_conversations()
    
    def delete_conversation(self, conv_id: str):
        """删除会话"""
        if conv_id in self.conversations:
            del self.conversations[conv_id]
            self._save_conversations()
    
    def get_all_conversations(self) -> List[Dict]:
        """获取所有会话（按更新时间倒序）"""
        convs = list(self.conversations.values())
        convs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return convs
    
    def clear_conversation(self, conv_id: str):
        """清空会话消息"""
        if conv_id in self.conversations:
            self.conversations[conv_id]["messages"] = []
            self.conversations[conv_id]["updated_at"] = datetime.now().isoformat()
            self._save_conversations()

# ============================================================================
# CHAT
# ============================================================================

def generate_cot_response(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    if not retrieved_docs:
        return "⚠️ **No Relevant Cases Found**\n\nPlease add documents to the knowledge base or try a different query."
    
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
        {"role":"system","content":SYSTEM_INSTRUCTION},
        {"role":"user","content":f"""Query: "{query}"

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
# UI
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
        .workspace-card { background: var(--card-bg); padding: 1.0rem; border-radius: 0.5rem; box-shadow: var(--shadow); color: var(--text-primary); }
        .tag-badge { display:inline-block; padding:0.25rem 0.75rem; margin:0.25rem 0.4rem 0.25rem 0; border-radius:1rem; font-size:0.875rem; font-weight:700; color:#0b1220; }
        .tag-equipment { background-color: #93c5fd; }
        .tag-industry  { background-color: #86efac; }
        .tag-timeline  { background-color: #fde68a; }
        .conv-item { padding: 0.5rem; margin: 0.25rem 0; border-radius: 0.375rem; cursor: pointer; background: var(--card-bg); }
        .conv-item:hover { background: #1e293b; }
        .conv-item-active { background: #1e40af !important; }
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
        .workspace-card { background: var(--card-bg); padding: 1.0rem; border-radius: 0.5rem; box-shadow: var(--shadow); color: var(--text-primary); }
        .tag-badge { display:inline-block; padding:0.25rem 0.75rem; margin:0.25rem 0.4rem 0.25rem 0; border-radius:1rem; font-size:0.875rem; font-weight:700; color: var(--text-primary); }
        .tag-equipment { background-color: #dbeafe; }
        .tag-industry  { background-color: #dcfce7; }
        .tag-timeline  { background-color: #fef3c7; }
        .conv-item { padding: 0.5rem; margin: 0.25rem 0; border-radius: 0.375rem; cursor: pointer; background: var(--card-bg); border: 1px solid #e5e7eb; }
        .conv-item:hover { background: #f3f4f6; }
        .conv-item-active { background: #dbeafe !important; border-color: #3b82f6 !important; }
        .stChatMessage, .stMarkdown, p, li, label, span, div { color: var(--text-primary); }
        [data-testid="stMetricDelta"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--text-primary) !important; }
        #MainMenu, footer, header {visibility: hidden;}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Underwriting Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    kb = KnowledgeBase()
    conv_mgr = ConversationManager()
    
    # Session state initialization
    if "current_conv_id" not in st.session_state:
        # 创建默认会话或使用最新的会话
        all_convs = conv_mgr.get_all_conversations()
        if all_convs:
            st.session_state.current_conv_id = all_convs[0]["id"]
        else:
            st.session_state.current_conv_id = conv_mgr.create_conversation()
    
    # Appearance toggle
    with st.sidebar:
        st.markdown("### 🎨 Appearance")
        appearance = st.radio("Theme", ["Light", "Dark"], horizontal=True, key="appearance_choice")
    
    inject_css(appearance)
    
    # Title
    st.markdown('<div class="main-header">🤖 Underwriting Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">RAG + CoT | Unified Knowledge Base | Multi-Chat Sessions</div>', unsafe_allow_html=True)
    
    # Sidebar: Conversation Management
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 💬 Chat Sessions")
        
        # New Chat Button
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("➕ New Chat", use_container_width=True):
                new_conv_id = conv_mgr.create_conversation()
                st.session_state.current_conv_id = new_conv_id
                st.rerun()
        with col2:
            # Refresh button
            if st.button("🔄", use_container_width=True):
                st.rerun()
        
        # Conversation List
        all_convs = conv_mgr.get_all_conversations()
        
        if all_convs:
            st.markdown("---")
            for conv in all_convs:
                conv_id = conv["id"]
                is_active = (conv_id == st.session_state.current_conv_id)
                
                # Conversation item
                conv_class = "conv-item conv-item-active" if is_active else "conv-item"
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        f"{'📌' if is_active else '💬'} {conv['title'][:25]}...",
                        key=f"conv_{conv_id}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        st.session_state.current_conv_id = conv_id
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"del_{conv_id}", use_container_width=True):
                        conv_mgr.delete_conversation(conv_id)
                        # Switch to another conversation or create new one
                        remaining = conv_mgr.get_all_conversations()
                        if remaining:
                            st.session_state.current_conv_id = remaining[0]["id"]
                        else:
                            st.session_state.current_conv_id = conv_mgr.create_conversation()
                        st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Knowledge Base Stats")
        stats = kb.get_stats()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Documents", stats["total_documents"])
        with c2:
            st.metric("Size", f"{stats['total_size_mb']:.1f} MB")
        
        if stats["format_distribution"]:
            st.markdown("**Formats:**")
            for fmt, count in stats["format_distribution"].items():
                st.write(f"{SUPPORTED_FORMATS.get(fmt, fmt)}: {count}")
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Knowledge Base", "📤 Upload"])
    
    # ===== TAB 1: CHAT =====
    with tab1:
        current_conv = conv_mgr.get_conversation(st.session_state.current_conv_id)
        
        if not current_conv:
            st.error("Conversation not found. Creating new one...")
            st.session_state.current_conv_id = conv_mgr.create_conversation()
            st.rerun()
        
        st.markdown(f"### 💬 {current_conv['title']}")
        
        # Rename conversation
        with st.expander("✏️ Rename Chat"):
            new_title = st.text_input("New title", value=current_conv['title'])
            if st.button("Update Title"):
                conv_mgr.update_conversation_title(st.session_state.current_conv_id, new_title)
                st.success("Title updated!")
                st.rerun()
        
        # Display messages
        for msg in current_conv["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about underwriting cases..."):
            # Add user message
            conv_mgr.add_message(st.session_state.current_conv_id, "user", prompt)
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching knowledge base..."):
                    retrieved = kb.search_documents(prompt, top_k=3)
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
            
            # Add assistant message
            conv_mgr.add_message(st.session_state.current_conv_id, "assistant", resp)
            st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            if st.checkbox("Confirm clear all messages in this chat"):
                conv_mgr.clear_conversation(st.session_state.current_conv_id)
                st.success("Chat cleared!")
                st.rerun()
    
    # ===== TAB 2: KNOWLEDGE BASE =====
    with tab2:
        st.markdown("### 📄 Knowledge Base")
        
        if not kb.metadata:
            st.info("No documents yet. Upload in 'Upload' tab.")
        else:
            left, right = st.columns([1, 2.2])
            
            with left:
                st.markdown("#### 📚 Documents Browser")
                q = st.text_input("Search title/tags...", key="kb_search")
                fe = st.multiselect("🔧 Equipment", TAG_OPTIONS["equipment"])
                fi = st.multiselect("🏭 Industry", TAG_OPTIONS["industry"])
                ft = st.multiselect("📅 Timeline", TAG_OPTIONS["timeline"])
                
                docs = kb.metadata
                if q:
                    ql = q.lower()
                    docs = [d for d in docs if (ql in d["filename"].lower() or 
                            any(ql in tag.lower() for v in d["tags"].values() for tag in v))]
                if fe:
                    docs = [d for d in docs if any(t in d["tags"].get("equipment", []) for t in fe)]
                if fi:
                    docs = [d for d in docs if any(t in d["tags"].get("industry", []) for t in fi)]
                if ft:
                    docs = [d for d in docs if any(t in d["tags"].get("timeline", []) for t in ft)]
                
                docs = sorted(docs, key=lambda d: d.get("upload_date",""), reverse=True)
                
                options = {f"{SUPPORTED_FORMATS.get(d['file_format'],'📎')} {d['filename']} [{d['doc_id']}]": d["doc_id"] 
                          for d in docs}
                selected_id = st.radio("Documents", list(options.keys()), 
                                      index=0 if options else None, key="kb_selected")
                selected_doc = None
                if selected_id:
                    sel_id = options[selected_id]
                    selected_doc = next((d for d in docs if d["doc_id"] == sel_id), None)
                
                if selected_doc and st.button("🗑️ Delete Selected"):
                    kb.delete_document(selected_doc["doc_id"])
                    st.success("Document deleted!")
                    st.rerun()
            
            with right:
                st.markdown("#### 👀 Preview Original")
                if not selected_doc:
                    st.info("Select a document on the left to preview.")
                else:
                    doc = selected_doc
                    st.markdown(f"**{doc['filename']}**  \nID: `{doc['doc_id']}` | "
                              f"Format: **{doc['file_format'].upper()}** | "
                              f"Size: {doc['file_size_kb']:.1f} KB")
                    
                    with open(doc["file_path"], "rb") as f:
                        st.download_button("⬇️ Download", f, file_name=doc["filename"])
                    
                    ext = doc["file_format"]
                    path = doc["file_path"]
                    
                    if ext == "pdf":
                        try:
                            data_uri = file_to_data_uri(path, "application/pdf")
                            html = f'<iframe src="{data_uri}" width="100%" height="800px" style="border:none;"></iframe>'
                            st.components.v1.html(html, height=820, scrolling=True)
                        except Exception as e:
                            st.error(f"PDF preview failed: {e}")
                    elif ext in ["png","jpg","jpeg"]:
                        try:
                            st.image(path, use_column_width=True)
                        except Exception as e:
                            st.error(f"Image preview failed: {e}")
                    elif ext in ["docx","doc"]:
                        text = extract_text_from_docx(path) if ext == "docx" else "(DOC preview not supported)"
                        st.text_area("Extracted Text", value=text[:8000], height=400)
                    elif ext == "txt":
                        text = extract_text_from_txt(path)
                        st.text_area("Text File", value=text[:8000], height=400)
                    elif ext in ["xlsx","xls"]:
                        try:
                            df = pd.read_excel(path)
                            st.dataframe(df.head(200), use_container_width=True)
                        except Exception as e:
                            st.error(f"Excel preview failed: {e}")
                    else:
                        st.info("Preview not supported. Please download to view.")
                    
                    st.markdown("---")
                    st.markdown("**Tags & Case Info**")
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
    
    # ===== TAB 3: UPLOAD =====
    with tab3:
        st.markdown("### 📤 Upload Document (Auto-Tag)")
        st.caption("Upload files to the unified knowledge base. AI will automatically extract and tag.")
        
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
                    temp_id = f"TEMP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    ext = uploaded_file.name.split('.')[-1].lower()
                    temp_path = os.path.join(DOCUMENTS_DIR, f"{temp_id}.{ext}")
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    extracted_text = extract_text_from_file(temp_path, ext)
                    auto = auto_annotate_by_llm(extracted_text, uploaded_file.name)
                    
                    doc = kb.add_document(
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
                    
                    st.success(f"✅ Document uploaded: {doc['doc_id']}")
                    with st.expander("🔎 Auto-Tag Result"):
                        st.json(auto)

if __name__ == "__main__":
    main()
