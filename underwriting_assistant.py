"""
Underwriting Assistant - ChatGPT-style Layout
专业承保助手 - ChatGPT风格布局

Updates (2025-11-19):
- ✅ 左侧导航栏：Chats、New Chat、会话列表
- ✅ 右侧对话区域：完整对话 + 输入框
- ✅ 无Workspace，统一知识库
- ✅ 多会话管理，自动保存
- ✅ 外观切换、自动标签等功能保留
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

DEEPSEEK_API_KEY = "sk-99bba2ce117444e197270f17d303e74f"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

DATA_DIR = "data"
KNOWLEDGE_BASE_DIR = os.path.join(DATA_DIR, "knowledge_base")
DOCUMENTS_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "documents")
METADATA_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "metadata.json")
EMBEDDINGS_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "embeddings.json")
CONVERSATIONS_DIR = os.path.join(DATA_DIR, "conversations")

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

SUPPORTED_FORMATS = {
    "pdf": "📄", "docx": "📝", "doc": "📝", "txt": "📃",
    "xlsx": "📊", "xls": "📊", "png": "🖼️", "jpg": "🖼️", "jpeg": "🖼️"
}

TAG_OPTIONS = {
    "equipment": ["Gas Turbine", "Steam Turbine", "Boiler", "Generator", "Compressor", 
                  "Heat Exchanger", "Pump", "Transformer", "Motor", "Other"],
    "industry": ["Oil & Gas", "Power Generation", "Manufacturing", "Chemical", 
                 "Mining", "Refining", "Marine", "Aviation", "Other"],
    "timeline": ["2025-Q4", "2025-Q3", "2025-Q2", "2025-Q1", "2024", "2023", "Earlier"]
}

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
            return " ".join([page.extract_text() or "" for page in pdf_reader.pages])
    except:
        return ""

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except:
        return ""

def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""

def extract_text_from_file(file_path: str, file_format: str) -> str:
    if file_format == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_format in ["docx", "doc"]:
        return extract_text_from_docx(file_path)
    elif file_format == "txt":
        return extract_text_from_txt(file_path)
    return ""

def generate_embedding(text: str) -> List[float]:
    text_hash = hashlib.md5((text or "").encode()).hexdigest()
    fake = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
    return (fake + [0.0] * 1536)[:1536]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a*b for a,b in zip(v1, v2))
    m1 = sum(a*a for a in v1) ** 0.5
    m2 = sum(b*b for b in v2) ** 0.5
    return dot / (m1 * m2) if m1 and m2 else 0.0

def file_to_data_uri(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

# ============================================================================
# AUTO ANNOTATION
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
Rules: Return ONLY valid JSON. No commentary."""

def auto_annotate_by_llm(extracted_text: str, filename: str) -> Dict[str, Any]:
    user_prompt = f"FILENAME: {filename}\nTEXT:\n{(extracted_text or '')[:4000]}"
    content = call_deepseek_api(
        messages=[{"role":"system","content":AUTO_ANNOTATE_SYSTEM},
                  {"role":"user","content":user_prompt}],
        temperature=0.2, max_tokens=700
    )
    try:
        cleaned = content.strip().strip("`").split("\n", 1)[-1].rsplit("```", 1)[0] if "```" in content else content
        data = json.loads(cleaned.strip())
    except:
        data = {
            "tags":{"equipment":["Other"],"industry":["Other"],"timeline":["Earlier"]},
            "decision":"Pending","premium":0,"risk_level":"Medium",
            "case_summary":"Auto-tagging failed.","key_insights":"Please re-run."
        }
    data.setdefault("tags", {})
    for k in ["equipment","industry","timeline"]:
        data["tags"].setdefault(k, ["Other" if k!="timeline" else "Earlier"])
    for k,v in [("decision","Pending"),("premium",0),("risk_level","Medium"),("case_summary",""),("key_insights","")]:
        data.setdefault(k,v)
    return data

# ============================================================================
# KNOWLEDGE BASE
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
            with open(EMBEDDINGS_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_embeddings(self):
        with open(EMBEDDINGS_FILE, 'w') as f:
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
            "doc_id": doc_id, "filename": uploaded_file.name, "file_format": ext,
            "file_path": file_path, "file_size_kb": uploaded_file.size/1024,
            "upload_date": datetime.now().isoformat(), "tags": tags,
            "decision": decision, "premium": premium, "risk_level": risk_level,
            "case_summary": case_summary, "key_insights": key_insights,
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
        }

# ============================================================================
# CONVERSATION MANAGEMENT
# ============================================================================

class ConversationManager:
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
        conv_id = f"CONV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if not title:
            title = f"New Chat"
        self.conversations[conv_id] = {
            "id": conv_id, "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": []
        }
        self._save_conversations()
        return conv_id
    
    def add_message(self, conv_id: str, role: str, content: str):
        if conv_id in self.conversations:
            self.conversations[conv_id]["messages"].append({
                "role": role, "content": content, "timestamp": datetime.now().isoformat()
            })
            self.conversations[conv_id]["updated_at"] = datetime.now().isoformat()
            if role == "user" and len(self.conversations[conv_id]["messages"]) == 1:
                auto_title = content[:35] + ("..." if len(content) > 35 else "")
                self.conversations[conv_id]["title"] = auto_title
            self._save_conversations()
    
    def delete_conversation(self, conv_id: str):
        if conv_id in self.conversations:
            del self.conversations[conv_id]
            self._save_conversations()
    
    def get_conversation(self, conv_id: str) -> Dict:
        return self.conversations.get(conv_id, None)
    
    def get_all_conversations(self) -> List[Dict]:
        convs = list(self.conversations.values())
        convs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return convs
    
    def update_conversation_title(self, conv_id: str, title: str):
        if conv_id in self.conversations:
            self.conversations[conv_id]["title"] = title
            self.conversations[conv_id]["updated_at"] = datetime.now().isoformat()
            self._save_conversations()
    
    def clear_conversation(self, conv_id: str):
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
Decision: {doc["decision"]} | Premium: ${doc["premium"]:,} | Risk: {doc["risk_level"]}

Summary: {doc["case_summary"]}
Insights: {doc["key_insights"]}
"""
    
    messages = [
        {"role":"system","content":SYSTEM_INSTRUCTION},
        {"role":"user","content":f'Query: "{query}"\n\nRetrieved Cases:\n{docs_text}\n\nAnalyze using 5-step CoT framework and provide: Decision + Premium Range + Sources'}
    ]
    return call_deepseek_api(messages)

# ============================================================================
# CSS
# ============================================================================

def inject_css(theme: str):
    if theme == "Dark":
        css = """
        <style>
        /* Dark Theme */
        :root {
            --bg-primary: #0b1220;
            --bg-secondary: #1a2332;
            --bg-tertiary: #243447;
            --text-primary: #e5e7eb;
            --text-secondary: #9ca3af;
            --border: #374151;
            --active-bg: #1e40af;
            --hover-bg: #1e293b;
        }
        
        /* Hide Streamlit defaults */
        #MainMenu, footer, header {visibility: hidden;}
        .stApp {background: var(--bg-primary); color: var(--text-primary);}
        
        /* Navigation Bar */
        .nav-container {
            position: fixed;
            left: 0;
            top: 0;
            width: 280px;
            height: 100vh;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 1rem;
            z-index: 1000;
        }
        
        .nav-header {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }
        
        .nav-btn {
            width: 100%;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            color: var(--text-primary);
            cursor: pointer;
            text-align: left;
            font-size: 0.95rem;
            transition: all 0.2s;
        }
        
        .nav-btn:hover {
            background: var(--hover-bg);
            border-color: var(--active-bg);
        }
        
        .nav-btn-active {
            background: var(--active-bg) !important;
            border-color: var(--active-bg) !important;
        }
        
        .chat-list {
            flex: 1;
            overflow-y: auto;
            margin-top: 1rem;
        }
        
        .chat-item {
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .chat-item:hover {
            background: var(--hover-bg);
        }
        
        .chat-item-active {
            background: var(--active-bg) !important;
            border-color: var(--active-bg) !important;
        }
        
        .chat-title {
            flex: 1;
            color: var(--text-primary);
            font-size: 0.9rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .chat-delete {
            color: var(--text-secondary);
            font-size: 1.1rem;
            cursor: pointer;
            padding: 0.25rem;
        }
        
        .chat-delete:hover {
            color: #ef4444;
        }
        
        /* Main Content Area */
        .main-content {
            margin-left: 300px;
            padding: 2rem;
            min-height: 100vh;
        }
        
        /* Chat Messages */
        .chat-container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .message-user {
            background: var(--bg-tertiary);
            padding: 1rem;
            border-radius: 0.75rem;
            margin-bottom: 1rem;
            color: var(--text-primary);
        }
        
        .message-assistant {
            background: var(--bg-secondary);
            padding: 1rem;
            border-radius: 0.75rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border);
            color: var(--text-primary);
        }
        
        /* Tags */
        .tag-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            margin: 0.25rem 0.25rem 0.25rem 0;
            border-radius: 1rem;
            font-size: 0.875rem;
            font-weight: 600;
        }
        .tag-equipment { background: #3b82f6; color: white; }
        .tag-industry { background: #10b981; color: white; }
        .tag-timeline { background: #f59e0b; color: white; }
        
        /* Override Streamlit styles */
        .stChatMessage, .stMarkdown, p, li, label, span, div {
            color: var(--text-primary) !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            color: var(--text-primary) !important;
        }
        </style>
        """
    else:
        css = """
        <style>
        /* Light Theme */
        :root {
            --bg-primary: #f9fafb;
            --bg-secondary: #ffffff;
            --bg-tertiary: #f3f4f6;
            --text-primary: #111827;
            --text-secondary: #6b7280;
            --border: #e5e7eb;
            --active-bg: #3b82f6;
            --hover-bg: #f3f4f6;
        }
        
        #MainMenu, footer, header {visibility: hidden;}
        .stApp {background: var(--bg-primary); color: var(--text-primary);}
        
        .nav-container {
            position: fixed;
            left: 0;
            top: 0;
            width: 280px;
            height: 100vh;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 1rem;
            z-index: 1000;
        }
        
        .nav-header {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }
        
        .nav-btn {
            width: 100%;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            color: var(--text-primary);
            cursor: pointer;
            text-align: left;
            font-size: 0.95rem;
            transition: all 0.2s;
        }
        
        .nav-btn:hover {
            background: var(--hover-bg);
            border-color: var(--active-bg);
        }
        
        .nav-btn-active {
            background: var(--active-bg) !important;
            border-color: var(--active-bg) !important;
            color: white !important;
        }
        
        .chat-list {
            flex: 1;
            overflow-y: auto;
            margin-top: 1rem;
        }
        
        .chat-item {
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .chat-item:hover {
            background: var(--hover-bg);
            border-color: var(--active-bg);
        }
        
        .chat-item-active {
            background: var(--active-bg) !important;
            border-color: var(--active-bg) !important;
        }
        
        .chat-title {
            flex: 1;
            color: var(--text-primary);
            font-size: 0.9rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .chat-item-active .chat-title {
            color: white !important;
        }
        
        .chat-delete {
            color: var(--text-secondary);
            font-size: 1.1rem;
            cursor: pointer;
            padding: 0.25rem;
        }
        
        .chat-delete:hover {
            color: #ef4444;
        }
        
        .main-content {
            margin-left: 300px;
            padding: 2rem;
            min-height: 100vh;
        }
        
        .chat-container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .message-user {
            background: var(--bg-tertiary);
            padding: 1rem;
            border-radius: 0.75rem;
            margin-bottom: 1rem;
            color: var(--text-primary);
        }
        
        .message-assistant {
            background: var(--bg-secondary);
            padding: 1rem;
            border-radius: 0.75rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border);
            color: var(--text-primary);
        }
        
        .tag-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            margin: 0.25rem 0.25rem 0.25rem 0;
            border-radius: 1rem;
            font-size: 0.875rem;
            font-weight: 600;
        }
        .tag-equipment { background: #dbeafe; color: #1e40af; }
        .tag-industry { background: #d1fae5; color: #065f46; }
        .tag-timeline { background: #fef3c7; color: #92400e; }
        
        .stChatMessage, .stMarkdown, p, li, label, span, div {
            color: var(--text-primary) !important;
        }
        
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            color: var(--text-primary) !important;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Underwriting Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize
    kb = KnowledgeBase()
    conv_mgr = ConversationManager()
    
    # Session state
    if "current_conv_id" not in st.session_state:
        all_convs = conv_mgr.get_all_conversations()
        if all_convs:
            st.session_state.current_conv_id = all_convs[0]["id"]
        else:
            st.session_state.current_conv_id = conv_mgr.create_conversation()
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"
    
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"
    
    # Apply CSS
    inject_css(st.session_state.theme)
    
    # Left Navigation Bar (using columns to simulate fixed sidebar)
    all_convs = conv_mgr.get_all_conversations()
    
    col_nav, col_main = st.columns([1, 4])
    
    with col_nav:
        st.markdown('<div class="nav-header">🤖 Underwriting AI</div>', unsafe_allow_html=True)
        
        # Theme toggle
        if st.button("🎨 " + st.session_state.theme, use_container_width=True):
            st.session_state.theme = "Dark" if st.session_state.theme == "Light" else "Light"
            st.rerun()
        
        st.markdown("---")
        
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            new_id = conv_mgr.create_conversation()
            st.session_state.current_conv_id = new_id
            st.session_state.current_page = "chat"
            st.rerun()
        
        # Navigation Buttons
        if st.button("💬 Chats", use_container_width=True, 
                    type="primary" if st.session_state.current_page == "chat" else "secondary"):
            st.session_state.current_page = "chat"
            st.rerun()
        
        if st.button("📄 Knowledge Base", use_container_width=True,
                    type="primary" if st.session_state.current_page == "kb" else "secondary"):
            st.session_state.current_page = "kb"
            st.rerun()
        
        if st.button("📤 Upload", use_container_width=True,
                    type="primary" if st.session_state.current_page == "upload" else "secondary"):
            st.session_state.current_page = "upload"
            st.rerun()
        
        st.markdown("---")
        st.markdown("#### Recent Chats")
        
        # Chat List
        for conv in all_convs[:15]:  # Show max 15 recent chats
            conv_id = conv["id"]
            is_active = (conv_id == st.session_state.current_conv_id)
            
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(
                    f"{'📌' if is_active else '💬'} {conv['title'][:28]}",
                    key=f"nav_conv_{conv_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.current_conv_id = conv_id
                    st.session_state.current_page = "chat"
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"nav_del_{conv_id}", use_container_width=True):
                    conv_mgr.delete_conversation(conv_id)
                    remaining = conv_mgr.get_all_conversations()
                    if remaining:
                        st.session_state.current_conv_id = remaining[0]["id"]
                    else:
                        st.session_state.current_conv_id = conv_mgr.create_conversation()
                    st.rerun()
        
        # Stats at bottom
        st.markdown("---")
        stats = kb.get_stats()
        st.metric("📚 Documents", stats["total_documents"])
        st.caption(f"Size: {stats['total_size_mb']:.1f} MB")
    
    # Main Content Area
    with col_main:
        if st.session_state.current_page == "chat":
            render_chat_page(kb, conv_mgr)
        elif st.session_state.current_page == "kb":
            render_kb_page(kb)
        elif st.session_state.current_page == "upload":
            render_upload_page(kb)

def render_chat_page(kb: KnowledgeBase, conv_mgr: ConversationManager):
    current_conv = conv_mgr.get_conversation(st.session_state.current_conv_id)
    
    if not current_conv:
        st.error("Conversation not found")
        return
    
    st.title(f"💬 {current_conv['title']}")
    
    # Rename option
    with st.expander("✏️ Rename Chat"):
        new_title = st.text_input("New title", value=current_conv['title'], key="rename_input")
        if st.button("Update Title"):
            conv_mgr.update_conversation_title(st.session_state.current_conv_id, new_title)
            st.success("Updated!")
            st.rerun()
    
    # Display messages
    for msg in current_conv["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about underwriting cases..."):
        conv_mgr.add_message(st.session_state.current_conv_id, "user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching..."):
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
        
        conv_mgr.add_message(st.session_state.current_conv_id, "assistant", resp)
        st.rerun()
    
    # Clear chat
    if st.button("🗑️ Clear Chat History"):
        if st.checkbox("Confirm"):
            conv_mgr.clear_conversation(st.session_state.current_conv_id)
            st.success("Cleared!")
            st.rerun()

def render_kb_page(kb: KnowledgeBase):
    st.title("📄 Knowledge Base")
    
    if not kb.metadata:
        st.info("No documents. Upload in Upload tab.")
        return
    
    # Search and filters
    q = st.text_input("🔍 Search", placeholder="Search documents...")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fe = st.multiselect("🔧 Equipment", TAG_OPTIONS["equipment"])
    with col2:
        fi = st.multiselect("🏭 Industry", TAG_OPTIONS["industry"])
    with col3:
        ft = st.multiselect("📅 Timeline", TAG_OPTIONS["timeline"])
    
    # Filter docs
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
    
    st.markdown(f"**{len(docs)} documents found**")
    
    # Display docs
    for doc in docs:
        with st.expander(f"{SUPPORTED_FORMATS.get(doc['file_format'],'📎')} {doc['filename']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**ID:** `{doc['doc_id']}`")
                st.markdown(f"**Decision:** {doc['decision']} | **Premium:** ${doc['premium']:,} | **Risk:** {doc['risk_level']}")
                
                tags_html = ""
                for t in doc["tags"].get("equipment", []):
                    tags_html += f'<span class="tag-badge tag-equipment">🔧 {t}</span>'
                for t in doc["tags"].get("industry", []):
                    tags_html += f'<span class="tag-badge tag-industry">🏭 {t}</span>'
                for t in doc["tags"].get("timeline", []):
                    tags_html += f'<span class="tag-badge tag-timeline">📅 {t}</span>'
                st.markdown(tags_html, unsafe_allow_html=True)
                
                st.write("**Summary:**")
                st.info(doc["case_summary"])
            
            with col2:
                with open(doc["file_path"], "rb") as f:
                    st.download_button("⬇️ Download", f, file_name=doc["filename"])
                
                if st.button("🗑️ Delete", key=f"del_{doc['doc_id']}"):
                    kb.delete_document(doc["doc_id"])
                    st.success("Deleted!")
                    st.rerun()

def render_upload_page(kb: KnowledgeBase):
    st.title("📤 Upload Document")
    
    with st.form("upload_form"):
        uploaded_file = st.file_uploader(
            "Choose document",
            type=list(SUPPORTED_FORMATS.keys())
        )
        submitted = st.form_submit_button("📤 Upload & Auto-Tag", use_container_width=True)
    
    if submitted and uploaded_file:
        with st.spinner("Processing..."):
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
            except:
                pass
            
            st.success(f"✅ Uploaded: {doc['doc_id']}")
            with st.expander("🔎 Auto-Tag Result"):
                st.json(auto)

if __name__ == "__main__":
    main()
