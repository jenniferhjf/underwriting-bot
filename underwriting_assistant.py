"""
Underwriting Assistant - ChatGPT-style Layout with Batch Upload
专业承保助手 - 支持批量上传（多文件、ZIP、文件夹）

Updates (2025-11-19):
- ✅ 支持上传多个文件
- ✅ 支持上传ZIP文件（自动解压）
- ✅ 支持文件夹上传（通过ZIP）
- ✅ 批量自动标签
- ✅ 进度显示
- ✅ ChatGPT风格布局保留
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
import zipfile
import tempfile
import shutil

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
    "xlsx": "📊", "xls": "📊", "png": "🖼️", "jpg": "🖼️", 
    "jpeg": "🖼️", "zip": "📦"  # 添加ZIP支持
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
# ZIP HANDLING
# ============================================================================

def extract_files_from_zip(zip_file) -> List[Dict[str, Any]]:
    """从ZIP文件中提取所有支持的文件"""
    extracted_files = []
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 保存ZIP文件
        zip_path = os.path.join(temp_dir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
        
        # 解压
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # 遍历所有文件
        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                if filename.startswith('.') or filename.startswith('__'):
                    continue  # 跳过隐藏文件和系统文件
                
                file_path = os.path.join(root, filename)
                file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
                
                # 检查是否是支持的格式
                if file_ext in ['pdf', 'docx', 'doc', 'txt', 'xlsx', 'xls', 'png', 'jpg', 'jpeg']:
                    # 读取文件内容
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    
                    # 创建类似上传文件的对象
                    extracted_files.append({
                        'name': filename,
                        'content': file_content,
                        'size': len(file_content),
                        'type': file_ext
                    })
        
    except Exception as e:
        st.error(f"ZIP解压失败: {str(e)}")
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    return extracted_files

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
    
    def add_document_from_content(self, filename: str, file_content: bytes, file_size: int,
                                   tags: Dict[str, List[str]], case_summary: str, key_insights: str,
                                   decision: str, premium: int, risk_level: str,
                                   extracted_text_preview: str = "") -> Dict[str, Any]:
        """从内容添加文档（用于批量上传）"""
        doc_id = f"DOC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(filename.encode()).hexdigest()[:6].upper()}"
        ext = filename.split('.')[-1].lower()
        saved_filename = f"{doc_id}.{ext}"
        file_path = os.path.join(DOCUMENTS_DIR, saved_filename)
        
        # 保存文件
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        full_text = f"{case_summary} {key_insights} {extracted_text_preview[:1000]}"
        embedding = generate_embedding(full_text)
        
        doc_meta = {
            "doc_id": doc_id, "filename": filename, "file_format": ext,
            "file_path": file_path, "file_size_kb": file_size/1024,
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
    
    def add_document(self, uploaded_file, tags: Dict[str, List[str]], 
                     case_summary: str, key_insights: str,
                     decision: str, premium: int, risk_level: str,
                     extracted_text_preview: str = "") -> Dict[str, Any]:
        """从上传文件添加文档"""
        return self.add_document_from_content(
            filename=uploaded_file.name,
            file_content=uploaded_file.getbuffer(),
            file_size=uploaded_file.size,
            tags=tags,
            case_summary=case_summary,
            key_insights=key_insights,
            decision=decision,
            premium=premium,
            risk_level=risk_level,
            extracted_text_preview=extracted_text_preview
        )
    
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
# CSS (同之前版本)
# ============================================================================

def inject_css(theme: str):
    if theme == "Dark":
        css = """
        <style>
        :root {
            --bg-primary: #0b1220; --bg-secondary: #1a2332; --bg-tertiary: #243447;
            --text-primary: #e5e7eb; --text-secondary: #9ca3af; --border: #374151;
            --active-bg: #1e40af; --hover-bg: #1e293b;
        }
        #MainMenu, footer, header {visibility: hidden;}
        .stApp {background: var(--bg-primary); color: var(--text-primary);}
        .tag-badge { display: inline-block; padding: 0.25rem 0.75rem; margin: 0.25rem; border-radius: 1rem; font-size: 0.875rem; font-weight: 600; }
        .tag-equipment { background: #3b82f6; color: white; }
        .tag-industry { background: #10b981; color: white; }
        .tag-timeline { background: #f59e0b; color: white; }
        .stChatMessage, .stMarkdown, p, li, label, span, div { color: var(--text-primary) !important; }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--text-primary) !important; }
        .upload-status { padding: 0.5rem; margin: 0.5rem 0; border-radius: 0.5rem; background: var(--bg-tertiary); }
        .upload-success { border-left: 3px solid #10b981; }
        .upload-error { border-left: 3px solid #ef4444; }
        .upload-processing { border-left: 3px solid #f59e0b; }
        </style>
        """
    else:
        css = """
        <style>
        :root {
            --bg-primary: #f9fafb; --bg-secondary: #ffffff; --bg-tertiary: #f3f4f6;
            --text-primary: #111827; --text-secondary: #6b7280; --border: #e5e7eb;
            --active-bg: #3b82f6; --hover-bg: #f3f4f6;
        }
        #MainMenu, footer, header {visibility: hidden;}
        .stApp {background: var(--bg-primary); color: var(--text-primary);}
        .tag-badge { display: inline-block; padding: 0.25rem 0.75rem; margin: 0.25rem; border-radius: 1rem; font-size: 0.875rem; font-weight: 600; }
        .tag-equipment { background: #dbeafe; color: #1e40af; }
        .tag-industry { background: #d1fae5; color: #065f46; }
        .tag-timeline { background: #fef3c7; color: #92400e; }
        .stChatMessage, .stMarkdown, p, li, label, span, div { color: var(--text-primary) !important; }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--text-primary) !important; }
        .upload-status { padding: 0.5rem; margin: 0.5rem 0; border-radius: 0.5rem; background: var(--bg-tertiary); }
        .upload-success { border-left: 3px solid #10b981; }
        .upload-error { border-left: 3px solid #ef4444; }
        .upload-processing { border-left: 3px solid #f59e0b; }
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
    
    kb = KnowledgeBase()
    conv_mgr = ConversationManager()
    
    if "current_conv_id" not in st.session_state:
        all_convs = conv_mgr.get_all_conversations()
        st.session_state.current_conv_id = all_convs[0]["id"] if all_convs else conv_mgr.create_conversation()
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"
    
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"
    
    inject_css(st.session_state.theme)
    
    all_convs = conv_mgr.get_all_conversations()
    col_nav, col_main = st.columns([1, 4])
    
    # Left Navigation (同之前版本，略)
    with col_nav:
        st.markdown("### 🤖 Underwriting AI")
        if st.button("🎨 " + st.session_state.theme, use_container_width=True):
            st.session_state.theme = "Dark" if st.session_state.theme == "Light" else "Light"
            st.rerun()
        st.markdown("---")
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            new_id = conv_mgr.create_conversation()
            st.session_state.current_conv_id = new_id
            st.session_state.current_page = "chat"
            st.rerun()
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
        for conv in all_convs[:15]:
            conv_id = conv["id"]
            is_active = (conv_id == st.session_state.current_conv_id)
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(f"{'📌' if is_active else '💬'} {conv['title'][:28]}",
                           key=f"nav_conv_{conv_id}", use_container_width=True,
                           type="primary" if is_active else "secondary"):
                    st.session_state.current_conv_id = conv_id
                    st.session_state.current_page = "chat"
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"nav_del_{conv_id}", use_container_width=True):
                    conv_mgr.delete_conversation(conv_id)
                    remaining = conv_mgr.get_all_conversations()
                    st.session_state.current_conv_id = remaining[0]["id"] if remaining else conv_mgr.create_conversation()
                    st.rerun()
        
        st.markdown("---")
        stats = kb.get_stats()
        st.metric("📚 Documents", stats["total_documents"])
        st.caption(f"Size: {stats['total_size_mb']:.1f} MB")
    
    # Main Content
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
    with st.expander("✏️ Rename Chat"):
        new_title = st.text_input("New title", value=current_conv['title'], key="rename_input")
        if st.button("Update Title"):
            conv_mgr.update_conversation_title(st.session_state.current_conv_id, new_title)
            st.success("Updated!")
            st.rerun()
    
    for msg in current_conv["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
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
    
    q = st.text_input("🔍 Search", placeholder="Search documents...")
    col1, col2, col3 = st.columns(3)
    with col1:
        fe = st.multiselect("🔧 Equipment", TAG_OPTIONS["equipment"])
    with col2:
        fi = st.multiselect("🏭 Industry", TAG_OPTIONS["industry"])
    with col3:
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
    st.markdown(f"**{len(docs)} documents found**")
    
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
    st.title("📤 Batch Upload Documents")
    st.markdown("**Upload multiple files, ZIP archives, or folders (as ZIP)**")
    
    # 上传方式选择
    upload_mode = st.radio(
        "Upload Mode",
        ["📄 Multiple Files", "📦 ZIP Archive (Auto-extract)", "📁 Folder (Upload as ZIP)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if upload_mode == "📄 Multiple Files":
        st.markdown("### Upload Multiple Files")
        st.caption("Select multiple files at once")
        
        uploaded_files = st.file_uploader(
            "Choose documents",
            type=['pdf', 'docx', 'doc', 'txt', 'xlsx', 'xls', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,  # 关键：允许多文件
            help="Hold Ctrl/Cmd to select multiple files"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files selected")
            
            # 显示文件列表
            with st.expander(f"📋 File List ({len(uploaded_files)} files)"):
                for i, f in enumerate(uploaded_files, 1):
                    st.write(f"{i}. {SUPPORTED_FORMATS.get(f.name.split('.')[-1].lower(), '📎')} {f.name} ({f.size/1024:.1f} KB)")
            
            if st.button("🚀 Process All Files", type="primary", use_container_width=True):
                process_batch_upload(kb, uploaded_files)
    
    elif upload_mode == "📦 ZIP Archive (Auto-extract)":
        st.markdown("### Upload ZIP Archive")
        st.caption("System will automatically extract and process all supported files")
        
        zip_file = st.file_uploader(
            "Choose ZIP file",
            type=['zip'],
            help="Upload a ZIP containing documents"
        )
        
        if zip_file:
            st.success(f"✅ ZIP file selected: {zip_file.name} ({zip_file.size/1024:.1f} KB)")
            
            if st.button("🚀 Extract & Process ZIP", type="primary", use_container_width=True):
                with st.spinner("📦 Extracting ZIP file..."):
                    extracted_files = extract_files_from_zip(zip_file)
                
                if extracted_files:
                    st.success(f"✅ Extracted {len(extracted_files)} files from ZIP")
                    
                    with st.expander(f"📋 Extracted Files ({len(extracted_files)} files)"):
                        for i, f in enumerate(extracted_files, 1):
                            st.write(f"{i}. {SUPPORTED_FORMATS.get(f['type'], '📎')} {f['name']} ({f['size']/1024:.1f} KB)")
                    
                    process_batch_upload_from_content(kb, extracted_files)
                else:
                    st.warning("No supported files found in ZIP")
    
    else:  # Folder as ZIP
        st.markdown("### Upload Folder as ZIP")
        st.info("""
        **How to upload a folder:**
        1. Compress your folder into a ZIP file
        2. Upload the ZIP file below
        3. System will extract and process all files
        
        **Example:**
        - Right-click folder → "Compress"
        - Or use command: `zip -r folder.zip your_folder/`
        """)
        
        folder_zip = st.file_uploader(
            "Choose folder (as ZIP)",
            type=['zip'],
            help="Upload your folder compressed as ZIP"
        )
        
        if folder_zip:
            st.success(f"✅ Folder ZIP selected: {folder_zip.name} ({folder_zip.size/1024:.1f} KB)")
            
            if st.button("🚀 Extract & Process Folder", type="primary", use_container_width=True):
                with st.spinner("📁 Extracting folder..."):
                    extracted_files = extract_files_from_zip(folder_zip)
                
                if extracted_files:
                    st.success(f"✅ Extracted {len(extracted_files)} files from folder")
                    
                    with st.expander(f"📋 Folder Contents ({len(extracted_files)} files)"):
                        for i, f in enumerate(extracted_files, 1):
                            st.write(f"{i}. {SUPPORTED_FORMATS.get(f['type'], '📎')} {f['name']} ({f['size']/1024:.1f} KB)")
                    
                    process_batch_upload_from_content(kb, extracted_files)
                else:
                    st.warning("No supported files found in folder")

def process_batch_upload(kb: KnowledgeBase, uploaded_files: List):
    """处理批量上传（从上传文件对象）"""
    st.markdown("---")
    st.markdown("### 🔄 Processing Files...")
    
    progress_bar = st.progress(0)
    status_container = st.container()
    
    success_count = 0
    error_count = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        
        with status_container:
            st.markdown(f"**Processing ({i+1}/{len(uploaded_files)}):** {uploaded_file.name}")
        
        try:
            # 保存临时文件
            ext = uploaded_file.name.split('.')[-1].lower()
            temp_path = os.path.join(DOCUMENTS_DIR, f"TEMP_{i}.{ext}")
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 提取文本
            extracted_text = extract_text_from_file(temp_path, ext)
            
            # 自动标签
            with st.spinner(f"🤖 Auto-tagging {uploaded_file.name}..."):
                auto = auto_annotate_by_llm(extracted_text, uploaded_file.name)
            
            # 添加到知识库
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
            
            # 清理临时文件
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
            
            with status_container:
                st.markdown(f'<div class="upload-status upload-success">✅ {uploaded_file.name} → {doc["doc_id"]}</div>', 
                          unsafe_allow_html=True)
            
            success_count += 1
            
        except Exception as e:
            with status_container:
                st.markdown(f'<div class="upload-status upload-error">❌ {uploaded_file.name}: {str(e)}</div>', 
                          unsafe_allow_html=True)
            error_count += 1
    
    progress_bar.progress(1.0)
    
    st.markdown("---")
    st.markdown("### 📊 Upload Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Success", success_count)
    with col2:
        st.metric("❌ Errors", error_count)
    with col3:
        st.metric("📊 Total", len(uploaded_files))
    
    if success_count > 0:
        st.balloons()

def process_batch_upload_from_content(kb: KnowledgeBase, extracted_files: List[Dict]):
    """处理批量上传（从提取的内容）"""
    st.markdown("---")
    st.markdown("### 🔄 Processing Files...")
    
    progress_bar = st.progress(0)
    status_container = st.container()
    
    success_count = 0
    error_count = 0
    
    for i, file_info in enumerate(extracted_files):
        progress = (i + 1) / len(extracted_files)
        progress_bar.progress(progress)
        
        with status_container:
            st.markdown(f"**Processing ({i+1}/{len(extracted_files)}):** {file_info['name']}")
        
        try:
            # 保存临时文件
            ext = file_info['type']
            temp_path = os.path.join(DOCUMENTS_DIR, f"TEMP_{i}.{ext}")
            
            with open(temp_path, "wb") as f:
                f.write(file_info['content'])
            
            # 提取文本
            extracted_text = extract_text_from_file(temp_path, ext)
            
            # 自动标签
            with st.spinner(f"🤖 Auto-tagging {file_info['name']}..."):
                auto = auto_annotate_by_llm(extracted_text, file_info['name'])
            
            # 添加到知识库
            doc = kb.add_document_from_content(
                filename=file_info['name'],
                file_content=file_info['content'],
                file_size=file_info['size'],
                tags=auto["tags"],
                case_summary=auto["case_summary"],
                key_insights=auto["key_insights"],
                decision=auto["decision"],
                premium=int(auto.get("premium", 0) or 0),
                risk_level=auto["risk_level"],
                extracted_text_preview=extracted_text[:800]
            )
            
            # 清理临时文件
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
            
            with status_container:
                st.markdown(f'<div class="upload-status upload-success">✅ {file_info["name"]} → {doc["doc_id"]}</div>', 
                          unsafe_allow_html=True)
            
            success_count += 1
            
        except Exception as e:
            with status_container:
                st.markdown(f'<div class="upload-status upload-error">❌ {file_info["name"]}: {str(e)}</div>', 
                          unsafe_allow_html=True)
            error_count += 1
    
    progress_bar.progress(1.0)
    
    st.markdown("---")
    st.markdown("### 📊 Upload Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Success", success_count)
    with col2:
        st.metric("❌ Errors", error_count)
    with col3:
        st.metric("📊 Total", len(extracted_files))
    
    if success_count > 0:
        st.balloons()

if __name__ == "__main__":
    main()
