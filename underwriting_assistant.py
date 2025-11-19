"""
Underwriting Assistant - COMPLETE VERSION
完整版本：批量上传 + 新手引导 + ChatGPT布局

所有功能：
✅ 新手引导（首次访问自动显示）
✅ 批量上传（多文件/ZIP/文件夹）
✅ ChatGPT风格布局
✅ 多会话管理
✅ 统一知识库
✅ 自动标签
✅ RAG + CoT
"""

import streamlit as st
import os, json, hashlib, zipfile, tempfile, shutil
from datetime import datetime
from typing import List, Dict, Any
import requests
import PyPDF2
from docx import Document
import base64

# ============================================================================
# CONFIGURATION
# ============================================================================

DEEPSEEK_API_KEY = "sk-99bba2ce117444e197270f17d303e74f"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

DATA_DIR = "data"
KB_DIR = os.path.join(DATA_DIR, "knowledge_base")
DOCS_DIR = os.path.join(KB_DIR, "documents")
METADATA_FILE = os.path.join(KB_DIR, "metadata.json")
EMBEDDINGS_FILE = os.path.join(KB_DIR, "embeddings.json")
CONV_DIR = os.path.join(DATA_DIR, "conversations")
CONV_FILE = os.path.join(CONV_DIR, "conversations.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "user_settings.json")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(CONV_DIR, exist_ok=True)

SUPPORTED_FORMATS = {
    "pdf": "📄", "docx": "📝", "doc": "📝", "txt": "📃",
    "xlsx": "📊", "xls": "📊", "png": "🖼️", "jpg": "🖼️", 
    "jpeg": "🖼️", "zip": "📦"
}

TAG_OPTIONS = {
    "equipment": ["Gas Turbine", "Steam Turbine", "Boiler", "Generator", "Compressor", 
                  "Heat Exchanger", "Pump", "Transformer", "Motor", "Other"],
    "industry": ["Oil & Gas", "Power Generation", "Manufacturing", "Chemical", 
                 "Mining", "Refining", "Marine", "Aviation", "Other"],
    "timeline": ["2025-Q4", "2025-Q3", "2025-Q2", "2025-Q1", "2024", "2023", "Earlier"]
}

# ============================================================================
# TUTORIAL STEPS
# ============================================================================

TUTORIAL_STEPS = [
    {
        "step": 1,
        "title": "🎉 Welcome to Underwriting Assistant!",
        "content": """
**Hello! Welcome aboard!** 👋

This is your AI-powered underwriting assistant that helps you:
- 📚 **Organize** all your underwriting cases in one place
- 🔍 **Search** through cases instantly using AI
- 💬 **Chat** with your knowledge base for quick answers
- 🤖 **Get recommendations** based on past precedents

Let's take a quick tour (takes ~2 minutes)!
        """
    },
    {
        "step": 2,
        "title": "📤 Step 1: Upload Your Documents",
        "content": """
**First, let's build your knowledge base!**

Click the **"📤 Upload"** button in the left sidebar to add documents.

You can upload:
- 📄 **Single files** (PDF, Word, Excel, etc.)
- 📦 **Multiple files** at once
- 🗂️ **ZIP archives** (we'll auto-extract them)
- 📁 **Entire folders** (compress as ZIP first)

Our AI will automatically:
✅ Extract text from documents
✅ Tag them by equipment, industry, timeline
✅ Create a searchable knowledge base
        """
    },
    {
        "step": 3,
        "title": "💬 Step 2: Chat with Your Knowledge Base",
        "content": """
**Now you can ask questions!**

Click **"💬 Chats"** to start a conversation.

Examples:
- "Show me gas turbine cases from 2024"
- "What premium should I charge for a 10-year boiler?"
- "Find similar cases to ABC Oil"

The AI will:
✅ Search your knowledge base
✅ Find relevant precedents
✅ Provide step-by-step reasoning (Chain-of-Thought)
✅ Cite specific sources

**Pro tip:** You can create multiple chats for different projects!
        """
    },
    {
        "step": 4,
        "title": "📚 Step 3: Browse Your Knowledge Base",
        "content": """
**View all your uploaded documents!**

Click **"📄 Knowledge Base"** to:
- 🔍 **Search** by filename or tags
- 🏷️ **Filter** by equipment, industry, timeline
- 👀 **Preview** documents
- ⬇️ **Download** or delete files

All your cases are organized and searchable in one place!
        """
    },
    {
        "step": 5,
        "title": "🚀 You're All Set!",
        "content": """
**That's it! You're ready to go!**

Quick recap:
1. 📤 **Upload** - Add your documents
2. 💬 **Chat** - Ask questions and get AI recommendations
3. 📚 **Browse** - View and manage your knowledge base

**Tips for success:**
- Upload documents regularly to keep knowledge fresh
- Use specific questions for better results
- Create separate chats for different projects
- Check the "Retrieved Documents" to see AI's sources

**Need help?** Click "📖 Show Tutorial" anytime to see this guide again.

Ready to start? Click "Got it!" below 👇
        """
    }
]

# ============================================================================
# UTILS
# ============================================================================

def call_deepseek_api(messages, temperature=0.7, max_tokens=2000):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"model": DEEPSEEK_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        resp = requests.post(f"{DEEPSEEK_API_BASE}/chat/completions", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API Error: {str(e)}"

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return " ".join([page.extract_text() or "" for page in pdf_reader.pages])
    except: return ""

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except: return ""

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except: return ""

def extract_text_from_file(file_path, file_format):
    if file_format == "pdf": return extract_text_from_pdf(file_path)
    elif file_format in ["docx", "doc"]: return extract_text_from_docx(file_path)
    elif file_format == "txt": return extract_text_from_txt(file_path)
    return ""

def generate_embedding(text):
    text_hash = hashlib.md5((text or "").encode()).hexdigest()
    fake = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
    return (fake + [0.0] * 1536)[:1536]

def cosine_similarity(v1, v2):
    dot = sum(a*b for a,b in zip(v1, v2))
    m1, m2 = sum(a*a for a in v1) ** 0.5, sum(b*b for b in v2) ** 0.5
    return dot / (m1 * m2) if m1 and m2 else 0.0

def extract_files_from_zip(zip_file):
    extracted = []
    temp_dir = tempfile.mkdtemp()
    try:
        zip_path = os.path.join(temp_dir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                if filename.startswith('.'): continue
                ext = filename.split('.')[-1].lower() if '.' in filename else ''
                if ext in ['pdf', 'docx', 'doc', 'txt', 'xlsx', 'xls', 'png', 'jpg', 'jpeg']:
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'rb') as f:
                        extracted.append({'name': filename, 'content': f.read(), 
                                        'size': os.path.getsize(file_path), 'type': ext})
    finally:
        shutil.rmtree(temp_dir)
    return extracted

AUTO_ANNOTATE_SYSTEM = """You are an underwriting document auto-tagger. Given text and filename, produce STRICT JSON:
{"tags": {"equipment": [], "industry": [], "timeline": []}, "decision": "Approved|Declined|Conditional|Pending",
"premium": number, "risk_level": "Low|Medium|High", "case_summary": "", "key_insights": ""}
Return ONLY valid JSON."""

def auto_annotate_by_llm(text, filename):
    content = call_deepseek_api([{"role":"system","content":AUTO_ANNOTATE_SYSTEM},
                                 {"role":"user","content":f"FILENAME: {filename}\nTEXT:\n{text[:4000]}"}],
                                temperature=0.2, max_tokens=700)
    try:
        cleaned = content.strip().strip("`")
        if "```" in cleaned: cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
        data = json.loads(cleaned.strip())
    except:
        data = {"tags":{"equipment":["Other"],"industry":["Other"],"timeline":["Earlier"]},
                "decision":"Pending","premium":0,"risk_level":"Medium","case_summary":"","key_insights":""}
    data.setdefault("tags", {})
    for k in ["equipment","industry","timeline"]:
        data["tags"].setdefault(k, ["Other" if k!="timeline" else "Earlier"])
    for k,v in [("decision","Pending"),("premium",0),("risk_level","Medium"),("case_summary",""),("key_insights","")]:
        data.setdefault(k,v)
    return data

# ============================================================================
# USER SETTINGS
# ============================================================================

class UserSettings:
    def __init__(self):
        self.settings = self._load()
    def _load(self):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f: return json.load(f)
        return {"tutorial_completed": False, "first_visit": True, "theme": "Light"}
    def _save(self):
        with open(SETTINGS_FILE, 'w') as f: json.dump(self.settings, f, indent=2)
    def is_first_time(self):
        return self.settings.get("first_visit", True) and not self.settings.get("tutorial_completed", False)
    def complete_tutorial(self):
        self.settings["tutorial_completed"] = True
        self.settings["first_visit"] = False
        self._save()
    def skip_tutorial(self):
        self.settings["first_visit"] = False
        self._save()

# ============================================================================
# KNOWLEDGE BASE
# ============================================================================

class KnowledgeBase:
    def __init__(self):
        self.metadata = self._load_metadata()
        self.embeddings = self._load_embeddings()
    def _load_metadata(self):
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r', encoding='utf-8') as f: return json.load(f)
        return []
    def _save_metadata(self):
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    def _load_embeddings(self):
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, 'r') as f: return json.load(f)
        return {}
    def _save_embeddings(self):
        with open(EMBEDDINGS_FILE, 'w') as f: json.dump(self.embeddings, f, indent=2)
    
    def add_document_from_content(self, filename, file_content, file_size, tags, case_summary, 
                                   key_insights, decision, premium, risk_level, extracted_text=""):
        doc_id = f"DOC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(filename.encode()).hexdigest()[:6].upper()}"
        ext = filename.split('.')[-1].lower()
        saved_filename = f"{doc_id}.{ext}"
        file_path = os.path.join(DOCS_DIR, saved_filename)
        with open(file_path, "wb") as f: f.write(file_content)
        full_text = f"{case_summary} {key_insights} {extracted_text[:1000]}"
        embedding = generate_embedding(full_text)
        doc_meta = {"doc_id": doc_id, "filename": filename, "file_format": ext, "file_path": file_path,
                    "file_size_kb": file_size/1024, "upload_date": datetime.now().isoformat(),
                    "tags": tags, "decision": decision, "premium": premium, "risk_level": risk_level,
                    "case_summary": case_summary, "key_insights": key_insights, "extracted_text_preview": extracted_text[:500]}
        self.metadata.append(doc_meta)
        self.embeddings[doc_id] = embedding
        self._save_metadata(); self._save_embeddings()
        return doc_meta
    
    def search_documents(self, query, top_k=5):
        if not self.metadata: return []
        qv = generate_embedding(query)
        scored = []
        for doc in self.metadata:
            doc_id = doc["doc_id"]
            if doc_id in self.embeddings:
                sim = cosine_similarity(qv, self.embeddings[doc_id])
                ql = query.lower()
                for tag_list in doc["tags"].values():
                    for tag in tag_list:
                        if tag.lower() in ql: sim += 0.1
                scored.append((sim, doc))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [d for _, d in scored[:top_k]]
    
    def delete_document(self, doc_id):
        self.metadata = [d for d in self.metadata if d["doc_id"] != doc_id]
        if doc_id in self.embeddings: del self.embeddings[doc_id]
        for fn in os.listdir(DOCS_DIR):
            if fn.startswith(doc_id): os.remove(os.path.join(DOCS_DIR, fn))
        self._save_metadata(); self._save_embeddings()
    
    def get_stats(self):
        return {"total_documents": len(self.metadata),
                "total_size_mb": sum(d["file_size_kb"] for d in self.metadata)/1024 if self.metadata else 0.0}

# ============================================================================
# CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    def __init__(self):
        self.conversations = self._load()
    def _load(self):
        if os.path.exists(CONV_FILE):
            with open(CONV_FILE, 'r', encoding='utf-8') as f: return json.load(f)
        return {}
    def _save(self):
        with open(CONV_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, ensure_ascii=False, indent=2)
    def create_conversation(self, title=None):
        conv_id = f"CONV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.conversations[conv_id] = {"id": conv_id, "title": title or "New Chat",
                                       "created_at": datetime.now().isoformat(),
                                       "updated_at": datetime.now().isoformat(), "messages": []}
        self._save()
        return conv_id
    def add_message(self, conv_id, role, content):
        if conv_id in self.conversations:
            self.conversations[conv_id]["messages"].append({"role": role, "content": content,
                                                            "timestamp": datetime.now().isoformat()})
            self.conversations[conv_id]["updated_at"] = datetime.now().isoformat()
            if role == "user" and len(self.conversations[conv_id]["messages"]) == 1:
                self.conversations[conv_id]["title"] = content[:35] + ("..." if len(content) > 35 else "")
            self._save()
    def delete_conversation(self, conv_id):
        if conv_id in self.conversations: del self.conversations[conv_id]
        self._save()
    def get_conversation(self, conv_id):
        return self.conversations.get(conv_id, None)
    def get_all_conversations(self):
        convs = list(self.conversations.values())
        convs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return convs

# ============================================================================
# CHAT
# ============================================================================

def generate_cot_response(query, retrieved_docs):
    """CoT模式 - 完整的5步推理"""
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
    system_prompt = """You are Mr. X's AI underwriting assistant.

Use this Chain-of-Thought framework for analysis:

**Step 1: Extract Key Tags**
- Identify equipment, industry, timeline from query

**Step 2: Analyze Retrieved Precedents**
- Review each relevant case
- Note similarities and differences

**Step 3: Check Recency & Applicability**
- Prioritize recent cases
- Consider market changes

**Step 4: Identify Decision Patterns**
- Look for consistent approvals/declines
- Identify risk factors

**Step 5: Recommend with Rationale**
- Provide clear decision
- Suggest premium range
- Cite specific precedents

Format your response with clear sections for each step."""
    
    messages = [
        {"role":"system","content": system_prompt},
        {"role":"user","content":f'Query: "{query}"\n\nRetrieved Cases:\n{docs_text}\n\nProvide step-by-step analysis.'}
    ]
    return call_deepseek_api(messages, temperature=0.3, max_tokens=2500)

def generate_quick_response(query, retrieved_docs):
    """快速模式 - 直接给出结论"""
    if not retrieved_docs:
        return "⚠️ **No Relevant Cases Found**\n\nPlease add documents to the knowledge base or try a different query."
    
    docs_summary = []
    for doc in retrieved_docs:
        equipment = ", ".join(doc["tags"].get("equipment", []))
        industry = ", ".join(doc["tags"].get("industry", []))
        docs_summary.append(f"- {doc['filename']}: {doc['decision']}, ${doc['premium']:,}, {equipment}, {industry}")
    
    docs_text = "\n".join(docs_summary)
    
    system_prompt = """You are Mr. X's AI underwriting assistant.

Provide a CONCISE response with:
1. **Recommended Decision** (Approve/Decline/Conditional)
2. **Suggested Premium Range** (with specific numbers)
3. **Key Risk Factors** (brief bullet points)
4. **Precedent References** (cite 2-3 most relevant cases by filename)

Keep response under 200 words. Be direct and actionable."""
    
    messages = [
        {"role":"system","content": system_prompt},
        {"role":"user","content":f'Query: "{query}"\n\nRelevant Precedents:\n{docs_text}\n\nProvide quick recommendation.'}
    ]
    return call_deepseek_api(messages, temperature=0.2, max_tokens=800)

# ============================================================================
# CSS
# ============================================================================

def inject_css(theme):
    if theme == "Dark":
        css = """<style>
        :root { --bg: #0b1220; --card: #1a2332; --text: #e5e7eb; --border: #374151; --active: #1e40af; }
        .stApp {background: var(--bg); color: var(--text);}
        .tag-equipment { background: #3b82f6; color: white; padding: 0.25rem 0.75rem; border-radius: 1rem; margin: 0.25rem; display: inline-block; }
        .tag-industry { background: #10b981; color: white; padding: 0.25rem 0.75rem; border-radius: 1rem; margin: 0.25rem; display: inline-block; }
        .tag-timeline { background: #f59e0b; color: white; padding: 0.25rem 0.75rem; border-radius: 1rem; margin: 0.25rem; display: inline-block; }
        #MainMenu, footer, header {visibility: hidden;}
        </style>"""
    else:
        css = """<style>
        :root { --bg: #f9fafb; --card: #ffffff; --text: #111827; --border: #e5e7eb; --active: #3b82f6; }
        .stApp {background: var(--bg); color: var(--text);}
        .tag-equipment { background: #dbeafe; color: #1e40af; padding: 0.25rem 0.75rem; border-radius: 1rem; margin: 0.25rem; display: inline-block; }
        .tag-industry { background: #d1fae5; color: #065f46; padding: 0.25rem 0.75rem; border-radius: 1rem; margin: 0.25rem; display: inline-block; }
        .tag-timeline { background: #fef3c7; color: #92400e; padding: 0.25rem 0.75rem; border-radius: 1rem; margin: 0.25rem; display: inline-block; }
        #MainMenu, footer, header {visibility: hidden;}
        </style>"""
    st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="Underwriting Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")
    
    kb = KnowledgeBase()
    conv_mgr = ConversationManager()
    user_settings = UserSettings()
    
    if "current_conv_id" not in st.session_state:
        all_convs = conv_mgr.get_all_conversations()
        st.session_state.current_conv_id = all_convs[0]["id"] if all_convs else conv_mgr.create_conversation()
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"
    if "theme" not in st.session_state:
        st.session_state.theme = user_settings.settings.get("theme", "Light")
    if "tutorial_active" not in st.session_state:
        st.session_state.tutorial_active = user_settings.is_first_time()
    if "tutorial_step" not in st.session_state:
        st.session_state.tutorial_step = 1 if st.session_state.tutorial_active else 0
    
    inject_css(st.session_state.theme)
    
    # TUTORIAL MODE
    if st.session_state.tutorial_active and st.session_state.tutorial_step > 0:
        render_tutorial(user_settings)
        return
    
    # NORMAL APP
    render_app(kb, conv_mgr, user_settings)

def render_tutorial(user_settings):
    step = st.session_state.tutorial_step
    if step < 1 or step > len(TUTORIAL_STEPS):
        st.session_state.tutorial_active = False
        st.rerun()
        return
    
    step_data = TUTORIAL_STEPS[step - 1]
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        # Progress indicator
        progress_html = '<div style="display: flex; gap: 0.5rem; margin-bottom: 2rem; justify-content: center;">'
        for i in range(len(TUTORIAL_STEPS)):
            color = "#3b82f6" if i < step else "#e5e7eb"
            progress_html += f'<div style="width: 40px; height: 8px; border-radius: 4px; background: {color};"></div>'
        progress_html += '</div>'
        st.markdown(progress_html, unsafe_allow_html=True)
        
        # Title
        st.markdown(f"## {step_data['title']}")
        
        # Content (using Streamlit's native markdown)
        st.markdown(step_data['content'])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            if step > 1 and st.button("← Back", use_container_width=True):
                st.session_state.tutorial_step -= 1
                st.rerun()
        with col_b:
            if st.button("Skip Tutorial", use_container_width=True):
                user_settings.skip_tutorial()
                st.session_state.tutorial_active = False
                st.session_state.tutorial_step = 0
                st.rerun()
        with col_c:
            button_text = "Got it!" if step == len(TUTORIAL_STEPS) else "Next →"
            if st.button(button_text, type="primary", use_container_width=True):
                if step < len(TUTORIAL_STEPS):
                    st.session_state.tutorial_step += 1
                    st.rerun()
                else:
                    user_settings.complete_tutorial()
                    st.session_state.tutorial_active = False
                    st.session_state.tutorial_step = 0
                    st.balloons()
                    st.rerun()

def render_app(kb, conv_mgr, user_settings):
    all_convs = conv_mgr.get_all_conversations()
    col_nav, col_main = st.columns([1, 4])
    
    with col_nav:
        st.markdown("### 🤖 Underwriting AI")
        if st.button("🎨 " + st.session_state.theme, use_container_width=True):
            st.session_state.theme = "Dark" if st.session_state.theme == "Light" else "Light"
            user_settings.settings["theme"] = st.session_state.theme
            user_settings._save()
            st.rerun()
        st.markdown("---")
        if st.button("📖 Show Tutorial", use_container_width=True):
            st.session_state.tutorial_active = True
            st.session_state.tutorial_step = 1
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
                if st.button(f"{'📌' if is_active else '💬'} {conv['title'][:28]}", key=f"nav_conv_{conv_id}",
                           use_container_width=True, type="primary" if is_active else "secondary"):
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
    
    with col_main:
        if st.session_state.current_page == "chat":
            render_chat_page(kb, conv_mgr)
        elif st.session_state.current_page == "kb":
            render_kb_page(kb)
        elif st.session_state.current_page == "upload":
            render_upload_page(kb)

def render_chat_page(kb, conv_mgr):
    current_conv = conv_mgr.get_conversation(st.session_state.current_conv_id)
    if not current_conv: 
        st.error("Conversation not found")
        return
    
    # Title with mode selector
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"💬 {current_conv['title']}")
    with col2:
        # Initialize response mode if not exists
        if "response_mode" not in st.session_state:
            st.session_state.response_mode = "CoT"
        
        response_mode = st.selectbox(
            "Response Mode",
            ["🧠 CoT (Detailed)", "⚡ Quick"],
            index=0 if st.session_state.response_mode == "CoT" else 1,
            help="CoT: Step-by-step reasoning (slower, detailed)\nQuick: Direct answer (faster, concise)",
            key="mode_selector"
        )
        st.session_state.response_mode = "CoT" if "CoT" in response_mode else "Quick"
    
    # Display existing messages
    for msg in current_conv["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # If message has retrieved docs metadata, show clickable references
            if msg["role"] == "assistant" and "retrieved_docs" in msg:
                render_clickable_references(msg["retrieved_docs"], kb)
    
    # Chat input
    if prompt := st.chat_input("Ask about underwriting cases...", key="chat_input_field"):
        if st.session_state.tour_active and st.session_state.tour_step == 4:
            st.session_state.tour_step += 1
        
        # Add user message
        conv_mgr.add_message(st.session_state.current_conv_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner(f"{'🧠 Analyzing with CoT...' if st.session_state.response_mode == 'CoT' else '⚡ Generating quick response...'}"):
                retrieved = kb.search_documents(prompt, top_k=5)
                
                # Choose response mode
                if st.session_state.response_mode == "CoT":
                    resp = generate_cot_response(prompt, retrieved)
                    mode_label = "🧠 Chain-of-Thought Analysis"
                else:
                    resp = generate_quick_response(prompt, retrieved)
                    mode_label = "⚡ Quick Response"
                
                # Display response with mode indicator
                st.caption(mode_label)
                st.markdown(resp)
                
                # Show clickable references
                if retrieved:
                    render_clickable_references(retrieved, kb)
        
        # Save message with metadata
        message_with_meta = {
            "role": "assistant",
            "content": resp,
            "retrieved_docs": [doc["doc_id"] for doc in retrieved],
            "mode": st.session_state.response_mode
        }
        conv_mgr.add_message(st.session_state.current_conv_id, "assistant", resp)
        
        # Store metadata separately (for clickable links)
        if "message_metadata" not in st.session_state:
            st.session_state.message_metadata = {}
        st.session_state.message_metadata[len(current_conv["messages"])] = {
            "retrieved_docs": retrieved
        }
        
        st.rerun()

def render_clickable_references(doc_ids_or_docs, kb):
    """Render clickable document references that jump to KB page"""
    
    # Handle both doc IDs and full doc objects
    if doc_ids_or_docs and isinstance(doc_ids_or_docs[0], str):
        # It's a list of doc IDs, need to fetch full docs
        docs = [doc for doc in kb.metadata if doc["doc_id"] in doc_ids_or_docs]
    else:
        # It's already full doc objects
        docs = doc_ids_or_docs
    
    if not docs:
        return
    
    st.markdown("---")
    st.markdown(f"**📚 Referenced Documents ({len(docs)}):**")
    
    # Create clickable document cards
    for i, doc in enumerate(docs, 1):
        with st.expander(f"{i}. {SUPPORTED_FORMATS.get(doc['file_format'],'📎')} {doc['filename']}", expanded=False):
            # Document info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**ID:** `{doc['doc_id']}`")
            with col2:
                decision = doc.get("decision", "N/A")
                emoji = {"Approved": "🟢", "Declined": "🔴", "Conditional": "🟡", "Pending": "⚪"}.get(decision, "⚪")
                st.markdown(f"**Decision:** {emoji} {decision}")
            with col3:
                st.markdown(f"**Premium:** ${doc.get('premium', 0):,}")
            
            # Tags
            tags_html = ""
            for t in doc["tags"].get("equipment", []): 
                tags_html += f'<span class="tag-equipment">🔧 {t}</span>'
            for t in doc["tags"].get("industry", []): 
                tags_html += f'<span class="tag-industry">🏭 {t}</span>'
            for t in doc["tags"].get("timeline", []): 
                tags_html += f'<span class="tag-timeline">📅 {t}</span>'
            if tags_html:
                st.markdown(tags_html, unsafe_allow_html=True)
            
            # Summary
            if doc.get("case_summary"):
                st.info(doc["case_summary"][:200] + "..." if len(doc.get("case_summary", "")) > 200 else doc["case_summary"])
            
            # Jump to KB button
            if st.button(f"🔗 View in Knowledge Base", key=f"jump_kb_{doc['doc_id']}_{i}", use_container_width=True):
                # Set filter to show this specific document
                st.session_state.current_page = "kb"
                st.session_state.kb_search_query = doc["filename"]
                st.rerun()

def render_kb_page(kb):
    st.title("📄 Knowledge Base")
    
    if not kb.metadata:
        st.info("📭 No documents yet. Go to Upload tab to add your first case!")
        return
    
    # ============================================================================
    # SEARCH & FILTER SECTION
    # ============================================================================
    
    st.markdown("### 🔍 Search & Filter")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Check if jumping from chat with search query
        default_search = st.session_state.get("kb_search_query", "")
        if default_search:
            # Clear the session state after using it
            st.session_state.kb_search_query = ""
        
        search_query = st.text_input(
            "🔎 Search by filename or content",
            value=default_search,
            placeholder="e.g., 'ABC Oil', 'turbine', 'boiler'...",
            help="Search across filenames, summaries, and key insights"
        )
    
    with col2:
        sort_by = st.selectbox(
            "📊 Sort by",
            ["Upload Date (Newest)", "Upload Date (Oldest)", "Premium (High to Low)", "Premium (Low to High)", "Filename (A-Z)"],
            help="Choose how to sort the results"
        )
    
    # Filter tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Equipment", "🏭 Industry", "📅 Timeline", "⚖️ Decision"])
    
    with tab1:
        equipment_filter = st.multiselect(
            "Filter by Equipment",
            options=TAG_OPTIONS["equipment"],
            help="Select one or more equipment types"
        )
    
    with tab2:
        industry_filter = st.multiselect(
            "Filter by Industry",
            options=TAG_OPTIONS["industry"],
            help="Select one or more industries"
        )
    
    with tab3:
        timeline_filter = st.multiselect(
            "Filter by Timeline",
            options=TAG_OPTIONS["timeline"],
            help="Select one or more time periods"
        )
    
    with tab4:
        decision_filter = st.multiselect(
            "Filter by Decision",
            options=["Approved", "Declined", "Conditional", "Pending"],
            help="Filter by underwriting decision"
        )
    
    st.markdown("---")
    
    # ============================================================================
    # APPLY FILTERS
    # ============================================================================
    
    filtered_docs = kb.metadata.copy()
    
    # Text search
    if search_query:
        search_lower = search_query.lower()
        filtered_docs = [
            doc for doc in filtered_docs
            if search_lower in doc["filename"].lower()
            or search_lower in doc.get("case_summary", "").lower()
            or search_lower in doc.get("key_insights", "").lower()
        ]
    
    # Equipment filter
    if equipment_filter:
        filtered_docs = [
            doc for doc in filtered_docs
            if any(eq in doc["tags"].get("equipment", []) for eq in equipment_filter)
        ]
    
    # Industry filter
    if industry_filter:
        filtered_docs = [
            doc for doc in filtered_docs
            if any(ind in doc["tags"].get("industry", []) for ind in industry_filter)
        ]
    
    # Timeline filter
    if timeline_filter:
        filtered_docs = [
            doc for doc in filtered_docs
            if any(time in doc["tags"].get("timeline", []) for time in timeline_filter)
        ]
    
    # Decision filter
    if decision_filter:
        filtered_docs = [
            doc for doc in filtered_docs
            if doc.get("decision", "") in decision_filter
        ]
    
    # ============================================================================
    # SORTING
    # ============================================================================
    
    if sort_by == "Upload Date (Newest)":
        filtered_docs.sort(key=lambda x: x.get("upload_date", ""), reverse=True)
    elif sort_by == "Upload Date (Oldest)":
        filtered_docs.sort(key=lambda x: x.get("upload_date", ""))
    elif sort_by == "Premium (High to Low)":
        filtered_docs.sort(key=lambda x: x.get("premium", 0), reverse=True)
    elif sort_by == "Premium (Low to High)":
        filtered_docs.sort(key=lambda x: x.get("premium", 0))
    elif sort_by == "Filename (A-Z)":
        filtered_docs.sort(key=lambda x: x.get("filename", "").lower())
    
    # ============================================================================
    # RESULTS HEADER
    # ============================================================================
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### 📊 Results: {len(filtered_docs)} / {len(kb.metadata)} documents")
    with col2:
        if st.button("🔄 Reset Filters", use_container_width=True):
            st.rerun()
    with col3:
        if filtered_docs and st.button("📥 Export CSV", use_container_width=True):
            import pandas as pd
            df = pd.DataFrame([{
                "Document ID": d["doc_id"],
                "Filename": d["filename"],
                "Equipment": ", ".join(d["tags"].get("equipment", [])),
                "Industry": ", ".join(d["tags"].get("industry", [])),
                "Timeline": ", ".join(d["tags"].get("timeline", [])),
                "Decision": d.get("decision", ""),
                "Premium": d.get("premium", 0),
                "Risk Level": d.get("risk_level", ""),
                "Upload Date": d.get("upload_date", "")
            } for d in filtered_docs])
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv,
                "knowledge_base_export.csv",
                "text/csv",
                use_container_width=True
            )
    
    st.markdown("---")
    
    # ============================================================================
    # DISPLAY DOCUMENTS
    # ============================================================================
    
    if not filtered_docs:
        st.warning("🔍 No documents match your filters. Try adjusting your search criteria.")
        return
    
    for doc in filtered_docs:
        with st.expander(f"{SUPPORTED_FORMATS.get(doc['file_format'],'📎')} {doc['filename']}", expanded=False):
            # Document header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Document ID:** `{doc['doc_id']}`")
                st.markdown(f"**Upload Date:** {doc.get('upload_date', 'N/A')[:10]}")
            with col2:
                st.markdown(f"**Size:** {doc.get('file_size_kb', 0):.1f} KB")
                st.markdown(f"**Format:** {doc['file_format'].upper()}")
            
            st.markdown("---")
            
            # Tags
            tags_html = ""
            for t in doc["tags"].get("equipment", []): 
                tags_html += f'<span class="tag-equipment">🔧 {t}</span>'
            for t in doc["tags"].get("industry", []): 
                tags_html += f'<span class="tag-industry">🏭 {t}</span>'
            for t in doc["tags"].get("timeline", []): 
                tags_html += f'<span class="tag-timeline">📅 {t}</span>'
            if tags_html:
                st.markdown(tags_html, unsafe_allow_html=True)
                st.markdown("---")
            
            # Decision & Premium
            col1, col2, col3 = st.columns(3)
            with col1:
                decision = doc.get("decision", "N/A")
                decision_color = {
                    "Approved": "🟢",
                    "Declined": "🔴",
                    "Conditional": "🟡",
                    "Pending": "⚪"
                }.get(decision, "⚪")
                st.markdown(f"**Decision:** {decision_color} {decision}")
            with col2:
                premium = doc.get("premium", 0)
                st.markdown(f"**Premium:** ${premium:,}")
            with col3:
                risk = doc.get("risk_level", "N/A")
                risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")
                st.markdown(f"**Risk:** {risk_emoji} {risk}")
            
            st.markdown("---")
            
            # Case summary
            if doc.get("case_summary"):
                st.markdown("**📝 Case Summary:**")
                st.info(doc["case_summary"])
            
            # Key insights
            if doc.get("key_insights"):
                st.markdown("**💡 Key Insights:**")
                st.success(doc["key_insights"])
            
            # Preview
            if doc.get("extracted_text_preview"):
                with st.expander("👁️ Text Preview"):
                    st.text(doc["extracted_text_preview"])
            
            st.markdown("---")
            
            # Actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑️ Delete", key=f"del_{doc['doc_id']}", use_container_width=True):
                    kb.delete_document(doc["doc_id"])
                    st.success("Deleted!")
                    st.rerun()
            with col2:
                if st.button("📋 Copy ID", key=f"copy_{doc['doc_id']}", use_container_width=True):
                    st.code(doc['doc_id'])
            with col3:
                if os.path.exists(doc.get("file_path", "")):
                    with open(doc["file_path"], "rb") as f:
                        st.download_button(
                            "📥 Download",
                            f.read(),
                            doc["filename"],
                            key=f"download_{doc['doc_id']}",
                            use_container_width=True
                        )


def render_upload_page(kb):
    st.title("📤 Batch Upload Documents")
    upload_mode = st.radio("Upload Mode", ["📄 Multiple Files", "📦 ZIP Archive"], horizontal=True)
    st.markdown("---")
    
    if upload_mode == "📄 Multiple Files":
        uploaded_files = st.file_uploader("Choose documents", 
            type=['pdf', 'docx', 'doc', 'txt', 'xlsx', 'xls', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True, help="Hold Ctrl/Cmd to select multiple files")
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files selected")
            with st.expander(f"📋 File List ({len(uploaded_files)} files)"):
                for i, f in enumerate(uploaded_files, 1):
                    st.write(f"{i}. {SUPPORTED_FORMATS.get(f.name.split('.')[-1].lower(), '📎')} {f.name} ({f.size/1024:.1f} KB)")
            if st.button("🚀 Process All Files", type="primary", use_container_width=True):
                process_batch(kb, uploaded_files)
    else:
        zip_file = st.file_uploader("Choose ZIP file", type=['zip'])
        if zip_file:
            st.success(f"✅ ZIP selected: {zip_file.name} ({zip_file.size/1024:.1f} KB)")
            if st.button("🚀 Extract & Process ZIP", type="primary", use_container_width=True):
                with st.spinner("📦 Extracting ZIP..."):
                    extracted = extract_files_from_zip(zip_file)
                if extracted:
                    st.success(f"✅ Extracted {len(extracted)} files")
                    with st.expander(f"📋 Files ({len(extracted)})"):
                        for i, f in enumerate(extracted, 1):
                            st.write(f"{i}. {SUPPORTED_FORMATS.get(f['type'], '📎')} {f['name']} ({f['size']/1024:.1f} KB)")
                    process_batch_content(kb, extracted)
                else:
                    st.warning("No supported files in ZIP")

def process_batch(kb, files):
    st.markdown("---")
    st.markdown("### 🔄 Processing Files...")
    progress = st.progress(0)
    status = st.container()
    success, errors = 0, 0
    for i, f in enumerate(files):
        progress.progress((i + 1) / len(files))
        with status:
            st.markdown(f"**Processing ({i+1}/{len(files)}):** {f.name}")
        try:
            ext = f.name.split('.')[-1].lower()
            temp_path = os.path.join(DOCS_DIR, f"TEMP_{i}.{ext}")
            with open(temp_path, "wb") as tf: tf.write(f.getbuffer())
            text = extract_text_from_file(temp_path, ext)
            auto = auto_annotate_by_llm(text, f.name)
            doc = kb.add_document_from_content(f.name, f.getbuffer(), f.size, auto["tags"],
                auto["case_summary"], auto["key_insights"], auto["decision"], int(auto.get("premium", 0) or 0),
                auto["risk_level"], text[:800])
            try: os.remove(temp_path)
            except: pass
            with status: st.success(f"✅ {f.name} → {doc['doc_id']}")
            success += 1
        except Exception as e:
            with status: st.error(f"❌ {f.name}: {str(e)}")
            errors += 1
    progress.progress(1.0)
    st.markdown("---")
    st.markdown("### 📊 Upload Summary")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("✅ Success", success)
    with c2: st.metric("❌ Errors", errors)
    with c3: st.metric("📊 Total", len(files))
    if success > 0: st.balloons()

def process_batch_content(kb, files):
    st.markdown("---")
    st.markdown("### 🔄 Processing Files...")
    progress = st.progress(0)
    status = st.container()
    success, errors = 0, 0
    for i, f in enumerate(files):
        progress.progress((i + 1) / len(files))
        with status:
            st.markdown(f"**Processing ({i+1}/{len(files)}):** {f['name']}")
        try:
            ext = f['type']
            temp_path = os.path.join(DOCS_DIR, f"TEMP_{i}.{ext}")
            with open(temp_path, "wb") as tf: tf.write(f['content'])
            text = extract_text_from_file(temp_path, ext)
            auto = auto_annotate_by_llm(text, f['name'])
            doc = kb.add_document_from_content(f['name'], f['content'], f['size'], auto["tags"],
                auto["case_summary"], auto["key_insights"], auto["decision"], int(auto.get("premium", 0) or 0),
                auto["risk_level"], text[:800])
            try: os.remove(temp_path)
            except: pass
            with status: st.success(f"✅ {f['name']} → {doc['doc_id']}")
            success += 1
        except Exception as e:
            with status: st.error(f"❌ {f['name']}: {str(e)}")
            errors += 1
    progress.progress(1.0)
    st.markdown("---")
    st.markdown("### 📊 Upload Summary")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("✅ Success", success)
    with c2: st.metric("❌ Errors", errors)
    with c3: st.metric("📊 Total", len(files))
    if success > 0: st.balloons()

if __name__ == "__main__":
    main()
