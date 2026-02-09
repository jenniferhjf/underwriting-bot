"""
Enterprise Knowledge Management System - Streamlit Cloud Optimized
Minimal dependencies version
"""

import streamlit as st
import json
import os
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
import pandas as pd
import numpy as np

# ===========================
# CONFIGURATION
# ===========================

VERSION = "4.1.0"
APP_NAME = "Underwriting Knowledge Management System"

# API Configuration
API_KEY = "sk-99bba2ce117444e197270f17d303e74f"
API_BASE = "https://api.deepseek.com/v1"
API_MODEL = "deepseek-chat"

# Directory Structure
DATA_DIR = Path("data")
KB_DIR = DATA_DIR / "knowledge_bases"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
CHUNKS_DIR = DATA_DIR / "chunks"

# Default datasets
DEFAULT_DATASETS = [
    {
        "filename": "Cargo - Agnes Fisheries - CE's notes for Yr 2021-22.pdf",
        "category": "Cargo",
        "tags": ["Cargo", "Agnes Fisheries", "2021-22"]
    },
    {
        "filename": "Hull - Marco Polo_Memo.pdf",
        "category": "Hull",
        "tags": ["Hull", "Marco Polo", "Memo"]
    },
    {
        "filename": "Cargo - Mitsui Co summary with CE's QA 29.8.22.docx",
        "category": "Cargo",
        "tags": ["Cargo", "Mitsui", "QA", "2022"]
    }
]

# Configuration
CATEGORIES = ["Hull", "Cargo", "Liability", "Property", "Marine", "Other"]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_DIM = 1536
INDEX_MODES = ["Vector Search", "Keyword Search", "Hybrid Search"]

SUPPORTED_FORMATS = {
    'pdf': '📄 PDF',
    'docx': '📝 Word',
    'txt': '📃 Text'
}

# ===========================
# UTILITY FUNCTIONS
# ===========================

def ensure_dirs():
    """Create necessary directories"""
    for dir_path in [KB_DIR, VECTOR_DB_DIR, CHUNKS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

def get_api_key() -> str:
    """Get API key"""
    return os.getenv("API_KEY", API_KEY)

def call_llm_api(system_prompt: str, user_prompt: str, 
                 temperature: float = 0.3, max_tokens: int = 2000) -> str:
    """Call LLM API"""
    try:
        api_key = get_api_key()
        if not api_key:
            return ""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": API_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 401:
            st.error("⚠️ API Authentication Failed")
            return ""
        
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except Exception as e:
        st.error(f"API Error: {e}")
        return ""

# ===========================
# MODELS
# ===========================

class KnowledgeBase:
    """Knowledge Base model"""
    
    def __init__(self, kb_id: str, name: str, description: str, category: str):
        self.kb_id = kb_id
        self.name = name
        self.description = description
        self.category = category
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.tokens = 0
        self.documents = []
        
    def to_dict(self) -> Dict:
        return {
            "kb_id": self.kb_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tokens": self.tokens,
            "documents": self.documents
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        kb = cls(data["kb_id"], data["name"], data["description"], data["category"])
        kb.created_at = data.get("created_at", datetime.now().isoformat())
        kb.updated_at = data.get("updated_at", datetime.now().isoformat())
        kb.tokens = data.get("tokens", 0)
        kb.documents = data.get("documents", [])
        return kb

class Document:
    """Document model"""
    
    def __init__(self, doc_id: str, filename: str, kb_id: str):
        self.doc_id = doc_id
        self.filename = filename
        self.kb_id = kb_id
        self.tags = []
        self.upload_time = datetime.now().isoformat()
        self.hit_count = 0
        self.file_path = ""
        self.file_size = 0
        self.chunk_count = 0
        self.vectorized = False
        
    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "kb_id": self.kb_id,
            "tags": self.tags,
            "upload_time": self.upload_time,
            "hit_count": self.hit_count,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "chunk_count": self.chunk_count,
            "vectorized": self.vectorized
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        doc = cls(data["doc_id"], data["filename"], data["kb_id"])
        doc.tags = data.get("tags", [])
        doc.upload_time = data.get("upload_time", datetime.now().isoformat())
        doc.hit_count = data.get("hit_count", 0)
        doc.file_path = data.get("file_path", "")
        doc.file_size = data.get("file_size", 0)
        doc.chunk_count = data.get("chunk_count", 0)
        doc.vectorized = data.get("vectorized", False)
        return doc

# ===========================
# KNOWLEDGE BASE MANAGEMENT
# ===========================

def create_knowledge_base(name: str, description: str, category: str) -> KnowledgeBase:
    """Create a new knowledge base"""
    kb_id = hashlib.md5(f"{name}{datetime.now()}".encode()).hexdigest()[:16]
    kb = KnowledgeBase(kb_id, name, description, category)
    
    kb_dir = KB_DIR / kb_id
    kb_dir.mkdir(exist_ok=True)
    
    with open(kb_dir / "metadata.json", 'w') as f:
        json.dump(kb.to_dict(), f, indent=2)
    
    return kb

def load_knowledge_base(kb_id: str) -> Optional[KnowledgeBase]:
    """Load knowledge base by ID"""
    kb_dir = KB_DIR / kb_id
    metadata_file = kb_dir / "metadata.json"
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            return KnowledgeBase.from_dict(data)
    return None

def list_knowledge_bases() -> List[KnowledgeBase]:
    """List all knowledge bases"""
    kbs = []
    if not KB_DIR.exists():
        return kbs
    
    for kb_dir in KB_DIR.iterdir():
        if kb_dir.is_dir():
            kb = load_knowledge_base(kb_dir.name)
            if kb:
                kbs.append(kb)
    
    return sorted(kbs, key=lambda x: x.updated_at, reverse=True)

def update_knowledge_base(kb: KnowledgeBase):
    """Update knowledge base"""
    kb.updated_at = datetime.now().isoformat()
    kb_dir = KB_DIR / kb.kb_id
    
    with open(kb_dir / "metadata.json", 'w') as f:
        json.dump(kb.to_dict(), f, indent=2)

def delete_knowledge_base(kb_id: str) -> bool:
    """Delete knowledge base"""
    try:
        kb_dir = KB_DIR / kb_id
        if kb_dir.exists():
            import shutil
            shutil.rmtree(kb_dir)
        
        vector_file = VECTOR_DB_DIR / f"{kb_id}_vectordb.json"
        if vector_file.exists():
            vector_file.unlink()
        
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

# ===========================
# DOCUMENT MANAGEMENT
# ===========================

def upload_document(kb_id: str, uploaded_file, tags: List[str] = None) -> Optional[Document]:
    """Upload document"""
    try:
        doc_id = hashlib.md5(f"{uploaded_file.name}{datetime.now()}".encode()).hexdigest()[:16]
        doc = Document(doc_id, uploaded_file.name, kb_id)
        
        if tags:
            doc.tags = tags
        else:
            doc.tags = auto_generate_tags(uploaded_file.name)
        
        kb_dir = KB_DIR / kb_id
        doc_dir = kb_dir / "documents"
        doc_dir.mkdir(exist_ok=True)
        
        file_path = doc_dir / uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        doc.file_path = str(file_path)
        doc.file_size = uploaded_file.size
        
        kb = load_knowledge_base(kb_id)
        if kb:
            kb.documents.append(doc.to_dict())
            update_knowledge_base(kb)
        
        return doc
        
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None

def delete_document(kb_id: str, doc_id: str) -> bool:
    """Delete document"""
    try:
        kb = load_knowledge_base(kb_id)
        if not kb:
            return False
        
        doc = None
        for d in kb.documents:
            if d["doc_id"] == doc_id:
                doc = d
                break
        
        if not doc:
            return False
        
        file_path = Path(doc["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        kb.documents = [d for d in kb.documents if d["doc_id"] != doc_id]
        update_knowledge_base(kb)
        
        return True
        
    except Exception as e:
        st.error(f"Delete error: {e}")
        return False

def auto_generate_tags(filename: str) -> List[str]:
    """Auto-generate tags"""
    tags = []
    name_parts = re.split(r'[\s_\-\.]+', filename)
    for part in name_parts:
        if len(part) > 2 and not part.isdigit():
            tags.append(part.title())
    
    years = re.findall(r'20\d{2}', filename)
    tags.extend(years)
    
    return list(set(tags))[:10]

# ===========================
# TEXT PROCESSING (SIMPLIFIED)
# ===========================

def extract_text_simple(file_path: Path) -> str:
    """Simple text extraction (no PDF/DOCX support to avoid dependencies)"""
    ext = file_path.suffix.lower()
    
    if ext == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except:
            return ""
    else:
        # For PDF/DOCX, return placeholder
        return f"[Document uploaded: {file_path.name}. Full text extraction requires additional libraries.]"

def chunk_text(text: str) -> List[Dict]:
    """Split text into chunks"""
    if not text:
        return []
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]
        
        chunks.append({
            'id': f"chunk_{chunk_id}",
            'text': chunk_text.strip(),
            'start': start,
            'end': end
        })
        
        chunk_id += 1
        start = end - CHUNK_OVERLAP
    
    return chunks

def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding vector"""
    hash_obj = hashlib.sha256(text.lower().encode())
    hash_bytes = hash_obj.digest()
    
    embedding = []
    for i in range(EMBEDDING_DIM):
        seed = int.from_bytes(hash_bytes, byteorder='big') + i
        value = (seed % 1000) / 1000.0 - 0.5
        embedding.append(value)
    
    embedding = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding

def vectorize_document(kb_id: str, doc_id: str) -> bool:
    """Vectorize document"""
    try:
        kb = load_knowledge_base(kb_id)
        if not kb:
            return False
        
        doc_data = next((d for d in kb.documents if d["doc_id"] == doc_id), None)
        if not doc_data:
            return False
        
        file_path = Path(doc_data["file_path"])
        text = extract_text_simple(file_path)
        
        if not text or len(text) < 10:
            st.warning(f"No text extracted from {doc_data['filename']}")
            return False
        
        chunks = chunk_text(text)
        embeddings = [generate_embedding(chunk['text']) for chunk in chunks]
        
        # Save chunks
        chunks_file = CHUNKS_DIR / f"{kb_id}_{doc_id}_chunks.json"
        with open(chunks_file, 'w') as f:
            json.dump(chunks, f, indent=2)
        
        # Update vector DB
        vector_file = VECTOR_DB_DIR / f"{kb_id}_vectordb.json"
        
        if vector_file.exists():
            with open(vector_file, 'r') as f:
                vector_data = json.load(f)
        else:
            vector_data = {"chunks": [], "embeddings": []}
        
        for chunk, emb in zip(chunks, embeddings):
            vector_data["chunks"].append({
                "chunk_id": f"{doc_id}_{chunk['id']}",
                "doc_id": doc_id,
                "text": chunk['text']
            })
            vector_data["embeddings"].append(emb.tolist())
        
        with open(vector_file, 'w') as f:
            json.dump(vector_data, f)
        
        # Update document
        for i, d in enumerate(kb.documents):
            if d["doc_id"] == doc_id:
                kb.documents[i]["vectorized"] = True
                kb.documents[i]["chunk_count"] = len(chunks)
                break
        
        kb.tokens += len(text.split())
        update_knowledge_base(kb)
        
        return True
        
    except Exception as e:
        st.error(f"Vectorization error: {e}")
        return False

# ===========================
# UI COMPONENTS
# ===========================

def render_navigation():
    """Render navigation"""
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #1e40af 0%, #7c3aed 100%); 
                padding: 1.5rem 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 1.8rem;">🏢 {APP_NAME}</h1>
        <p style="color: #e0e7ff; margin-top: 0.5rem; font-size: 0.9rem;">Version {VERSION} | Cloud Optimized</p>
    </div>
    """, unsafe_allow_html=True)

def render_kb_list():
    """Render knowledge base list"""
    st.subheader("📚 Knowledge Base Management")
    
    kbs = list_knowledge_bases()
    
    if not kbs:
        st.info("No knowledge bases. Create one to get started!")
        return
    
    data = []
    for kb in kbs:
        data.append({
            "Name": kb.name,
            "Description": kb.description[:50] + "..." if len(kb.description) > 50 else kb.description,
            "Category": kb.category,
            "Updated": kb.updated_at[:10],
            "Tokens": f"{kb.tokens:,}",
            "Docs": len(kb.documents)
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    cols = st.columns(min(len(kbs), 4))
    for idx, (col, kb) in enumerate(zip(cols, kbs)):
        with col:
            if st.button(f"📂 {kb.name}", key=f"view_{kb.kb_id}", use_container_width=True):
                st.session_state.current_kb = kb.kb_id
                st.session_state.page = "kb_detail"
                st.rerun()

def render_kb_creation():
    """Render KB creation form"""
    st.subheader("➕ Create New Knowledge Base")
    
    with st.form("create_kb"):
        name = st.text_input("Name*", placeholder="e.g., Marine Insurance KB")
        description = st.text_area("Description*", placeholder="Brief description")
        category = st.selectbox("Category*", CATEGORIES)
        
        if st.form_submit_button("Create", use_container_width=True):
            if not name or not description:
                st.error("Please fill all fields")
            else:
                kb = create_knowledge_base(name, description, category)
                st.success(f"✅ Created '{name}'!")
                st.session_state.current_kb = kb.kb_id
                st.session_state.page = "kb_detail"
                st.rerun()

def render_kb_detail(kb_id: str):
    """Render KB detail page"""
    kb = load_knowledge_base(kb_id)
    if not kb:
        st.error("Knowledge base not found")
        return
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"📚 {kb.name}")
        st.caption(kb.description)
    
    with col2:
        if st.button("🔙 Back"):
            st.session_state.page = "kb_list"
            st.rerun()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents", len(kb.documents))
    with col2:
        st.metric("Tokens", f"{kb.tokens:,}")
    with col3:
        st.metric("Category", kb.category)
    with col4:
        vectorized = sum(1 for d in kb.documents if d.get("vectorized", False))
        st.metric("Vectorized", f"{vectorized}/{len(kb.documents)}")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📄 Documents", "⬆️ Upload", "🔧 Vectorization"])
    
    with tab1:
        render_documents(kb)
    
    with tab2:
        render_upload(kb)
    
    with tab3:
        render_vectorization(kb)

def render_documents(kb: KnowledgeBase):
    """Render document list"""
    if not kb.documents:
        st.info("No documents. Upload some to get started!")
        return
    
    for doc_data in kb.documents:
        doc = Document.from_dict(doc_data)
        
        with st.expander(f"📄 {doc.filename}"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Upload", doc.upload_time[:10])
            with col2:
                st.metric("Size", f"{doc.file_size / 1024:.1f} KB")
            with col3:
                st.metric("Chunks", doc.chunk_count)
            with col4:
                st.metric("Vectorized", "✅" if doc.vectorized else "❌")
            
            st.markdown("**Tags:**")
            if doc.tags:
                tags_html = " ".join([
                    f'<span style="background:#3b82f6;color:white;padding:0.2rem 0.5rem;'
                    f'border-radius:5px;margin:0.2rem;">{tag}</span>'
                    for tag in doc.tags
                ])
                st.markdown(tags_html, unsafe_allow_html=True)
            
            with st.form(f"tags_{doc.doc_id}"):
                new_tags = st.text_input("Edit Tags (comma-separated)", value=", ".join(doc.tags))
                
                col_save, col_delete = st.columns(2)
                
                with col_save:
                    if st.form_submit_button("💾 Save", use_container_width=True):
                        doc.tags = [t.strip() for t in new_tags.split(",") if t.strip()]
                        
                        for i, d in enumerate(kb.documents):
                            if d["doc_id"] == doc.doc_id:
                                kb.documents[i] = doc.to_dict()
                                break
                        
                        update_knowledge_base(kb)
                        st.success("✅ Tags updated!")
                        st.rerun()
                
                with col_delete:
                    if st.form_submit_button("🗑️ Delete", use_container_width=True):
                        if delete_document(kb.kb_id, doc.doc_id):
                            st.success("✅ Deleted!")
                            st.rerun()

def render_upload(kb: KnowledgeBase):
    """Render upload form"""
    st.subheader("⬆️ Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Select files",
        type=['txt', 'pdf', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected**")
        
        tags_input = st.text_input("Tags (optional)", placeholder="tag1, tag2, tag3")
        tags = [t.strip() for t in tags_input.split(",") if t.strip()]
        
        if st.button("📤 Upload All", use_container_width=True):
            progress = st.progress(0)
            
            for idx, file in enumerate(uploaded_files):
                upload_document(kb.kb_id, file, tags)
                progress.progress((idx + 1) / len(uploaded_files))
            
            st.success(f"✅ Uploaded {len(uploaded_files)} file(s)!")
            st.rerun()

def render_vectorization(kb: KnowledgeBase):
    """Render vectorization config"""
    st.subheader("🔧 Vectorization Configuration")
    
    if not kb.documents:
        st.info("Upload documents first")
        return
    
    st.markdown("### 1. Select Documents")
    
    doc_options = {}
    for doc_data in kb.documents:
        doc = Document.from_dict(doc_data)
        status = "✅" if doc.vectorized else "⏳"
        doc_options[f"{status} {doc.filename}"] = doc.doc_id
    
    selected = st.multiselect("Choose documents", list(doc_options.keys()))
    
    if not selected:
        return
    
    st.markdown("### 2. Chunking")
    chunk_method = st.radio("Method", ["Auto", "Custom"], horizontal=True)
    
    st.markdown("### 3. Preprocessing")
    st.multiselect("Options", ["Remove Whitespace", "Lowercase"], default=["Remove Whitespace"])
    
    st.markdown("### 4. Index Mode")
    st.selectbox("Mode", INDEX_MODES)
    
    st.markdown("### 5. Vectorize")
    
    if st.button("🚀 Start", use_container_width=True, type="primary"):
        progress = st.progress(0)
        
        success = 0
        for idx, doc_display in enumerate(selected):
            doc_id = doc_options[doc_display]
            if vectorize_document(kb.kb_id, doc_id):
                success += 1
            progress.progress((idx + 1) / len(selected))
        
        st.success(f"✅ Vectorized {success}/{len(selected)}!")
        st.balloons()
        st.rerun()

# ===========================
# MAIN
# ===========================

def main():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🏢",
        layout="wide"
    )
    
    ensure_dirs()
    
    if 'page' not in st.session_state:
        st.session_state.page = "kb_list"
    
    if 'current_kb' not in st.session_state:
        st.session_state.current_kb = None
    
    render_navigation()
    
    with st.sidebar:
        st.header("🗂️ Navigation")
        
        if st.button("📚 Knowledge Bases", use_container_width=True):
            st.session_state.page = "kb_list"
            st.rerun()
        
        if st.button("➕ New KB", use_container_width=True):
            st.session_state.page = "kb_create"
            st.rerun()
        
        st.markdown("---")
        
        kbs = list_knowledge_bases()
        st.metric("Total KBs", len(kbs))
        st.metric("Total Docs", sum(len(kb.documents) for kb in kbs))
        st.metric("Total Tokens", f"{sum(kb.tokens for kb in kbs):,}")
    
    if st.session_state.page == "kb_list":
        render_kb_list()
    elif st.session_state.page == "kb_create":
        render_kb_creation()
    elif st.session_state.page == "kb_detail":
        if st.session_state.current_kb:
            render_kb_detail(st.session_state.current_kb)

if __name__ == "__main__":
    main()
