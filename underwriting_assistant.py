"""
Enhanced Underwriting Assistant - Enterprise Knowledge Management System
Complete knowledge base management with navigation, CRUD operations, and vectorization

========================================
IMPORTANT: API KEY CONFIGURATION
========================================
API key is configured at line 50.
If authentication fails, update the API_KEY value:

API_KEY = "your-deepseek-api-key-here"

Get your API key from: https://platform.deepseek.com/api_keys
========================================
"""

import streamlit as st
import json
import os
import hashlib
import re
import traceback
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
import requests
import pandas as pd

# Vector database imports
import numpy as np
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ===========================
# CONFIGURATION
# ===========================

VERSION = "4.0.0"
APP_NAME = "Underwriting Knowledge Management System"

# API Configuration - HARDCODED
API_KEY = "sk-99bba2ce117444e197270f17d303e74f"
API_BASE = "https://api.deepseek.com/v1"
API_MODEL = "deepseek-chat"

# Directory Structure
DATA_DIR = Path("data")
KB_DIR = DATA_DIR / "knowledge_bases"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
CHUNKS_DIR = DATA_DIR / "chunks"
PROMPTS_DIR = DATA_DIR / "prompts"
APPS_DIR = DATA_DIR / "applications"
CONFIG_DIR = DATA_DIR / "config"

# Default datasets to load
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

# Knowledge categories
CATEGORIES = ["Hull", "Cargo", "Liability", "Property", "Marine", "Other"]

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_DIM = 1536

# Supported file formats
SUPPORTED_FORMATS = {
    'pdf': '📄 PDF',
    'docx': '📝 Word',
    'doc': '📝 Word',
    'txt': '📃 Text',
    'csv': '📊 CSV',
    'xlsx': '📊 Excel'
}

# Index modes
INDEX_MODES = ["Vector Search", "Keyword Search", "Hybrid Search"]

# ===========================
# UTILITY FUNCTIONS
# ===========================

def ensure_dirs():
    """Create necessary directories"""
    for dir_path in [KB_DIR, VECTOR_DB_DIR, CHUNKS_DIR, PROMPTS_DIR, APPS_DIR, CONFIG_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

def get_api_key() -> str:
    """Get API key"""
    env_key = os.getenv("API_KEY")
    if env_key:
        return env_key
    
    if 'api_key' in st.session_state and st.session_state.api_key:
        return st.session_state.api_key
    
    return API_KEY

def call_llm_api(system_prompt: str, user_prompt: str, 
                 temperature: float = 0.3, max_tokens: int = 2000) -> str:
    """Call LLM API for text generation"""
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
            st.error(f"⚠️ API Authentication Failed! Update API_KEY at line 50 in app.py")
            return ""
        
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content']
        
    except Exception as e:
        st.error(f"API Error: {e}")
        return ""

# ===========================
# KNOWLEDGE BASE MODELS
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
        kb = cls(
            data["kb_id"],
            data["name"],
            data["description"],
            data["category"]
        )
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
        doc = cls(
            data["doc_id"],
            data["filename"],
            data["kb_id"]
        )
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
    
    # Create directory
    kb_dir = KB_DIR / kb_id
    kb_dir.mkdir(exist_ok=True)
    
    # Save metadata
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
    """Update knowledge base metadata"""
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
        
        # Delete vector DB
        vector_file = VECTOR_DB_DIR / f"{kb_id}_vectordb.json"
        if vector_file.exists():
            vector_file.unlink()
        
        return True
    except Exception as e:
        st.error(f"Error deleting KB: {e}")
        return False

# ===========================
# DOCUMENT MANAGEMENT
# ===========================

def upload_document(kb_id: str, uploaded_file, tags: List[str] = None) -> Optional[Document]:
    """Upload document to knowledge base"""
    try:
        doc_id = hashlib.md5(f"{uploaded_file.name}{datetime.now()}".encode()).hexdigest()[:16]
        doc = Document(doc_id, uploaded_file.name, kb_id)
        
        if tags:
            doc.tags = tags
        
        # Save file
        kb_dir = KB_DIR / kb_id
        doc_dir = kb_dir / "documents"
        doc_dir.mkdir(exist_ok=True)
        
        file_path = doc_dir / uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        doc.file_path = str(file_path)
        doc.file_size = uploaded_file.size
        
        # Auto-generate tags
        if not tags:
            doc.tags = auto_generate_tags(uploaded_file.name, "")
        
        # Update KB
        kb = load_knowledge_base(kb_id)
        if kb:
            kb.documents.append(doc.to_dict())
            update_knowledge_base(kb)
        
        return doc
        
    except Exception as e:
        st.error(f"Error uploading document: {e}")
        return None

def load_document(kb_id: str, doc_id: str) -> Optional[Document]:
    """Load document by ID"""
    kb = load_knowledge_base(kb_id)
    if not kb:
        return None
    
    for doc_data in kb.documents:
        if doc_data["doc_id"] == doc_id:
            return Document.from_dict(doc_data)
    
    return None

def delete_document(kb_id: str, doc_id: str) -> bool:
    """Delete document from knowledge base"""
    try:
        kb = load_knowledge_base(kb_id)
        if not kb:
            return False
        
        # Find and remove document
        doc = None
        for d in kb.documents:
            if d["doc_id"] == doc_id:
                doc = d
                break
        
        if not doc:
            return False
        
        # Delete file
        file_path = Path(doc["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        # Remove from KB
        kb.documents = [d for d in kb.documents if d["doc_id"] != doc_id]
        update_knowledge_base(kb)
        
        return True
        
    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return False

def auto_generate_tags(filename: str, content: str) -> List[str]:
    """Auto-generate tags for document"""
    tags = []
    
    # Extract from filename
    name_parts = re.split(r'[\s_\-\.]+', filename)
    for part in name_parts:
        if len(part) > 2 and not part.isdigit():
            tags.append(part.title())
    
    # Extract years
    years = re.findall(r'20\d{2}', filename)
    tags.extend(years)
    
    return list(set(tags))[:10]

# ===========================
# TEXT EXTRACTION
# ===========================

def extract_text_from_file(file_path: Path) -> Tuple[str, List[Dict]]:
    """Extract text and images from file"""
    ext = file_path.suffix.lower().lstrip('.')
    
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['docx', 'doc']:
        return extract_text_from_docx(file_path)
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return (f.read(), [])
    else:
        return ("", [])

def extract_text_from_pdf(file_path: Path) -> Tuple[str, List[Dict]]:
    """Extract text from PDF"""
    try:
        import fitz
        doc = fitz.open(file_path)
        
        text_content = []
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_content.append(text)
            
            # Extract images
            for img_index, img in enumerate(page.get_images()):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    images.append({
                        'page': page_num + 1,
                        'size': len(base_image["image"])
                    })
                except:
                    continue
        
        doc.close()
        return ("\n\n".join(text_content), images)
        
    except Exception as e:
        st.warning(f"PDF extraction error: {e}")
        return ("", [])

def extract_text_from_docx(file_path: Path) -> Tuple[str, List[Dict]]:
    """Extract text from DOCX"""
    try:
        from docx import Document
        doc = Document(file_path)
        text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return (text, [])
    except Exception as e:
        st.warning(f"DOCX extraction error: {e}")
        return ("", [])

# ===========================
# CHUNKING & VECTORIZATION
# ===========================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Split text into chunks"""
    if not text:
        return []
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        # Try to end at sentence boundary
        if end < len(text):
            for delimiter in ['. ', '.\n', '! ', '? ']:
                last_delim = chunk_text.rfind(delimiter)
                if last_delim > chunk_size * 0.7:
                    end = start + last_delim + 1
                    chunk_text = text[start:end]
                    break
        
        chunks.append({
            'id': f"chunk_{chunk_id}",
            'text': chunk_text.strip(),
            'start': start,
            'end': end
        })
        
        chunk_id += 1
        start = end - overlap
    
    return chunks

def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding vector"""
    text_lower = text.lower().strip()
    hash_obj = hashlib.sha256(text_lower.encode())
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

def vectorize_document(kb_id: str, doc_id: str, index_mode: str = "Vector Search") -> bool:
    """Vectorize document and add to vector database"""
    try:
        doc = load_document(kb_id, doc_id)
        if not doc:
            return False
        
        # Extract text
        file_path = Path(doc.file_path)
        text, images = extract_text_from_file(file_path)
        
        if not text:
            st.warning(f"No text extracted from {doc.filename}")
            return False
        
        # Chunk text
        chunks = chunk_text(text)
        doc.chunk_count = len(chunks)
        
        # Generate embeddings
        embeddings = []
        for chunk in chunks:
            emb = generate_embedding(chunk['text'])
            embeddings.append(emb)
        
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
            vector_data = {"chunks": [], "embeddings": [], "doc_map": {}}
        
        # Add new data
        for chunk, emb in zip(chunks, embeddings):
            chunk_id = f"{doc_id}_{chunk['id']}"
            vector_data["chunks"].append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": chunk['text']
            })
            vector_data["embeddings"].append(emb.tolist())
            vector_data["doc_map"][chunk_id] = doc_id
        
        # Save vector DB
        with open(vector_file, 'w') as f:
            json.dump(vector_data, f)
        
        # Update document
        doc.vectorized = True
        kb = load_knowledge_base(kb_id)
        for i, d in enumerate(kb.documents):
            if d["doc_id"] == doc_id:
                kb.documents[i] = doc.to_dict()
                break
        
        # Update tokens
        kb.tokens += len(text.split())
        update_knowledge_base(kb)
        
        return True
        
    except Exception as e:
        st.error(f"Vectorization error: {e}")
        traceback.print_exc()
        return False

# ===========================
# DEFAULT DATASET LOADING
# ===========================

def load_default_datasets():
    """Load default datasets into Default knowledge base"""
    try:
        # Create or load Default KB
        kbs = list_knowledge_bases()
        default_kb = None
        
        for kb in kbs:
            if kb.name == "Default":
                default_kb = kb
                break
        
        if not default_kb:
            default_kb = create_knowledge_base(
                "Default",
                "Default knowledge base with sample datasets",
                "Mixed"
            )
        
        # Check which files need to be loaded
        existing_files = [doc["filename"] for doc in default_kb.documents]
        
        loaded_count = 0
        for dataset in DEFAULT_DATASETS:
            filename = dataset["filename"]
            
            if filename in existing_files:
                continue
            
            if not Path(filename).exists():
                continue
            
            # Create uploaded file object
            with open(filename, 'rb') as f:
                file_content = f.read()
            
            class UploadedFile:
                def __init__(self, name, content):
                    self.name = name
                    self.size = len(content)
                    self._content = content
                
                def getvalue(self):
                    return self._content
            
            uploaded_file = UploadedFile(filename, file_content)
            
            # Upload document
            doc = upload_document(default_kb.kb_id, uploaded_file, dataset["tags"])
            
            if doc:
                loaded_count += 1
        
        return loaded_count > 0
        
    except Exception as e:
        st.error(f"Error loading default datasets: {e}")
        return False

# ===========================
# UI COMPONENTS
# ===========================

def render_navigation():
    """Render navigation bar"""
    st.markdown("""
    <style>
    .nav-container {
        background: linear-gradient(90deg, #1e40af 0%, #7c3aed 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .nav-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .nav-subtitle {
        color: #e0e7ff;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="nav-container">
        <div class="nav-title">🏢 {APP_NAME}</div>
        <div class="nav-subtitle">Version {VERSION} | Enterprise Knowledge Management</div>
    </div>
    """, unsafe_allow_html=True)

def render_kb_list():
    """Render knowledge base list"""
    st.subheader("📚 Knowledge Base Management")
    
    kbs = list_knowledge_bases()
    
    if not kbs:
        st.info("No knowledge bases found. Create one to get started!")
        return
    
    # Create DataFrame
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
    
    # Display table
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Actions
    st.markdown("---")
    
    cols = st.columns(len(kbs))
    for idx, (col, kb) in enumerate(zip(cols, kbs)):
        with col:
            if st.button(f"📂 {kb.name}", key=f"view_kb_{kb.kb_id}", use_container_width=True):
                st.session_state.current_kb = kb.kb_id
                st.session_state.page = "kb_detail"
                st.rerun()

def render_kb_creation():
    """Render knowledge base creation form"""
    st.subheader("➕ Create New Knowledge Base")
    
    with st.form("create_kb_form"):
        name = st.text_input("Knowledge Base Name*", placeholder="e.g., Marine Insurance KB")
        description = st.text_area("Description*", placeholder="Brief description of this knowledge base")
        category = st.selectbox("Category*", CATEGORIES)
        
        submitted = st.form_submit_button("Create Knowledge Base", use_container_width=True)
        
        if submitted:
            if not name or not description:
                st.error("Please fill in all required fields")
            else:
                kb = create_knowledge_base(name, description, category)
                st.success(f"✅ Knowledge base '{name}' created successfully!")
                st.session_state.current_kb = kb.kb_id
                st.session_state.page = "kb_detail"
                st.rerun()

def render_kb_detail(kb_id: str):
    """Render knowledge base detail page"""
    kb = load_knowledge_base(kb_id)
    if not kb:
        st.error("Knowledge base not found")
        return
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"📚 {kb.name}")
        st.caption(kb.description)
    
    with col2:
        if st.button("🔙 Back to List"):
            st.session_state.page = "kb_list"
            st.rerun()
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents", len(kb.documents))
    with col2:
        st.metric("Tokens", f"{kb.tokens:,}")
    with col3:
        st.metric("Category", kb.category)
    with col4:
        vectorized_count = sum(1 for d in kb.documents if d.get("vectorized", False))
        st.metric("Vectorized", f"{vectorized_count}/{len(kb.documents)}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📄 Documents", "⬆️ Upload", "🔧 Vectorization"])
    
    with tab1:
        render_document_list(kb)
    
    with tab2:
        render_document_upload(kb)
    
    with tab3:
        render_vectorization(kb)

def render_document_list(kb: KnowledgeBase):
    """Render document list"""
    if not kb.documents:
        st.info("No documents in this knowledge base. Upload some to get started!")
        return
    
    st.subheader("📄 Documents")
    
    for doc_data in kb.documents:
        doc = Document.from_dict(doc_data)
        
        with st.expander(f"📄 {doc.filename}"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Upload Time", doc.upload_time[:10])
            with col2:
                st.metric("Size", f"{doc.file_size / 1024:.1f} KB")
            with col3:
                st.metric("Chunks", doc.chunk_count)
            with col4:
                status = "✅ Yes" if doc.vectorized else "❌ No"
                st.metric("Vectorized", status)
            
            # Tags
            st.markdown("**Tags:**")
            if doc.tags:
                tags_html = " ".join([f'<span style="background:#3b82f6;color:white;padding:0.2rem 0.5rem;border-radius:5px;margin:0.2rem;">{tag}</span>' for tag in doc.tags])
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.caption("No tags")
            
            # Edit tags
            with st.form(f"edit_tags_{doc.doc_id}"):
                new_tags = st.text_input("Edit Tags (comma-separated)", value=", ".join(doc.tags))
                
                col_save, col_delete = st.columns(2)
                
                with col_save:
                    if st.form_submit_button("💾 Save Tags", use_container_width=True):
                        doc.tags = [t.strip() for t in new_tags.split(",") if t.strip()]
                        
                        # Update KB
                        for i, d in enumerate(kb.documents):
                            if d["doc_id"] == doc.doc_id:
                                kb.documents[i] = doc.to_dict()
                                break
                        
                        update_knowledge_base(kb)
                        st.success("✅ Tags updated!")
                        st.rerun()
                
                with col_delete:
                    if st.form_submit_button("🗑️ Delete Document", use_container_width=True):
                        if delete_document(kb.kb_id, doc.doc_id):
                            st.success(f"✅ Deleted {doc.filename}")
                            st.rerun()

def render_document_upload(kb: KnowledgeBase):
    """Render document upload form"""
    st.subheader("⬆️ Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Select files to upload",
        type=list(SUPPORTED_FORMATS.keys()),
        accept_multiple_files=True,
        help="Supported formats: PDF, Word, TXT, CSV, Excel"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected**")
        
        # Tag input
        tags_input = st.text_input("Tags (comma-separated, optional)", placeholder="e.g., Hull, 2024, Policy")
        tags = [t.strip() for t in tags_input.split(",") if t.strip()]
        
        if st.button("📤 Upload All", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            uploaded_count = 0
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Uploading {file.name}...")
                
                doc = upload_document(kb.kb_id, file, tags)
                if doc:
                    uploaded_count += 1
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.empty()
            progress_bar.empty()
            
            st.success(f"✅ Uploaded {uploaded_count} document(s) successfully!")
            st.rerun()

def render_vectorization(kb: KnowledgeBase):
    """Render vectorization configuration"""
    st.subheader("🔧 Vectorization Configuration")
    
    if not kb.documents:
        st.info("Upload documents first before vectorization")
        return
    
    # Select documents
    st.markdown("### 1. Select Documents")
    
    doc_options = {}
    for doc_data in kb.documents:
        doc = Document.from_dict(doc_data)
        status = "✅" if doc.vectorized else "⏳"
        doc_options[f"{status} {doc.filename}"] = doc.doc_id
    
    selected_docs = st.multiselect(
        "Choose documents to vectorize",
        options=list(doc_options.keys()),
        help="Select one or more documents"
    )
    
    if not selected_docs:
        return
    
    # Chunking configuration
    st.markdown("### 2. Chunking Configuration")
    
    chunk_method = st.radio(
        "Chunking Method",
        ["Auto Chunking", "Custom Chunking"],
        horizontal=True
    )
    
    if chunk_method == "Custom Chunking":
        col1, col2 = st.columns(2)
        with col1:
            custom_chunk_size = st.number_input("Chunk Size", value=CHUNK_SIZE, min_value=100, max_value=2000)
        with col2:
            custom_overlap = st.number_input("Overlap Size", value=CHUNK_OVERLAP, min_value=0, max_value=200)
    
    # Data preprocessing
    st.markdown("### 3. Data Preprocessing")
    
    preprocessing_options = st.multiselect(
        "Preprocessing Steps",
        ["Remove Extra Whitespace", "Remove Special Characters", "Lowercase Conversion"],
        default=["Remove Extra Whitespace"]
    )
    
    # Index configuration
    st.markdown("### 4. Index Configuration")
    
    index_mode = st.selectbox("Index Mode", INDEX_MODES)
    
    # Vectorization
    st.markdown("### 5. Start Vectorization")
    
    if st.button("🚀 Start Vectorization", use_container_width=True, type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        success_count = 0
        for idx, doc_display in enumerate(selected_docs):
            doc_id = doc_options[doc_display]
            
            status_text.text(f"Vectorizing {doc_display}...")
            
            if vectorize_document(kb.kb_id, doc_id, index_mode):
                success_count += 1
            
            progress_bar.progress((idx + 1) / len(selected_docs))
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Vectorized {success_count}/{len(selected_docs)} document(s)!")
        st.balloons()
        st.rerun()

# ===========================
# MAIN APPLICATION
# ===========================

def main():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    ensure_dirs()
    
    # Session state
    if 'page' not in st.session_state:
        st.session_state.page = "kb_list"
    
    if 'current_kb' not in st.session_state:
        st.session_state.current_kb = None
    
    if 'initial_load_done' not in st.session_state:
        st.session_state.initial_load_done = False
    
    # Navigation
    render_navigation()
    
    # Sidebar
    with st.sidebar:
        st.header("🗂️ Navigation")
        
        if st.button("📚 Knowledge Bases", use_container_width=True):
            st.session_state.page = "kb_list"
            st.rerun()
        
        if st.button("➕ New Knowledge Base", use_container_width=True):
            st.session_state.page = "kb_create"
            st.rerun()
        
        st.markdown("---")
        
        # Statistics
        kbs = list_knowledge_bases()
        st.metric("Total KBs", len(kbs))
        
        total_docs = sum(len(kb.documents) for kb in kbs)
        st.metric("Total Documents", total_docs)
        
        total_tokens = sum(kb.tokens for kb in kbs)
        st.metric("Total Tokens", f"{total_tokens:,}")
    
    # Load default datasets on first run
    if not st.session_state.initial_load_done:
        with st.spinner("Loading default datasets..."):
            load_default_datasets()
        st.session_state.initial_load_done = True
    
    # Page routing
    if st.session_state.page == "kb_list":
        render_kb_list()
    
    elif st.session_state.page == "kb_create":
        render_kb_creation()
    
    elif st.session_state.page == "kb_detail":
        if st.session_state.current_kb:
            render_kb_detail(st.session_state.current_kb)
        else:
            st.session_state.page = "kb_list"
            st.rerun()

if __name__ == "__main__":
    main()
