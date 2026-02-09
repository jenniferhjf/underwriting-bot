"""
Professional Underwriting Assistant with Full RAG Pipeline
==========================================================

Features:
- Multi-format document extraction (PDF, DOCX, XLSX, PPTX)
- Text chunking with overlap
- Embedding generation (OpenAI/Local models)
- Vector database (FAISS/ChromaDB)
- Semantic search with reranking
- LLM integration with context
"""

import streamlit as st
import os
import json
from datetime import datetime
from pathlib import Path
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

# Document Processing
import PyPDF2
from docx import Document as DocxDocument
from pptx import Presentation
import openpyxl

# RAG Components
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import requests

# ===========================
# Configuration
# ===========================

VERSION = "3.0.0-RAG"
APP_TITLE = "Professional Underwriting Assistant - RAG System"

# API Configuration
DEFAULT_API_KEY = os.getenv("API_KEY", "sk-99bba2ce117444e197270f17d303e74f")
API_BASE = "https://api.deepseek.com/v1"
API_MODEL = "deepseek-chat"

# RAG Configuration
CHUNK_SIZE = 512  # Characters per chunk
CHUNK_OVERLAP = 128  # Overlap between chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
TOP_K_RETRIEVAL = 10  # Initial retrieval
TOP_K_RERANK = 3  # After reranking

# Directory Structure
DATA_DIR = Path("data")
WORKSPACES_DIR = DATA_DIR / "workspaces"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CHUNKS_DIR = DATA_DIR / "chunks"

# Supported formats
SUPPORTED_FORMATS = {
    'pdf': '📄 PDF',
    'docx': '📝 Word',
    'xlsx': '📊 Excel',
    'pptx': '📽️ PowerPoint',
    'txt': '📃 Text'
}

# ===========================
# Initialize Components
# ===========================

@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model for embeddings"""
    return SentenceTransformer(EMBEDDING_MODEL)

def ensure_dirs():
    """Create necessary directories"""
    for dir_path in [WORKSPACES_DIR, VECTOR_DB_DIR, EMBEDDINGS_DIR, CHUNKS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

# ===========================
# Document Extraction
# ===========================

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF"""
    try:
        text_parts = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")
        return "\n\n".join(text_parts)
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return ""

def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from DOCX including tables"""
    try:
        doc = DocxDocument(file_path)
        text_parts = []
        
        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        
        # Extract tables
        for table_idx, table in enumerate(doc.tables):
            text_parts.append(f"\n[Table {table_idx + 1}]")
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                text_parts.append(" | ".join(row_data))
        
        return "\n".join(text_parts)
    except Exception as e:
        st.error(f"DOCX extraction error: {e}")
        return ""

def extract_text_from_xlsx(file_path: Path) -> str:
    """Extract text from Excel"""
    try:
        wb = openpyxl.load_workbook(file_path)
        text_parts = []
        
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            text_parts.append(f"\n[Sheet: {sheet_name}]")
            
            for row in sheet.iter_rows(values_only=True):
                row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
                if row_text.strip():
                    text_parts.append(row_text)
        
        return "\n".join(text_parts)
    except Exception as e:
        st.error(f"Excel extraction error: {e}")
        return ""

def extract_text_from_pptx(file_path: Path) -> str:
    """Extract text from PowerPoint"""
    try:
        prs = Presentation(file_path)
        text_parts = []
        
        for slide_idx, slide in enumerate(prs.slides):
            text_parts.append(f"\n[Slide {slide_idx + 1}]")
            
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text_parts.append(shape.text)
        
        return "\n".join(text_parts)
    except Exception as e:
        st.error(f"PowerPoint extraction error: {e}")
        return ""

def extract_text_from_txt(file_path: Path) -> str:
    """Extract text from TXT"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        st.error(f"TXT extraction error: {e}")
        return ""

def extract_text_from_file(file_path: Path) -> str:
    """Route extraction based on file type"""
    ext = file_path.suffix.lower().lstrip('.')
    
    extractors = {
        'pdf': extract_text_from_pdf,
        'docx': extract_text_from_docx,
        'xlsx': extract_text_from_xlsx,
        'pptx': extract_text_from_pptx,
        'txt': extract_text_from_txt
    }
    
    extractor = extractors.get(ext)
    if extractor:
        return extractor(file_path)
    else:
        st.error(f"Unsupported format: {ext}")
        return ""

# ===========================
# Text Chunking
# ===========================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks
    
    Returns:
        List of chunks with metadata
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    chunk_id = 0
    
    while start < text_length:
        end = start + chunk_size
        
        # Extract chunk
        chunk_text = text[start:end]
        
        # Try to break at sentence boundary
        if end < text_length:
            # Look for sentence ending
            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size * 0.5:  # At least 50% of chunk
                end = start + break_point + 1
                chunk_text = text[start:end]
        
        # Create chunk metadata
        chunk = {
            'chunk_id': chunk_id,
            'text': chunk_text.strip(),
            'start_char': start,
            'end_char': end,
            'length': len(chunk_text)
        }
        
        chunks.append(chunk)
        
        # Move to next chunk with overlap
        start = end - overlap
        chunk_id += 1
    
    return chunks

# ===========================
# Embedding Generation
# ===========================

def generate_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Generate embeddings for a list of texts
    
    Args:
        texts: List of text strings
        model: Sentence transformer model
        
    Returns:
        Numpy array of embeddings
    """
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

# ===========================
# Vector Database (FAISS)
# ===========================

class VectorDatabase:
    """FAISS-based vector database for semantic search"""
    
    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS index
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        self.chunks = []
        self.doc_metadata = {}
    
    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray, doc_id: str, doc_name: str):
        """
        Add document chunks to vector database
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Numpy array of embeddings
            doc_id: Document identifier
            doc_name: Document name
        """
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store chunks with metadata
        for chunk, embedding in zip(chunks, embeddings):
            chunk['doc_id'] = doc_id
            chunk['doc_name'] = doc_name
            chunk['embedding_norm'] = np.linalg.norm(embedding)
            self.chunks.append(chunk)
        
        # Store document metadata
        self.doc_metadata[doc_id] = {
            'doc_name': doc_name,
            'num_chunks': len(chunks),
            'added_at': datetime.now().isoformat()
        }
    
    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[Dict, float]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Prepare results
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(similarity)))
        
        return results
    
    def save(self, workspace_name: str):
        """Save index and metadata"""
        save_dir = VECTOR_DB_DIR / workspace_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_dir / "faiss.index"))
        
        # Save chunks and metadata
        with open(save_dir / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        with open(save_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.doc_metadata, f, indent=2)
    
    def load(self, workspace_name: str):
        """Load index and metadata"""
        load_dir = VECTOR_DB_DIR / workspace_name
        
        if not load_dir.exists():
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(load_dir / "faiss.index"))
            
            # Load chunks
            with open(load_dir / "chunks.json", 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            # Load metadata
            with open(load_dir / "metadata.json", 'r', encoding='utf-8') as f:
                self.doc_metadata = json.load(f)
            
            return True
        except Exception as e:
            st.error(f"Error loading vector database: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'total_chunks': self.index.ntotal,
            'total_documents': len(self.doc_metadata),
            'dimension': self.dimension
        }

# ===========================
# Reranking
# ===========================

class BM25Reranker:
    """BM25-based reranker for retrieved chunks"""
    
    def __init__(self):
        self.bm25 = None
        self.corpus_texts = []
    
    def fit(self, texts: List[str]):
        """Fit BM25 on corpus"""
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_texts = texts
    
    def rerank(self, query: str, candidates: List[Tuple[Dict, float]], top_k: int = TOP_K_RERANK) -> List[Tuple[Dict, float]]:
        """
        Rerank candidates using BM25
        
        Args:
            query: Query string
            candidates: List of (chunk, score) tuples
            top_k: Number of results to return
            
        Returns:
            Reranked list of (chunk, combined_score) tuples
        """
        if not candidates or not self.bm25:
            return candidates[:top_k]
        
        # Get BM25 scores
        tokenized_query = query.lower().split()
        candidate_texts = [chunk['text'] for chunk, _ in candidates]
        
        # Fit BM25 on candidates if needed
        if not self.corpus_texts or set(candidate_texts) != set(self.corpus_texts):
            self.fit(candidate_texts)
        
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Combine with vector similarity scores
        combined_results = []
        for (chunk, vec_score), bm25_score in zip(candidates, bm25_scores):
            # Weighted combination: 60% vector similarity, 40% BM25
            combined_score = 0.6 * vec_score + 0.4 * (bm25_score / (max(bm25_scores) + 1e-6))
            combined_results.append((chunk, combined_score))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results[:top_k]

# ===========================
# Prompt Template
# ===========================

def create_rag_prompt(query: str, context_chunks: List[Tuple[Dict, float]]) -> str:
    """
    Create prompt with retrieved context
    
    Args:
        query: User query
        context_chunks: List of (chunk, score) tuples
        
    Returns:
        Formatted prompt string
    """
    # Build context section
    context_parts = []
    for idx, (chunk, score) in enumerate(context_chunks, 1):
        context_parts.append(f"""
Document: {chunk['doc_name']}
Chunk {chunk['chunk_id']} (Relevance: {score:.3f})
---
{chunk['text']}
---
""")
    
    context_text = "\n".join(context_parts)
    
    # Create full prompt
    prompt = f"""You are a professional underwriting assistant. Use the following retrieved context to answer the user's question.

RETRIEVED CONTEXT:
{context_text}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Cite the specific document and chunk number when referencing information
3. If the context doesn't contain enough information, say so clearly
4. Provide specific details like amounts, dates, and terms when available
5. Keep your answer concise and professional

ANSWER:"""
    
    return prompt

# ===========================
# LLM API Call
# ===========================

def call_llm_api(prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> str:
    """Call LLM API with context"""
    try:
        api_key = st.session_state.get('api_key', DEFAULT_API_KEY)
        
        if not api_key:
            return "Error: API key not configured"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": API_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
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
        
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content']
        
    except Exception as e:
        return f"Error calling LLM: {e}"

# ===========================
# RAG Pipeline
# ===========================

def rag_query(query: str, vector_db: VectorDatabase, embedding_model: SentenceTransformer, 
              reranker: BM25Reranker) -> Dict[str, Any]:
    """
    Complete RAG pipeline
    
    Args:
        query: User query
        vector_db: Vector database instance
        embedding_model: Embedding model
        reranker: Reranker instance
        
    Returns:
        Dictionary with answer and metadata
    """
    # Step 1: Generate query embedding
    query_embedding = embedding_model.encode([query])[0]
    
    # Step 2: Retrieve top-k chunks
    candidates = vector_db.search(query_embedding, top_k=TOP_K_RETRIEVAL)
    
    if not candidates:
        return {
            'answer': "No relevant documents found in the database.",
            'sources': [],
            'num_candidates': 0,
            'num_reranked': 0
        }
    
    # Step 3: Rerank
    reranked_chunks = reranker.rerank(query, candidates, top_k=TOP_K_RERANK)
    
    # Step 4: Create prompt
    prompt = create_rag_prompt(query, reranked_chunks)
    
    # Step 5: Generate answer with LLM
    answer = call_llm_api(prompt)
    
    # Prepare result
    sources = [
        {
            'doc_name': chunk['doc_name'],
            'chunk_id': chunk['chunk_id'],
            'score': score,
            'text_preview': chunk['text'][:200] + "..."
        }
        for chunk, score in reranked_chunks
    ]
    
    return {
        'answer': answer,
        'sources': sources,
        'num_candidates': len(candidates),
        'num_reranked': len(reranked_chunks),
        'prompt_tokens': len(prompt.split())
    }

# ===========================
# Workspace Management
# ===========================

def create_workspace(name: str):
    """Create new workspace"""
    workspace_dir = WORKSPACES_DIR / name
    workspace_dir.mkdir(exist_ok=True)
    
    metadata = {
        "name": name,
        "created_at": datetime.now().isoformat(),
        "documents": []
    }
    
    with open(workspace_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def load_workspace(name: str) -> Optional[Dict]:
    """Load workspace metadata"""
    workspace_dir = WORKSPACES_DIR / name
    metadata_file = workspace_dir / "metadata.json"
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return None

def list_workspaces() -> List[str]:
    """List all workspaces"""
    if not WORKSPACES_DIR.exists():
        return []
    return [d.name for d in WORKSPACES_DIR.iterdir() if d.is_dir()]

def add_document_to_workspace(workspace_name: str, uploaded_file, 
                               vector_db: VectorDatabase, embedding_model: SentenceTransformer) -> bool:
    """
    Process and add document to workspace
    
    Full RAG pipeline:
    1. Extract text from file
    2. Chunk text
    3. Generate embeddings
    4. Index in vector database
    """
    try:
        # Save file
        workspace_dir = WORKSPACES_DIR / workspace_name
        file_path = workspace_dir / uploaded_file.name
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Step 1: Extract text
        st.info("📄 Step 1/4: Extracting text...")
        text = extract_text_from_file(file_path)
        
        if not text:
            st.error("No text extracted from document")
            return False
        
        st.success(f"✓ Extracted {len(text)} characters")
        
        # Step 2: Chunk text
        st.info("✂️ Step 2/4: Chunking text...")
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        st.success(f"✓ Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings
        st.info("🔢 Step 3/4: Generating embeddings...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = generate_embeddings(chunk_texts, embedding_model)
        st.success(f"✓ Generated {len(embeddings)} embeddings")
        
        # Step 4: Add to vector database
        st.info("💾 Step 4/4: Indexing in vector database...")
        doc_id = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
        vector_db.add_documents(chunks, embeddings, doc_id, uploaded_file.name)
        vector_db.save(workspace_name)
        st.success(f"✓ Indexed document: {uploaded_file.name}")
        
        # Update workspace metadata
        metadata = load_workspace(workspace_name)
        metadata['documents'].append({
            'filename': uploaded_file.name,
            'doc_id': doc_id,
            'size': uploaded_file.size,
            'num_chunks': len(chunks),
            'upload_date': datetime.now().isoformat()
        })
        
        with open(workspace_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
        
    except Exception as e:
        st.error(f"Error processing document: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False

# ===========================
# Streamlit UI
# ===========================

def render_header():
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, #1e40af 0%, #7c3aed 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>📋 {APP_TITLE}</h1>
        <p style='color: #e0e7ff; margin-top: 0.5rem;'>Version {VERSION} | Full RAG Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="RAG Underwriting Assistant",
        page_icon="📋",
        layout="wide"
    )
    
    ensure_dirs()
    render_header()
    
    # Initialize session state
    if 'current_workspace' not in st.session_state:
        st.session_state.current_workspace = "Default"
    
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = VectorDatabase()
    
    if 'reranker' not in st.session_state:
        st.session_state.reranker = BM25Reranker()
    
    # Load embedding model
    embedding_model = load_embedding_model()
    
    # Sidebar
    with st.sidebar:
        st.header("🗂️ Workspace")
        
        workspaces = list_workspaces()
        if not workspaces:
            create_workspace("Default")
            workspaces = ["Default"]
        
        selected_workspace = st.selectbox(
            "Select Workspace:",
            workspaces,
            index=workspaces.index(st.session_state.current_workspace) if st.session_state.current_workspace in workspaces else 0
        )
        
        if selected_workspace != st.session_state.current_workspace:
            st.session_state.current_workspace = selected_workspace
            # Load vector DB for this workspace
            st.session_state.vector_db = VectorDatabase()
            st.session_state.vector_db.load(selected_workspace)
            st.rerun()
        
        st.markdown("---")
        
        # Create new workspace
        with st.expander("➕ New Workspace"):
            new_name = st.text_input("Workspace Name:")
            if st.button("Create"):
                if new_name:
                    create_workspace(new_name)
                    st.success(f"Created: {new_name}")
                    st.rerun()
        
        st.markdown("---")
        
        # Vector DB stats
        stats = st.session_state.vector_db.get_stats()
        st.metric("Documents", stats['total_documents'])
        st.metric("Chunks", stats['total_chunks'])
        st.metric("Dimension", stats['dimension'])
        
        st.markdown("---")
        
        # API Config
        with st.expander("⚙️ API Key"):
            api_key = st.text_input("DeepSeek API Key:", type="password", value=DEFAULT_API_KEY)
            if st.button("Save API Key"):
                st.session_state.api_key = api_key
                st.success("✓ Saved")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Query (RAG)", "⬆️ Upload Documents", "📊 Document Explorer"])
    
    # TAB 1: RAG Query
    with tab1:
        st.header("💬 Query with RAG")
        
        if stats['total_documents'] == 0:
            st.warning("⚠️ No documents in database. Please upload documents first.")
        else:
            st.info(f"📚 Ready to query {stats['total_documents']} document(s) with {stats['total_chunks']} chunks")
            
            query = st.text_area(
                "Enter your question:",
                placeholder="e.g., What are the key terms of the MSC insurance policy?",
                height=100
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                top_k_retrieval = st.slider("Initial Retrieval (Top-K)", 5, 20, TOP_K_RETRIEVAL)
            with col2:
                top_k_rerank = st.slider("After Reranking (Top-K)", 1, 10, TOP_K_RERANK)
            with col3:
                temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.3)
            
            if st.button("🔍 Search & Answer", type="primary"):
                if not query:
                    st.warning("Please enter a question")
                else:
                    with st.spinner("Running RAG pipeline..."):
                        result = rag_query(
                            query,
                            st.session_state.vector_db,
                            embedding_model,
                            st.session_state.reranker
                        )
                        
                        # Display answer
                        st.markdown("### 💡 Answer")
                        st.markdown(result['answer'])
                        
                        st.markdown("---")
                        
                        # Display sources
                        st.markdown("### 📚 Sources")
                        st.caption(f"Retrieved {result['num_candidates']} candidates → Reranked to {result['num_reranked']}")
                        
                        for idx, source in enumerate(result['sources'], 1):
                            with st.expander(f"Source {idx}: {source['doc_name']} (Score: {source['score']:.3f})"):
                                st.markdown(f"**Chunk ID:** {source['chunk_id']}")
                                st.markdown(f"**Preview:**\n{source['text_preview']}")
                        
                        # Display stats
                        with st.expander("📊 Pipeline Stats"):
                            st.json({
                                'Candidates Retrieved': result['num_candidates'],
                                'Top-K After Reranking': result['num_reranked'],
                                'Prompt Tokens (approx)': result['prompt_tokens']
                            })
    
    # TAB 2: Upload
    with tab2:
        st.header("⬆️ Upload Documents")
        
        st.markdown("""
        Upload documents to build your knowledge base. Supported formats:
        - 📄 PDF
        - 📝 Word (.docx)
        - 📊 Excel (.xlsx)
        - 📽️ PowerPoint (.pptx)
        - 📃 Text (.txt)
        """)
        
        uploaded_files = st.file_uploader(
            "Choose files:",
            type=list(SUPPORTED_FORMATS.keys()),
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} file(s)")
            
            if st.button("📤 Process & Index"):
                progress = st.progress(0)
                
                for idx, file in enumerate(uploaded_files):
                    st.markdown(f"### Processing: {file.name}")
                    
                    success = add_document_to_workspace(
                        st.session_state.current_workspace,
                        file,
                        st.session_state.vector_db,
                        embedding_model
                    )
                    
                    if success:
                        st.success(f"✅ Completed: {file.name}")
                    else:
                        st.error(f"❌ Failed: {file.name}")
                    
                    progress.progress((idx + 1) / len(uploaded_files))
                
                st.balloons()
                st.info("🔄 Reloading workspace...")
                st.rerun()
    
    # TAB 3: Explorer
    with tab3:
        st.header("📊 Document Explorer")
        
        metadata = load_workspace(st.session_state.current_workspace)
        
        if not metadata or not metadata.get('documents'):
            st.info("No documents yet")
        else:
            docs = metadata['documents']
            
            # Display as table
            df = pd.DataFrame(docs)
            st.dataframe(
                df[['filename', 'num_chunks', 'size', 'upload_date']],
                use_container_width=True
            )
            
            # Document details
            st.markdown("---")
            st.subheader("Document Details")
            
            for doc in docs:
                with st.expander(f"📄 {doc['filename']}"):
                    st.json(doc)

if __name__ == "__main__":
    main()
