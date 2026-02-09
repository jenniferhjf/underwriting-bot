"""
Professional Underwriting RAG System with Handwriting Recognition
=================================================================

Complete Pipeline:
1. Load preset datasets
2. Detect handwritten vs printed text
3. OCR processing for both types
4. Text chunking with overlap
5. Embedding generation
6. Vector database indexing
7. Semantic search with reranking
8. LLM-powered question answering
"""

import streamlit as st
import os
import json
from datetime import datetime
from pathlib import Path
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import io

# Document & Image Processing
import PyPDF2
from pdf2image import convert_from_path
from docx import Document as DocxDocument
from PIL import Image
import cv2
import numpy as np

# OCR
import easyocr
import pytesseract

# RAG Components
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import requests
import pandas as pd

# ===========================
# Configuration
# ===========================

VERSION = "4.0.0-FINAL"
APP_TITLE = "Underwriting Repository - RAG with Handwriting Recognition"

# Preset Datasets (Default cases to load)
PRESET_DATASETS = [
    "Cargo - Agnes Fisheries - CE's notes for Yr 2021-22.pdf",
    "Hull - Marco Polo_Memo.pdf",
    "Cargo - Mitsui Co summary with CE's QA 29.8.22.docx"
]

# API Configuration
DEFAULT_API_KEY = "sk-99bba2ce117444e197270f17d303e74f"
API_BASE = "https://api.deepseek.com/v1"
API_MODEL = "deepseek-chat"

# RAG Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 3

# OCR Configuration
TESSERACT_LANG = "eng+chi_sim"
EASYOCR_LANGS = ['en', 'ch_sim']

# Directories
DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
OCR_RESULTS_DIR = DATA_DIR / "ocr_results"

# ===========================
# Directory Setup
# ===========================

def ensure_directories():
    """Create necessary directories"""
    for dir_path in [DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, VECTOR_DB_DIR, OCR_RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

# ===========================
# Model Loading
# ===========================

@st.cache_resource
def load_embedding_model():
    """Load embedding model"""
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_ocr_reader():
    """Load EasyOCR reader"""
    return easyocr.Reader(EASYOCR_LANGS, gpu=False)

# ===========================
# Step 1: Text Extraction & OCR
# ===========================

def convert_pdf_to_images(file_path: Path) -> List[Image.Image]:
    """Convert PDF pages to images for OCR processing"""
    try:
        images = convert_from_path(str(file_path), dpi=200)
        return images
    except Exception as e:
        st.error(f"PDF to image conversion error: {e}")
        return []

def detect_handwritten_regions(image_np: np.ndarray, min_width=50, min_height=20) -> List[Dict]:
    """
    Detect potential handwritten regions in image
    Uses contour detection to find text regions
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        if w > min_width and h > min_height:
            region = {
                'region_id': idx,
                'bbox': (x, y, w, h),
                'area': w * h,
                'aspect_ratio': w / h if h > 0 else 0
            }
            regions.append(region)
    
    return regions

def classify_text_type(image_crop: np.ndarray) -> str:
    """
    Classify if text region is handwritten or printed
    Uses edge density heuristic
    """
    gray = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Simple heuristic: handwritten has more irregular edges
    if edge_density > 0.15:
        return "handwritten"
    else:
        return "printed"

def perform_ocr(image: Image.Image, reader: easyocr.Reader, text_type: str = "printed") -> str:
    """
    Perform OCR on image
    Uses EasyOCR for handwritten, Tesseract for printed
    """
    try:
        image_np = np.array(image)
        
        if text_type == "handwritten":
            # EasyOCR better for handwritten
            results = reader.readtext(image_np)
            text = " ".join([detection[1] for detection in results])
        else:
            # Tesseract for printed text
            text = pytesseract.image_to_string(image, lang=TESSERACT_LANG)
        
        return text.strip()
    except Exception as e:
        st.error(f"OCR error: {e}")
        return ""

def process_pdf_with_ocr(file_path: Path, doc_id: str, reader: easyocr.Reader) -> Dict:
    """
    Complete PDF processing with handwriting detection
    
    Pipeline:
    1. Convert PDF to images
    2. Detect handwritten regions
    3. Classify each region
    4. Perform OCR (appropriate method)
    5. Save results
    """
    results = {
        'doc_id': doc_id,
        'filename': file_path.name,
        'pages': []
    }
    
    # Convert to images
    images = convert_pdf_to_images(file_path)
    
    if not images:
        return results
    
    progress_bar = st.progress(0)
    
    for page_idx, page_image in enumerate(images):
        page_result = {
            'page_num': page_idx + 1,
            'full_printed_text': "",
            'handwritten_regions': []
        }
        
        image_np = np.array(page_image)
        
        # Full page OCR for printed text
        full_page_text = perform_ocr(page_image, reader, "printed")
        page_result['full_printed_text'] = full_page_text
        
        # Detect handwritten regions
        regions = detect_handwritten_regions(image_np)
        
        for region in regions:
            x, y, w, h = region['bbox']
            crop = image_np[y:y+h, x:x+w]
            
            # Classify region
            text_type = classify_text_type(crop)
            
            if text_type == "handwritten":
                # OCR handwritten region
                crop_pil = Image.fromarray(crop)
                ocr_text = perform_ocr(crop_pil, reader, "handwritten")
                
                if ocr_text:
                    # Save handwritten region image
                    region_img_path = OCR_RESULTS_DIR / f"{doc_id}_p{page_idx+1}_r{region['region_id']}.png"
                    crop_pil.save(region_img_path)
                    
                    page_result['handwritten_regions'].append({
                        'region_id': region['region_id'],
                        'bbox': region['bbox'],
                        'text': ocr_text,
                        'image_path': str(region_img_path)
                    })
        
        results['pages'].append(page_result)
        progress_bar.progress((page_idx + 1) / len(images))
    
    # Save OCR results
    results_path = OCR_RESULTS_DIR / f"{doc_id}_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from DOCX"""
    try:
        doc = DocxDocument(file_path)
        text_parts = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        
        for table in doc.tables:
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                text_parts.append(" | ".join(row_data))
        
        return "\n".join(text_parts)
    except Exception as e:
        st.error(f"DOCX extraction error: {e}")
        return ""

# ===========================
# Step 2: Text Chunking
# ===========================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text
        chunk_size: Characters per chunk
        overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size * 0.5:
                end = start + break_point + 1
                chunk_text = text[start:end]
        
        chunks.append({
            'chunk_id': chunk_id,
            'text': chunk_text.strip(),
            'start_char': start,
            'end_char': end
        })
        
        start = end - overlap
        chunk_id += 1
    
    return chunks

# ===========================
# Step 3: Embedding Generation
# ===========================

def generate_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for text chunks"""
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings

# ===========================
# Step 4: Vector Database
# ===========================

class VectorDatabase:
    """FAISS-based vector database for semantic search"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []
        self.doc_metadata = {}
    
    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray, 
                     doc_id: str, doc_name: str, source_type: str = "printed"):
        """Add document chunks to vector database"""
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['doc_id'] = doc_id
            chunk['doc_name'] = doc_name
            chunk['source_type'] = source_type
            self.chunks.append(chunk)
        
        if doc_id not in self.doc_metadata:
            self.doc_metadata[doc_id] = {
                'doc_name': doc_name,
                'num_chunks': 0,
                'printed_chunks': 0,
                'handwritten_chunks': 0
            }
        
        self.doc_metadata[doc_id]['num_chunks'] += len(chunks)
        if source_type == "printed":
            self.doc_metadata[doc_id]['printed_chunks'] += len(chunks)
        else:
            self.doc_metadata[doc_id]['handwritten_chunks'] += len(chunks)
    
    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[Dict, float]]:
        """Search for similar chunks"""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        similarities, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(similarity)))
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        total_printed = sum(meta['printed_chunks'] for meta in self.doc_metadata.values())
        total_handwritten = sum(meta['handwritten_chunks'] for meta in self.doc_metadata.values())
        
        return {
            'total_chunks': self.index.ntotal,
            'total_documents': len(self.doc_metadata),
            'printed_chunks': total_printed,
            'handwritten_chunks': total_handwritten,
            'dimension': self.dimension
        }
    
    def save(self, path: Path):
        """Save vector database"""
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        with open(path / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        with open(path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.doc_metadata, f, indent=2)
    
    def load(self, path: Path) -> bool:
        """Load vector database"""
        try:
            self.index = faiss.read_index(str(path / "faiss.index"))
            
            with open(path / "chunks.json", 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            with open(path / "metadata.json", 'r', encoding='utf-8') as f:
                self.doc_metadata = json.load(f)
            
            return True
        except:
            return False

# ===========================
# Step 5-6: BM25 Reranking
# ===========================

class BM25Reranker:
    """BM25 reranker for retrieved chunks"""
    
    def __init__(self):
        self.bm25 = None
    
    def fit(self, texts: List[str]):
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)
    
    def rerank(self, query: str, candidates: List[Tuple[Dict, float]], 
              top_k: int = TOP_K_RERANK) -> List[Tuple[Dict, float]]:
        """Rerank candidates using BM25"""
        if not candidates:
            return []
        
        texts = [chunk['text'] for chunk, _ in candidates]
        self.fit(texts)
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        combined = []
        for (chunk, vec_score), bm25_score in zip(candidates, bm25_scores):
            combined_score = 0.6 * vec_score + 0.4 * (bm25_score / (max(bm25_scores) + 1e-6))
            combined.append((chunk, combined_score))
        
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

# ===========================
# Step 7: Prompt Template
# ===========================

def create_rag_prompt(query: str, context_chunks: List[Tuple[Dict, float]]) -> str:
    """Create RAG prompt with retrieved context"""
    context_parts = []
    for idx, (chunk, score) in enumerate(context_chunks, 1):
        source_label = "🖨️ PRINTED" if chunk.get('source_type') == 'printed' else "✍️ HANDWRITTEN"
        context_parts.append(f"""
[Source {idx}] {source_label}
Document: {chunk['doc_name']}
Chunk ID: {chunk['chunk_id']}
Relevance: {score:.3f}
---
{chunk['text']}
---
""")
    
    context_text = "\n".join(context_parts)
    
    prompt = f"""You are a professional underwriting assistant. Answer based on the retrieved context below.

RETRIEVED CONTEXT:
{context_text}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Cite source number and type (PRINTED/HANDWRITTEN)
3. If insufficient information, state clearly
4. Be specific with amounts, dates, terms
5. Note handwritten sources may have OCR errors

ANSWER:"""
    
    return prompt

# ===========================
# Step 8: LLM API Call
# ===========================

def call_llm_api(prompt: str) -> str:
    """Call LLM API for answer generation"""
    try:
        api_key = st.session_state.get('api_key', DEFAULT_API_KEY)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": API_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
        
    except Exception as e:
        return f"Error: {e}"

# ===========================
# Complete RAG Pipeline
# ===========================

def rag_query(query: str, vector_db: VectorDatabase, embedding_model: SentenceTransformer,
              reranker: BM25Reranker) -> Dict:
    """
    Complete RAG pipeline execution
    
    Steps:
    5. Query embedding
    6. Vector search (retrieve top-k)
    7. BM25 reranking
    8. Prompt assembly and LLM call
    """
    # Step 5: Query embedding
    query_embedding = embedding_model.encode([query])[0]
    
    # Step 6: Retrieve top-k chunks
    candidates = vector_db.search(query_embedding, top_k=TOP_K_RETRIEVAL)
    
    if not candidates:
        return {
            'answer': "No relevant documents found.",
            'sources': [],
            'num_candidates': 0
        }
    
    # Step 7: Rerank
    reranked = reranker.rerank(query, candidates, top_k=TOP_K_RERANK)
    
    # Step 8: Create prompt and call LLM
    prompt = create_rag_prompt(query, reranked)
    answer = call_llm_api(prompt)
    
    sources = [
        {
            'doc_name': chunk['doc_name'],
            'chunk_id': chunk['chunk_id'],
            'source_type': chunk.get('source_type', 'unknown'),
            'score': score,
            'text_preview': chunk['text'][:200] + "..."
        }
        for chunk, score in reranked
    ]
    
    return {
        'answer': answer,
        'sources': sources,
        'num_candidates': len(candidates)
    }

# ===========================
# Preset Dataset Loading
# ===========================

def load_preset_datasets(vector_db: VectorDatabase, embedding_model: SentenceTransformer, 
                        ocr_reader: easyocr.Reader):
    """
    Load and process preset datasets
    
    Complete pipeline for each file:
    1. Text extraction with OCR
    2. Chunking
    3. Embedding
    4. Indexing
    """
    st.info("🔄 Loading preset datasets...")
    
    # Check if already loaded
    if VECTOR_DB_DIR.exists() and (VECTOR_DB_DIR / "faiss.index").exists():
        if vector_db.load(VECTOR_DB_DIR):
            st.success("✅ Loaded existing vector database")
            return
    
    # Process each preset file
    for filename in PRESET_DATASETS:
        file_path = UPLOADS_DIR / filename
        
        if not file_path.exists():
            st.warning(f"⚠️ Preset file not found: {filename}. Please place in {UPLOADS_DIR}")
            continue
        
        st.info(f"📄 Processing: {filename}")
        
        doc_id = hashlib.md5(filename.encode()).hexdigest()[:8]
        
        # Step 1: Extract text
        if filename.endswith('.pdf'):
            # OCR processing for PDF
            ocr_results = process_pdf_with_ocr(file_path, doc_id, ocr_reader)
            
            # Extract printed text
            printed_text = "\n\n".join([
                page['full_printed_text'] for page in ocr_results['pages']
            ])
            
            # Extract handwritten text
            handwritten_text = "\n\n".join([
                region['text'] 
                for page in ocr_results['pages'] 
                for region in page['handwritten_regions']
            ])
            
            # Step 2-4: Chunk, embed, and index printed text
            if printed_text.strip():
                printed_chunks = chunk_text(printed_text)
                if printed_chunks:
                    printed_texts = [c['text'] for c in printed_chunks]
                    printed_embeddings = generate_embeddings(printed_texts, embedding_model)
                    vector_db.add_documents(printed_chunks, printed_embeddings, doc_id, filename, "printed")
            
            # Step 2-4: Chunk, embed, and index handwritten text
            if handwritten_text.strip():
                hw_chunks = chunk_text(handwritten_text)
                if hw_chunks:
                    hw_texts = [c['text'] for c in hw_chunks]
                    hw_embeddings = generate_embeddings(hw_texts, embedding_model)
                    vector_db.add_documents(hw_chunks, hw_embeddings, doc_id, filename, "handwritten")
        
        elif filename.endswith('.docx'):
            # Direct text extraction for DOCX
            text = extract_text_from_docx(file_path)
            if text.strip():
                chunks = chunk_text(text)
                if chunks:
                    texts = [c['text'] for c in chunks]
                    embeddings = generate_embeddings(texts, embedding_model)
                    vector_db.add_documents(chunks, embeddings, doc_id, filename, "printed")
    
    # Save vector database
    vector_db.save(VECTOR_DB_DIR)
    st.success("✅ All preset datasets processed and indexed")

# ===========================
# Streamlit UI
# ===========================

def render_header():
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, #1e3a8a 0%, #7c3aed 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>📋 {APP_TITLE}</h1>
        <p style='color: #e0e7ff; margin-top: 0.5rem;'>Version {VERSION}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Underwriting RAG",
        page_icon="📋",
        layout="wide"
    )
    
    ensure_directories()
    render_header()
    
    # Initialize session state
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = VectorDatabase()
    
    if 'reranker' not in st.session_state:
        st.session_state.reranker = BM25Reranker()
    
    if 'datasets_loaded' not in st.session_state:
        st.session_state.datasets_loaded = False
    
    # Load models
    embedding_model = load_embedding_model()
    ocr_reader = load_ocr_reader()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        with st.expander("🔑 API Key"):
            api_key = st.text_input("DeepSeek API Key:", type="password", value=DEFAULT_API_KEY)
            if st.button("Save Key"):
                st.session_state.api_key = api_key
                st.success("✓ Saved")
        
        st.markdown("---")
        
        # Load preset datasets
        if not st.session_state.datasets_loaded:
            if st.button("📥 Load Preset Datasets", type="primary"):
                with st.spinner("Processing datasets..."):
                    load_preset_datasets(st.session_state.vector_db, embedding_model, ocr_reader)
                st.session_state.datasets_loaded = True
                st.rerun()
        else:
            st.success("✅ Datasets Loaded")
            if st.button("🔄 Reload"):
                st.session_state.datasets_loaded = False
                st.session_state.vector_db = VectorDatabase()
                st.rerun()
        
        st.markdown("---")
        
        # Stats
        stats = st.session_state.vector_db.get_stats()
        st.metric("Documents", stats['total_documents'])
        st.metric("Total Chunks", stats['total_chunks'])
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🖨️ Printed", stats['printed_chunks'])
        with col2:
            st.metric("✍️ Handwritten", stats['handwritten_chunks'])
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 AI Chat (RAG)", 
        "📄 Document Viewer", 
        "🗄️ Vector Database", 
        "✍️ OCR Results"
    ])
    
    # TAB 1: AI Chat with RAG
    with tab1:
        st.header("💬 AI Chat with RAG Pipeline")
        
        if stats['total_documents'] == 0:
            st.warning("⚠️ Please load preset datasets first")
        else:
            st.markdown("""
            **RAG Pipeline Steps:**
            - **Step 5**: Query vectorization
            - **Step 6**: Semantic search (retrieve top-k)
            - **Step 7**: BM25 reranking + context assembly
            - **Step 8**: LLM generation
            """)
            
            query = st.text_area(
                "Enter your question:",
                placeholder="e.g., What are the retention terms in Agnes Fisheries case?",
                height=100
            )
            
            if st.button("🔍 Search & Answer", type="primary"):
                if query:
                    with st.spinner("Running RAG pipeline..."):
                        result = rag_query(
                            query,
                            st.session_state.vector_db,
                            embedding_model,
                            st.session_state.reranker
                        )
                        
                        st.markdown("### 💡 Answer")
                        st.markdown(result['answer'])
                        
                        st.markdown("---")
                        st.markdown("### 📚 Retrieved Sources")
                        
                        for idx, source in enumerate(result['sources'], 1):
                            source_icon = "🖨️" if source['source_type'] == 'printed' else "✍️"
                            with st.expander(
                                f"{source_icon} Source {idx}: {source['doc_name']} (Score: {source['score']:.3f})"
                            ):
                                st.markdown(f"**Type:** {source['source_type'].upper()}")
                                st.markdown(f"**Chunk ID:** {source['chunk_id']}")
                                st.markdown(f"**Preview:**\n{source['text_preview']}")
    
    # TAB 2: Document Viewer
    with tab2:
        st.header("📄 Document Viewer (Original Files)")
        
        st.markdown("View original documents and their extracted content")
        
        ocr_results_files = list(OCR_RESULTS_DIR.glob("*_results.json"))
        
        if not ocr_results_files:
            st.info("No processed documents yet")
        else:
            selected_file = st.selectbox(
                "Select document:",
                [f.stem.replace("_results", "") for f in ocr_results_files]
            )
            
            if selected_file:
                results_path = OCR_RESULTS_DIR / f"{selected_file}_results.json"
                
                with open(results_path, 'r', encoding='utf-8') as f:
                    ocr_results = json.load(f)
                
                st.subheader(f"📄 {ocr_results['filename']}")
                
                for page in ocr_results['pages']:
                    with st.expander(f"Page {page['page_num']}", expanded=False):
                        st.markdown("#### 🖨️ Printed Text")
                        st.text_area(
                            "Full page printed text:",
                            value=page['full_printed_text'],
                            height=200,
                            key=f"printed_{page['page_num']}"
                        )
                        
                        if page['handwritten_regions']:
                            st.markdown("#### ✍️ Handwritten Regions")
                            st.info(f"Found {len(page['handwritten_regions'])} handwritten region(s)")
                            
                            for region in page['handwritten_regions']:
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    if Path(region['image_path']).exists():
                                        img = Image.open(region['image_path'])
                                        st.image(img, caption=f"Region {region['region_id']}", use_column_width=True)
                                
                                with col2:
                                    st.markdown(f"**Region {region['region_id']}**")
                                    st.markdown(f"**Bounding Box:** {region['bbox']}")
                                    st.markdown("**OCR Transcription:**")
                                    st.code(region['text'], language=None)
    
    # TAB 3: Vector Database
    with tab3:
        st.header("🗄️ Vector Database Explorer")
        
        stats = st.session_state.vector_db.get_stats()
        
        # Overall stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Chunks", stats['total_chunks'])
        with col2:
            st.metric("Documents", stats['total_documents'])
        with col3:
            st.metric("🖨️ Printed", stats['printed_chunks'])
        with col4:
            st.metric("✍️ Handwritten", stats['handwritten_chunks'])
        
        st.markdown("---")
        
        # Document breakdown
        st.subheader("Document Breakdown")
        
        doc_data = []
        for doc_id, meta in st.session_state.vector_db.doc_metadata.items():
            doc_data.append({
                'Document': meta['doc_name'],
                'Total Chunks': meta['num_chunks'],
                '🖨️ Printed': meta['printed_chunks'],
                '✍️ Handwritten': meta['handwritten_chunks']
            })
        
        if doc_data:
            df = pd.DataFrame(doc_data)
            st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        
        # Sample chunks
        st.subheader("Sample Chunks (First 10)")
        
        if st.session_state.vector_db.chunks:
            sample_size = min(10, len(st.session_state.vector_db.chunks))
            samples = st.session_state.vector_db.chunks[:sample_size]
            
            for idx, chunk in enumerate(samples, 1):
                source_icon = "🖨️" if chunk.get('source_type') == 'printed' else "✍️"
                with st.expander(f"{source_icon} Chunk {idx}: {chunk['doc_name']} (ID: {chunk['chunk_id']})"):
                    st.json({
                        'chunk_id': chunk['chunk_id'],
                        'doc_name': chunk['doc_name'],
                        'source_type': chunk.get('source_type', 'unknown'),
                        'text': chunk['text'][:500] + "..."
                    })
    
    # TAB 4: OCR Results
    with tab4:
        st.header("✍️ OCR Processing Results")
        
        st.markdown("""
        View detailed OCR processing results:
        - Handwritten region detection
        - Text classification (printed vs handwritten)
        - OCR transcriptions
        """)
        
        ocr_results_files = list(OCR_RESULTS_DIR.glob("*_results.json"))
        
        if not ocr_results_files:
            st.info("No OCR results yet. Load preset datasets to process documents.")
        else:
            for results_file in ocr_results_files:
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                with st.expander(f"📄 {results['filename']}", expanded=False):
                    st.markdown(f"**Document ID:** `{results['doc_id']}`")
                    st.markdown(f"**Total Pages:** {len(results['pages'])}")
                    
                    for page in results['pages']:
                        st.markdown(f"#### Page {page['page_num']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Printed Text Length", len(page['full_printed_text']))
                        with col2:
                            st.metric("Handwritten Regions", len(page['handwritten_regions']))
                        
                        if page['handwritten_regions']:
                            st.markdown("**Handwritten Regions:**")
                            
                            for region in page['handwritten_regions']:
                                col_a, col_b = st.columns([1, 3])
                                
                                with col_a:
                                    if Path(region['image_path']).exists():
                                        img = Image.open(region['image_path'])
                                        st.image(img, width=200)
                                
                                with col_b:
                                    st.markdown(f"**Region {region['region_id']}**")
                                    st.markdown(f"Bbox: {region['bbox']}")
                                    st.code(region['text'])
                        
                        st.markdown("---")

if __name__ == "__main__":
    main()
