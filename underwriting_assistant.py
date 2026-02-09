"""
Professional Underwriting RAG System with Handwriting Recognition
=================================================================

Complete Pipeline:
1. File Upload & Preset Loading
2. Hybrid OCR processing (Tesseract + EasyOCR)
3. Text chunking
4. Embedding & Vector Indexing
5. Semantic Search + Reranking
6. LLM Q&A
"""

import streamlit as st
import os
import json
from pathlib import Path
import hashlib
from typing import List, Dict, Any, Tuple

# Document & Image Processing
from pdf2image import convert_from_path
from docx import Document as DocxDocument
from PIL import Image
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

VERSION = "4.2.0-Upload-Enabled"
APP_TITLE = "Underwriting Repository - RAG with Handwriting Recognition"

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
VECTOR_DB_DIR = DATA_DIR / "vector_db"
OCR_RESULTS_DIR = DATA_DIR / "ocr_results"

# Preset Datasets (files expected to be in data/uploads folder)
PRESET_DATASETS = [
    "Cargo - Agnes Fisheries - CE's notes for Yr 2021-22.pdf",
    "Hull - Marco Polo_Memo.pdf",
    "Cargo - Mitsui Co summary with CE's QA 29.8.22.docx"
]

# ===========================
# Directory Setup
# ===========================

def ensure_directories():
    """Create necessary directories"""
    for dir_path in [DATA_DIR, UPLOADS_DIR, VECTOR_DB_DIR, OCR_RESULTS_DIR]:
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
# OCR & Processing Logic
# ===========================

def convert_pdf_to_images(file_path: Path) -> List[Image.Image]:
    """Convert PDF pages to images for OCR processing"""
    try:
        images = convert_from_path(str(file_path), dpi=200)
        return images
    except Exception as e:
        st.error(f"PDF conversion error: {e}")
        return []

def perform_hybrid_ocr(image: Image.Image, reader: easyocr.Reader, page_idx: int, doc_id: str) -> Dict:
    """Hybrid OCR: Tesseract for full page, EasyOCR for specific regions"""
    page_result = {
        'page_num': page_idx + 1,
        'full_printed_text': "",
        'handwritten_regions': []
    }

    try:
        # 1. Full Page Tesseract
        full_text = pytesseract.image_to_string(image, lang=TESSERACT_LANG)
        page_result['full_printed_text'] = full_text.strip()

        # 2. EasyOCR for detection
        image_np = np.array(image)
        detections = reader.readtext(image_np)

        for idx, (bbox_points, text, conf) in enumerate(detections):
            x_coords = [p[0] for p in bbox_points]
            y_coords = [p[1] for p in bbox_points]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            w = x_max - x_min
            h = y_max - y_min

            # Filter tiny noise
            if w < 20 or h < 10:
                continue

            try:
                crop = image.crop((x_min, y_min, x_max, y_max))
                region_id = idx
                region_img_path = OCR_RESULTS_DIR / f"{doc_id}_p{page_idx+1}_r{region_id}.png"
                crop.save(region_img_path)

                page_result['handwritten_regions'].append({
                    'region_id': region_id,
                    'bbox': (x_min, y_min, w, h),
                    'text': text,
                    'confidence': float(conf),
                    'image_path': str(region_img_path)
                })
            except Exception:
                continue

    except Exception as e:
        print(f"OCR Warning: {e}")

    return page_result

def process_pdf_with_ocr(file_path: Path, doc_id: str, reader: easyocr.Reader) -> Dict:
    """Complete PDF processing pipeline"""
    results = {
        'doc_id': doc_id,
        'filename': file_path.name,
        'pages': []
    }
    
    images = convert_pdf_to_images(file_path)
    
    if not images:
        return results
    
    # Show progress only if running in main thread context
    progress_bar = st.progress(0)
    
    for page_idx, page_image in enumerate(images):
        page_result = perform_hybrid_ocr(page_image, reader, page_idx, doc_id)
        results['pages'].append(page_result)
        progress_bar.progress((page_idx + 1) / len(images))
    
    # Save OCR results JSON
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
        st.error(f"DOCX error: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Split text into chunks"""
    if not text: return []
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
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

def generate_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

# ===========================
# Vector Database
# ===========================

class VectorDatabase:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []
        self.doc_metadata = {}
    
    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray, 
                      doc_id: str, doc_name: str, source_type: str = "printed"):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['doc_id'] = doc_id
            chunk['doc_name'] = doc_name
            chunk['source_type'] = source_type
            self.chunks.append(chunk)
        
        if doc_id not in self.doc_metadata:
            self.doc_metadata[doc_id] = {'doc_name': doc_name, 'num_chunks': 0}
        
        self.doc_metadata[doc_id]['num_chunks'] += len(chunks)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Dict, float]]:
        if self.index.ntotal == 0: return []
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        similarities, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(similarity)))
        return results
    
    def get_stats(self) -> Dict:
        return {
            'total_chunks': self.index.ntotal,
            'total_documents': len(self.doc_metadata),
            'dimension': self.dimension
        }
    
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        with open(path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.doc_metadata, f, indent=2)
    
    def load(self, path: Path) -> bool:
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
# Reranking & LLM
# ===========================

class BM25Reranker:
    def __init__(self):
        self.bm25 = None
    
    def fit(self, texts: List[str]):
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)
    
    def rerank(self, query: str, candidates: List[Tuple[Dict, float]], top_k: int = 3) -> List[Tuple[Dict, float]]:
        if not candidates: return []
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

def call_llm_api(prompt: str) -> str:
    try:
        api_key = st.session_state.get('api_key', DEFAULT_API_KEY)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": API_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3, "max_tokens": 2000
        }
        response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

# ===========================
# Pipeline Execution
# ===========================

def process_document(file_path: Path, vector_db: VectorDatabase, 
                     embedding_model: SentenceTransformer, ocr_reader: easyocr.Reader):
    """
    Unified processing logic for both preset and uploaded files
    """
    filename = file_path.name
    doc_id = hashlib.md5(filename.encode()).hexdigest()[:8]
    
    st.info(f"📄 Processing: {filename}")
    
    # 1. Extraction & OCR
    if filename.endswith('.pdf'):
        ocr_results = process_pdf_with_ocr(file_path, doc_id, ocr_reader)
        
        printed_text = "\n\n".join([p['full_printed_text'] for p in ocr_results['pages']])
        handwritten_text = "\n\n".join([r['text'] for p in ocr_results['pages'] for r in p['handwritten_regions']])
        
        # 2. Chunking & Indexing
        if printed_text.strip():
            chunks = chunk_text(printed_text)
            if chunks:
                vecs = generate_embeddings([c['text'] for c in chunks], embedding_model)
                vector_db.add_documents(chunks, vecs, doc_id, filename, "printed")
        
        if handwritten_text.strip():
            chunks = chunk_text(handwritten_text)
            if chunks:
                vecs = generate_embeddings([c['text'] for c in chunks], embedding_model)
                vector_db.add_documents(chunks, vecs, doc_id, filename, "extracted")
                
    elif filename.endswith('.docx'):
        text = extract_text_from_docx(file_path)
        if text.strip():
            chunks = chunk_text(text)
            if chunks:
                vecs = generate_embeddings([c['text'] for c in chunks], embedding_model)
                vector_db.add_documents(chunks, vecs, doc_id, filename, "printed")

def handle_file_upload(uploaded_files, vector_db, embedding_model, ocr_reader):
    """Handle new file uploads from the UI"""
    for uploaded_file in uploaded_files:
        # Save file to UPLOADS_DIR
        file_path = UPLOADS_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process immediately
        process_document(file_path, vector_db, embedding_model, ocr_reader)
    
    # Save DB after processing all files
    vector_db.save(VECTOR_DB_DIR)
    st.success("✅ Uploads processed and indexed!")

def load_preset_datasets(vector_db, embedding_model, ocr_reader):
    """Load only if DB is empty"""
    if VECTOR_DB_DIR.exists() and (VECTOR_DB_DIR / "faiss.index").exists():
        if vector_db.load(VECTOR_DB_DIR):
            st.success("✅ Loaded existing database")
            return

    st.info("🔄 Initializing with preset datasets...")
    for filename in PRESET_DATASETS:
        file_path = UPLOADS_DIR / filename
        if file_path.exists():
            process_document(file_path, vector_db, embedding_model, ocr_reader)
    
    vector_db.save(VECTOR_DB_DIR)
    st.success("✅ Initialization complete")

# ===========================
# Streamlit UI
# ===========================

def main():
    st.set_page_config(page_title="Underwriting RAG", page_icon="📋", layout="wide")
    ensure_directories()
    
    # State Init
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = VectorDatabase()
    if 'reranker' not in st.session_state:
        st.session_state.reranker = BM25Reranker()
    
    # Load Models
    embedding_model = load_embedding_model()
    ocr_reader = load_ocr_reader()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # 1. API Key
        with st.expander("🔑 API Key"):
            key = st.text_input("DeepSeek Key:", type="password", value=DEFAULT_API_KEY)
            if st.button("Save Key"):
                st.session_state.api_key = key
                st.success("Saved")

        st.markdown("---")
        
        # 2. File Upload Section (NEW FEATURE)
        st.header("📂 Data Management")
        uploaded_files = st.file_uploader(
            "Upload New Documents", 
            type=['pdf', 'docx'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🚀 Process Uploads"):
                with st.spinner("Processing uploaded files..."):
                    handle_file_upload(
                        uploaded_files, 
                        st.session_state.vector_db, 
                        embedding_model, 
                        ocr_reader
                    )
        
        st.markdown("---")
        
        # 3. Preset Loading
        if st.button("📥 Load/Reset Presets"):
            load_preset_datasets(st.session_state.vector_db, embedding_model, ocr_reader)
            
        # Stats
        stats = st.session_state.vector_db.get_stats()
        st.metric("Total Documents", stats['total_documents'])
        st.metric("Total Chunks", stats['total_chunks'])

    # Main Area
    st.markdown(f"### 📋 {APP_TITLE}")
    
    tab1, tab2, tab3 = st.tabs(["💬 RAG Chat", "📄 Document Viewer", "🗄️ Database Info"])
    
    with tab1:
        query = st.text_area("Question:", height=100, placeholder="e.g. What are the retention terms?")
        if st.button("🔍 Analyze"):
            # RAG Pipeline
            q_emb = embedding_model.encode([query])[0]
            candidates = st.session_state.vector_db.search(q_emb)
            reranked = st.session_state.reranker.rerank(query, candidates)
            
            # Context Building
            context_txt = ""
            for i, (chunk, score) in enumerate(reranked, 1):
                context_txt += f"[Source {i}] {chunk['doc_name']} ({chunk['source_type']})\n{chunk['text']}\n\n"
            
            # LLM Call
            prompt = f"Answer strictly based on context:\n\n{context_txt}\n\nQuestion: {query}"
            answer = call_llm_api(prompt)
            
            st.markdown("### Answer")
            st.write(answer)
            
            with st.expander("References"):
                for i, (chunk, score) in enumerate(reranked, 1):
                    st.markdown(f"**{i}. {chunk['doc_name']}** (Score: {score:.3f})")
                    st.caption(chunk['text'][:200] + "...")

    with tab2:
        st.header("Document Viewer")
        ocr_files = list(OCR_RESULTS_DIR.glob("*_results.json"))
        if ocr_files:
            sel_file = st.selectbox("Select File", [f.stem.replace("_results", "") for f in ocr_files])
            if sel_file:
                with open(OCR_RESULTS_DIR / f"{sel_file}_results.json", 'r') as f:
                    res = json.load(f)
                st.json(res)
        else:
            st.info("No processed documents found.")

    with tab3:
        st.dataframe(pd.DataFrame(st.session_state.vector_db.doc_metadata).T)

if __name__ == "__main__":
    main()

