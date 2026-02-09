import streamlit as st
import os
import json
import shutil
from pathlib import Path
import hashlib
from typing import List, Dict, Tuple
import sys
import subprocess

# Document & Image Processing
try:
    from pdf2image import convert_from_path
    from docx import Document as DocxDocument
    from PIL import Image
    import numpy as np
except ImportError as e:
    st.error(f"Missing Dependency: {e}")
    st.stop()

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

APP_TITLE = "Underwriting RAG - Debug Mode"
DEFAULT_API_KEY = "sk-99bba2ce117444e197270f17d303e74f"
API_BASE = "https://api.deepseek.com/v1"
API_MODEL = "deepseek-chat"

# Directories
DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
OCR_RESULTS_DIR = DATA_DIR / "ocr_results"

# OCR Config
TESSERACT_LANG = "eng+chi_sim"
EASYOCR_LANGS = ['en', 'ch_sim']

# ===========================
# System Check (Run Once)
# ===========================
def check_system_dependencies():
    """Verify external tools are installed"""
    errors = []
    
    # Check Poppler (for PDF)
    try:
        subprocess.run(["pdftoppm", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        errors.append("❌ Poppler is NOT installed. PDFs cannot be processed.")
    
    # Check Tesseract
    try:
        subprocess.run(["tesseract", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        errors.append("❌ Tesseract is NOT installed. OCR will fail.")
        
    return errors

# ===========================
# Core Logic
# ===========================

def ensure_directories():
    for dir_path in [DATA_DIR, UPLOADS_DIR, VECTOR_DB_DIR, OCR_RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def load_models():
    """Load AI Models"""
    with st.spinner("Loading AI Models (Embedding + OCR)..."):
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        ocr_reader = easyocr.Reader(EASYOCR_LANGS, gpu=False)
    return embed_model, ocr_reader

def process_pdf(file_path: Path, doc_id: str, reader: easyocr.Reader) -> str:
    """Process PDF with Debug Prints"""
    st.write(f"🔍 [Debug] Converting PDF: {file_path.name}")
    
    try:
        # Check file size
        size = file_path.stat().st_size
        if size == 0:
            st.error("❌ File is empty (0 bytes). Upload failed.")
            return ""
            
        images = convert_from_path(str(file_path), dpi=200)
        st.write(f"✅ [Debug] Converted to {len(images)} images")
    except Exception as e:
        st.error(f"❌ PDF Conversion Failed: {e}")
        st.info("💡 Hint: If on Streamlit Cloud, ensure packages.txt contains 'poppler-utils'")
        return ""

    full_text = []
    progress_bar = st.progress(0)
    
    for i, img in enumerate(images):
        st.caption(f"Processing Page {i+1}/{len(images)}...")
        
        # 1. Tesseract (Printed)
        try:
            text = pytesseract.image_to_string(img, lang=TESSERACT_LANG)
            full_text.append(text)
        except Exception as e:
            st.warning(f"Tesseract Error on page {i+1}: {e}")

        # 2. EasyOCR (Handwriting/Details)
        try:
            img_np = np.array(img)
            results = reader.readtext(img_np, detail=0)
            full_text.append(" ".join(results))
        except Exception as e:
            st.warning(f"EasyOCR Error on page {i+1}: {e}")
            
        progress_bar.progress((i + 1) / len(images))
        
    return "\n".join(full_text)

def process_docx(file_path: Path) -> str:
    st.write(f"🔍 [Debug] Reading DOCX: {file_path.name}")
    try:
        doc = DocxDocument(file_path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        st.write(f"✅ [Debug] Extracted {len(text)} characters")
        return text
    except Exception as e:
        st.error(f"❌ DOCX Error: {e}")
        return ""

def chunk_text(text: str) -> List[str]:
    chunks = []
    chunk_size = 500
    overlap = 50
    
    if not text: return []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    
    return chunks

# ===========================
# Vector DB Class
# ===========================
class VectorDB:
    def __init__(self):
        self.index = faiss.IndexFlatIP(384)
        self.docs = [] # Metadata store
        self.chunks = [] # Text store
        
    def add(self, text_chunks, embeddings, filename):
        if len(text_chunks) == 0: return
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        for chunk in text_chunks:
            self.chunks.append({
                "text": chunk,
                "source": filename
            })
            
    def search(self, query_vec, k=5):
        if self.index.ntotal == 0: return []
        D, I = self.index.search(query_vec, k)
        results = []
        for idx in I[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

    def save(self):
        faiss.write_index(self.index, str(VECTOR_DB_DIR / "index.faiss"))
        with open(VECTOR_DB_DIR / "chunks.json", "w") as f:
            json.dump(self.chunks, f)
            
    def load(self):
        if (VECTOR_DB_DIR / "index.faiss").exists():
            self.index = faiss.read_index(str(VECTOR_DB_DIR / "index.faiss"))
            with open(VECTOR_DB_DIR / "chunks.json", "r") as f:
                self.chunks = json.load(f)
            return True
        return False

# ===========================
# Main UI
# ===========================
def main():
    st.set_page_config(layout="wide", page_title="Debug RAG")
    ensure_directories()
    
    # 1. Dependency Check
    sys_errors = check_system_dependencies()
    if sys_errors:
        for err in sys_errors:
            st.error(err)
        st.stop() # Stop app if system tools missing

    # 2. State & Models
    if 'db' not in st.session_state:
        st.session_state.db = VectorDB()
        if st.session_state.db.load():
            st.toast("Database loaded from disk")
            
    embed_model, ocr_reader = load_models()

    # 3. Sidebar - Upload
    with st.sidebar:
        st.header("📂 Upload Center")
        files = st.file_uploader("Upload Files", accept_multiple_files=True, type=['pdf', 'docx'])
        
        if files and st.button("🚀 Start Processing"):
            for f in files:
                save_path = UPLOADS_DIR / f.name
                
                # Debug: Write file
                with open(save_path, "wb") as dest:
                    dest.write(f.getbuffer())
                st.write(f"💾 [Debug] Saved {f.name} to disk")
                
                # Processing
                extracted_text = ""
                if f.name.endswith(".pdf"):
                    extracted_text = process_pdf(save_path, "temp_id", ocr_reader)
                elif f.name.endswith(".docx"):
                    extracted_text = process_docx(save_path)
                
                # Indexing
                if extracted_text:
                    chunks = chunk_text(extracted_text)
                    st.write(f"🧩 [Debug] Generated {len(chunks)} chunks")
                    
                    if chunks:
                        vecs = embed_model.encode(chunks, convert_to_numpy=True)
                        st.session_state.db.add(chunks, vecs, f.name)
                        st.success(f"✅ Indexed {f.name}")
                    else:
                        st.warning(f"⚠️ No text found in chunks for {f.name}")
                else:
                    st.error(f"❌ No text extracted from {f.name}")

            # Persist
            st.session_state.db.save()
            st.success("💾 Database Saved!")

    # 4. Main Chat
    st.title("💬 Underwriting Assistant (Debug Mode)")
    
    # Check DB status
    doc_count = len(set(c['source'] for c in st.session_state.db.chunks))
    st.info(f"📚 Database contains {st.session_state.db.index.ntotal} chunks from {doc_count} documents.")
    
    query = st.text_input("Ask a question:")
    if query:
        if st.session_state.db.index.ntotal == 0:
            st.error("Database is empty! Please upload a document first.")
        else:
            # Retrieval
            q_vec = embed_model.encode([query], convert_to_numpy=True)
            results = st.session_state.db.search(q_vec)
            
            # Display Context
            context_str = ""
            with st.expander("🔍 Debug: Retrieved Context", expanded=False):
                for i, r in enumerate(results):
                    st.write(f"**Source:** {r['source']}")
                    st.text(r['text'])
                    context_str += f"Source: {r['source']}\nContent: {r['text']}\n\n"
            
            # LLM Call
            prompt = f"Context:\n{context_txt}\n\nQuestion: {query}\nAnswer:"
            
            try:
                headers = {"Authorization": f"Bearer {DEFAULT_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": API_MODEL,
                    "messages": [{"role": "user", "content": f"Answer based on this context:\n\n{context_str}\n\nQuestion: {query}"}]
                }
                resp = requests.post(f"{API_BASE}/chat/completions", json=payload, headers=headers)
                
                if resp.status_code == 200:
                    st.markdown("### 🤖 Answer")
                    st.write(resp.json()['choices'][0]['message']['content'])
                else:
                    st.error(f"API Error: {resp.text}")
            except Exception as e:
                st.error(f"Request Failed: {e}")

if __name__ == "__main__":
    main()
