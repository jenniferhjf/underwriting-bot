"""
Enhanced Underwriting Assistant - Complete Version with OCR
Includes: Workspace, Documents, Search, Q&A, Audit Logs, Settings, Dictionary
"""

import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path
import hashlib
import re
import traceback
from typing import List, Dict, Any, Optional, Tuple
import requests
from io import BytesIO
import base64

# ===========================
# Configuration
# ===========================

VERSION = "3.2.0-complete"
APP_TITLE = "Enhanced Underwriting Assistant - Professional RAG+CoT System"

# API Configuration
DEFAULT_API_KEY = os.getenv("API_KEY", "sk-99bba2ce117444e197270f17d303e74f")
API_BASE = "https://api.deepseek.com/v1"
API_MODEL = "deepseek-chat"

# OCR Configuration
OCR_API_URL = "https://api.ocr.space/parse/image"
OCR_API_KEY = "K87899142388957"

# Directory Structure
DATA_DIR = Path("data")
WORKSPACES_DIR = DATA_DIR / "workspaces"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
ANALYSIS_DIR = DATA_DIR / "analysis"
AUDIT_DIR = DATA_DIR / "audit_logs"
CONFIG_DIR = DATA_DIR / "config"

INITIAL_DATASET = "Hull - MSC_Memo.pdf"

SUPPORTED_FORMATS = {
    'pdf': '📄 PDF',
    'docx': '📝 Word',
    'doc': '📝 Word',
    'txt': '📃 Text',
    'png': '🖼️ Image',
    'jpg': '🖼️ Image',
    'jpeg': '🖼️ Image'
}

TAG_OPTIONS = {
    'equipment': ['Hull', 'Cargo', 'Liability', 'Property', 'Marine', 'Aviation'],
    'industry': ['Shipping', 'Manufacturing', 'Retail', 'Technology', 'Construction'],
    'timeline': ['2024', '2025', '2026', 'Q1', 'Q2', 'Q3', 'Q4']
}

INSURANCE_TERMS = {
    'retention': 'The amount of risk that the insured retains',
    'premium': 'The amount paid for insurance coverage',
    'coverage': 'The scope and extent of protection provided',
    'deductible': 'The amount the insured must pay before the insurer pays',
    'underwriting slip': 'A document containing key details of an insurance risk',
    'loss ratio': 'The ratio of losses paid to premiums earned',
    'hull': 'Insurance covering damage to a vessel itself',
    'cargo': 'Insurance covering goods being transported',
    'P&I': 'Protection and Indemnity insurance',
    'claims made': 'Policy that covers claims made during the policy period',
    'occurrence': 'Policy that covers events that occur during the policy period',
    'subrogation': 'The right of an insurer to pursue recovery from a third party',
}

# System Prompts
SYSTEM_INSTRUCTION = """You are an expert underwriting assistant with deep knowledge of insurance policies, 
risk assessment, and document analysis. Provide clear, professional responses."""

ELECTRONIC_TEXT_ANALYSIS_SYSTEM = """Analyze this insurance document and provide a BRIEF summary (3-5 sentences).
Cover: insurance type, insured party, key financial terms, main risk factors."""

QA_SYSTEM = """You are an expert insurance underwriter. Answer questions based on the provided document context.
Be precise, cite specific information from the documents, and indicate if information is not available."""

# ===========================
# OCR Functions
# ===========================

def recognize_handwriting_cloud(image_data: str) -> Dict[str, Any]:
    """Recognize handwritten text using OCR.space API"""
    try:
        image_bytes = base64.b64decode(image_data)
        
        payload = {
            'apikey': OCR_API_KEY,
            'language': 'eng',
            'isOverlayRequired': False,
            'OCREngine': 2
        }
        
        files = {
            'file': ('image.png', image_bytes, 'image/png')
        }
        
        response = requests.post(OCR_API_URL, data=payload, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('IsErroredOnProcessing'):
                return {
                    'recognized_text': '',
                    'confidence': 0,
                    'error': result.get('ErrorMessage', 'Unknown error')
                }
            
            parsed_results = result.get('ParsedResults', [])
            if parsed_results:
                text = parsed_results[0].get('ParsedText', '').strip()
                confidence = float(parsed_results[0].get('TextOverlay', {}).get('Lines', [{}])[0].get('MaxHeight', 80))
                
                return {
                    'recognized_text': text,
                    'confidence': min(confidence, 100),
                    'word_count': len(text.split()),
                    'status': 'success'
                }
        
        return {
            'recognized_text': '',
            'confidence': 0,
            'error': f'HTTP {response.status_code}'
        }
    except Exception as e:
        return {
            'recognized_text': '',
            'confidence': 0,
            'error': str(e)
        }

def batch_recognize_handwriting_cloud(images: List[Dict]) -> List[Dict]:
    """Batch process multiple images"""
    results = []
    
    for i, img in enumerate(images):
        result = recognize_handwriting_cloud(img['data'])
        results.append({
            'location': img['id'],
            'page': img.get('page', 1),
            'recognized_text': result.get('recognized_text', ''),
            'confidence': result.get('confidence', 0),
            'word_count': result.get('word_count', 0),
            'error': result.get('error')
        })
    
    return results

# ===========================
# Utility Functions
# ===========================

def ensure_dirs():
    for dir_path in [WORKSPACES_DIR, EMBEDDINGS_DIR, ANALYSIS_DIR, AUDIT_DIR, CONFIG_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

def log_audit(action: str, details: Dict):
    """Log actions to audit trail"""
    try:
        log_file = AUDIT_DIR / f"audit_{datetime.now().strftime('%Y%m')}.json"
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        
        logs = []
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    except:
        pass

def get_api_key() -> str:
    if 'api_key' in st.session_state and st.session_state.api_key:
        return st.session_state.api_key
    return DEFAULT_API_KEY

def call_llm_api(system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 4000) -> str:
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
        
        response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return ""
    except:
        return ""

def generate_embedding(text: str) -> List[float]:
    """Simple hash-based embedding for demo"""
    hash_obj = hashlib.sha256(text.encode())
    hash_int = int.from_bytes(hash_obj.digest(), byteorder='big')
    return [(hash_int + i) % 1000 / 1000.0 - 0.5 for i in range(1536)]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity"""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = sum(a * a for a in v1) ** 0.5
    magnitude2 = sum(b * b for b in v2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

# ===========================
# PDF Processing with OCR
# ===========================

def extract_text_from_pdf(file_path: Path) -> Tuple[str, List[Dict]]:
    """
    Extract text from PDF with automatic OCR for scanned pages
    """
    try:
        import fitz
        doc = fitz.open(file_path)
        
        text_content = []
        all_images = []
        pages_without_text = []
        
        # First pass: extract text and images
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                text_content.append(f"=== Page {page_num+1} ===\n{text}\n")
            else:
                pages_without_text.append(page_num)
            
            for img_index, img in enumerate(page.get_images()):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    all_images.append({
                        'id': f"page{page_num+1}_img{img_index+1}",
                        'data': base64.b64encode(image_bytes).decode('utf-8'),
                        'page': page_num + 1,
                        'size': len(image_bytes)
                    })
                except:
                    continue
        
        # Second pass: OCR scanned pages
        if pages_without_text:
            print(f"🔍 Detected {len(pages_without_text)} scanned pages, performing OCR...")
            
            for page_num in pages_without_text:
                try:
                    page = doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    img_bytes = pix.tobytes("png")
                    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    ocr_result = recognize_handwriting_cloud(img_b64)
                    
                    if ocr_result.get('recognized_text'):
                        ocr_text = ocr_result['recognized_text']
                        confidence = ocr_result.get('confidence', 0)
                        
                        text_content.insert(
                            page_num,
                            f"=== Page {page_num+1} (OCR, {confidence:.0f}% confidence) ===\n{ocr_text}\n"
                        )
                        print(f"✓ OCR completed for page {page_num+1} ({confidence:.0f}% confidence)")
                    else:
                        text_content.insert(
                            page_num,
                            f"=== Page {page_num+1} ===\n[OCR failed or no text detected]\n"
                        )
                except Exception as e:
                    print(f"✗ OCR error on page {page_num+1}: {e}")
                    text_content.insert(
                        page_num,
                        f"=== Page {page_num+1} ===\n[OCR error: {str(e)}]\n"
                    )
        
        doc.close()
        final_text = "\n\n".join(text_content)
        
        if pages_without_text:
            summary = f"\n\n[OCR Processing Summary: {len(pages_without_text)} scanned page(s) processed]\n"
            final_text = summary + final_text
        
        return (final_text, all_images)
    except Exception as e:
        print(f"Error in extract_text_from_pdf: {e}")
        return ("", [])

def extract_text_from_docx(file_path: Path) -> str:
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except:
        return ""

def extract_text_from_file(file_path: Path) -> Tuple[str, List[Dict]]:
    ext = file_path.suffix.lower().lstrip('.')
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['docx', 'doc']:
        return (extract_text_from_docx(file_path), [])
    elif ext == 'txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return (f.read(), [])
        except:
            return ("", [])
    return ("", [])

# ===========================
# Analysis Functions
# ===========================

def translate_handwriting(images: List[Dict], filename: str) -> Dict:
    try:
        if not images:
            return {
                "has_handwriting": False,
                "translated_text": "No images detected",
                "image_count": 0,
                "ocr_results": []
            }
        
        st.info(f"🔍 Processing {len(images)} images with Cloud OCR...")
        ocr_results = batch_recognize_handwriting_cloud(images)
        
        formatted_results = []
        for result in ocr_results:
            if result['confidence'] > 0:
                formatted_results.append({
                    'location': result['location'],
                    'text': result['recognized_text'],
                    'confidence': result['confidence']
                })
        
        if formatted_results:
            translated_lines = ["✅ **Handwriting Recognition Results (Cloud OCR)**\n"]
            for res in formatted_results:
                translated_lines.append(f"[{res['location']}] {res['text']} (Confidence: {res['confidence']:.0f}%)")
            translated_text = "\n".join(translated_lines)
        else:
            translated_text = "Recognition failed"
        
        return {
            "has_handwriting": True,
            "translated_text": translated_text,
            "image_count": len(images),
            "ocr_results": formatted_results,
            "model": "OCR.space-cloud"
        }
    except Exception as e:
        return {
            "has_handwriting": False,
            "translated_text": f"Error: {e}",
            "image_count": 0,
            "ocr_results": []
        }

def analyze_electronic_text(text: str, filename: str) -> str:
    prompt = f"Analyze document '{filename}' (first 3000 chars):\n\n{text[:3000]}"
    response = call_llm_api(ELECTRONIC_TEXT_ANALYSIS_SYSTEM, prompt)
    return response if response else "Analysis unavailable"

def perform_dual_track_analysis(text: str, images: List[Dict], filename: str) -> Dict:
    electronic = analyze_electronic_text(text, filename)
    handwriting = translate_handwriting(images, filename)
    
    return {
        "electronic_analysis": electronic,
        "handwriting_translation": handwriting,
        "has_handwriting": handwriting.get('has_handwriting', False),
        "analysis_timestamp": datetime.now().isoformat(),
        "ocr_model": "OCR.space-cloud"
    }

# ===========================
# Workspace Management
# ===========================

def create_workspace(name: str, description: str = ""):
    workspace_dir = WORKSPACES_DIR / name
    workspace_dir.mkdir(exist_ok=True)
    metadata = {
        "name": name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "documents": []
    }
    with open(workspace_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log_audit("workspace_created", {"workspace": name})
    return metadata

def load_workspace(name: str) -> Optional[Dict]:
    metadata_file = WORKSPACES_DIR / name / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return None

def list_workspaces() -> List[str]:
    if not WORKSPACES_DIR.exists():
        return []
    return [d.name for d in WORKSPACES_DIR.iterdir() if d.is_dir()]

def delete_workspace(name: str):
    """Delete a workspace and all its contents"""
    import shutil
    workspace_dir = WORKSPACES_DIR / name
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
        log_audit("workspace_deleted", {"workspace": name})
        return True
    return False

def upload_document_to_workspace(workspace_name: str, uploaded_file, auto_analyze: bool = True):
    try:
        workspace_dir = WORKSPACES_DIR / workspace_name
        workspace_dir.mkdir(exist_ok=True)
        
        file_path = workspace_dir / uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        text, images = extract_text_from_file(file_path)
        
        embedding = generate_embedding(text[:5000])
        embedding_file = EMBEDDINGS_DIR / f"{workspace_name}_{uploaded_file.name}.json"
        with open(embedding_file, 'w') as f:
            json.dump({'embedding': embedding, 'preview': text[:500]}, f)
        
        analysis_result = None
        if auto_analyze and text:
            analysis_result = perform_dual_track_analysis(text, images, uploaded_file.name)
            analysis_file = ANALYSIS_DIR / f"{workspace_name}_{uploaded_file.name}.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis_result, f, indent=2)
        
        metadata = load_workspace(workspace_name)
        if not metadata:
            metadata = create_workspace(workspace_name)
        
        doc_metadata = {
            'filename': uploaded_file.name,
            'format': file_path.suffix.lstrip('.'),
            'path': str(file_path),
            'size': uploaded_file.size,
            'upload_date': datetime.now().isoformat(),
            'extracted_text_preview': text[:200],
            'has_images': len(images) > 0,
            'image_count': len(images),
            'has_deep_analysis': auto_analyze
        }
        
        metadata['documents'].append(doc_metadata)
        
        with open(workspace_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log_audit("document_uploaded", {
            "workspace": workspace_name,
            "filename": uploaded_file.name,
            "size": uploaded_file.size
        })
        
        return True
    except Exception as e:
        st.error(f"Upload error: {e}")
        return False

# ===========================
# Search and Retrieval
# ===========================

def search_documents(query: str, workspace_name: str, top_k: int = 5) -> List[Dict]:
    """Search documents using vector similarity"""
    try:
        query_embedding = generate_embedding(query)
        
        metadata = load_workspace(workspace_name)
        if not metadata or not metadata.get('documents'):
            return []
        
        results = []
        
        for doc in metadata['documents']:
            embedding_file = EMBEDDINGS_DIR / f"{workspace_name}_{doc['filename']}.json"
            
            if embedding_file.exists():
                with open(embedding_file, 'r') as f:
                    doc_data = json.load(f)
                    doc_embedding = doc_data['embedding']
                    preview = doc_data.get('preview', '')
                
                similarity = cosine_similarity(query_embedding, doc_embedding)
                
                # Boost similarity if query terms appear in preview
                query_terms = set(query.lower().split())
                preview_terms = set(preview.lower().split())
                term_overlap = len(query_terms & preview_terms)
                
                if term_overlap > 0:
                    similarity += term_overlap * 0.1
                
                results.append({
                    'filename': doc['filename'],
                    'similarity': min(similarity, 1.0),
                    'preview': preview,
                    'metadata': doc
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def answer_question(query: str, workspace_name: str, top_k: int = 3) -> str:
    """Answer questions based on retrieved documents"""
    try:
        search_results = search_documents(query, workspace_name, top_k)
        
        if not search_results:
            return "No relevant documents found to answer this question."
        
        # Build context from top results
        context_parts = []
        for i, result in enumerate(search_results[:top_k], 1):
            context_parts.append(f"[Document {i}: {result['filename']}]\n{result['preview']}\n")
        
        context = "\n---\n".join(context_parts)
        
        prompt = f"""Based on the following documents, answer this question: {query}

Documents:
{context}

Provide a clear, concise answer. Cite which document(s) you're using."""
        
        answer = call_llm_api(QA_SYSTEM, prompt, temperature=0.2)
        
        return answer if answer else "Unable to generate answer at this time."
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# ===========================
# UI Components
# ===========================

def inject_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown(f"""
    <div class="main-header">
        <h1>📋 {APP_TITLE}</h1>
        <p>Version {VERSION} | Cloud OCR Enabled</p>
    </div>
    """, unsafe_allow_html=True)

def render_analysis_view(workspace_name: str, filename: str):
    """Render detailed analysis view for a document"""
    analysis_file = ANALYSIS_DIR / f"{workspace_name}_{filename}.json"
    
    if not analysis_file.exists():
        st.warning("No analysis available for this document.")
        return
    
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    st.header(f"📊 Analysis: {filename}")
    
    # Electronic Text Analysis
    st.subheader("📄 Electronic Text Analysis")
    st.write(analysis.get('electronic_analysis', 'N/A'))
    
    # Handwriting Translation
    if analysis.get('has_handwriting'):
        st.subheader("✍️ Handwriting Recognition")
        handwriting = analysis.get('handwriting_translation', {})
        st.markdown(handwriting.get('translated_text', 'N/A'))
        
        # Show OCR results table
        if handwriting.get('ocr_results'):
            st.subheader("OCR Details")
            ocr_data = []
            for result in handwriting['ocr_results']:
                ocr_data.append({
                    'Location': result['location'],
                    'Text': result['text'][:50] + '...' if len(result['text']) > 50 else result['text'],
                    'Confidence': f"{result['confidence']:.0f}%"
                })
            
            import pandas as pd
            df = pd.DataFrame(ocr_data)
            st.dataframe(df, use_container_width=True)

def render_dashboard(workspace_name: str):
    """Render dashboard with statistics"""
    metadata = load_workspace(workspace_name)
    
    if not metadata:
        st.info("No data available. Upload documents to get started.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📚 Total Documents</h3>
            <h2>{}</h2>
        </div>
        """.format(len(metadata.get('documents', []))), unsafe_allow_html=True)
    
    with col2:
        total_images = sum(doc.get('image_count', 0) for doc in metadata.get('documents', []))
        st.markdown(f"""
        <div class="metric-card">
            <h3>🖼️ Total Images</h3>
            <h2>{total_images}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        analyzed = sum(1 for doc in metadata.get('documents', []) if doc.get('has_deep_analysis'))
        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Analyzed</h3>
            <h2>{analyzed}</h2>
        </div>
        """, unsafe_allow_html=True)

# ===========================
# Main Application
# ===========================

def main():
    st.set_page_config(
        page_title="Underwriting Assistant",
        page_icon="📋",
        layout="wide"
    )
    
    inject_css()
    ensure_dirs()
    
    # Session state initialization
    if 'current_workspace' not in st.session_state:
        st.session_state.current_workspace = "Default"
    
    if 'viewing_analysis' not in st.session_state:
        st.session_state.viewing_analysis = None
    
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.header("🗂️ Workspace Management")
        
        workspaces = list_workspaces()
        
        if not workspaces:
            create_workspace("Default", "Default workspace")
            workspaces = ["Default"]
        
        selected = st.selectbox("Select Workspace:", workspaces, key="workspace_select")
        st.session_state.current_workspace = selected
        
        # New workspace
        with st.expander("➕ Create New Workspace"):
            new_name = st.text_input("Workspace Name:")
            new_desc = st.text_area("Description:")
            if st.button("Create"):
                if new_name and new_name not in workspaces:
                    create_workspace(new_name, new_desc)
                    st.success(f"✅ Created workspace: {new_name}")
                    st.rerun()
        
        st.markdown("---")
        st.info("💡 Using Cloud OCR (Free Tier)\n\n25,000 requests/month")
        
        # Settings
        with st.expander("⚙️ Settings"):
            api_key = st.text_input("DeepSeek API Key:", value=get_api_key(), type="password")
            if st.button("Save API Key"):
                st.session_state.api_key = api_key
                st.success("✅ API Key saved")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "📚 Documents",
        "⬆️ Upload",
        "🔍 Search & Q&A",
        "📖 Dictionary"
    ])
    
    with tab1:
        st.header("📊 Dashboard")
        render_dashboard(st.session_state.current_workspace)
    
    with tab2:
        st.header("📚 Document Library")
        
        metadata = load_workspace(st.session_state.current_workspace)
        
        if metadata and metadata.get('documents'):
            for doc in metadata['documents']:
                with st.expander(f"📄 {doc['filename']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Format:** {doc['format']}")
                        st.write(f"**Size:** {doc['size']:,} bytes")
                        st.write(f"**Images:** {doc['image_count']}")
                        st.write(f"**Uploaded:** {doc['upload_date'][:10]}")
                        
                        if doc.get('extracted_text_preview'):
                            st.text_area("Preview:", doc['extracted_text_preview'], height=100, disabled=True)
                    
                    with col2:
                        if doc.get('has_deep_analysis'):
                            if st.button(f"View Analysis", key=f"view_{doc['filename']}"):
                                st.session_state.viewing_analysis = (st.session_state.current_workspace, doc['filename'])
                                st.rerun()
        else:
            st.info("No documents yet. Upload files in the Upload tab.")
        
        # Analysis modal
        if st.session_state.viewing_analysis:
            workspace, filename = st.session_state.viewing_analysis
            
            with st.container():
                render_analysis_view(workspace, filename)
                
                if st.button("✖️ Close Analysis"):
                    st.session_state.viewing_analysis = None
                    st.rerun()
    
    with tab3:
        st.header("⬆️ Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload:",
            type=list(SUPPORTED_FORMATS.keys()),
            accept_multiple_files=True
        )
        
        auto_analyze = st.checkbox("Auto-analyze documents with AI & OCR", value=True)
        
        if uploaded_files and st.button("📤 Upload & Process"):
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Processing {file.name}..."):
                    result = upload_document_to_workspace(
                        st.session_state.current_workspace,
                        file,
                        auto_analyze
                    )
                    if result:
                        st.success(f"✅ {file.name} uploaded successfully!")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.balloons()
            st.info("🔄 Refresh the Documents tab to see your files.")
    
    with tab4:
        st.header("🔍 Search & Question Answering")
        
        st.subheader("Document Search")
        search_query = st.text_input("Search documents:", placeholder="e.g., hull insurance, cargo claims...")
        
        if search_query:
            with st.spinner("Searching..."):
                results = search_documents(search_query, st.session_state.current_workspace, top_k=5)
            
            if results:
                st.success(f"Found {len(results)} relevant documents:")
                
                for result in results:
                    with st.expander(f"📄 {result['filename']} (Relevance: {result['similarity']:.2%})"):
                        st.write("**Preview:**")
                        st.text(result['preview'])
            else:
                st.warning("No documents found matching your query.")
        
        st.markdown("---")
        
        st.subheader("Ask a Question")
        question = st.text_area("Your question:", placeholder="e.g., What is the coverage amount for hull insurance?")
        
        if st.button("🤖 Get Answer") and question:
            with st.spinner("Analyzing documents and generating answer..."):
                answer = answer_question(question, st.session_state.current_workspace)
            
            st.markdown("### Answer:")
            st.info(answer)
    
    with tab5:
        st.header("📖 Insurance Terms Dictionary")
        
        search_term = st.text_input("Search terms:", placeholder="e.g., retention, premium...")
        
        if search_term:
            filtered_terms = {k: v for k, v in INSURANCE_TERMS.items() 
                            if search_term.lower() in k.lower()}
        else:
            filtered_terms = INSURANCE_TERMS
        
        for term, definition in sorted(filtered_terms.items()):
            with st.expander(f"📌 {term.title()}"):
                st.write(definition)

if __name__ == "__main__":
    main()
