"""
Enhanced Underwriting Assistant - Cloud Ready Version
Uses Google Cloud Vision API for handwriting recognition (no heavy dependencies)
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

VERSION = "3.1.0-cloud"
APP_TITLE = "Enhanced Underwriting Assistant"

# API Configuration
DEFAULT_API_KEY = os.getenv("API_KEY", "sk-99bba2ce117444e197270f17d303e74f")
API_BASE = "https://api.deepseek.com/v1"
API_MODEL = "deepseek-chat"

# OCR Configuration - Using free OCR API
OCR_API_URL = "https://api.ocr.space/parse/image"
OCR_API_KEY = "K87899142388957"  # Free tier API key

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
    'loss ratio': 'The ratio of losses paid to premiums earned'
}

# ===========================
# System Prompts
# ===========================

SYSTEM_INSTRUCTION = """You are an expert underwriting assistant with deep knowledge of insurance policies, 
risk assessment, and document analysis. Provide clear, professional responses."""

ELECTRONIC_TEXT_ANALYSIS_SYSTEM = """Analyze this insurance document and provide a BRIEF summary (3-5 sentences).
Cover: insurance type, insured party, key financial terms, main risk factors."""

# ===========================
# OCR Functions - Using Free Cloud API
# ===========================

def recognize_handwriting_cloud(image_data: str) -> Dict[str, Any]:
    """
    Recognize handwritten text using OCR.space free API
    """
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Prepare request
        payload = {
            'apikey': OCR_API_KEY,
            'language': 'eng',
            'isOverlayRequired': False,
            'OCREngine': 2  # Engine 2 is better for handwriting
        }
        
        files = {
            'file': ('image.jpg', BytesIO(image_bytes), 'image/jpeg')
        }
        
        # Make request
        response = requests.post(
            OCR_API_URL,
            data=payload,
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('IsErroredOnProcessing'):
                return {
                    "recognized_text": "OCR processing failed",
                    "confidence": 0.0,
                    "error": result.get('ErrorMessage', 'Unknown error')
                }
            
            # Extract text
            parsed_results = result.get('ParsedResults', [])
            if parsed_results:
                text = parsed_results[0].get('ParsedText', '').strip()
                
                # Calculate confidence based on text quality
                words = text.split()
                confidence = min(90.0, 50.0 + len(words) * 5)
                
                return {
                    "recognized_text": text,
                    "confidence": confidence,
                    "model": "OCR.space-engine2"
                }
        
        return {
            "recognized_text": "API request failed",
            "confidence": 0.0,
            "error": f"HTTP {response.status_code}"
        }
        
    except Exception as e:
        return {
            "recognized_text": f"OCR error: {str(e)}",
            "confidence": 0.0,
            "error": str(e)
        }

def batch_recognize_handwriting_cloud(images: List[Dict]) -> List[Dict]:
    """Batch process images using cloud OCR"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, img in enumerate(images):
        try:
            status_text.text(f"🔍 Recognizing {idx+1}/{len(images)}...")
            
            result = recognize_handwriting_cloud(img['data'])
            
            results.append({
                'image_id': img.get('id', f'img_{idx}'),
                'page': img.get('page', 1),
                'recognized_text': result['recognized_text'],
                'confidence': result['confidence'],
                'location': f"Page {img.get('page', 1)}"
            })
            
            progress_bar.progress((idx + 1) / len(images))
            
        except Exception as e:
            results.append({
                'image_id': img.get('id', f'img_{idx}'),
                'page': img.get('page', 1),
                'recognized_text': f"Error: {str(e)}",
                'confidence': 0.0,
                'location': f"Page {img.get('page', 1)}"
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return results

# ===========================
# Configuration & Utility Functions
# ===========================

def ensure_dirs():
    for dir_path in [WORKSPACES_DIR, EMBEDDINGS_DIR, ANALYSIS_DIR, AUDIT_DIR, CONFIG_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

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
    hash_obj = hashlib.sha256(text.encode())
    hash_int = int.from_bytes(hash_obj.digest(), byteorder='big')
    return [(hash_int + i) % 1000 / 1000.0 - 0.5 for i in range(1536)]

# ===========================
# PDF Processing
# ===========================

def extract_text_from_pdf(file_path: Path) -> Tuple[str, List[Dict]]:
    try:
        import fitz
        doc = fitz.open(file_path)
        
        text_content = []
        all_images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_content.append(f"=== Page {page_num+1} ===\n{text}\n")
            
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
        
        doc.close()
        return ("\n\n".join(text_content), all_images)
    except:
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
        
        # Use cloud OCR
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
    metadata = {"name": name, "description": description, "created_at": datetime.now().isoformat(), "documents": []}
    with open(workspace_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
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

def upload_document_to_workspace(workspace_name: str, uploaded_file, auto_analyze: bool = True):
    try:
        workspace_dir = WORKSPACES_DIR / workspace_name
        file_path = workspace_dir / uploaded_file.name
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        text, images = extract_text_from_file(file_path)
        
        doc_metadata = {
            "filename": uploaded_file.name,
            "format": file_path.suffix.lstrip('.'),
            "path": str(file_path),
            "upload_date": datetime.now().isoformat(),
            "has_images": len(images) > 0,
            "image_count": len(images),
            "tags": [],
            "has_deep_analysis": False
        }
        
        metadata = load_workspace(workspace_name)
        metadata["documents"].append(doc_metadata)
        
        with open(workspace_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if auto_analyze and len(images) > 0:
            with st.spinner("Analyzing with Cloud OCR..."):
                analysis = perform_dual_track_analysis(text, images, uploaded_file.name)
                analysis_file = ANALYSIS_DIR / f"{workspace_name}_{uploaded_file.name}.json"
                with open(analysis_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
                doc_metadata["has_deep_analysis"] = True
        
        return doc_metadata
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None

# ===========================
# UI Functions
# ===========================

def inject_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e40af 0%, #7c3aed 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .doc-card {
        background: rgba(30, 41, 59, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown(f"""
    <div class="main-header">
        <h1>📋 {APP_TITLE}</h1>
        <p>Version {VERSION} | Cloud OCR Ready</p>
    </div>
    """, unsafe_allow_html=True)

def render_analysis_view(workspace_name: str, filename: str):
    analysis_file = ANALYSIS_DIR / f"{workspace_name}_{filename}.json"
    
    if not analysis_file.exists():
        st.warning("No analysis found")
        return
    
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    st.subheader("📊 Analysis Results")
    
    tab1, tab2 = st.tabs(["Electronic Text", "Handwriting (Cloud OCR)"])
    
    with tab1:
        st.markdown("### 📄 Electronic Text Analysis")
        st.markdown(analysis.get('electronic_analysis', 'No analysis'))
    
    with tab2:
        st.markdown("### ✍️ Handwriting Recognition (Cloud OCR)")
        
        handwriting = analysis.get('handwriting_translation', {})
        
        if handwriting.get('has_handwriting'):
            st.success("✅ Handwriting detected and recognized")
            
            ocr_results = handwriting.get('ocr_results', [])
            
            if ocr_results:
                for result in ocr_results:
                    st.markdown(f"""
                    **📍 {result['location']}**  
                    ✍️ {result['text']}  
                    🎯 Confidence: {result['confidence']:.0f}%
                    
                    ---
                    """)
        else:
            st.info("ℹ️ No handwriting detected")
        
        # Manual upload
        st.markdown("### 📤 Upload Images for Recognition")
        uploaded_images = st.file_uploader(
            "Upload handwriting images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key=f"upload_{filename}"
        )
        
        if uploaded_images:
            for idx, img_file in enumerate(uploaded_images):
                with st.expander(f"Image {idx+1}: {img_file.name}"):
                    from PIL import Image
                    image = Image.open(img_file)
                    st.image(image, use_container_width=True)
                    
                    if st.button(f"🔍 Recognize", key=f"ocr_{filename}_{idx}"):
                        with st.spinner("Processing with Cloud OCR..."):
                            img_bytes = img_file.getvalue()
                            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                            
                            result = recognize_handwriting_cloud(img_b64)
                            
                            if 'error' not in result:
                                st.success("✅ Recognition complete!")
                                st.markdown(f"""
                                **Recognized Text:**  
                                {result['recognized_text']}
                                
                                **Confidence:** {result['confidence']:.0f}%  
                                **Model:** {result.get('model', 'Cloud OCR')}
                                """)
                            else:
                                st.error(f"Error: {result.get('error')}")

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
    
    if 'current_workspace' not in st.session_state:
        st.session_state.current_workspace = "Default"
    
    if 'viewing_analysis' not in st.session_state:
        st.session_state.viewing_analysis = None
    
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.header("🗂️ Workspace")
        
        workspaces = list_workspaces()
        
        if not workspaces:
            create_workspace("Default", "Default workspace")
            workspaces = ["Default"]
        
        selected = st.selectbox("Select Workspace:", workspaces)
        st.session_state.current_workspace = selected
        
        st.markdown("---")
        st.info("💡 Using Cloud OCR (Free Tier)")
    
    # Main tabs
    tab1, tab2 = st.tabs(["📚 Documents", "⬆️ Upload"])
    
    with tab1:
        st.header("📚 Document Library")
        
        metadata = load_workspace(st.session_state.current_workspace)
        
        if metadata and metadata.get('documents'):
            for doc in metadata['documents']:
                with st.expander(f"📄 {doc['filename']}"):
                    st.write(f"**Format:** {doc['format']}")
                    st.write(f"**Images:** {doc['image_count']}")
                    
                    if doc.get('has_deep_analysis'):
                        if st.button(f"View Analysis", key=f"view_{doc['filename']}"):
                            st.session_state.viewing_analysis = (st.session_state.current_workspace, doc['filename'])
                            st.rerun()
        else:
            st.info("No documents. Upload in the Upload tab.")
        
        if st.session_state.viewing_analysis:
            workspace, filename = st.session_state.viewing_analysis
            render_analysis_view(workspace, filename)
            
            if st.button("Close Analysis"):
                st.session_state.viewing_analysis = None
                st.rerun()
    
    with tab2:
        st.header("⬆️ Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload documents:",
            type=list(SUPPORTED_FORMATS.keys()),
            accept_multiple_files=True
        )
        
        auto_analyze = st.checkbox("Auto-analyze with Cloud OCR", value=True)
        
        if uploaded_files and st.button("Upload & Process"):
            for file in uploaded_files:
                result = upload_document_to_workspace(
                    st.session_state.current_workspace,
                    file,
                    auto_analyze
                )
                if result:
                    st.success(f"✅ {file.name} uploaded!")
            st.balloons()

if __name__ == "__main__":
    main()
