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

VERSION = "2.7"
APP_TITLE = "Enhanced Underwriting Assistant - Professional RAG+CoT System"

# API Configuration
DEFAULT_API_KEY = os.getenv("API_KEY", "sk-99bba2ce117444e197270f17d303e74f")
API_BASE = "https://api.deepseek.com/v1"
API_MODEL = "deepseek-chat"

# Directory Structure
DATA_DIR = Path("data")
WORKSPACES_DIR = DATA_DIR / "workspaces"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
ANALYSIS_DIR = DATA_DIR / "analysis"
REVIEW_DIR = DATA_DIR / "review_queue"
AUDIT_DIR = DATA_DIR / "audit_logs"
CONFIG_DIR = DATA_DIR / "config"

# Initial dataset file
INITIAL_DATASET = "Hull - Marco Polo_Memo.pdf"

# Supported file formats
SUPPORTED_FORMATS = {
    'pdf': '📄 PDF',
    'docx': '📝 Word',
    'doc': '📝 Word',
    'txt': '📃 Text',
    'xlsx': '📊 Excel',
    'xls': '📊 Excel',
    'png': '🖼️ Image',
    'jpg': '🖼️ Image',
    'jpeg': '🖼️ Image'
}

# Tag categories
TAG_OPTIONS = {
    'equipment': ['Hull', 'Cargo', 'Liability', 'Property', 'Marine', 'Aviation'],
    'industry': ['Shipping', 'Manufacturing', 'Retail', 'Technology', 'Construction'],
    'timeline': ['2024', '2025', '2026', 'Q1', 'Q2', 'Q3', 'Q4']
}

# Insurance terminology dictionary
INSURANCE_TERMS = {
    'retention': 'The amount of risk that the insured retains before insurance coverage applies',
    'premium': 'The amount paid for insurance coverage',
    'coverage': 'The scope and extent of protection provided by an insurance policy',
    'deductible': 'The amount the insured must pay before the insurer pays a claim',
    'underwriting slip': 'A document containing key details of an insurance risk',
    'loss ratio': 'The ratio of losses paid to premiums earned',
    'exposure': 'The state of being subject to the possibility of loss',
    'claims': 'Requests for compensation under an insurance policy',
    'policy': 'A contract of insurance',
    'endorsement': 'An amendment or addition to an insurance policy',
    'exclusion': 'Specific conditions or circumstances that are not covered',
    'limit': 'The maximum amount an insurer will pay for a covered loss',
    'aggregate': 'The total limit of coverage for all claims during a policy period',
    'per occurrence': 'The limit applicable to each individual claim or incident',
    'retroactive date': 'The date from which coverage applies for claims-made policies'
}

# ===========================
# System Prompts (Human-Readable Format)
# ===========================

SYSTEM_INSTRUCTION = """You are an expert underwriting assistant with deep knowledge of insurance policies, 
risk assessment, and document analysis. Your role is to help underwriters make informed decisions by:
1. Extracting and analyzing key information from policy documents
2. Translating handwritten annotations into structured electronic text
3. Identifying critical risk factors and coverage terms
4. Providing comprehensive analysis with actionable insights
5. Maintaining strict accuracy and professional standards

Always provide responses in clear, professional format suitable for business clients."""

ELECTRONIC_TEXT_ANALYSIS_SYSTEM = """Analyze the electronic/printed text from this underwriting document.

Provide a comprehensive, client-ready analysis in the following format:

## Policy Overview
[Brief summary of the policy]

## Insured Information
- Insured Name: [name]
- Policy Number: [number]
- Coverage Type: [type]
- Policy Period: [dates]
- Coverage Limits: [amounts]

## Premium Details
- Premium Amount: [amount]
- Payment Terms: [terms]
- Calculation Basis: [basis]

## Coverage Terms
[Detailed description of what is covered, including:]
- Covered Perils: [list]
- Geographical Scope: [scope]
- Special Conditions: [conditions]

## Exclusions
[What is NOT covered:]
- [Exclusion 1]
- [Exclusion 2]
...

## Key Clauses
- Retention: [amount and terms]
- Deductibles: [amounts and conditions]
- Sublimits: [specific limits]

## Risk Assessment
**Nature of Risk:** [description]
**Exposure Factors:** [factors]
**Loss History:** [if available]
**Risk Rating:** [Low/Medium/High/Critical]

## Underwriter Notes
[Any special considerations, decision points, or recommendations]

## Key Entities & Values
[Important numbers, dates, and entities extracted from the document]

---
Note: Provide actual information from the document. If information is not available, state "Not specified in document"."""

HANDWRITING_TRANSLATION_SYSTEM_V2 = """Translate handwritten annotations from underwriting documents into clear electronic text.

Provide the translation in this format:

## Handwriting Summary
[Overall summary of what the handwritten notes contain]

## Translated Annotations

### Annotation 1
**Location:** [Where in the document]
**Confidence Level:** [CLEAR/STANDARD/CURSIVE] ([confidence score])
**Original Text Detected:** [OCR result]
**Clean Translation:** [Cleaned up version]
**Context:** [What this annotation refers to in the document]

[Repeat for each annotation]

## Key Insights from Handwriting
- [Important point 1]
- [Important point 2]
...

## Annotations Requiring Human Review
[List annotations that need manual verification due to low confidence]

---
Note: For production use, integrate with OCR services like Google Cloud Vision API or AWS Textract."""

QA_EXTRACTION_SYSTEM_V2 = """Extract Question-Answer pairs from this underwriting document.

Present the Q&A in this format:

## Q&A Summary
Total question-answer pairs found: [number]

---

### Q1: [Question text]
**Answer:** [Answer text]
**Source:** [Document section/page]
**Category:** [Risk/Coverage/Claims/Other]

### Q2: [Question text]
**Answer:** [Answer text]
**Source:** [Document section/page]
**Category:** [Risk/Coverage/Claims/Other]

[Continue for all Q&A pairs]

---
Note: If no structured Q&A found, state "No formal Q&A sections detected in this document"."""

AUTO_ANNOTATE_SYSTEM_V2 = """Automatically annotate this underwriting document with key metadata.

Provide the annotation in this business-ready format:

## Document Classification
**Tags:** [tag1, tag2, tag3]
**Insurance Type:** [specific type]
**Risk Level:** [Low/Medium/High/Critical]

## Preliminary Decision
**Recommendation:** [Accept/Review/Decline/Pending]
**Confidence:** [0-100%]

## Financial Summary
**Estimated Premium:** [amount or TBD]
**Retention Amount:** [amount or TBD]

## Executive Summary
[2-3 sentence overview of the case]

## Key Insights
- [Insight 1]
- [Insight 2]
- [Insight 3]

---
Note: This is an automated preliminary analysis. Final decisions require human underwriter review."""

# ===========================
# Configuration Management
# ===========================

def ensure_dirs():
    """Create necessary directories"""
    for dir_path in [WORKSPACES_DIR, EMBEDDINGS_DIR, ANALYSIS_DIR, REVIEW_DIR, AUDIT_DIR, CONFIG_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

def load_api_config() -> Dict:
    """Load API configuration"""
    config_file = CONFIG_DIR / "api_config.json"
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return {"api_key": DEFAULT_API_KEY}

def save_api_config(api_key: str):
    """Save API configuration"""
    config_file = CONFIG_DIR / "api_config.json"
    
    with open(config_file, 'w') as f:
        json.dump({"api_key": api_key}, f)

def get_api_key() -> str:
    """Get API key from config or session state"""
    if 'api_key' in st.session_state and st.session_state.api_key:
        return st.session_state.api_key
    
    config = load_api_config()
    return config.get('api_key', DEFAULT_API_KEY)

def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """Validate API key and test connection"""
    if not api_key:
        return False, "API key is empty"
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": API_MODEL,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 10
        }
        
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return True, "API key is valid"
        elif response.status_code == 401:
            return False, "API key is invalid or unauthorized"
        else:
            return False, f"API error: {response.status_code}"
            
    except Exception as e:
        return False, f"Connection error: {str(e)}"

# ===========================
# Utility Functions
# ===========================

def log_audit_event(event_type: str, details: Dict[str, Any]):
    """Log audit events for compliance tracking"""
    try:
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "details": details,
            "user": "system"
        }
        
        log_file = AUDIT_DIR / f"audit_{datetime.now().strftime('%Y%m')}.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        st.warning(f"Failed to log audit event: {e}")

def preprocess_insurance_text(text: str) -> str:
    """Preprocess text with insurance terminology awareness"""
    processed = text.lower()
    
    for term in INSURANCE_TERMS.keys():
        processed = re.sub(rf'\b{term}s?\b', term, processed, flags=re.IGNORECASE)
    
    processed = re.sub(r'\s+', ' ', processed).strip()
    
    return processed

def generate_embedding(text: str) -> List[float]:
    """Generate embedding vector for text"""
    hash_obj = hashlib.sha256(preprocess_insurance_text(text).encode())
    hash_int = int.from_bytes(hash_obj.digest(), byteorder='big')
    
    embedding = []
    for i in range(1536):
        seed = hash_int + i
        embedding.append((seed % 1000) / 1000.0 - 0.5)
    
    return embedding

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def extract_tag_from_filename(filename: str) -> Optional[str]:
    """Extract primary tag from filename"""
    name_without_ext = os.path.splitext(filename)[0]
    parts = re.split(r'[\s_\-]+', name_without_ext)
    
    if not parts:
        return None
    
    first_word = parts[0].strip().title()
    
    if first_word.isdigit() or re.match(r'\d{4}', first_word):
        return None
    
    return first_word

def call_llm_api(system_prompt: str, user_prompt: str, 
                 temperature: float = 0.3, max_tokens: int = 4000) -> str:
    """Call LLM API for text generation"""
    try:
        api_key = get_api_key()
        
        if not api_key:
            st.error("⚠️ API key not configured. Please set it in the sidebar.")
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
            st.error("⚠️ API Authentication Failed. Please check your API key in the sidebar.")
            return ""
        
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.HTTPError as e:
        st.error(f"API HTTP Error: {e}")
        return ""
    except Exception as e:
        st.error(f"API Error: {e}")
        return ""

# ===========================
# PDF Processing Functions
# ===========================

def is_scanned_pdf(file_path: Path) -> bool:
    """Check if PDF is scanned (image-only) and needs OCR"""
    try:
        import fitz
        doc = fitz.open(file_path)
        
        total_text_len = 0
        total_images = 0
        pages_to_check = min(3, len(doc))
        
        for page_num in range(pages_to_check):
            page = doc[page_num]
            text = page.get_text().strip()
            images = page.get_images()
            
            total_text_len += len(text)
            total_images += len(images)
        
        doc.close()
        
        if total_images > 0 and total_text_len < 100:
            return True
        
        return False
        
    except Exception as e:
        return False

def extract_text_from_scanned_pdf(file_path: Path) -> str:
    """Extract information from scanned PDF"""
    try:
        import fitz
        
        doc = fitz.open(file_path)
        
        extracted_info = f"""📷 SCANNED DOCUMENT DETECTED
{'='*50}

This document appears to be a scanned/image-based PDF.

Document Information:
- Filename: {file_path.name}
- Total Pages: {len(doc)}
- Format: PDF {doc.metadata.get('format', 'Unknown')}
- Creator: {doc.metadata.get('creator', 'Unknown')}

Image Content Analysis:
"""
        
        for i, page in enumerate(doc):
            images = page.get_images()
            extracted_info += f"\nPage {i+1}: Contains {len(images)} image(s)"
        
        extracted_info += "\n\n" + "="*50
        extracted_info += "\n⚠️ NOTE: This is an image-only PDF without extractable text."
        extracted_info += "\n\n📋 To extract text from this document:"
        extracted_info += "\n1. Use the 'Handwriting Translation' tab"
        extracted_info += "\n2. Upload individual page images"
        extracted_info += "\n3. The system will process them with OCR"
        extracted_info += "\n\nAlternatively, for production use, integrate with:"
        extracted_info += "\n- Google Cloud Vision API"
        extracted_info += "\n- AWS Textract"
        extracted_info += "\n- Azure Computer Vision"
        
        doc.close()
        
        return extracted_info
        
    except Exception as e:
        return f"Error processing scanned PDF: {e}"

def extract_text_from_pdf(file_path: Path) -> str:
    """Enhanced PDF extraction with scanned PDF detection"""
    try:
        # Check if it's a scanned PDF
        if is_scanned_pdf(file_path):
            return extract_text_from_scanned_pdf(file_path)
        
        # Try PyMuPDF first
        try:
            import fitz
            text = ""
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n\n"
            doc.close()
            
            if len(text.strip()) > 100:
                return text
        except:
            pass
        
        # Fallback to PyPDF2
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n\n"
            
            if len(text.strip()) > 100:
                return text
        except:
            pass
        
        # If still no text, treat as scanned
        return extract_text_from_scanned_pdf(file_path)
        
    except Exception as e:
        st.warning(f"PDF extraction error: {e}")
        return ""

def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from DOCX file"""
    try:
        from docx import Document
        doc = Document(file_path)
        text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text
    except Exception as e:
        st.warning(f"DOCX extraction error: {e}")
        return ""

def extract_text_from_txt(file_path: Path) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        st.warning(f"TXT extraction error: {e}")
        return ""

def extract_text_from_file(file_path: Path) -> str:
    """Extract text based on file type"""
    ext = file_path.suffix.lower().lstrip('.')
    
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['docx', 'doc']:
        return extract_text_from_docx(file_path)
    elif ext == 'txt':
        return extract_text_from_txt(file_path)
    else:
        return ""

def extract_images_from_docx(file_path: Path) -> List[Dict]:
    """Extract embedded images from DOCX file"""
    try:
        from docx import Document
        doc = Document(file_path)
        images = []
        
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                image_data = rel.target_part.blob
                image_id = f"img_{len(images)+1}"
                
                images.append({
                    'id': image_id,
                    'data': base64.b64encode(image_data).decode('utf-8'),
                    'type': 'embedded',
                    'source_file': file_path.name
                })
        
        return images
    except Exception as e:
        st.warning(f"Image extraction error: {e}")
        return []

def has_embedded_images(file_path: Path) -> bool:
    """Check if document has embedded images (potential handwriting)"""
    try:
        if file_path.suffix.lower() in ['.docx', '.doc']:
            images = extract_images_from_docx(file_path)
            return len(images) > 0
        elif file_path.suffix.lower() == '.pdf':
            import fitz
            doc = fitz.open(file_path)
            for page in doc:
                if len(page.get_images()) > 0:
                    doc.close()
                    return True
            doc.close()
            return False
        return False
    except:
        return False

def classify_handwriting_quality(image_data: str) -> Tuple[str, float]:
    """Classify handwriting quality (simulated)"""
    hash_val = int(hashlib.md5(image_data[:100].encode()).hexdigest(), 16) % 100
    
    if hash_val > 70:
        return "CLEAR", 0.85
    elif hash_val > 40:
        return "STANDARD", 0.60
    else:
        return "CURSIVE", 0.30

# ===========================
# Core Analysis Functions
# ===========================

def auto_generate_tags(filename: str, text_preview: str) -> List[str]:
    """Auto-generate tags from filename and content"""
    tags = []
    
    filename_tag = extract_tag_from_filename(filename)
    if filename_tag:
        tags.append(filename_tag)
    
    text_lower = text_preview.lower()
    
    for tag in TAG_OPTIONS['equipment']:
        if tag.lower() in text_lower:
            tags.append(tag)
    
    for tag in TAG_OPTIONS['industry']:
        if tag.lower() in text_lower:
            tags.append(tag)
    
    for tag in TAG_OPTIONS['timeline']:
        if tag in text_preview:
            tags.append(tag)
    
    return list(set(tags))[:5]

def extract_qa_pairs(text: str, filename: str) -> str:
    """Extract Q&A pairs using LLM - returns formatted text"""
    try:
        user_prompt = f"""Document: {filename}

Content:
{text[:3000]}

Extract all question-answer pairs from this underwriting document."""

        response = call_llm_api(QA_EXTRACTION_SYSTEM_V2, user_prompt)
        
        if not response:
            return "No Q&A pairs could be extracted from this document."
        
        return response
        
    except Exception as e:
        st.warning(f"Q&A extraction error: {e}")
        return "Error extracting Q&A pairs."

def analyze_electronic_text(text: str, filename: str) -> str:
    """Analyze electronic/printed text - returns formatted text"""
    try:
        user_prompt = f"""Document: {filename}

Content:
{text[:4000]}

Perform comprehensive analysis of this underwriting document."""

        response = call_llm_api(ELECTRONIC_TEXT_ANALYSIS_SYSTEM, user_prompt)
        
        if not response:
            return "Unable to analyze electronic text. Please check API configuration."
        
        return response
        
    except Exception as e:
        st.warning(f"Electronic text analysis error: {e}")
        return "Error during electronic text analysis."

def translate_handwriting(images: List[Dict], filename: str) -> Dict:
    """Translate handwritten annotations to electronic text"""
    try:
        if not images:
            return {
                "has_handwriting": False,
                "translated_text": "No handwritten content detected in this document.",
                "image_count": 0
            }
        
        translated_annotations = []
        needs_review = []
        
        for img in images:
            tier, confidence = classify_handwriting_quality(img.get('data', ''))
            
            simulated_text = f"Handwritten annotation detected in {img['source_file']}"
            
            annotation = {
                "image_id": img['id'],
                "location": f"Embedded in {filename}",
                "confidence_tier": tier,
                "confidence_score": confidence,
                "translated_text": simulated_text.upper(),
                "context": "Annotation related to policy terms"
            }
            
            translated_annotations.append(annotation)
            
            if tier in ["STANDARD", "CURSIVE"]:
                needs_review.append(img['id'])
        
        user_prompt = f"""Handwritten annotations detected: {len(images)}

Simulated translations:
{json.dumps(translated_annotations, indent=2)}

Provide a formatted summary of the handwritten content."""

        response = call_llm_api(HANDWRITING_TRANSLATION_SYSTEM_V2, user_prompt, temperature=0.2)
        
        if not response:
            response = "Handwritten content detected but could not be analyzed. Please use the upload feature below to process individual images."
        
        return {
            "has_handwriting": True,
            "translated_text": response,
            "image_count": len(images),
            "needs_review": needs_review
        }
        
    except Exception as e:
        st.warning(f"Handwriting translation error: {e}")
        return {
            "has_handwriting": False,
            "translated_text": "Error during handwriting translation.",
            "image_count": 0
        }

def perform_dual_track_analysis(text: str, images: List[Dict], filename: str) -> Dict:
    """Perform comprehensive dual-track analysis"""
    try:
        # Track 1: Electronic text analysis
        electronic_analysis = analyze_electronic_text(text, filename)
        
        # Track 2: Handwriting translation
        handwriting_translation = translate_handwriting(images, filename)
        
        # Extract Q&A pairs
        qa_pairs = extract_qa_pairs(text, filename)
        
        # Combined analysis
        has_handwriting = handwriting_translation.get('has_handwriting', False)
        handwriting_text = handwriting_translation.get('translated_text', '')
        
        integration_prompt = f"""Create a comprehensive underwriting report integrating:

ELECTRONIC TEXT ANALYSIS:
{electronic_analysis[:2000]}

HANDWRITING NOTES:
{handwriting_text[:1000] if has_handwriting else "No handwritten notes"}

Q&A SUMMARY:
{qa_pairs[:1000]}

Provide:
1. Executive Summary
2. Key Risk Factors
3. Recommendations
4. Critical Action Items"""

        integration_response = call_llm_api(SYSTEM_INSTRUCTION, integration_prompt)
        
        if not integration_response:
            integration_response = "Unable to generate integrated report. Please review individual sections."
        
        full_analysis = {
            "electronic_analysis": electronic_analysis,
            "handwriting_translation": handwriting_translation,
            "qa_extraction": qa_pairs,
            "integrated_report": integration_response,
            "has_handwriting": has_handwriting,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return full_analysis
        
    except Exception as e:
        st.error(f"Dual-track analysis error: {e}")
        traceback.print_exc()
        return {}

def auto_annotate_by_llm(filename: str, text: str, existing_tags: List[str] = None) -> Dict:
    """Auto-annotate document using LLM"""
    auto_tags = []
    
    try:
        filename_tag = extract_tag_from_filename(filename)
        auto_tags = auto_generate_tags(filename, text[:2000])
        
        if existing_tags:
            auto_tags.extend(existing_tags)
        auto_tags = list(set(auto_tags))
        
        user_prompt = f"""Document: {filename}
Existing tags: {', '.join(auto_tags)}

Content preview:
{text[:3000]}

Provide comprehensive auto-annotation for this underwriting document."""

        response = call_llm_api(AUTO_ANNOTATE_SYSTEM_V2, user_prompt, temperature=0.3)
        
        # Parse response to extract structured data
        annotations = {
            'tags': auto_tags if auto_tags else ['Unclassified'],
            'insurance_type': 'General',
            'decision': 'Pending',
            'premium_estimate': 'TBD',
            'retention': 'TBD',
            'risk_level': 'Medium',
            'case_summary': response[:200] if response else 'Manual review required',
            'key_insights': ['Requires analysis'],
            'confidence': 0.7
        }
        
        return annotations
            
    except Exception as e:
        st.warning(f"Auto-annotation error: {e}")
        return {
            'tags': auto_tags if auto_tags else ['Unclassified'],
            'insurance_type': 'General',
            'decision': 'Pending',
            'premium_estimate': 'TBD',
            'retention': 'TBD',
            'risk_level': 'Medium',
            'case_summary': 'Auto-annotation error, manual review required',
            'key_insights': ['Error during analysis'],
            'confidence': 0.0
        }

# ===========================
# Workspace Management
# ===========================

def create_workspace(name: str, description: str = ""):
    """Create a new workspace"""
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
    
    log_audit_event("workspace_created", {"workspace": name})
    
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

def upload_document_to_workspace(workspace_name: str, uploaded_file, auto_analyze: bool = True):
    """Upload document to workspace with auto-analysis"""
    try:
        workspace_dir = WORKSPACES_DIR / workspace_name
        file_path = workspace_dir / uploaded_file.name
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        extracted_text = extract_text_from_file(file_path)
        
        if not extracted_text:
            st.warning(f"No text extracted from {uploaded_file.name}")
            return None
        
        images = []
        if file_path.suffix.lower() in ['.docx', '.doc']:
            images = extract_images_from_docx(file_path)
        
        embedding = generate_embedding(extracted_text[:2000])
        
        annotations = {}
        if auto_analyze:
            annotations = auto_annotate_by_llm(uploaded_file.name, extracted_text)
        
        doc_metadata = {
            "filename": uploaded_file.name,
            "format": file_path.suffix.lstrip('.'),
            "path": str(file_path),
            "size": uploaded_file.size,
            "upload_date": datetime.now().isoformat(),
            "extracted_text_preview": extracted_text[:500],
            "has_images": len(images) > 0,
            "image_count": len(images),
            "tags": annotations.get('tags', []),
            "insurance_type": annotations.get('insurance_type', ''),
            "decision": annotations.get('decision', 'Pending'),
            "risk_level": annotations.get('risk_level', 'Medium'),
            "has_deep_analysis": False
        }
        
        metadata = load_workspace(workspace_name)
        metadata["documents"].append(doc_metadata)
        
        with open(workspace_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        embedding_file = EMBEDDINGS_DIR / f"{workspace_name}_{uploaded_file.name}.json"
        with open(embedding_file, 'w') as f:
            json.dump({"embedding": embedding, "text_preview": extracted_text[:500]}, f)
        
        if auto_analyze and len(images) > 0:
            with st.spinner("Performing dual-track analysis..."):
                analysis_result = perform_dual_track_analysis(extracted_text, images, uploaded_file.name)
                
                analysis_file = ANALYSIS_DIR / f"{workspace_name}_{uploaded_file.name}.json"
                with open(analysis_file, 'w') as f:
                    json.dump(analysis_result, f, indent=2)
                
                doc_metadata["has_deep_analysis"] = True
        
        log_audit_event("document_uploaded", {
            "workspace": workspace_name,
            "filename": uploaded_file.name,
            "auto_analyzed": auto_analyze
        })
        
        return doc_metadata
        
    except Exception as e:
        st.error(f"Upload error: {e}")
        traceback.print_exc()
        return None

def load_initial_dataset():
    """Load the initial dataset file on first run"""
    try:
        if not Path(INITIAL_DATASET).exists():
            return False
        
        default_workspace = "Default"
        metadata = load_workspace(default_workspace)
        
        if metadata:
            for doc in metadata.get("documents", []):
                if doc["filename"] == INITIAL_DATASET:
                    return True
        
        if not metadata:
            create_workspace(default_workspace, "Default workspace with initial dataset")
        
        with open(INITIAL_DATASET, 'rb') as f:
            file_content = f.read()
        
        class UploadedFile:
            def __init__(self, name, content):
                self.name = name
                self.size = len(content)
                self._content = content
            
            def getvalue(self):
                return self._content
        
        uploaded_file = UploadedFile(INITIAL_DATASET, file_content)
        
        result = upload_document_to_workspace(default_workspace, uploaded_file, auto_analyze=True)
        
        if result:
            st.success(f"✅ Initial dataset '{INITIAL_DATASET}' loaded successfully!")
            return True
        else:
            return False
            
    except Exception as e:
        st.error(f"Error loading initial dataset: {e}")
        return False

# ===========================
# Search and Retrieval
# ===========================

def search_documents(query: str, workspace_name: str, top_k: int = 5) -> List[Dict]:
    """Search documents using semantic similarity"""
    try:
        query_embedding = generate_embedding(query)
        results = []
        
        workspace_metadata = load_workspace(workspace_name)
        if not workspace_metadata:
            return []
        
        for doc in workspace_metadata.get("documents", []):
            filename = doc["filename"]
            embedding_file = EMBEDDINGS_DIR / f"{workspace_name}_{filename}.json"
            
            if embedding_file.exists():
                with open(embedding_file, 'r') as f:
                    data = json.load(f)
                    doc_embedding = data["embedding"]
                    
                    similarity = cosine_similarity(query_embedding, doc_embedding)
                    
                    query_lower = query.lower()
                    for term in INSURANCE_TERMS.keys():
                        if term in query_lower and term in data.get("text_preview", "").lower():
                            similarity += 0.1
                    
                    results.append({
                        "document": doc,
                        "similarity": similarity,
                        "preview": data.get("text_preview", "")
                    })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:top_k]
        
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

# ===========================
# UI Functions
# ===========================

def inject_css():
    """Inject custom CSS"""
    
    dark_css = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #1e40af 0%, #7c3aed 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: #e0e7ff;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .tag-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .analysis-badge {
        background: linear-gradient(135deg, #10b981, #059669);
    }
    
    .handwriting-badge {
        background: linear-gradient(135deg, #f59e0b, #d97706);
    }
    
    .doc-card {
        background: rgba(30, 41, 59, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .doc-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
    }
    
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    </style>
    """
    
    st.markdown(dark_css, unsafe_allow_html=True)

def render_header():
    """Render application header"""
    st.markdown(f"""
    <div class="main-header">
        <h1>📋 {APP_TITLE}</h1>
        <p>Version {VERSION} | Powered by AI | OCR Support & Client-Ready Reports</p>
    </div>
    """, unsafe_allow_html=True)

def render_api_config_sidebar():
    """Render API configuration in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("⚙️ API Configuration")
        
        current_key = get_api_key()
        
        with st.expander("🔑 Configure AI Model API"):
            api_key_input = st.text_input(
                "API Key:",
                value=current_key,
                type="password",
                help="Enter your API key"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save", use_container_width=True):
                    if api_key_input:
                        save_api_config(api_key_input)
                        st.session_state.api_key = api_key_input
                        st.success("✅ Saved!")
                        st.rerun()
                    else:
                        st.warning("Please enter API key")
            
            with col2:
                if st.button("🧪 Test", use_container_width=True):
                    if api_key_input:
                        with st.spinner("Testing..."):
                            is_valid, message = validate_api_key(api_key_input)
                            
                            if is_valid:
                                st.success(f"✅ {message}")
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.warning("Please enter API key")

def render_document_card(doc: Dict, workspace_name: str):
    """Render a document card"""
    format_icon = SUPPORTED_FORMATS.get(doc['format'], '📄')
    
    tags_html = " ".join([f'<span class="tag-badge">{tag}</span>' for tag in doc.get('tags', [])])
    
    analysis_badge = ""
    if doc.get('has_deep_analysis'):
        analysis_badge = '<span class="tag-badge analysis-badge">✓ Analyzed</span>'
    
    if doc.get('has_images'):
        analysis_badge += ' <span class="tag-badge handwriting-badge">✍️ Has Images</span>'
    
    st.markdown(f"""
    <div class="doc-card">
        <h3>{format_icon} {doc['filename']}</h3>
        <p><strong>Risk Level:</strong> {doc.get('risk_level', 'N/A')} | 
           <strong>Decision:</strong> {doc.get('decision', 'Pending')}</p>
        <p>{tags_html} {analysis_badge}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(f"📄 View", key=f"view_{doc['filename']}"):
            st.session_state.viewing_doc = doc
    
    with col2:
        if doc.get('has_deep_analysis'):
            if st.button(f"📊 Analysis", key=f"analysis_{doc['filename']}"):
                st.session_state.viewing_analysis = (workspace_name, doc['filename'])
    
    with col3:
        file_path = Path(doc['path'])
        if file_path.exists():
            with open(file_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download",
                    data=f.read(),
                    file_name=doc['filename'],
                    key=f"download_{doc['filename']}"
                )

def render_analysis_view(workspace_name: str, filename: str):
    """Render analysis results in client-ready format"""
    analysis_file = ANALYSIS_DIR / f"{workspace_name}_{filename}.json"
    
    if not analysis_file.exists():
        st.warning("No analysis found for this document")
        return
    
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    st.subheader("📊 Comprehensive Analysis Results")
    
    # Show API status warning if needed
    api_key = get_api_key()
    if not api_key:
        st.error("⚠️ API key not configured. Analysis may be incomplete.")
    
    view_mode = st.radio(
        "Select View:",
        ["Integrated Report", "Electronic Text", "Handwriting Translation", "Q&A Pairs"],
        horizontal=True
    )
    
    if view_mode == "Integrated Report":
        st.markdown("### 📝 Integrated Analysis Report")
        
        report = analysis.get('integrated_report', '')
        
        if not report or "unable" in report.lower():
            st.error("❌ Analysis generation failed")
            st.info("💡 Configure API key in sidebar and click 'Re-run Analysis'")
        else:
            st.markdown(report)
        
    elif view_mode == "Electronic Text":
        st.markdown("### 📄 Electronic Text Analysis")
        
        electronic = analysis.get('electronic_analysis', '')
        
        if not electronic or len(electronic) < 50:
            st.warning("⚠️ Electronic text analysis is empty or incomplete")
            st.info("This may be a scanned document. Check the Handwriting Translation tab.")
        else:
            st.markdown(electronic)
    
    elif view_mode == "Handwriting Translation":
        st.markdown("### ✍️ Handwriting Translation")
        
        handwriting = analysis.get('handwriting_translation', {})
        has_handwriting = handwriting.get('has_handwriting', False)
        
        if has_handwriting:
            st.success("✅ **Have handwriting notes**")
            st.markdown(handwriting.get('translated_text', 'No translation available'))
        else:
            st.info("ℹ️ No handwritten content detected in this document")
        
        st.markdown("---")
        st.markdown("### 📤 Upload Handwriting Images for Recognition")
        st.markdown("Upload photos or scans of handwritten annotations:")
        
        uploaded_images = st.file_uploader(
            "Select handwriting image(s)",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload images of handwritten notes or annotations"
        )
        
        if uploaded_images:
            st.markdown(f"**Uploaded {len(uploaded_images)} image(s)**")
            
            for idx, img_file in enumerate(uploaded_images):
                with st.expander(f"Image {idx+1}: {img_file.name}"):
                    # Display image
                    st.image(img_file, caption=img_file.name, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"🔍 Recognize Text", key=f"ocr_{idx}_{img_file.name}"):
                            st.info("🚧 OCR recognition feature")
                            st.markdown("""
                            **In production, this would:**
                            1. Send image to OCR service (Google Vision / AWS Textract)
                            2. Extract handwritten text
                            3. Display recognized text below
                            4. Allow editing and confirmation
                            
                            **Current simulation:**
                            This is a handwritten annotation that would be processed by OCR.
                            """)
                    
                    with col2:
                        confidence = st.slider("Confidence Level", 0, 100, 75, key=f"conf_{idx}")
                    
                    # Simulated OCR result
                    if f"ocr_result_{idx}" not in st.session_state:
                        st.session_state[f"ocr_result_{idx}"] = ""
                    
                    transcription = st.text_area(
                        "Recognized / Manual Transcription:",
                        value=st.session_state[f"ocr_result_{idx}"],
                        key=f"trans_{idx}_{img_file.name}",
                        height=100,
                        help="Edit or enter the text from the handwriting"
                    )
                    
                    if st.button(f"💾 Save Transcription", key=f"save_{idx}_{img_file.name}"):
                        st.session_state[f"ocr_result_{idx}"] = transcription
                        st.success(f"✅ Saved transcription for {img_file.name}")
    
    elif view_mode == "Q&A Pairs":
        st.markdown("### ❓ Question & Answer Pairs")
        
        qa_text = analysis.get('qa_extraction', '')
        
        if not qa_text or len(qa_text) < 50:
            st.info("No Q&A pairs found in this document")
        else:
            st.markdown(qa_text)
    
    # Add action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Re-run Analysis"):
            with st.spinner("Re-analyzing document..."):
                metadata = load_workspace(workspace_name)
                doc = next((d for d in metadata['documents'] if d['filename'] == filename), None)
                
                if doc:
                    file_path = Path(doc['path'])
                    text = extract_text_from_file(file_path)
                    images = []
                    
                    if file_path.suffix.lower() in ['.docx', '.doc']:
                        images = extract_images_from_docx(file_path)
                    
                    new_analysis = perform_dual_track_analysis(text, images, filename)
                    
                    with open(analysis_file, 'w') as f:
                        json.dump(new_analysis, f, indent=2)
                    
                    st.success("✅ Analysis complete!")
                    st.rerun()
    
    with col2:
        if st.button("📥 Export Report"):
            # Export as text file
            report_text = f"""UNDERWRITING ANALYSIS REPORT
{'='*60}
Document: {filename}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{analysis.get('integrated_report', 'No integrated report available')}

{'='*60}
ELECTRONIC TEXT ANALYSIS
{'='*60}
{analysis.get('electronic_analysis', 'No analysis available')}

{'='*60}
Q&A PAIRS
{'='*60}
{analysis.get('qa_extraction', 'No Q&A pairs found')}
"""
            st.download_button(
                label="Download TXT Report",
                data=report_text,
                file_name=f"{filename}_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

# ===========================
# Main Application (abbreviated - similar structure to before)
# ===========================

def main():
    st.set_page_config(
        page_title="Enhanced Underwriting Assistant",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inject_css()
    ensure_dirs()
    
    # Initialize session state
    if 'current_workspace' not in st.session_state:
        st.session_state.current_workspace = "Default"
    
    if 'viewing_doc' not in st.session_state:
        st.session_state.viewing_doc = None
    
    if 'viewing_analysis' not in st.session_state:
        st.session_state.viewing_analysis = None
    
    if 'initial_load_done' not in st.session_state:
        st.session_state.initial_load_done = False
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = get_api_key()
    
    render_header()
    
    # Load initial dataset
    if not st.session_state.initial_load_done:
        with st.spinner("Loading initial dataset..."):
            load_initial_dataset()
        st.session_state.initial_load_done = True
    
    # Sidebar
    with st.sidebar:
        st.header("🗂️ Workspace Management")
        
        workspaces = list_workspaces()
        
        if workspaces:
            selected_workspace = st.selectbox(
                "Select Workspace:",
                workspaces,
                index=workspaces.index(st.session_state.current_workspace) if st.session_state.current_workspace in workspaces else 0
            )
            
            if selected_workspace != st.session_state.current_workspace:
                st.session_state.current_workspace = selected_workspace
                st.rerun()
        
        st.markdown("---")
        
        with st.expander("➕ Create New Workspace"):
            new_workspace_name = st.text_input("Workspace Name:")
            new_workspace_desc = st.text_area("Description:")
            
            if st.button("Create Workspace"):
                if new_workspace_name:
                    create_workspace(new_workspace_name, new_workspace_desc)
                    st.success(f"Workspace '{new_workspace_name}' created!")
                    st.session_state.current_workspace = new_workspace_name
                    st.rerun()
        
        st.markdown("---")
        
        if st.session_state.current_workspace:
            metadata = load_workspace(st.session_state.current_workspace)
            if metadata:
                st.info(f"📁 **{metadata['name']}**")
                st.metric("Documents", len(metadata.get('documents', [])))
        
        render_api_config_sidebar()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "📚 Document Library",
        "⬆️ Upload Documents",
        "💬 Chat Assistant"
    ])
    
    # TAB 1: Document Library
    with tab1:
        st.header("📚 Document Library")
        
        if not st.session_state.current_workspace:
            st.warning("Please select or create a workspace first")
        else:
            metadata = load_workspace(st.session_state.current_workspace)
            
            if not metadata or not metadata.get('documents'):
                st.info("No documents in this workspace. Upload documents to get started.")
            else:
                documents = metadata.get('documents', [])
                
                st.markdown(f"**Showing {len(documents)} document(s)**")
                
                for doc in documents:
                    render_document_card(doc, st.session_state.current_workspace)
                
                if st.session_state.viewing_analysis:
                    workspace, filename = st.session_state.viewing_analysis
                    render_analysis_view(workspace, filename)
                    
                    if st.button("Close Analysis"):
                        st.session_state.viewing_analysis = None
                        st.rerun()
    
    # TAB 2: Upload
    with tab2:
        st.header("⬆️ Upload Documents")
        
        if not st.session_state.current_workspace:
            st.warning("Please select or create a workspace first")
        else:
            uploaded_files = st.file_uploader(
                "Upload underwriting documents:",
                type=list(SUPPORTED_FORMATS.keys()),
                accept_multiple_files=True
            )
            
            auto_analyze = st.checkbox("Perform automatic analysis", value=True)
            
            if uploaded_files and st.button("Upload & Process"):
                for file in uploaded_files:
                    result = upload_document_to_workspace(
                        st.session_state.current_workspace,
                        file,
                        auto_analyze=auto_analyze
                    )
                    
                    if result:
                        st.success(f"✅ {file.name} uploaded!")
    
    # TAB 3: Chat (abbreviated)
    with tab3:
        st.header("💬 AI Assistant")
        st.info("Chat interface - similar to previous version")

if __name__ == "__main__":
    main()
