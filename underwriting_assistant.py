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

VERSION = "2.6"
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
# System Prompts
# ===========================

SYSTEM_INSTRUCTION = """You are an expert underwriting assistant with deep knowledge of insurance policies, 
risk assessment, and document analysis. Your role is to help underwriters make informed decisions by:
1. Extracting and analyzing key information from policy documents
2. Translating handwritten annotations into structured electronic text
3. Identifying critical risk factors and coverage terms
4. Providing comprehensive analysis with actionable insights
5. Maintaining strict accuracy and professional standards

Always structure your responses in clear, professional JSON format when requested."""

ELECTRONIC_TEXT_ANALYSIS_SYSTEM = """Analyze the electronic/printed text from this underwriting document.

Extract and structure:
1. Policy Details: insured name, policy number, coverage type, policy period, limits
2. Premium Information: premium amount, payment terms, calculation basis
3. Coverage Terms: covered perils, geographical scope, special conditions
4. Exclusions: what is NOT covered
5. Key Clauses: retention amounts, deductibles, sublimits
6. Risk Assessment: nature of risk, exposure factors, loss history
7. Underwriter Notes: any decision points or special considerations

Return ONLY valid JSON with this structure:
{
  "policy_details": {...},
  "premium_info": {...},
  "coverage_terms": {...},
  "exclusions": [...],
  "key_clauses": {...},
  "risk_assessment": {...},
  "underwriter_notes": [...],
  "key_entities": [...]
}"""

HANDWRITING_TRANSLATION_SYSTEM = """Translate and analyze handwritten annotations from this underwriting document.

Your task:
1. **OCR Confidence Assessment**: Evaluate handwriting quality (Clear/Standard/Cursive)
2. **Text Translation**: Convert handwritten text to clean electronic text
3. **Context Preservation**: Maintain the meaning and intent of annotations
4. **Location Tracking**: Note which page/section the handwriting refers to
5. **Summary**: Provide a brief summary of handwritten content

Handwriting Quality Tiers:
- CLEAR (70-100%): Clean, printed handwriting - high confidence OCR
- STANDARD (40-70%): Normal cursive - moderate confidence, may need human review
- CURSIVE (<40%): Difficult to read - requires human verification

Return ONLY valid JSON:
{
  "translated_annotations": [
    {
      "image_id": "string",
      "location": "string (e.g., Page 2, margin)",
      "confidence_tier": "CLEAR|STANDARD|CURSIVE",
      "confidence_score": 0.0-1.0,
      "original_text_detected": "string",
      "translated_text": "string (clean electronic version)",
      "context": "string (what this annotation refers to)"
    }
  ],
  "handwriting_summary": "string (overall summary of handwritten content)",
  "needs_human_review": ["list of image_ids that need verification"],
  "key_insights_from_handwriting": ["list of important points from annotations"]
}"""

QA_EXTRACTION_SYSTEM = """Extract Question-Answer pairs from this underwriting document.

Look for:
1. Formal Q&A sections (A1, A2, Q1, Q2 format)
2. Email correspondence with questions and responses
3. Underwriter queries and client responses
4. Risk assessment questions

Return ONLY valid JSON:
{
  "qa_pairs": [
    {
      "question_id": "string",
      "question": "string",
      "answer": "string",
      "source": "string (document section/page)",
      "category": "string (risk/coverage/claims/other)"
    }
  ],
  "total_pairs": number
}"""

AUTO_ANNOTATE_SYSTEM = """Automatically annotate this underwriting document with key metadata.

Analyze the document and extract:
1. Tags: relevant categorization (equipment/industry/timeline)
2. Insurance Type: specific type of insurance
3. Decision: preliminary risk decision (Accept/Review/Decline/Pending)
4. Premium: estimated premium amount
5. Risk Level: Low/Medium/High/Critical
6. Case Summary: brief executive summary
7. Key Insights: most important findings

Return ONLY valid JSON:
{
  "tags": ["tag1", "tag2", "tag3"],
  "insurance_type": "string",
  "decision": "Accept|Review|Decline|Pending",
  "premium_estimate": "string",
  "retention": "string",
  "risk_level": "Low|Medium|High|Critical",
  "case_summary": "string (2-3 sentences)",
  "key_insights": ["insight1", "insight2", "insight3"],
  "confidence": 0.0-1.0
}"""

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
    
    # Test connection
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
    
    # Normalize insurance terms
    for term in INSURANCE_TERMS.keys():
        processed = re.sub(rf'\b{term}s?\b', term, processed, flags=re.IGNORECASE)
    
    # Clean extra whitespace
    processed = re.sub(r'\s+', ' ', processed).strip()
    
    return processed

def generate_embedding(text: str) -> List[float]:
    """Generate embedding vector for text (simplified version using hash)"""
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
    """Extract primary tag from filename (first word before space/underscore/dash)"""
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
            return "{}"
        
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
            return "{}"
        
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.HTTPError as e:
        st.error(f"API HTTP Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Response: {e.response.text}")
        return "{}"
    except Exception as e:
        st.error(f"API Error: {e}")
        return "{}"

# ===========================
# File Processing Functions
# ===========================

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF file"""
    try:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
        return text
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
    """Extract embedded images from DOCX file (simulated handwriting)"""
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
    
    # Add filename-based tag
    filename_tag = extract_tag_from_filename(filename)
    if filename_tag:
        tags.append(filename_tag)
    
    # Extract from content
    text_lower = text_preview.lower()
    
    # Equipment tags
    for tag in TAG_OPTIONS['equipment']:
        if tag.lower() in text_lower:
            tags.append(tag)
    
    # Industry tags
    for tag in TAG_OPTIONS['industry']:
        if tag.lower() in text_lower:
            tags.append(tag)
    
    # Timeline tags
    for tag in TAG_OPTIONS['timeline']:
        if tag in text_preview:
            tags.append(tag)
    
    return list(set(tags))[:5]

def extract_qa_pairs(text: str, filename: str) -> Dict:
    """Extract Q&A pairs using LLM"""
    try:
        user_prompt = f"""Document: {filename}

Content:
{text[:3000]}

Extract all question-answer pairs from this underwriting document."""

        response = call_llm_api(QA_EXTRACTION_SYSTEM, user_prompt)
        
        if response == "{}":
            return {"qa_pairs": [], "total_pairs": 0}
        
        qa_data = json.loads(response)
        return qa_data
        
    except json.JSONDecodeError:
        return {"qa_pairs": [], "total_pairs": 0}
    except Exception as e:
        st.warning(f"Q&A extraction error: {e}")
        return {"qa_pairs": [], "total_pairs": 0}

def analyze_electronic_text(text: str, filename: str) -> Dict:
    """Analyze electronic/printed text"""
    try:
        user_prompt = f"""Document: {filename}

Content:
{text[:4000]}

Perform comprehensive analysis of this underwriting document."""

        response = call_llm_api(ELECTRONIC_TEXT_ANALYSIS_SYSTEM, user_prompt)
        
        if response == "{}":
            return {
                "policy_details": {},
                "premium_info": {},
                "coverage_terms": {},
                "exclusions": [],
                "key_clauses": {},
                "risk_assessment": {},
                "underwriter_notes": [],
                "key_entities": []
            }
        
        analysis = json.loads(response)
        return analysis
        
    except json.JSONDecodeError:
        return {
            "policy_details": {},
            "premium_info": {},
            "coverage_terms": {},
            "exclusions": [],
            "key_clauses": {},
            "risk_assessment": {},
            "underwriter_notes": [],
            "key_entities": []
        }
    except Exception as e:
        st.warning(f"Electronic text analysis error: {e}")
        return {}

def translate_handwriting(images: List[Dict], filename: str) -> Dict:
    """Translate handwritten annotations to electronic text"""
    try:
        if not images:
            return {
                "translated_annotations": [],
                "handwriting_summary": "No handwritten content detected",
                "needs_human_review": [],
                "key_insights_from_handwriting": []
            }
        
        translated_annotations = []
        needs_review = []
        
        for img in images:
            tier, confidence = classify_handwriting_quality(img.get('data', ''))
            
            simulated_text = f"Handwritten note from {img['source_file']}"
            
            annotation = {
                "image_id": img['id'],
                "location": f"Embedded in {filename}",
                "confidence_tier": tier,
                "confidence_score": confidence,
                "original_text_detected": simulated_text,
                "translated_text": simulated_text.upper(),
                "context": "Annotation related to policy terms"
            }
            
            translated_annotations.append(annotation)
            
            if tier in ["STANDARD", "CURSIVE"]:
                needs_review.append(img['id'])
        
        user_prompt = f"""Handwritten annotations detected: {len(images)}

Simulated translations:
{json.dumps(translated_annotations, indent=2)}

Provide a summary of the handwritten content and key insights."""

        response = call_llm_api(HANDWRITING_TRANSLATION_SYSTEM, user_prompt, temperature=0.2)
        
        if response == "{}":
            return {
                "translated_annotations": translated_annotations,
                "handwriting_summary": "Handwritten annotations detected and translated",
                "needs_human_review": needs_review,
                "key_insights_from_handwriting": ["Manual review recommended for accuracy"]
            }
        
        try:
            result = json.loads(response)
        except:
            result = {
                "translated_annotations": translated_annotations,
                "handwriting_summary": "Handwritten annotations detected and translated",
                "needs_human_review": needs_review,
                "key_insights_from_handwriting": ["Manual review recommended for accuracy"]
            }
        
        return result
        
    except Exception as e:
        st.warning(f"Handwriting translation error: {e}")
        traceback.print_exc()
        return {
            "translated_annotations": [],
            "handwriting_summary": "Translation failed",
            "needs_human_review": [],
            "key_insights_from_handwriting": []
        }

def perform_dual_track_analysis(text: str, images: List[Dict], filename: str) -> Dict:
    """Perform comprehensive dual-track analysis with handwriting integration"""
    try:
        # Track 1: Electronic text analysis
        electronic_analysis = analyze_electronic_text(text, filename)
        
        # Track 2: Handwriting translation
        handwriting_translation = translate_handwriting(images, filename)
        
        # Extract Q&A pairs
        qa_pairs = extract_qa_pairs(text, filename)
        
        # Integrate handwriting into full analysis
        translated_text = "\n".join([
            ann['translated_text'] 
            for ann in handwriting_translation.get('translated_annotations', [])
        ])
        
        # Combined analysis
        integration_prompt = f"""Integrate the following analysis components into a comprehensive underwriting report:

ELECTRONIC TEXT ANALYSIS:
{json.dumps(electronic_analysis, indent=2)}

TRANSLATED HANDWRITING:
{handwriting_translation.get('handwriting_summary', 'None')}
{translated_text}

Q&A PAIRS:
{json.dumps(qa_pairs, indent=2)}

Provide:
1. Executive Summary (incorporating handwritten insights)
2. Risk Assessment
3. Recommendations
4. Critical Action Items"""

        integration_response = call_llm_api(SYSTEM_INSTRUCTION, integration_prompt)
        
        # Combine all results
        full_analysis = {
            "electronic_analysis": electronic_analysis,
            "handwriting_translation": handwriting_translation,
            "qa_extraction": qa_pairs,
            "integrated_report": integration_response if integration_response != "{}" else "Analysis integration failed - API may be unavailable",
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return full_analysis
        
    except Exception as e:
        st.error(f"Dual-track analysis error: {e}")
        traceback.print_exc()
        return {}

def auto_annotate_by_llm(filename: str, text: str, existing_tags: List[str] = None) -> Dict:
    """Auto-annotate document using LLM"""
    # Initialize auto_tags at the beginning
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

        response = call_llm_api(AUTO_ANNOTATE_SYSTEM, user_prompt, temperature=0.3)
        
        if response == "{}":
            return {
                'tags': auto_tags if auto_tags else ['Unclassified'],
                'insurance_type': 'General',
                'decision': 'Pending',
                'premium_estimate': 'TBD',
                'retention': 'TBD',
                'risk_level': 'Medium',
                'case_summary': 'API unavailable - manual review required',
                'key_insights': ['Requires manual analysis'],
                'confidence': 0.3
            }
        
        try:
            annotations = json.loads(response)
            
            # Merge filename-based tags
            if filename_tag and filename_tag not in annotations.get('tags', []):
                annotations['tags'].insert(0, filename_tag)
            
            # Ensure required fields
            annotations.setdefault('tags', auto_tags if auto_tags else ['Unclassified'])
            annotations.setdefault('insurance_type', 'General')
            annotations.setdefault('decision', 'Pending')
            annotations.setdefault('premium_estimate', 'TBD')
            annotations.setdefault('retention', 'TBD')
            annotations.setdefault('risk_level', 'Medium')
            annotations.setdefault('case_summary', 'Underwriting case requires review')
            annotations.setdefault('key_insights', ['Comprehensive analysis needed'])
            annotations.setdefault('confidence', 0.7)
            
            return annotations
            
        except json.JSONDecodeError:
            return {
                'tags': auto_tags if auto_tags else ['Unclassified'],
                'insurance_type': 'General',
                'decision': 'Pending',
                'premium_estimate': 'TBD',
                'retention': 'TBD',
                'risk_level': 'Medium',
                'case_summary': 'Auto-annotation failed, manual review required',
                'key_insights': ['Requires manual analysis'],
                'confidence': 0.3
            }
            
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
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Extract text
        extracted_text = extract_text_from_file(file_path)
        
        if not extracted_text:
            st.warning(f"No text extracted from {uploaded_file.name}")
            return None
        
        # Extract images (for handwriting)
        images = []
        if file_path.suffix.lower() in ['.docx', '.doc']:
            images = extract_images_from_docx(file_path)
        
        # Generate embedding
        embedding = generate_embedding(extracted_text[:2000])
        
        # Auto-annotate
        annotations = {}
        if auto_analyze:
            annotations = auto_annotate_by_llm(uploaded_file.name, extracted_text)
        
        # Create document metadata
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
        
        # Update workspace metadata
        metadata = load_workspace(workspace_name)
        metadata["documents"].append(doc_metadata)
        
        with open(workspace_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save embedding
        embedding_file = EMBEDDINGS_DIR / f"{workspace_name}_{uploaded_file.name}.json"
        with open(embedding_file, 'w') as f:
            json.dump({"embedding": embedding, "text_preview": extracted_text[:500]}, f)
        
        # Perform deep analysis if requested
        if auto_analyze and len(images) > 0:
            with st.spinner("Performing dual-track analysis..."):
                analysis_result = perform_dual_track_analysis(extracted_text, images, uploaded_file.name)
                
                # Save analysis
                analysis_file = ANALYSIS_DIR / f"{workspace_name}_{uploaded_file.name}.json"
                with open(analysis_file, 'w') as f:
                    json.dump(analysis_result, f, indent=2)
                
                doc_metadata["has_deep_analysis"] = True
                
                # Add handwriting to review queue if needed
                handwriting_data = analysis_result.get('handwriting_translation', {})
                needs_review = handwriting_data.get('needs_human_review', [])
                
                if needs_review:
                    for img_id in needs_review:
                        review_item = {
                            "document": uploaded_file.name,
                            "workspace": workspace_name,
                            "image_id": img_id,
                            "added_date": datetime.now().isoformat(),
                            "status": "pending",
                            "confidence": 0.5
                        }
                        
                        review_file = REVIEW_DIR / f"{workspace_name}_{uploaded_file.name}_{img_id}.json"
                        with open(review_file, 'w') as f:
                            json.dump(review_item, f, indent=2)
        
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
            st.warning(f"Initial dataset file '{INITIAL_DATASET}' not found in root directory")
            return False
        
        # Check if already loaded
        default_workspace = "Default"
        metadata = load_workspace(default_workspace)
        
        if metadata:
            for doc in metadata.get("documents", []):
                if doc["filename"] == INITIAL_DATASET:
                    return True
        
        # Create default workspace if it doesn't exist
        if not metadata:
            create_workspace(default_workspace, "Default workspace with initial dataset")
        
        # Load the file
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
            st.error(f"Failed to load initial dataset")
            return False
            
    except Exception as e:
        st.error(f"Error loading initial dataset: {e}")
        traceback.print_exc()
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
                    
                    # Boost score for insurance terms
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
    
    .review-pending {
        background: linear-gradient(135deg, #f59e0b, #d97706);
    }
    
    .review-completed {
        background: linear-gradient(135deg, #10b981, #059669);
    }
    
    .confidence-clear {
        background: linear-gradient(135deg, #10b981, #059669);
    }
    
    .confidence-standard {
        background: linear-gradient(135deg, #f59e0b, #d97706);
    }
    
    .confidence-cursive {
        background: linear-gradient(135deg, #ef4444, #dc2626);
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
    
    .review-card {
        background: rgba(30, 41, 59, 0.9);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #dc2626;
        margin-bottom: 1.5rem;
    }
    
    .stChatMessage {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
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
    
    .stMetric {
        background: rgba(30, 41, 59, 0.6);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    .api-warning {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """
    
    st.markdown(dark_css, unsafe_allow_html=True)

def render_header():
    """Render application header"""
    st.markdown(f"""
    <div class="main-header">
        <h1>📋 {APP_TITLE}</h1>
        <p>Version {VERSION} | Powered by AI | Handwriting Translation & Dual-Track Analysis</p>
    </div>
    """, unsafe_allow_html=True)

def render_api_config_sidebar():
    """Render API configuration in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("⚙️ API Configuration")
        
        current_key = get_api_key()
        
        with st.expander("🔑 Configure AI Model API", expanded=not current_key):
            api_key_input = st.text_input(
                "API Key:",
                value=current_key,
                type="password",
                help="Enter your API key for AI model access"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save", use_container_width=True):
                    if api_key_input:
                        save_api_config(api_key_input)
                        st.session_state.api_key = api_key_input
                        st.success("✅ API key saved!")
                        st.rerun()
                    else:
                        st.warning("Please enter an API key")
            
            with col2:
                if st.button("🧪 Test", use_container_width=True):
                    if api_key_input:
                        with st.spinner("Testing API key..."):
                            is_valid, message = validate_api_key(api_key_input)
                            
                            if is_valid:
                                st.success(f"✅ {message}")
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.warning("Please enter an API key")

def render_document_card(doc: Dict, workspace_name: str):
    """Render a document card"""
    format_icon = SUPPORTED_FORMATS.get(doc['format'], '📄')
    
    tags_html = " ".join([f'<span class="tag-badge">{tag}</span>' for tag in doc.get('tags', [])])
    
    analysis_badge = ""
    if doc.get('has_deep_analysis'):
        analysis_badge = '<span class="tag-badge analysis-badge">✓ Analyzed</span>'
    
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
    """Render analysis results with debug info"""
    analysis_file = ANALYSIS_DIR / f"{workspace_name}_{filename}.json"
    
    if not analysis_file.exists():
        st.warning("No analysis found for this document")
        return
    
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    st.subheader("📊 Comprehensive Analysis Results")
    
    # Show API status
    api_key = get_api_key()
    if not api_key:
        st.error("⚠️ API key not configured. Analysis may be incomplete.")
        st.info("Please configure an API key in the sidebar (⚙️ API Configuration)")
    
    view_mode = st.radio(
        "Select View:",
        ["Integrated Report", "Electronic Text", "Handwriting Translation", "Q&A Pairs"],
        horizontal=True
    )
    
    if view_mode == "Integrated Report":
        st.markdown("### 📝 Integrated Analysis Report")
        
        report = analysis.get('integrated_report', '')
        
        # Check if report is empty or error message
        if not report or report == "{}" or "API" in report or "failed" in report.lower():
            st.error("❌ Analysis generation failed or incomplete")
            st.markdown("""
            **Possible reasons:**
            1. API key is not configured
            2. API connection failed
            3. API quota exceeded
            
            **How to fix:**
            1. Configure API key in the sidebar (⚙️ API Configuration)
            2. Click "🧪 Test" to verify the connection
            3. Click "🔄 Re-run Analysis" below
            """)
            
            # Show raw data for debugging
            with st.expander("🔍 Debug Information"):
                st.json(analysis)
        else:
            st.markdown(report)
        
    elif view_mode == "Electronic Text":
        st.markdown("### 📄 Electronic Text Analysis")
        electronic = analysis.get('electronic_analysis', {})
        
        if not electronic or len(electronic) == 0:
            st.warning("⚠️ Electronic text analysis is empty. API may have failed.")
            with st.expander("🔍 View Raw Data"):
                st.json(analysis)
        else:
            with st.expander("Policy Details", expanded=True):
                st.json(electronic.get('policy_details', {}))
            
            with st.expander("Risk Assessment"):
                st.json(electronic.get('risk_assessment', {}))
            
            with st.expander("Coverage Terms"):
                st.json(electronic.get('coverage_terms', {}))
    
    elif view_mode == "Handwriting Translation":
        st.markdown("### ✍️ Handwriting Translation Results")
        handwriting = analysis.get('handwriting_translation', {})
        
        st.markdown(f"**Summary:** {handwriting.get('handwriting_summary', 'N/A')}")
        
        st.markdown("#### Translated Annotations:")
        annotations = handwriting.get('translated_annotations', [])
        
        if not annotations:
            st.info("No handwritten content detected in this document")
        else:
            for annotation in annotations:
                tier = annotation['confidence_tier']
                confidence = annotation['confidence_score']
                
                tier_class = f"confidence-{tier.lower()}"
                
                st.markdown(f"""
                <div class="review-card">
                    <p><strong>Location:</strong> {annotation['location']}</p>
                    <p><strong>Confidence:</strong> <span class="tag-badge {tier_class}">{tier} ({confidence:.2f})</span></p>
                    <p><strong>Translated Text:</strong> {annotation['translated_text']}</p>
                    <p><strong>Context:</strong> {annotation.get('context', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        if handwriting.get('key_insights_from_handwriting'):
            st.markdown("#### Key Insights from Handwriting:")
            for insight in handwriting['key_insights_from_handwriting']:
                st.markdown(f"- {insight}")
    
    elif view_mode == "Q&A Pairs":
        st.markdown("### ❓ Question & Answer Pairs")
        qa_data = analysis.get('qa_extraction', {})
        
        pairs = qa_data.get('qa_pairs', [])
        
        if not pairs:
            st.info("No Q&A pairs found in this document")
        else:
            for qa in pairs:
                with st.expander(f"Q: {qa.get('question', 'N/A')}"):
                    st.markdown(f"**Answer:** {qa.get('answer', 'N/A')}")
                    st.markdown(f"**Source:** {qa.get('source', 'N/A')}")
                    st.markdown(f"**Category:** {qa.get('category', 'N/A')}")
    
    # Add reanalysis button
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Re-run Analysis"):
            with st.spinner("Re-analyzing document..."):
                # Load document
                metadata = load_workspace(workspace_name)
                doc = next((d for d in metadata['documents'] if d['filename'] == filename), None)
                
                if doc:
                    file_path = Path(doc['path'])
                    text = extract_text_from_file(file_path)
                    images = []
                    
                    if file_path.suffix.lower() in ['.docx', '.doc']:
                        images = extract_images_from_docx(file_path)
                    
                    # Perform analysis
                    new_analysis = perform_dual_track_analysis(text, images, filename)
                    
                    # Save results
                    with open(analysis_file, 'w') as f:
                        json.dump(new_analysis, f, indent=2)
                    
                    st.success("✅ Analysis complete! Refreshing...")
                    st.rerun()
    
    with col2:
        if st.button("🔍 View Full JSON"):
            st.json(analysis)

# ===========================
# Main Application
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
    
    if 'current_review_index' not in st.session_state:
        st.session_state.current_review_index = 0
    
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
        
        if not workspaces:
            st.info("No workspaces found. Create one to get started.")
        else:
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
                else:
                    st.warning("Please enter a workspace name")
        
        st.markdown("---")
        
        if st.session_state.current_workspace:
            metadata = load_workspace(st.session_state.current_workspace)
            if metadata:
                st.info(f"📁 **{metadata['name']}**\n\n{metadata.get('description', 'No description')}")
                st.metric("Documents", len(metadata.get('documents', [])))
        
        render_api_config_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chat Assistant",
        "📚 Document Library",
        "⬆️ Upload Documents",
        "📊 Analysis Dashboard",
        "✍️ Handwriting Review"
    ])
    
    # TAB 1: Chat Assistant
    with tab1:
        st.header("💬 AI Underwriting Assistant")
        st.markdown("Ask questions about your documents, policies, and underwriting decisions.")
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about underwriting, policies, risk assessment..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    search_results = search_documents(
                        prompt,
                        st.session_state.current_workspace,
                        top_k=3
                    )
                    
                    context = "\n\n".join([
                        f"Document: {r['document']['filename']}\n{r['preview']}"
                        for r in search_results
                    ])
                    
                    user_prompt = f"""User Question: {prompt}

Relevant Documents:
{context}

Provide a comprehensive answer based on the available documents."""

                    response = call_llm_api(SYSTEM_INSTRUCTION, user_prompt)
                    
                    if response == "{}":
                        response = "I apologize, but I'm unable to process your request at this time. Please ensure the API key is configured correctly in the sidebar."
                    
                    st.markdown(response)
                    
                    if search_results:
                        with st.expander("📚 Sources"):
                            for r in search_results:
                                st.markdown(f"- **{r['document']['filename']}** (similarity: {r['similarity']:.2f})")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # TAB 2: Document Library
    with tab2:
        st.header("📚 Document Library")
        
        if not st.session_state.current_workspace:
            st.warning("Please select or create a workspace first")
        else:
            metadata = load_workspace(st.session_state.current_workspace)
            
            if not metadata or not metadata.get('documents'):
                st.info("No documents in this workspace. Upload documents to get started.")
            else:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    search_query = st.text_input("🔍 Search documents:", placeholder="Enter keywords...")
                
                with col2:
                    filter_type = st.selectbox("Filter by:", ["All", "PDF", "Word", "Excel", "Images"])
                
                documents = metadata.get('documents', [])
                
                if filter_type != "All":
                    filter_map = {
                        "PDF": "pdf",
                        "Word": ["docx", "doc"],
                        "Excel": ["xlsx", "xls"],
                        "Images": ["png", "jpg", "jpeg"]
                    }
                    filter_formats = filter_map[filter_type]
                    if isinstance(filter_formats, list):
                        documents = [d for d in documents if d['format'] in filter_formats]
                    else:
                        documents = [d for d in documents if d['format'] == filter_formats]
                
                if search_query:
                    documents = [d for d in documents if search_query.lower() in d['filename'].lower()]
                
                st.markdown(f"**Showing {len(documents)} document(s)**")
                
                for doc in documents:
                    render_document_card(doc, st.session_state.current_workspace)
                
                if st.session_state.viewing_doc:
                    with st.expander("📄 Document Details", expanded=True):
                        doc = st.session_state.viewing_doc
                        st.json(doc)
                        
                        if st.button("Close Details"):
                            st.session_state.viewing_doc = None
                            st.rerun()
                
                if st.session_state.viewing_analysis:
                    workspace, filename = st.session_state.viewing_analysis
                    render_analysis_view(workspace, filename)
                    
                    if st.button("Close Analysis"):
                        st.session_state.viewing_analysis = None
                        st.rerun()
    
    # TAB 3: Upload Documents
    with tab3:
        st.header("⬆️ Upload Documents")
        
        if not st.session_state.current_workspace:
            st.warning("Please select or create a workspace first")
        else:
            st.markdown(f"**Current Workspace:** {st.session_state.current_workspace}")
            
            uploaded_files = st.file_uploader(
                "Upload underwriting documents:",
                type=list(SUPPORTED_FORMATS.keys()),
                accept_multiple_files=True,
                help="Supported formats: PDF, Word, Excel, Text, Images"
            )
            
            auto_analyze = st.checkbox("Perform automatic analysis", value=True,
                                      help="Includes handwriting translation and dual-track analysis")
            
            if uploaded_files and st.button("Upload & Process"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    
                    result = upload_document_to_workspace(
                        st.session_state.current_workspace,
                        file,
                        auto_analyze=auto_analyze
                    )
                    
                    if result:
                        st.success(f"✅ {file.name} uploaded successfully!")
                    else:
                        st.error(f"❌ Failed to upload {file.name}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("Upload complete!")
                st.balloons()
    
    # TAB 4: Analysis Dashboard
    with tab4:
        st.header("📊 Analysis Dashboard")
        
        if not st.session_state.current_workspace:
            st.warning("Please select a workspace first")
        else:
            metadata = load_workspace(st.session_state.current_workspace)
            
            if not metadata or not metadata.get('documents'):
                st.info("No documents to analyze")
            else:
                documents = metadata.get('documents', [])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Documents", len(documents))
                
                with col2:
                    analyzed = sum(1 for d in documents if d.get('has_deep_analysis'))
                    st.metric("Analyzed", analyzed)
                
                with col3:
                    high_risk = sum(1 for d in documents if d.get('risk_level') == 'High')
                    st.metric("High Risk", high_risk)
                
                with col4:
                    pending = sum(1 for d in documents if d.get('decision') == 'Pending')
                    st.metric("Pending Review", pending)
                
                st.markdown("---")
                
                st.subheader("Document Analysis Status")
                
                for doc in documents:
                    with st.expander(f"{doc['filename']} - {doc.get('decision', 'Pending')}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Risk Level:** {doc.get('risk_level', 'N/A')}")
                            st.markdown(f"**Tags:** {', '.join(doc.get('tags', []))}")
                            st.markdown(f"**Has Images:** {'Yes' if doc.get('has_images') else 'No'}")
                        
                        with col2:
                            if doc.get('has_deep_analysis'):
                                if st.button("View Full Analysis", key=f"view_full_{doc['filename']}"):
                                    st.session_state.viewing_analysis = (st.session_state.current_workspace, doc['filename'])
                                    st.rerun()
                            else:
                                if st.button("Run Analysis", key=f"run_{doc['filename']}"):
                                    with st.spinner("Analyzing..."):
                                        file_path = Path(doc['path'])
                                        text = extract_text_from_file(file_path)
                                        images = []
                                        
                                        if file_path.suffix.lower() in ['.docx', '.doc']:
                                            images = extract_images_from_docx(file_path)
                                        
                                        analysis = perform_dual_track_analysis(text, images, doc['filename'])
                                        
                                        analysis_file = ANALYSIS_DIR / f"{st.session_state.current_workspace}_{doc['filename']}.json"
                                        with open(analysis_file, 'w') as f:
                                            json.dump(analysis, f, indent=2)
                                        
                                        doc['has_deep_analysis'] = True
                                        
                                        with open(WORKSPACES_DIR / st.session_state.current_workspace / "metadata.json", 'w') as f:
                                            json.dump(metadata, f, indent=2)
                                        
                                        st.success("Analysis complete!")
                                        st.rerun()
    
    # TAB 5: Handwriting Review
    with tab5:
        st.header("✍️ Handwriting Review Workbench")
        st.markdown("Human-in-the-loop verification of handwritten content translation")
        
        review_files = list(REVIEW_DIR.glob("*.json"))
        
        if not review_files:
            st.info("✅ No handwritten content requires review at this time")
        else:
            col1, col2, col3 = st.columns(3)
            
            pending = sum(1 for f in review_files if json.loads(f.read_text()).get('status') == 'pending')
            completed = len(review_files) - pending
            
            with col1:
                st.metric("Total Items", len(review_files))
            with col2:
                st.metric("Pending", pending)
            with col3:
                st.metric("Completed", completed)
            
            st.markdown("---")
            
            if st.session_state.current_review_index >= len(review_files):
                st.session_state.current_review_index = 0
            
            current_idx = st.session_state.current_review_index
            review_file = review_files[current_idx]
            review_data = json.loads(review_file.read_text())
            
            st.markdown(f"**Review Item {current_idx + 1} of {len(review_files)}**")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Original Handwriting")
                st.info(f"**Document:** {review_data['document']}")
                st.info(f"**Image ID:** {review_data['image_id']}")
                st.info(f"**Confidence:** {review_data.get('confidence', 0.5):.2f}")
                
                st.markdown("*[Original handwriting image would appear here]*")
            
            with col2:
                st.markdown("### Translation / Transcription")
                
                analysis_file = ANALYSIS_DIR / f"{review_data['workspace']}_{review_data['document']}.json"
                
                matching_ann = None
                if analysis_file.exists():
                    analysis = json.loads(analysis_file.read_text())
                    handwriting = analysis.get('handwriting_translation', {})
                    
                    for ann in handwriting.get('translated_annotations', []):
                        if ann['image_id'] == review_data['image_id']:
                            matching_ann = ann
                            break
                    
                    if matching_ann:
                        st.markdown(f"**Auto-translated:** {matching_ann['translated_text']}")
                        st.markdown(f"**Confidence Tier:** {matching_ann['confidence_tier']}")
                    else:
                        st.warning("Translation not found")
                else:
                    st.warning("Analysis file not found")
            
            st.markdown("---")
            st.markdown("### Review Actions")
            
            form_key = f"review_form_{review_data['document']}_{review_data['image_id']}_{current_idx}"
            
            with st.form(key=form_key):
                transcription = st.text_area(
                    "Manual Transcription (edit if needed):",
                    value=matching_ann.get('translated_text', '') if matching_ann else '',
                    key=f"transcribe_{form_key}",
                    height=100
                )
                
                notes = st.text_area(
                    "Reviewer Notes:",
                    key=f"notes_{form_key}",
                    height=80
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    approve_btn = st.form_submit_button("✅ Approve", use_container_width=True)
                with col2:
                    modify_btn = st.form_submit_button("✏️ Modify & Approve", use_container_width=True)
                with col3:
                    reject_btn = st.form_submit_button("❌ Mark Unreadable", use_container_width=True)
                
                if approve_btn or modify_btn or reject_btn:
                    review_data['status'] = 'completed'
                    review_data['reviewed_at'] = datetime.now().isoformat()
                    review_data['final_transcription'] = transcription
                    review_data['reviewer_notes'] = notes
                    
                    if reject_btn:
                        review_data['readable'] = False
                    else:
                        review_data['readable'] = True
                    
                    with open(review_file, 'w') as f:
                        json.dump(review_data, f, indent=2)
                    
                    log_audit_event("handwriting_reviewed", {
                        "document": review_data['document'],
                        "image_id": review_data['image_id'],
                        "status": review_data['status']
                    })
                    
                    st.success("✅ Review submitted!")
                    st.info("Click 'Next Item' to continue reviewing")
            
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("⬅️ Previous", disabled=(current_idx == 0)):
                    st.session_state.current_review_index -= 1
                    st.rerun()
            
            with col2:
                st.markdown(f"<center>Item {current_idx + 1} / {len(review_files)}</center>", unsafe_allow_html=True)
            
            with col3:
                if st.button("Next ➡️", disabled=(current_idx >= len(review_files) - 1)):
                    st.session_state.current_review_index += 1
                    st.rerun()
            
            st.markdown("---")
            if st.button("📥 Export All Reviews"):
                export_data = []
                for rf in review_files:
                    export_data.append(json.loads(rf.read_text()))
                
                export_json = json.dumps(export_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=export_json,
                    file_name=f"handwriting_reviews_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
