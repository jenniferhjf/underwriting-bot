"""
Enhanced Underwriting Assistant - Professional RAG+CoT System
Version 2.3 - Claude API Integration (English Only)

Features:
- Claude 3.5 Sonnet API
- Filename-based automatic tagging
- Handwriting classification: Conclusion/Todo/Risk/Communication
- Initial dataset: Hull - Marco Polo_Memo.pdf
"""

import streamlit as st
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import requests
import PyPDF2
from docx import Document as DocxDocument
import pandas as pd
from PIL import Image
import io
import zipfile
import tempfile
import re
from collections import defaultdict
import traceback

# ============================================================================
# CONFIGURATION - CLAUDE API
# ============================================================================

CLAUDE_API_KEY = "sk-yNCM9ClUBDJFsgZ0rktNPkGBTymQWL2rtyag5Rov3buMxRHt"
CLAUDE_API_BASE = "https://api.anthropic.com/v1"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

DATA_DIR = "data"
WORKSPACES_DIR = os.path.join(DATA_DIR, "workspaces")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
REVIEW_DIR = os.path.join(DATA_DIR, "review_queue")
AUDIT_DIR = os.path.join(DATA_DIR, "audit_logs")
INITIAL_DATASET_DIR = "."  # Root directory for initial dataset

for dir_path in [WORKSPACES_DIR, EMBEDDINGS_DIR, ANALYSIS_DIR, REVIEW_DIR, AUDIT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

SUPPORTED_FORMATS = {
    "pdf": "PDF", "docx": "Word", "doc": "Word",
    "txt": "Text", "xlsx": "Excel", "xls": "Excel",
    "png": "Image", "jpg": "Image", "jpeg": "Image"
}

TAG_OPTIONS = {
    "equipment": ["Gas Turbine", "Steam Turbine", "Boiler", "Generator", "Compressor", 
                  "Heat Exchanger", "Pump", "Transformer", "Motor", "Vessel", "Hull", "Other"],
    "industry": ["Oil & Gas", "Power Generation", "Manufacturing", "Chemical", 
                 "Mining", "Refining", "Marine", "Aviation", "Cargo", "Property", "Liability", "Other"],
    "timeline": ["2026-Q1", "2025-Q4", "2025-Q3", "2025-Q2", "2025-Q1", "2024", "2023", "Earlier"]
}

HANDWRITING_CATEGORIES = {
    "conclusion": "Conclusion",
    "todo": "To-Do",
    "risk": "Risk Factor",
    "communication": "External Communication"
}

INSURANCE_TERMS = {
    "retention": "deductible/self-insured retention",
    "premium": "insurance premium",
    "coverage": "insurance coverage",
    "deductible": "amount paid before insurance kicks in",
    "underwriting slip": "underwriting document",
    "loss ratio": "claims to premium ratio",
    "exposure": "risk exposure",
    "claims": "insurance claims",
    "policy": "insurance policy",
    "endorsement": "policy amendment",
    "exclusion": "coverage exclusion",
    "limit": "coverage limit",
    "aggregate": "total coverage limit",
    "per occurrence": "per incident limit",
    "retroactive date": "coverage start date"
}

# ============================================================================
# CLAUDE API INTEGRATION
# ============================================================================

def call_claude_api(system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """
    Call Claude API with system and user prompts
    
    Args:
        system_prompt: System instruction for Claude
        user_prompt: User's actual query/request
        temperature: Randomness (0.0-1.0)
        max_tokens: Maximum response length
    
    Returns:
        Claude's response text
    """
    headers = {
        "Content-Type": "application/json",
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    }
    
    try:
        resp = requests.post(
            f"{CLAUDE_API_BASE}/messages",
            headers=headers,
            json=payload,
            timeout=90
        )
        resp.raise_for_status()
        response_data = resp.json()
        
        if "content" in response_data and len(response_data["content"]) > 0:
            return response_data["content"][0]["text"]
        else:
            return "Error: No response from Claude"
            
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

SYSTEM_INSTRUCTION = """Role: You are an AI underwriting assistant with deep insurance domain knowledge

Task: Answer underwriting queries using retrieved cases with insurance-specific understanding

Process: Think step-by-step using this framework:

Step 1: Extract key insurance terms and tags from query
Step 2: Analyze retrieved precedents (prioritize recent cases)
Step 3: Check recency, applicability & similar risk profiles
Step 4: Identify decision patterns and retention strategies
Step 5: Recommend with rationale citing specific clauses

Output: Provide decision + premium range + retention terms + sources with page references"""

ELECTRONIC_TEXT_ANALYSIS_SYSTEM = """You are analyzing ELECTRONIC PRINTED TEXT from insurance documents.

Focus on:
1. Policy schedules and coverage tables
2. Premium calculations and retention amounts
3. Loss statistics and claim history
4. Coverage terms and conditions
5. Insured values and limits
6. Effective dates and policy periods
7. Named insureds and beneficiaries
8. Special clauses and endorsements

Extract structured data:
- Policy numbers and reference IDs
- Financial figures (premiums, limits, deductibles)
- Dates (inception, expiry, retroactive)
- Coverage scope and exclusions
- Loss ratios and experience modifications

Output: JSON with:
{
  "policy_info": {
    "policy_number": "...",
    "insured": "...",
    "period": "...",
    "coverage_type": "..."
  },
  "financial_terms": {
    "premium": 0,
    "retention": 0,
    "limit": 0,
    "currency": "USD"
  },
  "coverage_terms": [...],
  "loss_statistics": {
    "tables": [...],
    "loss_ratio": 0
  },
  "special_clauses": [...]
}"""

HANDWRITTEN_CLASSIFICATION_SYSTEM = """You are analyzing and CLASSIFYING HANDWRITTEN ANNOTATIONS from insurance underwriting documents.

Your task: Extract handwritten text and classify into 4 categories:

1. **Conclusion** - Final decisions, approvals, authorizations
   Keywords: "approved", "declined", "accepted", "agreed", "confirmed", "OK", checkmarks, signatures
   Examples: "Approved subject to...", "Agree with terms", "OK to proceed"

2. **To-Do** - Action items, follow-ups, pending tasks
   Keywords: "need to", "follow up", "check", "verify", "confirm", "pending", "ask"
   Examples: "Need to verify loss history", "Follow up with broker", "Check references"

3. **Risk Factor** - Identified risks, concerns, warnings
   Keywords: "risk", "concern", "warning", "high", "caution", "watch", "monitor", "flag"
   Examples: "High risk - monitor closely", "Concern about claims history", "Watch exposure"

4. **External Communication** - Messages for clients/brokers, negotiation points
   Keywords: "tell broker", "inform client", "communicate", "proposal", "offer", "counter"
   Examples: "Tell broker we can offer...", "Communicate terms to client", "Counter with..."

For each handwritten annotation, provide:
{
  "annotations": [
    {
      "text": "OCR recognized handwritten text",
      "category": "conclusion|todo|risk|communication",
      "confidence": 0.0-1.0,
      "location": "Page X, margin/top/bottom",
      "context": "related printed content context",
      "urgency": "high|medium|low",
      "assignee": "identified responsible person (if any)"
    }
  ],
  "summary_by_category": {
    "conclusion": ["..."],
    "todo": ["..."],
    "risk": ["..."],
    "communication": ["..."]
  },
  "needs_human_review": [
    {"reason": "low confidence", "image_id": "..."}
  ]
}

Important:
- Try to extract text even if confidence is low (mark for review)
- Use context from surrounding printed text to infer meaning
- Flag unclear items for human review
- Note any signatures, initials, or dates"""

QA_EXTRACTION_SYSTEM = """You are extracting Question-Answer pairs from insurance documents.

Recognize patterns:
1. Formal Q&A format:
   - "Q1: ..." -> "A1: ..."
   - "Question: ..." -> "Answer: ..."
2. Email correspondence:
   - "From: [Broker] Q: ..." -> "From: [Underwriter] Re: ..."
3. Margin annotations:
   - Question mark next to clause -> Handwritten answer
4. Meeting notes:
   - "CE asked: ..." -> "Response: ..."

For each Q&A pair extract:
{
  "question_id": "Q1/Email-001",
  "question_text": "...",
  "answer_text": "...",
  "source_type": "formal/email/annotation/verbal",
  "page_number": 0,
  "asked_by": "...",
  "answered_by": "...",
  "date": "...",
  "status": "certified/pending/uncertified",
  "confidence": 0.0-1.0
}

Return array of Q&A objects."""

AUTO_ANNOTATE_SYSTEM = """You are an insurance underwriting document auto-tagger with domain expertise.

Analyze the document and extract:
1. Insurance type (Cargo/Property/Liability/Equipment Breakdown/Marine Hull/etc.)
2. Equipment/Asset types covered
3. Industry sector
4. Key financial terms
5. Decision status
6. Risk indicators

Return STRICT JSON:
{
  "tags": {
    "equipment": ["..."],
    "industry": ["..."],
    "timeline": ["YYYY-QX"]
  },
  "insurance_type": "Cargo|Property|Liability|Equipment|Marine|Hull|Other",
  "decision": "Approved|Declined|Conditional|Pending",
  "premium": 0,
  "retention": 0,
  "limit": 0,
  "currency": "USD",
  "risk_level": "Low|Medium|Medium-High|High|Critical",
  "case_summary": "Brief summary focusing on coverage, insured, and key terms",
  "key_insights": "Notable risk factors, loss history, special conditions",
  "extracted_clauses": [],
  "confidence_score": 0.0-1.0
}

Rules:
- Focus on insurance-specific terminology
- Extract policy numbers and reference IDs
- Identify retention/deductible amounts
- Note any special endorsements
- Flag high-risk indicators
"""

# ============================================================================
# AUDIT LOGGING
# ============================================================================

def log_audit_event(workspace_name: str, event_type: str, details: Dict[str, Any], user: str = "system"):
    """Log audit events for compliance and tracking"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "workspace": workspace_name,
            "event_type": event_type,
            "user": user,
            "details": details
        }
        
        log_file = os.path.join(AUDIT_DIR, f"{workspace_name}_audit.jsonl")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Audit logging failed: {e}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def preprocess_insurance_text(text: str) -> str:
    """Preprocess text with insurance domain knowledge"""
    processed = text
    for term, definition in INSURANCE_TERMS.items():
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        processed = pattern.sub(f"{term}({definition})", processed)
    
    processed = re.sub(r'\b(USD|US\$)\s*(\d+)', r'USD \2', processed)
    processed = re.sub(r'\b(GBP|£)\s*(\d+)', r'GBP \2', processed)
    
    return processed

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF with better error handling"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_parts = []
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"\n--- PAGE {i+1} ---\n{page_text}")
            return "".join(text_parts)
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX preserving structure"""
    try:
        doc = DocxDocument(file_path)
        parts = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text)
        
        for i, table in enumerate(doc.tables):
            parts.append(f"\n--- TABLE {i+1} ---")
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                parts.append(" | ".join(row_data))
        
        return "\n".join(parts)
    except Exception as e:
        return f"Error extracting DOCX: {str(e)}"

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT with encoding detection"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return "Error: Could not decode text file"

def extract_text_from_file(file_path: str, file_format: str) -> str:
    """Unified text extraction with preprocessing"""
    extractors = {
        "pdf": extract_text_from_pdf,
        "docx": extract_text_from_docx,
        "doc": extract_text_from_docx,
        "txt": extract_text_from_txt
    }
    raw_text = extractors.get(file_format, lambda x: "")(file_path)
    return preprocess_insurance_text(raw_text)

def generate_embedding(text: str) -> List[float]:
    """Generate embedding with insurance domain context"""
    processed_text = preprocess_insurance_text(text)
    text_hash = hashlib.md5(processed_text.encode()).hexdigest()
    fake = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
    return (fake + [0.0] * 1536)[:1536]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between vectors"""
    dot = sum(a*b for a,b in zip(v1, v2))
    m1, m2 = sum(a*a for a in v1) ** 0.5, sum(b*b for b in v2) ** 0.5
    return dot / (m1 * m2) if m1 and m2 else 0.0

# ============================================================================
# FILENAME-BASED TAGGING
# ============================================================================

def extract_tag_from_filename(filename: str) -> str:
    """
    Extract first word from filename as main tag
    
    Examples:
    - "Vessel_Insurance_2024.pdf" -> "Vessel"
    - "Hull - Marco Polo_Memo.pdf" -> "Hull"
    - "Turbine Maintenance Report.docx" -> "Turbine"
    - "2024-Marine-Policy.pdf" -> "Marine" (skip year)
    """
    name_without_ext = os.path.splitext(filename)[0]
    tokens = re.split(r'[-_\s]+', name_without_ext)
    
    valid_tokens = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            continue
        if re.match(r'^\d{4}[Qq]\d$', token):
            continue
        valid_tokens.append(token)
    
    if valid_tokens:
        return valid_tokens[0].capitalize()
    else:
        return "Document"

def auto_generate_tags(filename: str, text_preview: str = "") -> Dict[str, List[str]]:
    """Generate tags based on filename and text content"""
    tags = {
        "equipment": [],
        "industry": [],
        "timeline": []
    }
    
    main_tag = extract_tag_from_filename(filename)
    tags["equipment"].append(main_tag)
    
    text_lower = text_preview.lower()
    for industry in TAG_OPTIONS["industry"]:
        if industry.lower() in text_lower:
            if industry not in tags["industry"]:
                tags["industry"].append(industry)
    
    if not tags["industry"]:
        tags["industry"].append("Other")
    
    current_year = 2026
    for year in range(current_year, current_year - 5, -1):
        if str(year) in filename or str(year) in text_preview:
            quarter_match = re.search(rf'{year}[-_\s]*[Qq]?([1-4])', filename + text_preview)
            if quarter_match:
                tags["timeline"].append(f"{year}-Q{quarter_match.group(1)}")
            else:
                tags["timeline"].append(str(year))
            break
    
    if not tags["timeline"]:
        tags["timeline"].append("Earlier")
    
    return tags

# ============================================================================
# IMAGE & HANDWRITING PROCESSING
# ============================================================================

def extract_images_from_docx(docx_path: str, output_dir: str) -> List[Dict[str, str]]:
    """Extract all images from DOCX file with metadata"""
    images_info = []
    try:
        os.makedirs(output_dir, exist_ok=True)
        with zipfile.ZipFile(docx_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.startswith('word/media/'):
                    filename = os.path.basename(file_info.filename)
                    if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
                        extracted_path = zip_ref.extract(file_info, output_dir)
                        
                        try:
                            img = Image.open(extracted_path)
                            width, height = img.size
                            has_handwriting = width >= 300 and height >= 300
                        except:
                            width, height = 0, 0
                            has_handwriting = False
                        
                        images_info.append({
                            "image_id": filename.split('.')[0],
                            "filename": filename,
                            "path": extracted_path,
                            "format": filename.split('.')[-1].lower(),
                            "page_number": len(images_info) + 1,
                            "likely_handwriting": has_handwriting,
                            "width": width,
                            "height": height
                        })
        return images_info
    except Exception as e:
        st.error(f"Error extracting images: {e}")
        return []

def classify_handwriting_quality(image_info: Dict) -> Tuple[str, float]:
    """Classify handwriting quality into tiers"""
    width = image_info.get("width", 0)
    height = image_info.get("height", 0)
    
    if width * height > 1000000:
        return "CLEAR", 0.75
    elif width * height > 400000:
        return "STANDARD", 0.55
    else:
        return "CURSIVE", 0.30

# ============================================================================
# ANALYSIS FUNCTIONS (Using Claude)
# ============================================================================

def extract_qa_pairs(text: str, filename: str) -> List[Dict[str, Any]]:
    """Extract Question-Answer pairs using Claude"""
    
    user_prompt = f"""Document: {filename}

Text content:
{text[:5000]}

Extract all Question-Answer pairs following the guidelines.
Return ONLY a valid JSON array of Q&A objects."""
    
    response = call_claude_api(
        system_prompt=QA_EXTRACTION_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=2000
    )
    
    try:
        cleaned = response.strip().strip("`")
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        qa_pairs = json.loads(cleaned.strip())
        
        if isinstance(qa_pairs, list):
            return qa_pairs
        elif isinstance(qa_pairs, dict) and "qa_pairs" in qa_pairs:
            return qa_pairs["qa_pairs"]
        else:
            return []
    except:
        return []

def analyze_electronic_text(extracted_text: str, filename: str) -> Dict[str, Any]:
    """Analyze electronic/printed text using Claude"""
    
    user_prompt = f"""Document: {filename}

Electronic Printed Text Content:
{extracted_text[:5000]}

Analyze the ELECTRONIC/PRINTED content following the insurance document framework.
Focus on policy terms, financial figures, coverage details, and formal terms.

Return ONLY valid JSON."""
    
    response = call_claude_api(
        system_prompt=ELECTRONIC_TEXT_ANALYSIS_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=2000
    )
    
    try:
        cleaned = response.strip().strip("`")
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        result = json.loads(cleaned.strip())
        result["content_type"] = "electronic_text"
        result["analysis_timestamp"] = datetime.now().isoformat()
        return result
    except Exception as e:
        return {
            "raw_response": response,
            "error": str(e),
            "note": "Could not parse as JSON",
            "content_type": "electronic_text"
        }

def analyze_and_classify_handwriting(image_info: List[Dict], extracted_text: str, doc_id: str) -> Dict[str, Any]:
    """Analyze and classify handwritten annotations using Claude"""
    
    handwriting_items = []
    needs_review = []
    
    for img in image_info:
        if img.get("likely_handwriting", False):
            tier, confidence = classify_handwriting_quality(img)
            
            item = {
                "image_id": img["image_id"],
                "page": img.get("page_number", 0),
                "tier": tier,
                "estimated_confidence": confidence,
                "status": "auto_processed" if tier == "CLEAR" else "needs_review"
            }
            
            handwriting_items.append(item)
            
            if tier in ["STANDARD", "CURSIVE"] or confidence < 0.6:
                needs_review.append({
                    "doc_id": doc_id,
                    "image_id": img["image_id"],
                    "image_path": img["path"],
                    "page": img.get("page_number", 0),
                    "tier": tier,
                    "confidence": confidence,
                    "reason": f"Confidence below threshold: {confidence:.2%}",
                    "review_status": "pending",
                    "created_at": datetime.now().isoformat()
                })
    
    if needs_review:
        review_file = os.path.join(REVIEW_DIR, f"{doc_id}_review_queue.json")
        with open(review_file, 'w', encoding='utf-8') as f:
            json.dump(needs_review, f, ensure_ascii=False, indent=2)
    
    user_prompt = f"""Based on document analysis with handwriting regions detected:

CONTEXT FROM PRINTED TEXT:
{extracted_text[:2500]}

HANDWRITING REGIONS DETECTED: {len(handwriting_items)}

YOUR TASK: Extract and CLASSIFY all handwritten annotations into these categories:
1. Conclusion - Final decisions, approvals
2. To-Do - Action items, follow-ups
3. Risk Factor - Identified risks, concerns
4. External Communication - Messages for clients/brokers

For each annotation:
- Extract the handwritten text (even if low confidence)
- Classify into one of the 4 categories
- Provide location and context
- Note confidence level
- Flag if needs human review

Return the JSON structure as specified."""
    
    response = call_claude_api(
        system_prompt=HANDWRITTEN_CLASSIFICATION_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=2500
    )
    
    try:
        cleaned = response.strip().strip("`")
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        result = json.loads(cleaned.strip())
    except:
        result = {
            "raw_response": response,
            "note": "Could not parse as JSON",
            "annotations": [],
            "summary_by_category": {
                "conclusion": [],
                "todo": [],
                "risk": [],
                "communication": []
            }
        }
    
    result["handwriting_items"] = handwriting_items
    result["review_queue_count"] = len(needs_review)
    result["content_type"] = "handwritten_annotations"
    result["analysis_timestamp"] = datetime.now().isoformat()
    
    return result

def perform_dual_track_analysis(file_path: str, file_format: str, filename: str, doc_id: str, workspace_name: str) -> Dict[str, Any]:
    """Perform enhanced DUAL-TRACK analysis using Claude"""
    
    analysis_result = {
        "doc_id": doc_id,
        "filename": filename,
        "analysis_date": datetime.now().isoformat(),
        "file_format": file_format,
        "dual_track_analysis": {
            "electronic_text": {},
            "handwritten_annotations": {},
            "qa_pairs": [],
            "integration_summary": ""
        },
        "images_by_page": [],
        "metadata": {},
        "review_queue": []
    }
    
    try:
        log_audit_event(workspace_name, "analysis_started", {
            "doc_id": doc_id,
            "filename": filename,
            "file_format": file_format
        })
        
        st.info("Track 1: Analyzing Electronic/Printed Text...")
        extracted_text = extract_text_from_file(file_path, file_format)
        analysis_result["metadata"]["text_length"] = len(extracted_text)
        analysis_result["metadata"]["has_text"] = len(extracted_text) > 0
        
        if extracted_text:
            electronic_analysis = analyze_electronic_text(extracted_text, filename)
            analysis_result["dual_track_analysis"]["electronic_text"] = electronic_analysis
            
            st.info("Extracting Question-Answer pairs...")
            qa_pairs = extract_qa_pairs(extracted_text, filename)
            analysis_result["dual_track_analysis"]["qa_pairs"] = qa_pairs
            analysis_result["metadata"]["qa_pairs_count"] = len(qa_pairs)
        
        st.info("Track 2: Processing & Classifying Handwritten Content...")
        images_info = []
        temp_image_dir = tempfile.mkdtemp(prefix=f"doc_analysis_{doc_id}_")
        
        if file_format == "docx":
            images_info = extract_images_from_docx(file_path, temp_image_dir)
        
        analysis_result["metadata"]["image_count"] = len(images_info)
        analysis_result["metadata"]["has_images"] = len(images_info) > 0
        
        for img in images_info:
            tier, confidence = classify_handwriting_quality(img)
            analysis_result["images_by_page"].append({
                "page": img.get("page_number", 0),
                "image_id": img["image_id"],
                "filename": img["filename"],
                "format": img["format"],
                "likely_handwriting": img.get("likely_handwriting", False),
                "quality_tier": tier,
                "confidence_estimate": confidence
            })
        
        if images_info:
            handwriting_analysis = analyze_and_classify_handwriting(images_info, extracted_text, doc_id)
            analysis_result["dual_track_analysis"]["handwritten_annotations"] = handwriting_analysis
            analysis_result["metadata"]["review_queue_count"] = handwriting_analysis.get("review_queue_count", 0)
            
            annotations = handwriting_analysis.get("annotations", [])
            category_count = {
                "conclusion": 0,
                "todo": 0,
                "risk": 0,
                "communication": 0
            }
            for ann in annotations:
                cat = ann.get("category", "")
                if cat in category_count:
                    category_count[cat] += 1
            analysis_result["metadata"]["annotation_categories"] = category_count
            
            review_file = os.path.join(REVIEW_DIR, f"{doc_id}_review_queue.json")
            if os.path.exists(review_file):
                with open(review_file, 'r', encoding='utf-8') as f:
                    analysis_result["review_queue"] = json.load(f)
        
        st.info("Integrating Analysis Results...")
        
        handwriting_summary = analysis_result["dual_track_analysis"]["handwritten_annotations"].get("summary_by_category", {})
        
        integration_prompt = f"""Integrate these analysis tracks into a comprehensive underwriting report:

ELECTRONIC TEXT ANALYSIS:
{json.dumps(analysis_result["dual_track_analysis"]["electronic_text"], indent=2, ensure_ascii=False)[:2000]}

HANDWRITTEN ANNOTATIONS (CLASSIFIED):
Conclusions: {len(handwriting_summary.get('conclusion', []))} items
To-Do: {len(handwriting_summary.get('todo', []))} items
Risks: {len(handwriting_summary.get('risk', []))} items
Communications: {len(handwriting_summary.get('communication', []))} items

Details:
{json.dumps(handwriting_summary, indent=2, ensure_ascii=False)[:1500]}

Q&A PAIRS: {len(analysis_result["dual_track_analysis"]["qa_pairs"])}

Create a structured report with:
1. Executive Summary - Key decisions and conclusions
2. Action Items - All to-do items with urgency
3. Risk Assessment - Identified risk factors
4. External Communication Plan - Client/broker communication points
5. Policy Details - From electronic text
6. Q&A Summary - Key questions and answers
7. Recommendations - Next steps for underwriter

Format as clear markdown with sections."""
        
        integration_summary = call_claude_api(
            system_prompt="You are a senior insurance underwriting analyst creating structured reports.",
            user_prompt=integration_prompt,
            temperature=0.4,
            max_tokens=3000
        )
        
        analysis_result["dual_track_analysis"]["integration_summary"] = integration_summary
        
        analysis_file = os.path.join(ANALYSIS_DIR, f"{doc_id}_dual_analysis.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        log_audit_event(workspace_name, "analysis_completed", {
            "doc_id": doc_id,
            "text_length": analysis_result["metadata"]["text_length"],
            "image_count": analysis_result["metadata"]["image_count"],
            "qa_pairs": len(analysis_result["dual_track_analysis"]["qa_pairs"]),
            "review_queue": analysis_result["metadata"].get("review_queue_count", 0),
            "annotation_categories": analysis_result["metadata"].get("annotation_categories", {})
        })
        
        st.success("Dual-track analysis completed!")
        
        if analysis_result["metadata"].get("review_queue_count", 0) > 0:
            st.warning(f"{analysis_result['metadata']['review_queue_count']} handwritten items need human review.")
        
        return analysis_result
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.error(traceback.format_exc())
        log_audit_event(workspace_name, "analysis_failed", {
            "doc_id": doc_id,
            "error": str(e)
        })
        analysis_result["error"] = str(e)
        return analysis_result

def auto_annotate_by_llm(extracted_text: str, filename: str) -> Dict[str, Any]:
    """Auto-annotate document using Claude"""
    user_prompt = f"FILENAME: {filename}\n\nTEXT:\n{(extracted_text or '')[:5000]}"
    
    content = call_claude_api(
        system_prompt=AUTO_ANNOTATE_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=1000
    )
    
    try:
        cleaned = content.strip().strip("`")
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        data = json.loads(cleaned)
    except:
        data = {}
    
    defaults = {
        "tags": {"equipment": ["Other"], "industry": ["Other"], "timeline": ["Earlier"]},
        "insurance_type": "Other",
        "decision": "Pending",
        "premium": 0,
        "retention": 0,
        "limit": 0,
        "currency": "USD",
        "risk_level": "Medium",
        "case_summary": "Auto-tagging in progress. Please review.",
        "key_insights": "Awaiting detailed analysis.",
        "extracted_clauses": [],
        "confidence_score": 0.5
    }
    
    for key, default_value in defaults.items():
        data.setdefault(key, default_value)
    
    return data

# ============================================================================
# WORKSPACE CLASS
# ============================================================================

class Workspace:
    def __init__(self, name: str):
        self.name = name
        self.workspace_dir = os.path.join(WORKSPACES_DIR, name)
        self.documents_dir = os.path.join(self.workspace_dir, "documents")
        self.metadata_file = os.path.join(self.workspace_dir, "metadata.json")
        self.embeddings_file = os.path.join(self.workspace_dir, "embeddings.json")
        os.makedirs(self.documents_dir, exist_ok=True)
        self.metadata = self._load_metadata()
        self.embeddings = self._load_embeddings()
    
    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_metadata(self):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def _load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_embeddings(self):
        with open(self.embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(self.embeddings, f, indent=2)
    
    def add_document(self, uploaded_file, tags: Dict[str, List[str]], 
                     case_summary: str, key_insights: str, decision: str, 
                     premium: int, risk_level: str, extracted_text_preview: str = "",
                     has_deep_analysis: bool = False, insurance_type: str = "Other",
                     retention: int = 0, limit: int = 0) -> Dict[str, Any]:
        """Add document with enhanced metadata"""
        doc_id = f"DOC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(uploaded_file.name.encode()).hexdigest()[:6].upper()}"
        ext = uploaded_file.name.split('.')[-1].lower()
        filename = f"{doc_id}.{ext}"
        file_path = os.path.join(self.documents_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        full_text = f"{insurance_type} {case_summary} {key_insights} {extracted_text_preview[:1000]}"
        embedding = generate_embedding(full_text)
        
        doc_meta = {
            "doc_id": doc_id,
            "filename": uploaded_file.name,
            "file_format": ext,
            "file_path": file_path,
            "file_size_kb": uploaded_file.size/1024,
            "upload_date": datetime.now().isoformat(),
            "tags": tags,
            "insurance_type": insurance_type,
            "decision": decision,
            "premium": premium,
            "retention": retention,
            "limit": limit,
            "risk_level": risk_level,
            "case_summary": case_summary,
            "key_insights": key_insights,
            "extracted_text_preview": extracted_text_preview[:500],
            "has_deep_analysis": has_deep_analysis,
            "review_status": "not_started"
        }
        
        self.metadata.append(doc_meta)
        self.embeddings[doc_id] = embedding
        self._save_metadata()
        self._save_embeddings()
        
        log_audit_event(self.name, "document_added", {
            "doc_id": doc_id,
            "filename": uploaded_file.name,
            "insurance_type": insurance_type,
            "main_tag": tags.get("equipment", ["Other"])[0]
        })
        
        return doc_meta
    
    def search_documents(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None):
        """Enhanced search with insurance-aware ranking"""
        if not self.metadata:
            return []
        
        qv = generate_embedding(query)
        scored = []
        
        for doc in self.metadata:
            if filters:
                if filters.get("insurance_type") and doc.get("insurance_type") != filters["insurance_type"]:
                    continue
                if filters.get("risk_level") and doc.get("risk_level") != filters["risk_level"]:
                    continue
                if filters.get("decision") and doc.get("decision") != filters["decision"]:
                    continue
            
            doc_id = doc["doc_id"]
            if doc_id in self.embeddings:
                sim = cosine_similarity(qv, self.embeddings[doc_id])
                
                ql = query.lower()
                for term in INSURANCE_TERMS.keys():
                    if term in ql and term in doc.get("case_summary", "").lower():
                        sim += 0.15
                
                for tag_list in doc["tags"].values():
                    for tag in tag_list:
                        if tag.lower() in ql:
                            sim += 0.1
                
                if doc.get("has_deep_analysis"):
                    sim += 0.05
                
                upload_date = doc.get("upload_date", "")
                if upload_date and ("2026" in upload_date or "2025" in upload_date):
                    sim += 0.08
                
                scored.append((sim, doc))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [d for _, d in scored[:top_k]]
    
    def delete_document(self, doc_id: str):
        """Delete document with audit logging"""
        doc = next((d for d in self.metadata if d["doc_id"] == doc_id), None)
        
        self.metadata = [d for d in self.metadata if d["doc_id"] != doc_id]
        if doc_id in self.embeddings:
            del self.embeddings[doc_id]
        
        for fn in os.listdir(self.documents_dir):
            if fn.startswith(doc_id):
                try:
                    os.remove(os.path.join(self.documents_dir, fn))
                except:
                    pass
        
        analysis_file = os.path.join(ANALYSIS_DIR, f"{doc_id}_dual_analysis.json")
        if os.path.exists(analysis_file):
            try:
                os.remove(analysis_file)
            except:
                pass
        
        review_file = os.path.join(REVIEW_DIR, f"{doc_id}_review_queue.json")
        if os.path.exists(review_file):
            try:
                os.remove(review_file)
            except:
                pass
        
        self._save_metadata()
        self._save_embeddings()
        
        log_audit_event(self.name, "document_deleted", {
            "doc_id": doc_id,
            "filename": doc.get("filename") if doc else "unknown"
        })
    
    def get_stats(self):
        """Get workspace statistics"""
        return {
            "total_documents": len(self.metadata),
            "total_size_mb": sum(d["file_size_kb"] for d in self.metadata)/1024 if self.metadata else 0.0,
            "format_distribution": self._get_fmt_dist(),
            "decision_distribution": self._get_decision_dist(),
            "insurance_type_distribution": self._get_insurance_type_dist(),
            "analyzed_documents": sum(1 for d in self.metadata if d.get("has_deep_analysis", False)),
            "pending_reviews": self._count_pending_reviews()
        }
    
    def _get_fmt_dist(self):
        dist = {}
        for d in self.metadata:
            dist[d["file_format"]] = dist.get(d["file_format"], 0) + 1
        return dist
    
    def _get_decision_dist(self):
        dist = {}
        for d in self.metadata:
            dist[d["decision"]] = dist.get(d["decision"], 0) + 1
        return dist
    
    def _get_insurance_type_dist(self):
        dist = {}
        for d in self.metadata:
            itype = d.get("insurance_type", "Other")
            dist[itype] = dist.get(itype, 0) + 1
        return dist
    
    def _count_pending_reviews(self):
        """Count documents with pending handwriting reviews"""
        count = 0
        for doc in self.metadata:
            review_file = os.path.join(REVIEW_DIR, f"{doc['doc_id']}_review_queue.json")
            if os.path.exists(review_file):
                try:
                    with open(review_file, 'r') as f:
                        queue = json.load(f)
                        count += len([item for item in queue if item.get("review_status") == "pending"])
                except:
                    pass
        return count

def get_all_workspaces() -> List[str]:
    if not os.path.exists(WORKSPACES_DIR):
        return []
    return [d for d in os.listdir(WORKSPACES_DIR) if os.path.isdir(os.path.join(WORKSPACES_DIR, d))]

def create_workspace(name: str) -> Workspace:
    ws = Workspace(name)
    log_audit_event(name, "workspace_created", {"name": name})
    return ws

# ============================================================================
# INITIAL DATASET LOADER
# ============================================================================

def load_initial_dataset(workspace: Workspace):
    """Load Hull - Marco Polo_Memo.pdf as initial dataset if workspace is empty"""
    if len(workspace.metadata) > 0:
        return None
    
    initial_file_path = os.path.join(INITIAL_DATASET_DIR, "Hull - Marco Polo_Memo.pdf")
    
    if not os.path.exists(initial_file_path):
        return None
    
    try:
        # Extract text
        extracted_text = extract_text_from_pdf(initial_file_path)
        
        # Generate tags from filename
        auto_tags = auto_generate_tags("Hull - Marco Polo_Memo.pdf", extracted_text[:1000])
        
        # Auto-annotate
        auto_annotation = auto_annotate_by_llm(extracted_text, "Hull - Marco Polo_Memo.pdf")
        
        # Merge tags
        auto_annotation["tags"]["equipment"] = auto_tags["equipment"] + [
            t for t in auto_annotation["tags"].get("equipment", []) 
            if t not in auto_tags["equipment"]
        ]
        auto_annotation["tags"]["industry"] = auto_tags["industry"]
        auto_annotation["tags"]["timeline"] = auto_tags["timeline"]
        
        # Create uploaded file object
        class InitialFileWrapper:
            def __init__(self, path, name):
                self.path = path
                self.name = name
                with open(path, 'rb') as f:
                    self.content = f.read()
                self.size = len(self.content)
            
            def getbuffer(self):
                return self.content
        
        uploaded_file = InitialFileWrapper(initial_file_path, "Hull - Marco Polo_Memo.pdf")
        
        # Add to workspace
        doc = workspace.add_document(
            uploaded_file=uploaded_file,
            tags=auto_annotation["tags"],
            case_summary=auto_annotation["case_summary"],
            key_insights=auto_annotation["key_insights"],
            decision=auto_annotation["decision"],
            premium=int(auto_annotation.get("premium", 0) or 0),
            risk_level=auto_annotation["risk_level"],
            extracted_text_preview=extracted_text[:1000],
            has_deep_analysis=False,
            insurance_type=auto_annotation.get("insurance_type", "Hull"),
            retention=int(auto_annotation.get("retention", 0) or 0),
            limit=int(auto_annotation.get("limit", 0) or 0)
        )
        
        return doc
        
    except Exception as e:
        st.error(f"Failed to load initial dataset: {e}")
        return None

# ============================================================================
# REVIEW WORKBENCH FUNCTIONS
# ============================================================================

def get_all_pending_reviews(workspace: Workspace) -> List[Dict[str, Any]]:
    """Get all pending handwriting reviews across workspace"""
    all_reviews = []
    
    for doc in workspace.metadata:
        review_file = os.path.join(REVIEW_DIR, f"{doc['doc_id']}_review_queue.json")
        if os.path.exists(review_file):
            try:
                with open(review_file, 'r', encoding='utf-8') as f:
                    queue = json.load(f)
                    for item in queue:
                        if item.get("review_status") == "pending":
                            item["doc_filename"] = doc["filename"]
                            all_reviews.append(item)
            except:
                pass
    
    return all_reviews

def submit_handwriting_review(doc_id: str, image_id: str, 
                              transcribed_text: str, category: str,
                              reviewer_notes: str,
                              is_accurate: bool, reviewer: str = "User"):
    """Submit handwriting review result with category"""
    review_file = os.path.join(REVIEW_DIR, f"{doc_id}_review_queue.json")
    
    if not os.path.exists(review_file):
        return False
    
    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            queue = json.load(f)
        
        updated = False
        for item in queue:
            if item["image_id"] == image_id:
                item["review_status"] = "completed"
                item["transcribed_text"] = transcribed_text
                item["category"] = category
                item["reviewer_notes"] = reviewer_notes
                item["is_accurate"] = is_accurate
                item["reviewed_by"] = reviewer
                item["reviewed_at"] = datetime.now().isoformat()
                updated = True
                break
        
        if updated:
            with open(review_file, 'w', encoding='utf-8') as f:
                json.dump(queue, f, ensure_ascii=False, indent=2)
            
            log_audit_event("global", "handwriting_reviewed", {
                "doc_id": doc_id,
                "image_id": image_id,
                "category": category,
                "reviewer": reviewer,
                "is_accurate": is_accurate
            })
        
        return updated
    except Exception as e:
        st.error(f"Error submitting review: {e}")
        return False

# ============================================================================
# CHAT FUNCTION
# ============================================================================

def generate_cot_response(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """Generate Chain-of-Thought response using Claude"""
    if not retrieved_docs:
        return """**No Relevant Cases Found**

Please add documents to this workspace or try refining your query with specific insurance terms (e.g., hull, cargo, retention, loss ratio)."""
    
    docs_text = ""
    for i, doc in enumerate(retrieved_docs, 1):
        equipment = ", ".join(doc["tags"].get("equipment", []))
        industry = ", ".join(doc["tags"].get("industry", []))
        timeline = ", ".join(doc["tags"].get("timeline", []))
        insurance_type = doc.get("insurance_type", "Other")
        
        docs_text += f"""
{'='*70}
CASE #{i}: {doc["doc_id"]}
{'='*70}
File: {doc["filename"]} ({doc["file_format"].upper()})
Type: {insurance_type}
Tags: {equipment} | {industry} | {timeline}

Decision: {doc["decision"]}
Premium: ${doc["premium"]:,}
Retention: ${doc.get("retention", 0):,}
Limit: ${doc.get("limit", 0):,}
Risk Level: {doc["risk_level"]}

Case Summary:
{doc["case_summary"]}

Key Insights:
{doc["key_insights"]}

{'[DUAL-TRACK ANALYZED]' if doc.get('has_deep_analysis') else ''}

"""
    
    user_prompt = f"""Query: "{query}"

Retrieved Cases:
{docs_text}

Please analyze using the 5-step insurance underwriting framework:
1. Extract key insurance terms and risk factors from query
2. Analyze retrieved precedents (focus on similar risks and recent cases)
3. Check recency & applicability (prioritize 2025-2026 cases)
4. Identify decision patterns and retention/premium strategies
5. Recommend with rationale citing specific clauses and page references

Provide:
- **Decision Recommendation** (Approved/Declined/Conditional)
- **Premium Range** (with calculation basis)
- **Retention Terms** (recommended deductible)
- **Special Conditions** (if any)
- **Source References** (case IDs and specific terms)
"""
    
    return call_claude_api(
        system_prompt=SYSTEM_INSTRUCTION,
        user_prompt=user_prompt,
        temperature=0.6,
        max_tokens=2500
    )

# ============================================================================
# UI STYLING (FIXED - English Only)
# ============================================================================

def inject_css(appearance: str):
    """Enhanced CSS with category badges"""
    if appearance == "Dark":
        css = """
        <style>
        :root {
            --text-primary: #e5e7eb; --text-secondary: #cbd5e1; --muted: #9ca3af;
            --bg-app: #0b1220; --card-bg: #101826; --shadow: 0 1px 3px rgba(0,0,0,0.5);
            --brand: #93c5fd; --green: #86efac; --amber: #fde68a; --purple: #c084fc;
            --red: #fca5a5; --blue: #60a5fa;
        }
        .stApp { background-color: var(--bg-app); color: var(--text-primary); }
        .main-header { font-size: 2rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.25rem; }
        .sub-header { font-size: 1rem; color: var(--muted); margin-bottom: 1.0rem; }
        .tag-badge { display:inline-block; padding:0.25rem 0.75rem; margin:0.25rem 0.4rem 0.25rem 0; border-radius:1rem; font-size:0.875rem; font-weight:700; color:#0b1220; }
        .tag-equipment { background-color: #93c5fd; }
        .tag-industry { background-color: #86efac; }
        .tag-timeline { background-color: #fde68a; }
        .tag-insurance { background-color: #c084fc; }
        
        .category-conclusion { background-color: #86efac; color: #0b1220; padding:0.5rem 1rem; border-radius:0.5rem; font-weight:700; margin:0.5rem; display:inline-block; }
        .category-todo { background-color: #fde68a; color: #0b1220; padding:0.5rem 1rem; border-radius:0.5rem; font-weight:700; margin:0.5rem; display:inline-block; }
        .category-risk { background-color: #fca5a5; color: #0b1220; padding:0.5rem 1rem; border-radius:0.5rem; font-weight:700; margin:0.5rem; display:inline-block; }
        .category-communication { background-color: #93c5fd; color: #0b1220; padding:0.5rem 1rem; border-radius:0.5rem; font-weight:700; margin:0.5rem; display:inline-block; }
        
        .analysis-badge { background-color: #c084fc; color:#0b1220; padding:0.25rem 0.75rem; border-radius:1rem; font-size:0.875rem; font-weight:700; }
        .review-pending { background-color: #fca5a5; color:#0b1220; padding:0.25rem 0.75rem; border-radius:1rem; font-size:0.875rem; font-weight:700; }
        .review-completed { background-color: #86efac; color:#0b1220; padding:0.25rem 0.75rem; border-radius:1rem; font-size:0.875rem; font-weight:700; }
        .confidence-clear { color: #86efac; font-weight: 700; }
        .confidence-standard { color: #fde68a; font-weight: 700; }
        .confidence-cursive { color: #fca5a5; font-weight: 700; }
        .review-card { background: var(--card-bg); padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid #fca5a5; margin: 1rem 0; }
        .stChatMessage, .stMarkdown, p, li, label, span, div { color: var(--text-primary); }
        [data-testid="stMetricDelta"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--text-primary) !important; }
        #MainMenu, footer, header {visibility: hidden;}
        </style>
        """
    else:
        css = """
        <style>
        :root {
            --text-primary: #0f172a; --text-secondary: #374151; --muted: #6b7280;
            --bg-app: #f5f7fa; --card-bg: #ffffff; --shadow: 0 1px 3px rgba(0,0,0,0.1);
            --brand: #1e40af; --green: #166534; --amber: #92400e; --purple: #7c3aed;
            --red: #dc2626; --blue: #2563eb;
        }
        .stApp { background-color: var(--bg-app); color: var(--text-primary); }
        .main-header { font-size: 2rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.25rem; }
        .sub-header { font-size: 1rem; color: var(--muted); margin-bottom: 1.0rem; }
        .tag-badge { display:inline-block; padding:0.25rem 0.75rem; margin:0.25rem 0.4rem 0.25rem 0; border-radius:1rem; font-size:0.875rem; font-weight:700; }
        .tag-equipment { background-color: #dbeafe; color: #0f172a; }
        .tag-industry { background-color: #dcfce7; color: #0f172a; }
        .tag-timeline { background-color: #fef3c7; color: #0f172a; }
        .tag-insurance { background-color: #e9d5ff; color: #0f172a; }
        
        .category-conclusion { background-color: #dcfce7; color: #166534; padding:0.5rem 1rem; border-radius:0.5rem; font-weight:700; margin:0.5rem; display:inline-block; border: 2px solid #86efac; }
        .category-todo { background-color: #fef3c7; color: #92400e; padding:0.5rem 1rem; border-radius:0.5rem; font-weight:700; margin:0.5rem; display:inline-block; border: 2px solid #fde68a; }
        .category-risk { background-color: #fee2e2; color: #dc2626; padding:0.5rem 1rem; border-radius:0.5rem; font-weight:700; margin:0.5rem; display:inline-block; border: 2px solid #fca5a5; }
        .category-communication { background-color: #dbeafe; color: #1e40af; padding:0.5rem 1rem; border-radius:0.5rem; font-weight:700; margin:0.5rem; display:inline-block; border: 2px solid #93c5fd; }
        
        .analysis-badge { background-color: #e9d5ff; color: #0f172a; padding:0.25rem 0.75rem; border-radius:1rem; font-size:0.875rem; font-weight:700; }
        .review-pending { background-color: #fee2e2; color: #dc2626; padding:0.25rem 0.75rem; border-radius:1rem; font-size:0.875rem; font-weight:700; }
        .review-completed { background-color: #dcfce7; color: #166534; padding:0.25rem 0.75rem; border-radius:1rem; font-size:0.875rem; font-weight:700; }
        .confidence-clear { color: #166534; font-weight: 700; }
        .confidence-standard { color: #92400e; font-weight: 700; }
        .confidence-cursive { color: #dc2626; font-weight: 700; }
        .review-card { background: var(--card-bg); padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid #dc2626; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .stChatMessage, .stMarkdown, p, li, label, span, div { color: var(--text-primary); }
        [data-testid="stMetricDelta"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--text-primary) !important; }
        #MainMenu, footer, header {visibility: hidden;}
        </style>
        """
    
    st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="Insurance Underwriting Assistant",
        page_icon="shield",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "upload_success" not in st.session_state:
        st.session_state.upload_success = False
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "current_review_index" not in st.session_state:
        st.session_state.current_review_index = 0
    if "review_submitted" not in st.session_state:
        st.session_state.review_submitted = False
    if "initial_dataset_loaded" not in st.session_state:
        st.session_state.initial_dataset_loaded = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Appearance")
        appearance = st.radio("Theme", ["Light", "Dark"], horizontal=True, key="appearance_choice")
    
    inject_css(appearance)
    
    # Header
    st.markdown('<div class="main-header">Insurance Underwriting Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Claude 3.5 | Filename-based Tagging | Handwriting Classification: Conclusion/Todo/Risk/Communication</div>', unsafe_allow_html=True)
    
    # Workspace Management
    with st.sidebar:
        st.markdown("### Workspaces")
        workspaces = get_all_workspaces()
        
        with st.expander("New Workspace"):
            new_ws_name = st.text_input("Workspace Name", placeholder="e.g., Marine Hull 2026")
            if st.button("Create", key="create_ws_btn"):
                if new_ws_name and new_ws_name not in workspaces:
                    create_workspace(new_ws_name)
                    st.success(f"Created: {new_ws_name}")
                    st.rerun()
                elif new_ws_name in workspaces:
                    st.error("Already exists")
                else:
                    st.error("Enter a name")
        
        if not workspaces:
            st.info("No workspaces. Create one above.")
            st.stop()
        
        selected_ws = st.selectbox("Select Workspace", workspaces, key="workspace_selector")
        workspace = Workspace(selected_ws)
        
        # Load initial dataset on first run
        if not st.session_state.initial_dataset_loaded:
            initial_doc = load_initial_dataset(workspace)
            if initial_doc:
                st.success("Loaded initial dataset: Hull - Marco Polo_Memo.pdf")
            st.session_state.initial_dataset_loaded = True
        
        stats = workspace.get_stats()
        
        st.markdown("---")
        st.markdown("### Stats")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Docs", stats["total_documents"])
        with c2:
            st.metric("Size", f"{stats['total_size_mb']:.1f}MB")
        
        c3, c4 = st.columns(2)
        with c3:
            st.metric("Analyzed", stats.get('analyzed_documents', 0))
        with c4:
            st.metric("Reviews", stats.get('pending_reviews', 0))
        
        if stats.get("insurance_type_distribution"):
            with st.expander("Types"):
                for itype, count in stats["insurance_type_distribution"].items():
                    st.write(f"• {itype}: {count}")
        
        st.markdown("---")
        if st.button("Delete Workspace", key="delete_ws"):
            if st.checkbox(f"Confirm delete", key="confirm_del"):
                import shutil
                shutil.rmtree(workspace.workspace_dir)
                st.success("Deleted!")
                st.rerun()
    
    # Main Tabs
    tabs = st.tabs([
        "Chat",
        "Documents",
        "Upload",
        "Analysis",
        "Review"
    ])
    
    # TAB 1: Chat
    with tabs[0]:
        st.markdown("### Chat")
        
        if stats["total_documents"] == 0:
            st.warning("No documents yet.")
        
        with st.expander("Filters"):
            col1, col2, col3 = st.columns(3)
            with col1:
                ins_types = ["All"] + list(set(d.get("insurance_type", "Other") for d in workspace.metadata))
                filter_ins = st.selectbox("Type", ins_types, key="flt_ins")
            with col2:
                filter_risk = st.selectbox("Risk", ["All", "Low", "Medium", "Medium-High", "High", "Critical"], key="flt_risk")
            with col3:
                filter_dec = st.selectbox("Decision", ["All", "Approved", "Declined", "Conditional", "Pending"], key="flt_dec")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        
        if prompt := st.chat_input("Ask about underwriting cases..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    filters = {}
                    if filter_ins != "All":
                        filters["insurance_type"] = filter_ins
                    if filter_risk != "All":
                        filters["risk_level"] = filter_risk
                    if filter_dec != "All":
                        filters["decision"] = filter_dec
                    
                    retrieved = workspace.search_documents(prompt, top_k=5, filters=filters)
                    resp = generate_cot_response(prompt, retrieved)
                    st.markdown(resp)
                    
                    if retrieved:
                        with st.expander(f"{len(retrieved)} Retrieved Cases"):
                            for d in retrieved:
                                st.markdown(f"**{d['doc_id']}** - {d['filename']}")
                                tags_html = f'<span class="tag-insurance">{d.get("insurance_type", "Other")}</span>'
                                for t in d["tags"].get("equipment", [])[:2]:
                                    tags_html += f'<span class="tag-badge tag-equipment">{t}</span>'
                                st.markdown(tags_html, unsafe_allow_html=True)
                                st.markdown("---")
            
            st.session_state.messages.append({"role": "assistant", "content": resp})
    
    # TAB 2: Documents
    with tabs[1]:
        st.markdown("### Documents")
        if not workspace.metadata:
            st.info("No documents yet.")
        else:
            left, right = st.columns([1, 2.2])
            
            with left:
                st.markdown("#### Filter")
                q = st.text_input("Search", key="doc_search")
                
                docs = workspace.metadata
                
                if q:
                    ql = q.lower()
                    docs = [d for d in docs if ql in d["filename"].lower() or ql in d.get("case_summary", "").lower()]
                
                docs = sorted(docs, key=lambda d: d.get("upload_date", ""), reverse=True)
                
                st.caption(f"{len(docs)} found")
                
                options = {f"{d['filename'][:40]}...": d["doc_id"] for d in docs}
                
                selected_id = None
                if options:
                    selected_id = st.radio("Docs", list(options.keys()), index=0, key="doc_sel", label_visibility="collapsed")
                
                selected_doc = None
                if selected_id:
                    sel_id = options[selected_id]
                    selected_doc = next((d for d in docs if d["doc_id"] == sel_id), None)
                
                if selected_doc:
                    if st.button("Delete", use_container_width=True, key="del_doc"):
                        workspace.delete_document(selected_doc["doc_id"])
                        st.success("Deleted!")
                        st.rerun()
            
            with right:
                st.markdown("#### Preview")
                if not selected_doc:
                    st.info("Select a document")
                else:
                    doc = selected_doc
                    st.markdown(f"### {doc['filename']}")
                    
                    main_tag = doc["tags"].get("equipment", ["Other"])[0]
                    st.caption(f"Main Tag: **{main_tag}** (from filename)")
                    
                    status_html = f'<span class="tag-insurance">{doc.get("insurance_type", "Other")}</span>'
                    if doc.get("has_deep_analysis"):
                        status_html += ' <span class="analysis-badge">Analyzed</span>'
                    st.markdown(status_html, unsafe_allow_html=True)
                    
                    with open(doc["file_path"], "rb") as f:
                        st.download_button("Download", f, file_name=doc["filename"], use_container_width=True, key=f"dl_{doc['doc_id']}")
                    
                    with st.expander("Content", expanded=False):
                        ext = doc["file_format"]
                        if ext in ["docx", "doc"]:
                            text = extract_text_from_docx(doc["file_path"]) if ext == "docx" else ""
                            st.text_area("", value=text[:3000], height=250, key=f"prev_{doc['doc_id']}", label_visibility="collapsed")
                        elif ext == "txt":
                            text = extract_text_from_txt(doc["file_path"])
                            st.text_area("", value=text[:3000], height=250, key=f"prevt_{doc['doc_id']}", label_visibility="collapsed")
                        elif ext == "pdf":
                            st.info("PDF - Download to view")
                        else:
                            st.info("Preview not available")
                    
                    with st.expander("Metadata"):
                        st.write(f"**Summary:** {doc['case_summary']}")
                        st.write(f"**Insights:** {doc['key_insights']}")
    
    # TAB 3: Upload
    with tabs[2]:
        st.markdown("### Upload")
        st.caption("Main tag will be extracted from filename first word")
        
        if st.session_state.upload_success:
            st.success("Uploaded!")
            st.session_state.upload_success = False
        
        with st.form("upload_form"):
            uploaded_file = st.file_uploader(
                "Choose file",
                type=list(SUPPORTED_FORMATS.keys()),
                key="file_up"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                auto_tag = st.checkbox("Auto-Tag (AI)", value=True, key="auto_t")
            with col2:
                deep_analysis = st.checkbox("Deep Analysis", value=False, key="deep_a")
            
            submitted = st.form_submit_button("Upload", use_container_width=True)
        
        if submitted and uploaded_file:
            with st.spinner("Processing..."):
                try:
                    temp_id = f"TEMP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    ext = uploaded_file.name.split('.')[-1].lower()
                    temp_path = os.path.join(workspace.documents_dir, f"{temp_id}.{ext}")
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    extracted_text = extract_text_from_file(temp_path, ext)
                    
                    auto_tags = auto_generate_tags(uploaded_file.name, extracted_text[:1000])
                    
                    st.success(f"Extracted main tag: **{auto_tags['equipment'][0]}** from filename")
                    
                    if auto_tag:
                        auto = auto_annotate_by_llm(extracted_text, uploaded_file.name)
                        auto["tags"]["equipment"] = auto_tags["equipment"] + [t for t in auto["tags"].get("equipment", []) if t not in auto_tags["equipment"]]
                        auto["tags"]["industry"] = auto_tags["industry"]
                        auto["tags"]["timeline"] = auto_tags["timeline"]
                    else:
                        auto = {
                            "tags": auto_tags,
                            "insurance_type": "Other",
                            "decision": "Pending",
                            "premium": 0,
                            "retention": 0,
                            "limit": 0,
                            "risk_level": "Medium",
                            "case_summary": "Manual review",
                            "key_insights": "Not tagged"
                        }
                    
                    doc = workspace.add_document(
                        uploaded_file=uploaded_file,
                        tags=auto["tags"],
                        case_summary=auto["case_summary"],
                        key_insights=auto["key_insights"],
                        decision=auto["decision"],
                        premium=int(auto.get("premium", 0) or 0),
                        risk_level=auto["risk_level"],
                        extracted_text_preview=extracted_text[:1000],
                        has_deep_analysis=False,
                        insurance_type=auto.get("insurance_type", "Other"),
                        retention=int(auto.get("retention", 0) or 0),
                        limit=int(auto.get("limit", 0) or 0)
                    )
                    
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except:
                        pass
                    
                    if auto_tag:
                        with st.expander("Auto-Tag Results"):
                            st.write(f"**Main Tag (from filename):** {auto_tags['equipment'][0]}")
                            st.write(f"**Type:** {auto.get('insurance_type')}")
                            st.write(f"**Decision:** {auto['decision']}")
                    
                    if deep_analysis:
                        st.info("Starting analysis...")
                        analysis_result = perform_dual_track_analysis(
                            file_path=doc["file_path"],
                            file_format=doc["file_format"],
                            filename=doc["filename"],
                            doc_id=doc["doc_id"],
                            workspace_name=workspace.name
                        )
                        
                        for d in workspace.metadata:
                            if d["doc_id"] == doc["doc_id"]:
                                d["has_deep_analysis"] = True
                                break
                        workspace._save_metadata()
                        
                        st.balloons()
                    
                    st.session_state.upload_success = True
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed: {e}")
                    st.error(traceback.format_exc())
        elif submitted:
            st.error("Select a file")
    
    # TAB 4: Analysis
    with tabs[3]:
        st.markdown("### Analysis")
        st.caption("Handwriting classification: Conclusion | Todo | Risk | Communication")
        
        if not workspace.metadata:
            st.info("No documents.")
        else:
            doc_opts = {f"{d['filename'][:50]}": d["doc_id"] for d in workspace.metadata}
            
            sel_analysis = st.selectbox("Select:", list(doc_opts.keys()), key="ana_sel")
            
            if sel_analysis:
                ana_doc_id = doc_opts[sel_analysis]
                ana_doc = next((d for d in workspace.metadata if d["doc_id"] == ana_doc_id), None)
                
                if ana_doc:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{ana_doc['filename']}**")
                    with col2:
                        if ana_doc.get("has_deep_analysis"):
                            st.success("Analyzed")
                        else:
                            st.info("Not analyzed")
                    
                    if st.button("Start Analysis", type="primary", use_container_width=True, key="start_ana"):
                        analysis_result = perform_dual_track_analysis(
                            file_path=ana_doc["file_path"],
                            file_format=ana_doc["file_format"],
                            filename=ana_doc["filename"],
                            doc_id=ana_doc["doc_id"],
                            workspace_name=workspace.name
                        )
                        
                        for doc in workspace.metadata:
                            if doc["doc_id"] == ana_doc_id:
                                doc["has_deep_analysis"] = True
                                break
                        workspace._save_metadata()
                        
                        st.balloons()
                        st.rerun()
                    
                    ana_file = os.path.join(ANALYSIS_DIR, f"{ana_doc_id}_dual_analysis.json")
                    if os.path.exists(ana_file):
                        st.markdown("---")
                        st.markdown("### Results")
                        
                        try:
                            with open(ana_file, 'r', encoding='utf-8') as f:
                                ana_data = json.load(f)
                            
                            cat_counts = ana_data['metadata'].get('annotation_categories', {})
                            if cat_counts:
                                st.markdown("#### Handwriting Categories")
                                c1, c2, c3, c4 = st.columns(4)
                                with c1:
                                    st.metric("Conclusion", cat_counts.get('conclusion', 0))
                                with c2:
                                    st.metric("To-Do", cat_counts.get('todo', 0))
                                with c3:
                                    st.metric("Risk", cat_counts.get('risk', 0))
                                with c4:
                                    st.metric("Communication", cat_counts.get('communication', 0))
                            
                            view = st.radio("View:", ["Summary", "Classifications", "Electronic", "Q&A"], horizontal=True, key="view")
                            
                            if view == "Summary":
                                st.markdown(ana_data.get("dual_track_analysis", {}).get("integration_summary", "No summary"))
                            
                            elif view == "Classifications":
                                st.markdown("#### Classified Handwriting Annotations")
                                
                                hw_data = ana_data.get("dual_track_analysis", {}).get("handwritten_annotations", {})
                                annotations = hw_data.get("annotations", [])
                                
                                if annotations:
                                    by_cat = {
                                        "conclusion": [],
                                        "todo": [],
                                        "risk": [],
                                        "communication": []
                                    }
                                    
                                    for ann in annotations:
                                        cat = ann.get("category", "")
                                        if cat in by_cat:
                                            by_cat[cat].append(ann)
                                    
                                    if by_cat["conclusion"]:
                                        st.markdown('<div class="category-conclusion">Conclusions</div>', unsafe_allow_html=True)
                                        for ann in by_cat["conclusion"]:
                                            with st.expander(f"{ann.get('text', '')[:60]}..."):
                                                st.write(f"**Text:** {ann.get('text')}")
                                                st.write(f"**Location:** {ann.get('location')}")
                                                st.write(f"**Confidence:** {ann.get('confidence', 0):.0%}")
                                                if ann.get('context'):
                                                    st.write(f"**Context:** {ann.get('context')}")
                                    
                                    if by_cat["todo"]:
                                        st.markdown('<div class="category-todo">To-Do</div>', unsafe_allow_html=True)
                                        for ann in by_cat["todo"]:
                                            with st.expander(f"{ann.get('text', '')[:60]}..."):
                                                st.write(f"**Text:** {ann.get('text')}")
                                                st.write(f"**Urgency:** {ann.get('urgency', 'medium')}")
                                                st.write(f"**Assignee:** {ann.get('assignee', 'TBD')}")
                                                st.write(f"**Location:** {ann.get('location')}")
                                    
                                    if by_cat["risk"]:
                                        st.markdown('<div class="category-risk">Risk Factors</div>', unsafe_allow_html=True)
                                        for ann in by_cat["risk"]:
                                            with st.expander(f"{ann.get('text', '')[:60]}..."):
                                                st.write(f"**Text:** {ann.get('text')}")
                                                st.write(f"**Urgency:** {ann.get('urgency', 'high')}")
                                                st.write(f"**Location:** {ann.get('location')}")
                                    
                                    if by_cat["communication"]:
                                        st.markdown('<div class="category-communication">External Communication</div>', unsafe_allow_html=True)
                                        for ann in by_cat["communication"]:
                                            with st.expander(f"{ann.get('text', '')[:60]}..."):
                                                st.write(f"**Text:** {ann.get('text')}")
                                                st.write(f"**Location:** {ann.get('location')}")
                                else:
                                    st.info("No handwriting annotations found")
                            
                            elif view == "Electronic":
                                st.json(ana_data.get("dual_track_analysis", {}).get("electronic_text", {}))
                            
                            elif view == "Q&A":
                                qa = ana_data.get("dual_track_analysis", {}).get("qa_pairs", [])
                                if qa:
                                    for i, q in enumerate(qa, 1):
                                        with st.expander(f"Q{i}: {q.get('question_text', '')[:50]}..."):
                                            st.write(f"**Q:** {q.get('question_text')}")
                                            st.write(f"**A:** {q.get('answer_text')}")
                                else:
                                    st.info("No Q&A pairs")
                            
                            st.download_button(
                                "Download JSON",
                                json.dumps(ana_data, indent=2, ensure_ascii=False),
                                file_name=f"{ana_doc_id}_analysis.json",
                                mime="application/json",
                                key="dl_ana"
                            )
                        except Exception as e:
                            st.error(f"Error: {e}")
    
    # TAB 5: Review
    with tabs[4]:
        st.markdown("### Review Workbench")
        st.caption("Review and classify handwriting")
        
        pending = get_all_pending_reviews(workspace)
        
        if not pending:
            st.success("No pending reviews!")
        else:
            st.warning(f"{len(pending)} items need review")
            
            if st.session_state.review_submitted:
                st.success("Submitted!")
                st.session_state.review_submitted = False
                if st.button("Refresh"):
                    st.session_state.current_review_index = 0
                    st.rerun()
            
            idx = st.session_state.current_review_index
            
            if idx < len(pending):
                item = pending[idx]
                
                st.progress((idx + 1) / len(pending))
                st.caption(f"{idx + 1} / {len(pending)}")
                
                doc_id = item.get("doc_id")
                image_id = item.get("image_id")
                
                st.markdown(f"**{item.get('doc_filename')}** - Page {item.get('page', 0)}")
                
                form_key = f"rev_{doc_id}_{image_id}_{idx}"
                
                with st.form(form_key):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        img_path = item.get("image_path")
                        if img_path and os.path.exists(img_path):
                            try:
                                st.image(img_path, use_column_width=True)
                            except:
                                st.error("Image error")
                        else:
                            st.info("No image")
                    
                    with col2:
                        st.markdown("**Classify handwriting:**")
                        category = st.selectbox(
                            "Category",
                            options=list(HANDWRITING_CATEGORIES.keys()),
                            format_func=lambda x: f"{HANDWRITING_CATEGORIES[x]} ({x})",
                            key=f"cat_{form_key}"
                        )
                        
                        transcribed = st.text_area(
                            "Transcription",
                            placeholder="Type text...",
                            height=120,
                            key=f"trans_{form_key}"
                        )
                        
                        notes = st.text_area(
                            "Notes",
                            placeholder="Context...",
                            height=80,
                            key=f"notes_{form_key}"
                        )
                        
                        accurate = st.checkbox("OCR accurate", key=f"acc_{form_key}")
                        reviewer = st.text_input("Name", value="Underwriter", key=f"revr_{form_key}")
                    
                    col_sub, col_skip = st.columns(2)
                    with col_sub:
                        submit = st.form_submit_button("Submit", use_container_width=True)
                    with col_skip:
                        skip = st.form_submit_button("Skip", use_container_width=True)
                    
                    if submit:
                        if not transcribed:
                            st.error("Provide transcription")
                        else:
                            success = submit_handwriting_review(
                                doc_id=doc_id,
                                image_id=image_id,
                                transcribed_text=transcribed,
                                category=category,
                                reviewer_notes=notes,
                                is_accurate=accurate,
                                reviewer=reviewer
                            )
                            
                            if success:
                                st.session_state.review_submitted = True
                                st.session_state.current_review_index = 0
                                st.rerun()
                    
                    if skip:
                        st.session_state.current_review_index = (idx + 1) % len(pending)
                        st.rerun()
                
                st.markdown("---")
                n1, n2, n3 = st.columns([1, 2, 1])
                with n1:
                    if st.button("Previous", disabled=(idx == 0), use_container_width=True, key="prev"):
                        st.session_state.current_review_index = idx - 1
                        st.rerun()
                with n2:
                    st.markdown(f"<center>{idx + 1} / {len(pending)}</center>", unsafe_allow_html=True)
                with n3:
                    if st.button("Next", disabled=(idx >= len(pending) - 1), use_container_width=True, key="next"):
                        st.session_state.current_review_index = idx + 1
                        st.rerun()

if __name__ == "__main__":
    main()
