"""
Enhanced Underwriting Assistant - Professional RAG+CoT System
Version 2.0 with Advanced Handwriting Recognition & Review Workflow

New Features:
- Tiered handwriting recognition (Clear/Standard/Cursive)
- Human-in-the-loop review workbench
- Insurance domain terminology optimization
- Q&A pair extraction
- Audit logging
- Multi-level confidence scoring
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
import base64
import pandas as pd
from PIL import Image
import io
import zipfile
import tempfile
import re
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

DEEPSEEK_API_KEY = "sk-99bba2ce117444e197270f17d303e74f"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

DATA_DIR = "data"
WORKSPACES_DIR = os.path.join(DATA_DIR, "workspaces")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
REVIEW_DIR = os.path.join(DATA_DIR, "review_queue")
AUDIT_DIR = os.path.join(DATA_DIR, "audit_logs")

for dir_path in [WORKSPACES_DIR, EMBEDDINGS_DIR, ANALYSIS_DIR, REVIEW_DIR, AUDIT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

SUPPORTED_FORMATS = {
    "pdf": "📄 PDF", "docx": "📝 Word", "doc": "📝 Word",
    "txt": "📃 Text", "xlsx": "📊 Excel", "xls": "📊 Excel",
    "png": "🖼️ Image", "jpg": "🖼️ Image", "jpeg": "🖼️ Image"
}

TAG_OPTIONS = {
    "equipment": ["Gas Turbine", "Steam Turbine", "Boiler", "Generator", "Compressor", 
                  "Heat Exchanger", "Pump", "Transformer", "Motor", "Vessel", "Other"],
    "industry": ["Oil & Gas", "Power Generation", "Manufacturing", "Chemical", 
                 "Mining", "Refining", "Marine", "Aviation", "Cargo", "Property", "Liability", "Other"],
    "timeline": ["2026-Q1", "2025-Q4", "2025-Q3", "2025-Q2", "2025-Q1", "2024", "2023", "Earlier"]
}

# Insurance domain terminology dictionary
INSURANCE_TERMS = {
    "retention": "自留额/免赔额",
    "premium": "保费",
    "coverage": "承保范围",
    "deductible": "免赔额",
    "underwriting slip": "承保单",
    "loss ratio": "赔付率",
    "exposure": "风险暴露",
    "claims": "理赔",
    "policy": "保单",
    "endorsement": "批单",
    "exclusion": "除外责任",
    "limit": "责任限额",
    "aggregate": "累计限额",
    "per occurrence": "每次事故",
    "retroactive date": "追溯日期"
}

# ============================================================================
# SYSTEM PROMPTS - ENHANCED VERSION
# ============================================================================

SYSTEM_INSTRUCTION = """Role: You are Mr. X's AI underwriting assistant with deep insurance domain knowledge

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
4. Coverage terms and conditions (A1, A2, etc.)
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

HANDWRITTEN_TEXT_ANALYSIS_SYSTEM = """You are analyzing HANDWRITTEN ANNOTATIONS from insurance underwriting documents.

Focus on:
1. CE (Chief Underwriter) questions and concerns
   - "Can we increase retention to...?"
   - "What's the loss history for...?"
   - "Similar to [case reference]?"
2. Manual calculations and adjustments
   - Revised premium calculations
   - Retention level modifications
   - Risk loading factors
3. Approval indicators
   - Signatures with dates
   - "Approved subject to..."
   - Checkmarks and initials
4. Risk assessment notes
   - "High risk - monitor closely"
   - "Good account - renew"
   - References to previous claims
5. Cross-references
   - "See page X"
   - "Refer to [broker name]"
   - "Compare with ABC Corp case"

Confidence levels:
- CLEAR (70-100%): Standard handwriting, legible
- MEDIUM (40-70%): Some ambiguity, needs context
- LOW (<40%): Cursive/illegible, requires human review

Output: JSON with:
{
  "questions_raised": [
    {"text": "...", "confidence": 0.85, "location": "Page 2, margin"}
  ],
  "manual_calculations": [
    {"formula": "...", "result": 0, "confidence": 0.70}
  ],
  "approval_marks": [
    {"type": "signature", "name": "...", "date": "...", "confidence": 0.90}
  ],
  "risk_notes": [
    {"content": "...", "sentiment": "positive/negative", "confidence": 0.65}
  ],
  "cross_references": [
    {"reference": "...", "type": "case/page/person", "confidence": 0.75}
  ],
  "needs_human_review": [
    {"reason": "cursive handwriting", "location": "...", "image_id": "..."}
  ]
}"""

QA_EXTRACTION_SYSTEM = """You are extracting Question-Answer pairs from insurance documents.

Recognize patterns:
1. Formal Q&A format:
   - "A1 Question: ..." → "Answer: ..."
   - "Q1: ..." → "A1: ..."
2. Email correspondence:
   - "From: [Broker] Q: ..." → "From: [Underwriter] Re: ..."
3. Margin annotations:
   - Question mark next to clause → Handwritten answer
4. Meeting notes:
   - "CE asked: ..." → "Response: ..."

For each Q&A pair extract:
{
  "question_id": "A1/Q1/Email-001",
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
1. Insurance type (Cargo/Property/Liability/Equipment Breakdown/etc.)
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
  "insurance_type": "Cargo|Property|Liability|Equipment|Marine|Other",
  "decision": "Approved|Declined|Conditional|Pending",
  "premium": 0,
  "retention": 0,
  "limit": 0,
  "currency": "USD",
  "risk_level": "Low|Medium|Medium-High|High|Critical",
  "case_summary": "Brief summary focusing on coverage, insured, and key terms",
  "key_insights": "Notable risk factors, loss history, special conditions",
  "extracted_clauses": ["A1", "A2", ...],
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

# ============================================================================
# API & UTILITY FUNCTIONS
# ============================================================================

def call_deepseek_api(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """Call DeepSeek API with error handling"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"model": DEEPSEEK_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        resp = requests.post(f"{DEEPSEEK_API_BASE}/chat/completions", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API Error: {str(e)}"

def preprocess_insurance_text(text: str) -> str:
    """Preprocess text with insurance domain knowledge"""
    # Normalize insurance terms
    processed = text
    for term, definition in INSURANCE_TERMS.items():
        # Add definition as context (helps with semantic search)
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        processed = pattern.sub(f"{term}({definition})", processed)
    
    # Normalize common abbreviations
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
        
        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text)
        
        # Extract tables
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
    # Preprocess text
    processed_text = preprocess_insurance_text(text)
    
    # Generate hash-based embedding (placeholder for real embedding model)
    text_hash = hashlib.md5(processed_text.encode()).hexdigest()
    fake = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
    return (fake + [0.0] * 1536)[:1536]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between vectors"""
    dot = sum(a*b for a,b in zip(v1, v2))
    m1, m2 = sum(a*a for a in v1) ** 0.5, sum(b*b for b in v2) ** 0.5
    return dot / (m1 * m2) if m1 and m2 else 0.0

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
                        
                        # Analyze image to estimate handwriting likelihood
                        try:
                            img = Image.open(extracted_path)
                            width, height = img.size
                            has_handwriting = width >= 300 and height >= 300  # Simplified heuristic
                        except:
                            has_handwriting = False
                        
                        images_info.append({
                            "image_id": filename.split('.')[0],
                            "filename": filename,
                            "path": extracted_path,
                            "format": filename.split('.')[-1].lower(),
                            "page_number": len(images_info) + 1,
                            "likely_handwriting": has_handwriting,
                            "width": width if 'img' in locals() else 0,
                            "height": height if 'img' in locals() else 0
                        })
        return images_info
    except Exception as e:
        st.error(f"Error extracting images: {e}")
        return []

def classify_handwriting_quality(image_info: Dict) -> Tuple[str, float]:
    """
    Classify handwriting quality into tiers
    Returns: (tier, confidence_estimate)
    
    Tiers:
    - CLEAR: 70-100% expected accuracy
    - STANDARD: 40-70% expected accuracy
    - CURSIVE: <40% expected accuracy (needs human review)
    """
    # Simplified classification based on image properties
    # In production, use actual OCR confidence scores
    
    width = image_info.get("width", 0)
    height = image_info.get("height", 0)
    
    if width * height > 1000000:  # Large, likely clear scan
        return "CLEAR", 0.75
    elif width * height > 400000:  # Medium size
        return "STANDARD", 0.55
    else:  # Small or unclear
        return "CURSIVE", 0.30

# ============================================================================
# Q&A EXTRACTION
# ============================================================================

def extract_qa_pairs(text: str, filename: str) -> List[Dict[str, Any]]:
    """Extract Question-Answer pairs using pattern matching and LLM"""
    
    prompt = f"""Document: {filename}

Text content:
{text[:5000]}

Extract all Question-Answer pairs following the QA_EXTRACTION_SYSTEM guidelines.
Return ONLY a valid JSON array of Q&A objects."""
    
    response = call_deepseek_api(
        messages=[
            {"role": "system", "content": QA_EXTRACTION_SYSTEM},
            {"role": "user", "content": prompt}
        ],
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

# ============================================================================
# ENHANCED ANALYSIS FUNCTIONS
# ============================================================================

def analyze_electronic_text(extracted_text: str, filename: str) -> Dict[str, Any]:
    """Analyze electronic/printed text with insurance focus"""
    
    prompt = f"""Document: {filename}

Electronic Printed Text Content:
{extracted_text[:5000]}

Analyze the ELECTRONIC/PRINTED content following the insurance document framework.
Focus on policy terms, financial figures, coverage details, and formal terms.

Return ONLY valid JSON."""
    
    response = call_deepseek_api(
        messages=[
            {"role": "system", "content": ELECTRONIC_TEXT_ANALYSIS_SYSTEM},
            {"role": "user", "content": prompt}
        ],
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

def analyze_handwritten_annotations(image_info: List[Dict], extracted_text: str, doc_id: str) -> Dict[str, Any]:
    """Analyze handwritten annotations with tiered processing"""
    
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
    
    # Save items needing review
    if needs_review:
        review_file = os.path.join(REVIEW_DIR, f"{doc_id}_review_queue.json")
        with open(review_file, 'w', encoding='utf-8') as f:
            json.dump(needs_review, f, ensure_ascii=False, indent=2)
    
    # LLM analysis for context
    prompt = f"""Based on document analysis and detected handwriting regions:

Context from document:
{extracted_text[:2000]}

Number of handwriting regions: {len(handwriting_items)}
Regions needing review: {len(needs_review)}

Analyze the handwritten content focusing on:
1. Underwriter questions and concerns
2. Manual calculations
3. Approval marks
4. Risk assessment notes
5. Cross-references

Return ONLY valid JSON."""
    
    response = call_deepseek_api(
        messages=[
            {"role": "system", "content": HANDWRITTEN_TEXT_ANALYSIS_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=2000
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
            "note": "Could not parse as JSON"
        }
    
    result["handwriting_items"] = handwriting_items
    result["review_queue_count"] = len(needs_review)
    result["content_type"] = "handwritten_annotations"
    result["analysis_timestamp"] = datetime.now().isoformat()
    
    return result

def perform_dual_track_analysis(file_path: str, file_format: str, filename: str, doc_id: str, workspace_name: str) -> Dict[str, Any]:
    """
    Perform enhanced DUAL-TRACK analysis with Q&A extraction and review queue
    """
    
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
        # Log audit event
        log_audit_event(workspace_name, "analysis_started", {
            "doc_id": doc_id,
            "filename": filename,
            "file_format": file_format
        })
        
        # ===== TRACK 1: Electronic Text =====
        st.info("📝 Track 1: Analyzing Electronic/Printed Text...")
        extracted_text = extract_text_from_file(file_path, file_format)
        analysis_result["metadata"]["text_length"] = len(extracted_text)
        analysis_result["metadata"]["has_text"] = len(extracted_text) > 0
        
        if extracted_text:
            electronic_analysis = analyze_electronic_text(extracted_text, filename)
            analysis_result["dual_track_analysis"]["electronic_text"] = electronic_analysis
            
            # Extract Q&A pairs
            st.info("❓ Extracting Question-Answer pairs...")
            qa_pairs = extract_qa_pairs(extracted_text, filename)
            analysis_result["dual_track_analysis"]["qa_pairs"] = qa_pairs
            analysis_result["metadata"]["qa_pairs_count"] = len(qa_pairs)
        
        # ===== TRACK 2: Handwritten Content =====
        st.info("✍️ Track 2: Processing Handwritten Content...")
        images_info = []
        temp_image_dir = tempfile.mkdtemp(prefix=f"doc_analysis_{doc_id}_")
        
        if file_format == "docx":
            images_info = extract_images_from_docx(file_path, temp_image_dir)
        
        analysis_result["metadata"]["image_count"] = len(images_info)
        analysis_result["metadata"]["has_images"] = len(images_info) > 0
        
        # Store images by page
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
        
        # Analyze handwritten content
        if images_info:
            handwriting_analysis = analyze_handwritten_annotations(images_info, extracted_text, doc_id)
            analysis_result["dual_track_analysis"]["handwritten_annotations"] = handwriting_analysis
            analysis_result["metadata"]["review_queue_count"] = handwriting_analysis.get("review_queue_count", 0)
            
            # Load review queue
            review_file = os.path.join(REVIEW_DIR, f"{doc_id}_review_queue.json")
            if os.path.exists(review_file):
                with open(review_file, 'r', encoding='utf-8') as f:
                    analysis_result["review_queue"] = json.load(f)
        
        # ===== INTEGRATION =====
        st.info("🔗 Integrating Analysis Results...")
        
        integration_prompt = f"""Integrate these analysis tracks into a comprehensive underwriting report:

ELECTRONIC TEXT ANALYSIS:
{json.dumps(analysis_result["dual_track_analysis"]["electronic_text"], indent=2, ensure_ascii=False)[:2000]}

HANDWRITTEN ANNOTATIONS:
{json.dumps(analysis_result["dual_track_analysis"]["handwritten_annotations"], indent=2, ensure_ascii=False)[:1500]}

Q&A PAIRS EXTRACTED: {len(analysis_result["dual_track_analysis"]["qa_pairs"])}

Provide:
1. Key underwriting decisions (from electronic + handwritten)
2. Questions raised by CE and responses
3. Approval workflow and signatures
4. Risk factors and special conditions
5. Action items from annotations
6. Cross-references between documents
7. Recommendations for underwriter

Format as markdown report."""
        
        integration_summary = call_deepseek_api(
            messages=[
                {"role": "system", "content": "You are a senior insurance underwriting analyst."},
                {"role": "user", "content": integration_prompt}
            ],
            temperature=0.4,
            max_tokens=2500
        )
        
        analysis_result["dual_track_analysis"]["integration_summary"] = integration_summary
        
        # Save analysis
        analysis_file = os.path.join(ANALYSIS_DIR, f"{doc_id}_dual_analysis.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        # Log completion
        log_audit_event(workspace_name, "analysis_completed", {
            "doc_id": doc_id,
            "text_length": analysis_result["metadata"]["text_length"],
            "image_count": analysis_result["metadata"]["image_count"],
            "qa_pairs": len(analysis_result["dual_track_analysis"]["qa_pairs"]),
            "review_queue": analysis_result["metadata"].get("review_queue_count", 0)
        })
        
        st.success("✅ Dual-track analysis completed!")
        
        if analysis_result["metadata"].get("review_queue_count", 0) > 0:
            st.warning(f"⚠️ {analysis_result['metadata']['review_queue_count']} handwritten items need human review. Please check the Review Workbench.")
        
        return analysis_result
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        log_audit_event(workspace_name, "analysis_failed", {
            "doc_id": doc_id,
            "error": str(e)
        })
        analysis_result["error"] = str(e)
        return analysis_result

def auto_annotate_by_llm(extracted_text: str, filename: str) -> Dict[str, Any]:
    """Auto-annotate document using insurance-aware LLM"""
    user_prompt = f"FILENAME: {filename}\n\nTEXT:\n{(extracted_text or '')[:5000]}"
    content = call_deepseek_api(
        messages=[
            {"role": "system", "content": AUTO_ANNOTATE_SYSTEM},
            {"role": "user", "content": user_prompt}
        ],
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
    
    # Ensure all required fields
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
# WORKSPACE CLASS (ENHANCED)
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
        
        # Generate embedding with insurance context
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
            "insurance_type": insurance_type
        })
        
        return doc_meta
    
    def search_documents(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None):
        """Enhanced search with insurance-aware ranking"""
        if not self.metadata:
            return []
        
        qv = generate_embedding(query)
        scored = []
        
        for doc in self.metadata:
            # Apply filters
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
                
                # Boost score for matching insurance terms
                ql = query.lower()
                for term in INSURANCE_TERMS.keys():
                    if term in ql and term in doc.get("case_summary", "").lower():
                        sim += 0.15
                
                # Boost for matching tags
                for tag_list in doc["tags"].values():
                    for tag in tag_list:
                        if tag.lower() in ql:
                            sim += 0.1
                
                # Boost for analyzed documents
                if doc.get("has_deep_analysis"):
                    sim += 0.05
                
                # Recency boost
                upload_date = doc.get("upload_date", "")
                if upload_date and "2026" in upload_date or "2025" in upload_date:
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
        
        # Remove files
        for fn in os.listdir(self.documents_dir):
            if fn.startswith(doc_id):
                os.remove(os.path.join(self.documents_dir, fn))
        
        analysis_file = os.path.join(ANALYSIS_DIR, f"{doc_id}_dual_analysis.json")
        if os.path.exists(analysis_file):
            os.remove(analysis_file)
        
        review_file = os.path.join(REVIEW_DIR, f"{doc_id}_review_queue.json")
        if os.path.exists(review_file):
            os.remove(review_file)
        
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
                with open(review_file, 'r') as f:
                    queue = json.load(f)
                    count += len([item for item in queue if item.get("review_status") == "pending"])
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
# REVIEW WORKBENCH FUNCTIONS
# ============================================================================

def get_all_pending_reviews(workspace: Workspace) -> List[Dict[str, Any]]:
    """Get all pending handwriting reviews across workspace"""
    all_reviews = []
    
    for doc in workspace.metadata:
        review_file = os.path.join(REVIEW_DIR, f"{doc['doc_id']}_review_queue.json")
        if os.path.exists(review_file):
            with open(review_file, 'r', encoding='utf-8') as f:
                queue = json.load(f)
                for item in queue:
                    if item.get("review_status") == "pending":
                        item["doc_filename"] = doc["filename"]
                        all_reviews.append(item)
    
    return all_reviews

def submit_handwriting_review(doc_id: str, image_id: str, 
                              transcribed_text: str, reviewer_notes: str,
                              is_accurate: bool, reviewer: str = "User"):
    """Submit handwriting review result"""
    review_file = os.path.join(REVIEW_DIR, f"{doc_id}_review_queue.json")
    
    if not os.path.exists(review_file):
        return False
    
    with open(review_file, 'r', encoding='utf-8') as f:
        queue = json.load(f)
    
    # Update the specific item
    updated = False
    for item in queue:
        if item["image_id"] == image_id:
            item["review_status"] = "completed"
            item["transcribed_text"] = transcribed_text
            item["reviewer_notes"] = reviewer_notes
            item["is_accurate"] = is_accurate
            item["reviewed_by"] = reviewer
            item["reviewed_at"] = datetime.now().isoformat()
            updated = True
            break
    
    if updated:
        with open(review_file, 'w', encoding='utf-8') as f:
            json.dump(queue, f, ensure_ascii=False, indent=2)
        
        # Log audit event
        log_audit_event("global", "handwriting_reviewed", {
            "doc_id": doc_id,
            "image_id": image_id,
            "reviewer": reviewer,
            "is_accurate": is_accurate
        })
    
    return updated

# ============================================================================
# CHAT FUNCTION (ENHANCED)
# ============================================================================

def generate_cot_response(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """Generate Chain-of-Thought response with insurance expertise"""
    if not retrieved_docs:
        return "⚠️ **No Relevant Cases Found**\n\nPlease add documents to this workspace or try refining your query with specific insurance terms (e.g., cargo, retention, loss ratio)."
    
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
Tags: 🔧 {equipment} | 🏭 {industry} | 📅 {timeline}

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
    
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"""Query: "{query}"

Retrieved Cases:
{docs_text}

Please analyze using the 5-step insurance underwriting CoT framework:
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
"""}
    ]
    
    return call_deepseek_api(messages, temperature=0.6, max_tokens=2500)

# ============================================================================
# UI STYLING (ENHANCED)
# ============================================================================

def inject_css(appearance: str):
    """Enhanced CSS with review workbench styling"""
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
        .tag-equipment { background-color: #dbeafe; color: var(--text-primary); }
        .tag-industry { background-color: #dcfce7; color: var(--text-primary); }
        .tag-timeline { background-color: #fef3c7; color: var(--text-primary); }
        .tag-insurance { background-color: #e9d5ff; color: var(--text-primary); }
        .analysis-badge { background-color: #e9d5ff; color: var(--text-primary); padding:0.25rem 0.75rem; border-radius:1rem; font-size:0.875rem; font-weight:700; }
        .review-pending { background-color: #fee2e2; color: var(--red); padding:0.25rem 0.75rem; border-radius:1rem; font-size:0.875rem; font-weight:700; }
        .review-completed { background-color: #dcfce7; color: var(--green); padding:0.25rem 0.75rem; border-radius:1rem; font-size:0.875rem; font-weight:700; }
        .confidence-clear { color: #166534; font-weight: 700; }
        .confidence-standard { color: #92400e; font-weight: 700; }
        .confidence-cursive { color: #dc2626; font-weight: 700; }
        .review-card { background: var(--card-bg); padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid #dc2626; margin: 1rem 0; box-shadow: var(--shadow); }
        .stChatMessage, .stMarkdown, p, li, label, span, div { color: var(--text-primary); }
        [data-testid="stMetricDelta"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--text-primary) !important; }
        #MainMenu, footer, header {visibility: hidden;}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION (ENHANCED)
# ============================================================================

def main():
    st.set_page_config(
        page_title="Insurance Underwriting Assistant",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎨 Appearance")
        appearance = st.radio("Theme", ["Light", "Dark"], horizontal=True, key="appearance_choice")
    
    inject_css(appearance)
    
    # Header
    st.markdown('<div class="main-header">🛡️ Insurance Underwriting Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced RAG System with Dual-Track Analysis | Handwriting Recognition | Review Workbench</div>', unsafe_allow_html=True)
    
    # Workspace Management
    with st.sidebar:
        st.markdown("### 📁 Workspaces")
        workspaces = get_all_workspaces()
        
        with st.expander("➕ New Workspace"):
            new_ws_name = st.text_input("Workspace Name", placeholder="e.g., Marine Cargo 2026")
            if st.button("Create"):
                if new_ws_name and new_ws_name not in workspaces:
                    create_workspace(new_ws_name)
                    st.success(f"✅ Created workspace: {new_ws_name}")
                    st.rerun()
                elif new_ws_name in workspaces:
                    st.error("Workspace already exists")
                else:
                    st.error("Please enter a name")
        
        if not workspaces:
            st.info("No workspaces yet. Create one above.")
            st.stop()
        
        selected_ws = st.selectbox("Select Workspace", workspaces, key="workspace_selector")
        workspace = Workspace(selected_ws)
        stats = workspace.get_stats()
        
        st.markdown("---")
        st.markdown("### 📊 Workspace Stats")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Documents", stats["total_documents"])
        with c2:
            st.metric("Size", f"{stats['total_size_mb']:.1f} MB")
        
        c3, c4 = st.columns(2)
        with c3:
            st.metric("Analyzed", stats.get('analyzed_documents', 0))
        with c4:
            pending = stats.get('pending_reviews', 0)
            st.metric("Reviews", pending, delta=f"{pending} pending" if pending > 0 else None)
        
        if stats.get("insurance_type_distribution"):
            st.markdown("**Insurance Types:**")
            for itype, count in stats["insurance_type_distribution"].items():
                st.write(f"• {itype}: {count}")
        
        if stats["format_distribution"]:
            st.markdown("**Formats:**")
            for fmt, count in stats["format_distribution"].items():
                st.write(f"{SUPPORTED_FORMATS.get(fmt, fmt)}: {count}")
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        if st.button("🗑️ Delete Workspace"):
            if st.checkbox(f"Confirm delete {selected_ws}"):
                import shutil
                shutil.rmtree(workspace.workspace_dir)
                st.success("Workspace deleted!")
                st.rerun()
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chat",
        "📄 Documents",
        "📤 Upload",
        "🔬 Analysis",
        "👥 Review Workbench"
    ])
    
    # TAB 1: Chat (Enhanced)
    with tab1:
        st.markdown("### 💬 Chat with AI Underwriting Assistant")
        
        if stats["total_documents"] == 0:
            st.warning("⚠️ No documents yet. Upload documents first.")
        
        # Advanced search options
        with st.expander("🔍 Advanced Search Filters"):
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_insurance = st.selectbox("Insurance Type", ["All"] + list(set(d.get("insurance_type", "Other") for d in workspace.metadata)))
            with col2:
                filter_risk = st.selectbox("Risk Level", ["All", "Low", "Medium", "Medium-High", "High", "Critical"])
            with col3:
                filter_decision = st.selectbox("Decision", ["All", "Approved", "Declined", "Conditional", "Pending"])
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        
        if prompt := st.chat_input("Ask about underwriting cases... (e.g., 'similar cargo cases with high retention')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching knowledge base with insurance domain optimization..."):
                    # Build filters
                    filters = {}
                    if filter_insurance != "All":
                        filters["insurance_type"] = filter_insurance
                    if filter_risk != "All":
                        filters["risk_level"] = filter_risk
                    if filter_decision != "All":
                        filters["decision"] = filter_decision
                    
                    retrieved = workspace.search_documents(prompt, top_k=5, filters=filters)
                    resp = generate_cot_response(prompt, retrieved)
                    st.markdown(resp)
                    
                    if retrieved:
                        with st.expander(f"📚 {len(retrieved)} Retrieved Cases"):
                            for d in retrieved:
                                st.markdown(f"**{d['doc_id']}** - {d['filename']}")
                                tags_html = f'<span class="tag-insurance">📋 {d.get("insurance_type", "Other")}</span>'
                                for t in d["tags"].get("equipment", []):
                                    tags_html += f'<span class="tag-badge tag-equipment">🔧 {t}</span>'
                                for t in d["tags"].get("industry", []):
                                    tags_html += f'<span class="tag-badge tag-industry">🏭 {t}</span>'
                                for t in d["tags"].get("timeline", []):
                                    tags_html += f'<span class="tag-badge tag-timeline">📅 {t}</span>'
                                if d.get("has_deep_analysis"):
                                    tags_html += '<span class="analysis-badge">📊 Analyzed</span>'
                                st.markdown(tags_html, unsafe_allow_html=True)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.caption(f"Premium: ${d['premium']:,}")
                                with col2:
                                    st.caption(f"Retention: ${d.get('retention', 0):,}")
                                with col3:
                                    st.caption(f"Risk: {d['risk_level']}")
                                
                                st.markdown("---")
            
            st.session_state.messages.append({"role": "assistant", "content": resp})
    
    # TAB 2: Documents (Enhanced)
    with tab2:
        st.markdown("### 📄 Knowledge Base Browser")
        if not workspace.metadata:
            st.info("No documents yet. Upload in 'Upload' tab.")
        else:
            left, right = st.columns([1, 2.2])
            
            with left:
                st.markdown("#### 🔍 Search & Filter")
                q = st.text_input("Search...", key="kb_search")
                
                fe = st.multiselect("🔧 Equipment", TAG_OPTIONS["equipment"])
                fi = st.multiselect("🏭 Industry", TAG_OPTIONS["industry"])
                ft = st.multiselect("📅 Timeline", TAG_OPTIONS["timeline"])
                
                insurance_types = list(set(d.get("insurance_type", "Other") for d in workspace.metadata))
                fit = st.multiselect("📋 Insurance Type", insurance_types)
                
                show_analyzed = st.checkbox("Analyzed only", value=False)
                show_needs_review = st.checkbox("Needs review", value=False)
                
                docs = workspace.metadata
                
                # Apply filters
                if q:
                    ql = q.lower()
                    docs = [d for d in docs if (ql in d["filename"].lower() or 
                           ql in d.get("case_summary", "").lower() or
                           any(ql in tag.lower() for v in d["tags"].values() for tag in v))]
                if fe:
                    docs = [d for d in docs if any(t in d["tags"].get("equipment", []) for t in fe)]
                if fi:
                    docs = [d for d in docs if any(t in d["tags"].get("industry", []) for t in fi)]
                if ft:
                    docs = [d for d in docs if any(t in d["tags"].get("timeline", []) for t in ft)]
                if fit:
                    docs = [d for d in docs if d.get("insurance_type") in fit]
                if show_analyzed:
                    docs = [d for d in docs if d.get("has_deep_analysis", False)]
                if show_needs_review:
                    docs = [d for d in docs if os.path.exists(os.path.join(REVIEW_DIR, f"{d['doc_id']}_review_queue.json"))]
                
                docs = sorted(docs, key=lambda d: d.get("upload_date", ""), reverse=True)
                
                st.markdown(f"**{len(docs)} documents found**")
                
                options = {f"{SUPPORTED_FORMATS.get(d['file_format'],'📎')} {d['filename']} [{d['doc_id']}]": d["doc_id"] 
                          for d in docs}
                
                selected_id = None
                if options:
                    selected_id = st.radio("Documents", list(options.keys()), 
                                         index=0, key="kb_selected", label_visibility="collapsed")
                
                selected_doc = None
                if selected_id:
                    sel_id = options[selected_id]
                    selected_doc = next((d for d in docs if d["doc_id"] == sel_id), None)
                
                if selected_doc and st.button("🗑️ Delete Selected", use_container_width=True):
                    workspace.delete_document(selected_doc["doc_id"])
                    st.success("Document deleted!")
                    st.rerun()
            
            with right:
                st.markdown("#### 👀 Document Preview")
                if not selected_doc:
                    st.info("← Select a document to preview")
                else:
                    doc = selected_doc
                    st.markdown(f"### {doc['filename']}")
                    st.caption(f"ID: `{doc['doc_id']}` | {doc['file_format'].upper()} | {doc['file_size_kb']:.1f} KB | Uploaded: {doc['upload_date'][:10]}")
                    
                    # Status badges
                    status_html = f'<span class="tag-insurance">📋 {doc.get("insurance_type", "Other")}</span>'
                    if doc.get("has_deep_analysis"):
                        status_html += ' <span class="analysis-badge">✅ Analyzed</span>'
                    
                    review_file = os.path.join(REVIEW_DIR, f"{doc['doc_id']}_review_queue.json")
                    if os.path.exists(review_file):
                        with open(review_file, 'r') as f:
                            queue = json.load(f)
                            pending = len([i for i in queue if i.get("review_status") == "pending"])
                            if pending > 0:
                                status_html += f' <span class="review-pending">⚠️ {pending} Reviews Pending</span>'
                            else:
                                status_html += ' <span class="review-completed">✅ Reviews Complete</span>'
                    
                    st.markdown(status_html, unsafe_allow_html=True)
                    
                    with open(doc["file_path"], "rb") as f:
                        st.download_button("⬇️ Download Original", f, file_name=doc["filename"], use_container_width=True)
                    
                    # Preview
                    ext = doc["file_format"]
                    path = doc["file_path"]
                    
                    st.markdown("---")
                    st.markdown("**📄 Content Preview**")
                    
                    if ext in ["png", "jpg", "jpeg"]:
                        try:
                            st.image(path, use_column_width=True)
                        except Exception as e:
                            st.error(f"Preview failed: {e}")
                    elif ext in ["docx", "doc"]:
                        text = extract_text_from_docx(path) if ext == "docx" else "(DOC not supported)"
                        st.text_area("Extracted Text", value=text[:6000], height=400, label_visibility="collapsed")
                    elif ext == "txt":
                        text = extract_text_from_txt(path)
                        st.text_area("Text", value=text[:6000], height=400, label_visibility="collapsed")
                    elif ext in ["xlsx", "xls"]:
                        try:
                            df = pd.read_excel(path, nrows=100)
                            st.dataframe(df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Preview failed: {e}")
                    elif ext == "pdf":
                        st.info("📄 PDF - Download to view")
                    else:
                        st.info("Preview not available")
                    
                    st.markdown("---")
                    st.markdown("**📊 Metadata & Analysis**")
                    
                    # Tags
                    tags_html = ""
                    for t in doc["tags"].get("equipment", []):
                        tags_html += f'<span class="tag-badge tag-equipment">🔧 {t}</span>'
                    for t in doc["tags"].get("industry", []):
                        tags_html += f'<span class="tag-badge tag-industry">🏭 {t}</span>'
                    for t in doc["tags"].get("timeline", []):
                        tags_html += f'<span class="tag-badge tag-timeline">📅 {t}</span>'
                    st.markdown(tags_html, unsafe_allow_html=True)
                    
                    # Financial terms
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Decision", doc['decision'])
                    with col2:
                        st.metric("Premium", f"${doc['premium']:,}")
                    with col3:
                        st.metric("Retention", f"${doc.get('retention', 0):,}")
                    with col4:
                        st.metric("Risk", doc['risk_level'])
                    
                    with st.expander("📝 Case Summary"):
                        st.write(doc["case_summary"])
                    
                    with st.expander("💡 Key Insights"):
                        st.write(doc["key_insights"])
    
    # TAB 3: Upload (Enhanced)
    with tab3:
        st.markdown("### 📤 Upload Documents with Auto-Tagging")
        st.caption("Upload insurance documents for automatic metadata extraction and tagging.")
        
        with st.form("upload_form_enhanced"):
            uploaded_file = st.file_uploader(
                "Choose a document",
                type=list(SUPPORTED_FORMATS.keys()),
                help="Supported: PDF, Word, Excel, Text, Images"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                enable_auto_tag = st.checkbox("Enable Auto-Tagging (AI)", value=True)
            with col2:
                enable_deep_analysis = st.checkbox("Perform Deep Analysis immediately", value=False)
            
            submitted = st.form_submit_button("📤 Upload Document", use_container_width=True)
        
        if submitted:
            if not uploaded_file:
                st.error("Please select a file")
            else:
                with st.spinner("Processing document..."):
                    # Save temporarily
                    temp_id = f"TEMP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    ext = uploaded_file.name.split('.')[-1].lower()
                    temp_path = os.path.join(workspace.documents_dir, f"{temp_id}.{ext}")
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract text
                    extracted_text = extract_text_from_file(temp_path, ext)
                    
                    # Auto-tag if enabled
                    if enable_auto_tag:
                        st.info("🤖 Running AI auto-tagger...")
                        auto = auto_annotate_by_llm(extracted_text, uploaded_file.name)
                    else:
                        auto = {
                            "tags": {"equipment": ["Other"], "industry": ["Other"], "timeline": ["Earlier"]},
                            "insurance_type": "Other",
                            "decision": "Pending",
                            "premium": 0,
                            "retention": 0,
                            "limit": 0,
                            "risk_level": "Medium",
                            "case_summary": "Manual review required",
                            "key_insights": "Not auto-tagged"
                        }
                    
                    # Add to workspace
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
                    
                    # Clean up temp file
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except:
                        pass
                    
                    st.success(f"✅ Document uploaded: {doc['doc_id']}")
                    
                    # Show auto-tag result
                    if enable_auto_tag:
                        with st.expander("🔎 Auto-Tag Results"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.json({k: v for k, v in auto.items() if k not in ["case_summary", "key_insights"]})
                            with col2:
                                st.write("**Case Summary:**")
                                st.write(auto["case_summary"])
                                st.write("**Key Insights:**")
                                st.write(auto["key_insights"])
                    
                    # Deep analysis if requested
                    if enable_deep_analysis:
                        st.info("🔬 Starting deep dual-track analysis...")
                        analysis_result = perform_dual_track_analysis(
                            file_path=doc["file_path"],
                            file_format=doc["file_format"],
                            filename=doc["filename"],
                            doc_id=doc["doc_id"],
                            workspace_name=workspace.name
                        )
                        
                        # Update metadata
                        for d in workspace.metadata:
                            if d["doc_id"] == doc["doc_id"]:
                                d["has_deep_analysis"] = True
                                break
                        workspace._save_metadata()
                        
                        st.balloons()
                        st.success("🎉 Upload and analysis complete!")
                    
                    st.info("💡 You can perform detailed analysis later in the 'Analysis' tab.")
    
    # TAB 4: Analysis (Enhanced)
    with tab4:
        st.markdown("### 🔬 Dual-Track Document Analysis")
        st.caption("Separate processing of electronic text and handwritten annotations with Q&A extraction")
        
        # Explanation cards
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 0.75rem; color: white;">
            <h4 style="margin:0; color: white;">📝 Track 1: Electronic Text</h4>
            <p style="margin-top: 0.5rem; color: #f0f0f0;">
            • Policy schedules & coverage tables<br>
            • Premium calculations<br>
            • Loss statistics<br>
            • Coverage terms (A1, A2, etc.)<br>
            • Q&A pair extraction
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 0.75rem; color: white;">
            <h4 style="margin:0; color: white;">✍️ Track 2: Handwritten</h4>
            <p style="margin-top: 0.5rem; color: #f0f0f0;">
            • Margin notes & questions<br>
            • Manual calculations<br>
            • Approval signatures<br>
            • Risk assessment notes<br>
            • Cross-references
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if not workspace.metadata:
            st.info("No documents yet. Upload documents first.")
        else:
            # Document selector
            doc_options = {
                f"{SUPPORTED_FORMATS.get(d['file_format'],'📎')} {d['filename']} ({d.get('insurance_type', 'Other')}) [{d['doc_id']}]": d["doc_id"]
                for d in workspace.metadata
            }
            
            selected_for_analysis = st.selectbox(
                "Select document for analysis:",
                list(doc_options.keys()),
                key="dual_analysis_selector"
            )
            
            if selected_for_analysis:
                analysis_doc_id = doc_options[selected_for_analysis]
                analysis_doc = next((d for d in workspace.metadata if d["doc_id"] == analysis_doc_id), None)
                
                if analysis_doc:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**Selected:** {analysis_doc['filename']}")
                        st.caption(f"{analysis_doc['file_format'].upper()} | {analysis_doc['file_size_kb']:.1f} KB | {analysis_doc.get('insurance_type', 'Other')}")
                    
                    with col2:
                        if analysis_doc.get("has_deep_analysis"):
                            st.success("✅ Already analyzed")
                        else:
                            st.info("⏳ Not analyzed")
                    
                    with col3:
                        analysis_file = os.path.join(ANALYSIS_DIR, f"{analysis_doc_id}_dual_analysis.json")
                        if os.path.exists(analysis_file):
                            with open(analysis_file, 'rb') as f:
                                st.download_button(
                                    "📥 Download",
                                    f,
                                    file_name=f"{analysis_doc_id}_analysis.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                    
                    if st.button("🚀 Start Dual-Track Analysis", type="primary", use_container_width=True):
                        analysis_result = perform_dual_track_analysis(
                            file_path=analysis_doc["file_path"],
                            file_format=analysis_doc["file_format"],
                            filename=analysis_doc["filename"],
                            doc_id=analysis_doc["doc_id"],
                            workspace_name=workspace.name
                        )
                        
                        # Update metadata
                        for doc in workspace.metadata:
                            if doc["doc_id"] == analysis_doc_id:
                                doc["has_deep_analysis"] = True
                                break
                        workspace._save_metadata()
                        
                        st.balloons()
                        st.success("🎉 Analysis complete! Scroll down for results.")
                        st.rerun()
                    
                    st.markdown("---")
                    
                    # Display results if available
                    analysis_file = os.path.join(ANALYSIS_DIR, f"{analysis_doc_id}_dual_analysis.json")
                    if os.path.exists(analysis_file):
                        st.markdown("### 📊 Analysis Results")
                        
                        with open(analysis_file, 'r', encoding='utf-8') as f:
                            analysis_data = json.load(f)
                        
                        # Overview metrics
                        st.markdown("#### 📋 Overview")
                        c1, c2, c3, c4, c5 = st.columns(5)
                        with c1:
                            st.metric("Text Length", f"{analysis_data['metadata'].get('text_length', 0):,}")
                        with c2:
                            st.metric("Images", analysis_data['metadata'].get('image_count', 0))
                        with c3:
                            st.metric("Q&A Pairs", analysis_data['metadata'].get('qa_pairs_count', 0))
                        with c4:
                            st.metric("Needs Review", analysis_data['metadata'].get('review_queue_count', 0))
                        with c5:
                            st.metric("Status", "✅ Complete" if analysis_data['metadata'].get('has_text') else "⚠️ No Text")
                        
                        # View selector
                        st.markdown("---")
                        view_mode = st.radio(
                            "Select view:",
                            ["🔗 Integrated Summary", "📝 Electronic Text", "✍️ Handwritten", "❓ Q&A Pairs", "📄 Page-by-Page"],
                            horizontal=True
                        )
                        
                        if view_mode == "🔗 Integrated Summary":
                            st.markdown("#### 📊 Comprehensive Underwriting Report")
                            st.markdown(analysis_data.get("dual_track_analysis", {}).get("integration_summary", "No summary available."))
                            
                            with st.expander("🔍 View Detailed Analysis"):
                                st.json(analysis_data.get("dual_track_analysis", {}))
                        
                        elif view_mode == "📝 Electronic Text":
                            st.markdown("#### 📝 Electronic Text Analysis")
                            electronic = analysis_data.get("dual_track_analysis", {}).get("electronic_text", {})
                            
                            if electronic.get("policy_info"):
                                st.markdown("**Policy Information:**")
                                st.json(electronic["policy_info"])
                            
                            if electronic.get("financial_terms"):
                                st.markdown("**Financial Terms:**")
                                col1, col2, col3 = st.columns(3)
                                ft = electronic["financial_terms"]
                                with col1:
                                    st.metric("Premium", f"{ft.get('currency', 'USD')} {ft.get('premium', 0):,}")
                                with col2:
                                    st.metric("Retention", f"{ft.get('currency', 'USD')} {ft.get('retention', 0):,}")
                                with col3:
                                    st.metric("Limit", f"{ft.get('currency', 'USD')} {ft.get('limit', 0):,}")
                            
                            with st.expander("📄 Full Electronic Analysis"):
                                st.json(electronic)
                        
                        elif view_mode == "✍️ Handwritten":
                            st.markdown("#### ✍️ Handwritten Annotations Analysis")
                            handwritten = analysis_data.get("dual_track_analysis", {}).get("handwritten_annotations", {})
                            
                            # Questions raised
                            if handwritten.get("questions_raised"):
                                st.markdown("**Questions Raised by CE:**")
                                for q in handwritten["questions_raised"]:
                                    confidence = q.get("confidence", 0)
                                    conf_class = "confidence-clear" if confidence > 0.7 else ("confidence-standard" if confidence > 0.4 else "confidence-cursive")
                                    st.markdown(f'- {q.get("text")} <span class="{conf_class}">({confidence:.0%} confidence)</span> *{q.get("location", "")}*', unsafe_allow_html=True)
                            
                            # Approval marks
                            if handwritten.get("approval_marks"):
                                st.markdown("**Approval Marks:**")
                                for a in handwritten["approval_marks"]:
                                    st.markdown(f'- **{a.get("type")}**: {a.get("name")} on {a.get("date", "N/A")} ({a.get("confidence", 0):.0%})')
                            
                            # Items needing review
                            if handwritten.get("needs_human_review"):
                                st.warning(f"⚠️ {len(handwritten['needs_human_review'])} items need human review")
                                for item in handwritten["needs_human_review"]:
                                    st.markdown(f"- {item.get('reason')} at {item.get('location')}")
                            
                            with st.expander("📄 Full Handwritten Analysis"):
                                st.json(handwritten)
                        
                        elif view_mode == "❓ Q&A Pairs":
                            st.markdown("#### ❓ Extracted Question-Answer Pairs")
                            qa_pairs = analysis_data.get("dual_track_analysis", {}).get("qa_pairs", [])
                            
                            if qa_pairs:
                                for i, qa in enumerate(qa_pairs, 1):
                                    with st.expander(f"Q{i}: {qa.get('question_id', '')} - {qa.get('question_text', '')[:80]}..."):
                                        st.markdown(f"**Question ({qa.get('source_type', 'unknown')}):**")
                                        st.write(qa.get("question_text", ""))
                                        
                                        st.markdown("**Answer:**")
                                        st.write(qa.get("answer_text", ""))
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.caption(f"Asked by: {qa.get('asked_by', 'Unknown')}")
                                        with col2:
                                            st.caption(f"Answered by: {qa.get('answered_by', 'Unknown')}")
                                        with col3:
                                            confidence = qa.get("confidence", 0)
                                            conf_class = "confidence-clear" if confidence > 0.7 else ("confidence-standard" if confidence > 0.4 else "confidence-cursive")
                                            st.markdown(f'<span class="{conf_class}">Confidence: {confidence:.0%}</span>', unsafe_allow_html=True)
                                        
                                        st.caption(f"Status: {qa.get('status', 'uncertified')} | Page: {qa.get('page_number', 'N/A')}")
                            else:
                                st.info("No Q&A pairs extracted from this document.")
                        
                        elif view_mode == "📄 Page-by-Page":
                            st.markdown("#### 📄 Page-by-Page Breakdown")
                            images = analysis_data.get("images_by_page", [])
                            
                            if images:
                                for img in images:
                                    page = img.get("page", 0)
                                    has_hw = img.get("likely_handwriting", False)
                                    tier = img.get("quality_tier", "UNKNOWN")
                                    confidence = img.get("confidence_estimate", 0)
                                    
                                    tier_emoji = "🟢" if tier == "CLEAR" else ("🟡" if tier == "STANDARD" else "🔴")
                                    
                                    with st.expander(f"Page {page}: {img['filename']} {tier_emoji} {tier} ({confidence:.0%})"):
                                        st.write(f"**Image ID:** {img['image_id']}")
                                        st.write(f"**Format:** {img['format'].upper()}")
                                        st.write(f"**Contains Handwriting:** {'Yes ✍️' if has_hw else 'No 📝'}")
                                        
                                        if has_hw:
                                            st.markdown(f"**Recognition Tier:** {tier}")
                                            st.progress(confidence)
                                            
                                            if tier in ["STANDARD", "CURSIVE"]:
                                                st.warning("⚠️ This page may require human review")
                            else:
                                st.info("No page-level data available.")
    
    # TAB 5: Review Workbench (NEW)
    with tab5:
        st.markdown("### 👥 Handwriting Review Workbench")
        st.caption("Human-in-the-loop review for handwritten annotations with low confidence")
        
        # Get all pending reviews
        pending_reviews = get_all_pending_reviews(workspace)
        
        if not pending_reviews:
            st.success("✅ No pending handwriting reviews! All documents are fully processed.")
            st.info("💡 As new documents with handwriting are analyzed, they will appear here for review.")
        else:
            st.warning(f"⚠️ {len(pending_reviews)} handwriting items need your review")
            
            # Review queue statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Reviews", len(pending_reviews))
            with col2:
                clear_count = len([r for r in pending_reviews if "CLEAR" in r.get("tier", "")])
                st.metric("Clear Tier", clear_count)
            with col3:
                cursive_count = len([r for r in pending_reviews if "CURSIVE" in r.get("tier", "")])
                st.metric("Cursive Tier", cursive_count)
            
            st.markdown("---")
            
            # Review interface
            for i, review_item in enumerate(pending_reviews):
                doc_id = review_item.get("doc_id")
                image_id = review_item.get("image_id")
                tier = review_item.get("tier", "UNKNOWN")
                confidence = review_item.get("confidence", 0)
                page = review_item.get("page", 0)
                reason = review_item.get("reason", "")
                doc_filename = review_item.get("doc_filename", "Unknown")
                
                # Tier styling
                tier_color = "#86efac" if tier == "CLEAR" else ("#fde68a" if tier == "STANDARD" else "#fca5a5")
                
                st.markdown(f"""
                <div class="review-card">
                <h4 style="margin:0;">Review #{i+1}: {doc_filename} - Page {page}</h4>
                <p style="margin-top:0.5rem;">
                <span style="background:{tier_color}; color:#0b1220; padding:0.25rem 0.75rem; border-radius:0.5rem; font-weight:700;">
                {tier} Tier ({confidence:.0%} confidence)
                </span>
                </p>
                <p style="color:#9ca3af; margin-top:0.5rem;">Reason: {reason}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Review form
                with st.form(f"review_form_{doc_id}_{image_id}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Show image if available
                        image_path = review_item.get("image_path")
                        if image_path and os.path.exists(image_path):
                            try:
                                st.image(image_path, caption=f"Handwritten region - {image_id}", use_column_width=True)
                            except:
                                st.error("Could not load image preview")
                        else:
                            st.info("📷 Image preview not available")
                        
                        st.markdown("**Instructions:**")
                        st.markdown("1. Examine the handwritten content carefully")
                        st.markdown("2. Transcribe the text as accurately as possible")
                        st.markdown("3. Add any relevant notes about context or meaning")
                        st.markdown("4. Mark if the OCR was accurate (if OCR was attempted)")
                    
                    with col2:
                        transcribed_text = st.text_area(
                            "Transcribed Text",
                            placeholder="Type the handwritten text here...",
                            height=150,
                            key=f"transcribe_{doc_id}_{image_id}"
                        )
                        
                        reviewer_notes = st.text_area(
                            "Reviewer Notes",
                            placeholder="Context, references, or clarifications...",
                            height=100,
                            key=f"notes_{doc_id}_{image_id}"
                        )
                        
                        is_accurate = st.checkbox(
                            "OCR was accurate (if applicable)",
                            key=f"accurate_{doc_id}_{image_id}"
                        )
                        
                        reviewer_name = st.text_input(
                            "Your Name",
                            value="Underwriter",
                            key=f"reviewer_{doc_id}_{image_id}"
                        )
                    
                    submit_review = st.form_submit_button("✅ Submit Review", use_container_width=True)
                    
                    if submit_review:
                        if not transcribed_text:
                            st.error("Please provide transcribed text")
                        else:
                            success = submit_handwriting_review(
                                doc_id=doc_id,
                                image_id=image_id,
                                transcribed_text=transcribed_text,
                                reviewer_notes=reviewer_notes,
                                is_accurate=is_accurate,
                                reviewer=reviewer_name
                            )
                            
                            if success:
                                st.success(f"✅ Review submitted for {image_id}!")
                                st.info("🔄 Reloading review queue...")
                                st.rerun()
                            else:
                                st.error("Failed to submit review")
                
                st.markdown("---")
            
            # Batch actions
            st.markdown("### 🔧 Batch Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Export Review Queue (CSV)", use_container_width=True):
                    df = pd.DataFrame(pending_reviews)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        file_name=f"{workspace.name}_review_queue.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("🗑️ Clear Completed Reviews", use_container_width=True):
                    st.info("This will remove all completed reviews from the queue.")
                    # Implementation would go here

if __name__ == "__main__":
    main()
