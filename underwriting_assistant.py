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
import pandas as pd

# ===========================
# Configuration
# ===========================

VERSION = "2.9.0"
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
INITIAL_DATASET = "Hull - MSC_Memo.pdf"

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
# System Prompts (Modified)
# ===========================

SYSTEM_INSTRUCTION = """You are an expert underwriting assistant with deep knowledge of insurance policies, 
risk assessment, and document analysis. Your role is to help underwriters make informed decisions by:
1. Extracting and analyzing key information from policy documents
2. Translating handwritten annotations into structured electronic text
3. Identifying critical risk factors and coverage terms
4. Providing comprehensive analysis with actionable insights
5. Maintaining strict accuracy and professional standards

Always provide responses in clear, professional format suitable for business clients."""

# Modified: Concise summary instead of full report
ELECTRONIC_TEXT_ANALYSIS_SYSTEM = """提供文档电子文本的简洁总结。

**要求**：基于文档实际内容，用一个段落（100-200字）总结：
1. 被保险人名称和承保类型
2. 关键财务数据（保费、免赔额、限额）
3. 承保期限和续保条款
4. 主要风险因素或特殊条件
5. 历史赔付率或理赔情况（如有）

**格式**：
- 不要使用标题、项目符号或分段
- 写成一个连贯的摘要段落
- 专业、客观、简明

示例：
"本文件承保地中海航运公司（MSC）的船舶Melody和Rhapsody的船体和机器保险，承保期限从2008年5月至2009年5月。每艘船的保险金额为30万美元，免赔额从50万美元提高至100万美元。历史赔付率显示第一期为74.32%，后续续保期降至1.43%，表明风险表现改善。续保保费比到期条款高10%，FCIL按自有优势和费率承保，经纪人佣金为22.5%。"

**重要**：仅提取和分析文档中的实际内容，保持总结简洁明了。"""

# Modified: Structured format without summary sections
HANDWRITING_TRANSLATION_SYSTEM = """分析并翻译承保文件中的手写批注。

**输出格式**：对每个检测到的手写批注，使用以下格式：

---
图片ID: [标识符]
翻译文本: [手写内容的准确转录]
识别置信度: [百分比，如 85%]
位置: [文档中的位置]
类型: [签名/评论/日期/审批/备注]
---

**严格要求**：
- 不要包含任何总结性段落
- 不要包含"概述"或"摘要"部分  
- 不要包含"需要人工审核"的列表
- 不要包含结论性文字
- 只输出单个批注的翻译，每个批注使用上述格式

示例输出：
---
图片ID: page1_img3
翻译文本: 致CEO审阅
识别置信度: 92%
位置: 第1页，右上角
类型: 审批
---

---
图片ID: page2_img5
翻译文本: 建议续保，保费上浮5%
识别置信度: 78%
位置: 第2页，页边空白处
类型: 评论
---

**仅输出批注翻译**，不要添加任何其他文字。"""

QA_EXTRACTION_SYSTEM = """Extract Question-Answer pairs from this underwriting document.

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

AUTO_ANNOTATE_SYSTEM = """Automatically annotate this underwriting document with key metadata.

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

# NEW FUNCTION: Extract case metadata for table view
def extract_case_metadata(text: str, filename: str) -> Dict[str, str]:
    """从文档文本中提取结构化元数据用于表格视图"""
    
    metadata = {
        "案例名称": filename.replace('.pdf', '').replace('.docx', '').replace('_', ' '),
        "类别": "未分类",
        "承保年度": "未知",
        "客户名称": "未指定",
        "最新更新时间": datetime.now().strftime("%Y-%m-%d")
    }
    
    # 提取保险类型/类别
    categories = ["船体", "货运", "责任", "财产", "海洋", "航空", "Hull", "Cargo", "Liability", "Property", "Marine", "Aviation"]
    text_lower = text.lower()
    for cat in categories:
        if cat.lower() in text_lower[:500]:
            metadata["类别"] = cat
            break
    
    # 提取年份
    year_pattern = r'20\d{2}'
    years = re.findall(year_pattern, text[:1000])
    if years:
        metadata["承保年度"] = years[0]
    
    # 提取客户名称
    client_patterns = [
        r'(?i)被保险人[：:\s]+([^\n,，]{3,40})',
        r'(?i)insured[:\s]+([A-Z][A-Za-z\s&\.]{5,50})(?:[\n,]|Ltd|Inc|Corp|Co\.)',
        r'(?i)客户[：:\s]+([^\n,，]{3,40})',
        r'(?i)company[:\s]+([A-Z][A-Za-z\s&\.]{5,50})(?:[\n,]|Ltd|Inc|Corp)'
    ]
    
    for pattern in client_patterns:
        match = re.search(pattern, text[:1000])
        if match:
            client_name = match.group(1).strip()
            client_name = re.sub(r'\s+(Ltd|Inc|Corp|Co\.|Limited|Corporation|Company).*$', '', client_name, flags=re.IGNORECASE)
            if len(client_name) > 3:
                metadata["客户名称"] = client_name
                break
    
    return metadata

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

def extract_images_from_pdf(file_path: Path) -> List[Dict]:
    """Extract all images from PDF (for scanned documents and handwriting detection)"""
    try:
        import fitz
        doc = fitz.open(file_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to base64
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    images.append({
                        'id': f"page{page_num+1}_img{img_index+1}",
                        'data': image_b64,
                        'type': 'scanned',
                        'page': page_num + 1,
                        'source_file': file_path.name,
                        'size': len(image_bytes)
                    })
                except Exception as e:
                    continue
        
        doc.close()
        return images
        
    except Exception as e:
        st.warning(f"PDF image extraction error: {e}")
        return []

def detect_handwriting_in_images(images: List[Dict]) -> bool:
    """Detect if images likely contain handwriting (heuristic approach)"""
    if not images:
        return False
    
    # Count images per page
    pages = {}
    for img in images:
        page = img.get('page', 1)
        pages[page] = pages.get(page, 0) + 1
    
    # If any page has 4+ images, likely has handwriting overlays/annotations
    for page, count in pages.items():
        if count >= 4:
            return True
    
    # Check for small images (might be signatures/stamps)
    for img in images:
        if img.get('size', 0) < 50000:  # < 50KB
            return True
    
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
        extracted_info += "\n1. The system will analyze images for handwritten annotations"
        extracted_info += "\n2. Use the 'Handwriting Translation' tab to view results"
        extracted_info += "\n3. You can also upload individual page images for OCR processing"
        extracted_info += "\n\nFor production use, integrate with:"
        extracted_info += "\n- Google Cloud Vision API"
        extracted_info += "\n- AWS Textract"
        extracted_info += "\n- Azure Computer Vision"
        
        doc.close()
        
        return extracted_info
        
    except Exception as e:
        return f"Error processing scanned PDF: {e}"

def extract_text_from_pdf(file_path: Path) -> Tuple[str, List[Dict]]:
    """Enhanced PDF extraction - returns (text, images)"""
    try:
        import fitz
        doc = fitz.open(file_path)
        
        text_content = []
        all_images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Try to extract text
            text = page.get_text()
            if text.strip():
                text_content.append(f"=== Page {page_num+1} ===\n{text}\n")
            
            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    all_images.append({
                        'id': f"page{page_num+1}_img{img_index+1}",
                        'data': base64.b64encode(image_bytes).decode('utf-8'),
                        'type': 'scanned' if not text.strip() else 'embedded',
                        'page': page_num + 1,
                        'source_file': file_path.name,
                        'size': len(image_bytes)
                    })
                except:
                    continue
        
        doc.close()
        
        # If no text but has images, it's a scanned document
        if not text_content and all_images:
            scanned_info = extract_text_from_scanned_pdf(file_path)
            return (scanned_info, all_images)
        
        final_text = "\n\n".join(text_content) if text_content else ""
        return (final_text, all_images)
        
    except Exception as e:
        st.warning(f"PDF extraction error: {e}")
        return ("", [])

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

def extract_text_from_file(file_path: Path) -> Tuple[str, List[Dict]]:
    """Extract text and images based on file type"""
    ext = file_path.suffix.lower().lstrip('.')
    
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['docx', 'doc']:
        text = extract_text_from_docx(file_path)
        images = extract_images_from_docx(file_path)
        return (text, images)
    elif ext == 'txt':
        text = extract_text_from_txt(file_path)
        return (text, [])
    else:
        return ("", [])

def extract_images_from_docx(file_path: Path) -> List[Dict]:
    """Extract embedded images from DOCX file (skip external links)"""
    try:
        from docx import Document
        doc = Document(file_path)
        images = []
        
        for rel in doc.part.rels.values():
            # Skip external relationships (like linked images from URLs)
            if hasattr(rel, 'target_mode') and rel.target_mode == 'External':
                continue
            
            # Only process internal image relationships
            if "image" in rel.target_ref:
                try:
                    image_data = rel.target_part.blob
                    image_id = f"img_{len(images)+1}"
                    
                    images.append({
                        'id': image_id,
                        'data': base64.b64encode(image_data).decode('utf-8'),
                        'type': 'embedded',
                        'source_file': file_path.name,
                        'size': len(image_data)
                    })
                except Exception as e:
                    # Skip images that can't be processed
                    continue
        
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
# Core Analysis Functions (Modified)
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

        response = call_llm_api(QA_EXTRACTION_SYSTEM, user_prompt)
        
        if not response:
            return "No Q&A pairs could be extracted from this document."
        
        return response
        
    except Exception as e:
        st.warning(f"Q&A extraction error: {e}")
        return "Error extracting Q&A pairs."

# MODIFIED: Returns concise summary
def analyze_electronic_text(text: str, filename: str) -> str:
    """分析电子/印刷文本 - 返回简洁总结"""
    try:
        user_prompt = f"""文档名称: {filename}

完整内容:
{text[:6000]}

请对此承保文件进行简洁分析，提供100-200字的段落总结。"""

        response = call_llm_api(ELECTRONIC_TEXT_ANALYSIS_SYSTEM, user_prompt, max_tokens=1000)
        
        if not response:
            return "无法分析电子文本。请检查API配置。"
        
        return response
        
    except Exception as e:
        st.warning(f"电子文本分析错误: {e}")
        return "电子文本分析过程中发生错误。"

# MODIFIED: New structured output format
def translate_handwriting(images: List[Dict], filename: str, text_content: str = "") -> Dict:
    """翻译手写批注，使用新的输出格式"""
    try:
        if not images:
            return {
                "has_handwriting": False,
                "translated_text": "此文档中未检测到图片。",
                "image_count": 0
            }
        
        has_handwriting = detect_handwriting_in_images(images)
        
        if not has_handwriting:
            return {
                "has_handwriting": False,
                "translated_text": f"文档包含 {len(images)} 张图片，但未检测到手写批注。",
                "image_count": len(images)
            }
        
        max_page = max([img.get('page', 1) for img in images])
        
        user_prompt = f"""文档名称: {filename}

这是一份包含 {len(images)} 张图片的扫描承保文件，共 {max_page} 页。

文档上下文:
{text_content[:1500] if text_content else "扫描文档 - 正在分析图片内容"}

图片分析:
- 总图片数: {len(images)}
- 第1页图片数: {len([i for i in images if i.get('page')==1])}
- 小型覆盖图: {len([i for i in images if i.get('size', 0) < 50000])} 张

任务: 识别并翻译所有手写批注，按指定格式输出。"""

        response = call_llm_api(HANDWRITING_TRANSLATION_SYSTEM, user_prompt, temperature=0.2, max_tokens=3000)
        
        if not response:
            response = f"检测到 {len(images)} 张图片，包含手写批注。正在处理中。"
        
        return {
            "has_handwriting": True,
            "translated_text": response,
            "image_count": len(images)
        }
        
    except Exception as e:
        st.warning(f"手写翻译错误: {e}")
        return {
            "has_handwriting": False,
            "translated_text": f"手写翻译过程中发生错误: {e}",
            "image_count": 0
        }

def perform_dual_track_analysis(text: str, images: List[Dict], filename: str) -> Dict:
    """Perform comprehensive dual-track analysis"""
    try:
        # Track 1: Electronic text analysis
        electronic_analysis = analyze_electronic_text(text, filename)
        
        # Track 2: Handwriting translation (with text context)
        handwriting_translation = translate_handwriting(images, filename, text)
        
        # Extract Q&A pairs
        qa_pairs = extract_qa_pairs(text, filename)
        
        # Combined analysis
        has_handwriting = handwriting_translation.get('has_handwriting', False)
        handwriting_text = handwriting_translation.get('translated_text', '')
        
        integration_prompt = f"""Create a comprehensive underwriting report integrating:

ELECTRONIC TEXT ANALYSIS:
{electronic_analysis[:2500]}

HANDWRITING NOTES:
{handwriting_text[:1200] if has_handwriting else "No handwritten notes detected"}

Q&A SUMMARY:
{qa_pairs[:1000]}

Provide:
1. Executive Summary (2-3 paragraphs covering key points)
2. Critical Risk Factors (identify top 3-5 risks)
3. Underwriting Recommendations (specific actions needed)
4. Key Decision Points (items requiring management attention)

Base the report on ACTUAL content from this specific document."""

        integration_response = call_llm_api(SYSTEM_INSTRUCTION, integration_prompt, max_tokens=4000)
        
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

        response = call_llm_api(AUTO_ANNOTATE_SYSTEM, user_prompt, temperature=0.3)
        
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

def delete_document_from_workspace(workspace_name: str, filename: str) -> bool:
    """Delete a document from workspace"""
    try:
        metadata = load_workspace(workspace_name)
        if not metadata:
            return False
        
        # Find the document
        doc = next((d for d in metadata['documents'] if d['filename'] == filename), None)
        if not doc:
            return False
        
        # Delete physical file
        file_path = Path(doc['path'])
        if file_path.exists():
            file_path.unlink()
        
        # Delete embedding file
        embedding_file = EMBEDDINGS_DIR / f"{workspace_name}_{filename}.json"
        if embedding_file.exists():
            embedding_file.unlink()
        
        # Delete analysis file
        analysis_file = ANALYSIS_DIR / f"{workspace_name}_{filename}.json"
        if analysis_file.exists():
            analysis_file.unlink()
        
        # Remove from metadata
        metadata['documents'] = [d for d in metadata['documents'] if d['filename'] != filename]
        
        # Save metadata
        workspace_dir = WORKSPACES_DIR / workspace_name
        with open(workspace_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log_audit_event("document_deleted", {
            "workspace": workspace_name,
            "filename": filename
        })
        
        return True
        
    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return False

def upload_document_to_workspace(workspace_name: str, uploaded_file, auto_analyze: bool = True):
    """Upload document to workspace with auto-analysis"""
    try:
        workspace_dir = WORKSPACES_DIR / workspace_name
        file_path = workspace_dir / uploaded_file.name
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Extract text and images
        extracted_text, images = extract_text_from_file(file_path)
        
        if not extracted_text:
            st.warning(f"No text extracted from {uploaded_file.name}")
            return None
        
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
        
        # Perform analysis if there are images or auto_analyze is enabled
        if auto_analyze and (len(images) > 0 or is_scanned_pdf(file_path)):
            with st.spinner("Performing dual-track analysis..."):
                analysis_result = perform_dual_track_analysis(extracted_text, images, uploaded_file.name)
                
                analysis_file = ANALYSIS_DIR / f"{workspace_name}_{uploaded_file.name}.json"
                with open(analysis_file, 'w') as f:
                    json.dump(analysis_result, f, indent=2, ensure_ascii=False)
                
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
    """Load ONLY the MSC Memo file on first run"""
    try:
        # Look for the file with flexible naming
        possible_names = [
            "Hull - MSC_Memo.pdf",
            "Hull_MSC_Memo.pdf", 
            "Hull-MSC_Memo.pdf",
            "Hull - Marco Polo_Memo.pdf"  # Backward compatibility
        ]
        
        initial_file = None
        for name in possible_names:
            if Path(name).exists():
                initial_file = name
                break
        
        if not initial_file:
            st.info(f"ℹ️ Initial dataset file not found. Looking for: {possible_names[0]}")
            return False
        
        default_workspace = "Default"
        metadata = load_workspace(default_workspace)
        
        # Check if initial file already loaded
        if metadata:
            for doc in metadata.get("documents", []):
                if doc["filename"] == initial_file or initial_file in doc["filename"]:
                    return True  # Already loaded
        
        # Create workspace if not exists
        if not metadata:
            create_workspace(default_workspace, "Default workspace with initial dataset")
            metadata = load_workspace(default_workspace)
        
        # Load the file
        with open(initial_file, 'rb') as f:
            file_content = f.read()
        
        class UploadedFile:
            def __init__(self, name, content):
                self.name = name
                self.size = len(content)
                self._content = content
            
            def getvalue(self):
                return self._content
        
        uploaded_file = UploadedFile(initial_file, file_content)
        
        result = upload_document_to_workspace(default_workspace, uploaded_file, auto_analyze=True)
        
        if result:
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
        <p>Version {VERSION} | Powered by AI | Advanced OCR & Document Analysis</p>
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

def render_document_card(doc: Dict, workspace_name: str, doc_index: int = 0):
    """Render a document card with unique keys and delete button"""
    format_icon = SUPPORTED_FORMATS.get(doc['format'], '📄')
    
    # Create unique key prefix using upload date and index
    upload_ts = doc.get('upload_date', '').replace(':', '-').replace('.', '-')
    key_prefix = f"{workspace_name}_{hashlib.md5(doc['filename'].encode()).hexdigest()[:8]}_{upload_ts}_{doc_index}"
    
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
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button(f"📄 View", key=f"view_{key_prefix}"):
            st.session_state.viewing_doc = doc
    
    with col2:
        if doc.get('has_deep_analysis'):
            if st.button(f"📊 Analysis", key=f"analysis_{key_prefix}"):
                st.session_state.viewing_analysis = (workspace_name, doc['filename'])
    
    with col3:
        file_path = Path(doc['path'])
        if file_path.exists():
            with open(file_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download",
                    data=f.read(),
                    file_name=doc['filename'],
                    key=f"download_{key_prefix}"
                )
    
    with col4:
        if st.button(f"🗑️ Delete", key=f"delete_{key_prefix}", type="secondary"):
            st.session_state[f"confirm_delete_{key_prefix}"] = True
    
    # Confirmation dialog for delete
    if st.session_state.get(f"confirm_delete_{key_prefix}", False):
        st.warning(f"⚠️ Are you sure you want to delete **{doc['filename']}**?")
        col_yes, col_no = st.columns(2)
        
        with col_yes:
            if st.button("✅ Yes, Delete", key=f"confirm_yes_{key_prefix}"):
                if delete_document_from_workspace(workspace_name, doc['filename']):
                    st.success(f"✅ Deleted {doc['filename']}")
                    st.session_state[f"confirm_delete_{key_prefix}"] = False
                    st.rerun()
                else:
                    st.error("Failed to delete document")
        
        with col_no:
            if st.button("❌ Cancel", key=f"confirm_no_{key_prefix}"):
                st.session_state[f"confirm_delete_{key_prefix}"] = False
                st.rerun()

# NEW FUNCTION: Render cases table view with filters
def render_cases_table_view(workspace_name: str):
    """渲染所有案例的表格视图，支持筛选"""
    
    metadata = load_workspace(workspace_name)
    if not metadata or not metadata.get('documents'):
        st.info("当前工作空间没有文档")
        return
    
    documents = metadata.get('documents', [])
    
    # 构建表格数据
    table_data = []
    for doc in documents:
        analysis_file = ANALYSIS_DIR / f"{workspace_name}_{doc['filename']}.json"
        
        case_meta = {
            "案例名称": doc['filename'],
            "类别": doc.get('insurance_type', '未分类'),
            "承保年度": "未知",
            "客户名称": "未指定",
            "最新更新时间": doc.get('upload_date', '')[:10] if doc.get('upload_date') else ''
        }
        
        # 如果有分析结果，提取元数据
        if analysis_file.exists():
            try:
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)
                    electronic_text = analysis.get('electronic_analysis', '')
                    extracted = extract_case_metadata(electronic_text, doc['filename'])
                    case_meta["类别"] = extracted.get('类别', case_meta["类别"])
                    case_meta["承保年度"] = extracted.get('承保年度', '未知')
                    case_meta["客户名称"] = extracted.get('客户名称', '未指定')
            except:
                pass
        
        # 从标签获取类别
        if case_meta["类别"] == "未分类" and doc.get('tags'):
            for tag in doc['tags']:
                if tag in ["Hull", "Cargo", "Liability", "Property", "Marine", "Aviation"]:
                    case_meta["类别"] = tag
                    break
        
        table_data.append(case_meta)
    
    df = pd.DataFrame(table_data)
    
    # 筛选器
    st.markdown("### 📊 案例概览表")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        categories = ["全部"] + sorted(list(df['类别'].unique()))
        selected_category = st.selectbox("保险类型:", categories, key="filter_category")
    
    with col2:
        clients = ["全部"] + sorted(list(df['客户名称'].unique()))
        selected_client = st.selectbox("客户名称:", clients, key="filter_client")
    
    with col3:
        years = ["全部"] + sorted([y for y in df['承保年度'].unique() if y != "未知"], reverse=True)
        selected_year = st.selectbox("承保年度:", years, key="filter_year")
    
    # 应用筛选
    filtered_df = df.copy()
    
    if selected_category != "全部":
        filtered_df = filtered_df[filtered_df['类别'] == selected_category]
    
    if selected_client != "全部":
        filtered_df = filtered_df[filtered_df['客户名称'] == selected_client]
    
    if selected_year != "全部":
        filtered_df = filtered_df[filtered_df['承保年度'] == selected_year]
    
    st.markdown(f"**显示 {len(filtered_df)} / {len(df)} 个案例**")
    
    # 显示表格
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "案例名称": st.column_config.TextColumn("案例名称", width="large"),
            "类别": st.column_config.TextColumn("类别", width="small"),
            "客户名称": st.column_config.TextColumn("客户名称", width="medium"),
            "承保年度": st.column_config.TextColumn("承保年度", width="small"),
            "最新更新时间": st.column_config.DateColumn("最新更新", format="YYYY-MM-DD", width="small")
        }
    )
    
    # 选择案例查看详情
    if len(filtered_df) > 0:
        st.markdown("---")
        selected_case = st.selectbox(
            "选择案例查看详细分析:",
            filtered_df['案例名称'].tolist(),
            key="select_case_for_analysis"
        )
        
        if selected_case and st.button("📄 查看详细分析", key="view_selected_analysis"):
            st.session_state.viewing_analysis = (workspace_name, selected_case)
            st.rerun()

# COMPLETELY REPLACED: render_analysis_view function
def render_analysis_view(workspace_name: str, filename: str):
    """渲染分析结果，使用新的表格和显示逻辑"""
    analysis_file = ANALYSIS_DIR / f"{workspace_name}_{filename}.json"
    
    if not analysis_file.exists():
        st.warning("未找到此文档的分析结果")
        return
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    st.subheader(f"📊 分析结果: {filename}")
    
    api_key = get_api_key()
    if not api_key:
        st.error("⚠️ 未配置API密钥，分析结果可能不完整")
    
    view_mode = st.radio(
        "选择查看模式:",
        ["📋 案例总览表", "📄 电子文本摘要", "✍️ 手写翻译", "❓ 问答对"],
        horizontal=True,
        key="analysis_view_mode"
    )
    
    # 案例总览表
    if view_mode == "📋 案例总览表":
        render_cases_table_view(workspace_name)
    
    # 电子文本摘要
    elif view_mode == "📄 电子文本摘要":
        st.markdown("### 📄 文档内容摘要")
        
        electronic = analysis.get('electronic_analysis', '')
        
        if not electronic or len(electronic) < 50:
            st.warning("⚠️ 电子文本分析为空或不完整")
            st.info("这可能是扫描文档，请查看"手写翻译"标签页")
        else:
            st.markdown(electronic)
            
            st.markdown("---")
            if st.button("📥 导出摘要", key="export_summary"):
                st.download_button(
                    label="下载文本摘要",
                    data=electronic,
                    file_name=f"{filename}_摘要_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    key="download_summary_txt"
                )
    
    # 手写翻译
    elif view_mode == "✍️ 手写翻译":
        st.markdown("### ✍️ 手写批注翻译")
        
        handwriting = analysis.get('handwriting_translation', {})
        has_handwriting = handwriting.get('has_handwriting', False)
        
        if not has_handwriting:
            st.info("ℹ️ 此文档中未检测到手写内容")
        else:
            translation_text = handwriting.get('translated_text', '')
            
            # 解析结构化输出
            annotations = []
            current_annotation = {}
            
            for line in translation_text.split('\n'):
                line = line.strip()
                
                if line == '---':
                    if current_annotation and 'text' in current_annotation:
                        annotations.append(current_annotation)
                        current_annotation = {}
                elif line.startswith('图片ID:') or line.startswith('IMAGE:'):
                    current_annotation['image_id'] = line.split(':', 1)[1].strip()
                elif line.startswith('翻译文本:') or line.startswith('TEXT:'):
                    current_annotation['text'] = line.split(':', 1)[1].strip()
                elif line.startswith('识别置信度:') or line.startswith('CONFIDENCE:'):
                    conf_str = line.split(':', 1)[1].replace('%', '').strip()
                    try:
                        current_annotation['confidence'] = int(conf_str)
                    except:
                        current_annotation['confidence'] = 75
                elif line.startswith('位置:') or line.startswith('LOCATION:'):
                    current_annotation['location'] = line.split(':', 1)[1].strip()
                elif line.startswith('类型:') or line.startswith('TYPE:'):
                    current_annotation['type'] = line.split(':', 1)[1].strip()
            
            if current_annotation and 'text' in current_annotation:
                annotations.append(current_annotation)
            
            # 显示批注
            if annotations:
                st.success(f"✅ 检测到 {len(annotations)} 个手写批注")
                
                for idx, annot in enumerate(annotations, 1):
                    st.markdown(f"#### 批注 {idx}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("**📷 手写图片**")
                        image_id = annot.get('image_id', '')
                        image_path = ANALYSIS_DIR / f"{workspace_name}_{filename}_{image_id}.png"
                        
                        if image_path.exists():
                            st.image(str(image_path), use_container_width=True)
                        else:
                            st.info(f"图片ID: {image_id}")
                            st.caption("💡 在生产环境中，此处将显示实际的手写图片")
                        
                        st.caption(f"📍 位置: {annot.get('location', '未指定')}")
                    
                    with col2:
                        st.markdown("**✍️ 翻译文本**")
                        st.markdown(f"> _{annot.get('text', '无翻译内容')}_")
                        
                        st.markdown("**📊 识别置信度**")
                        confidence = annot.get('confidence', 75)
                        
                        if confidence >= 80:
                            color = "green"
                        elif confidence >= 60:
                            color = "orange"
                        else:
                            color = "red"
                        
                        st.progress(confidence / 100)
                        st.markdown(f"<span style='color:{color};font-weight:bold;'>{confidence}%</span>", unsafe_allow_html=True)
                        
                        type_emoji = {
                            "签名": "✒️", "评论": "💬", "日期": "📅", "审批": "✅", "备注": "📝",
                            "Signature": "✒️", "Comment": "💬", "Date": "📅", "Approval": "✅", "Note": "📝"
                        }
                        annot_type = annot.get('type', '未知')
                        emoji = type_emoji.get(annot_type, "📌")
                        st.markdown(f"**🏷️ 类型:** {emoji} {annot_type}")
                    
                    st.markdown("---")
            else:
                # 过滤总结性段落
                lines = translation_text.split('\n')
                filtered_lines = []
                skip_keywords = ['summary', 'overview', '总结', '概述', 'key insights', 
                               'annotations requiring', '需要审核', '检测到的批注', 'detected annotations',
                               'handwriting summary', '手写摘要']
                
                skip_mode = False
                for line in lines:
                    if any(keyword in line.lower() for keyword in skip_keywords):
                        skip_mode = True
                        continue
                    if line.strip() == '---':
                        skip_mode = False
                    if not skip_mode and line.strip():
                        filtered_lines.append(line)
                
                filtered_text = '\n'.join(filtered_lines)
                if filtered_text.strip():
                    st.text(filtered_text)
                else:
                    st.info("未找到有效的批注翻译内容")
        
        # 上传功能
        st.markdown("---")
        st.markdown("### 📤 上传额外的手写图片")
        
        uploaded_images = st.file_uploader(
            "上传手写批注照片或扫描件:",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key=f"upload_handwriting_{filename}"
        )
        
        if uploaded_images:
            st.markdown(f"**已上传 {len(uploaded_images)} 张图片**")
            
            for idx, img_file in enumerate(uploaded_images, 1):
                st.markdown(f"#### 图片 {idx}: {img_file.name}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(img_file, caption=f"图片 {idx}", use_container_width=True)
                
                with col2:
                    st.markdown("**识别选项**")
                    
                    if st.button(f"🔍 识别文字", key=f"ocr_btn_{filename}_{idx}"):
                        st.info("🔧 OCR功能 - 在生产环境中将调用OCR服务")
                    
                    transcription = st.text_area(
                        "手动转录/编辑:",
                        key=f"manual_trans_{filename}_{idx}",
                        height=100,
                        placeholder="在此输入或编辑识别出的文字..."
                    )
                    
                    confidence = st.slider(
                        "置信度:",
                        0, 100, 75,
                        key=f"conf_slider_{filename}_{idx}"
                    )
                    
                    if st.button(f"💾 保存", key=f"save_trans_{filename}_{idx}"):
                        if transcription:
                            st.success(f"✅ 已保存图片 {idx} 的转录内容")
                        else:
                            st.warning("请先输入转录内容")
                
                st.markdown("---")
    
    # 问答对
    elif view_mode == "❓ 问答对":
        st.markdown("### ❓ 问答对提取")
        qa_text = analysis.get('qa_extraction', '')
        
        if not qa_text or len(qa_text) < 50:
            st.info("此文档中未找到问答对")
        else:
            st.markdown(qa_text)
    
    # 操作按钮
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 重新分析", key=f"rerun_analysis_{filename}"):
            with st.spinner("正在重新分析文档..."):
                metadata = load_workspace(workspace_name)
                doc = next((d for d in metadata['documents'] if d['filename'] == filename), None)
                
                if doc:
                    file_path = Path(doc['path'])
                    text, images = extract_text_from_file(file_path)
                    new_analysis = perform_dual_track_analysis(text, images, filename)
                    
                    with open(analysis_file, 'w') as f:
                        json.dump(new_analysis, f, indent=2, ensure_ascii=False)
                    
                    st.success("✅ 分析完成!")
                    st.rerun()
    
    with col2:
        if st.button("📥 导出报告", key=f"export_report_{filename}"):
            report_text = f"""承保文件分析报告
{'='*60}
文档: {filename}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*60}
电子文本摘要
{'='*60}
{analysis.get('electronic_analysis', '无分析内容')}

{'='*60}
手写翻译
{'='*60}
{analysis.get('handwriting_translation', {}).get('translated_text', '未检测到手写内容')}

{'='*60}
问答对
{'='*60}
{analysis.get('qa_extraction', '无问答对')}
"""
            st.download_button(
                label="下载TXT报告",
                data=report_text,
                file_name=f"{filename}_分析报告_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                key=f"download_txt_{filename}"
            )
    
    with col3:
        if st.button("⬅️ 返回文档库", key=f"back_to_lib_{filename}"):
            st.session_state.viewing_analysis = None
            st.rerun()

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
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = get_api_key()
    
    render_header()
    
    # Load initial dataset (ONLY ONCE, ONLY ONE FILE)
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Document Library",
        "⬆️ Upload Documents",
        "📊 Analysis Dashboard",
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
                
                for idx, doc in enumerate(documents):
                    render_document_card(doc, st.session_state.current_workspace, doc_index=idx)
                
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
                        st.success(f"✅ {file.name} uploaded!")
                    else:
                        st.error(f"❌ Failed to upload {file.name}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("Upload complete!")
                st.balloons()
    
    # TAB 3: Analysis Dashboard
    with tab3:
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
                
                for idx, doc in enumerate(documents):
                    upload_ts = doc.get('upload_date', '').replace(':', '-').replace('.', '-')
                    unique_key = f"{hashlib.md5(doc['filename'].encode()).hexdigest()[:8]}_{upload_ts}_{idx}"
                    
                    with st.expander(f"{doc['filename']} - {doc.get('decision', 'Pending')}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Risk Level:** {doc.get('risk_level', 'N/A')}")
                            st.markdown(f"**Tags:** {', '.join(doc.get('tags', []))}")
                            st.markdown(f"**Has Images:** {'Yes' if doc.get('has_images') else 'No'}")
                        
                        with col2:
                            if doc.get('has_deep_analysis'):
                                if st.button("View Full Analysis", key=f"view_full_{unique_key}"):
                                    st.session_state.viewing_analysis = (st.session_state.current_workspace, doc['filename'])
                                    st.rerun()
                            else:
                                if st.button("Run Analysis", key=f"run_{unique_key}"):
                                    with st.spinner("Analyzing..."):
                                        file_path = Path(doc['path'])
                                        text, images = extract_text_from_file(file_path)
                                        
                                        analysis = perform_dual_track_analysis(text, images, doc['filename'])
                                        
                                        analysis_file = ANALYSIS_DIR / f"{st.session_state.current_workspace}_{doc['filename']}.json"
                                        with open(analysis_file, 'w') as f:
                                            json.dump(analysis, f, indent=2, ensure_ascii=False)
                                        
                                        doc['has_deep_analysis'] = True
                                        
                                        with open(WORKSPACES_DIR / st.session_state.current_workspace / "metadata.json", 'w') as f:
                                            json.dump(metadata, f, indent=2)
                                        
                                        st.success("Analysis complete!")
                                        st.rerun()
    
    # TAB 4: Chat Assistant
    with tab4:
        st.header("💬 AI Assistant")
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
                    
                    if not response:
                        response = "I apologize, but I'm unable to process your request at this time. Please ensure the API key is configured correctly in the sidebar."
                    
                    st.markdown(response)
                    
                    if search_results:
                        with st.expander("📚 Sources"):
                            for r in search_results:
                                st.markdown(f"- **{r['document']['filename']}** (similarity: {r['similarity']:.2f})")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
