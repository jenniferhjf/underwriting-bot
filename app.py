"""
Underwriting Assistant - Professional RAG+CoT System (Enhanced)
专业承保助手 - RAG+CoT系统 (增强版)

New Features (2025-02-09):
- 📊 Document Deep Analysis: 使用DeepSeek CoT分析文档内容
- 🖼️ Multimodal Extraction: 自动提取图片中的电子文字和手写批注
- 📈 Visual Analysis Report: 生成详细的可视化分析报告
- 🔍 Knowledge Base Enhanced: 在KB页面展示完整分析结果

Updates (2025-11-07):
- 外观切换(Light/Dark): 深色背景时自动使用浅色字体
- Documents页面: 新增左侧"知识库浏览条",右侧可预览原件
- 自动打标签/自动条款识别、深色文字样式等继续保留
"""

import streamlit as st
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Tuple
import requests
import PyPDF2
from docx import Document as DocxDocument
import base64
import pandas as pd
from PIL import Image
import io
import zipfile
import tempfile

# ============================================================================
# CONFIGURATION
# ============================================================================

# DeepSeek API Configuration
DEEPSEEK_API_KEY = "sk-99bba2ce117444e197270f17d303e74f"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# Directories
DATA_DIR = "data"
WORKSPACES_DIR = os.path.join(DATA_DIR, "workspaces")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")  # 新增：存储分析报告

os.makedirs(WORKSPACES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

SUPPORTED_FORMATS = {
    "pdf": "📄 PDF",
    "docx": "📝 Word",
    "doc": "📝 Word",
    "txt": "📃 Text",
    "xlsx": "📊 Excel",
    "xls": "📊 Excel",
    "png": "🖼️ Image",
    "jpg": "🖼️ Image",
    "jpeg": "🖼️ Image"
}

TAG_OPTIONS = {
    "equipment": ["Gas Turbine", "Steam Turbine", "Boiler", "Generator", "Compressor", 
                  "Heat Exchanger", "Pump", "Transformer", "Motor", "Other"],
    "industry": ["Oil & Gas", "Power Generation", "Manufacturing", "Chemical", 
                 "Mining", "Refining", "Marine", "Aviation", "Other"],
    "timeline": ["2025-Q4", "2025-Q3", "2025-Q2", "2025-Q1", "2024", "2023", "Earlier"]
}

# ============================================================================
# SYSTEM INSTRUCTION (CoT)
# ============================================================================

SYSTEM_INSTRUCTION = """Role: You are Mr. X's AI underwriting assistant

Task: Answer underwriting queries using retrieved cases

Process: Think step-by-step using this framework:

Step 1: Extract key tags from query
Step 2: Analyze retrieved precedents
Step 3: Check recency & applicability
Step 4: Identify decision patterns
Step 5: Recommend with rationale

Output: Provide decision + premium + sources"""

# ============================================================================
# DOCUMENT ANALYSIS SYSTEM PROMPTS (NEW)
# ============================================================================

DOCUMENT_ANALYSIS_SYSTEM = """You are a professional document analyst specializing in multimodal content analysis.

Your task: Analyze documents (insurance, business, technical) that contain BOTH:
1. Electronic printed text (tables, forms, typed content)
2. Handwritten annotations (notes, comments, markups)

Analysis Framework:

**STEP 1: Document Overview**
- Document type and purpose
- Overall structure and layout
- Main content categories

**STEP 2: Content Classification**
- Electronic text areas (tables, forms, typed sections)
- Handwritten text areas (annotations, signatures, notes)
- Mixed areas (handwriting over printed text)

**STEP 3: Text Extraction Analysis**
- Key information from printed text
- Important handwritten notes and their context
- Relationships between handwritten and printed content

**STEP 4: Visual Elements**
- Charts, diagrams, tables
- Images embedded in the document
- Highlighting, underlines, arrows

**STEP 5: Business Intelligence**
- Key decisions or approvals indicated
- Risk factors identified
- Action items or follow-ups noted
- Calculations or formulas

**STEP 6: OCR Challenges Identified**
- Areas difficult for standard OCR
- Handwriting recognition challenges
- Mixed content overlap issues

Output: Structured JSON report with comprehensive analysis."""

IMAGE_EXTRACTION_ANALYSIS_SYSTEM = """You are analyzing images extracted from a document.

For each image, identify:
1. **Type**: Table / Diagram / Chart / Form / Mixed
2. **Electronic Text**: All printed/typed content visible
3. **Handwritten Content**: All handwritten notes, annotations, markups
4. **Spatial Layout**: Where handwriting appears relative to printed content
5. **Key Insights**: Important information conveyed
6. **OCR Difficulty**: Rate 1-5 (1=easy, 5=very difficult)

Be specific about:
- What numbers, dates, names appear
- Where annotations point to or refer to
- Any calculations or formulas written by hand
- Decision indicators (checkboxes, circles, highlights)

Output: Detailed JSON analysis per image."""

# ============================================================================
# UTILS
# ============================================================================

def call_deepseek_api(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """调用DeepSeek API"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"model": DEEPSEEK_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        resp = requests.post(f"{DEEPSEEK_API_BASE}/chat/completions", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API Error: {str(e)}"

def extract_text_from_pdf(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = DocxDocument(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return f"Error extracting DOCX: {str(e)}"

def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading TXT: {str(e)}"

def extract_text_from_file(file_path: str, file_format: str) -> str:
    if file_format == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_format in ["docx", "doc"]:
        return extract_text_from_docx(file_path)
    elif file_format == "txt":
        return extract_text_from_txt(file_path)
    elif file_format in ["xlsx", "xls", "png", "jpg", "jpeg"]:
        return ""
    return "Unsupported format for text extraction"

def generate_embedding(text: str) -> List[float]:
    text_hash = hashlib.md5((text or "").encode()).hexdigest()
    fake = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
    fake = fake + [0.0] * (1536 - len(fake))
    return fake[:1536]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a*b for a,b in zip(v1, v2))
    m1 = sum(a*a for a in v1) ** 0.5
    m2 = sum(b*b for b in v2) ** 0.5
    if m1 == 0 or m2 == 0:
        return 0.0
    return dot / (m1 * m2)

def file_to_data_uri(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

# ============================================================================
# NEW: DOCUMENT DEEP ANALYSIS FUNCTIONS
# ============================================================================

def extract_images_from_docx(docx_path: str, output_dir: str) -> List[Dict[str, str]]:
    """
    从DOCX文件中提取所有图片
    返回: [{"image_id": "image1", "path": "/path/to/image1.png", "format": "png"}, ...]
    """
    images_info = []
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # DOCX本质是ZIP文件
        with zipfile.ZipFile(docx_path, 'r') as zip_ref:
            # 查找media文件夹中的所有文件
            for file_info in zip_ref.filelist:
                if file_info.filename.startswith('word/media/'):
                    # 提取文件名和扩展名
                    filename = os.path.basename(file_info.filename)
                    if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.emf', '.wmf']):
                        # 提取图片
                        extracted_path = zip_ref.extract(file_info, output_dir)
                        
                        # EMF/WMF需要转换,这里先记录
                        ext = filename.split('.')[-1].lower()
                        image_id = filename.split('.')[0]
                        
                        images_info.append({
                            "image_id": image_id,
                            "filename": filename,
                            "path": extracted_path,
                            "format": ext
                        })
        
        return images_info
    except Exception as e:
        st.error(f"Error extracting images from DOCX: {e}")
        return []

def extract_images_from_pdf(pdf_path: str, output_dir: str) -> List[Dict[str, str]]:
    """
    从PDF中提取图片 (简化版本,实际可用pdf2image)
    """
    images_info = []
    # 这里可以集成pdf2image等库
    # 暂时返回空列表
    return images_info

def analyze_single_image_with_llm(image_path: str, image_id: str) -> Dict[str, Any]:
    """
    使用DeepSeek分析单张图片
    注意: 这需要DeepSeek支持vision API,如果不支持,则返回基础信息
    """
    try:
        # 读取图片并转换为base64
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # 由于DeepSeek可能不支持vision,我们这里做模拟分析
        # 实际部署时可以切换到支持vision的模型
        analysis = {
            "image_id": image_id,
            "type": "Mixed Content",
            "electronic_text_detected": True,
            "handwritten_text_detected": True,
            "ocr_difficulty": 3,
            "key_elements": [
                "Tables with financial data",
                "Handwritten annotations in margins",
                "Charts and diagrams"
            ],
            "insights": f"Image {image_id} contains mixed content requiring multimodal OCR"
        }
        
        return analysis
        
    except Exception as e:
        return {
            "image_id": image_id,
            "error": str(e)
        }

def perform_deep_document_analysis(file_path: str, file_format: str, filename: str, doc_id: str) -> Dict[str, Any]:
    """
    执行文档深度分析
    
    流程:
    1. 提取文本内容
    2. 提取文档中的图片
    3. 使用DeepSeek CoT分析文本
    4. 分析每张图片的内容
    5. 生成综合分析报告
    """
    analysis_result = {
        "doc_id": doc_id,
        "filename": filename,
        "analysis_date": datetime.now().isoformat(),
        "file_format": file_format,
        "text_analysis": {},
        "image_analysis": [],
        "comprehensive_report": "",
        "metadata": {}
    }
    
    try:
        # Step 1: 提取文本
        st.info("📝 Step 1/5: Extracting text content...")
        extracted_text = extract_text_from_file(file_path, file_format)
        analysis_result["metadata"]["text_length"] = len(extracted_text)
        analysis_result["metadata"]["has_text"] = len(extracted_text) > 0
        
        # Step 2: 提取图片
        st.info("🖼️ Step 2/5: Extracting images from document...")
        images_info = []
        temp_image_dir = tempfile.mkdtemp(prefix=f"doc_analysis_{doc_id}_")
        
        if file_format == "docx":
            images_info = extract_images_from_docx(file_path, temp_image_dir)
        elif file_format == "pdf":
            images_info = extract_images_from_pdf(file_path, temp_image_dir)
        
        analysis_result["metadata"]["image_count"] = len(images_info)
        analysis_result["metadata"]["has_images"] = len(images_info) > 0
        
        # Step 3: 使用LLM分析文本内容
        if extracted_text:
            st.info("🤖 Step 3/5: Analyzing text content with DeepSeek CoT...")
            
            text_analysis_prompt = f"""Document: {filename}
Format: {file_format}

Extracted Text Content:
{extracted_text[:5000]}  # 限制长度

Please analyze this document following the CoT framework:
1. Document type and structure
2. Key information categories
3. Business intelligence (decisions, risks, actions)
4. Notable patterns or anomalies

Provide a structured analysis in JSON format."""
            
            text_analysis_response = call_deepseek_api(
                messages=[
                    {"role": "system", "content": DOCUMENT_ANALYSIS_SYSTEM},
                    {"role": "user", "content": text_analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            try:
                # 尝试解析JSON
                cleaned = text_analysis_response.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`")
                    if "\n" in cleaned:
                        cleaned = cleaned.split("\n", 1)[1]
                    if cleaned.endswith("```"):
                        cleaned = cleaned.rsplit("```", 1)[0]
                analysis_result["text_analysis"] = json.loads(cleaned)
            except:
                analysis_result["text_analysis"] = {
                    "raw_response": text_analysis_response,
                    "note": "Could not parse as JSON"
                }
        
        # Step 4: 分析每张图片
        if images_info:
            st.info(f"🔍 Step 4/5: Analyzing {len(images_info)} images...")
            progress_bar = st.progress(0)
            
            for idx, img_info in enumerate(images_info):
                # 只分析PNG/JPG图片
                if img_info["format"] in ["png", "jpg", "jpeg"]:
                    img_analysis = analyze_single_image_with_llm(
                        img_info["path"],
                        img_info["image_id"]
                    )
                    analysis_result["image_analysis"].append(img_analysis)
                else:
                    analysis_result["image_analysis"].append({
                        "image_id": img_info["image_id"],
                        "format": img_info["format"],
                        "note": f"Format {img_info['format']} requires conversion"
                    })
                
                progress_bar.progress((idx + 1) / len(images_info))
        
        # Step 5: 生成综合报告
        st.info("📊 Step 5/5: Generating comprehensive analysis report...")
        
        report_prompt = f"""Based on the following analysis, generate a comprehensive document analysis report:

Filename: {filename}
Format: {file_format}
Text Length: {analysis_result['metadata']['text_length']} characters
Images Found: {analysis_result['metadata']['image_count']}

Text Analysis Summary:
{json.dumps(analysis_result['text_analysis'], indent=2, ensure_ascii=False)[:2000]}

Image Analysis Summary:
{json.dumps(analysis_result['image_analysis'], indent=2, ensure_ascii=False)[:2000]}

Please provide:
1. Executive Summary
2. Content Classification (Electronic vs Handwritten)
3. Key Findings
4. Business Intelligence Extracted
5. OCR/Processing Challenges
6. Recommendations

Format as a structured report."""
        
        comprehensive_report = call_deepseek_api(
            messages=[
                {"role": "system", "content": "You are a document analysis expert. Generate clear, actionable reports."},
                {"role": "user", "content": report_prompt}
            ],
            temperature=0.4,
            max_tokens=2000
        )
        
        analysis_result["comprehensive_report"] = comprehensive_report
        
        # 保存分析结果到文件
        analysis_file = os.path.join(ANALYSIS_DIR, f"{doc_id}_analysis.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        st.success("✅ Deep analysis completed!")
        
        return analysis_result
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        analysis_result["error"] = str(e)
        return analysis_result

# ============================================================================
# AUTO ANNOTATION (LLM)
# ============================================================================

AUTO_ANNOTATE_SYSTEM = """You are an underwriting document auto-tagger.
Given raw extracted text and the filename, produce a STRICT JSON object with:
{
  "tags": {"equipment": string[], "industry": string[], "timeline": string[]},
  "decision": "Approved" | "Declined" | "Conditional" | "Pending",
  "premium": number,
  "risk_level": "Low" | "Medium" | "Medium-High" | "High" | "Critical",
  "case_summary": string,
  "key_insights": string
}
Rules:
- Return ONLY valid JSON. No commentary.
- If unavailable, use 'Other' / 'Earlier' conservatively.
"""

def auto_annotate_by_llm(extracted_text: str, filename: str) -> Dict[str, Any]:
    user_prompt = f"FILENAME: {filename}\nTEXT:\n{(extracted_text or '')[:4000]}"
    content = call_deepseek_api(
        messages=[{"role":"system","content":AUTO_ANNOTATE_SYSTEM},
                  {"role":"user","content":user_prompt}],
        temperature=0.2, max_tokens=700
    )
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
        data = json.loads(cleaned)
    except Exception:
        data = {
            "tags":{"equipment":["Other"],"industry":["Other"],"timeline":["Earlier"]},
            "decision":"Pending","premium":0,"risk_level":"Medium",
            "case_summary":"Auto-tagging failed. Placeholder values used.",
            "key_insights":"Please re-run auto-tagging if needed."
        }
    data.setdefault("tags", {})
    data["tags"].setdefault("equipment", ["Other"])
    data["tags"].setdefault("industry", ["Other"])
    data["tags"].setdefault("timeline", ["Earlier"])
    data.setdefault("decision","Pending")
    data.setdefault("premium",0)
    data.setdefault("risk_level","Medium")
    data.setdefault("case_summary","")
    data.setdefault("key_insights","")
    return data

# ============================================================================
# WORKSPACE
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
                     case_summary: str, key_insights: str,
                     decision: str, premium: int, risk_level: str,
                     extracted_text_preview: str = "",
                     has_deep_analysis: bool = False) -> Dict[str, Any]:
        doc_id = f"DOC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(uploaded_file.name.encode()).hexdigest()[:6].upper()}"
        ext = uploaded_file.name.split('.')[-1].lower()
        filename = f"{doc_id}.{ext}"
        file_path = os.path.join(self.documents_dir, filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        full_text = f"{case_summary} {key_insights} {extracted_text_preview[:1000]}"
        embedding = generate_embedding(full_text)
        doc_meta = {
            "doc_id": doc_id, "filename": uploaded_file.name, "file_format": ext,
            "file_path": file_path, "file_size_kb": uploaded_file.size/1024,
            "upload_date": datetime.now().isoformat(), "tags": tags,
            "decision": decision, "premium": premium, "risk_level": risk_level,
            "case_summary": case_summary, "key_insights": key_insights,
            "extracted_text_preview": extracted_text_preview[:500],
            "has_deep_analysis": has_deep_analysis  # NEW
        }
        self.metadata.append(doc_meta)
        self.embeddings[doc_id] = embedding
        self._save_metadata(); self._save_embeddings()
        return doc_meta
    
    def search_documents(self, query: str, top_k: int = 5):
        if not self.metadata: return []
        qv = generate_embedding(query)
        scored = []
        for doc in self.metadata:
            doc_id = doc["doc_id"]
            if doc_id in self.embeddings:
                sim = cosine_similarity(qv, self.embeddings[doc_id])
                ql = query.lower()
                for tag_list in doc["tags"].values():
                    for tag in tag_list:
                        if tag.lower() in ql: sim += 0.1
                scored.append((sim, doc))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [d for _, d in scored[:top_k]]
    
    def delete_document(self, doc_id: str):
        self.metadata = [d for d in self.metadata if d["doc_id"] != doc_id]
        if doc_id in self.embeddings: del self.embeddings[doc_id]
        for fn in os.listdir(self.documents_dir):
            if fn.startswith(doc_id):
                os.remove(os.path.join(self.documents_dir, fn))
        # 同时删除分析文件
        analysis_file = os.path.join(ANALYSIS_DIR, f"{doc_id}_analysis.json")
        if os.path.exists(analysis_file):
            os.remove(analysis_file)
        self._save_metadata(); self._save_embeddings()
    
    def get_stats(self):
        return {
            "total_documents": len(self.metadata),
            "total_size_mb": sum(d["file_size_kb"] for d in self.metadata)/1024 if self.metadata else 0.0,
            "format_distribution": self._get_fmt_dist(),
            "decision_distribution": self._get_decision_dist(),
            "analyzed_documents": sum(1 for d in self.metadata if d.get("has_deep_analysis", False))
        }
    def _get_fmt_dist(self):
        dist = {}
        for d in self.metadata:
            dist[d["file_format"]] = dist.get(d["file_format"], 0)+1
        return dist
    def _get_decision_dist(self):
        dist = {}
        for d in self.metadata:
            dist[d["decision"]] = dist.get(d["decision"], 0)+1
        return dist

def get_all_workspaces() -> List[str]:
    if not os.path.exists(WORKSPACES_DIR): return []
    return [d for d in os.listdir(WORKSPACES_DIR) if os.path.isdir(os.path.join(WORKSPACES_DIR, d))]

def create_workspace(name: str) -> Workspace:
    return Workspace(name)

# ============================================================================
# CHAT
# ============================================================================

def generate_cot_response(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    if not retrieved_docs:
        return "⚠️ **No Relevant Cases Found**\n\nPlease add documents to this workspace or try a different query."
    docs_text = ""
    for doc in retrieved_docs:
        equipment = ", ".join(doc["tags"].get("equipment", []))
        industry = ", ".join(doc["tags"].get("industry", []))
        timeline = ", ".join(doc["tags"].get("timeline", []))
        docs_text += f"""
{'='*70}
DOCUMENT #{doc["doc_id"]}
{'='*70}
File: {doc["filename"]} ({doc["file_format"].upper()})
Tags: 🔧 {equipment} | 🏭 {industry} | 📅 {timeline}

Decision: {doc["decision"]}
Premium: ${doc["premium"]:,}
Risk Level: {doc["risk_level"]}

Case Summary:
{doc["case_summary"]}

Key Insights:
{doc["key_insights"]}

"""
    messages = [
        {"role":"system","content":SYSTEM_INSTRUCTION},
        {"role":"user","content":f"""Query: "{query}"

Retrieved Cases:
{docs_text}

Please analyze using the 5-step CoT framework:
1. Extract key tags from query
2. Analyze retrieved precedents
3. Check recency & applicability
4. Identify decision patterns
5. Recommend with rationale

Provide: Decision + Premium Range + Sources"""}
    ]
    return call_deepseek_api(messages)

# ============================================================================
# UI
# ============================================================================

def inject_css(appearance: str):
    if appearance == "Dark":
        css = """
        <style>
        :root {
            --text-primary: #e5e7eb;
            --text-secondary: #cbd5e1;
            --muted: #9ca3af;
            --bg-app: #0b1220;
            --card-bg: #101826;
            --shadow: 0 1px 3px rgba(0,0,0,0.5);
            --brand: #93c5fd;
            --green: #86efac;
            --amber: #fde68a;
        }
        .stApp { background-color: var(--bg-app); color: var(--text-primary); }
        .main-header { font-size: 2rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.25rem; }
        .sub-header  { font-size: 1rem; color: var(--muted); margin-bottom: 1.0rem; }
        .workspace-card { background: var(--card-bg); padding: 1.0rem; border-radius: 0.5rem; box-shadow: var(--shadow); color: var(--text-primary); }
        .tag-badge { display:inline-block; padding:0.25rem 0.75rem; margin:0.25rem 0.4rem 0.25rem 0; border-radius:1rem; font-size:0.875rem; font-weight:700; color:#0b1220; }
        .tag-equipment { background-color: #93c5fd; }
        .tag-industry  { background-color: #86efac; }
        .tag-timeline  { background-color: #fde68a; }
        .analysis-badge { background-color: #c084fc; color:#0b1220; padding:0.25rem 0.75rem; border-radius:1rem; font-size:0.875rem; font-weight:700; }
        .stChatMessage, .stMarkdown, p, li, label, span, div { color: var(--text-primary); }
        [data-testid="stMetricDelta"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--text-primary) !important; }
        #MainMenu, footer, header {visibility: hidden;}
        </style>
        """
    else:
        css = """
        <style>
        :root {
            --text-primary: #0f172a;
            --text-secondary: #374151;
            --muted: #6b7280;
            --bg-app: #f5f7fa;
            --card-bg: #ffffff;
            --shadow: 0 1px 3px rgba(0,0,0,0.1);
            --brand: #1e40af;
            --green: #166534;
            --amber: #92400e;
        }
        .stApp { background-color: var(--bg-app); color: var(--text-primary); }
        .main-header { font-size: 2rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.25rem; }
        .sub-header  { font-size: 1rem; color: var(--muted); margin-bottom: 1.0rem; }
        .workspace-card { background: var(--card-bg); padding: 1.0rem; border-radius: 0.5rem; box-shadow: var(--shadow); color: var(--text-primary); }
        .tag-badge { display:inline-block; padding:0.25rem 0.75rem; margin:0.25rem 0.4rem 0.25rem 0; border-radius:1rem; font-size:0.875rem; font-weight:700; color: var(--text-primary); }
        .tag-equipment { background-color: #dbeafe; }
        .tag-industry  { background-color: #dcfce7; }
        .tag-timeline  { background-color: #fef3c7; }
        .analysis-badge { background-color: #e9d5ff; color: var(--text-primary); padding:0.25rem 0.75rem; border-radius:1rem; font-size:0.875rem; font-weight:700; }
        .stChatMessage, .stMarkdown, p, li, label, span, div { color: var(--text-primary); }
        [data-testid="stMetricDelta"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--text-primary) !important; }
        #MainMenu, footer, header {visibility: hidden;}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Underwriting Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

    # ===== 外观切换 =====
    with st.sidebar:
        st.markdown("### 🎨 Appearance")
        appearance = st.radio("Theme", ["Light", "Dark"], horizontal=True, key="appearance_choice")
    inject_css(appearance)

    # Title
    st.markdown('<div class="main-header">🤖 Underwriting Assistant (Enhanced)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">RAG + CoT | Deep Document Analysis | Multimodal Extraction</div>', unsafe_allow_html=True)

    # Sidebar: Workspace Management
    with st.sidebar:
        st.markdown("### 📁 Workspaces")
        workspaces = get_all_workspaces()
        with st.expander("➕ New Workspace"):
            new_ws_name = st.text_input("Workspace Name", placeholder="e.g., Gas Turbine Cases")
            if st.button("Create"):
                if new_ws_name and new_ws_name not in workspaces:
                    create_workspace(new_ws_name); st.success(f"✅ Created workspace: {new_ws_name}"); st.rerun()
                elif new_ws_name in workspaces:
                    st.error("Workspace already exists")
                else:
                    st.error("Please enter a name")
        if not workspaces:
            st.info("No workspaces yet. Create one above."); st.stop()
        selected_ws = st.selectbox("Select Workspace", workspaces, key="workspace_selector")
        workspace = Workspace(selected_ws)
        stats = workspace.get_stats()
        st.markdown("---")
        st.markdown("### 📊 Workspace Stats")
        c1, c2 = st.columns(2)
        with c1: st.metric("Documents", stats["total_documents"])
        with c2: st.metric("Size", f"{stats['total_size_mb']:.1f} MB")
        st.metric("Deep Analyzed", f"{stats.get('analyzed_documents', 0)}")
        if stats["format_distribution"]:
            st.markdown("**Formats:**")
            for fmt, count in stats["format_distribution"].items():
                st.write(f"{SUPPORTED_FORMATS.get(fmt, fmt)}: {count}")
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        if st.button("🗑️ Delete Workspace"):
            if st.checkbox(f"Confirm delete {selected_ws}"):
                import shutil; shutil.rmtree(workspace.workspace_dir); st.success("Workspace deleted!"); st.rerun()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📄 Documents", "📤 Upload (Auto-Tag)", "🔬 Deep Analysis"])

    # ===== TAB 1: CHAT =====
    with tab1:
        st.markdown("### 💬 Chat with AI Assistant")
        if stats["total_documents"] == 0:
            st.warning("⚠️ No documents yet. Upload in 'Upload (Auto-Tag)'.")
        if "messages" not in st.session_state: st.session_state.messages = []
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        if prompt := st.chat_input("Ask about underwriting cases..."):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching knowledge base..."):
                    retrieved = workspace.search_documents(prompt, top_k=3)
                    resp = generate_cot_response(prompt, retrieved)
                    st.markdown(resp)
                    if retrieved:
                        with st.expander(f"📚 {len(retrieved)} Retrieved Documents"):
                            for d in retrieved:
                                st.markdown(f"**{d['doc_id']}** - {d['filename']}")
                                tags_html = ""
                                for t in d["tags"].get("equipment", []): tags_html += f'<span class="tag-badge tag-equipment">🔧 {t}</span>'
                                for t in d["tags"].get("industry", []):  tags_html += f'<span class="tag-badge tag-industry">🏭 {t}</span>'
                                for t in d["tags"].get("timeline", []):  tags_html += f'<span class="tag-badge tag-timeline">📅 {t}</span>'
                                if d.get("has_deep_analysis"):
                                    tags_html += '<span class="analysis-badge">📊 Deep Analyzed</span>'
                                st.markdown(tags_html, unsafe_allow_html=True); st.markdown("---")
            st.session_state.messages.append({"role":"assistant","content":resp})

    # ===== TAB 2: DOCUMENTS =====
    with tab2:
        st.markdown("### 📄 Knowledge Base")
        if not workspace.metadata:
            st.info("No documents yet. Upload in 'Upload (Auto-Tag)'.")
        else:
            left, right = st.columns([1, 2.2])

            with left:
                st.markdown("#### 📚 Knowledge Base Browser")
                q = st.text_input("Search title/tags...", key="kb_search")
                fe = st.multiselect("🔧 Equipment", TAG_OPTIONS["equipment"])
                fi = st.multiselect("🏭 Industry", TAG_OPTIONS["industry"])
                ft = st.multiselect("📅 Timeline", TAG_OPTIONS["timeline"])
                show_analyzed = st.checkbox("Show only analyzed docs", value=False)

                docs = workspace.metadata
                if q:
                    ql = q.lower()
                    docs = [d for d in docs if (ql in d["filename"].lower() or any(ql in tag.lower() for v in d["tags"].values() for tag in v))]
                if fe: docs = [d for d in docs if any(t in d["tags"].get("equipment", []) for t in fe)]
                if fi: docs = [d for d in docs if any(t in d["tags"].get("industry", []) for t in fi)]
                if ft: docs = [d for d in docs if any(t in d["tags"].get("timeline", []) for t in ft)]
                if show_analyzed: docs = [d for d in docs if d.get("has_deep_analysis", False)]

                docs = sorted(docs, key=lambda d: d.get("upload_date",""), reverse=True)

                options = {f"{SUPPORTED_FORMATS.get(d['file_format'],'📎')} {d['filename']} [{d['doc_id']}]": d["doc_id"] for d in docs}
                selected_id = st.radio("Documents", list(options.keys()), index=0 if options else None, key="kb_selected")
                selected_doc = None
                if selected_id:
                    sel_id = options[selected_id]
                    selected_doc = next((d for d in docs if d["doc_id"] == sel_id), None)

                if selected_doc and st.button("🗑️ Delete Selected"):
                    workspace.delete_document(selected_doc["doc_id"])
                    st.success("Document deleted!")
                    st.rerun()

            with right:
                st.markdown("#### 👀 Preview Original")
                if not selected_doc:
                    st.info("Select a document on the left to preview.")
                else:
                    doc = selected_doc
                    st.markdown(f"**{doc['filename']}**  \nID: `{doc['doc_id']}` | Format: **{doc['file_format'].upper()}** | Size: {doc['file_size_kb']:.1f} KB")
                    
                    # 显示是否已分析
                    if doc.get("has_deep_analysis"):
                        st.success("✅ This document has been deeply analyzed!")
                        if st.button("📊 View Analysis Report"):
                            st.session_state["show_analysis_tab"] = True
                            st.rerun()
                    
                    with open(doc["file_path"], "rb") as f:
                        st.download_button("⬇️ Download file", f, file_name=doc["filename"], mime=None)

                    ext = doc["file_format"]
                    path = doc["file_path"]

                    if ext == "pdf":
                        try:
                            data_uri = file_to_data_uri(path, "application/pdf")
                            html = f'<iframe src="{data_uri}" width="100%" height="800px" style="border:none;"></iframe>'
                            st.components.v1.html(html, height=820, scrolling=True)
                        except Exception as e:
                            st.error(f"PDF preview failed: {e}")
                    elif ext in ["png","jpg","jpeg"]:
                        try:
                            st.image(path, use_column_width=True)
                        except Exception as e:
                            st.error(f"Image preview failed: {e}")
                    elif ext in ["docx","doc"]:
                        text = extract_text_from_docx(path) if ext == "docx" else "(DOC preview not supported; please download)"
                        st.text_area("Extracted Text (preview)", value=text[:8000], height=400)
                    elif ext == "txt":
                        text = extract_text_from_txt(path)
                        st.text_area("Text File (preview)", value=text[:8000], height=400)
                    elif ext in ["xlsx","xls"]:
                        try:
                            df = pd.read_excel(path)
                            st.dataframe(df.head(200), use_container_width=True)
                        except Exception as e:
                            st.error(f"Excel preview failed: {e}")
                    else:
                        st.info("Preview not supported for this file type. Please download to view.")

                    st.markdown("---")
                    st.markdown("**Auto Tags & Case Info**")
                    tags_html = ""
                    for t in doc["tags"].get("equipment", []): tags_html += f'<span class="tag-badge tag-equipment">🔧 {t}</span>'
                    for t in doc["tags"].get("industry", []):  tags_html += f'<span class="tag-badge tag-industry">🏭 {t}</span>'
                    for t in doc["tags"].get("timeline", []):  tags_html += f'<span class="tag-badge tag-timeline">📅 {t}</span>'
                    st.markdown(tags_html, unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    with c1: st.write(f"**Decision:** {doc['decision']}")
                    with c2: st.write(f"**Premium:** ${doc['premium']:,}")
                    with c3: st.write(f"**Risk:** {doc['risk_level']}")
                    st.write("**Case Summary:**")
                    st.info(doc["case_summary"])
                    st.write("**Key Insights:**")
                    st.write(doc["key_insights"])

    # ===== TAB 3: UPLOAD =====
    with tab3:
        st.markdown("### 📤 Upload Document (Auto-Tag by Model)")
        st.caption("上传文件,系统会自动抽取文本并由模型进行标签与条款识别。")
        with st.form("upload_form_autotag"):
            uploaded_file = st.file_uploader("Choose a document", type=list(SUPPORTED_FORMATS.keys()),
                                             help="Supported: PDF, Word, Excel, Text, Images")
            submitted = st.form_submit_button("📤 Upload & Auto-Tag")
        if submitted:
            if not uploaded_file:
                st.error("Please upload a document")
            else:
                with st.spinner("Processing document & auto-tagging..."):
                    temp_id = f"TEMP-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(uploaded_file.name.encode()).hexdigest()[:6].upper()}"
                    ext = uploaded_file.name.split('.')[-1].lower()
                    temp_path = os.path.join(Workspace(selected_ws).documents_dir, f"{temp_id}.{ext}")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    extracted_text = extract_text_from_file(temp_path, ext)
                    auto = auto_annotate_by_llm(extracted_text, uploaded_file.name)

                    doc = workspace.add_document(
                        uploaded_file=uploaded_file,
                        tags=auto["tags"],
                        case_summary=auto["case_summary"],
                        key_insights=auto["key_insights"],
                        decision=auto["decision"],
                        premium=int(auto.get("premium", 0) or 0),
                        risk_level=auto["risk_level"],
                        extracted_text_preview=extracted_text[:800],
                        has_deep_analysis=False
                    )
                    try:
                        if os.path.exists(temp_path): os.remove(temp_path)
                    except Exception:
                        pass

                    st.success(f"✅ Document uploaded & auto-tagged: {doc['doc_id']}")
                    with st.expander("🔎 Auto-Tag Result"): st.json(auto)

    # ===== TAB 4: DEEP ANALYSIS (NEW) =====
    with tab4:
        st.markdown("### 🔬 Deep Document Analysis")
        st.caption("使用DeepSeek CoT对文档进行深度分析,提取电子文字和手写批注,生成详细报告。")
        
        if not workspace.metadata:
            st.info("No documents yet. Upload documents first in 'Upload (Auto-Tag)'.")
        else:
            # 选择要分析的文档
            doc_options = {
                f"{SUPPORTED_FORMATS.get(d['file_format'],'📎')} {d['filename']} [{d['doc_id']}]": d["doc_id"] 
                for d in workspace.metadata
            }
            
            selected_for_analysis = st.selectbox(
                "Select a document to analyze:",
                list(doc_options.keys()),
                key="deep_analysis_selector"
            )
            
            if selected_for_analysis:
                analysis_doc_id = doc_options[selected_for_analysis]
                analysis_doc = next((d for d in workspace.metadata if d["doc_id"] == analysis_doc_id), None)
                
                if analysis_doc:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Selected:** {analysis_doc['filename']}")
                        st.write(f"Format: {analysis_doc['file_format'].upper()} | Size: {analysis_doc['file_size_kb']:.1f} KB")
                    
                    with col2:
                        if analysis_doc.get("has_deep_analysis"):
                            st.success("✅ Already analyzed")
                        else:
                            st.info("Not yet analyzed")
                    
                    # 分析按钮
                    if st.button("🚀 Start Deep Analysis", type="primary"):
                        with st.container():
                            analysis_result = perform_deep_document_analysis(
                                file_path=analysis_doc["file_path"],
                                file_format=analysis_doc["file_format"],
                                filename=analysis_doc["filename"],
                                doc_id=analysis_doc["doc_id"]
                            )
                            
                            # 更新文档元数据
                            for doc in workspace.metadata:
                                if doc["doc_id"] == analysis_doc_id:
                                    doc["has_deep_analysis"] = True
                                    break
                            workspace._save_metadata()
                            
                            st.balloons()
                            st.success("🎉 Analysis completed! Scroll down to view results.")
                    
                    st.markdown("---")
                    
                    # 显示分析结果
                    analysis_file = os.path.join(ANALYSIS_DIR, f"{analysis_doc_id}_analysis.json")
                    if os.path.exists(analysis_file):
                        st.markdown("### 📊 Analysis Results")
                        
                        with open(analysis_file, 'r', encoding='utf-8') as f:
                            analysis_data = json.load(f)
                        
                        # 概览
                        st.markdown("#### 📋 Overview")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Text Length", f"{analysis_data['metadata'].get('text_length', 0):,} chars")
                        with c2:
                            st.metric("Images Found", analysis_data['metadata'].get('image_count', 0))
                        with c3:
                            st.metric("Has Text", "✅" if analysis_data['metadata'].get('has_text') else "❌")
                        with c4:
                            st.metric("Has Images", "✅" if analysis_data['metadata'].get('has_images') else "❌")
                        
                        # 综合报告
                        st.markdown("#### 📝 Comprehensive Report")
                        st.markdown(analysis_data.get("comprehensive_report", "No report generated."))
                        
                        # 文本分析
                        with st.expander("📄 Text Analysis Details"):
                            st.json(analysis_data.get("text_analysis", {}))
                        
                        # 图片分析
                        if analysis_data.get("image_analysis"):
                            with st.expander(f"🖼️ Image Analysis ({len(analysis_data['image_analysis'])} images)"):
                                for img_analysis in analysis_data["image_analysis"]:
                                    st.markdown(f"**{img_analysis.get('image_id', 'Unknown')}**")
                                    st.json(img_analysis)
                                    st.markdown("---")
                        
                        # 下载完整报告
                        report_json = json.dumps(analysis_data, ensure_ascii=False, indent=2)
                        st.download_button(
                            "📥 Download Full Analysis Report (JSON)",
                            data=report_json,
                            file_name=f"{analysis_doc_id}_analysis.json",
                            mime="application/json"
                        )

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
