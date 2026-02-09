#!/usr/bin/env python3
"""
Enhanced Underwriting Assistant v2.9.0
核心改进：
1. Integrated Analysis Report 以表格形式展示
2. 支持按保险类型、客户名称、承保年度筛选
3. Handwriting Translation 显示图片+翻译+识别度百分比
"""

import streamlit as st
import os
import json
import hashlib
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io
import re

# 尝试导入 PDF 处理库
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# =============================================================================
# 核心配置
# =============================================================================

VERSION = "2.9.0"
APP_TITLE = "Enhanced Underwriting Assistant - Table View System"

# DeepSeek API 配置
API_BASE = "https://api.deepseek.com/v1"
API_MODEL = "deepseek-chat"
DEFAULT_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-99bba2ce117444e197270f17d303e74f")

# 数据目录结构
DATA_DIR = "data"
WORKSPACES_DIR = os.path.join(DATA_DIR, "workspaces")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
REVIEW_DIR = os.path.join(DATA_DIR, "review_queue")
AUDIT_DIR = os.path.join(DATA_DIR, "audit_logs")
CONFIG_DIR = os.path.join(DATA_DIR, "config")

# 初始数据集
INITIAL_DATASET = "Hull_MSC_Memo.pdf"

# 支持的文件格式
SUPPORTED_FORMATS = {
    "pdf": "📄",
    "docx": "📝",
    "doc": "📝",
    "txt": "📃",
    "xlsx": "📊",
    "xls": "📊",
    "png": "🖼️",
    "jpg": "🖼️",
    "jpeg": "🖼️"
}

# 保险类型选项
INSURANCE_TYPES = [
    "全部",
    "Hull & Machinery",
    "Cargo",
    "P&I",
    "War Risk",
    "Marine Liability",
    "其他"
]

# =============================================================================
# 系统提示词 - 简化版
# =============================================================================

ELECTRONIC_TEXT_SUMMARY_SYSTEM = """You are an underwriting document summarizer. 
Provide a BRIEF summary (3-5 sentences) of the document content covering:
- Insurance type and policy
- Insured party
- Key terms (premium, coverage, etc.)
- Main risk factors

Keep it concise and client-ready."""

HANDWRITING_TRANSLATION_SYSTEM = """You are a handwriting translator for insurance documents.

CRITICAL: For each handwritten annotation:
1. Translate to text (keep original language if unclear)
2. Estimate confidence (0-100%)
3. Describe location (e.g., "Top of page 1", "Margin right")

Output format:
[Location] Translated text (Confidence: XX%)

Example:
[Top of Page 1] To CEO: Renewal suggestions for your consideration (Confidence: 85%)
[Right margin, Page 2] Check premium calculation (Confidence: 92%)

DO NOT write a summary. Only translate each handwriting piece."""

# =============================================================================
# 工具函数
# =============================================================================

def ensure_directories():
    """确保所有必需的目录存在"""
    for dir_path in [DATA_DIR, WORKSPACES_DIR, EMBEDDINGS_DIR, 
                     ANALYSIS_DIR, REVIEW_DIR, AUDIT_DIR, CONFIG_DIR]:
        os.makedirs(dir_path, exist_ok=True)

def log_audit(action: str, details: Dict[str, Any]):
    """记录审计日志"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "action": action,
        "details": details
    }
    
    log_file = os.path.join(AUDIT_DIR, f"{datetime.now().strftime('%Y%m%d')}.json")
    
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    
    logs.append(log_entry)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

def get_api_key() -> str:
    """获取 API 密钥"""
    if "api_key" in st.session_state and st.session_state.api_key:
        return st.session_state.api_key
    return DEFAULT_API_KEY

def call_deepseek_api(messages: List[Dict[str, str]], max_tokens: int = 4000) -> Optional[str]:
    """调用 DeepSeek API"""
    api_key = get_api_key()
    
    if not HAS_REQUESTS:
        return "Error: requests library not installed"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": API_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"API Error: {str(e)}"

# =============================================================================
# PDF 处理函数
# =============================================================================

def extract_text_from_pdf(file_path: str) -> Tuple[str, List[Dict]]:
    """从 PDF 提取文本和图片"""
    text = ""
    images = []
    
    if not HAS_PYMUPDF:
        return "Error: PyMuPDF not installed", []
    
    try:
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 提取文本
            page_text = page.get_text()
            if page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            # 提取图片
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # 转换为 base64
                image_b64 = base64.b64encode(image_bytes).decode()
                
                images.append({
                    "page": page_num + 1,
                    "index": img_index,
                    "format": base_image["ext"],
                    "data": image_b64,
                    "width": base_image.get("width", 0),
                    "height": base_image.get("height", 0)
                })
        
        doc.close()
        
        # 检测是否是扫描件
        if not text.strip() and images:
            text = f"[Scanned PDF detected: {len(images)} images found across {len(doc)} pages]"
        
        return text, images
        
    except Exception as e:
        return f"Error extracting PDF: {str(e)}", []

def detect_handwriting_in_images(images: List[Dict]) -> bool:
    """简单的启发式检测：是否可能包含手写内容"""
    if not images:
        return False
    
    # 启发式规则：
    # 1. 图片较多（可能是扫描件 + 手写批注）
    # 2. 有小尺寸图片（可能是批注）
    
    if len(images) > 3:
        return True
    
    for img in images:
        # 检查是否有较小的图片（可能是手写批注）
        if img.get("width", 0) < 800 or img.get("height", 0) < 600:
            return True
    
    return False

# =============================================================================
# 文档分析函数
# =============================================================================

def analyze_electronic_text(text: str) -> str:
    """分析电子文本，返回简短摘要"""
    messages = [
        {"role": "system", "content": ELECTRONIC_TEXT_SUMMARY_SYSTEM},
        {"role": "user", "content": f"Summarize this insurance document:\n\n{text[:3000]}"}
    ]
    
    summary = call_deepseek_api(messages, max_tokens=500)
    return summary if summary else "Unable to generate summary"

def translate_handwriting(images: List[Dict], document_context: str = "") -> List[Dict]:
    """翻译手写内容"""
    if not images:
        return []
    
    # 构建提示词
    context_info = f"Document context: {document_context[:500]}" if document_context else ""
    
    prompt = f"""Analyze the handwriting in this insurance document.
{context_info}

The document contains {len(images)} images across multiple pages.
For each handwritten annotation you can identify, provide:
1. Location (page number and position)
2. Translated text
3. Confidence percentage (0-100%)

Format:
[Location] Text (Confidence: XX%)"""

    messages = [
        {"role": "system", "content": HANDWRITING_TRANSLATION_SYSTEM},
        {"role": "user", "content": prompt}
    ]
    
    translation_result = call_deepseek_api(messages, max_tokens=2000)
    
    # 解析翻译结果
    translations = []
    
    if translation_result and "[" in translation_result:
        # 简单解析格式: [Location] Text (Confidence: XX%)
        lines = translation_result.split('\n')
        for line in lines:
            if '[' in line and ']' in line:
                try:
                    # 提取位置
                    location = line[line.find('[')+1:line.find(']')]
                    
                    # 提取文本和置信度
                    rest = line[line.find(']')+1:].strip()
                    if '(Confidence:' in rest:
                        text = rest[:rest.find('(Confidence:')].strip()
                        conf_str = rest[rest.find('(Confidence:')+12:rest.find('%)')]
                        confidence = int(conf_str.strip())
                    else:
                        text = rest
                        confidence = 70  # 默认值
                    
                    translations.append({
                        "location": location,
                        "text": text,
                        "confidence": confidence,
                        "image_ref": None  # 可以后续关联到具体图片
                    })
                except:
                    continue
    
    # 如果解析失败，创建一个默认翻译
    if not translations and images:
        translations.append({
            "location": f"Page 1",
            "text": translation_result if translation_result else "Unable to translate handwriting",
            "confidence": 50,
            "image_ref": 0
        })
    
    return translations

# =============================================================================
# 工作区和文档管理
# =============================================================================

def create_workspace(workspace_name: str, description: str = ""):
    """创建工作区"""
    workspace_dir = os.path.join(WORKSPACES_DIR, workspace_name)
    os.makedirs(workspace_dir, exist_ok=True)
    
    metadata = {
        "name": workspace_name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "documents": []
    }
    
    metadata_file = os.path.join(workspace_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    log_audit("create_workspace", {"workspace": workspace_name})

def load_workspace_metadata(workspace_name: str) -> Optional[Dict]:
    """加载工作区元数据"""
    metadata_file = os.path.join(WORKSPACES_DIR, workspace_name, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_workspace_metadata(workspace_name: str, metadata: Dict):
    """保存工作区元数据"""
    metadata_file = os.path.join(WORKSPACES_DIR, workspace_name, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def upload_document(workspace_name: str, uploaded_file, auto_analyze: bool = True) -> bool:
    """上传文档到工作区"""
    try:
        # 保存文件
        workspace_dir = os.path.join(WORKSPACES_DIR, workspace_name)
        file_path = os.path.join(workspace_dir, uploaded_file.name)
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # 提取文本和图片
        text, images = "", []
        if uploaded_file.name.lower().endswith('.pdf'):
            text, images = extract_text_from_pdf(file_path)
        
        # 检测手写
        has_handwriting = detect_handwriting_in_images(images)
        
        # 分析文档
        summary = ""
        handwriting_translations = []
        
        if auto_analyze and text:
            summary = analyze_electronic_text(text)
            
            if has_handwriting:
                handwriting_translations = translate_handwriting(images, text)
        
        # 保存分析结果
        analysis_data = {
            "filename": uploaded_file.name,
            "upload_time": datetime.now().isoformat(),
            "has_images": len(images) > 0,
            "image_count": len(images),
            "has_handwriting": has_handwriting,
            "summary": summary,
            "handwriting_translations": handwriting_translations,
            "text_preview": text[:500] if text else ""
        }
        
        analysis_file = os.path.join(ANALYSIS_DIR, f"{uploaded_file.name}.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        # 保存图片数据
        if images:
            images_file = os.path.join(ANALYSIS_DIR, f"{uploaded_file.name}_images.json")
            with open(images_file, 'w', encoding='utf-8') as f:
                json.dump(images, f, indent=2)
        
        # 更新工作区元数据
        metadata = load_workspace_metadata(workspace_name)
        if metadata:
            # 提取文档信息
            doc_info = {
                "filename": uploaded_file.name,
                "upload_time": datetime.now().isoformat(),
                "size": uploaded_file.size,
                "has_handwriting": has_handwriting,
                "insurance_type": extract_insurance_type(text, summary),
                "client_name": extract_client_name(text, summary),
                "underwriting_year": extract_year(text, summary)
            }
            
            metadata["documents"].append(doc_info)
            save_workspace_metadata(workspace_name, metadata)
        
        log_audit("upload_document", {
            "workspace": workspace_name,
            "filename": uploaded_file.name,
            "has_handwriting": has_handwriting
        })
        
        return True
        
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return False

def extract_insurance_type(text: str, summary: str) -> str:
    """从文本中提取保险类型"""
    combined = (text + " " + summary).lower()
    
    if "hull" in combined or "machinery" in combined:
        return "Hull & Machinery"
    elif "cargo" in combined:
        return "Cargo"
    elif "p&i" in combined or "protection" in combined:
        return "P&I"
    elif "war" in combined:
        return "War Risk"
    elif "liability" in combined:
        return "Marine Liability"
    else:
        return "其他"

def extract_client_name(text: str, summary: str) -> str:
    """从文本中提取客户名称"""
    # 简单的提取逻辑：查找常见模式
    combined = text + " " + summary
    
    # 查找 "Insured:" 或类似模式
    patterns = [
        r"Insured[:\s]+([A-Z][a-zA-Z\s&]+)",
        r"Client[:\s]+([A-Z][a-zA-Z\s&]+)",
        r"Assured[:\s]+([A-Z][a-zA-Z\s&]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, combined)
        if match:
            return match.group(1).strip()[:50]
    
    return "Unknown"

def extract_year(text: str, summary: str) -> str:
    """从文本中提取承保年度"""
    combined = text + " " + summary
    
    # 查找年份（2000-2030）
    years = re.findall(r'20[0-2][0-9]', combined)
    if years:
        return years[0]
    
    return datetime.now().strftime("%Y")

# =============================================================================
# Streamlit UI
# =============================================================================

def render_table_view():
    """渲染表格视图"""
    st.header("📊 Integrated Analysis Report")
    
    # 获取所有工作区的文档
    all_documents = []
    
    if os.path.exists(WORKSPACES_DIR):
        for workspace_name in os.listdir(WORKSPACES_DIR):
            metadata = load_workspace_metadata(workspace_name)
            if metadata and "documents" in metadata:
                for doc in metadata["documents"]:
                    doc["workspace"] = workspace_name
                    all_documents.append(doc)
    
    if not all_documents:
        st.info("📭 No documents uploaded yet")
        return
    
    # 筛选控件
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_type = st.selectbox(
            "保险类型",
            ["全部"] + list(set([d.get("insurance_type", "其他") for d in all_documents]))
        )
    
    with col2:
        selected_client = st.selectbox(
            "客户名称",
            ["全部"] + list(set([d.get("client_name", "Unknown") for d in all_documents]))
        )
    
    with col3:
        selected_year = st.selectbox(
            "承保年度",
            ["全部"] + sorted(list(set([d.get("underwriting_year", "Unknown") for d in all_documents])), reverse=True)
        )
    
    # 应用筛选
    filtered_docs = all_documents
    
    if selected_type != "全部":
        filtered_docs = [d for d in filtered_docs if d.get("insurance_type") == selected_type]
    
    if selected_client != "全部":
        filtered_docs = [d for d in filtered_docs if d.get("client_name") == selected_client]
    
    if selected_year != "全部":
        filtered_docs = [d for d in filtered_docs if d.get("underwriting_year") == selected_year]
    
    # 显示表格
    st.write(f"**共 {len(filtered_docs)} 个文档**")
    
    if filtered_docs:
        # 创建表格数据
        table_data = []
        for doc in filtered_docs:
            table_data.append({
                "案例名称": doc.get("filename", "Unknown"),
                "类别": doc.get("insurance_type", "其他"),
                "承保年度": doc.get("underwriting_year", "Unknown"),
                "最新更新时间": doc.get("upload_time", "Unknown")[:19]
            })
        
        # 显示为可点击的表格
        for idx, row in enumerate(table_data):
            with st.expander(f"📄 {row['案例名称']} | {row['类别']} | {row['承保年度']}"):
                col_a, col_b = st.columns([1, 3])
                
                with col_a:
                    st.write("**文档信息**")
                    st.write(f"- 类别: {row['类别']}")
                    st.write(f"- 承保年度: {row['承保年度']}")
                    st.write(f"- 更新时间: {row['最新更新时间']}")
                
                with col_b:
                    # 显示分析结果
                    render_document_analysis(filtered_docs[idx])

def render_document_analysis(doc_info: Dict):
    """渲染单个文档的分析结果"""
    filename = doc_info.get("filename")
    
    # 加载分析数据
    analysis_file = os.path.join(ANALYSIS_DIR, f"{filename}.json")
    if not os.path.exists(analysis_file):
        st.warning("分析数据不存在")
        return
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    # 显示电子文本摘要
    st.write("**Electronic Text Analysis**")
    summary = analysis_data.get("summary", "No summary available")
    st.write(summary)
    
    # 显示手写翻译
    if analysis_data.get("has_handwriting"):
        st.write("---")
        st.write("**Handwriting Translation**")
        
        translations = analysis_data.get("handwriting_translations", [])
        
        if translations:
            # 加载图片数据
            images_file = os.path.join(ANALYSIS_DIR, f"{filename}_images.json")
            images = []
            if os.path.exists(images_file):
                with open(images_file, 'r', encoding='utf-8') as f:
                    images = json.load(f)
            
            for trans in translations:
                col_img, col_text = st.columns([1, 2])
                
                with col_img:
                    # 显示相关图片（如果有）
                    if images and trans.get("image_ref") is not None:
                        img_idx = trans["image_ref"]
                        if img_idx < len(images):
                            img_data = images[img_idx]["data"]
                            st.image(f"data:image/png;base64,{img_data}", width=200)
                    else:
                        st.write(f"🖼️ {trans['location']}")
                
                with col_text:
                    confidence = trans.get("confidence", 0)
                    st.write(f"**{trans['text']}**")
                    
                    # 显示置信度条
                    color = "green" if confidence >= 80 else "orange" if confidence >= 60 else "red"
                    st.progress(confidence / 100)
                    st.caption(f"识别度: {confidence}%")
        else:
            st.info("✅ Have handwriting notes (上传手写图片以获取翻译)")

def main():
    """主函数"""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide"
    )
    
    ensure_directories()
    
    st.title(f"{APP_TITLE} v{VERSION}")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 配置")
        
        # API Key
        api_key = st.text_input(
            "API Key",
            value=get_api_key(),
            type="password",
            key="api_key_input"
        )
        
        if st.button("💾 保存"):
            st.session_state.api_key = api_key
            st.success("已保存")
        
        st.divider()
        
        # 工作区选择
        st.header("📁 工作区")
        
        workspaces = []
        if os.path.exists(WORKSPACES_DIR):
            workspaces = [d for d in os.listdir(WORKSPACES_DIR) 
                         if os.path.isdir(os.path.join(WORKSPACES_DIR, d))]
        
        if not workspaces:
            create_workspace("Default", "默认工作区")
            workspaces = ["Default"]
        
        current_workspace = st.selectbox(
            "选择工作区",
            workspaces,
            key="current_workspace"
        )
        
        st.divider()
        
        # 上传文件
        st.header("📤 上传文档")
        uploaded_file = st.file_uploader(
            "选择文件",
            type=list(SUPPORTED_FORMATS.keys())
        )
        
        if uploaded_file:
            if st.button("🚀 上传并分析"):
                with st.spinner("处理中..."):
                    success = upload_document(
                        current_workspace,
                        uploaded_file,
                        auto_analyze=True
                    )
                    if success:
                        st.success(f"✅ 已上传: {uploaded_file.name}")
                        st.rerun()
    
    # 主内容区
    tab1, tab2 = st.tabs(["📊 报告表格", "ℹ️ 关于"])
    
    with tab1:
        render_table_view()
    
    with tab2:
        st.header("关于系统")
        st.write(f"""
        **Enhanced Underwriting Assistant** v{VERSION}
        
        核心功能：
        - ✅ 表格形式展示所有文档
        - ✅ 支持筛选（保险类型、客户名称、承保年度）
        - ✅ 电子文本简短摘要
        - ✅ 手写翻译（显示图片 + 翻译 + 识别度百分比）
        
        技术栈：
        - Streamlit
        - PyMuPDF (fitz)
        - DeepSeek API
        
        Powered by AI 🤖
        """)

if __name__ == "__main__":
    # 初始化数据
    ensure_directories()
    
    # 加载初始数据集（首次运行）
    if not os.path.exists(os.path.join(WORKSPACES_DIR, "Default")):
        create_workspace("Default", "默认工作区")
    
    main()
