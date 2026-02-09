"""
Enhanced Underwriting Assistant v2.8.3 - Fixed Version
=======================================================
Fixed: Streamlit image() TypeError
- All use_container_width=True replaced with use_column_width=True
- Compatible with Streamlit < 1.23.0

Date: 2026-02-09
Version: 2.8.3-fixed
"""

import streamlit as st
import json
import os
import base64
from datetime import datetime
from pathlib import Path
import io
from typing import Dict, List, Optional, Tuple
import hashlib

# PDF/DOCX processing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# OCR support
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# AI/API support
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    
    # API Settings
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    
    # OCR Settings
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
    
    # Application Settings
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "200"))  # MB
    
    # Data Storage
    DATA_DIR = Path("data")
    WORKSPACE_FILE = DATA_DIR / "workspaces.json"
    
    @classmethod
    def init_storage(cls):
        """Initialize storage directories"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        if not cls.WORKSPACE_FILE.exists():
            cls.WORKSPACE_FILE.write_text(json.dumps({}))


# Initialize configuration
Config.init_storage()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_doc_id(filename: str) -> str:
    """Generate unique document ID"""
    timestamp = datetime.now().isoformat()
    content = f"{filename}_{timestamp}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def safe_json_loads(data: str, default=None):
    """Safely load JSON with fallback"""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}


def format_timestamp(dt: datetime = None) -> str:
    """Format timestamp for display"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

class DocumentProcessor:
    """Handle document extraction and processing"""
    
    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, List[Dict]]:
        """Extract text and images from PDF"""
        if not PYMUPDF_AVAILABLE:
            return "", []
        
        text_content = []
        images = []
        
        try:
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Extract text
                page_text = page.get_text()
                if page_text.strip():
                    text_content.append(page_text)
                
                # Extract images
                image_list = page.get_images()
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Convert to base64
                        image_base64 = base64.b64encode(image_bytes).decode()
                        
                        images.append({
                            "id": f"page_{page_num + 1}_img_{img_index + 1}",
                            "data": f"data:image/{image_ext};base64,{image_base64}",
                            "page": page_num + 1,
                            "size": len(image_bytes),
                            "type": "embedded"
                        })
                    except Exception as e:
                        if Config.DEBUG_MODE:
                            st.error(f"Error extracting image: {e}")
                        continue
            
            pdf_document.close()
            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return "", []
        
        return "\n\n".join(text_content), images
    
    @staticmethod
    def extract_text_from_docx(file_bytes: bytes) -> str:
        """Extract text from DOCX"""
        if not DOCX_AVAILABLE:
            return ""
        
        try:
            doc = Document(io.BytesIO(file_bytes))
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            return "\n\n".join(text_content)
        except Exception as e:
            st.error(f"Error processing DOCX: {e}")
            return ""
    
    @staticmethod
    def perform_ocr(image_data: str) -> Dict:
        """Perform OCR on image data"""
        if not OCR_AVAILABLE:
            return {
                "text": "[OCR not available - pytesseract not installed]",
                "confidence": 0.0
            }
        
        try:
            # Decode base64 image
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Set tesseract path if configured
            if Config.TESSERACT_CMD:
                pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
            
            # Perform OCR
            text = pytesseract.image_to_string(image)
            
            # Get confidence (simplified)
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "text": text.strip(),
                "confidence": avg_confidence / 100.0  # Normalize to 0-1
            }
        except Exception as e:
            if Config.DEBUG_MODE:
                st.error(f"OCR Error: {e}")
            return {
                "text": f"[OCR Error: {str(e)}]",
                "confidence": 0.0
            }


# ============================================================================
# AI ANALYSIS
# ============================================================================

class AIAnalyzer:
    """Handle AI-powered document analysis"""
    
    def __init__(self):
        self.api_key = Config.DEEPSEEK_API_KEY
        self.api_base = Config.DEEPSEEK_API_BASE
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
        else:
            self.client = None
    
    def analyze_electronic_text(self, text: str, filename: str) -> str:
        """Generate brief summary of electronic text"""
        if not self.client:
            return self._fallback_summary(text)
        
        try:
            prompt = f"""Analyze this insurance document and provide a brief summary (3-5 sentences).

Document: {filename}

Content:
{text[:5000]}  # Limit to first 5000 chars

Provide a concise summary focusing on:
1. Document type and purpose
2. Key coverage details
3. Important dates or terms"""

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an insurance underwriting expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            if Config.DEBUG_MODE:
                st.error(f"AI Analysis Error: {e}")
            return self._fallback_summary(text)
    
    @staticmethod
    def _fallback_summary(text: str) -> str:
        """Generate simple fallback summary"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        preview = ' '.join(lines[:10])[:500]
        return f"Document preview: {preview}..." if len(preview) == 500 else preview


# ============================================================================
# WORKSPACE MANAGEMENT
# ============================================================================

class WorkspaceManager:
    """Manage workspaces and documents"""
    
    def __init__(self):
        self.workspace_file = Config.WORKSPACE_FILE
        self.workspaces = self._load_workspaces()
    
    def _load_workspaces(self) -> Dict:
        """Load workspaces from file"""
        try:
            with open(self.workspace_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            if Config.DEBUG_MODE:
                st.error(f"Error loading workspaces: {e}")
            return {}
    
    def _save_workspaces(self):
        """Save workspaces to file"""
        try:
            with open(self.workspace_file, 'w', encoding='utf-8') as f:
                json.dump(self.workspaces, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving workspaces: {e}")
    
    def create_workspace(self, name: str) -> bool:
        """Create new workspace"""
        if name in self.workspaces:
            return False
        
        self.workspaces[name] = {
            "created_at": format_timestamp(),
            "documents": {}
        }
        self._save_workspaces()
        return True
    
    def add_document(self, workspace: str, doc_data: Dict) -> bool:
        """Add document to workspace"""
        if workspace not in self.workspaces:
            return False
        
        doc_id = doc_data["document_id"]
        self.workspaces[workspace]["documents"][doc_id] = doc_data
        self._save_workspaces()
        return True
    
    def get_documents(self, workspace: str) -> Dict:
        """Get all documents in workspace"""
        if workspace not in self.workspaces:
            return {}
        return self.workspaces[workspace]["documents"]
    
    def get_document(self, workspace: str, doc_id: str) -> Optional[Dict]:
        """Get specific document"""
        docs = self.get_documents(workspace)
        return docs.get(doc_id)
    
    def list_workspaces(self) -> List[str]:
        """List all workspace names"""
        return list(self.workspaces.keys())


# ============================================================================
# UI RENDERING FUNCTIONS
# ============================================================================

def render_header():
    """Render application header"""
    st.title("🔍 Enhanced Underwriting Assistant")
    st.markdown("### AI-Powered Document Analysis with OCR & Handwriting Recognition")
    st.markdown("---")


def render_sidebar(workspace_manager: WorkspaceManager):
    """Render sidebar navigation"""
    st.sidebar.title("📂 Navigation")
    
    # Workspace selection
    st.sidebar.subheader("Workspace")
    workspaces = workspace_manager.list_workspaces()
    
    # Create new workspace
    with st.sidebar.expander("➕ Create New Workspace"):
        new_workspace = st.text_input("Workspace Name", key="new_workspace_input")
        if st.button("Create", key="create_workspace_btn"):
            if new_workspace:
                if workspace_manager.create_workspace(new_workspace):
                    st.success(f"✅ Created workspace: {new_workspace}")
                    st.rerun()
                else:
                    st.error("❌ Workspace already exists")
    
    # Select workspace
    if workspaces:
        selected_workspace = st.sidebar.selectbox(
            "Select Workspace",
            workspaces,
            key="workspace_selector"
        )
    else:
        st.sidebar.warning("No workspaces available. Create one above.")
        selected_workspace = None
    
    st.sidebar.markdown("---")
    
    # Main navigation
    st.sidebar.subheader("View")
    view_mode = st.sidebar.radio(
        "Select View",
        ["📤 Upload Document", "📊 Integrated Analysis", "📋 Document List"],
        key="view_mode"
    )
    
    return selected_workspace, view_mode


def render_upload_view(workspace: str, workspace_manager: WorkspaceManager):
    """Render document upload interface"""
    st.header("📤 Upload Document")
    
    if not workspace:
        st.warning("⚠️ Please select or create a workspace first.")
        return
    
    st.markdown(f"**Current Workspace:** `{workspace}`")
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx"],
        help="Upload PDF or DOCX files for analysis"
    )
    
    if uploaded_file:
        # Display file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"📄 **File:** {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        # Check file size
        if file_size_mb > Config.MAX_FILE_SIZE:
            st.error(f"❌ File too large. Maximum size: {Config.MAX_FILE_SIZE} MB")
            return
        
        # Process document
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                process_document(uploaded_file, workspace, workspace_manager)


def process_document(uploaded_file, workspace: str, workspace_manager: WorkspaceManager):
    """Process uploaded document"""
    try:
        # Read file
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        file_ext = Path(filename).suffix.lower()
        
        # Generate document ID
        doc_id = generate_doc_id(filename)
        
        # Extract content based on file type
        st.info("📖 Extracting content...")
        processor = DocumentProcessor()
        
        if file_ext == ".pdf":
            text_content, images = processor.extract_text_from_pdf(file_bytes)
        elif file_ext == ".docx":
            text_content = processor.extract_text_from_docx(file_bytes)
            images = []
        else:
            st.error("❌ Unsupported file type")
            return
        
        # AI Analysis
        analyzer = AIAnalyzer()
        
        st.info("🤖 Analyzing content...")
        if text_content:
            summary = analyzer.analyze_electronic_text(text_content, filename)
        else:
            summary = "No text content extracted."
        
        # OCR for images
        handwriting_data = []
        if images:
            st.info(f"🖼️ Processing {len(images)} images...")
            progress_bar = st.progress(0)
            
            for idx, img in enumerate(images):
                ocr_result = processor.perform_ocr(img["data"])
                handwriting_data.append({
                    **img,
                    "translated_text": ocr_result["text"],
                    "recognition_confidence": ocr_result["confidence"]
                })
                progress_bar.progress((idx + 1) / len(images))
            
            progress_bar.empty()
        
        # Prepare document data
        doc_data = {
            "document_id": doc_id,
            "filename": filename,
            "format": file_ext[1:],  # Remove dot
            "uploaded_at": format_timestamp(),
            "case_name": filename.replace(file_ext, ""),
            "category": "General Insurance",  # Default category
            "underwriting_year": datetime.now().year,
            "last_updated": format_timestamp(),
            "electronic_text": text_content,
            "electronic_summary": summary,
            "has_handwriting": len(handwriting_data) > 0,
            "handwriting_data": handwriting_data,
            "num_images": len(images)
        }
        
        # Save to workspace
        if workspace_manager.add_document(workspace, doc_data):
            st.success(f"✅ Document processed successfully!")
            st.success(f"📝 Document ID: `{doc_id}`")
            st.success(f"📊 Extracted {len(images)} images")
            st.success(f"✍️ Handwriting detected: {'Yes' if handwriting_data else 'No'}")
        else:
            st.error("❌ Failed to save document")
    
    except Exception as e:
        st.error(f"❌ Error processing document: {e}")
        if Config.DEBUG_MODE:
            st.exception(e)


def render_integrated_analysis(workspace: str, workspace_manager: WorkspaceManager):
    """Render integrated analysis report"""
    st.header("📊 Integrated Analysis Report")
    
    if not workspace:
        st.warning("⚠️ Please select a workspace first.")
        return
    
    documents = workspace_manager.get_documents(workspace)
    
    if not documents:
        st.info("📭 No documents in this workspace. Upload documents to get started.")
        return
    
    st.markdown(f"**Workspace:** `{workspace}` | **Total Documents:** {len(documents)}")
    st.markdown("---")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_category = st.selectbox(
            "Filter by Category",
            ["All"] + list(set(doc.get("category", "N/A") for doc in documents.values())),
            key="filter_category"
        )
    
    with col2:
        filter_year = st.selectbox(
            "Filter by Year",
            ["All"] + sorted(list(set(doc.get("underwriting_year", "N/A") for doc in documents.values())), reverse=True),
            key="filter_year"
        )
    
    with col3:
        search_name = st.text_input("Search by Name", key="search_name")
    
    # Apply filters
    filtered_docs = []
    for doc in documents.values():
        # Category filter
        if filter_category != "All" and doc.get("category") != filter_category:
            continue
        
        # Year filter
        if filter_year != "All" and doc.get("underwriting_year") != filter_year:
            continue
        
        # Name search
        if search_name and search_name.lower() not in doc.get("case_name", "").lower():
            continue
        
        filtered_docs.append(doc)
    
    st.markdown(f"**Showing:** {len(filtered_docs)} document(s)")
    st.markdown("---")
    
    # Display table
    if filtered_docs:
        # Create table data
        table_data = []
        for doc in filtered_docs:
            table_data.append({
                "Case Name": doc.get("case_name", "N/A"),
                "Category": doc.get("category", "N/A"),
                "Year": doc.get("underwriting_year", "N/A"),
                "Last Updated": doc.get("last_updated", "N/A"),
                "Document ID": doc.get("document_id", "N/A")
            })
        
        # Display as dataframe - FIXED: use_column_width
        st.dataframe(
            table_data,
            use_column_width=True,
            hide_index=True
        )
        
        # View details
        st.markdown("---")
        st.subheader("📋 View Document Details")
        
        doc_names = [doc.get("case_name", doc.get("filename", "Unknown")) for doc in filtered_docs]
        selected_doc_name = st.selectbox("Select Document", doc_names, key="select_doc_analysis")
        
        if selected_doc_name:
            # Find selected document
            selected_doc = next(
                (doc for doc in filtered_docs if doc.get("case_name", doc.get("filename")) == selected_doc_name),
                None
            )
            
            if selected_doc:
                render_analysis_view(workspace, selected_doc["filename"])
    else:
        st.info("No documents match the filters.")


def render_analysis_view(workspace: str, filename: str):
    """Render detailed analysis view for a document"""
    workspace_manager = st.session_state.get('workspace_manager')
    documents = workspace_manager.get_documents(workspace)
    
    # Find document by filename
    doc = next((d for d in documents.values() if d.get("filename") == filename), None)
    
    if not doc:
        st.error("❌ Document not found")
        return
    
    st.markdown("---")
    st.subheader(f"📄 {doc.get('filename', 'Unknown')}")
    
    # Document metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Document ID", doc.get("document_id", "N/A"))
    with col2:
        st.metric("Category", doc.get("category", "N/A"))
    with col3:
        st.metric("Year", doc.get("underwriting_year", "N/A"))
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📝 Electronic Text", "✍️ Handwriting Translation", "📊 Metadata"])
    
    with tab1:
        st.subheader("Electronic Text Analysis")
        summary = doc.get("electronic_summary", "No summary available.")
        st.info(summary)
        
        with st.expander("📖 View Full Text"):
            full_text = doc.get("electronic_text", "No text extracted.")
            st.text_area("Full Text Content", full_text, height=400, key=f"full_text_{doc['document_id']}")
    
    with tab2:
        st.subheader("Handwriting Translation")
        
        if not doc.get("has_handwriting", False):
            st.info("📭 No handwriting detected in this document.")
        else:
            handwriting_data = doc.get("handwriting_data", [])
            st.markdown(f"**Total Images:** {len(handwriting_data)}")
            st.markdown("---")
            
            for idx, hw_item in enumerate(handwriting_data):
                col_img, col_text = st.columns([1, 2])
                
                with col_img:
                    st.markdown(f"**Image {idx + 1}** (Page {hw_item.get('page', 'N/A')})")
                    
                    # Display image - FIXED: use_column_width instead of use_container_width
                    try:
                        # Convert base64 to image
                        if "base64," in hw_item.get("data", ""):
                            image_data = hw_item["data"].split("base64,")[1]
                            image_bytes = base64.b64decode(image_data)
                            st.image(image_bytes, use_column_width=True)
                        else:
                            st.warning("Invalid image data")
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")
                
                with col_text:
                    st.markdown("**Translated Text:**")
                    translated_text = hw_item.get("translated_text", "No translation available")
                    st.text_area(
                        "Translation",
                        translated_text,
                        height=150,
                        key=f"translation_{doc['document_id']}_{idx}",
                        label_visibility="collapsed"
                    )
                    
                    # Confidence indicator
                    confidence = hw_item.get("recognition_confidence", 0.0)
                    confidence_pct = confidence * 100
                    
                    # Color coding
                    if confidence >= 0.8:
                        color = "🟢"
                    elif confidence >= 0.5:
                        color = "🟡"
                    else:
                        color = "🔴"
                    
                    st.markdown(f"**Recognition Confidence:** {color} {confidence_pct:.1f}%")
                    st.progress(confidence, text=None)
                
                st.markdown("---")
    
    with tab3:
        st.subheader("Document Metadata")
        
        metadata = {
            "Document ID": doc.get("document_id", "N/A"),
            "Filename": doc.get("filename", "N/A"),
            "Format": doc.get("format", "N/A"),
            "Case Name": doc.get("case_name", "N/A"),
            "Category": doc.get("category", "N/A"),
            "Underwriting Year": doc.get("underwriting_year", "N/A"),
            "Uploaded At": doc.get("uploaded_at", "N/A"),
            "Last Updated": doc.get("last_updated", "N/A"),
            "Has Handwriting": "Yes" if doc.get("has_handwriting", False) else "No",
            "Number of Images": doc.get("num_images", 0)
        }
        
        for key, value in metadata.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**{key}:**")
            with col2:
                st.markdown(f"`{value}`")


def render_document_list(workspace: str, workspace_manager: WorkspaceManager):
    """Render document list view"""
    st.header("📋 Document List")
    
    if not workspace:
        st.warning("⚠️ Please select a workspace first.")
        return
    
    documents = workspace_manager.get_documents(workspace)
    
    if not documents:
        st.info("📭 No documents in this workspace.")
        return
    
    st.markdown(f"**Workspace:** `{workspace}` | **Total Documents:** {len(documents)}")
    st.markdown("---")
    
    # List documents
    for doc_id, doc in documents.items():
        with st.expander(f"📄 {doc.get('filename', 'Unknown')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Document ID:** `{doc_id}`")
                st.markdown(f"**Category:** {doc.get('category', 'N/A')}")
                st.markdown(f"**Year:** {doc.get('underwriting_year', 'N/A')}")
            
            with col2:
                st.markdown(f"**Uploaded:** {doc.get('uploaded_at', 'N/A')}")
                st.markdown(f"**Format:** {doc.get('format', 'N/A').upper()}")
                st.markdown(f"**Handwriting:** {'Yes ✍️' if doc.get('has_handwriting', False) else 'No'}")
            
            if st.button(f"View Details", key=f"view_{doc_id}"):
                st.session_state['view_doc'] = doc.get('filename')
                st.rerun()


def render_handwriting_translator():
    """Render handwriting translator tool (standalone)"""
    st.header("✍️ Handwriting Translator")
    st.markdown("Upload handwriting images for OCR recognition")
    st.markdown("---")
    
    uploaded_images = st.file_uploader(
        "Upload Handwriting Images",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        help="Upload images containing handwritten text"
    )
    
    if uploaded_images:
        processor = DocumentProcessor()
        
        for idx, img_file in enumerate(uploaded_images):
            st.markdown(f"### Image {idx + 1}: {img_file.name}")
            
            col_img, col_result = st.columns([1, 2])
            
            with col_img:
                # FIXED: use_column_width instead of use_container_width
                st.image(img_file, caption=img_file.name, use_column_width=True)
            
            with col_result:
                # Convert to base64
                img_bytes = img_file.read()
                img_base64 = base64.b64encode(img_bytes).decode()
                img_data = f"data:image/png;base64,{img_base64}"
                
                # Perform OCR
                with st.spinner("Recognizing..."):
                    ocr_result = processor.perform_ocr(img_data)
                
                st.markdown("**Recognized Text:**")
                st.text_area(
                    "Text",
                    ocr_result["text"],
                    height=150,
                    key=f"ocr_result_{idx}",
                    label_visibility="collapsed"
                )
                
                # Confidence
                confidence = ocr_result["confidence"]
                confidence_pct = confidence * 100
                
                if confidence >= 0.8:
                    color = "🟢"
                elif confidence >= 0.5:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.markdown(f"**Confidence:** {color} {confidence_pct:.1f}%")
                st.progress(confidence)
            
            st.markdown("---")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page config
    st.set_page_config(
        page_title="Enhanced Underwriting Assistant",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize workspace manager
    if 'workspace_manager' not in st.session_state:
        st.session_state['workspace_manager'] = WorkspaceManager()
    
    workspace_manager = st.session_state['workspace_manager']
    
    # Render header
    render_header()
    
    # Check dependencies
    warnings = []
    if not PYMUPDF_AVAILABLE:
        warnings.append("⚠️ PyMuPDF not installed - PDF processing limited")
    if not DOCX_AVAILABLE:
        warnings.append("⚠️ python-docx not installed - DOCX processing disabled")
    if not OCR_AVAILABLE:
        warnings.append("⚠️ pytesseract not installed - OCR disabled")
    if not OPENAI_AVAILABLE or not Config.DEEPSEEK_API_KEY:
        warnings.append("⚠️ AI analysis disabled - API key not configured")
    
    if warnings:
        with st.expander("⚠️ Configuration Warnings", expanded=False):
            for warning in warnings:
                st.warning(warning)
    
    # Render sidebar and get navigation
    selected_workspace, view_mode = render_sidebar(workspace_manager)
    
    # Render main content based on view mode
    if view_mode == "📤 Upload Document":
        render_upload_view(selected_workspace, workspace_manager)
    
    elif view_mode == "📊 Integrated Analysis":
        render_integrated_analysis(selected_workspace, workspace_manager)
    
    elif view_mode == "📋 Document List":
        render_document_list(selected_workspace, workspace_manager)
    
    # Check for view_doc in session state (from document list)
    if 'view_doc' in st.session_state and selected_workspace:
        render_analysis_view(selected_workspace, st.session_state['view_doc'])
        del st.session_state['view_doc']
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Enhanced Underwriting Assistant v2.8.3-fixed | "
        "Built with Streamlit | "
        "Fixed: use_container_width compatibility"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
