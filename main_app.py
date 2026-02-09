"""
Enhanced Underwriting RAG System - Main Application
====================================================
包含两个主要功能：
1. 数据预处理区 (Admin) - 上传文档、预处理、构建索引
2. RAG问答区 (User) - 问答检索、智能回答

Version: 1.0.0
Date: 2026-02-09
"""

import streamlit as st
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 导入自定义模块
try:
    from modules.preprocessor import DocumentPreprocessor
    from modules.vector_store import VectorStore
    from modules.embeddings import EmbeddingGenerator
    from modules.llm_client import LLMClient
    from modules.rag_pipeline import RAGPipeline
except ImportError:
    st.error("❌ 模块导入失败，请确保所有依赖已安装")
    st.stop()


# ============================================================================
# 配置
# ============================================================================

class Config:
    """应用配置"""
    
    # 数据目录
    DATA_DIR = Path("data")
    ELECTRONIC_DATA_FILE = DATA_DIR / "electronic_data.json"
    HANDWRITING_DATA_FILE = DATA_DIR / "handwriting_data.json"
    VECTOR_INDEX_FILE = DATA_DIR / "vector_index.faiss"
    METADATA_FILE = DATA_DIR / "metadata.json"
    
    # API配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    
    # Embedding配置
    EMBEDDING_MODEL = "text-embedding-3-small"  # 或使用本地模型
    EMBEDDING_DIMENSION = 1536
    
    # RAG配置
    TOP_K = 5  # 检索Top-K个相关段落
    CHUNK_SIZE = 500  # 文本分块大小
    CHUNK_OVERLAP = 50  # 分块重叠
    
    @classmethod
    def init_storage(cls):
        """初始化存储目录"""
        cls.DATA_DIR.mkdir(exist_ok=True)


Config.init_storage()


# ============================================================================
# 会话状态初始化
# ============================================================================

def init_session_state():
    """初始化会话状态"""
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = DocumentPreprocessor()
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    if 'embedding_generator' not in st.session_state:
        st.session_state.embedding_generator = EmbeddingGenerator()
    
    if 'llm_client' not in st.session_state:
        st.session_state.llm_client = LLMClient()
    
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    
    if 'knowledge_base_loaded' not in st.session_state:
        st.session_state.knowledge_base_loaded = False
    
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}


# ============================================================================
# 数据预处理区 (步骤 1-4)
# ============================================================================

def render_preprocessing_section():
    """渲染数据预处理区"""
    st.header("📤 数据预处理区 (Admin)")
    st.markdown("上传文档，进行预处理，构建知识库索引")
    st.markdown("---")
    
    # Tab: 上传与预处理 | 索引管理 | 数据查看
    tab1, tab2, tab3 = st.tabs(["📄 上传与预处理", "🔍 索引管理", "📊 数据查看"])
    
    with tab1:
        render_upload_and_process()
    
    with tab2:
        render_index_management()
    
    with tab3:
        render_data_viewer()


def render_upload_and_process():
    """上传文档并预处理"""
    st.subheader("步骤 1-4: 文档预处理流程")
    
    st.markdown("""
    **处理流程：**
    1. 🔍 识别电子文本 vs 手写文本
    2. ✂️ 文本分块 (Chunking)
    3. 🧮 向量化 (Embeddings)
    4. 💾 保存到数据库 + 构建索引
    """)
    
    # 文件上传
    uploaded_files = st.file_uploader(
        "上传文档 (PDF/DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="支持批量上传多个文档"
    )
    
    if uploaded_files:
        st.success(f"✅ 已选择 {len(uploaded_files)} 个文件")
        
        # 显示文件列表
        with st.expander("📋 查看文件列表"):
            for f in uploaded_files:
                st.markdown(f"- **{f.name}** ({f.size / 1024:.2f} KB)")
        
        # 处理配置
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("分块大小", min_value=100, max_value=2000, value=Config.CHUNK_SIZE)
        with col2:
            chunk_overlap = st.number_input("分块重叠", min_value=0, max_value=200, value=Config.CHUNK_OVERLAP)
        
        # 开始处理按钮
        if st.button("🚀 开始预处理", type="primary"):
            process_documents(uploaded_files, chunk_size, chunk_overlap)


def process_documents(uploaded_files, chunk_size: int, chunk_overlap: int):
    """
    执行完整的文档预处理流程
    步骤 1: 文件处理 (电子文本 vs 手写文本分离)
    步骤 2: 文本分块
    步骤 3: 向量化
    步骤 4: 构建索引
    """
    preprocessor = st.session_state.preprocessor
    embedding_generator = st.session_state.embedding_generator
    vector_store = st.session_state.vector_store
    
    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_electronic_data = []
    all_handwriting_data = []
    
    for idx, uploaded_file in enumerate(uploaded_files):
        # 更新进度
        progress = (idx + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"处理文件 {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        try:
            # === 步骤 1: 文件处理和分离 ===
            st.info(f"📖 步骤 1/4: 提取和分离内容 - {uploaded_file.name}")
            
            file_bytes = uploaded_file.read()
            result = preprocessor.process_document(
                file_bytes=file_bytes,
                filename=uploaded_file.name
            )
            
            electronic_text = result.get("electronic_text", "")
            handwriting_images = result.get("handwriting_images", [])
            metadata = result.get("metadata", {})
            
            st.success(f"✅ 步骤 1 完成: 电子文本 {len(electronic_text)} 字符, 手写图像 {len(handwriting_images)} 张")
            
            # === 步骤 2: 文本分块 ===
            if electronic_text:
                st.info(f"✂️ 步骤 2/4: 文本分块 - {uploaded_file.name}")
                
                chunks = preprocessor.chunk_text(
                    text=electronic_text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                st.success(f"✅ 步骤 2 完成: 生成 {len(chunks)} 个文本块")
                
                # === 步骤 3: 向量化 ===
                st.info(f"🧮 步骤 3/4: 向量化 - {uploaded_file.name}")
                
                embeddings = []
                for chunk in chunks:
                    embedding = embedding_generator.generate_embedding(chunk["text"])
                    chunk["embedding"] = embedding
                    embeddings.append(embedding)
                
                st.success(f"✅ 步骤 3 完成: 生成 {len(embeddings)} 个向量")
                
                # 保存电子文本数据
                doc_data = {
                    "doc_id": result["doc_id"],
                    "filename": uploaded_file.name,
                    "metadata": metadata,
                    "chunks": chunks
                }
                all_electronic_data.append(doc_data)
            
            # 处理手写图像
            if handwriting_images:
                st.info(f"✍️ 处理手写图像 - {uploaded_file.name}")
                
                for img in handwriting_images:
                    # OCR识别
                    ocr_result = preprocessor.perform_ocr(img["data"])
                    img["ocr_text"] = ocr_result["text"]
                    img["confidence"] = ocr_result["confidence"]
                    img["doc_id"] = result["doc_id"]
                
                all_handwriting_data.extend(handwriting_images)
                st.success(f"✅ 手写图像处理完成: {len(handwriting_images)} 张")
        
        except Exception as e:
            st.error(f"❌ 处理文件失败: {uploaded_file.name}")
            st.error(f"错误: {str(e)}")
            continue
    
    # === 步骤 4: 构建索引并保存 ===
    if all_electronic_data:
        st.info("💾 步骤 4/4: 保存数据并构建向量索引")
        
        # 保存JSON数据
        save_processed_data(all_electronic_data, all_handwriting_data)
        
        # 构建向量索引
        vector_store.build_index(all_electronic_data)
        
        st.success("✅ 步骤 4 完成: 数据已保存，向量索引已构建")
        st.success(f"📊 总计处理: {len(all_electronic_data)} 个文档, {len(all_handwriting_data)} 张手写图像")
        
        # 更新会话状态
        st.session_state.knowledge_base_loaded = True
        st.balloons()
    else:
        st.warning("⚠️ 没有提取到有效的电子文本数据")
    
    progress_bar.empty()
    status_text.empty()


def save_processed_data(electronic_data: List[Dict], handwriting_data: List[Dict]):
    """保存预处理后的数据"""
    try:
        # 保存电子文本数据
        with open(Config.ELECTRONIC_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                "processed_at": datetime.now().isoformat(),
                "total_documents": len(electronic_data),
                "documents": electronic_data
            }, f, indent=2, ensure_ascii=False)
        
        # 保存手写数据
        with open(Config.HANDWRITING_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                "processed_at": datetime.now().isoformat(),
                "total_images": len(handwriting_data),
                "images": handwriting_data
            }, f, indent=2, ensure_ascii=False)
        
        # 保存元数据
        metadata = {
            "last_updated": datetime.now().isoformat(),
            "total_documents": len(electronic_data),
            "total_handwriting_images": len(handwriting_data)
        }
        with open(Config.METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        st.success(f"✅ 数据已保存到 {Config.DATA_DIR}")
    
    except Exception as e:
        st.error(f"❌ 保存数据失败: {e}")


def render_index_management():
    """索引管理"""
    st.subheader("🔍 向量索引管理")
    
    # 检查索引状态
    if Config.VECTOR_INDEX_FILE.exists():
        st.success("✅ 向量索引已存在")
        
        # 显示索引信息
        vector_store = st.session_state.vector_store
        info = vector_store.get_index_info()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("索引向量数", info.get("num_vectors", "N/A"))
        with col2:
            st.metric("向量维度", info.get("dimension", "N/A"))
        with col3:
            st.metric("索引类型", info.get("index_type", "N/A"))
        
        # 重建索引
        if st.button("🔄 重建索引"):
            with st.spinner("重建索引中..."):
                rebuild_index()
    else:
        st.warning("⚠️ 向量索引不存在，请先进行文档预处理")
        
        if st.button("🏗️ 从现有数据构建索引"):
            if Config.ELECTRONIC_DATA_FILE.exists():
                with st.spinner("构建索引中..."):
                    build_index_from_existing_data()
            else:
                st.error("❌ 没有找到预处理数据，请先上传文档")


def rebuild_index():
    """重建向量索引"""
    try:
        if Config.ELECTRONIC_DATA_FILE.exists():
            with open(Config.ELECTRONIC_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            vector_store = st.session_state.vector_store
            vector_store.build_index(data["documents"])
            
            st.success("✅ 索引重建成功")
        else:
            st.error("❌ 没有找到数据文件")
    except Exception as e:
        st.error(f"❌ 重建索引失败: {e}")


def build_index_from_existing_data():
    """从现有数据构建索引"""
    rebuild_index()


def render_data_viewer():
    """数据查看器"""
    st.subheader("📊 数据查看")
    
    # 选择数据类型
    data_type = st.radio("选择数据类型", ["电子文本", "手写图像", "元数据"])
    
    if data_type == "电子文本":
        if Config.ELECTRONIC_DATA_FILE.exists():
            with open(Config.ELECTRONIC_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            st.json(data, expanded=False)
            st.download_button(
                "📥 下载电子文本数据",
                data=json.dumps(data, indent=2, ensure_ascii=False),
                file_name="electronic_data.json",
                mime="application/json"
            )
        else:
            st.info("📭 暂无电子文本数据")
    
    elif data_type == "手写图像":
        if Config.HANDWRITING_DATA_FILE.exists():
            with open(Config.HANDWRITING_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            st.markdown(f"**总计:** {data.get('total_images', 0)} 张图像")
            
            # 显示前几张图像
            for idx, img in enumerate(data.get("images", [])[:5]):
                with st.expander(f"图像 {idx + 1}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**文档ID:** {img.get('doc_id', 'N/A')}")
                        st.markdown(f"**页码:** {img.get('page', 'N/A')}")
                        st.markdown(f"**置信度:** {img.get('confidence', 0) * 100:.1f}%")
                    with col2:
                        st.markdown("**OCR文本:**")
                        st.text(img.get('ocr_text', 'N/A')[:200])
        else:
            st.info("📭 暂无手写图像数据")
    
    else:  # 元数据
        if Config.METADATA_FILE.exists():
            with open(Config.METADATA_FILE, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            st.json(metadata)
        else:
            st.info("📭 暂无元数据")


# ============================================================================
# RAG 问答区 (步骤 5-8)
# ============================================================================

def render_rag_section():
    """渲染RAG问答区"""
    st.header("💬 RAG 问答区 (User)")
    st.markdown("基于知识库的智能问答系统")
    st.markdown("---")
    
    # 检查知识库是否已加载
    if not check_knowledge_base():
        st.warning("⚠️ 知识库尚未加载或不存在")
        st.info("👉 请先在 **数据预处理区** 上传文档并完成预处理")
        return
    
    # 加载知识库
    if not st.session_state.knowledge_base_loaded:
        with st.spinner("📚 加载知识库中..."):
            load_knowledge_base()
    
    st.success("✅ 知识库已加载")
    
    # 显示知识库统计
    display_kb_stats()
    
    st.markdown("---")
    
    # 问答界面
    render_qa_interface()


def check_knowledge_base() -> bool:
    """检查知识库是否存在"""
    return (Config.ELECTRONIC_DATA_FILE.exists() and 
            Config.VECTOR_INDEX_FILE.exists())


def load_knowledge_base():
    """加载知识库到内存"""
    try:
        # 加载向量存储
        vector_store = st.session_state.vector_store
        vector_store.load_index(Config.VECTOR_INDEX_FILE)
        
        # 加载文档数据
        with open(Config.ELECTRONIC_DATA_FILE, 'r', encoding='utf-8') as f:
            electronic_data = json.load(f)
        
        st.session_state.electronic_data = electronic_data
        st.session_state.knowledge_base_loaded = True
        
        st.success("✅ 知识库加载完成")
    
    except Exception as e:
        st.error(f"❌ 加载知识库失败: {e}")
        st.session_state.knowledge_base_loaded = False


def display_kb_stats():
    """显示知识库统计信息"""
    try:
        with open(Config.METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 文档数", metadata.get("total_documents", 0))
        with col2:
            st.metric("✍️ 手写图像", metadata.get("total_handwriting_images", 0))
        with col3:
            last_updated = metadata.get("last_updated", "N/A")
            if last_updated != "N/A":
                last_updated = last_updated.split("T")[0]
            st.metric("🔄 最后更新", last_updated)
    
    except Exception as e:
        st.warning(f"无法加载统计信息: {e}")


def render_qa_interface():
    """问答界面"""
    st.subheader("🤔 提出您的问题")
    
    # 问题输入
    query = st.text_area(
        "输入您的问题:",
        height=100,
        placeholder="例如：这份保单的承保范围是什么？",
        help="输入关于保险文档的问题"
    )
    
    # 高级选项
    with st.expander("⚙️ 高级选项"):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("检索Top-K", min_value=1, max_value=10, value=Config.TOP_K)
        with col2:
            show_sources = st.checkbox("显示来源", value=True)
    
    # 提交按钮
    if st.button("🔍 搜索答案", type="primary"):
        if query.strip():
            with st.spinner("🤔 思考中..."):
                answer_question(query, top_k, show_sources)
        else:
            st.warning("⚠️ 请输入问题")


def answer_question(query: str, top_k: int, show_sources: bool):
    """
    执行完整的RAG流程回答问题
    步骤 5: 问题向量化
    步骤 6: 知识库检索
    步骤 7: 组装Context
    步骤 8: LLM生成答案
    """
    try:
        rag_pipeline = st.session_state.rag_pipeline
        
        # === 步骤 5: 问题向量化 ===
        st.info("🧮 步骤 5/8: 问题向量化...")
        query_embedding = st.session_state.embedding_generator.generate_embedding(query)
        st.success("✅ 步骤 5 完成")
        
        # === 步骤 6: 知识库检索 ===
        st.info(f"🔍 步骤 6/8: 检索Top-{top_k}相关段落...")
        vector_store = st.session_state.vector_store
        search_results = vector_store.search(query_embedding, top_k=top_k)
        st.success(f"✅ 步骤 6 完成: 找到 {len(search_results)} 个相关段落")
        
        # === 步骤 7: 组装Context ===
        st.info("📝 步骤 7/8: 组装上下文...")
        context = rag_pipeline.build_context(search_results)
        st.success("✅ 步骤 7 完成")
        
        # === 步骤 8: LLM生成答案 ===
        st.info("🤖 步骤 8/8: 生成答案...")
        llm_client = st.session_state.llm_client
        answer = llm_client.generate_answer(query, context)
        st.success("✅ 步骤 8 完成")
        
        # 显示答案
        st.markdown("---")
        st.subheader("💡 答案")
        st.markdown(answer)
        
        # 显示来源
        if show_sources:
            st.markdown("---")
            st.subheader("📚 相关来源")
            for idx, result in enumerate(search_results):
                with st.expander(f"来源 {idx + 1}: {result['filename']} (相似度: {result['score']:.3f})"):
                    st.markdown(f"**页码:** {result.get('page', 'N/A')}")
                    st.markdown(f"**内容:**")
                    st.text(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])
    
    except Exception as e:
        st.error(f"❌ 回答问题失败: {e}")
        if Config.DEBUG_MODE:
            st.exception(e)


# ============================================================================
# 主应用
# ============================================================================

def main():
    """主应用入口"""
    
    # 页面配置
    st.set_page_config(
        page_title="Enhanced Underwriting RAG System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 初始化会话状态
    init_session_state()
    
    # 标题
    st.title("🔍 Enhanced Underwriting RAG System")
    st.markdown("### AI驱动的保险文档知识库与智能问答系统")
    
    # 侧边栏 - 模式选择
    st.sidebar.title("📂 功能选择")
    mode = st.sidebar.radio(
        "选择模式",
        ["💬 RAG 问答区", "📤 数据预处理区"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 系统状态")
    
    # 显示知识库状态
    if check_knowledge_base():
        st.sidebar.success("✅ 知识库已就绪")
    else:
        st.sidebar.warning("⚠️ 知识库未就绪")
    
    # 显示API状态
    if Config.DEEPSEEK_API_KEY:
        st.sidebar.success("✅ API已配置")
    else:
        st.sidebar.warning("⚠️ API未配置")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ 系统信息")
    st.sidebar.markdown(f"**版本:** 1.0.0")
    st.sidebar.markdown(f"**更新:** 2026-02-09")
    
    # 主要内容区域
    st.markdown("---")
    
    if mode == "📤 数据预处理区":
        render_preprocessing_section()
    else:
        render_rag_section()
    
    # 页脚
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Enhanced Underwriting RAG System v1.0.0 | "
        "Built with Streamlit & DeepSeek"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
