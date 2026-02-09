"""
Enhanced Underwriting RAG System - Standalone Version
Version: 3.1.0 (Single File Edition)
All modules integrated into one file for easy GitHub deployment
"""

import streamlit as st
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================
# Configuration
# ============================================
st.set_page_config(
    page_title="Enhanced Underwriting RAG System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables with defaults
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))

# ============================================
# Module 1: OCR (English Handwriting)
# ============================================
class SimpleOCR:
    """Simple OCR for English text and handwriting"""
    
    def __init__(self, engine='tesseract'):
        self.engine = engine
        self._init_engine()
    
    def _init_engine(self):
        """Initialize OCR engine"""
        try:
            if self.engine == 'tesseract':
                import pytesseract
                self.recognizer = pytesseract
            elif self.engine == 'easyocr':
                import easyocr
                self.recognizer = easyocr.Reader(['en'])
            elif self.engine == 'trocr':
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                from PIL import Image
                self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
                self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
                self.Image = Image
            logger.info(f"OCR engine initialized: {self.engine}")
        except Exception as e:
            logger.warning(f"OCR engine {self.engine} failed to load: {e}")
            self.engine = 'none'
    
    def recognize(self, image_path: str) -> Dict[str, Any]:
        """Recognize text from image"""
        try:
            if self.engine == 'tesseract':
                import pytesseract
                from PIL import Image
                img = Image.open(image_path)
                text = pytesseract.image_to_string(img, lang='eng')
                return {'text': text.strip(), 'confidence': 0.85}
            
            elif self.engine == 'easyocr':
                results = self.recognizer.readtext(image_path)
                text = ' '.join([item[1] for item in results])
                conf = np.mean([item[2] for item in results]) if results else 0
                return {'text': text, 'confidence': conf}
            
            elif self.engine == 'trocr':
                image = self.Image.open(image_path).convert("RGB")
                pixel_values = self.processor(image, return_tensors="pt").pixel_values
                generated_ids = self.model.generate(pixel_values)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return {'text': text, 'confidence': 0.90}
            
            else:
                return {'text': '', 'confidence': 0}
                
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return {'text': '', 'confidence': 0}

# ============================================
# Module 2: Document Preprocessor
# ============================================
class DocumentProcessor:
    """Process PDF and DOCX documents"""
    
    def __init__(self, ocr: Optional[SimpleOCR] = None):
        self.ocr = ocr or SimpleOCR()
    
    def extract_text_from_pdf(self, file_path: str) -> List[Dict]:
        """Extract text from PDF"""
        try:
            import PyPDF2
            pages = []
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    pages.append({
                        'page': i + 1,
                        'text': text,
                        'type': 'electronic'
                    })
            return pages
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return []
    
    def extract_text_from_docx(self, file_path: str) -> List[Dict]:
        """Extract text from DOCX"""
        try:
            import docx
            doc = docx.Document(file_path)
            pages = []
            text = '\n'.join([para.text for para in doc.paragraphs])
            pages.append({
                'page': 1,
                'text': text,
                'type': 'electronic'
            })
            return pages
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            return []
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into chunks"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks

# ============================================
# Module 3: Embeddings
# ============================================
class EmbeddingGenerator:
    """Generate embeddings for text"""
    
    def __init__(self, backend='local'):
        self.backend = backend
        self.dimension = 384  # Default for local
        self._init_backend()
    
    def _init_backend(self):
        """Initialize embedding backend"""
        try:
            if self.backend == 'local':
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.dimension = 384
                logger.info("Local embeddings initialized")
            
            elif self.backend == 'deepseek':
                import openai
                self.client = openai.OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_API_BASE
                )
                self.dimension = 1536
                logger.info("DeepSeek embeddings initialized")
                
        except Exception as e:
            logger.error(f"Embedding init error: {e}")
            # Fallback to random embeddings
            self.backend = 'random'
            self.dimension = 384
    
    def generate(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            if self.backend == 'local':
                embedding = self.model.encode(text)
                return embedding.tolist()
            
            elif self.backend == 'deepseek':
                response = self.client.embeddings.create(
                    input=text,
                    model="deepseek-embedding"
                )
                return response.data[0].embedding
            
            else:
                # Random fallback
                return np.random.randn(self.dimension).tolist()
                
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return np.random.randn(self.dimension).tolist()
    
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return [self.generate(text) for text in texts]

# ============================================
# Module 4: Vector Store (FAISS)
# ============================================
class VectorStore:
    """FAISS-based vector store"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self._init_index()
    
    def _init_index(self):
        """Initialize FAISS index"""
        try:
            import faiss
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"FAISS index initialized (dim={self.dimension})")
        except Exception as e:
            logger.error(f"FAISS init error: {e}")
            self.index = None
    
    def add_embeddings(self, embeddings: List[List[float]], metadata: List[Dict]):
        """Add embeddings to index"""
        if self.index is None:
            return
        
        try:
            embeddings_array = np.array(embeddings).astype('float32')
            self.index.add(embeddings_array)
            self.metadata.extend(metadata)
            logger.info(f"Added {len(embeddings)} vectors to index")
        except Exception as e:
            logger.error(f"Add embeddings error: {e}")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar vectors"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        try:
            query_array = np.array([query_embedding]).astype('float32')
            distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.metadata):
                    results.append({
                        'metadata': self.metadata[idx],
                        'score': float(dist),
                        'text': self.metadata[idx].get('text', '')
                    })
            
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def count(self) -> int:
        """Get number of vectors"""
        return self.index.ntotal if self.index else 0

# ============================================
# Module 5: LLM Client (DeepSeek)
# ============================================
class LLMClient:
    """DeepSeek API client"""
    
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.model = model
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client for DeepSeek"""
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=DEEPSEEK_API_BASE
            )
            logger.info("LLM client initialized")
        except Exception as e:
            logger.error(f"LLM init error: {e}")
            self.client = None
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response from LLM"""
        if not self.client or not self.api_key:
            return "Error: LLM client not initialized. Please set DEEPSEEK_API_KEY."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Error: {str(e)}"

# ============================================
# RAG Pipeline
# ============================================
class RAGSystem:
    """Complete RAG system"""
    
    def __init__(self):
        self.ocr = SimpleOCR(engine=OCR_ENGINE)
        self.processor = DocumentProcessor(self.ocr)
        self.embedder = EmbeddingGenerator(backend=EMBEDDING_BACKEND)
        self.vector_store = VectorStore(dimension=self.embedder.dimension)
        self.llm = LLMClient()
    
    def process_document(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """Process document through Steps 1-4"""
        result = {
            'file_name': file_name,
            'chunks': [],
            'status': 'processing'
        }
        
        try:
            # Step 1-2: Extract text
            if file_path.endswith('.pdf'):
                pages = self.processor.extract_text_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                pages = self.processor.extract_text_from_docx(file_path)
            else:
                result['status'] = 'error'
                result['error'] = 'Unsupported file format'
                return result
            
            # Step 3: Chunk text
            all_text = '\n\n'.join([p['text'] for p in pages])
            chunks = self.processor.chunk_text(all_text, CHUNK_SIZE, CHUNK_OVERLAP)
            
            # Step 4: Generate embeddings
            embeddings = self.embedder.generate_batch(chunks)
            
            # Store
            metadata = [{
                'file_name': file_name,
                'chunk_index': i,
                'text': chunk
            } for i, chunk in enumerate(chunks)]
            
            self.vector_store.add_embeddings(embeddings, metadata)
            
            result['chunks'] = chunks
            result['status'] = 'success'
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer question through Steps 5-8"""
        result = {
            'question': question,
            'answer': '',
            'context': []
        }
        
        try:
            # Step 5: Embed question
            question_embedding = self.embedder.generate(question)
            
            # Step 6: Retrieve similar chunks
            search_results = self.vector_store.search(question_embedding, top_k)
            
            if not search_results:
                result['answer'] = "No relevant information found in the knowledge base."
                return result
            
            # Step 7: Build context
            context_texts = []
            for i, item in enumerate(search_results, 1):
                context_texts.append(f"[Passage {i}]\n{item['text']}\n")
                result['context'].append({
                    'passage_id': i,
                    'text': item['text'],
                    'score': item['score'],
                    'source': item['metadata'].get('file_name', 'Unknown')
                })
            
            context = "\n".join(context_texts)
            
            # Step 8: Generate answer
            prompt = f"""You are an expert underwriting assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the information in the context
2. If the context doesn't contain enough information, say "I don't have enough information"
3. Be concise and precise
4. Cite specific passages if relevant

Answer:"""
            
            answer = self.llm.generate(prompt)
            result['answer'] = answer
            
        except Exception as e:
            logger.error(f"Question answering error: {e}")
            result['answer'] = f"Error: {str(e)}"
        
        return result

# ============================================
# Session State
# ============================================
def init_session_state():
    """Initialize session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.rag_system = RAGSystem()
        st.session_state.documents = []
        st.session_state.chat_history = []
        
        # Create data directory
        os.makedirs("data/uploads", exist_ok=True)
        logger.info("Session initialized")

# ============================================
# UI Components
# ============================================
def render_sidebar():
    """Render sidebar"""
    with st.sidebar:
        st.title("📄 RAG System")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["📤 Upload", "🔍 Vector DB", "💬 Chat", "📊 Stats"],
            key="nav"
        )
        
        st.markdown("---")
        st.subheader("System Info")
        
        if st.session_state.rag_system:
            st.metric("Documents", len(st.session_state.documents))
            st.metric("Vectors", st.session_state.rag_system.vector_store.count())
            st.metric("Chats", len(st.session_state.chat_history))
        
        st.markdown("---")
        with st.expander("⚙️ Config"):
            st.write(f"**Embeddings:** {EMBEDDING_BACKEND}")
            st.write(f"**OCR:** {OCR_ENGINE}")
            st.write(f"**LLM:** {DEEPSEEK_MODEL}")
        
        return page

def page_upload():
    """Upload page"""
    st.title("📤 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=['pdf', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🚀 Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            for file in uploaded_files:
                # Save file
                file_path = f"data/uploads/{file.name}"
                with open(file_path, 'wb') as f:
                    f.write(file.getbuffer())
                
                # Process
                result = st.session_state.rag_system.process_document(file_path, file.name)
                
                if result['status'] == 'success':
                    st.session_state.documents.append(result)
                    st.success(f"✅ Processed {file.name}: {len(result['chunks'])} chunks")
                else:
                    st.error(f"❌ Error processing {file.name}: {result.get('error', 'Unknown')}")
    
    # Show processed documents
    if st.session_state.documents:
        st.markdown("---")
        st.subheader("📋 Processed Documents")
        for doc in st.session_state.documents:
            with st.expander(f"📄 {doc['file_name']}"):
                st.write(f"**Chunks:** {len(doc['chunks'])}")
                if doc['chunks']:
                    st.text_area("Sample", doc['chunks'][0][:500], height=150)

def page_vector_db():
    """Vector DB page"""
    st.title("🔍 Vector Database")
    
    rag = st.session_state.rag_system
    
    if rag.vector_store.count() == 0:
        st.warning("⚠️ No vectors in database. Please upload documents first.")
        return
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Vectors", rag.vector_store.count())
    col2.metric("Dimension", rag.vector_store.dimension)
    col3.metric("Files", len(st.session_state.documents))
    
    st.markdown("---")
    st.subheader("📊 Metadata")
    
    if rag.vector_store.metadata:
        df = pd.DataFrame(rag.vector_store.metadata)
        st.dataframe(df, use_container_width=True)

def page_chat():
    """Chat page"""
    st.title("💬 Chat Interface")
    
    if st.session_state.rag_system.vector_store.count() == 0:
        st.warning("⚠️ No documents loaded. Please upload documents first.")
        return
    
    # Chat history
    st.subheader("📜 Chat History")
    for i, chat in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**Q{i+1}:** {chat['question']}")
            st.markdown(f"**A{i+1}:** {chat['answer']}")
            
            if chat.get('context'):
                with st.expander(f"📚 Context ({len(chat['context'])} passages)"):
                    for p in chat['context']:
                        st.markdown(f"**Passage {p['passage_id']}** (Score: {p['score']:.4f})")
                        st.text(p['text'][:300])
                        st.caption(f"Source: {p['source']}")
            
            st.markdown("---")
    
    # Question input
    st.subheader("❓ Ask a Question")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input("Enter your question", key="question")
    with col2:
        top_k = st.number_input("Top-K", 1, 10, TOP_K)
    
    if st.button("🚀 Ask", type="primary") and question:
        with st.spinner("Processing..."):
            result = st.session_state.rag_system.answer_question(question, top_k)
            st.session_state.chat_history.append(result)
            st.rerun()

def page_stats():
    """Statistics page"""
    st.title("📊 Statistics")
    
    if not st.session_state.documents:
        st.warning("⚠️ No documents processed yet.")
        return
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(st.session_state.documents))
    col2.metric("Total Chunks", sum(len(d['chunks']) for d in st.session_state.documents))
    col3.metric("Questions Asked", len(st.session_state.chat_history))
    
    st.markdown("---")
    st.subheader("📈 Document Details")
    
    data = [{
        'Document': doc['file_name'],
        'Chunks': len(doc['chunks']),
        'Status': doc['status']
    } for doc in st.session_state.documents]
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

# ============================================
# Main App
# ============================================
def main():
    """Main application"""
    init_session_state()
    
    # Custom CSS
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    page = render_sidebar()
    
    # Pages
    if page == "📤 Upload":
        page_upload()
    elif page == "🔍 Vector DB":
        page_vector_db()
    elif page == "💬 Chat":
        page_chat()
    elif page == "📊 Stats":
        page_stats()

if __name__ == "__main__":
    main()
