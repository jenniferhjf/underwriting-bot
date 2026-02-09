"""
Document Preprocessor Module
============================
步骤 1-2: 文档处理、电子文本vs手写文本分离、文本分块

功能:
- PDF/DOCX文档解析
- 电子文本提取
- 手写图像识别和分离
- 文本分块 (Chunking)
- OCR识别

Version: 1.0.0
"""

import io
import base64
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

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


class DocumentPreprocessor:
    """文档预处理器"""
    
    def __init__(self):
        self.chunk_size = 500
        self.chunk_overlap = 50
    
    def process_document(self, file_bytes: bytes, filename: str) -> Dict:
        """
        处理文档 - 步骤1
        返回: {
            'doc_id': str,
            'filename': str,
            'electronic_text': str,
            'handwriting_images': List[Dict],
            'metadata': Dict
        }
        """
        # 生成文档ID
        doc_id = self._generate_doc_id(filename)
        
        # 根据文件类型处理
        file_ext = Path(filename).suffix.lower()
        
        if file_ext == '.pdf':
            electronic_text, images = self._extract_from_pdf(file_bytes)
        elif file_ext == '.docx':
            electronic_text = self._extract_from_docx(file_bytes)
            images = []
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
        
        # 提取元数据
        metadata = self._extract_metadata(filename, electronic_text)
        
        return {
            'doc_id': doc_id,
            'filename': filename,
            'electronic_text': electronic_text,
            'handwriting_images': images,
            'metadata': metadata
        }
    
    def chunk_text(self, text: str, chunk_size: int = None, 
                   chunk_overlap: int = None) -> List[Dict]:
        """
        文本分块 - 步骤2
        返回: List[{
            'chunk_id': int,
            'text': str,
            'start_char': int,
            'end_char': int
        }]
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 尝试在句子边界分割
            if end < len(text):
                # 查找最近的句号、问号、感叹号
                for i in range(end, start + chunk_size // 2, -1):
                    if text[i] in '.!?\n。！？':
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'start_char': start,
                    'end_char': end
                })
                chunk_id += 1
            
            # 移动到下一个块，考虑重叠
            start = end - chunk_overlap
        
        return chunks
    
    def perform_ocr(self, image_data: str) -> Dict:
        """
        执行OCR识别
        返回: {'text': str, 'confidence': float}
        """
        if not OCR_AVAILABLE:
            return {
                'text': '[OCR不可用 - 请安装pytesseract]',
                'confidence': 0.0
            }
        
        try:
            # 解码base64图像
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # 执行OCR
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            
            # 获取置信度
            data = pytesseract.image_to_data(image, lang='chi_sim+eng', 
                                            output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': text.strip(),
                'confidence': avg_confidence / 100.0
            }
        
        except Exception as e:
            return {
                'text': f'[OCR错误: {str(e)}]',
                'confidence': 0.0
            }
    
    def _extract_from_pdf(self, file_bytes: bytes) -> Tuple[str, List[Dict]]:
        """从PDF提取文本和图像"""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF未安装，无法处理PDF")
        
        text_content = []
        images = []
        
        try:
            pdf_document = fitz.open(stream=file_bytes, filetype='pdf')
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # 提取文本
                page_text = page.get_text()
                if page_text.strip():
                    text_content.append(page_text)
                
                # 提取图像
                image_list = page.get_images()
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image['image']
                        image_ext = base_image['ext']
                        
                        # 转换为base64
                        image_base64 = base64.b64encode(image_bytes).decode()
                        
                        images.append({
                            'id': f'page_{page_num + 1}_img_{img_index + 1}',
                            'data': f'data:image/{image_ext};base64,{image_base64}',
                            'page': page_num + 1,
                            'size': len(image_bytes),
                            'type': 'embedded'
                        })
                    except Exception as e:
                        print(f"提取图像错误: {e}")
                        continue
            
            pdf_document.close()
        
        except Exception as e:
            raise Exception(f"PDF处理错误: {e}")
        
        return '\n\n'.join(text_content), images
    
    def _extract_from_docx(self, file_bytes: bytes) -> str:
        """从DOCX提取文本"""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx未安装，无法处理DOCX")
        
        try:
            doc = Document(io.BytesIO(file_bytes))
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            return '\n\n'.join(text_content)
        
        except Exception as e:
            raise Exception(f"DOCX处理错误: {e}")
    
    def _generate_doc_id(self, filename: str) -> str:
        """生成文档ID"""
        timestamp = datetime.now().isoformat()
        content = f"{filename}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _extract_metadata(self, filename: str, text: str) -> Dict:
        """提取文档元数据"""
        # 简单的元数据提取
        metadata = {
            'filename': filename,
            'text_length': len(text),
            'processed_at': datetime.now().isoformat()
        }
        
        # 尝试从文件名提取信息
        if 'hull' in filename.lower():
            metadata['category'] = 'Marine Insurance'
        elif 'cargo' in filename.lower():
            metadata['category'] = 'Cargo Insurance'
        else:
            metadata['category'] = 'General Insurance'
        
        # 提取年份
        import re
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            metadata['year'] = int(year_match.group())
        else:
            metadata['year'] = datetime.now().year
        
        return metadata


if __name__ == '__main__':
    # 测试代码
    preprocessor = DocumentPreprocessor()
    
    # 测试文本分块
    sample_text = "这是一个测试文本。" * 100
    chunks = preprocessor.chunk_text(sample_text)
    print(f"生成了 {len(chunks)} 个文本块")
