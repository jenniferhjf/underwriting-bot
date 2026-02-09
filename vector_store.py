"""
Vector Store Module
===================
步骤 4 & 6: 向量存储、索引构建、相似度检索

功能:
- 使用FAISS构建向量索引
- 向量检索
- 索引持久化

Version: 1.0.0
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: FAISS未安装，向量检索功能将不可用")


class VectorStore:
    """向量存储和检索"""
    
    def __init__(self, dimension: int = 1536):
        """
        初始化向量存储
        
        Args:
            dimension: 向量维度
        """
        self.dimension = dimension
        self.index = None
        self.documents = []  # 存储文档元数据
        self.index_built = False
        
        if not FAISS_AVAILABLE:
            print("警告: FAISS不可用，无法构建向量索引")
    
    def build_index(self, documents: List[Dict], save_path: Optional[str] = None):
        """
        构建向量索引 - 步骤4
        
        Args:
            documents: 文档列表，每个文档包含chunks字段
            save_path: 索引保存路径
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS未安装")
        
        # 收集所有embeddings和对应的文档信息
        all_embeddings = []
        all_metadata = []
        
        for doc in documents:
            doc_id = doc.get('doc_id')
            filename = doc.get('filename')
            chunks = doc.get('chunks', [])
            
            for chunk in chunks:
                if 'embedding' in chunk:
                    embedding = chunk['embedding']
                    all_embeddings.append(embedding)
                    
                    # 保存元数据
                    all_metadata.append({
                        'doc_id': doc_id,
                        'filename': filename,
                        'chunk_id': chunk['chunk_id'],
                        'text': chunk['text'],
                        'page': chunk.get('page', 'N/A')
                    })
        
        if not all_embeddings:
            raise ValueError("没有找到任何embeddings")
        
        # 转换为numpy数组
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # 创建FAISS索引
        # 使用IndexFlatIP (内积，适合归一化向量)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # 归一化向量（对于余弦相似度）
        faiss.normalize_L2(embeddings_array)
        
        # 添加向量到索引
        self.index.add(embeddings_array)
        
        # 保存元数据
        self.documents = all_metadata
        self.index_built = True
        
        print(f"✅ 索引构建完成: {self.index.ntotal} 个向量")
        
        # 保存索引
        if save_path:
            self.save_index(save_path)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        检索相似文档 - 步骤6
        
        Args:
            query_embedding: 查询向量
            top_k: 返回Top-K个结果
            
        Returns:
            检索结果列表
        """
        if not self.index_built or self.index is None:
            raise ValueError("索引未构建，请先调用build_index()")
        
        # 转换为numpy数组并归一化
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # 执行检索
        scores, indices = self.index.search(query_vector, top_k)
        
        # 组装结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['score'] = float(score)
                results.append(result)
        
        return results
    
    def save_index(self, save_path: str):
        """
        保存索引到文件
        
        Args:
            save_path: 保存路径（例如: data/vector_index.faiss）
        """
        if not self.index_built:
            raise ValueError("索引未构建")
        
        # 保存FAISS索引
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(save_path))
        
        # 保存元数据
        metadata_path = save_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dimension': self.dimension,
                'num_vectors': self.index.ntotal,
                'documents': self.documents
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 索引已保存到: {save_path}")
    
    def load_index(self, load_path: str):
        """
        从文件加载索引
        
        Args:
            load_path: 索引文件路径
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS未安装")
        
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {load_path}")
        
        # 加载FAISS索引
        self.index = faiss.read_index(str(load_path))
        
        # 加载元数据
        metadata_path = load_path.with_suffix('.metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.dimension = metadata['dimension']
                self.documents = metadata['documents']
        
        self.index_built = True
        print(f"✅ 索引已加载: {self.index.ntotal} 个向量")
    
    def get_index_info(self) -> Dict:
        """获取索引信息"""
        if not self.index_built:
            return {
                'status': 'not_built',
                'num_vectors': 0,
                'dimension': self.dimension
            }
        
        return {
            'status': 'ready',
            'num_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'num_documents': len(set(doc['doc_id'] for doc in self.documents)),
            'index_type': 'FAISS IndexFlatIP'
        }
    
    def delete_index(self):
        """删除索引"""
        self.index = None
        self.documents = []
        self.index_built = False
        print("✅ 索引已删除")


class HybridVectorStore(VectorStore):
    """混合向量存储（支持多种检索策略）"""
    
    def __init__(self, dimension: int = 1536):
        super().__init__(dimension)
        self.text_index = {}  # 文本索引（用于关键词检索）
    
    def build_index(self, documents: List[Dict], save_path: Optional[str] = None):
        """构建向量索引和文本索引"""
        # 调用父类方法构建向量索引
        super().build_index(documents, save_path)
        
        # 构建文本索引
        for idx, doc_meta in enumerate(self.documents):
            text = doc_meta['text'].lower()
            words = text.split()
            
            for word in words:
                if word not in self.text_index:
                    self.text_index[word] = []
                self.text_index[word].append(idx)
        
        print(f"✅ 文本索引构建完成: {len(self.text_index)} 个词")
    
    def hybrid_search(self, query_embedding: List[float], 
                     query_text: str, top_k: int = 5,
                     vector_weight: float = 0.7) -> List[Dict]:
        """
        混合检索：向量检索 + 关键词检索
        
        Args:
            query_embedding: 查询向量
            query_text: 查询文本
            top_k: 返回Top-K个结果
            vector_weight: 向量检索权重(0-1)
            
        Returns:
            检索结果列表
        """
        # 向量检索
        vector_results = self.search(query_embedding, top_k * 2)
        
        # 关键词检索
        query_words = query_text.lower().split()
        keyword_scores = {}
        
        for word in query_words:
            if word in self.text_index:
                for idx in self.text_index[word]:
                    keyword_scores[idx] = keyword_scores.get(idx, 0) + 1
        
        # 合并分数
        combined_results = {}
        
        for result in vector_results:
            # 找到对应的索引
            for idx, doc in enumerate(self.documents):
                if (doc['doc_id'] == result['doc_id'] and 
                    doc['chunk_id'] == result['chunk_id']):
                    
                    vector_score = result['score']
                    keyword_score = keyword_scores.get(idx, 0) / max(len(query_words), 1)
                    
                    # 组合分数
                    combined_score = (vector_weight * vector_score + 
                                    (1 - vector_weight) * keyword_score)
                    
                    result['combined_score'] = combined_score
                    result['keyword_score'] = keyword_score
                    combined_results[idx] = result
                    break
        
        # 按组合分数排序
        sorted_results = sorted(combined_results.values(), 
                              key=lambda x: x['combined_score'], 
                              reverse=True)
        
        return sorted_results[:top_k]


if __name__ == '__main__':
    # 测试代码
    print("测试向量存储...")
    
    # 创建测试数据
    test_documents = [
        {
            'doc_id': 'test_001',
            'filename': 'test.pdf',
            'chunks': [
                {
                    'chunk_id': 0,
                    'text': 'This is a test sentence.',
                    'embedding': np.random.randn(1536).tolist()
                },
                {
                    'chunk_id': 1,
                    'text': 'Another test sentence.',
                    'embedding': np.random.randn(1536).tolist()
                }
            ]
        }
    ]
    
    # 构建索引
    vector_store = VectorStore()
    vector_store.build_index(test_documents)
    
    # 测试检索
    query_embedding = np.random.randn(1536).tolist()
    results = vector_store.search(query_embedding, top_k=2)
    print(f"检索到 {len(results)} 个结果")
    
    # 显示索引信息
    info = vector_store.get_index_info()
    print(f"索引信息: {info}")
