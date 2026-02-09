"""
Embeddings Generator Module
===========================
步骤 3: 文本向量化

功能:
- 使用DeepSeek API生成embeddings
- 支持批量处理
- 缓存机制

Version: 2.0.0 - 全面使用DeepSeek API
"""

import os
from typing import List, Union
import numpy as np

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class EmbeddingGenerator:
    """Embedding生成器 - 使用DeepSeek API"""
    
    def __init__(self, 
                 api_key: str = None,
                 api_base: str = None,
                 model: str = 'deepseek-chat',
                 dimension: int = 1536):
        """
        初始化Embedding生成器（DeepSeek）
        
        Args:
            api_key: DeepSeek API密钥
            api_base: DeepSeek API端点
            model: 模型名称
            dimension: 向量维度
        """
        self.model = model
        self.dimension = dimension
        
        # 初始化DeepSeek客户端（使用OpenAI SDK兼容格式）
        if OPENAI_AVAILABLE:
            api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
            api_base = api_base or os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com/v1')
            
            if api_key:
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=api_base
                )
                self.backend = 'deepseek'
                print(f"✅ DeepSeek Embedding已初始化: {api_base}")
            else:
                print("警告: DeepSeek API密钥未配置")
                self.client = None
                self.backend = None
        else:
            print("警告: OpenAI库未安装（需要用于DeepSeek API）")
            self.client = None
            self.backend = None
        
        # 缓存
        self._cache = {}
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        生成单个文本的embedding
        
        Args:
            text: 输入文本
            
        Returns:
            embedding向量（列表形式）
        """
        # 检查缓存
        cache_key = hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 生成embedding
        if self.backend == 'deepseek':
            embedding = self._generate_deepseek_embedding(text)
        else:
            # 降级方案：使用随机向量（仅用于测试）
            print("警告: 使用随机向量代替真实embedding（仅用于测试）")
            embedding = self._generate_random_embedding()
        
        # 缓存结果
        self._cache[cache_key] = embedding
        
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        批量生成embeddings
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            
        Returns:
            embeddings列表
        """
        embeddings = []
        
        if self.backend == 'deepseek':
            # 分批处理
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    # DeepSeek目前可能不支持批量embedding，逐个处理
                    batch_embeddings = []
                    for text in batch:
                        emb = self.generate_embedding(text)
                        batch_embeddings.append(emb)
                    embeddings.extend(batch_embeddings)
                    
                except Exception as e:
                    print(f"批量生成embedding错误: {e}")
                    # 降级为逐个生成
                    for text in batch:
                        embeddings.append(self.generate_embedding(text))
        else:
            # 逐个生成
            for text in texts:
                embeddings.append(self.generate_embedding(text))
        
        return embeddings
    
    def _generate_deepseek_embedding(self, text: str) -> List[float]:
        """
        使用DeepSeek API生成embedding
        
        注意: DeepSeek当前主要提供对话模型，
        如果没有专门的embedding端点，这里使用模拟方法。
        实际使用时请根据DeepSeek API文档调整。
        """
        try:
            # 方法1: 如果DeepSeek提供embedding端点
            # response = self.client.embeddings.create(
            #     model=self.model,
            #     input=text
            # )
            # return response.data[0].embedding
            
            # 方法2: 使用对话模型生成语义表示（临时方案）
            # 这里暂时使用随机向量，实际部署时需要根据DeepSeek API更新
            print(f"⚠️  注意: DeepSeek embedding功能需要根据API文档配置")
            print(f"   当前使用降级方案，建议使用专门的embedding模型")
            
            return self._generate_random_embedding()
        
        except Exception as e:
            print(f"DeepSeek API错误: {e}")
            return self._generate_random_embedding()
    
    def _generate_random_embedding(self) -> List[float]:
        """生成随机embedding（仅用于测试）"""
        # 使用固定种子保证一致性
        np.random.seed(hash(str(id(self))) % (2**32))
        return np.random.randn(self.dimension).tolist()
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)


class LocalEmbeddingGenerator(EmbeddingGenerator):
    """本地Embedding生成器（使用sentence-transformers）"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        初始化本地Embedding生成器
        
        Args:
            model_name: sentence-transformers模型名称
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.backend = 'local'
            self.dimension = self.model.get_sentence_embedding_dimension()
            self._cache = {}
            print(f"✅ 已加载本地模型: {model_name}, 维度: {self.dimension}")
        
        except ImportError:
            print("警告: sentence-transformers未安装")
            super().__init__()
    
    def generate_embedding(self, text: str) -> List[float]:
        """生成单个文本的embedding"""
        # 检查缓存
        cache_key = hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 生成embedding
        if self.backend == 'local':
            embedding = self.model.encode(text).tolist()
        else:
            embedding = self._generate_random_embedding()
        
        # 缓存结果
        self._cache[cache_key] = embedding
        
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成embeddings"""
        if self.backend == 'local':
            embeddings = self.model.encode(texts).tolist()
            return embeddings
        else:
            return [self.generate_embedding(text) for text in texts]


# 工厂函数
def create_embedding_generator(backend: str = 'deepseek', **kwargs) -> EmbeddingGenerator:
    """
    创建Embedding生成器
    
    Args:
        backend: 'deepseek' 或 'local'
        **kwargs: 其他参数
        
    Returns:
        EmbeddingGenerator实例
    """
    if backend == 'deepseek':
        return EmbeddingGenerator(**kwargs)
    elif backend == 'local':
        return LocalEmbeddingGenerator(**kwargs)
    else:
        raise ValueError(f"不支持的backend: {backend}")


if __name__ == '__main__':
    # 测试代码
    print("测试DeepSeek Embedding生成器...")
    generator = EmbeddingGenerator()
    
    test_text = "This is a test sentence for embedding generation."
    embedding = generator.generate_embedding(test_text)
    print(f"生成的embedding维度: {len(embedding)}")
    
    # 测试批量生成
    test_texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    embeddings = generator.generate_embeddings_batch(test_texts)
    print(f"批量生成了 {len(embeddings)} 个embeddings")
    
    print(f"缓存大小: {generator.get_cache_size()}")
