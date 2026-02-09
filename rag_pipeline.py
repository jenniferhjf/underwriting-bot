"""
RAG Pipeline Module
===================
步骤 5-8: RAG完整流程编排

功能:
- 问题处理和向量化（步骤5）
- 知识库检索（步骤6）
- Context组装（步骤7）
- LLM生成答案（步骤8）

Version: 1.0.0
"""

from typing import List, Dict, Optional
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .llm_client import LLMClient


class RAGPipeline:
    """RAG流程编排器"""
    
    def __init__(self,
                 embedding_generator: EmbeddingGenerator = None,
                 vector_store: VectorStore = None,
                 llm_client: LLMClient = None):
        """
        初始化RAG Pipeline
        
        Args:
            embedding_generator: Embedding生成器
            vector_store: 向量存储
            llm_client: LLM客户端
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_store = vector_store or VectorStore()
        self.llm_client = llm_client or LLMClient()
    
    def query(self, question: str, top_k: int = 5, 
             return_sources: bool = True) -> Dict:
        """
        执行完整的RAG查询流程
        
        步骤5: 问题向量化
        步骤6: 知识库检索
        步骤7: Context组装
        步骤8: LLM生成答案
        
        Args:
            question: 用户问题
            top_k: 检索Top-K个相关段落
            return_sources: 是否返回来源信息
            
        Returns:
            {
                'answer': str,  # 生成的答案
                'sources': List[Dict],  # 来源信息（如果return_sources=True）
                'context': str  # 使用的上下文
            }
        """
        # === 步骤5: 问题向量化 ===
        query_embedding = self.embedding_generator.generate_embedding(question)
        
        # === 步骤6: 知识库检索 ===
        search_results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # === 步骤7: Context组装 ===
        context = self.build_context(search_results)
        
        # === 步骤8: LLM生成答案 ===
        answer = self.llm_client.generate_answer(question, context)
        
        # 组装返回结果
        result = {
            'answer': answer,
            'context': context
        }
        
        if return_sources:
            result['sources'] = self._format_sources(search_results)
        
        return result
    
    def build_context(self, search_results: List[Dict]) -> str:
        """
        组装Context - 步骤7
        
        Args:
            search_results: 检索结果列表
            
        Returns:
            组装好的上下文字符串
        """
        if not search_results:
            return "[未找到相关信息]"
        
        context_parts = []
        
        for idx, result in enumerate(search_results, 1):
            # 格式化每个结果
            source_info = f"[来源 {idx}] 文档: {result['filename']}"
            if 'page' in result and result['page'] != 'N/A':
                source_info += f", 页码: {result['page']}"
            
            content = f"{source_info}\n{result['text']}\n"
            context_parts.append(content)
        
        return "\n---\n".join(context_parts)
    
    def _format_sources(self, search_results: List[Dict]) -> List[Dict]:
        """格式化来源信息"""
        sources = []
        
        for result in search_results:
            source = {
                'filename': result['filename'],
                'page': result.get('page', 'N/A'),
                'text_preview': result['text'][:200] + '...' if len(result['text']) > 200 else result['text'],
                'score': result.get('score', 0.0)
            }
            sources.append(source)
        
        return sources


class StreamingRAGPipeline(RAGPipeline):
    """支持流式输出的RAG Pipeline"""
    
    def query_stream(self, question: str, top_k: int = 5):
        """
        流式执行RAG查询
        
        Yields:
            生成的文本片段
        """
        # 步骤5: 问题向量化
        query_embedding = self.embedding_generator.generate_embedding(question)
        
        # 步骤6: 知识库检索
        search_results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # 步骤7: Context组装
        context = self.build_context(search_results)
        
        # 步骤8: LLM流式生成答案
        for chunk in self.llm_client.generate_answer_stream(question, context):
            yield chunk


class HybridRAGPipeline(RAGPipeline):
    """混合检索RAG Pipeline（向量+关键词）"""
    
    def query(self, question: str, top_k: int = 5,
             vector_weight: float = 0.7,
             return_sources: bool = True) -> Dict:
        """
        使用混合检索的RAG查询
        
        Args:
            question: 用户问题
            top_k: 检索Top-K个相关段落
            vector_weight: 向量检索权重
            return_sources: 是否返回来源信息
            
        Returns:
            查询结果
        """
        # 步骤5: 问题向量化
        query_embedding = self.embedding_generator.generate_embedding(question)
        
        # 步骤6: 混合检索
        if hasattr(self.vector_store, 'hybrid_search'):
            search_results = self.vector_store.hybrid_search(
                query_embedding, 
                question, 
                top_k=top_k,
                vector_weight=vector_weight
            )
        else:
            # 降级为普通向量检索
            search_results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # 步骤7: Context组装
        context = self.build_context(search_results)
        
        # 步骤8: LLM生成答案
        answer = self.llm_client.generate_answer(question, context)
        
        # 组装返回结果
        result = {
            'answer': answer,
            'context': context
        }
        
        if return_sources:
            result['sources'] = self._format_sources(search_results)
        
        return result


class ConversationalRAGPipeline(RAGPipeline):
    """支持多轮对话的RAG Pipeline"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []
    
    def query(self, question: str, top_k: int = 5,
             use_history: bool = True,
             return_sources: bool = True) -> Dict:
        """
        带对话历史的RAG查询
        
        Args:
            question: 用户问题
            top_k: 检索Top-K个相关段落
            use_history: 是否使用对话历史
            return_sources: 是否返回来源信息
            
        Returns:
            查询结果
        """
        # 如果使用历史，可能需要改写问题
        if use_history and self.conversation_history:
            question = self._rewrite_question_with_history(question)
        
        # 执行标准RAG流程
        result = super().query(question, top_k, return_sources)
        
        # 保存到对话历史
        self.conversation_history.append({
            'question': question,
            'answer': result['answer']
        })
        
        # 限制历史长度
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
        
        return result
    
    def _rewrite_question_with_history(self, question: str) -> str:
        """
        根据对话历史改写问题
        
        例如: "那它的期限呢？" -> "保单的期限是多久？"
        """
        # 简单实现：附加上下文
        if len(self.conversation_history) > 0:
            last_qa = self.conversation_history[-1]
            context_hint = f"[基于上一个问题: {last_qa['question'][:50]}]"
            return f"{context_hint} {question}"
        
        return question
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        if hasattr(self.llm_client, 'clear_history'):
            self.llm_client.clear_history()


# 工厂函数
def create_rag_pipeline(mode: str = 'standard', **kwargs) -> RAGPipeline:
    """
    创建RAG Pipeline
    
    Args:
        mode: 'standard', 'streaming', 'hybrid', 'conversational'
        **kwargs: 其他参数
        
    Returns:
        RAGPipeline实例
    """
    if mode == 'standard':
        return RAGPipeline(**kwargs)
    elif mode == 'streaming':
        return StreamingRAGPipeline(**kwargs)
    elif mode == 'hybrid':
        return HybridRAGPipeline(**kwargs)
    elif mode == 'conversational':
        return ConversationalRAGPipeline(**kwargs)
    else:
        raise ValueError(f"不支持的mode: {mode}")


if __name__ == '__main__':
    # 测试代码
    print("测试RAG Pipeline...")
    
    pipeline = RAGPipeline()
    
    test_question = "这份保单的承保范围是什么？"
    
    # 注意：需要先构建向量索引才能测试
    print(f"问题: {test_question}")
    print("提示: 需要先构建向量索引才能完整测试RAG功能")
