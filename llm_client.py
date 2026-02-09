"""
LLM Client Module
=================
步骤 8: 调用大语言模型生成答案

功能:
- 调用DeepSeek API
- Prompt工程
- 流式输出支持

Version: 1.0.0
"""

import os
from typing import List, Dict, Optional, Generator

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMClient:
    """大语言模型客户端"""
    
    def __init__(self, 
                 api_key: str = None,
                 api_base: str = None,
                 model: str = 'deepseek-chat'):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
        """
        self.model = model
        
        # 获取API配置
        api_key = api_key or os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY')
        api_base = api_base or os.getenv('DEEPSEEK_API_BASE') or os.getenv('OPENAI_API_BASE')
        
        if not api_key:
            print("警告: API密钥未配置")
            self.client = None
        elif not OPENAI_AVAILABLE:
            print("警告: OpenAI库未安装")
            self.client = None
        else:
            # 初始化客户端
            if api_base:
                self.client = OpenAI(api_key=api_key, base_url=api_base)
            else:
                self.client = OpenAI(api_key=api_key)
            print(f"✅ LLM客户端已初始化: {model}")
    
    def generate_answer(self, question: str, context: str, 
                       max_tokens: int = 1000,
                       temperature: float = 0.3) -> str:
        """
        生成答案 - 步骤8
        
        Args:
            question: 用户问题
            context: 检索到的上下文
            max_tokens: 最大生成token数
            temperature: 温度参数
            
        Returns:
            生成的答案
        """
        if not self.client:
            return self._fallback_answer(question, context)
        
        # 构建prompt
        prompt = self._build_prompt(question, context)
        
        try:
            # 调用LLM API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
        
        except Exception as e:
            print(f"LLM API错误: {e}")
            return self._fallback_answer(question, context)
    
    def generate_answer_stream(self, question: str, context: str,
                              max_tokens: int = 1000,
                              temperature: float = 0.3) -> Generator[str, None, None]:
        """
        流式生成答案
        
        Args:
            question: 用户问题
            context: 检索到的上下文
            max_tokens: 最大生成token数
            temperature: 温度参数
            
        Yields:
            生成的文本片段
        """
        if not self.client:
            yield self._fallback_answer(question, context)
            return
        
        # 构建prompt
        prompt = self._build_prompt(question, context)
        
        try:
            # 流式调用
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            print(f"流式API错误: {e}")
            yield self._fallback_answer(question, context)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        构建prompt - 步骤7
        
        组合问题和上下文
        """
        prompt = f"""请基于以下上下文回答问题。如果上下文中没有相关信息，请明确说明。

上下文信息：
{context}

问题：{question}

请提供详细、准确的答案："""
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的保险核保助手，擅长分析保险文档并回答相关问题。

你的职责：
1. 基于提供的上下文信息准确回答问题
2. 如果上下文中没有足够信息，明确说明
3. 使用专业但易懂的语言
4. 提供具体的文档引用（如页码、段落）

回答风格：
- 简洁明了，突出重点
- 使用项目符号或编号列表组织信息
- 必要时提供解释或背景知识
"""
    
    def _fallback_answer(self, question: str, context: str) -> str:
        """降级回答（API不可用时）"""
        return f"""[AI服务暂时不可用]

基于检索到的上下文，以下是相关信息：

{context[:500]}...

请根据以上信息人工分析回答问题："{question}"
"""


class MultiTurnLLMClient(LLMClient):
    """支持多轮对话的LLM客户端"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []
    
    def generate_answer_with_history(self, question: str, context: str,
                                     max_tokens: int = 1000,
                                     temperature: float = 0.3) -> str:
        """
        带对话历史的答案生成
        
        Args:
            question: 用户问题
            context: 检索到的上下文
            max_tokens: 最大生成token数
            temperature: 温度参数
            
        Returns:
            生成的答案
        """
        if not self.client:
            return self._fallback_answer(question, context)
        
        # 构建prompt
        prompt = self._build_prompt(question, context)
        
        # 构建消息列表
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            }
        ]
        
        # 添加对话历史
        messages.extend(self.conversation_history)
        
        # 添加当前问题
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            # 调用LLM API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content.strip()
            
            # 保存到对话历史
            self.conversation_history.append({
                "role": "user",
                "content": prompt
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": answer
            })
            
            # 限制历史长度
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return answer
        
        except Exception as e:
            print(f"LLM API错误: {e}")
            return self._fallback_answer(question, context)
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print("✅ 对话历史已清空")
    
    def get_history_length(self) -> int:
        """获取对话历史长度"""
        return len(self.conversation_history)


# 工厂函数
def create_llm_client(provider: str = 'deepseek', 
                     multi_turn: bool = False,
                     **kwargs) -> LLMClient:
    """
    创建LLM客户端
    
    Args:
        provider: 'deepseek', 'openai', 等
        multi_turn: 是否支持多轮对话
        **kwargs: 其他参数
        
    Returns:
        LLMClient实例
    """
    # 根据provider设置默认参数
    if provider == 'deepseek':
        kwargs.setdefault('api_base', 'https://api.deepseek.com/v1')
        kwargs.setdefault('model', 'deepseek-chat')
    elif provider == 'openai':
        kwargs.setdefault('model', 'gpt-4')
    
    # 创建客户端
    if multi_turn:
        return MultiTurnLLMClient(**kwargs)
    else:
        return LLMClient(**kwargs)


if __name__ == '__main__':
    # 测试代码
    print("测试LLM客户端...")
    
    client = LLMClient()
    
    test_question = "这份保单的承保范围是什么？"
    test_context = """
    本保险合同承保范围包括：
    1. 船体损失
    2. 货物损失
    3. 第三方责任
    """
    
    if client.client:
        answer = client.generate_answer(test_question, test_context)
        print(f"问题: {test_question}")
        print(f"答案: {answer}")
    else:
        print("API未配置，跳过测试")
