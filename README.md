# Underwriting Assistant - 完整文档

## 📋 项目概述

专业的承保AI助手，基于你的PPT设计：
- **Multimodal Extraction**: 支持PDF、Word、Excel、图片等多格式文档
- **RAG + CoT Framework**: 检索增强生成 + 5步思维链推理
- **Workspace Management**: 工作区隔离，文档容器化管理
- **Vector Database**: 使用Embeddings进行语义搜索
- **Clean UI**: 简洁专业的界面设计

## 🎯 核心功能

### 1. Multimodal Extraction（多模态提取）

**支持格式：**
- 📄 PDF - 提取文本内容
- 📝 Word (.docx, .doc) - 提取段落文本
- 📃 Text (.txt) - 直接读取
- 📊 Excel (.xlsx, .xls) - 表格数据
- 🖼️ Images (.png, .jpg, .jpeg) - 图片文件

**提取流程：**
```
Document Upload → Text Extraction → Tagging → Embedding → Vector DB
```

### 2. RAG + CoT Framework

**为什么使用RAG + CoT？**

**Fast Retrieval（快速检索）：**
- 语义搜索在<0.1s内从Mr. X的知识库检索相关案例

**Explainable Reasoning（可解释推理）：**
- 5步CoT框架确保透明、可审计的推荐

**System Instruction (CoT):**
```
Role: You are Mr. X's AI underwriting assistant

Task: Answer underwriting queries using retrieved cases

Process: Think step-by-step using this framework:
  Step 1: Extract key tags from query
  Step 2: Analyze retrieved precedents
  Step 3: Check recency & applicability
  Step 4: Identify decision patterns
  Step 5: Recommend with rationale

Output: Provide decision + premium + sources
```

### 3. Chat-bot（对话机器人）

**WHAT is Chatbot?**
- Conversational AI Assistant
- 通过对话交互支持知识查询和承保决策

**WHY Chat-bot?**

Chat-bot vs QA-Bot + RAG:
- ✅ Multi-turn dialogue (多轮对话)
- ✅ Context understanding (上下文理解)
- ✅ More coherent responses (更连贯的响应)
- ✅ Learn from interactions (从交互中学习)
- ✅ Human-like reasoning dialogue (类人推理对话)

**HOW to Use?**
```
Underwriter 🧑 → Query Q's to Chatbox
                ↓
LLM → Finding + Think.......
  → Answer + Sources
```

### 4. Workspace Management（工作区管理）

**工作区概念：**
- 每个工作区是独立的文档容器
- 工作区之间完全隔离
- 可以按项目、客户、或时间段创建工作区

**工作区结构：**
```
data/
└── workspaces/
    ├── Gas Turbine Cases/
    │   ├── documents/
    │   │   ├── DOC-20241107-ABC123.pdf
    │   │   ├── DOC-20241107-DEF456.docx
    │   │   └── ...
    │   ├── metadata.json
    │   └── embeddings.json
    │
    ├── Oil & Gas Projects/
    │   ├── documents/
    │   ├── metadata.json
    │   └── embeddings.json
    │
    └── 2024 Q4 Cases/
        ├── documents/
        ├── metadata.json
        └── embeddings.json
```

### 5. Vector Database（向量数据库）

**Embedding Process:**
```
1. Document Text → Chunking (分块)
   - Split into 500-1000 token chunks
   - 100-token overlap for context

2. Chunks → Embedding Model
   - OpenAI text-embedding-3 (推荐)
   - 或 sentence-transformers (本地)
   - 生成 1536-dim vectors

3. Vectors → Vector DB
   - 存储在embeddings.json
   - Index: HNSW for fast retrieval
```

**Semantic Search:**
```
User Query → Query Embedding → Similarity Search → Top-K Documents
```

## 🚀 快速开始

### 安装

```bash
# 安装依赖
pip install -r requirements_assistant.txt
```

### 运行

```bash
# 启动应用
streamlit run underwriting_assistant.py
```

应用会在 `http://localhost:8501` 打开

## 📖 使用指南

### Step 1: 创建工作区

1. 在侧边栏点击 "➕ New Workspace"
2. 输入工作区名称，例如："Gas Turbine Cases 2024"
3. 点击 "Create"

### Step 2: 上传文档

1. 进入 "📄 Documents" → "➕ Upload Document"
2. 选择文件（PDF、Word、Excel等）
3. 添加多维标签：
   - 🔧 Equipment: Gas Turbine, Boiler, etc.
   - 🏭 Industry: Oil & Gas, Manufacturing, etc.
   - 📅 Timeline: 2024-Q4, 2024-Q3, etc.
4. 填写案例信息：
   - Decision: Approved/Declined/Conditional
   - Premium: 保费金额
   - Risk Level: Low/Medium/High
   - Case Summary: 案例摘要
   - Key Insights: 关键见解
5. 点击 "📤 Upload Document"

### Step 3: 与AI对话

1. 进入 "💬 Chat" 标签
2. 输入问题，例如：
   ```
   - "Show me gas turbine cases approved in 2024"
   - "10-year equipment in oil & gas, how to price?"
   - "Compare high risk vs low risk patterns"
   ```
3. AI会：
   - 搜索向量数据库
   - 检索最相关的3个案例
   - 使用5步CoT框架分析
   - 提供推荐和理由

### Step 4: 查看分析

1. 进入 "📊 Analytics" 标签
2. 查看：
   - 文档统计
   - 决策分布
   - 格式分布

## 🏗️ 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│                    (Streamlit UI)                        │
├─────────────────────────────────────────────────────────┤
│  💬 Chat    │  📄 Documents    │  📊 Analytics          │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              WORKSPACE MANAGEMENT LAYER                  │
│                                                          │
│  Workspace 1  │  Workspace 2  │  Workspace 3           │
│  (Isolated)   │  (Isolated)   │  (Isolated)            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              DOCUMENT PROCESSING LAYER                   │
│                                                          │
│  PDF Extract → Text  │  DOCX Extract → Text            │
│  TXT Read → Text     │  Image OCR → Text (Future)      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                EMBEDDING LAYER                           │
│                                                          │
│  Text → Chunking → Embedding Model → Vectors           │
│  (OpenAI text-embedding-3 or sentence-transformers)    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                VECTOR DATABASE                           │
│                                                          │
│  embeddings.json (Local Storage)                        │
│  or ChromaDB / Pinecone (Production)                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                 RAG RETRIEVAL                            │
│                                                          │
│  Query Embedding → Similarity Search → Top-K Docs       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              LLM (CoT REASONING)                         │
│                                                          │
│  DeepSeek API → 5-Step CoT → Response                   │
└─────────────────────────────────────────────────────────┘
```

### Data Flow（数据流）

**上传文档流程：**
```
1. User uploads document.pdf
   ↓
2. Extract text from PDF
   ↓
3. Combine: case_summary + key_insights + extracted_text
   ↓
4. Generate embedding vector (1536-dim)
   ↓
5. Save to workspace:
   - documents/DOC-xxx.pdf (原文件)
   - metadata.json (元数据)
   - embeddings.json (向量)
```

**查询响应流程：**
```
1. User query: "10-year gas turbine cases"
   ↓
2. Generate query embedding
   ↓
3. Similarity search in vector DB
   ↓
4. Retrieve top-3 most similar documents
   ↓
5. Format documents as context
   ↓
6. Send to DeepSeek API with CoT prompt
   ↓
7. LLM generates 5-step reasoning
   ↓
8. Display response + sources to user
```

## 🔧 技术细节

### Embedding实现

**当前版本（原型）：**
```python
def generate_embedding(text: str) -> List[float]:
    # Placeholder: Simple hash-based fake embedding
    text_hash = hashlib.md5(text.encode()).hexdigest()
    fake_embedding = [float(int(text_hash[i:i+2], 16)) / 255.0 
                      for i in range(0, 32, 2)]
    fake_embedding = fake_embedding + [0.0] * (1536 - len(fake_embedding))
    return fake_embedding[:1536]
```

**生产版本（推荐）：**

**选项1：OpenAI Embeddings**
```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

def generate_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding
```

**选项2：Sentence Transformers（本地）**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def generate_embedding(text: str) -> List[float]:
    return model.encode(text).tolist()
```

### Vector Search优化

**当前：简单余弦相似度**
```python
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2)
```

**优化：使用ChromaDB**
```python
import chromadb

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection("underwriting_docs")

# Add documents
collection.add(
    documents=[doc_text],
    embeddings=[embedding],
    metadatas=[metadata],
    ids=[doc_id]
)

# Query
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
```

### 文本提取增强

**当前支持：**
- ✅ PDF文本提取（PyPDF2）
- ✅ Word文档提取（python-docx）
- ✅ 纯文本读取

**未来增强：**
```python
# OCR for images/scanned PDFs
import pytesseract
from PIL import Image

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Excel parsing
import pandas as pd

def extract_text_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    text = df.to_string()
    return text
```

## 📊 5步CoT框架详解

### Step 1: Extract Key Tags from Query

**目标：** 识别查询中的关键标签

**示例：**
```
Query: "10-year gas turbine in oil & gas, how to price?"

Extracted Tags:
- Equipment: Gas Turbine
- Age: 10 years
- Industry: Oil & Gas
- Question Type: Pricing inquiry
```

### Step 2: Analyze Retrieved Precedents

**目标：** 审查检索到的案例

**示例：**
```
Retrieved 3 cases:
1. DOC-20230315-ABC123
   Tags: Gas Turbine, Oil & Gas, 2023-Q1
   Decision: Approved, Premium: $48k, Risk: Medium

2. DOC-20230820-DEF456
   Tags: Gas Turbine, Oil & Gas, 2023-Q3
   Decision: Conditional, Premium: $52k, Risk: Medium-High

3. DOC-20211110-GHI789
   Tags: Gas Turbine, Oil & Gas, 2021-Q4
   Decision: Declined, Risk: High (15 years old)
```

### Step 3: Check Recency & Applicability

**目标：** 评估案例的时效性和适用性

**示例：**
```
Recency Analysis:
- Case 1 & 2: From 2023 → Highly relevant
- Case 3: From 2021 → Less recent but shows boundary

Applicability:
- All 3 cases match equipment and industry
- Age range: 10-15 years (our query: 10 years)
- Case 1 closest match to query
```

### Step 4: Identify Decision Patterns

**目标：** 找出决策规律

**示例：**
```
Pattern Analysis:
1. Age threshold: 10y = Approved, 15y = Declined
   → 10y is borderline, acceptable with good maintenance

2. Maintenance quality is KEY differentiator:
   - Case 1 (Approved): "Excellent maintenance"
   - Case 2 (Conditional): "Average maintenance"
   - Case 3 (Declined): "Poor maintenance"

3. Industry sub-sector affects premium:
   - Upstream: Slightly higher premium
   - Downstream: Standard premium
```

### Step 5: Recommend with Rationale

**目标：** 提供明确推荐和理由

**示例：**
```
RECOMMENDATION:
Decision: CONDITIONAL APPROVAL
Premium Range: $50,000 - $54,000

Rationale:
Based on Case #DEF456's pattern, I recommend conditional 
approval for this 10-year gas turbine. The equipment age is 
at our threshold, and "average" maintenance (vs "excellent" 
in Case #ABC123) elevates risk.

Conditions:
1. Request full maintenance records (last 3 years)
2. Require third-party inspection report
3. Verify no major incident history

Premium Justification:
- If maintenance documentation is strong: $50k (closer to Case #ABC123)
- If documentation is incomplete: $54k (closer to Case #DEF456 conditional)
- If critical gaps found: Decline (like Case #GHI789)

Sources: DOC-20230315-ABC123, DOC-20230820-DEF456
```

## 🎨 UI设计说明

### 简洁专业的界面

**设计原则：**
1. **Clean** - 去除不必要的元素
2. **Professional** - 商务化配色和排版
3. **Functional** - 功能优先，一目了然

**配色方案：**
```css
Background: #f5f7fa (浅灰蓝)
Cards: #ffffff (白色)
Primary: #1f2937 (深灰)
Secondary: #6b7280 (中灰)

Tags:
- Equipment: #dbeafe (浅蓝)
- Industry: #dcfce7 (浅绿)
- Timeline: #fef3c7 (浅黄)
```

### 三个主要标签页

**1. 💬 Chat**
- 对话式界面
- 显示检索到的文档
- 实时AI响应

**2. 📄 Documents**
- 查看所有文档
- 多维度筛选
- 上传新文档

**3. 📊 Analytics**
- 工作区统计
- 决策分布图表
- 格式分布

## 🔒 数据隔离

### Workspace隔离机制

每个工作区完全独立：

```
Workspace A (Gas Turbine Cases):
- 只包含燃气轮机案例
- 只能搜索本工作区文档
- 向量数据库独立

Workspace B (Manufacturing Projects):
- 只包含制造业案例
- 完全与Workspace A隔离
- 独立的embeddings

→ 切换工作区 = 切换知识库
```

**优势：**
- 🔒 安全性：不同项目数据隔离
- 🎯 精确性：搜索更聚焦
- 📊 清晰性：统计更准确
- 🗂️ 组织性：文档管理有序

## ⚙️ 配置和自定义

### 修改API Key

编辑 `underwriting_assistant.py`:
```python
DEEPSEEK_API_KEY = "your-api-key"
```

### 修改Embedding模型

替换 `generate_embedding()` 函数：
```python
# 使用OpenAI
from openai import OpenAI
client = OpenAI(api_key="your-key")

def generate_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding
```

### 修改标签选项

编辑 `TAG_OPTIONS` 字典：
```python
TAG_OPTIONS = {
    "equipment": ["Your", "Custom", "Equipment", "List"],
    "industry": ["Your", "Industry", "List"],
    "timeline": ["2024", "2023", "etc"]
}
```

### 修改CoT Prompt

编辑 `SYSTEM_INSTRUCTION` 变量：
```python
SYSTEM_INSTRUCTION = """Your custom system instruction here"""
```

## 🚀 生产部署建议

### 1. 使用真实Embedding模型

```bash
pip install openai sentence-transformers
```

### 2. 使用向量数据库

推荐：ChromaDB, Pinecone, Weaviate

```bash
pip install chromadb
```

### 3. 添加用户认证

```python
import streamlit_authenticator as stauth
```

### 4. 使用云存储

- AWS S3
- Google Cloud Storage
- Azure Blob Storage

### 5. 添加日志和监控

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"User query: {query}")
logger.info(f"Retrieved docs: {len(docs)}")
```

### 6. 性能优化

- 缓存embeddings
- 异步处理
- 批量embedding生成

## 📝 常见问题

### Q1: Embedding是假的吗？

A: 当前版本使用简单的hash-based假embedding用于原型演示。生产环境应使用真实的embedding模型（OpenAI或sentence-transformers）。

### Q2: 如何提高搜索准确度？

A: 
1. 使用真实embedding模型
2. 增加文档数量
3. 优化case summary和key insights的质量
4. 调整top_k参数

### Q3: 工作区之间能共享文档吗？

A: 默认不能。工作区是隔离的。如果需要共享，可以导出文档并重新上传到另一个工作区。

### Q4: 支持网页抓取吗？

A: 当前版本不支持。未来可添加：
```python
import requests
from bs4 import BeautifulSoup

def fetch_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    return text
```

### Q5: 如何批量上传文档？

A: 当前需要逐个上传。未来可添加批量上传功能或CSV导入。

## 📞 技术支持

- DeepSeek API: https://platform.deepseek.com
- Streamlit Docs: https://docs.streamlit.io
- PyPDF2: https://pypdf2.readthedocs.io
- python-docx: https://python-docx.readthedocs.io

## 📄 许可证

原型演示项目，供教育和研究使用。

---

**Underwriting Assistant - Making Mr. X's Expertise Operational! 🚀**
