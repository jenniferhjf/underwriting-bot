# 🚀 Underwriting Assistant - Quick Start Guide

## ⚡ 3步快速启动

### Step 1: 安装依赖
```bash
pip install -r requirements_assistant.txt
```

### Step 2: 运行应用
```bash
streamlit run underwriting_assistant.py
```

### Step 3: 使用系统
浏览器自动打开 `http://localhost:8501`

---

## 📋 核心功能速览

### 1. Workspace Management (工作区管理)
```
侧边栏 → ➕ New Workspace → 输入名称 → Create
```

**工作区概念：**
- 每个工作区 = 独立的文档容器
- 工作区之间完全隔离
- 例如：创建"Gas Turbine Cases"、"Oil & Gas Projects"等

### 2. Document Upload (文档上传)
```
Documents标签 → ➕ Upload Document
```

**支持格式：**
- 📄 PDF
- 📝 Word (.docx, .doc)
- 📃 Text (.txt)
- 📊 Excel (.xlsx, .xls)
- 🖼️ Images (.png, .jpg, .jpeg)

**上传步骤：**
1. 选择文件
2. 添加多维标签：
   - 🔧 Equipment: Gas Turbine, Boiler, etc.
   - 🏭 Industry: Oil & Gas, Manufacturing, etc.
   - 📅 Timeline: 2024-Q4, 2024-Q3, etc.
3. 填写案例信息：
   - Decision, Premium, Risk Level
   - Case Summary, Key Insights
4. 点击上传

### 3. AI Chat (AI对话)
```
Chat标签 → 输入问题 → Enter
```

**示例问题：**
```
- "Show me gas turbine cases from 2024"
- "10-year equipment in oil & gas, how to price?"
- "Compare approved vs declined patterns"
- "High risk cases in manufacturing"
```

**AI响应包含：**
- 检索到的相关文档（Top 3）
- 5步Chain-of-Thought分析
- 决策推荐 + 保费范围
- 引用的Case ID

### 4. Analytics (分析)
```
Analytics标签 → 查看统计
```

**数据可视化：**
- 文档总数、批准率、拒绝率
- 决策分布图表
- 文档格式分布
- 存储使用情况

---

## 🎯 基于PPT的设计

### Solution Design架构

**1. Multimodal Extraction (多模态提取)**
```
Excel/Word/Notes → Document Parser → Extract text/tables/metadata
                                  ↓
                         OCR + Manual Validation
                                  ↓
                         Tagged Corpus (有标签的语料库)
```

**实现：**
- ✅ 支持PDF, Word, Excel, Images
- ✅ 文本提取 (PyPDF2, python-docx)
- ✅ 三维标签系统 (Equipment, Industry, Timeline)

**2. RAG + CoT Framework**
```
WHY RAG + CoT?
- Fast Retrieval: <0.1s 语义搜索
- Explainable Reasoning: 5步CoT确保透明推荐
```

**System Instruction (CoT):**
```
Role: You are Mr. X's AI underwriting assistant

Task: Answer underwriting queries using retrieved cases

Process: Think step-by-step:
  Step 1: Extract key tags from query
  Step 2: Analyze retrieved precedents
  Step 3: Check recency & applicability
  Step 4: Identify decision patterns
  Step 5: Recommend with rationale

Output: Provide decision + premium + sources
```

**3. Chat-bot Interface**
```
WHAT is Chatbot?
- Conversational AI Assistant
- 支持多轮对话和上下文理解

WHY Chat-bot vs QA-Bot?
✅ Multi-turn dialogue
✅ Context understanding
✅ Human-like reasoning

HOW to Use?
Underwriter → Query → LLM → Finding + Think → Answer + Sources
```

---

## 📁 项目文件结构

```
underwriting-assistant/
├── underwriting_assistant.py       # 主应用程序
├── requirements_assistant.txt      # 依赖包列表
├── README_UNDERWRITING_ASSISTANT.md # 完整文档
├── QUICKSTART_ASSISTANT.md         # 本文件
└── data/                           # 数据目录（自动创建）
    ├── workspaces/
    │   ├── Gas Turbine Cases/
    │   │   ├── documents/
    │   │   ├── metadata.json
    │   │   └── embeddings.json
    │   └── Oil & Gas Projects/
    │       ├── documents/
    │       ├── metadata.json
    │       └── embeddings.json
    └── embeddings/
```

---

## 🎨 UI界面说明

### 简洁专业的设计

**左侧边栏：**
- 📁 Workspaces (工作区列表)
- ➕ New Workspace (创建新工作区)
- 📊 Workspace Stats (统计信息)
- ⚙️ Settings (设置)

**主界面三个标签：**

**1. 💬 Chat**
- 对话式聊天界面
- 显示检索到的文档
- 实时AI响应

**2. 📄 Documents**
- 📋 View Documents (查看文档)
  - 多维度筛选
  - 标签展示
  - 删除操作
- ➕ Upload Document (上传文档)
  - 文件选择
  - 标签添加
  - 元数据填写

**3. 📊 Analytics**
- 统计指标卡片
- 决策分布图表
- 格式分布图表

---

## 🔧 配置说明

### API配置

文件：`underwriting_assistant.py` 第16行

```python
DEEPSEEK_API_KEY = "sk-99bba2ce117444e197270f17d303e74f"
```

**测试API：**
```bash
# 在Python中测试
import requests

headers = {
    "Authorization": "Bearer sk-99bba2ce117444e197270f17d303e74f"
}

response = requests.post(
    "https://api.deepseek.com/v1/chat/completions",
    headers=headers,
    json={
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Hello"}]
    }
)

print(response.json())
```

### 修改标签选项

文件：`underwriting_assistant.py` 第30-37行

```python
TAG_OPTIONS = {
    "equipment": ["你的设备列表"],
    "industry": ["你的行业列表"],
    "timeline": ["你的时间段列表"]
}
```

---

## 📊 使用示例

### 完整工作流程

**场景：Mr. X收到新RFQ**

**Step 1: 创建工作区**
```
侧边栏 → ➕ New Workspace
名称: "2024 Q4 New Cases"
→ Create
```

**Step 2: 上传历史案例**
```
Documents → ➕ Upload

文件: MrX_Gas_Turbine_History.pdf

标签:
🔧 Equipment: [Gas Turbine]
🏭 Industry: [Oil & Gas]
📅 Timeline: [2023, 2024-Q1, 2024-Q2]

案例信息:
Decision: Approved
Premium: $48,000
Risk: Medium
Summary: "Historical gas turbine cases showing approval patterns..."
Insights: "Key factors: maintenance quality, age <12 years, upstream sector..."

→ Upload
```

**Step 3: 查询相似案例**
```
Chat → 输入问题:
"10-year gas turbine in oil & gas upstream, average maintenance, how to price?"

→ AI响应:
```

**AI响应示例：**
```markdown
### STEP 1: Extract Key Tags
From your query, I identified:
- Equipment: Gas Turbine (10 years)
- Industry: Oil & Gas - Upstream
- Maintenance: Average quality
- Query type: Pricing inquiry

### STEP 2: Analyze Retrieved Precedents
Retrieved 3 relevant cases from your workspace:

**DOC-20230315-ABC123** (PDF)
🔧 Gas Turbine | 🏭 Oil & Gas | 📅 2023-Q1
Decision: Approved | Premium: $48,000 | Risk: Medium
Summary: Similar 10-year turbine, excellent maintenance

**DOC-20240520-DEF456** (DOCX)
🔧 Gas Turbine | 🏭 Oil & Gas | 📅 2024-Q2
Decision: Conditional | Premium: $52,000 | Risk: Medium-High
Summary: 11-year turbine, average maintenance

**DOC-20211110-GHI789** (Excel)
🔧 Gas Turbine | 🏭 Oil & Gas | 📅 2021-Q4
Decision: Declined | Premium: N/A | Risk: High
Summary: 15-year turbine, poor maintenance

### STEP 3: Check Recency & Applicability
- Cases ABC123 & DEF456 are from 2023-2024 → Highly relevant
- Case GHI789 from 2021 shows declining boundary
- All match equipment type and industry sector

### STEP 4: Identify Decision Patterns
Pattern Analysis:
1. **Age Threshold**: 10y approved, 15y declined → 10y is borderline
2. **Maintenance Quality** is KEY differentiator:
   - Excellent maintenance → Approved at $48k
   - Average maintenance → Conditional at $52k
   - Poor maintenance → Declined
3. **Upstream sector** tends to have slightly higher premiums

### STEP 5: Recommendation
**DECISION:** CONDITIONAL APPROVAL

**PREMIUM RANGE:** $50,000 - $54,000

**RATIONALE:**
Following the pattern in Case #DEF456, I recommend conditional 
approval for this 10-year gas turbine with average maintenance.

The equipment age (10 years) is at the acceptable threshold, but 
"average" maintenance (vs "excellent" in Case #ABC123) increases 
risk and warrants conditions.

**CONDITIONS:**
1. Request complete maintenance records (last 3 years)
2. Require third-party inspection report
3. Verify no major incidents or downtime

**PREMIUM JUSTIFICATION:**
- If maintenance documentation is strong → $50k (closer to #ABC123)
- If documentation has gaps → $54k (closer to #DEF456)
- If critical issues found → Decline (like #GHI789)

**SOURCES:**
- DOC-20230315-ABC123
- DOC-20240520-DEF456
- DOC-20211110-GHI789
```

---

## ⚠️ 重要提示

### 1. Embedding是原型版本

当前使用简单的hash-based假embedding。

**生产环境应替换为：**

```python
# 选项1: OpenAI (推荐)
from openai import OpenAI

client = OpenAI(api_key="your-key")

def generate_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

# 选项2: Sentence Transformers (本地)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def generate_embedding(text: str):
    return model.encode(text).tolist()
```

### 2. 数据存储

当前使用JSON文件存储。

**生产环境推荐：**
- ChromaDB (向量数据库)
- PostgreSQL + pgvector
- Pinecone / Weaviate

### 3. 文本提取

当前支持基本文本提取。

**增强功能：**
- OCR for scanned documents
- Table extraction from PDFs
- Excel data parsing
- Image preprocessing

---

## 🐛 故障排除

### 问题1: "ModuleNotFoundError"
```bash
pip install -r requirements_assistant.txt
```

### 问题2: PDF提取失败
- 确保PDF不是加密的
- 尝试其他PDF库：pdfplumber, pymupdf

### 问题3: API错误
- 检查API key是否正确
- 检查网络连接
- 查看DeepSeek API余额

### 问题4: 搜索不准确
- 使用真实embedding模型
- 增加文档数量
- 优化case summary质量

### 问题5: 工作区找不到
- 检查 `data/workspaces/` 目录
- 确保有写入权限

---

## 📚 延伸阅读

完整文档请查看：`README_UNDERWRITING_ASSISTANT.md`

内容包括：
- 详细技术架构
- 5步CoT框架详解
- 生产部署建议
- 性能优化指南
- API集成示例

---

## 🎉 Success Checklist

- [ ] 安装依赖包
- [ ] 启动应用
- [ ] 创建第一个工作区
- [ ] 上传第一个文档
- [ ] 在Chat中提问
- [ ] 收到AI的5步分析
- [ ] 查看Analytics统计
- [ ] 尝试多维度筛选

---

**Underwriting Assistant 让 Mr. X 的专业知识运营化！** 🚀

需要帮助？查看完整文档或联系技术支持。
