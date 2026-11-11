# Phase 5 · 文档 RAG 知识库 小白教程

> RAG（Retrieval-Augmented Generation，检索增强生成）让大模型能够回答关于你自己文档的问题。本教程对比 ChromaDB 和 Qdrant 两种向量数据库方案，在 Radxa Orion O6N 上构建本地知识库。
>
> **实验状态：⏳ 待执行（依赖 Phase 3/4）**

---

## 一、实验目标

用本地模型实现"问本地文档"的完整流程：输入一份 PDF/TXT 文档，能用自然语言提问并得到准确答案，全程不访问互联网。

---

## 二、背景知识

### 2.1 RAG 是什么？

普通大模型的问题：它只知道训练数据里的知识，不知道你私有的文档内容。

RAG 的解决方案：先把文档拆成片段、转成向量存好，提问时先找到最相关的片段，再把片段和问题一起给大模型回答。

```
文档 PDF/TXT/MD
  ↓ 解析分块（每块约 500 字）
  ↓ Embedding 模型（把文字转成向量）
  ↓ 存入向量数据库
  
用户提问
  ↓ Embedding（问题也转成向量）
  ↓ 向量检索（找最相似的文档片段）
  ↓ 拼接 prompt（问题 + 相关片段）
  ↓ 大模型回答
```

### 2.2 什么是向量和 Embedding？

向量（Vector）是一串数字，代表文字的"语义"。语义相近的句子，向量也相近。

Embedding 模型（如 bge-m3）把文字转成向量。这个过程在板子上用 CPU 完成，不需要 NPU。

### 2.3 ChromaDB vs Qdrant

| 对比 | ChromaDB | Qdrant |
|---|---|---|
| 部署方式 | Python 嵌入式 | 独立服务（Docker） |
| 适合场景 | 单机实验、小型知识库 | 多用户、持久化 |
| 资源占用 | 极低 | 中等 |
| 检索能力 | 基础 | 高级（过滤、分片） |
| 推荐用途 | Phase 5 验证 | Phase 6 生产部署 |

---

## 三、方案 A：ChromaDB（轻量本地）

### 3.1 安装

```bash
source /opt/edge-ai-lab/venv/bin/activate
pip install -i https://mirrors.aliyun.com/pypi/simple/ chromadb sentence-transformers langchain
```

### 3.2 文档入库

```python
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 加载 Embedding 模型（首次下载约 100MB）
encoder = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 初始化向量库
client = chromadb.PersistentClient(path='/opt/edge-ai-lab/services/chromadb')
collection = client.get_or_create_collection('docs')

# 分块
def ingest(text: str, source: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = encoder.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{source}_{i}" for i in range(len(chunks))]
    )
    print(f"Ingested {len(chunks)} chunks from {source}")

# 读取文档并入库
with open('/opt/edge-ai-lab/datasets/my_doc.txt', 'r') as f:
    ingest(f.read(), 'my_doc')
```

### 3.3 检索并回答

```python
import httpx

def rag_query(question: str, top_k: int = 3) -> str:
    # 检索相关片段
    q_emb = encoder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    context = '\n'.join(results['documents'][0])

    # 构造 prompt
    prompt = f"""根据以下文档内容回答问题。

文档内容：
{context}

问题：{question}

回答："""

    # 调用本地 LLM（通过 Phase 4 的网关）
    resp = httpx.post('http://localhost:8000/v1/chat/completions',
        json={"model": "qwen2.5:1.5b",
              "messages": [{"role": "user", "content": prompt}]})
    return resp.json()['choices'][0]['message']['content']

print(rag_query("这份文档的主要内容是什么？"))
```

---

## 四、方案 B：Qdrant（生产级）

### 4.1 Docker 部署 Qdrant

```bash
docker run -d \
    --name qdrant \
    -p 6333:6333 \
    -v /opt/edge-ai-lab/services/qdrant:/qdrant/storage \
    qdrant/qdrant:latest
```

### 4.2 安装客户端

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ qdrant-client
```

### 4.3 文档入库

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(host='localhost', port=6333)

# 创建集合
client.recreate_collection(
    collection_name='docs',
    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
)

# 入库（与 ChromaDB 类似，向量维度要匹配 Embedding 模型输出）
chunks = [...]  # 分块后的文本列表
embeddings = encoder.encode(chunks)
client.upsert(
    collection_name='docs',
    points=[PointStruct(id=i, vector=emb.tolist(), payload={"text": chunk})
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
)
```

---

## 五、性能对比指标

实验完成后，记录以下数据：

| 指标 | ChromaDB | Qdrant |
|---|---|---|
| 入库速度（文档/秒） | — | — |
| 检索延迟（Top-5，ms） | — | — |
| 内存占用（MB） | — | — |
| 10000 片段检索延迟 | — | — |

---

## 六、验收标准

- [ ] ChromaDB 方案：文档入库 + 检索问答通过
- [ ] Qdrant 方案：Docker 启动 + 文档入库 + 检索问答通过
- [ ] 两套方案性能数据记录完成
- [ ] 完整 RAG 流程（文档 → Embedding → 检索 → LLM 回答）端到端测试通过

---

*本教程对应实验：Phase 5，待执行。最后更新：2026-11-11。*
