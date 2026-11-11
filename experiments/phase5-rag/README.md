# Phase 5 · 文档 RAG 知识库 ⏳

**状态：未开始（依赖 Phase 4）**

## 实验目标

实现本地文档检索与知识问答，对比 ChromaDB 和 Qdrant 两种向量库方案，评估在边缘设备上的实用性。

## 两套方案

### 方案 A · ChromaDB（轻量本地）

- 嵌入式部署，零额外服务
- 适合单机实验和快速验证
- 评估点：索引速度、检索延迟、内存占用

### 方案 B · Qdrant（生产级）

- Docker 部署，支持 gRPC + REST
- 适合多用户、持久化场景
- 评估点：性能对比 ChromaDB、向量过滤、持久化方案

## 通用流程

```
文档（PDF / TXT / MD）
  └─ 解析与分块（LangChain / 自定义）
       └─ Embedding（本地模型）
            └─ 向量入库（ChromaDB / Qdrant）
                 └─ 语义检索 → LLM 生成答案
```

## Embedding 模型候选

| 模型 | 参数量 | 语言 |
|---|---|---|
| bge-small-zh-v1.5 | ~24M | 中文 |
| bge-m3 | ~570M | 多语言 |

## 对比指标

| 指标 | 说明 |
|---|---|
| 索引速度 | 文档入库耗时（文档数 / 秒） |
| 检索延迟 | Top-K 查询响应时间 |
| 内存占用 | 运行时 RSS |
| 检索准确率 | 人工评估 Top-5 相关性 |

## 目录规划

```
experiments/phase5-rag/
├── chroma/          # ChromaDB 方案
│   ├── ingest.py
│   └── query.py
├── qdrant/          # Qdrant 方案
│   ├── compose.yaml
│   ├── ingest.py
│   └── query.py
├── docs/            # 测试文档集
└── results/
    └── comparison.md
```
