"""Phase 5 RAG 知识库 - ChromaDB 方案"""
import chromadb
from chromadb.utils import embedding_functions
import httpx
import os
import json
import time

# 使用 bge-small-zh embedding (本地运行)
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
DB_PATH = "/opt/edge-ai-lab/services/chromadb"
COLLECTION_NAME = "edge_ai_docs"

# Phase 4 网关 URL
GATEWAY_URL = "http://localhost:8000"

# 示例文档集（关于 CIX P1 Edge AI Lab）
SAMPLE_DOCS = [
    "CIX P1 Edge AI Lab 是基于 Radxa Orion O6N 开发板的边缘 AI 实验平台。",
    "该平台搭载 Zhouyi V3 NPU，提供 30 TOPS 算力，支持 INT8 量化推理。",
    "CPU 采用 12 核 ARM big.LITTLE 架构：8 个 Cortex-A720 大核和 4 个 Cortex-A520 小核。",
    "内存为 32 GB LPDDR5，存储为 1 TB NVMe SSD。",
    "NPU 推理需要使用 onnxruntime-zhouyi，不能使用 PyPI 通用版 onnxruntime。",
    "Execution Provider 名称是 ZhouyiExecutionProvider，不是 CIXExecutionProvider。",
    "MobileNetV2 INT8 在 NPU 上比 CPU 快 8.6 倍，延迟仅 1.48 毫秒。",
    "YOLOv8n 目标检测在 NPU 上达到 90.9 FPS，比 CPU 快 17.8 倍。",
    "FlagOS 优化路线下 MiniCPM-V 4.7 的 PP512 达到 589 tok/s，TG128 达到 21.9 tok/s。",
    "llama.cpp 通用路线下 Qwen2.5-1.5B 的 PP512 为 117 tok/s，TG128 为 22.78 tok/s。",
]


def build_db():
    """构建向量数据库"""
    print("=== 初始化 ChromaDB ===")
    client = chromadb.PersistentClient(path=DB_PATH)

    # 使用 sentence-transformers 作为 embedding
    try:
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
    except Exception as e:
        print(f"sentence-transformers 不可用: {e}")
        print("使用默认 embedding 函数")
        emb_fn = embedding_functions.DefaultEmbeddingFunction()

    # 创建/获取集合
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # 入库
    print(f"=== 入库 {len(SAMPLE_DOCS)} 篇文档 ===")
    t0 = time.time()
    collection.add(
        documents=SAMPLE_DOCS,
        ids=[f"doc_{i}" for i in range(len(SAMPLE_DOCS))],
        metadatas=[{"source": "edge-ai-lab"} for _ in SAMPLE_DOCS]
    )
    elapsed = time.time() - t0
    print(f"入库耗时: {elapsed:.2f}s, 集合大小: {collection.count()}")

    return collection, emb_fn


def rag_query(collection, question: str, top_k: int = 3) -> dict:
    """RAG 检索 + LLM 回答"""
    print(f"\n=== 问题: {question} ===")

    # 1. 检索相关片段
    t0 = time.time()
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    retrieval_ms = (time.time() - t0) * 1000

    contexts = results["documents"][0]
    distances = results["distances"][0]
    print(f"检索耗时: {retrieval_ms:.1f}ms, Top-{top_k} 结果:")
    for i, (ctx, dist) in enumerate(zip(contexts, distances)):
        print(f"  [{i+1}] dist={dist:.4f}: {ctx[:60]}...")

    # 2. 构造 prompt
    context_str = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    prompt = f"""根据以下文档内容回答问题。如果文档中没有相关信息，请说明。

文档内容：
{context_str}

问题：{question}

回答："""

    # 3. 调用 LLM（通过 Phase 4 网关）
    t0 = time.time()
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{GATEWAY_URL}/v1/chat/completions",
                json={
                    "model": "qwen2.5",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 128,
                    "temperature": 0.3,
                }
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
            llm_ms = (time.time() - t0) * 1000
    except Exception as e:
        answer = f"LLM 调用失败: {e}"
        llm_ms = (time.time() - t0) * 1000

    print(f"LLM 耗时: {llm_ms:.1f}ms")
    print(f"回答: {answer}")

    return {
        "question": question,
        "retrieval_ms": round(retrieval_ms, 1),
        "llm_ms": round(llm_ms, 1),
        "contexts": contexts,
        "answer": answer,
    }


if __name__ == "__main__":
    collection, emb_fn = build_db()

    # 测试问题
    questions = [
        "这台设备的 NPU 算力是多少？",
        "MobileNetV2 在 NPU 上比 CPU 快多少倍？",
        "Execution Provider 的名称是什么？",
        "FlagOS 路线的 PP512 性能是多少？",
    ]

    results = []
    for q in questions:
        r = rag_query(collection, q, top_k=3)
        results.append(r)
        print()

    # 保存结果
    os.makedirs("/opt/edge-ai-lab/benchmarks", exist_ok=True)
    with open("/opt/edge-ai-lab/benchmarks/phase5_rag_chroma_2026-11-11.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("=== 保存结果到 phase5_rag_chroma_2026-11-11.json ===")
