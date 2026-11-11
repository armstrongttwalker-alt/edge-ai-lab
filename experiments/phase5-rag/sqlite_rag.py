"""
Phase 5 RAG 知识库 - 轻量版
用 sqlite3 + numpy 实现向量存储和余弦检索
不依赖 chromadb / sentence-transformers / torch
Embedding: onnxruntime + 板载已安装的 paraphrase-multilingual-minilm 模型
回退方案: TF-IDF 字符级向量（零依赖）
"""
import sqlite3
import numpy as np
import json
import httpx
import os
import time
import struct
import hashlib

DB_PATH = "/opt/edge-ai-lab/services/rag/vectors.db"
GATEWAY_URL = "http://localhost:8000"

# 示例文档集
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


class TFIDFEmbedder:
    """字符级 TF-IDF 向量，零依赖，支持中文"""
    def __init__(self, dim=256):
        self.dim = dim
        self.vocab = {}

    def _tokenize(self, text):
        # 字符级 unigram + bigram
        chars = list(text)
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        return chars + bigrams

    def fit(self, docs):
        # 建立词表
        all_tokens = set()
        for doc in docs:
            all_tokens.update(self._tokenize(doc))
        self.vocab = {t: i % self.dim for i, t in enumerate(sorted(all_tokens))}
        return self

    def embed(self, texts):
        vecs = []
        for text in texts:
            v = np.zeros(self.dim)
            tokens = self._tokenize(text)
            for t in tokens:
                if t in self.vocab:
                    v[self.vocab[t]] += 1
            # L2 normalize
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vecs.append(v)
        return np.array(vecs)


class VectorDB:
    """SQLite 向量数据库"""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY,
                text TEXT,
                vector BLOB
            )
        """)
        self.conn.commit()

    def add(self, texts, vectors):
        for text, vec in zip(texts, vectors):
            blob = vec.astype(np.float32).tobytes()
            self.conn.execute("INSERT INTO docs (text, vector) VALUES (?, ?)", (text, blob))
        self.conn.commit()

    def search(self, query_vec, top_k=3):
        rows = self.conn.execute("SELECT text, vector FROM docs").fetchall()
        if not rows:
            return []
        texts = [r[0] for r in rows]
        vecs = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        # Cosine similarity
        q = query_vec.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs_norm = vecs / (norms + 1e-9)
        scores = vecs_norm @ q
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [(texts[i], float(scores[i])) for i in top_idx]

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]


def build_db(embedder, db):
    print(f"=== 入库 {len(SAMPLE_DOCS)} 篇文档 ===")
    t0 = time.time()
    vecs = embedder.embed(SAMPLE_DOCS)
    db.add(SAMPLE_DOCS, vecs)
    elapsed = time.time() - t0
    print(f"入库耗时: {elapsed:.3f}s, 集合大小: {db.count()}")


def rag_query(embedder, db, question, top_k=3):
    print(f"\n=== 问题: {question} ===")
    t0 = time.time()
    q_vec = embedder.embed([question])[0]
    results = db.search(q_vec, top_k)
    retrieval_ms = (time.time() - t0) * 1000
    print(f"检索耗时: {retrieval_ms:.1f}ms")
    for i, (ctx, score) in enumerate(results):
        print(f"  [{i+1}] score={score:.4f}: {ctx[:60]}...")

    contexts = [r[0] for r in results]
    context_str = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    prompt = f"""根据以下文档内容回答问题。如果文档中没有相关信息，请说明。

文档内容：
{context_str}

问题：{question}

回答："""

    t0 = time.time()
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(f"{GATEWAY_URL}/v1/chat/completions", json={
                "model": "qwen2.5",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 128,
                "temperature": 0.3,
            })
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
        "top_contexts": contexts,
        "answer": answer,
    }


if __name__ == "__main__":
    embedder = TFIDFEmbedder(dim=256).fit(SAMPLE_DOCS)
    db = VectorDB(DB_PATH)

    if db.count() == 0:
        build_db(embedder, db)
    else:
        print(f"=== 复用已有数据库，共 {db.count()} 篇文档 ===")

    questions = [
        "这台设备的 NPU 算力是多少？",
        "MobileNetV2 在 NPU 上比 CPU 快多少倍？",
        "Execution Provider 的名称是什么？",
        "FlagOS 路线的 PP512 性能是多少？",
    ]

    results = []
    for q in questions:
        r = rag_query(embedder, db, q)
        results.append(r)

    os.makedirs("/opt/edge-ai-lab/benchmarks", exist_ok=True)
    with open("/opt/edge-ai-lab/benchmarks/phase5_rag_sqlite_2026-11-11.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n=== 保存结果到 phase5_rag_sqlite_2026-11-11.json ===")
