# -*- coding: utf-8 -*-
"""云端 RAG 模块：OpenRouter 嵌入 + pgvector 检索 + （由网关转发给）云端 chat。

设计（与板子能力边界匹配：板子只做编排/存取，重活在云端）：
- 嵌入：OpenRouter qwen3-embedding-8b，dimensions=1024（MRL 截断，适配 pgvector HNSW）
- 向量库：PostgreSQL + pgvector，表 rag_chunks(embedding vector(1024)) + HNSW 余弦
- 语料：仅英文（源语言）。已验证跨语言对齐良好，中文提问可命中英文块。
- 生成：不在本模块，由网关把检索到的上下文拼好后转发给云端 chat（DeepSeek）。

配置全部来自 /opt/edge-ai-lab/.env。
"""
import os, json, time, urllib.request
import psycopg2


def _load_env(path="/opt/edge-ai-lab/.env"):
    env = {}
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return env


class CloudRAG:
    def __init__(self):
        env = _load_env()
        self.embed_url = env.get("EMBED_BASE_URL", "").rstrip("/") + "/embeddings"
        self.embed_key = env.get("EMBED_API_KEY", "")
        self.embed_model = env.get("EMBED_MODEL", "qwen/qwen3-embedding-8b")
        self.dim = int(env.get("EMBED_DIM", "1024"))
        self.pg_dsn = env.get("PG_DSN", "")
        # reranker（cross-encoder 精排，治纯向量检索"提到≠定义"的病）
        self.rerank_url = env.get("RERANK_BASE_URL", "").rstrip("/") + "/rerank"
        self.rerank_key = env.get("EMBED_API_KEY", "")   # 复用 OpenRouter key
        self.rerank_model = env.get("RERANK_MODEL", "")
        self.rerank_on = bool(self.rerank_model and env.get("RERANK_BASE_URL"))
        self.ready = bool(self.embed_key and self.pg_dsn)

    def embed(self, text, retries=2, timeout=15):
        """调 OpenRouter 嵌入，dim=1024。快速失败：OpenRouter 断网时几秒内放弃，
        不死等（原 60s×5 会把网关拖死十几分钟 → 不响应 /health → 界面 model not found）。"""
        body = json.dumps({"model": self.embed_model, "input": text,
                           "dimensions": self.dim}).encode()
        last = None
        for i in range(retries):
            try:
                req = urllib.request.Request(
                    self.embed_url, data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.embed_key}"})
                r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
                return r["data"][0]["embedding"]
            except Exception as e:
                last = e
                if i < retries - 1:
                    time.sleep(2)
        raise last

    def _rerank(self, query, cands, top_k):
        """OpenRouter cohere/rerank-v3.5 精排。cands: [(src,title,content,cos)]。
        失败则退回向量序（不致命）。返回 top_k。"""
        if not self.rerank_on or not cands:
            return cands[:top_k]
        docs = [c[2] for c in cands]
        body = json.dumps({"model": self.rerank_model, "query": query,
                           "documents": docs, "top_n": top_k}).encode()
        for i in range(2):   # 快速失败：rerank 挂了退回向量序，不死等
            try:
                req = urllib.request.Request(
                    self.rerank_url, data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.rerank_key}"})
                r = json.loads(urllib.request.urlopen(req, timeout=15).read())
                out = []
                for item in r["results"]:
                    idx = item["index"]
                    s, t, ct, _ = cands[idx]
                    out.append((s, t, ct, float(item.get("relevance_score", 0.0))))
                return out
            except Exception:
                if i < 1:
                    time.sleep(2)
        return cands[:top_k]   # rerank 失败退回向量序

    _NOISE = ("source NOT LIKE '%%/model_readmes/%%' "
              "AND source NOT LIKE '%%/chip_adaptation_guide/%%' "
              "AND source NOT LIKE '%%/references/test/%%'")

    def _dense(self, cur, qv, k):
        """向量召回 top-k。返回 [(id, source, title, content)]。"""
        cur.execute(
            f"SELECT id, source, title, content FROM rag_chunks "
            f"WHERE {self._NOISE} ORDER BY embedding <=> %s::vector LIMIT %s",
            (qv, k))
        return cur.fetchall()

    def _fts(self, cur, query, k):
        """PostgreSQL 全文检索 top-k（english 分词，content_tsv GIN 索引）。
        对操作类/含具体关键词的问题强（实测 how-to 类排名碾压向量）。"""
        cur.execute(
            f"SELECT id, source, title, content FROM rag_chunks "
            f"WHERE {self._NOISE} AND content_tsv @@ websearch_to_tsquery('english', %s) "
            f"ORDER BY ts_rank(content_tsv, websearch_to_tsquery('english', %s)) DESC LIMIT %s",
            (query, query, k))
        return cur.fetchall()

    @staticmethod
    def _rrf(lists, k=60):
        """Reciprocal Rank Fusion：融合多路检索的排名（不比原始分数）。
        lists: 每路是 [(id, source, title, content)] 按各自排名。返回融合后 [(id,src,title,content)]。"""
        score, meta = {}, {}
        for lst in lists:
            for rank, row in enumerate(lst):
                rid = row[0]
                score[rid] = score.get(rid, 0.0) + 1.0 / (k + rank + 1)
                meta[rid] = row
        order = sorted(score, key=lambda x: -score[x])
        return [meta[rid] for rid in order]

    def search(self, query, top_k=8, coarse_k=50):
        """Hybrid 检索（实验 E 方案，实测最佳）：
        Dense(向量) + FTS(关键词) 各粗召回 coarse_k → RRF 融合 → cohere rerank 精排 top_k。
        - Dense 擅长定义/综合类，FTS 擅长操作/关键词类，RRF 让两者互补；
        - 去掉了 query 改写（实测对最终 MRR 无额外增益，却慢 10-20s 且多依赖一次 OpenRouter）。
        噪声目录 SQL 层排除。OpenRouter 断网时 dense 走不了，仍可退化为纯 FTS。"""
        conn = psycopg2.connect(self.pg_dsn)
        try:
            cur = conn.cursor()
            cur.execute("SET hnsw.ef_search = 200;")
            # Dense（嵌入可能因 OpenRouter 抖动失败 → 退化为纯 FTS）
            dense_rows = []
            try:
                qv = self.embed(query)
                dense_rows = self._dense(cur, qv, coarse_k)
            except Exception:
                dense_rows = []
            # FTS（本地，稳）
            try:
                fts_rows = self._fts(cur, query, coarse_k)
            except Exception:
                fts_rows = []
            cur.close()
        finally:
            conn.close()
        # RRF 融合两路
        fused = self._rrf([r for r in (dense_rows, fts_rows) if r])
        # 转成 rerank 所需的 (source, title, content, score) 形态
        cands = [(src, title, content, 0.0) for (_id, src, title, content) in fused]
        # cross-encoder 精排
        return self._rerank(query, cands, top_k)

    def count(self):
        """返回 (块数, 文档数)，供健康检查/启动日志。"""
        conn = psycopg2.connect(self.pg_dsn)
        try:
            cur = conn.cursor()
            cur.execute("SELECT count(*), count(DISTINCT source) FROM rag_chunks")
            n, ns = cur.fetchone()
            cur.close()
            return int(n), int(ns)
        finally:
            conn.close()
