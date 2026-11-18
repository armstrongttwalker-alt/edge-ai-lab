"""
Edge AI Gateway - FastAPI 推理网关
统一路由 LLM 和视觉模型推理请求
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import numpy as np
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edge-ai-gateway")

app = FastAPI(
    title="Edge AI Gateway",
    description="CIX P1 Edge AI Lab - 统一推理网关",
    version="1.0.0"
)

# Backend URLs
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")
NPU_WORKSPACE = "/opt/edge-ai-lab/npu-workspace"
RAG_DOCS_DIR = os.getenv("RAG_DOCS_DIR", "/opt/edge-ai-lab/rag-docs")

# ==================== 内置 RAG（服务端语义检索增强） ====================
# 绕开 Open WebUI 的 RAG 中间件，在网关内部完成"向量化→检索→注入→回答"。
# 向量模型：bge-m3（ONNX，1024 维，多语言，中文强）。
# 切块：langchain MarkdownHeaderTextSplitter（按标题层级切，每块带标题路径）。

import glob

EMBED_BASE = os.getenv(
    "EMBED_MODEL_DIR",
    "/opt/edge-ai-lab/models/models/BAAI--bge-m3/snapshots/master/onnx",
)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
RAG_COLLECTION = "edge_ai_rag_bge"   # 换模型后维度变了，用新 collection
EMBED_DIM = 1024


class _Embedder:
    """bge-m3 ONNX 语义向量化。模型直接输出 sentence_embedding（已池化）。"""
    def __init__(self):
        from transformers import AutoTokenizer
        import onnxruntime as ort
        self.tok = AutoTokenizer.from_pretrained(EMBED_BASE)
        self.sess = ort.InferenceSession(
            f"{EMBED_BASE}/model.onnx", providers=["CPUExecutionProvider"])
        self.innames = [i.name for i in self.sess.get_inputs()]
        self.outnames = [o.name for o in self.sess.get_outputs()]

    def embed(self, text):
        enc = self.tok(text, return_tensors="np", truncation=True,
                       max_length=512, padding=True)
        feed = {n: enc[n].astype(np.int64) for n in self.innames if n in enc}
        outs = self.sess.run(None, feed)
        # bge-m3 输出含 sentence_embedding；若不存在则退回 CLS pooling
        if "sentence_embedding" in self.outnames:
            emb = outs[self.outnames.index("sentence_embedding")][0]
        else:
            emb = outs[0][0, 0]  # last_hidden_state 的 CLS token
        return (emb / np.linalg.norm(emb)).astype(np.float32)


class _RagStore:
    """文档切块 → embedding → 存入 Qdrant，查询时做向量检索。"""
    def __init__(self):
        self.embedder = None
        self.client = None
        self.ready = False

    def _chunk(self, text):
        # 用 langchain MarkdownHeaderTextSplitter 按标题层级切块，
        # 每块带上完整标题路径（如 "FlagOS Overview > ... > Open-source core libraries"），
        # 保留文档层级，解决"子项脱离父标题"导致的检索召回偏差。
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter)
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1"), ("##", "H2"),
                                 ("###", "H3"), ("####", "H4")],
            strip_headers=False,
        )
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=100)
        chunks = []
        for d in header_splitter.split_text(text):
            path = " > ".join(str(v) for v in d.metadata.values())
            for piece in char_splitter.split_text(d.page_content):
                # 每块前面带标题路径，让向量同时编码"归属"信息
                chunks.append((path + "\n" + piece) if path else piece)
        return chunks

    def build(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        self.embedder = _Embedder()
        self.client = QdrantClient(url=QDRANT_URL)
        # 收集文档分块
        items = []  # (source, text)
        if os.path.isdir(RAG_DOCS_DIR):
            for fn in sorted(os.listdir(RAG_DOCS_DIR)):
                fp = os.path.join(RAG_DOCS_DIR, fn)
                if not os.path.isfile(fp):
                    continue
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    continue
                for ch in self._chunk(text):
                    items.append((fn, ch))
        if not items:
            return 0
        # 重建 collection
        self.client.recreate_collection(
            collection_name=RAG_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        points = []
        for i, (src, txt) in enumerate(items):
            vec = self.embedder.embed(txt)
            points.append(PointStruct(
                id=i, vector=vec.tolist(),
                payload={"source": src, "text": txt},
            ))
        self.client.upsert(collection_name=RAG_COLLECTION, points=points)
        self.ready = True
        return len(points)

    def search(self, query, top_k=4):
        if not self.ready:
            return []
        qv = self.embedder.embed(query).tolist()
        res = self.client.query_points(
            collection_name=RAG_COLLECTION, query=qv, limit=top_k,
        ).points
        return [(p.payload["source"], p.payload["text"], float(p.score))
                for p in res]


RAG = _RagStore()
try:
    _n = RAG.build()
    logger.info(f"RAG(embedding) indexed {_n} chunks into Qdrant collection '{RAG_COLLECTION}'")
except Exception as _e:
    logger.error(f"RAG build failed: {_e}")

# ==================== Models ====================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "qwen2.5"
    messages: List[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.7
    stream: bool = False

class VisionRequest(BaseModel):
    model: str = "mobilenetv2"
    image: Optional[List[List[List[List[float]]]]] = None  # [1,3,224,224]
    return_topk: int = 5

# ==================== Endpoints ====================

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "edge-ai-gateway", "version": "1.0.0"}

@app.get("/v1/models")
async def list_models():
    """列出可用模型（OpenAI 标准格式：data 数组）"""
    import time as _t
    created = int(_t.time())
    return {
        "object": "list",
        "data": [
            {"id": "qwen2.5", "object": "model", "created": created, "owned_by": "llama.cpp"},
            {"id": "qwen2.5-rag", "object": "model", "created": created, "owned_by": "edge-ai-lab"},
            {"id": "mobilenetv2", "object": "model", "created": created, "owned_by": "onnxruntime-zhouyi"},
            {"id": "yolov8n", "object": "model", "created": created, "owned_by": "onnxruntime-zhouyi"},
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    """LLM 对话 - 透传到 llama-server，支持流式(stream)和非流式。

    直接转发 Open WebUI 的原始请求体，不做字段裁剪，
    避免丢失 stream / tools / 采样参数等字段。
    """
    try:
        body = await raw_request.json()
    except Exception:
        raise HTTPException(400, "请求体不是合法 JSON")

    # 如果选的是 qwen2.5-rag：在网关内部做检索增强
    if body.get("model") == "qwen2.5-rag" and RAG.ready:
        msgs = body.get("messages", [])
        # 只保留真实的 user/assistant 对话，过滤 Open WebUI 注入的
        # role=tool（available_skills 等）和空消息，避免模型被带偏。
        clean = []
        for m in msgs:
            role = m.get("role")
            cont = str(m.get("content", "")).strip()
            if role in ("user", "assistant") and cont:
                clean.append({"role": role, "content": cont})

        last_user = next((m for m in reversed(clean)
                          if m["role"] == "user"), None)
        if last_user:
            last_q = last_user["content"]
            # 检索查询：合并最近几轮 user 消息，保留话题上下文。
            # 追问（如 "explain in details"）本身没有实质语义，
            # 带上前文才能检索到正确主题。
            user_turns = [m["content"] for m in clean if m["role"] == "user"]
            retrieval_query = " ".join(user_turns[-3:]) if len(user_turns) > 1 else last_q
            hits = RAG.search(retrieval_query, top_k=15)
            if hits:
                context = "\n\n".join(
                    f"[source:{src}]\n{txt}" for src, txt, _ in hits
                )
                # system 提示用中性英文写，避免把输出语言带偏；
                # 明确要求"跟随用户最后一条消息的语言"。
                sys_msg = {
                    "role": "system",
                    "content": (
                        "You are a knowledge-base QA assistant. Reference passages are "
                        "provided below, separated by [source]. Each passage begins with "
                        "its heading path in the form 'A > B > C' showing exactly where it "
                        "sits in the document.\n"
                        "Rules:\n"
                        "1. Answer ONLY from the reference passages. Do NOT use outside "
                        "knowledge and do NOT invent anything. If the passages do not "
                        "contain the answer, say so plainly.\n"
                        "2. Copy every proper name EXACTLY as written, with its full "
                        "prefix. Never abbreviate or drop a prefix: write 'FlagTree', "
                        "'FlagScale', 'FlagCX' in full — never 'Tree', 'Scale', 'CX'.\n"
                        "3. For list / enumeration questions (which components / libraries "
                        "/ tools / plugins), list EVERY matching item found in the "
                        "passages. Do NOT stop after a few, do NOT summarize mid-list, do "
                        "NOT repeat an item.\n"
                        "4. Use the heading path to decide category membership. Include "
                        "ONLY items that belong to the category being asked about; do NOT "
                        "pull in items from other categories. A category heading (e.g. "
                        "'Multi-domain operator libraries') is a group label, NOT an item "
                        "itself — list the named entities under it, not the label.\n"
                        "5. CRITICAL: Reply in the SAME language as the user's LAST "
                        "message. English question -> English answer; Chinese question -> "
                        "Chinese answer.\n\n"
                        f"=== REFERENCE PASSAGES ===\n{context}\n=== END ===\n\n"
                        f"(The user's last message is: {last_q})"
                    ),
                }
                body["messages"] = [sys_msg] + clean
            else:
                body["messages"] = clean
        # 转发给真实模型，并去掉可能触发工具调用的字段
        body["model"] = "qwen2.5"
        body.pop("tools", None)
        body.pop("tool_choice", None)

    is_stream = bool(body.get("stream", False))
    url = f"{LLAMA_SERVER_URL}/v1/chat/completions"

    if is_stream:
        # 流式：用 SSE 透传 llama-server 的分块响应
        async def event_stream():
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("POST", url, json=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        try:
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        except httpx.ConnectError:
            raise HTTPException(503, "llama-server 未启动，请先在 :8081 启动 llama-server")
    else:
        # 非流式：直接返回 JSON
        async with httpx.AsyncClient(timeout=300) as client:
            try:
                resp = await client.post(url, json=body)
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            except httpx.ConnectError:
                raise HTTPException(503, "llama-server 未启动，请先在 :8081 启动 llama-server")
            except Exception as e:
                raise HTTPException(500, f"LLM 推理失败: {str(e)}")

@app.post("/v1/vision/classify")
async def vision_classify(request: VisionRequest):
    """视觉分类 - 使用 NPU 推理 MobileNetV2"""
    import onnxruntime as ort

    os.chdir(NPU_WORKSPACE)

    model_path = f"/opt/edge-ai-lab/models/{request.model}/{request.model}_qdq.onnx"

    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["ZhouyiExecutionProvider", "CPUExecutionProvider"]
        )
        inp = sess.get_inputs()[0]

        # Use provided image or random
        if request.image:
            dummy = np.array(request.image, dtype=np.float32)
        else:
            dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)

        t0 = time.perf_counter()
        output = sess.run(None, {inp.name: dummy})
        latency_ms = (time.perf_counter() - t0) * 1000

        logits = output[0][0]
        top_k_idx = np.argsort(logits)[-request.return_topk:][::-1]
        top_k_probs = np.exp(logits[top_k_idx]) / np.sum(np.exp(logits))

        return {
            "model": request.model,
            "provider": "ZhouyiExecutionProvider",
            "latency_ms": round(latency_ms, 2),
            "top_k": [
                {"class_id": int(idx), "score": float(prob)}
                for idx, prob in zip(top_k_idx, top_k_probs)
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"视觉推理失败: {str(e)}")

@app.post("/v1/vision/detect")
async def vision_detect(request: VisionRequest):
    """目标检测 - 使用 NPU 推理 YOLOv8n"""
    import onnxruntime as ort

    os.chdir(NPU_WORKSPACE)

    model_path = "/opt/edge-ai-lab/models/yolo/yolov8n_qdq.onnx"

    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["ZhouyiExecutionProvider", "CPUExecutionProvider"]
        )
        inp = sess.get_inputs()[0]

        if request.image:
            dummy = np.array(request.image, dtype=np.float32)
        else:
            dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)

        t0 = time.perf_counter()
        output = sess.run(None, {inp.name: dummy})
        latency_ms = (time.perf_counter() - t0) * 1000

        # output shape: (1, 84, 8400) = [4 coords + 80 classes] x 8400 boxes
        return {
            "model": "yolov8n",
            "provider": "ZhouyiExecutionProvider",
            "latency_ms": round(latency_ms, 2),
            "output_shape": list(output[0].shape),
            "num_detections": int(output[0].shape[2]),
        }
    except Exception as e:
        raise HTTPException(500, f"检测推理失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
