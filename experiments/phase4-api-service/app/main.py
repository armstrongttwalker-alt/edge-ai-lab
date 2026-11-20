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
# 本地 bge-m3+Qdrant RAG 启动即重建，CPU 上很慢；默认跳过，只在显式开启时构建。
# 云端方案(qwen-rag)不需要它。
if os.getenv("ENABLE_LOCAL_RAG", "0") == "1":
    try:
        _n = RAG.build()
        logger.info(f"RAG(embedding) indexed {_n} chunks into Qdrant collection '{RAG_COLLECTION}'")
    except Exception as _e:
        logger.error(f"RAG build failed: {_e}")
else:
    logger.info("本地 bge-m3 RAG 已跳过(ENABLE_LOCAL_RAG!=1)，使用云端 qwen-rag")

# ==================== 云端 RAG（OpenRouter 嵌入 + pgvector + DeepSeek 生成）====================
from app.cloud_rag import CloudRAG
CLOUD_RAG = CloudRAG()
try:
    if CLOUD_RAG.ready:
        _cn, _cs = CLOUD_RAG.count()
        logger.info(f"CloudRAG 就绪：pgvector 内 {_cn} 块 / {_cs} 文档")
    else:
        logger.warning("CloudRAG 未就绪：.env 缺 EMBED_API_KEY 或 PG_DSN")
except Exception as _e:
    logger.error(f"CloudRAG 连接失败: {_e}")

# 云端 chat（由 .env 配置，OpenAI 兼容）
_ENV = {}
try:
    for _line in open("/opt/edge-ai-lab/.env", encoding="utf-8"):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1); _ENV[_k.strip()] = _v.strip()
except Exception:
    pass
CHAT_BASE_URL = _ENV.get("CHAT_BASE_URL", "").rstrip("/")
CHAT_API_KEY = _ENV.get("CHAT_API_KEY", "")
CHAT_MODEL = _ENV.get("CHAT_MODEL", "deepseek-flash")

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
            {"id": "qwen-rag", "object": "model", "created": created, "owned_by": "cloud-rag(pgvector+deepseek)"},
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

    # ===== 云端 RAG：qwen-rag（OpenRouter 嵌入 + pgvector 检索 + DeepSeek 生成）=====
    if body.get("model") == "qwen-rag" and CLOUD_RAG.ready:
        msgs = body.get("messages", [])
        # Open WebUI 的"系统自动请求"（生成标题/标签/follow-up）不是用户真实提问，
        # 不应触发完整 RAG（白跑 OpenRouter、与真实提问抢配额）。识别后直接转发
        # 给普通 chat 后端处理，不走检索增强。
        _joined = " ".join(str(m.get("content", "")) for m in msgs)
        _owui_markers = ("Generate a concise title", "Generate 1-3 broad tags",
                         "### Task:", "summarizing the chat history",
                         "Generate a brief 3-5 word title")
        if any(mk in _joined for mk in _owui_markers):
            logger.info("[qwen-rag] 识别为 Open WebUI 系统请求(标题/标签)，跳过 RAG，直接转发")
            body["model"] = "qwen2.5"
            body.pop("tools", None); body.pop("tool_choice", None)
            url = f"{LLAMA_SERVER_URL}/v1/chat/completions"
            async with httpx.AsyncClient(timeout=60) as client:
                try:
                    resp = await client.post(url, json=body)
                    return JSONResponse(content=resp.json(), status_code=resp.status_code)
                except Exception:
                    return JSONResponse(content={"choices":[{"message":{"role":"assistant","content":"Chat"}}]})
        # 诊断：打印 Open WebUI 发来的原始 messages 结构（角色+长度+前80字）
        try:
            _dbg = [{"role": m.get("role"),
                     "len": len(str(m.get("content", ""))),
                     "head": str(m.get("content", ""))[:80]} for m in msgs]
            logger.info(f"[qwen-rag] 收到 {len(msgs)} 条消息: {_dbg}")
        except Exception:
            pass
        # 过滤 Open WebUI 注入的 role=tool / 空消息，只留真实对话
        clean = [{"role": m.get("role"), "content": str(m.get("content", "")).strip()}
                 for m in msgs
                 if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()]
        last_user = next((m for m in reversed(clean) if m["role"] == "user"), None)
        if not last_user:
            raise HTTPException(400, "无有效用户消息")
        last_q = last_user["content"]
        # 检索查询：默认只用最后一条 user 消息，避免历史话题污染检索
        # （Open WebUI 会带整段历史，多轮时前一主题会把检索带偏）。
        retrieval_query = last_q
        logger.info(f"[qwen-rag] 检索查询: {retrieval_query[:120]}")
        try:
            hits = CLOUD_RAG.search(retrieval_query, top_k=8)
        except Exception as e:
            raise HTTPException(503, f"检索失败(嵌入/pgvector): {str(e)[:120]}")
        logger.info(f"[qwen-rag] 命中 {len(hits)} 块: " +
                    ", ".join(f"{s}({cos:.2f})" for s, t, c, cos in hits[:8]))
        context = "\n\n".join(
            f"[source:{src}]\n{content}" for src, title, content, cos in hits)
        sys_msg = {
            "role": "system",
            "content": (
                "You are a knowledge-base QA assistant for the FlagOS documentation. "
                "Reference passages are provided below, separated by [source]. Each "
                "passage starts with its heading path (A > B > C).\n"
                "General rules:\n"
                "1. Base your answer on the reference passages. Do NOT fabricate facts, "
                "versions, or descriptions that are not supported by the passages. Only "
                "if the passages are truly unrelated to the question should you say the "
                "information is not available.\n"
                "2. Copy proper names exactly, with the full Flag* prefix.\n"
                "3. Reply in the SAME language as the user's last message (passages are "
                "English; if the user asked in Chinese, answer in Chinese).\n"
                "\n"
                "How to answer by question type:\n"
                "- Definition questions ('what is X', 'X 是什么', 'introduce X'): Explain "
                "what X is using the descriptions in the passages. Synthesize a clear "
                "1-3 sentence definition even if the details are spread across several "
                "passages. Do NOT refuse just because there is no single verbatim "
                "definition sentence — compose one from what the passages say about X.\n"
                "- List questions ('what are the X', 'which X', 'X 有哪些'): List EVERY "
                "matching item found in the passages, grouped by category when natural. "
                "Be exhaustive; do not stop after a few.\n"
                "- How-to questions ('how to X', '如何 X'): Give the steps or commands "
                "from the passages, in order.\n"
                "- For every item you name, add its description ONLY if the passages "
                "provide one; never invent a meaning from the name alone.\n\n"
                f"=== REFERENCE PASSAGES ===\n{context}\n=== END ===\n\n"
                f"(The user's last message is: {last_q})"
            ),
        }
        fwd = {
            "model": CHAT_MODEL,
            "messages": [sys_msg] + clean,
            # DeepSeek deepseek-flash 会先 reasoning 再答，给足 token 否则答案被挤空
            "max_tokens": int(body.get("max_tokens") or 1500),
            "temperature": min(float(body.get("temperature", 0.2) or 0.2), 0.3),
            "stream": bool(body.get("stream", False)),
        }
        url = f"{CHAT_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {CHAT_API_KEY}"}
        if fwd["stream"]:
            async def cloud_stream():
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream("POST", url, json=fwd, headers=headers) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
            return StreamingResponse(cloud_stream(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient(timeout=300) as client:
                try:
                    resp = await client.post(url, json=fwd, headers=headers)
                    return JSONResponse(content=resp.json(), status_code=resp.status_code)
                except Exception as e:
                    raise HTTPException(502, f"云端 chat 调用失败: {str(e)[:120]}")

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

            # 空泛/总览型问法（"详细介绍"/"explain in details"/"有哪些"）向量发散，
            # 常检索不到具体定义块 → 模型只拿到名字、没拿到定义 → 看名字臆造描述
            # （实测 FlagCX 被编成 "complex number operations"）。
            # 对这类问法：扩展检索词 + 扩大召回，尽量把各板块的定义块都捞回来。
            # 注：这是适配当前小语料的启发式；语料上规模后应换 LLM query 改写 /
            # reranker（纯 CPU 下 LLM 改写会给每次问答再加一次推理，故此处从简）。
            _ql = last_q.lower()
            _broad_kw = ["explain", "overview", "detail", "introduce", "tell me about",
                         "介绍", "详细", "概述", "总览", "整体", "有哪些", "组成",
                         "architecture", "架构", "components", "核心"]
            is_broad = len(_ql) < 40 or any(k in _ql for k in _broad_kw)
            if is_broad:
                retrieval_query = (
                    retrieval_query +
                    " core libraries operators plugins developer tools platform "
                    "services domain-specific projects architecture components "
                    "核心库 算子库 插件 开发者工具 平台服务 领域项目 架构 组件"
                )
                top_k = 30
            else:
                top_k = 15
            hits = RAG.search(retrieval_query, top_k=top_k)
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
                        "Chinese answer.\n"
                        "6. ANTI-HALLUCINATION: Give an item a description ONLY if that "
                        "description appears in the passages. If a name appears but the "
                        "passages provide no description for it, list the name ALONE and "
                        "do NOT invent, guess, or infer a description from the name itself "
                        "(e.g. never guess that 'FlagCX' means complex numbers or "
                        "'FlagTree' means tree operations).\n\n"
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
        # 强制低温：Open WebUI 默认采样温度偏高（易漂移/臆造），
        # RAG 场景要忠实于文档，锁到 <=0.2。
        try:
            body["temperature"] = min(float(body.get("temperature", 0.2)), 0.2)
        except (TypeError, ValueError):
            body["temperature"] = 0.2

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
