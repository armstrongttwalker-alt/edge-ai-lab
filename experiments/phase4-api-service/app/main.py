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
