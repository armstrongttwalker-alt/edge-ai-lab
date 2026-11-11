# Phase 4 · FastAPI 推理服务 小白教程

> 本教程说明如何用 FastAPI 构建一个统一的 AI 推理网关，对外提供 OpenAI 兼容接口，对内路由到不同的推理后端（llama.cpp、Ollama、ONNX NPU）。
>
> **实验状态：⏳ 待执行（依赖 Phase 3）**

---

## 一、实验目标

构建一个运行在 Radxa Orion O6N 上的 FastAPI 服务，统一管理 LLM 和视觉模型推理，提供 OpenAI 兼容 HTTP 接口，支持 Docker 部署。

---

## 二、背景知识

### 2.1 为什么需要推理网关？

如果每个应用直接调用不同的推理引擎：
- Open WebUI 直接连 Ollama
- 自己写的脚本用 llama.cpp
- 视觉识别用 ONNX Runtime

未来换一个后端，所有调用方都要改。用网关统一入口，好处是：
- **后端可替换**：换推理引擎不影响调用方
- **统一日志**：所有请求集中记录
- **请求路由**：文字走 LLM，图片走视觉模型
- **速率控制**：防止并发推理把板子打满

### 2.2 OpenAI 兼容接口

OpenAI 的 `/v1/chat/completions` 接口已成为 LLM 推理的事实标准。Open WebUI、Dify 等工具都能接入兼容此接口的服务。

请求格式：
```json
{
  "model": "qwen2.5",
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "max_tokens": 100
}
```

---

## 三、服务架构

```
Open WebUI / curl / 任意客户端
        ↓ POST /v1/chat/completions
FastAPI Gateway（端口 8000）
        ├─ /v1/chat/completions → LLM Backend（Ollama / llama.cpp）
        ├─ /v1/vision           → ONNX NPU（ZhouyiEP）
        └─ /v1/embeddings       → Embedding 模型
```

---

## 四、项目结构

```
experiments/phase4-api-service/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── routers/
│   │   ├── chat.py          # LLM 路由
│   │   ├── vision.py        # 视觉模型路由
│   │   └── embeddings.py    # Embedding 路由
│   └── backends/
│       ├── ollama.py        # Ollama 后端
│       ├── llamacpp.py      # llama.cpp 后端
│       └── onnx_npu.py      # ONNX NPU 后端
├── Dockerfile
├── compose.yaml
└── requirements.txt
```

---

## 五、核心代码

### main.py

```python
from fastapi import FastAPI
from app.routers import chat, vision, embeddings

app = FastAPI(title="Edge AI Gateway", version="1.0.0")

app.include_router(chat.router, prefix="/v1")
app.include_router(vision.router, prefix="/v1")
app.include_router(embeddings.router, prefix="/v1")

@app.get("/health")
def health():
    return {"status": "ok"}
```

### routers/chat.py

```python
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import httpx, json

router = APIRouter()
OLLAMA_URL = "http://localhost:11434"

@router.post("/chat/completions")
async def chat_completions(request: dict):
    # 转发到 Ollama
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/v1/chat/completions",
            json=request
        )
    return resp.json()
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### compose.yaml

```yaml
services:
  api:
    build: .
    container_name: edge-ai-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    network_mode: host  # 访问宿主机的 Ollama
```

---

## 六、执行步骤

```bash
cd /opt/edge-ai-lab/experiments/phase4-api-service

# 安装依赖
pip install fastapi uvicorn httpx

# 本地运行测试
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 测试接口
curl http://localhost:8000/health
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"你好"}]}'

# Docker 部署
docker compose up -d
```

---

## 七、验收标准

- [ ] FastAPI 服务启动，`/health` 返回 OK
- [ ] `/v1/chat/completions` 能正确转发到 Ollama 并返回响应
- [ ] Docker Compose 部署成功
- [ ] Open WebUI 通过此网关访问本地模型

---

*本教程对应实验：Phase 4，待执行。最后更新：2026-11-11。*
