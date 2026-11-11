# Phase 4 · AI 推理 API 服务 ⏳

**状态：未开始（依赖 Phase 3）**

## 实验目标

构建统一的 FastAPI 推理服务，提供 OpenAI 兼容接口，支持 LLM 和视觉模型路由。将来更换推理后端，前端接口不变。

## 服务架构

```
Open WebUI
  └─ FastAPI Gateway（路由 / 鉴权 / 日志）
       ├─ /v1/chat/completions  → LLM（llama.cpp / Ollama / FlagOS）
       ├─ /v1/vision            → YOLO + NPU
       └─ /v1/embeddings        → Embedding 模型
```

## 技术选型

- **框架：** FastAPI + Uvicorn
- **接口规范：** OpenAI API 兼容（`/v1/chat/completions`）
- **部署：** Docker Compose

## 目录规划

```
experiments/phase4-api-service/
├── app/
│   ├── main.py
│   ├── routers/
│   │   ├── chat.py
│   │   ├── vision.py
│   │   └── embeddings.py
│   └── backends/
│       ├── llama_cpp.py
│       └── onnx_npu.py
├── Dockerfile
├── compose.yaml
└── requirements.txt
```

## 验收标准

- `curl http://<板端IP>:8000/v1/chat/completions` 返回正确响应
- LLM 和视觉路由均可独立调用
- Open WebUI 可通过该接口访问本地模型
