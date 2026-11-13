#!/bin/bash
# Phase 4 FastAPI 推理网关启动脚本
# 用法: bash start_gateway.sh
# 前提: llama-server 已在 :8081 运行

source /opt/edge-ai-lab/venv/bin/activate
export LD_LIBRARY_PATH=/usr/share/cix/lib:/usr/share/cix/lib/onnxruntime:$LD_LIBRARY_PATH
export LLAMA_SERVER_URL=http://localhost:8081
cd /opt/edge-ai-lab/services/api
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
