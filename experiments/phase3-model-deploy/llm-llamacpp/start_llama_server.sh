#!/bin/bash
# llama-server 启动脚本（LLM 后端，供 FastAPI 网关和 Open WebUI 使用）
# --jinja: 启用聊天模板和工具调用支持（Open WebUI 需要）
# 端口 8081（8080 留给 Open WebUI 的宿主机映射）

/usr/share/cix/bin/llama-server \
    -m /opt/edge-ai-lab/models/llm/qwen2.5-1.5b-q8_0.gguf \
    -t 8 \
    --host 0.0.0.0 \
    --port 8081 \
    --jinja \
    -n 128
