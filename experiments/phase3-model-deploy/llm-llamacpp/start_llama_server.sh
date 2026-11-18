#!/bin/bash
# llama-server 启动脚本（LLM 后端，供 FastAPI 网关和 Open WebUI 使用）
# --jinja: 启用聊天模板和工具调用支持（Open WebUI 需要）
# 端口 8081（8080 留给 Open WebUI 的宿主机映射）
# 模型：Qwen2.5-3B-Instruct Q8_0。实测 Q4_K_M 会损伤指令遵循
#   （专有名词写残 FlagTree→Tree、整串重复），Q8 修复，详见
#   experiments/phase5-rag/eval/README.md 的三轮量化对比。

/usr/share/cix/bin/llama-server \
    -m /opt/edge-ai-lab/models/llm/qwen2.5-3b-q8.gguf \
    -t 8 \
    --host 0.0.0.0 \
    --port 8081 \
    --jinja \
    -n 512 \
    -c 8192
