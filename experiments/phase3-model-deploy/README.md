# Phase 3 · AI 模型部署 🟡

**状态：LLM/FlagOS 子实验已完成，其余未开始**

## 子实验

| 子实验 | 内容 | 状态 |
|---|---|---|
| 3A · MobileNetV2 | ONNX INT8 视觉分类，NPU 推理 | ⏳ 依赖 Phase 1 |
| 3B · YOLO | 目标检测，NPU 推理，多路视频 | ⏳ |
| 3C · LLM FlagOS | MiniCPM-V 4.7，CPU 优化推理 | ✅ |
| 3D · llama.cpp | GGUF 量化模型，CPU 推理 | ⏳ |
| 3E · Ollama | 模型管理层，对接 Open WebUI | ⏳ |

## 已完成

**3C · FlagOS + vLLM：** 见 [llm-flagos/](./llm-flagos/)

PP512 提升 +223%，TG128 提升 +87%（相对 Minimal 基线）。

## LLM 实验模型候选

| 模型 | 参数量 | 用途 |
|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | 快速验证 |
| Qwen2.5-3B-Instruct | 3B | 日常实验 |
| Qwen2.5-7B-Instruct | 7B | 主力实验 |
| 14B 量化模型 | ~14B | 内存压力测试 |

注意：模型能否运行（内存限制）与能否全图 NPU 执行（算子覆盖率）是独立问题，须分开记录。
