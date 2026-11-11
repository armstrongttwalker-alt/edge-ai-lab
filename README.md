# CIX P1 Edge AI Lab

基于 **Radxa Orion O6N（CIX P1 芯片）** 的边缘 AI 实验室。系统验证 CPU / GPU / NPU 异构算力，逐步构建本地 AI 推理服务与文档问答能力，形成可复用的边缘 AI 实验记录。

## 硬件平台

| 项目 | 规格 |
|---|---|
| 开发板 | Radxa Orion O6N |
| SoC | CIX P1 |
| CPU | ARM64 · 12 核（8× A720 + 4× A520） |
| RAM | 32 GB |
| 存储 | 1 TB NVMe |
| NPU | Zhouyi V3（1 Cluster × 3 Core） |
| GPU | 集成 GPU |
| 操作系统 | Debian 12 |
| 内核 | 6.6.89-3-sky1 |
| Python | 3.11 |

## 项目目标

在 CIX P1 芯片上完成六个方向的实验，最终形成一套属于自己的边缘 AI 硬件实验记录与可运行的服务节点。

| 阶段 | 方向 | 状态 |
|---|---|---|
| Phase 0 | 基础环境检查 | ✅ 完成 |
| Phase 1 | NPU Bring-up | ✅ 完成（7/7） |
| Phase 2 | CPU / GPU / NPU 基准测试 | ⏳ 未开始 |
| Phase 3 | AI 模型部署 | 🟡 部分完成（LLM/FlagOS 已跑通） |
| Phase 4 | AI 推理 API 服务 | ⏳ 未开始 |
| Phase 5 | 文档 RAG 知识库 | ⏳ 未开始 |
| Phase 6 | Docker 部署与运维 | ⏳ 未开始 |

详细进度见 [PROGRESS.md](./PROGRESS.md)，完整规划见 [PROJECT_PLAN.md](./PROJECT_PLAN.md)。

## 两条技术路线

本项目并行探索两条互补的加速路线。

**NPU 路线（Phase 1–2）** — 通过 CIX 厂商软件栈调用 Zhouyi V3 NPU：

```
Python → ONNX Runtime (zhouyi) → CIX EP → NOE Runtime → aipu 驱动 → NPU
```

**CPU LLM 路线（Phase 3）** — 通过 FlagOS 在 ARM CPU 上优化大模型推理：

```
vLLM → vLLM-plugin-FL → FlagGems → FlagTree-CPU (MLIR/LLVM) → A720 Neon
```

## 已完成实验速览

**FlagOS LLM 推理**（MiniCPM-V 4.7，1B 多模态，W8A8 量化）：三路对比 Stock / Minimal / FlagOS。
- PP512：183 → 589 tok/s（**+223%**）
- TG128：11.7 → 21.9 tok/s（**+87%**）

详见 [experiments/phase3-model-deploy/llm-flagos/](./experiments/phase3-model-deploy/llm-flagos/)。

## 文档导航

- [PROJECT_PLAN.md](./PROJECT_PLAN.md) — 项目规划书，六阶段完整计划、验收标准、依赖关系
- [PROGRESS.md](./PROGRESS.md) — 实验进度记录，按时间线持续更新
- [CLAUDE.md](./CLAUDE.md) — Claude Code 工作指南
- `experiments/` — 各阶段实验记录、脚本与证据

## 目录结构

```
AI-LAB/
├── README.md
├── PROJECT_PLAN.md
├── PROGRESS.md
├── CLAUDE.md
└── experiments/
    ├── phase0-env-check/        # 基础环境检查
    ├── phase1-npu-bringup/      # NPU 启动与运行时验证
    ├── phase2-benchmark/        # 异构算力基准测试
    ├── phase3-model-deploy/     # 模型部署（MobileNetV2 / YOLO / LLM）
    │   └── llm-flagos/          # ✅ FlagOS + vLLM LLM 推理实验
    ├── phase4-api-service/      # FastAPI 推理服务
    ├── phase5-rag/              # 文档 RAG
    └── phase6-docker-ops/       # 部署与运维
```

## 实验原则

- 每条命令都有明确目的，不为完成命令而执行命令。
- 上一步未验证通过，不贸然进入下一步。
- 软件安装成功 ≠ 硬件加速成功，必须通过 Provider 分配与设备日志双重确认。
