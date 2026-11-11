# CIX P1 Edge AI Lab 项目规划书

## 背景

本项目基于 **Radxa Orion O6N** 开发板（搭载 CIX P1 SoC），系统验证边缘设备的异构 AI 算力，并逐步构建可运行的本地 AI 推理与文档问答服务。这不是一次性的跑分，而是建设一个长期可复用的实验平台与开源实验记录。

核心问题不是"这台设备能跑多大的模型"，而是"CPU / GPU / NPU 三种算力各自适合什么任务、实际加速效果如何、能支撑什么样的服务"。因此项目从最底层的 NPU 驱动验证开始，逐层向上推进到模型部署、推理服务与 RAG 知识库。

## 项目目标

完成七个阶段的实验，最终交付：

1. 一套经过验证的 CIX P1 异构算力实验环境
2. NPU 推理链路打通的证据（非 CPU 回退）
3. CPU / GPU / NPU 基准性能对比报告
4. 视觉模型与本地 LLM 部署方案
5. FastAPI 推理服务（OpenAI 兼容接口）
6. 本地文档 RAG 知识库（ChromaDB + Qdrant 双方案）
7. Docker 化部署与监控方案
8. 开源 GitHub 项目，包含实验脚本、文档与原始数据

## 硬件平台

| 项目 | 规格 |
|---|---|
| 开发板 | Radxa Orion O6N |
| SoC | CIX P1 |
| CPU | ARM64 · 12 核（8× A720 + 4× A520） |
| GPU | Mali G720（Vulkan / OpenCL） |
| NPU | Zhouyi V3（NeuralONE），1 Cluster × 3 Core，标称 30 TOPS |
| RAM | 32 GB LPDDR5 |
| 存储 | 1 TB NVMe SSD |
| 操作系统 | Debian 12 |
| 内核 | 6.6.89-3-sky1 |
| Python | 3.11 |
| 容器 | Docker |

注意：Radxa 文档 NPU 算力标注 28.8 TOPS，产品页标注 30 TOPS。实验报告应分别记录官方规格与实测性能，不混为一谈。

## 技术架构

采用"宿主机负责硬件管理，Docker 负责应用服务，独立实验环境负责 NPU 开发"的分层架构。

```
┌─────────────────────────────────────────────┐
│  Debian 12 Host + NVMe                       │
│  Kernel · Drivers · Docker · Storage · Logs  │
└─────────────────────────────────────────────┘
        ┌──────────────┬──────────────┐
        │              │              │
   NPU Lab        LLM Lab       Vision Lab
   CIX Runtime   llama.cpp     YOLO / OpenCV
   ONNX Runtime  Ollama
   (Zhouyi)      FlagOS/vLLM
        │              │              │
        └──────────────┴──────────────┘
                     │
           Benchmark Lab · Performance / Power
                     │
        ┌────────────┴────────────┐
        │  Docker Application Services │
        │  Open WebUI · FastAPI        │
        │  Qdrant · ChromaDB           │
        │  Grafana · n8n               │
        └─────────────────────────────┘
```

设计原则：NPU 实验初期直接在宿主机运行，不急于容器化。原因是 NPU 涉及内核驱动、用户态驱动、设备节点和运行时库，先在宿主机验证硬件、再研究容器映射，可减少排错变量。

## NPU 软件栈说明

NPU 推理需要六层软件协作，任何一层缺失都会导致计算任务回退到 CPU：

```
Python 应用
  └─ ONNX Runtime（onnxruntime_zhouyi，厂商定制版，非 PyPI 通用版）
       └─ CIX NPU Execution Provider
            └─ CIX NOE Runtime（libnoe.so）
                 └─ Linux aipu 驱动（/dev/aipu）
                      └─ Zhouyi V3 NPU
```

模型转换工具链（NeuralONE SDK / cixbuild）是 x86_64 格式，须在 PC 上完成模型转换与量化，再将结果复制到板端运行推理。

---

## 阶段规划

### Phase 0 · 基础环境检查 ✅ 已完成

**目标：** 确认 Debian 12 环境满足所有后续实验的前提条件。

**验收标准（已满足）：**
- OS：Debian 12，Kernel 6.6.89-3-sky1，ARM64
- CPU 12 核，RAM 32 GB，NVMe 1 TB
- NPU 驱动已加载：`lsmod | grep aipu` → `aipu 98304 0`
- Zhouyi V3 识别：`dmesg | grep -i aipu` → `AIPU detected: zhouyi-v3`
- 设备节点正常：`/dev/aipu`（crw-rw-rw-）
- CIX 用户态库已安装：`libnoe.so`、`libonnxruntime.so`
- Docker 已安装

**交付物：** 硬件环境报告（见 `experiments/phase0-env-check/`）

---

### Phase 1 · NPU Bring-up 🔄 进行中（Step 4/7）

**目标：** 通过 CIX 厂商软件栈，让 Python 程序真正在 Zhouyi V3 NPU 上执行 ONNX 模型推理，并排除 CPU 回退。

**子步骤：**

| 步骤 | 内容 | 状态 |
|---|---|---|
| Step 1 | 检查 OS 和硬件 | ✅ 完成 |
| Step 2 | 确认 aipu 驱动加载 | ✅ 完成 |
| Step 3 | 确认 `/dev/aipu` 和 NOE 运行库 | ✅ 完成 |
| Step 4 | 安装 NumPy + 厂商 ONNX Runtime wheel | 🔄 进行中（NumPy 下载超时） |
| Step 5 | 验证 CIX NPU Execution Provider 注册 | ⏳ |
| Step 6 | 加载 MobileNetV2 INT8 执行推理 | ⏳ |
| Step 7 | 确认 NPU 实际执行（排除 CPU fallback） | ⏳ |

**当前阻塞：** NumPy pip 安装超时。解决方案：用国内镜像源（如阿里云 `https://mirrors.aliyun.com/pypi/simple/`）或在 Windows 机器下载 ARM64 wheel 后 scp 传入。

**验收标准：**
1. `onnxruntime_zhouyi` 导入无报错
2. `CIXExecutionProvider` 出现在 `ort.get_available_providers()` 中
3. 推理 Session 创建时指定 `providers=["CIXExecutionProvider"]` 成功
4. 模型计算图主要节点分配给 NPU（非全部回退 CPU）
5. `dmesg` 或 NOE 日志显示 NPU 收到计算任务
6. MobileNetV2 推理结果数值正确

**Python 环境：**
```bash
source /opt/edge-ai-lab/venv/bin/activate
# venv 创建参数：--system-site-packages（复用系统已装的 NOE 包）
```

**交付物：** NPU 推理验证记录（见 `experiments/phase1-npu-bringup/`）

---

### Phase 2 · CPU / GPU / NPU 基准测试 ⏳

**目标：** 在相同模型和相同输入条件下，量化三种算力的推理延迟、吞吐量和内存占用，形成可重复的性能报告。

**基准模型：** MobileNetV2 INT8（视觉分类），统一输入 224×224。

**测试矩阵：**

| 测试项 | CPU EP | GPU | NPU（Zhouyi） |
|---|---|---|---|
| FP32 推理 | ✅ | 视驱动 | — |
| FP16 推理 | ✅ | 视驱动 | 视算子支持 |
| INT8 推理 | ✅ | 视后端 | ✅ 优先 |
| 延迟 (ms) | 测量 | 测量 | 测量 |
| 吞吐量 (FPS) | 测量 | 测量 | 测量 |
| P50 / P95 | 测量 | 测量 | 测量 |
| 内存 (MB) | 测量 | 测量 | 测量 |

注意：GPU 路线需单独验证 Mali G720 计算后端（OpenCL / Vulkan），不能以"硬件支持 OpenCL"推断 ONNX GPU EP 可直接调用。

**交付物：** 异构计算性能对比报告（见 `experiments/phase2-benchmark/`）

---

### Phase 3 · AI 模型部署 🟡 部分完成

**目标：** 完整覆盖视觉分类、目标检测、大语言模型三类任务的部署与性能验证。

#### 3A · MobileNetV2 视觉分类（依赖 Phase 1）
- INT8 ONNX 模型，NPU 推理验证
- 模型转换在 x86_64 PC 上使用 NeuralONE SDK 完成

#### 3B · YOLO 目标检测
- 参考 Radxa 官方多路 YOLO 视频识别项目
- 先用本地视频文件，再接 USB 摄像头
- 验证 NPU 推理、测试多路并行 FPS

#### 3C · 本地 LLM 推理

**路线 1：llama.cpp（GGUF 格式，CPU 路线）**

| 模型 | 用途 |
|---|---|
| Qwen2.5-1.5B-Instruct | 快速验证，低内存 |
| Qwen2.5-3B-Instruct | 日常实验 |
| Qwen2.5-7B-Instruct | 主力实验 |
| 14B 量化模型 | 内存压力测试 |

**路线 2：Ollama（管理层，对接 Open WebUI）**

**路线 3：FlagOS + vLLM ✅ 已跑通**
- 模型：MiniCPM-V 4.7（1B 多模态，BF16 视觉 + W8A8 语言）
- 软件栈：vLLM 0.24 + vLLM-plugin-FL + FlagGems + FlagTree-CPU (MLIR/LLVM)
- 执行单元：8× A720 Neon SDOT
- 结果：PP512 589 tok/s（+223% vs Minimal），TG128 21.9 tok/s（+87%）
- 记录：见 `experiments/phase3-model-deploy/llm-flagos/`

注意：LLM 能否运行（内存 + 推理引擎）与能否全图在 NPU 执行（NeuralONE 算子支持）是两个独立问题，实验报告须分别记录。

**交付物：** 各模型推理验证记录 + 性能数据

---

### Phase 4 · AI 推理 API 服务 ⏳

**目标：** 构建统一的 FastAPI 推理服务，提供 OpenAI 兼容接口，支持 LLM 和视觉模型路由。

**调用架构：**
```
Open WebUI（用户交互）
  └─ FastAPI Gateway（API 路由 / 鉴权 / 日志）
       ├─ LLM Backend（llama.cpp / Ollama / FlagOS）
       ├─ Vision Backend（YOLO + NPU）
       └─ Embedding Backend（CPU / NPU）
```

**关键设计原则：** 推理后端可替换，前端接口不变。

**交付物：** FastAPI 服务代码 + Docker Compose 配置（见 `experiments/phase4-api-service/`）

---

### Phase 5 · 文档 RAG 知识库 ⏳

**目标：** 实现本地文档检索与知识问答，对比 ChromaDB 和 Qdrant 两种向量库方案。

**方案 A：ChromaDB（轻量本地）**
- 轻量嵌入式，零依赖部署
- 适合单机实验和快速验证
- 评估点：索引速度、检索延迟、内存占用

**方案 B：Qdrant（生产级）**
- Docker 部署，支持 gRPC + REST
- 适合多用户、高并发场景
- 评估点：性能对比 ChromaDB、向量过滤能力

**通用流程：** 文档解析 → Embedding（本地模型）→ 向量入库 → 语义检索 → LLM 问答

**交付物：** 两套 RAG 方案实现 + 检索性能对比（见 `experiments/phase5-rag/`）

---

### Phase 6 · Docker 部署与实验室运维 ⏳

**目标：** 将核心服务容器化，建立监控与自动启动机制。

**部署服务优先级：**

| 服务 | 软件 | 优先级 |
|---|---|---|
| AI Web UI | Open WebUI | 高 |
| 推理 API | FastAPI | 高 |
| 向量数据库 | Qdrant | 中 |
| 监控 | Prometheus + Grafana | 中 |
| 工作流 | n8n | 低 |
| 数据库 | PostgreSQL | 低 |

**目录结构：**
```
/opt/edge-ai-lab/services/
├── open-webui/compose.yaml
├── api/Dockerfile + app/main.py
├── qdrant/compose.yaml
└── monitoring/compose.yaml
```

**交付物：** Docker Compose 配置 + 运维脚本（见 `experiments/phase6-docker-ops/`）

---

## 技术风险

| 风险 | 应对策略 |
|---|---|
| NPU 算子不完整，模型大量回退 CPU | 从 MobileNetV2 等小模型开始，逐步验证算子覆盖率 |
| Docker 无法传递 NPU 设备节点 | 先宿主机验证，再研究 `/dev/aipu` 映射与运行时库挂载 |
| Mali GPU 计算后端不完整 | 单独验证 OpenCL 可用性，不假设 GPU 推理开箱即用 |
| NeuralONE SDK 仅支持 x86_64 转换 | PC 转换 + 板端推理的分工流程，转换结果复制到板端 |
| 长时间推理过热 | 记录温度与频率，验证持续负载下的稳定性 |
| 7B/14B 模型内存不足 | 从 INT4/INT8 量化版本开始，控制并发 |
| 升级内核破坏驱动 | 暂时固定当前内核，先确认驱动组合稳定 |

---

## 实验规范

每个实验记录须包含：

1. **实验目标** — 要解决什么问题
2. **环境快照** — OS 版本、驱动版本、相关软件版本
3. **模型信息** — 名称、来源、精度格式、转换方式
4. **执行命令** — 完整可复现的命令，每个参数说明用途
5. **原始数据** — 保存完整输出，不只保留"好看的结果"
6. **验收标准** — 明确定义"实验通过"的条件
7. **结论与问题** — 实验发现和遗留问题

**三条核心原则：**
- 软件安装成功 ≠ 硬件加速成功
- 推理结果正确 ≠ NPU 实际参与计算
- 上一步未验证通过，不进入下一步

---

## GitHub 仓库结构

```
cix-p1-edge-ai-lab/
├── README.md
├── PROJECT_PLAN.md           # 本文件
├── PROGRESS.md               # 进度记录
├── CLAUDE.md                 # Claude Code 工作指南
├── docs/
│   ├── hardware.md
│   ├── npu-setup.md
│   ├── model-deployment.md
│   └── benchmark-methodology.md
├── experiments/
│   ├── phase0-env-check/
│   ├── phase1-npu-bringup/
│   ├── phase2-benchmark/
│   ├── phase3-model-deploy/
│   │   └── llm-flagos/       # ✅ 已完成
│   ├── phase4-api-service/
│   ├── phase5-rag/
│   └── phase6-docker-ops/
├── scripts/
│   ├── system-info.sh
│   ├── collect-metrics.sh
│   └── run-benchmark.sh
└── results/
    ├── raw/
    └── reports/
```

