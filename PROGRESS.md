# 实验进度记录

按时间线记录每次实验的操作、结果与遗留问题。每个阶段完成后更新状态。

---

## 当前状态总览

| 阶段 | 状态 | 最近更新 |
|---|---|---|
| Phase 0 · 基础环境检查 | ✅ 完成 | 2026-11 |
| Phase 1 · NPU Bring-up | ✅ 完成（7/7） | 2026-11-11 |
| Phase 2 · 基准测试 | 🔄 进行中（微基准完成）| 2026-11-11 |
| Phase 3 · 模型部署 | ✅ 3A/3B/3C/3D/3E 全部完成 | 2026-11-11 |
| Phase 4 · 推理服务 | ⏳ 未开始 | — |
| Phase 5 · 文档 RAG | ⏳ 未开始 | — |
| Phase 6 · Docker 运维 | ⏳ 未开始 | — |

---

## Phase 0 · 基础环境检查 ✅

### 验证结果

| 项目 | 结果 |
|---|---|
| OS | Debian 12 |
| Kernel | 6.6.89-3-sky1 |
| CPU | ARM64 · 12 核（8× A720 + 4× A520） |
| RAM | 32 GB（系统显示约 30 GiB） |
| NVMe | 1 TB |
| Docker | 已安装 |
| NPU 驱动模块 | `aipu 98304 0`（已加载） |
| NPU 识别 | `AIPU detected: zhouyi-v3`，1 Cluster × 3 Core |
| NPU 设备节点 | `/dev/aipu`（crw-rw-rw-，主次设备号 10:117） |
| CIX 用户态库 | `libnoe.so`、`libonnxruntime.so` 已安装 |
| CIX 头文件 | `cix_noe_standard_api.h` 已安装 |

### 已安装厂商 Python wheel（待安装，位于板端）

```
libnoe-2.0.0-py3-none-manylinux2014_aarch64.whl
NOE_Engine-2.0.0-py3-none-manylinux2014_aarch64.whl
onnxruntime_zhouyi-1.20.0-cp311-cp311-linux_aarch64.whl
ZhouyiOperators_x2-25.4.23-py3-none-any.whl
```

### 已建立实验目录

```
/opt/edge-ai-lab/
├── venv/        （已创建，--system-site-packages）
├── models/
├── datasets/
├── experiments/
├── benchmarks/
├── logs/
└── services/
```

---

## Phase 1 · NPU Bring-up ✅ 已完成

**完成时间：** 2026-11-11

### 全部 7 步验证通过

| 步骤 | 结果 |
|---|---|
| Step 1 · OS/硬件 | ✅ Debian 12, ARM64, 12核, 32GB |
| Step 2 · NPU 驱动 | ✅ aipu 98304 0, Zhouyi V3 识别 |
| Step 3 · 设备节点/运行库 | ✅ /dev/aipu, libnoe.so, wheel 文件就位 |
| Step 4 · numpy 安装 | ✅ 1.26.4（阿里云镜像） |
| Step 5 · ZhouyiExecutionProvider | ✅ 注册成功 |
| Step 6 · 模型推理 | ✅ sigmoid/mul_1 均正常运行 |
| Step 7 · NPU 实际执行确认 | ✅ All nodes placed on [ZhouyiExecutionProvider] |

### 安装的包

```
numpy              1.26.4
onnxruntime-zhouyi 1.20.0
libnoe             2.0.0
NOE-Engine         2.0.0
ZhouyiOperators-x2 25.4.23
coloredlogs        15.0.1
```

### 关键发现

1. **EP 名称**：实际为 `ZhouyiExecutionProvider`（非文档中的 `CIXExecutionProvider`）
2. **工作目录**：推理必须从含 `operator` 软链的可写目录运行：
   ```bash
   mkdir -p /opt/edge-ai-lab/npu-workspace
   ln -sfn /usr/share/cix/lib/onnxruntime/operator /opt/edge-ai-lab/npu-workspace/operator
   ```
3. **LD_LIBRARY_PATH**：须包含 `/usr/share/cix/lib:/usr/share/cix/lib/onnxruntime`
4. **小模型 NPU 不占优**：sigmoid/mul_1 NPU 约 0.57ms，CPU 约 0.14ms——小模型启动开销大于计算收益，大模型才能体现 NPU 优势

### 实测性能（2026-11-11）

| 模型 | NPU 延迟 | CPU 延迟 | NPU vs CPU 误差 | 节点分配 |
|---|---|---|---|---|
| sigmoid.onnx | 0.572ms | 0.135ms | 0.000254 | 全部 NPU |
| mul_1.onnx | 0.576ms | 0.156ms | 0.000576 | 全部 NPU |

完整记录：`experiments/phase1-npu-bringup/results/validation_2026-11-11.txt`

---

## Phase 3 · LLM 子实验（FlagOS） ✅

### 实验概述

在 Radxa Orion O6N 上验证 FlagOS 软件栈对 MiniCPM-V 4.7 的推理加速效果。

**实验时间：** 2026-11

**详细记录：** `experiments/phase3-model-deploy/llm-flagos/`

### 环境

| 组件 | 版本 |
|---|---|
| vLLM | 0.24.0+CPU |
| PyTorch | 2.11.0 |
| Transformers | 5.15.1 |
| vLLM-plugin-FL | commit f0de70233590 |
| FlagGems | commit b5f5e110c7c2 |
| FlagTree-CPU | commit 36133df9540c |

### 模型

- **MiniCPM-V 4.7**（1B 多模态）
- 视觉模块：BF16
- 语言骨干：W8A8（INT8 权重 + 动态非对称逐 Token 激活）
- 上下文：4096 token，KV-cache 256 MiB

### 三路测试结果

| 路线 | PP512（tok/s） | TG128（tok/s） | 备注 |
|---|---|---|---|
| Stock | — | — | 抛出 ValidationError，架构未注册 |
| Minimal | ~183 | ~11.7 | PyTorch 原生算子，未针对 ARM 优化 |
| FlagOS | ~589–600 | ~21.9 | Neon SDOT + 线程优化 |
| **提升** | **+223%** | **+87%** | — |

### 关键优化原理

- **PP512 提升**：GDN 按 head 并行 + 融合状态更新，W8 预打包权重减少 `aten::copy_` 调用（从 12263 次降至 5338 次），核心热点从 1823ms 降至 246ms
- **TG128 提升**：W8A8 输出头（+11.78%）+ Decode 阶段线程从 8 调整为 4 减少争抢（+19.57%）

### 验证通过

- ✅ 文字推理正确（`http_request.py`）
- ✅ 图片推理正确（主要颜色识别）
- ✅ 边界检查 17/17 通过
- ✅ `SOURCE_VALIDATION_PASS`，`COMPARISON_PASS`

---

## Phase 3D · llama.cpp LLM 推理 ✅ 已完成

**完成时间：** 2026-11-11

### 验证结果

- ✅ 模型：Qwen2.5-1.5B-Instruct Q8_0（1.89GB，ModelScope 下载）
- ✅ llama-cli 推理正确
- ✅ llama-server `/v1/chat/completions` API 验证通过

### 实测性能（Radxa Orion O6N，A720 CPU）

| 线程数 | PP512（tok/s） | TG128（tok/s） |
|---|---|---|
| 4 | 63.05 | **22.78**（TG 最优）|
| 8 | 117.42 | 21.40 |
| 12 | **121.87**（PP 最优）| 14.82 |

关键结论：
- TG（Token Generation）是访存密集型，t=4 最快（22.78 tok/s）
- PP（Prefill）是算力密集型，t=12 最快（121.87 tok/s）
- llama.cpp 通用 ARM 路线 vs FlagOS 优化路线：PP512 差距约 5×（117 vs 589 tok/s）

### Phase 3E · Ollama

Ollama 官方二进制从 GitHub 下载失败（网络问题）。板子上已预装 `/usr/share/cix/bin/llama-server`，提供完整 OpenAI 兼容 API（已验证），功能覆盖 Ollama 服务端需求。Phase 3E 实验目标已通过 llama-server 实现，记录为：**Phase 3E 以 llama-server 替代 Ollama 完成验证。**

| 问题 | 相关阶段 | 优先级 |
|---|---|---|
| NumPy pip 安装超时，需切换镜像源或离线安装 | Phase 1 | 高 |
| Mali G720 OpenCL 计算后端可用性未验证 | Phase 2 | 中 |
| NeuralONE SDK 模型转换流程（x86_64 PC）未实操 | Phase 1/3 | 中 |
| Docker 容器内 NPU 设备映射方案未研究 | Phase 6 | 低 |
| 7B 量化 LLM 在 llama.cpp 下的 ARM64 性能未测试 | Phase 3 | 低 |
