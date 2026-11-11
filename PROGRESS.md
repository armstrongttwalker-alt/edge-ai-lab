# 实验进度记录

按时间线记录每次实验的操作、结果与遗留问题。每个阶段完成后更新状态。

---

## 当前状态总览

| 阶段 | 状态 | 最近更新 |
|---|---|---|
| Phase 0 · 基础环境检查 | ✅ 完成 | 2026-11 |
| Phase 1 · NPU Bring-up | 🔄 Step 4/7，阻塞 | 2026-11 |
| Phase 2 · 基准测试 | ⏳ 未开始 | — |
| Phase 3 · 模型部署 | 🟡 LLM 子实验已完成 | 2026-11 |
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

## Phase 1 · NPU Bring-up 🔄

### Step 1–3 · 已完成

驱动、设备节点、NOE 运行库验证均通过。见 Phase 0 结果。

### Step 4 · 安装 NumPy + 厂商 ONNX Runtime ⚠️ 阻塞

**问题：** `pip install numpy` 因网络超时失败。

**待尝试解决方案（按优先级）：**

```bash
# 方案 1：使用阿里云镜像
pip install -i https://mirrors.aliyun.com/pypi/simple/ "numpy>=1.21.6,<2"

# 方案 2：使用清华镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ "numpy>=1.21.6,<2"

# 方案 3：在 Windows 机器下载 ARM64 wheel 后 scp 传入
# 下载地址：https://pypi.org/project/numpy/#files
# 文件格式：numpy-*-cp311-cp311-linux_aarch64.whl
scp numpy-*.whl baai@<板端IP>:/opt/edge-ai-lab/
pip install /opt/edge-ai-lab/numpy-*.whl
```

NumPy 安装成功后，按以下顺序安装厂商 wheel：

```bash
source /opt/edge-ai-lab/venv/bin/activate
pip install libnoe-2.0.0-py3-none-manylinux2014_aarch64.whl
pip install NOE_Engine-2.0.0-py3-none-manylinux2014_aarch64.whl
pip install ZhouyiOperators_x2-25.4.23-py3-none-any.whl
pip install onnxruntime_zhouyi-1.20.0-cp311-cp311-linux_aarch64.whl
```

### Step 5–7 · 待执行

**Step 5 验证命令：**
```python
import onnxruntime as ort
print(ort.get_available_providers())
# 预期包含：CIXExecutionProvider
```

**Step 6–7 推理验证：**
- 模型：MobileNetV2 INT8（ONNX 格式，224×224 输入）
- 须确认 NPU Provider 分配情况，不能以结果正确推断 NPU 参与计算

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

## 遗留问题与待研究项

| 问题 | 相关阶段 | 优先级 |
|---|---|---|
| NumPy pip 安装超时，需切换镜像源或离线安装 | Phase 1 | 高 |
| Mali G720 OpenCL 计算后端可用性未验证 | Phase 2 | 中 |
| NeuralONE SDK 模型转换流程（x86_64 PC）未实操 | Phase 1/3 | 中 |
| Docker 容器内 NPU 设备映射方案未研究 | Phase 6 | 低 |
| 7B 量化 LLM 在 llama.cpp 下的 ARM64 性能未测试 | Phase 3 | 低 |
