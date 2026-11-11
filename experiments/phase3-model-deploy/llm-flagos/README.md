# Phase 3C · FlagOS + vLLM LLM 推理实验 ✅

## 实验目标

在 Radxa Orion O6N（CIX P1）上验证 FlagOS 软件栈对 MiniCPM-V 4.7 多模态大模型的推理加速效果，对比 Stock / Minimal / FlagOS 三种引擎路线的吞吐量差异。

## 实验状态：已完成

**完成时间：** 2026-11

## 环境快照

### 软件版本（锁定）

| 组件 | 版本 |
|---|---|
| vLLM | 0.24.0+CPU |
| PyTorch | 2.11.0 |
| Transformers | 5.15.1 |
| vLLM-plugin-FL | commit f0de70233590 |
| FlagGems | commit b5f5e110c7c2 |
| FlagTree-CPU | commit 36133df9540c |
| Python | 3.11 |

### 模型

- **MiniCPM-V 4.7**（1B 多模态）
- 视觉模块：BF16（保留图片理解与 OCR 能力）
- 语言骨干：W8A8（INT8 权重 + 动态非对称逐 Token 激活）
- 上下文长度：4096 token
- CPU KV-cache：256 MiB

### 推理架构

```
客户端（文字/图片请求）
  └─ vLLM 0.24（调度、采样、内存管理）
       └─ vLLM-plugin-FL（模型注册与量化接入）
            └─ FlagGems（底层算子）
                 └─ FlagTree-CPU（MLIR/LLVM 编译）
                      └─ CIX P1 · 8× A720 · Neon SDOT
```

## 三路对比设计

| 路线 | 说明 | 预期 |
|---|---|---|
| Stock | 官方 vLLM 0.24 未注册 MiniCPM-V 4.7 | 抛出 `ValidationError` |
| Minimal | 打注册补丁，但走 PyTorch 通用算子 | 可运行但未优化 |
| FlagOS | 拦截替换算子，Neon 加速 W8 GEMM | 性能最优 |

## 执行流程

完整执行命令见 `inference-flagos.md`（源实验手册）。核心步骤：

1. **主机环境校验**（Step 01–02）：校验 wheel 哈希与镜像 ID，输出 `PRELOADED_CHECK_PASS` / `PREBUILT_CHECK_PASS`
2. **开发容器构建**（Step 03–06）：离线安装预编译 wheel，执行正确性检查，输出 `SOURCE_VALIDATION_PASS`
3. **HTTP 服务启动**（Step 07–08）：`vllm serve`，等待 `READY`，发送文字与图片请求
4. **性能跑分**（Step 09–11）：三路 `run_bench.sh`，`--repeats 5`（1 预热 + 5 测量取中位数），输出 `COMPARISON_PASS`
5. **边界检查**（Step 12）：17 项边界条件，输出 `BOUNDARIES 17 / 17`

## 测试结果

### 吞吐量对比

| 路线 | PP512（tok/s） | TG128（tok/s） |
|---|---|---|
| Stock | — | — |
| Minimal | ~183 | ~11.7 |
| FlagOS | ~589–600 | ~21.9 |
| **相对 Minimal 提升** | **+223.76%** | **+87.12%** |

指标说明：
- **PP512**：预填充，处理 512 输入 token，测算力（compute-bound）
- **TG128**：解码，生成 128 输出 token，测访存（memory-bound）

### 优化原理

**PP512 提升原因：**
- GDN 按 head 并行并融合状态更新，核心热点从 1823ms 降至 246ms
- W8 复用预打包权重，`aten::copy_` 调用从 12263 次降至 5338 次

**TG128 提升原因：**
- W8A8 输出头优化：+11.78%
- Decode 阶段线程从 8 调整为 4（减少切换开销）：+19.57%

### 推理正确性

- ✅ 文字推理：`http_request.py` 返回正确答案
- ✅ 图片推理：正确识别主要颜色（红色）
- ✅ Token usage 非零

## 验收标准（全部通过）

- [x] `PRELOADED_CHECK_PASS`
- [x] `PREBUILT_CHECK_PASS`
- [x] `SOURCE_VALIDATION_PASS`
- [x] 文字推理结果正确
- [x] 图片推理结果正确
- [x] `COMPARISON_PASS`
- [x] `BOUNDARIES 17 / 17`

## 结论

FlagOS 通过算子级优化与 ARM Neon 指令集加速，在 CIX P1 的 CPU 上将 MiniCPM-V 4.7 推理性能提升约 2–3 倍。该实验证明：未经优化的 PyTorch 通用算子无法发挥 ARM64 平台潜力，需要底层编译与算子替换才能达到实用水平。

## 关键文件

- `inference-flagos.md` — 完整实验手册（命令与原理解析）
- 证据目录（板端）：`/work/evidence/workshop-run01/`
  - `stock.json` / `minimal.json` / `flags.json` — 原始跑分数据
  - `comparison.json` — 对比结果
- HTTP 证据：`/work/evidence/http-run01/`
  - `text-response.json` / `image-response.json` / `server-full.log`

## 异常重试规范

- **材料安装失败**：不可直接覆盖，必须 `mv /work/envs /work/evidence/pip-retry-<时间戳>` 备份后重试
- **端口冲突**：先查 `server.log` 与 `docker logs`，确认属于本项目的旧服务后 `docker stop`，重试须更换容器名与输出目录
- **测量中断**：拒绝同名重试（防覆盖），重试输出文件必须改名
