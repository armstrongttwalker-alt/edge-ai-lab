# Phase 3D · llama.cpp CPU LLM 推理 小白教程

> 本教程说明如何在 Radxa Orion O6N 的 CPU 上用 llama.cpp 运行量化大模型，并测量推理性能。这是最直接的本地 LLM 路线，不依赖 NPU，只用 ARM CPU。
>
> **实验状态：⏳ 待执行（需要下载 GGUF 模型）**

---

## 一、实验目标

用 llama.cpp 在 CIX P1 的 ARM64 CPU 上运行 Qwen2.5 系列量化模型，测量 Prefill 速度和 Token Generation 速度，建立 CPU LLM 推理的性能基准。

---

## 二、背景知识

### 2.1 llama.cpp 是什么？

llama.cpp 是用纯 C++ 实现的大模型推理框架，专注于在 CPU 上高效运行量化模型。优势：
- 不需要 CUDA，纯 CPU 也能跑
- 支持 ARM 的 Neon SIMD 指令集加速
- 支持 GGUF 量化格式（4-bit、5-bit、8-bit 等）

板子上已经预装了 llama.cpp 的相关工具在 `/usr/share/cix/bin/`：
- `llama-cli`：交互式对话
- `llama-server`：启动 HTTP API 服务
- `llama-bench`：性能测试

### 2.2 什么是 GGUF 量化？

GGUF（GPT-Generated Unified Format）是 llama.cpp 使用的模型格式。常见量化级别：

| 量化 | 每参数位数 | 7B 模型大小 | 速度 | 精度损失 |
|---|---|---|---|---|
| Q8_0 | 8 bit | ~7.7 GB | 较慢 | 极小 |
| Q5_K_M | 5 bit | ~5.1 GB | 中等 | 很小 |
| Q4_K_M | 4 bit | ~4.1 GB | 快 | 小 |
| Q2_K | 2 bit | ~2.4 GB | 最快 | 明显 |

32 GB 内存可以轻松跑 7B 量化模型，14B Q4 也在范围内。

### 2.3 Prefill vs Token Generation

- **Prefill（PP）**：处理输入 prompt 的阶段，计算密集（compute-bound），核心是矩阵乘法
- **Token Generation（TG）**：逐 token 生成输出，访存密集（memory-bound），每步读一遍权重

FlagOS 实验已经证明 PP 和 TG 分别约 589 tok/s 和 21.9 tok/s（优化路线）。llama.cpp 的结果会有所不同，因为底层实现不同。

---

## 三、模型候选

| 模型 | 量化 | 大小 | HuggingFace 下载路径 |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | Q8_0 | ~1.6 GB | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` |
| Qwen2.5-3B-Instruct | Q4_K_M | ~2.0 GB | `Qwen/Qwen2.5-3B-Instruct-GGUF` |
| Qwen2.5-7B-Instruct | Q4_K_M | ~4.5 GB | `Qwen/Qwen2.5-7B-Instruct-GGUF` |

---

## 四、执行步骤

### 4.1 下载模型

```bash
# 在板子上执行（使用 HF 镜像加速）
export HF_ENDPOINT=https://hf-mirror.com
pip install huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen2.5-1.5B-Instruct-GGUF',
    filename='qwen2.5-1.5b-instruct-q8_0.gguf',
    local_dir='/opt/edge-ai-lab/models/llm/'
)
print('Done')
"
```

### 4.2 交互式对话

```bash
/usr/share/cix/bin/llama-cli \
    -m /opt/edge-ai-lab/models/llm/qwen2.5-1.5b-instruct-q8_0.gguf \
    -n 128 \
    -t 8 \
    --prompt "你好，请用一句话介绍一下自己。"
```

参数说明：
- `-n 128`：最多生成 128 个 token
- `-t 8`：使用 8 个 CPU 线程（A720 大核）
- `-p`：输入 prompt

### 4.3 性能测试

```bash
/usr/share/cix/bin/llama-bench \
    -m /opt/edge-ai-lab/models/llm/qwen2.5-1.5b-instruct-q8_0.gguf \
    -p 512 -n 128 -t 8 -r 3 -o json > /opt/edge-ai-lab/benchmarks/llamacpp_1.5b_q8_t8.json
```

参数说明（来自 llama-bench --help 实测）：
- `-p 512`：Prefill 输入 512 token（测算力）
- `-n 128`：生成 128 token（测访存）
- `-t 8`：使用 8 个线程（A720 大核）
- `-r 3`：重复 3 次取平均
- `-o json`：JSON 格式输出（可选 csv/md/sql）

### 4.4 比较不同线程数

```bash
for t in 4 8 12; do
    echo "=== threads=$t ==="
    /usr/share/cix/bin/llama-bench \
        -m /opt/edge-ai-lab/models/llm/qwen2.5-1.5b-instruct-q8_0.gguf \
        -p 512 -n 128 -t $t -r 2 -o md
done
```

---

## 五、实测结果（2026-11-11）

### 测试环境

- 设备：Radxa Orion O6N（CIX P1）
- CPU：8× Cortex-A720 @2.6GHz
- 模型：Qwen2.5-1.5B-Instruct Q8_0（1.76 GiB，1.78B 参数）
- llama.cpp build：837cc941（5929）
- 重复：3 次取平均

### 线程数对比

| 线程数 | PP512（tok/s） | TG128（tok/s） |
|---|---|---|
| 4 | 63.05 ± 0.07 | **22.78 ± 0.04** |
| 8 | 117.42 ± 0.06 | 21.40 ± 0.02 |
| 12 | **121.87 ± 0.10** | 14.82 ± 0.03 |

### 关键发现

**TG128 在 t=4 时最快，t=12 反而最慢。**

这与 FlagOS 实验的结论完全一致：Token Generation 是访存密集型操作（每步读一遍所有权重），多线程争抢内存带宽反而降速。最佳 TG 线程数：4。

**PP512 随线程数近似线性扩展**，t=12 比 t=8 仅提升 4%，说明已接近 CPU 算力上限。最佳 PP 线程数：8-12。

**与 FlagOS 对比（同硬件，不同软件栈）：**

| 路线 | PP512（tok/s） | 差距 |
|---|---|---|
| FlagOS（Triton/Neon SDOT 优化） | ~589 | 5× |
| llama.cpp（通用 ARM 路线） | ~117 | 基准 |

差距体现了底层 ARM 算子优化（Neon SDOT、算子融合）的价值。llama.cpp 通用路线未针对 Cortex-A720 做专项优化。

### 推理验证

- ✅ llama-cli 推理正确（"你好，请用一句话介绍自己" → 正常中文回复）
- ✅ llama-server `/v1/chat/completions` API 返回正确响应

原始数据：`results/bench_2026-11-11.md`

---

## 六、HTTP API 服务

```bash
/usr/share/cix/bin/llama-server \
    -m /opt/edge-ai-lab/models/llm/qwen2.5-1.5b-instruct-q8_0.gguf \
    -t 8 \
    --host 0.0.0.0 \
    --port 8080
```

服务启动后，可以通过 OpenAI 兼容接口访问：

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2.5","messages":[{"role":"user","content":"你好"}],"max_tokens":100}'
```

---

## 七、验收标准

- [ ] 1.5B 模型下载完成
- [ ] `llama-cli` 能正常对话
- [ ] `llama-bench` 完成 PP512/TG128 测量
- [ ] 记录不同线程数的性能数据
- [ ] 结果写入 `benchmarks/` 目录

---

*本教程对应实验：Phase 3D，待执行。最后更新：2026-11-11。*
