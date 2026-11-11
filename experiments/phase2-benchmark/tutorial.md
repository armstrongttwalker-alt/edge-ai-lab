# Phase 2 · CPU / GPU / NPU 基准测试 小白教程

> 本阶段用统一的模型和输入，量化三种算力的推理性能差异。目前已经完成首批 NPU vs CPU 微基准测试，真实视觉模型（MobileNetV2）测试需要先完成模型下载或转换。
>
> **实验状态：进行中（微基准已完成）**

---

## 一、实验目标

在相同模型、相同输入、相同测试流程下，对比 CPU / GPU / NPU 三种算力的推理延迟、吞吐量和资源占用，形成可重复的性能报告。

核心原则：**公平比较**。固定模型、固定 batch size、固定输入尺寸、固定线程数，只改变计算后端。

---

## 二、背景知识

### 2.1 为什么不能直接比较

三种算力的架构差异决定了它们适合的任务类型不同：

| 算力 | 架构特点 | 适合的任务 |
|---|---|---|
| CPU（A720） | 通用，单线程强，启动开销低 | 小模型、控制逻辑多的任务 |
| GPU（Mali G720） | 并行度高，带宽大 | 图形渲染、大批量并行计算 |
| NPU（Zhouyi V3） | 专为矩阵运算优化 | 中大型神经网络推理 |

**关键认识：** 小模型 NPU 反而可能比 CPU 慢——因为 NPU 的任务调度和编译开销大于计算收益。只有当模型足够大（卷积层多、矩阵乘法密集），NPU 的并行计算优势才能体现。

### 2.2 关键指标

| 指标 | 含义 | 单位 |
|---|---|---|
| **Latency** | 单次推理耗时 | ms |
| **P50** | 50% 请求的延迟中位数（典型值） | ms |
| **P95** | 95% 请求的延迟（尾部延迟） | ms |
| **Throughput** | 单位时间处理量 | FPS / tok/s |
| **Memory** | 推理时内存占用峰值 | MB |

P50 反映"大部分时候多快"，P95 反映"最坏情况多快"，两者都要测。

### 2.3 测试方法论

为保证结果可重复：

1. **预热**：先跑 10 次不计入统计，让缓存和频率稳定
2. **重复**：跑 200 次取统计值，消除偶然波动
3. **固定 CPU 频率**：如不固定，记录测试时的频率范围
4. **独占运行**：停掉其他占用 CPU 的服务
5. **同一输入**：所有测试用同一份随机种子生成的输入

---

## 三、测试矩阵

### 3.1 模型选择

| 模型 | 用途 | 输入 | 预期体现 |
|---|---|---|---|
| sigmoid.onnx | 微基准（3×4×5 张量） | 极小张量 | NPU 启动开销 |
| mul_1.onnx | 微基准（3×2 矩阵乘） | 极小张量 | NPU 启动开销 |
| MobileNetV2 INT8 | 视觉分类 | 224×224×3 | NPU 真正优势 |
| YOLOv8n | 目标检测 | 640×640×3 | NPU 大模型优势 |

### 3.2 精度 × 后端

| 精度 | CPU EP | Mali GPU | Zhouyi NPU |
|---|---|---|---|
| FP32 | ✅ | 视驱动 | — |
| FP16 | ✅ | 视驱动 | 视算子 |
| INT8 | ✅ | 视后端 | ✅ 优先 |

---

## 四、已完成的微基准测试（2026-11-11）

### 测试环境

- 设备：Radxa Orion O6N（CIX P1）
- CPU：8× Cortex-A720 @2.6GHz + 4× Cortex-A520 @1.8GHz
- NPU：Zhouyi V3（1 Cluster × 3 Core）
- Python：3.11，onnxruntime-zhouyi 1.20.0
- 重复：200 次（10 次预热 + 200 次测量）

### 测试结果

| 模型 | 后端 | avg (ms) | P50 (ms) | P95 (ms) |
|---|---|---|---|---|
| sigmoid (3×4×5) | ZhouyiEP | 0.321 | 0.3127 | 0.3649 |
| sigmoid (3×4×5) | CPU | 0.1403 | 0.1314 | 0.1829 |
| mul_1 (3×2) | ZhouyiEP | 0.3656 | 0.3153 | 0.3736 |
| mul_1 (3×2) | CPU | 0.17 | 0.1558 | 0.2154 |

### 结果解读

**结论：在极小模型上，NPU 比 CPU 慢约 2 倍。**

这不是 NPU 的问题，而是符合预期：
- sigmoid 模型只有 1 个节点、60 个元素的张量运算
- NPU 的任务调度、kernel 编译、数据搬运开销（约 0.3ms）远大于这点计算量
- CPU 直接在本地 cache 里就算完了，没有跨设备搬运

**这个结论非常重要**：它说明"NPU 一定比 CPU 快"是错误的。NPU 只在计算密集的模型（如 MobileNetV2 的 50+ 卷积层、YOLO 的检测网络）上才有优势。Phase 3 部署 MobileNetV2 后，会做真正的对比测试。

### 原始数据

完整 JSON 数据：`experiments/phase2-benchmark/results/micro_npu_cpu_2026-11-11.json`

---

## 五、NPU vs CPU 微基准测试脚本

```python
import onnxruntime as ort
import numpy as np
import os, time, json

os.chdir('/opt/edge-ai-lab/npu-workspace')
DATASETS = "/opt/edge-ai-lab/venv/lib/python3.11/site-packages/onnxruntime/datasets"

results = {}
for name, path, shape in [
    ("sigmoid", f"{DATASETS}/sigmoid.onnx", [3,4,5]),
    ("mul_1",   f"{DATASETS}/mul_1.onnx",   [3,2]),
]:
    for prov_name, providers in [
        ("ZhouyiEP", ["ZhouyiExecutionProvider","CPUExecutionProvider"]),
        ("CPU",      ["CPUExecutionProvider"]),
    ]:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(path, sess_options=opts, providers=providers)
        inp = sess.get_inputs()[0]
        dummy = np.random.rand(*shape).astype(np.float32)
        # 预热
        for _ in range(10): sess.run(None, {inp.name: dummy})
        # 测量
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            sess.run(None, {inp.name: dummy})
            times.append((time.perf_counter()-t0)*1000)
        times.sort()
        p50 = times[len(times)//2]
        p95 = times[int(len(times)*0.95)]
        avg = sum(times)/len(times)
        key = f"{name}_{prov_name}"
        results[key] = {"avg_ms": round(avg,4), "p50_ms": round(p50,4), "p95_ms": round(p95,4)}
        print(f"{key}: avg={avg:.4f}ms p50={p50:.4f}ms p95={p95:.4f}ms")

with open('/opt/edge-ai-lab/benchmarks/phase2_npu_cpu_micro.json', 'w') as f:
    json.dump(results, f, indent=2)
```

运行：

```bash
source /opt/edge-ai-lab/venv/bin/activate
export LD_LIBRARY_PATH=/usr/share/cix/lib:/usr/share/cix/lib/onnxruntime:$LD_LIBRARY_PATH
python3 /opt/edge-ai-lab/experiments/npu/bench_micro.py
```

---

## 六、待完成项

### 6.1 MobileNetV2 INT8 对比测试

依赖 Phase 3A 完成 MobileNetV2 模型准备（下载或 NeuralONE SDK 转换）。完成后用同一模型在 CPU 和 NPU 上跑，对比 224×224 输入的延迟。

### 6.2 GPU 路线验证

需要先确认 Mali G720 的 OpenCL/Vulkan 计算后端可用性：

```bash
# 检查 OpenCL
clinfo 2>/dev/null || echo "clinfo not installed"
ls /dev/mali* 2>/dev/null || echo "no mali device"
```

注意：GPU 推理后端依赖厂商驱动，不能假设"OpenCL 支持就等于 ONNX GPU EP 可用"。

### 6.3 功耗测量

需要外接功率计或读取板载功耗传感器：

```bash
# 查看是否有功耗传感器
cat /sys/class/power_supply/*/power_now 2>/dev/null
find /sys -name "*power*" 2>/dev/null | grep -i npu | head
```

---

## 七、验收标准

完整 Phase 2 通过需要：

- [x] NPU vs CPU 微基准（已完成，2026-11-11）
- [ ] MobileNetV2 INT8 CPU vs NPU 延迟对比
- [ ] P50 / P95 延迟记录
- [ ] GPU 后端可用性确认（如可用则加入对比）
- [ ] 完整测试报告写入 `results/` 目录

---

*本教程对应实验：Phase 2，进行中。最后更新：2026-11-11。*
