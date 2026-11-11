# Phase 3A · MobileNetV2 NPU 推理 小白教程

> 本教程面向第一次在 NPU 上部署视觉分类模型的读者。你将理解：为什么需要用专门工具转换模型、INT8 量化是怎么回事，以及如何验证模型真正跑在 NPU 上。
>
> **实验状态：⏳ 待执行（依赖 NeuralONE SDK 模型转换）**

---

## 一、实验目标

将 MobileNetV2 FP32 模型转换为 Zhouyi NPU 可用的 INT8 格式，部署到 Radxa Orion O6N，完成一次完整的视觉分类推理，并对比 CPU 与 NPU 的延迟差异。

---

## 二、背景知识

### 2.1 为什么需要模型转换？

ONNX 格式的模型不能直接在 NPU 上运行——就像 MP3 文件不能直接在磁带机上播放一样，需要先转换成 NPU 能理解的格式。

CIX P1 的 NPU 使用的编译工具链叫 **NeuralONE SDK**，核心工具是 `cixbuild`，它负责：
1. 检查模型算子兼容性
2. 进行 INT8 量化（降低精度，换取速度）
3. 编译成 NPU 的本地指令格式（`.bin`）

**重要限制：** NeuralONE SDK 中的 `cixbuild` 是 **x86_64** 格式，只能在 Linux PC 上运行，不能在 ARM64 板子上运行。这形成了"PC 转换 + 板端推理"的分工。

### 2.2 什么是 INT8 量化？

| 精度 | 存储一个数 | 计算速度 | 精度 |
|---|---|---|---|
| FP32 | 4 字节 | 基准 | 最高 |
| FP16 | 2 字节 | 约 2× | 轻微损失 |
| INT8 | 1 字节 | 约 4× | 有损（通常 <1% 精度下降） |

量化把原本用 32 位浮点数表示的权重，改用 8 位整数表示。好处是模型变小、计算加快；代价是精度略有损失（通常 Top-1 精度下降 <1%，可接受）。

### 2.3 MobileNetV2 是什么？

Google 专为移动设备设计的轻量图像分类模型，参数量约 3.4M，输入 224×224 RGB 图片，输出 1000 类概率分布（ImageNet 类别）。是验证 NPU 视觉推理能力的标准起点。

---

## 三、模型转换流程

### 重要发现：不需要 NeuralONE SDK

实验证明，可以直接在板子上用 `onnxruntime.quantization` 完成 FP32 → QDQ INT8 转换，不需要 x86_64 PC 和 cixbuild 工具链。

### 3.1 生成 FP32 模型（在 x86_64 机器上）

```python
import torch, torchvision

model = torchvision.models.mobilenet_v2(weights=None)
model.eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, 'mobilenetv2_fp32.onnx',
    input_names=['input'], output_names=['output'],
    opset_version=13, dynamo=False
)
```

### 3.2 传输到板子

```bash
scp mobilenetv2_fp32.onnx baai@192.168.15.195:/opt/edge-ai-lab/models/mobilenetv2/
```

### 3.3 在板子上量化为 QDQ INT8（无需 cixbuild）

```python
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
import numpy as np

class RandomCalibData(CalibrationDataReader):
    def __init__(self, n=20):
        self.data = iter([
            {"input": np.random.rand(1,3,224,224).astype(np.float32)}
            for _ in range(n)
        ])
    def get_next(self):
        return next(self.data, None)

quantize_static(
    'mobilenetv2_fp32.onnx',
    'mobilenetv2_qdq.onnx',
    RandomCalibData(20),
    quant_format=QuantFormat.QDQ,
    per_channel=False,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
)
```

注意：onnx 包会把 numpy 升到 2.x，需要降回 1.x：

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ "numpy>=1.21.6,<2"
```

---

## 四、板端推理脚本

```python
import onnxruntime as ort
import numpy as np
import os, time

os.chdir('/opt/edge-ai-lab/npu-workspace')

MODEL = '/opt/edge-ai-lab/models/mobilenetv2/mobilenetv2_qdq.onnx'
sess = ort.InferenceSession(
    MODEL,
    providers=["ZhouyiExecutionProvider", "CPUExecutionProvider"]
)
inp = sess.get_inputs()[0]
dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)

for _ in range(5): sess.run(None, {inp.name: dummy})  # warmup
times = []
for _ in range(50):
    t0 = time.perf_counter()
    out = sess.run(None, {inp.name: dummy})
    times.append((time.perf_counter()-t0)*1000)
times.sort()
print(f"NPU avg={sum(times)/len(times):.2f}ms P50={times[25]:.2f}ms P95={times[47]:.2f}ms")
print(f"Top-1: {np.argmax(out[0])}")
```

运行：

```bash
source /opt/edge-ai-lab/venv/bin/activate
export LD_LIBRARY_PATH=/usr/share/cix/lib:/usr/share/cix/lib/onnxruntime:$LD_LIBRARY_PATH
cd /opt/edge-ai-lab/npu-workspace
python3 infer_mobilenetv2.py
```

---

## 五、实测结果（2026-11-11）

| Provider | avg (ms) | P50 (ms) | P95 (ms) |
|---|---|---|---|
| ZhouyiEP（NPU） | **1.48** | 1.49 | 1.61 |
| CPU | 12.68 | 12.68 | 12.79 |
| **NPU 加速比** | **8.6×** | | |

节点分配（verbose 日志）：
```
Number of partitions supported by ZHOUYI: 1
number of nodes in the graph: 313
number of nodes supported by ZHOUYI: 313
All nodes placed on [ZhouyiExecutionProvider]. Number of nodes: 1
```

**全部 313 个节点分配给 NPU，零 CPU 回退。**

这是整个实验最重要的里程碑：在真实的 224×224 视觉模型上，NPU 相对 CPU 实现了 **8.6× 加速**，与之前小模型（sigmoid/mul_1）NPU 反而慢的情况形成鲜明对比，证明 NPU 优势在中大型模型上才能真正体现。

原始数据：`results/bench_2026-11-11.md`

---

*本教程对应实验：Phase 3A，待执行。最后更新：2026-11-11。*
