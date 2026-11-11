# Phase 3B · YOLO 目标检测 小白教程

> 本教程说明如何在 Radxa Orion O6N 上运行 YOLO 目标检测模型，重点验证 NPU 推理能力，并参考 Radxa 官方多路视频 YOLO 项目做进阶测试。
>
> **实验状态：⏳ 待执行（依赖 NeuralONE SDK 模型转换）**

---

## 一、实验目标

将 YOLOv8n 转换为 Zhouyi NPU 可用的 INT8 格式，在 Radxa Orion O6N 上对图片和视频流执行目标检测推理，测量 FPS 和延迟，对比 CPU 与 NPU 的性能差异。

---

## 二、背景知识

### 2.1 YOLO 是什么？

YOLO（You Only Look Once）是目前最流行的实时目标检测模型系列。它能在一张图里同时找出多个物体、画出边界框、标出类别（如"人"、"车"、"猫"）。

与 MobileNetV2（只分类"这张图是什么"）不同，YOLO 能回答"这张图里有什么、在哪里"。

### 2.2 为什么选 YOLOv8n？

YOLOv8n（nano）是 YOLOv8 系列中最小的版本，专为边缘设备设计：

| 模型 | 参数量 | 输入 | mAP | 适合场景 |
|---|---|---|---|---|
| YOLOv8n | 3.2M | 640×640 | 37.3 | 边缘设备，速度优先 |
| YOLOv8s | 11.2M | 640×640 | 44.9 | 均衡 |
| YOLOv8m | 25.9M | 640×640 | 50.2 | 精度优先 |

YOLOv8n 的 640×640 输入比 MobileNetV2 的 224×224 大很多，计算量更能体现 NPU 相对于 CPU 的优势。

### 2.3 计算图复杂度

YOLO 包含大量 Conv、BatchNorm、Relu、Concat、Upsample 操作，以及最后的 NMS（非极大值抑制）后处理。

**注意：** NMS 是控制流操作，NPU 通常不支持，会回退 CPU。这是正常行为，属于"部分图分配"——卷积主体在 NPU，NMS 后处理在 CPU。

---

## 三、模型转换流程

### 重要发现：同样无需 NeuralONE SDK

与 Phase 3A MobileNetV2 相同，直接用 `onnxruntime.quantization` QDQ 量化即可，不需要 cixbuild。

### 3.1 导出 YOLOv8n ONNX（在 x86_64 机器上）

```bash
pip install ultralytics
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640, opset=13, simplify=True)
"
```

### 3.2 传输到板子

```bash
scp yolov8n.onnx baai@192.168.15.195:/opt/edge-ai-lab/models/yolo/yolov8n_fp32.onnx
```

### 3.3 在板子上量化为 QDQ INT8

```python
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
import numpy as np

class CalibData(CalibrationDataReader):
    def __init__(self, n=20):
        self.data = iter([{"images": np.random.rand(1,3,640,640).astype(np.float32)} for _ in range(n)])
    def get_next(self):
        return next(self.data, None)

quantize_static(
    'yolov8n_fp32.onnx', 'yolov8n_qdq.onnx',
    CalibData(20),
    quant_format=QuantFormat.QDQ,
    per_channel=False,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
)
```

---

## 四、板端推理脚本

```python
import onnxruntime as ort
import numpy as np
import os, time

os.chdir('/opt/edge-ai-lab/npu-workspace')

MODEL = '/opt/edge-ai-lab/models/yolo/yolov8n_qdq.onnx'
sess = ort.InferenceSession(
    MODEL,
    providers=["ZhouyiExecutionProvider", "CPUExecutionProvider"]
)
inp = sess.get_inputs()[0]
print(f"输入: {inp.name}, shape={inp.shape}")  # images, [1,3,640,640]

dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
for _ in range(3): sess.run(None, {inp.name: dummy})  # warmup

times = []
for _ in range(30):
    t0 = time.perf_counter()
    output = sess.run(None, {inp.name: dummy})
    times.append((time.perf_counter()-t0)*1000)

times.sort()
avg = sum(times)/len(times)
print(f"NPU avg={avg:.2f}ms FPS={1000/avg:.1f}")
print(f"Output shape: {output[0].shape}")  # (1, 84, 8400)
```

运行：

```bash
source /opt/edge-ai-lab/venv/bin/activate
export LD_LIBRARY_PATH=/usr/share/cix/lib:/usr/share/cix/lib/onnxruntime:$LD_LIBRARY_PATH
cd /opt/edge-ai-lab/npu-workspace
python3 infer_yolo.py
```

---

## 五、实测结果（2026-11-11）

| Provider | avg (ms) | P50 (ms) | FPS |
|---|---|---|---|
| ZhouyiEP（NPU） | **11.00** | 10.98 | **90.9** |
| CPU | 195.71 | 194.92 | 5.1 |
| **NPU 加速比** | **17.8×** | | |

输出 shape：`(1, 84, 8400)`（84 = 4 坐标 + 80 COCO 类别，8400 = 检测格网）✅

**横向对比（同平台 NPU 加速比）：**

| 模型 | 输入尺寸 | NPU vs CPU |
|---|---|---|
| sigmoid/mul_1 | 极小张量 | 0.5×（NPU 反而慢）|
| MobileNetV2 INT8 | 224×224 | 8.6× |
| YOLOv8n INT8 | 640×640 | **17.8×** |

规律：模型越大、输入越大，NPU 加速比越高。大输入的卷积计算量远超 NPU 启动开销，NPU 并行优势充分发挥。

原始数据：`results/bench_2026-11-11.md`

---

*本教程对应实验：Phase 3B，待执行。最后更新：2026-11-11。*
