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

## 三、模型转换流程（x86_64 PC 执行）

### 3.1 获取 YOLOv8n ONNX 模型

```bash
pip install ultralytics
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640, opset=11)
print('Exported yolov8n.onnx')
"
```

### 3.2 检查算子兼容性

```bash
cixbuild check yolov8n.onnx
```

输出会列出哪些节点能在 NPU 上运行，哪些需要 CPU 回退。Conv、BatchNorm、Relu 等主干算子均在支持列表，NMS 相关部分会回退 CPU。

### 3.3 转换为 NPU 可用格式

```bash
cixbuild convert \
    --input yolov8n.onnx \
    --output yolov8n_int8_npu.onnx \
    --quant int8 \
    --calibration_data /path/to/coco_calib_images/
```

校准数据：从 COCO 验证集中取 100-200 张图片即可。

### 3.4 传输到板子

```bash
scp yolov8n_int8_npu.onnx baai@192.168.15.195:/opt/edge-ai-lab/models/yolo/
```

---

## 四、板端推理脚本

### 4.1 图片推理

```python
import onnxruntime as ort
import numpy as np
import os, time, cv2

os.chdir('/opt/edge-ai-lab/npu-workspace')

MODEL = '/opt/edge-ai-lab/models/yolo/yolov8n_int8_npu.onnx'

opts = ort.SessionOptions()
opts.log_severity_level = 1
sess = ort.InferenceSession(
    MODEL,
    sess_options=opts,
    providers=["ZhouyiExecutionProvider", "CPUExecutionProvider"]
)

inp = sess.get_inputs()[0]
print(f"输入: {inp.name}, shape={inp.shape}")  # 预期: [1, 3, 640, 640]

def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))[np.newaxis, :]  # [1, 3, 640, 640]

# 推理
image = preprocess('/opt/edge-ai-lab/datasets/test.jpg')

# 预热
for _ in range(5):
    sess.run(None, {inp.name: image})

# 测量
times = []
for _ in range(100):
    t0 = time.perf_counter()
    output = sess.run(None, {inp.name: image})
    times.append((time.perf_counter()-t0)*1000)

times.sort()
print(f"NPU avg={sum(times)/len(times):.2f}ms P50={times[50]:.2f}ms P95={times[95]:.2f}ms")
print(f"output shape: {output[0].shape}")  # 预期: [1, 84, 8400]
```

### 4.2 视频流推理（实时 FPS）

```python
import cv2, time

cap = cv2.VideoCapture(0)  # 0 = USB 摄像头，或填本地视频路径
frame_count = 0
t_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = preprocess_frame(frame)  # 参考上方 preprocess 函数
    output = sess.run(None, {inp.name: image})

    frame_count += 1
    fps = frame_count / (time.time() - t_start)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if frame_count % 30 == 0:
        print(f"FPS: {fps:.1f}")

cap.release()
```

### 4.3 Radxa 官方多路 YOLO

Radxa 提供了官方的多路视频 YOLO 示例，支持多路并行推理和实时显示：

```bash
# 参考官方文档
# https://docs.radxa.com/orion/o6/app-development/artificial-intelligence/cix-multistream-yolo
```

---

## 五、CPU vs NPU 对比测试

```python
for prov_name, providers in [
    ("ZhouyiEP", ["ZhouyiExecutionProvider", "CPUExecutionProvider"]),
    ("CPU",      ["CPUExecutionProvider"]),
]:
    sess = ort.InferenceSession(MODEL, providers=providers)
    # ... 同上测量流程
    print(f"{prov_name}: avg={avg:.2f}ms, FPS={1000/avg:.1f}")
```

预期结果（YOLOv8n INT8，640×640）：
- CPU：约 200-500ms（2-5 FPS）
- NPU：约 50-100ms（10-20 FPS）

这是 NPU 优势真正体现的场景——大输入、密集卷积计算。

---

## 六、节点分配验证

```python
opts = ort.SessionOptions()
opts.log_severity_level = 0  # verbose

sess = ort.InferenceSession(MODEL, sess_options=opts,
       providers=["ZhouyiExecutionProvider", "CPUExecutionProvider"])
```

查看日志中的关键信息：
- `Number of partitions supported by ZHOUYI: N` — N 越大越好
- 主干 Conv 层应分配给 NPU
- NMS 相关节点（`NonMaxSuppression`）会回退 CPU，这是正常的

---

## 七、验收标准

- [ ] `cixbuild` 成功转换 YOLOv8n（无致命不兼容算子）
- [ ] 板端 Session 创建成功，主干节点分配给 NPU
- [ ] 图片推理结果输出正确（`output[0].shape == [1, 84, 8400]`）
- [ ] NPU 推理 FPS 明显高于 CPU（预期 ≥2×）
- [ ] 延迟数据写入 `benchmarks/` 目录

---

*本教程对应实验：Phase 3B，待执行。最后更新：2026-11-11。*
