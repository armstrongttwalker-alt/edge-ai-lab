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

## 三、模型转换流程（x86_64 PC 执行）

### 3.1 安装 NeuralONE SDK

```bash
# 在 x86_64 Linux PC 上执行
wget https://dl.radxa.com/orion/common/cix_noe_sdk_25_q3_release.tar.gz
tar -xvf cix_noe_sdk_25_q3_release.tar.gz
cd cix_noe_sdk_25_q3_release/

# 创建 Python 3.10 环境
conda create -n cix-noe python=3.10
conda activate cix-noe
bash env_setup.sh

# 验证安装
cixbuild -v
```

### 3.2 下载 MobileNetV2 FP32 模型

```bash
# 从 ONNX Model Zoo 下载
wget https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx
```

### 3.3 检查算子兼容性

```bash
cixbuild check mobilenetv2-10.onnx
```

输出会列出哪些算子 NPU 支持，哪些需要回退 CPU。MobileNetV2 的算子（Conv、Relu、Add、GlobalAvgPool、Reshape）均在支持列表内。

### 3.4 转换为 INT8

```bash
cixbuild convert \
    --input mobilenetv2-10.onnx \
    --output mobilenetv2_int8_npu.onnx \
    --quant int8 \
    --calibration_data /path/to/calibration_images/
```

`--calibration_data`：量化需要用少量真实数据（约 100-200 张图片）校准统计分布。可以用 ImageNet 验证集的一小部分。

### 3.5 传输到板子

```bash
scp mobilenetv2_int8_npu.onnx baai@192.168.15.195:/opt/edge-ai-lab/models/mobilenetv2/
```

---

## 四、板端推理脚本

```python
import onnxruntime as ort
import numpy as np
import os, time

# 必须从含 operator 软链的目录运行
os.chdir('/opt/edge-ai-lab/npu-workspace')

MODEL = '/opt/edge-ai-lab/models/mobilenetv2/mobilenetv2_int8_npu.onnx'

# 创建 Session
opts = ort.SessionOptions()
opts.log_severity_level = 1  # 显示节点分配信息
sess = ort.InferenceSession(
    MODEL,
    sess_options=opts,
    providers=["ZhouyiExecutionProvider", "CPUExecutionProvider"]
)

inp = sess.get_inputs()[0]
print(f"输入: {inp.name}, shape={inp.shape}, type={inp.type}")

# 构造 224×224 RGB 图片（实际使用时替换为真实图片预处理结果）
dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)

# 预热
for _ in range(5):
    sess.run(None, {inp.name: dummy})

# 测量延迟
times = []
for _ in range(100):
    t0 = time.perf_counter()
    output = sess.run(None, {inp.name: dummy})
    times.append((time.perf_counter()-t0)*1000)

times.sort()
print(f"NPU avg={sum(times)/len(times):.2f}ms P50={times[50]:.2f}ms P95={times[95]:.2f}ms")
print(f"Top-1 预测类别: {np.argmax(output[0])}")
```

运行：

```bash
source /opt/edge-ai-lab/venv/bin/activate
export LD_LIBRARY_PATH=/usr/share/cix/lib:/usr/share/cix/lib/onnxruntime:$LD_LIBRARY_PATH
python3 /opt/edge-ai-lab/experiments/npu/infer_mobilenetv2.py
```

---

## 五、验收标准

- [ ] `cixbuild` 在 PC 上成功转换模型（无不兼容算子报错）
- [ ] 转换后模型传输到板子
- [ ] Session 创建时显示 NPU 节点分配（`placed on [ZhouyiExecutionProvider]`）
- [ ] 推理结果 shape 为 `(1, 1000)`
- [ ] NPU vs CPU 延迟对比：NPU 明显快于 CPU（预期 MobileNetV2 INT8 NPU 延迟 < 20ms）
- [ ] NPU vs CPU 输出误差 < 0.01

---

*本教程对应实验：Phase 3A，待执行。最后更新：2026-11-11。*
