# Phase 1 · NPU Bring-up 🔄

**状态：进行中（Step 4/7）**

## 实验目标

通过 CIX 厂商软件栈，让 Python 程序真正在 Zhouyi V3 NPU 上执行 ONNX 模型推理，并排除 CPU 回退。

## 子步骤进度

| 步骤 | 内容 | 状态 |
|---|---|---|
| Step 1 | 检查 OS 和硬件 | ✅ |
| Step 2 | 确认 aipu 驱动加载 | ✅ |
| Step 3 | 确认 `/dev/aipu` 和 NOE 运行库 | ✅ |
| Step 4 | 安装 NumPy + 厂商 ONNX Runtime wheel | 🔄 阻塞 |
| Step 5 | 验证 CIX NPU Execution Provider 注册 | ⏳ |
| Step 6 | 加载 MobileNetV2 INT8 执行推理 | ⏳ |
| Step 7 | 确认 NPU 实际执行（排除 CPU fallback） | ⏳ |

## 当前阻塞

NumPy pip 安装超时。解决方案：

```bash
# 切换国内镜像
pip install -i https://mirrors.aliyun.com/pypi/simple/ "numpy>=1.21.6,<2"

# 或从 Windows 机器传入 ARM64 wheel
scp numpy-*-cp311-cp311-linux_aarch64.whl baai@<板端IP>:/opt/edge-ai-lab/
pip install /opt/edge-ai-lab/numpy-*.whl
```

## Step 4 完成后的安装顺序

```bash
source /opt/edge-ai-lab/venv/bin/activate
pip install libnoe-2.0.0-py3-none-manylinux2014_aarch64.whl
pip install NOE_Engine-2.0.0-py3-none-manylinux2014_aarch64.whl
pip install ZhouyiOperators_x2-25.4.23-py3-none-any.whl
pip install onnxruntime_zhouyi-1.20.0-cp311-cp311-linux_aarch64.whl
```

## Step 5 验证命令

```python
import onnxruntime as ort
print(ort.get_available_providers())
# 预期输出包含：'CIXExecutionProvider'
```

## Step 6–7 推理验证

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(
    "mobilenetv2_int8.onnx",
    providers=["CIXExecutionProvider"]
)
inp = {sess.get_inputs()[0].name: np.zeros((1, 3, 224, 224), dtype=np.float32)}
out = sess.run(None, inp)
print(out)
```

验收标准见 [PROJECT_PLAN.md](../../PROJECT_PLAN.md#phase-1--npu-bring-up-)。

## 关键说明

- 必须使用 `onnxruntime_zhouyi`，不能用 PyPI 通用版 `onnxruntime`
- 推理结果正确 ≠ NPU 实际执行，须通过 Provider 分配与 `dmesg` 双重确认
