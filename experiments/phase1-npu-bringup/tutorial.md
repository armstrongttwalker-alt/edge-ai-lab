# Phase 1 · NPU Bring-up 小白教程

> 本教程面向第一次接触 Linux AI 推理环境的读者。从项目目标开始，逐层解释硬件、驱动、运行时、模型和实验步骤。读完这份教程，你应该能理解：为什么有 NPU 硬件不等于软件能用 NPU，以及我们如何一步步证明 NPU 真的在工作。

---

## 一、实验目标

让 Radxa Orion O6N 上的 Python 程序，通过 CIX 厂商提供的软件栈，真正调用 Zhouyi V3 NPU 执行一个 AI 模型（MobileNetV2 INT8）。

注意这句话里有两个关键词：
- **真正调用**：不是程序跑出结果就算成功，而是要证明计算确实发生在 NPU 上，不是 CPU 偷偷代劳。
- **通过厂商软件栈**：用 CIX 提供的 `onnxruntime_zhouyi`，不是 PyPI 上的通用 `onnxruntime`。

---

## 二、背景知识

在动手前，先把涉及的术语弄清楚。你不用全记，先建立大致印象，遇到再回查。

### 2.1 硬件层面

| 术语 | 小白理解 |
|---|---|
| **NPU** | 专门做 AI 计算的芯片，比 CPU 更擅长矩阵运算 |
| **AIPU** | AI Processing Unit，CIX 驱动和软件栈里用的名称，和 NPU 指同一个东西 |
| **Zhouyi V3** | 这块 NPU 的硬件架构名称（"周"系列第三版） |
| **Cluster / Core** | NPU 内部的计算单元组织方式。本设备是 1 Cluster × 3 Core |

### 2.2 软件层面

| 术语 | 小白理解 |
|---|---|
| **Driver（驱动）** | 让操作系统能管理硬件的程序 |
| **Kernel Module（内核模块）** | Linux 内核可以动态加载的功能单元，驱动通常以模块形式存在 |
| **Runtime（运行时）** | 让上层软件能调用硬件的环境 |
| **NOE** | CIX 自己的 NPU 软件接口和运行环境体系 |
| **设备节点 `/dev/aipu`** | Linux 暴露给应用程序的硬件访问入口 |

### 2.3 AI 模型层面

| 术语 | 小白理解 |
|---|---|
| **ONNX** | 一种 AI 模型交换格式，让模型能在不同推理框架间通用 |
| **ONNX Runtime** | 加载并执行 ONNX 模型的推理引擎 |
| **Execution Provider（EP）** | 决定模型计算任务交给哪个设备（CPU / GPU / NPU）的后端 |
| **INT8** | 8 位整数精度，常用于量化模型，降低计算和存储开销 |
| **CPU Fallback** | 本该在 NPU 跑的计算，因故退回 CPU 执行，我们要排除这种情况 |

### 2.4 三个最容易混的概念

- **Driver** 解决"硬件能不能被操作系统访问"
- **Runtime** 解决"软件如何使用硬件"
- **ONNX Runtime** 解决"AI 模型如何被执行"

这三层缺一不可，我们这次实验就是要把这三层全部打通。

---

## 三、NPU 推理的完整软件链路

下面这张图是整个实验的核心。要让一行 `model.run(image)` 真正跑在 NPU 上，中间要经过六层：

```
① Python 应用程序
      你写的 AI 程序
  ↓
② ONNX Runtime（onnxruntime_zhouyi）
      负责加载模型、组织计算任务
  ↓
③ CIX NPU Execution Provider
      把适合 NPU 的计算节点交给 NPU 后端
  ↓
④ CIX NOE Runtime（libnoe.so）
      CIX 厂商提供的 NPU 运行环境
  ↓
⑤ Linux aipu 驱动（/dev/aipu）
      内核驱动，负责把任务提交给硬件
  ↓
⑥ Zhouyi V3 NPU
      真正执行 AI 计算
```

**任何一层出问题，计算就无法到达 NPU。** 常见的失败情形：

| 缺少哪层 | 现象 |
|---|---|
| 没有驱动 | `lsmod \| grep aipu` 无输出，`/dev/aipu` 不存在 |
| 没有运行库 | `import onnxruntime` 报错，或 Provider 列表里没有 `CIXExecutionProvider` |
| 安装了错误的 ONNX Runtime | 用了 PyPI 通用版，不含 CIX EP，计算全走 CPU |
| 模型算子不支持 | ONNX Runtime 把不支持的节点自动回退到 CPU |

这就是为什么"程序跑出了结果"不够——我们必须逐层确认。

---

## 四、实验分为 7 个步骤

| 步骤 | 内容 | 状态 |
|---|---|---|
| Step 1 | 检查操作系统和硬件 | ✅ 已完成 |
| Step 2 | 确认 NPU 驱动已加载 | ✅ 已完成 |
| Step 3 | 确认设备节点和厂商运行库 | ✅ 已完成 |
| Step 4 | 安装 NumPy + 厂商 ONNX Runtime | 🔄 阻塞（NumPy 超时） |
| Step 5 | 验证 NPU Execution Provider | ⏳ |
| Step 6 | 加载 MobileNetV2 INT8 执行推理 | ⏳ |
| Step 7 | 确认 NPU 实际执行，排除 CPU 回退 | ⏳ |

---

## 五、已完成步骤详解（Step 1–3）

### Step 1：检查操作系统和硬件

**目的：** 确认设备环境满足运行 AI 软件的基本要求。

```bash
uname -a                   # Linux 内核版本和架构
cat /etc/os-release        # 操作系统版本
lscpu                      # CPU 信息（架构、核心数）
free -h                    # 内存
df -h                      # 磁盘空间
```

**你的设备验证结果：**

| 项目 | 结果 | 重要性 |
|---|---|---|
| OS | Debian 12 | 决定软件包格式（deb）|
| Kernel | 6.6.89-3-sky1 | 驱动与内核版本绑定 |
| CPU 架构 | ARM64 | 决定 wheel 文件格式（aarch64）|
| CPU 核心 | 12 | 推理时可用的线程数 |
| RAM | 32 GB | 决定能加载多大的模型 |
| NVMe | 1 TB | 存放模型和数据集 |

**为什么检查架构很关键？** 厂商提供的 ONNX Runtime wheel 文件名包含 `linux_aarch64`，表示只能在 ARM64 Linux 上使用。如果误将 x86_64 版本装上去，会报 ABI 不兼容错误。

---

### Step 2：确认 NPU 驱动已加载

**目的：** 确认 Linux 内核已经加载了管理 NPU 的驱动模块。

```bash
lsmod | grep aipu
```

`lsmod` 列出当前加载的所有内核模块，`grep aipu` 过滤出 NPU 相关的行。

**你的输出：**
```
aipu 98304 0
```

三列含义：模块名 `aipu`、占用内存 `98304` 字节、被其他模块引用次数 `0`。

```bash
dmesg | grep -i aipu
```

`dmesg` 读取内核日志，`-i` 表示不区分大小写。

**你的输出包含：**
```
AIPU detected: zhouyi-v3
ZHOUYI V3 AIPU
Enabled Partition Count: 1
Enabled Cluster Count: 1
Core Count per Cluster: 3
```

这说明：内核启动时 NPU 驱动正常初始化，识别到了 Zhouyi V3 架构，1 个 Cluster 下有 3 个计算 Core。

**结论：** 驱动正常，不需要重装驱动或升级内核。

---

### Step 3：确认设备节点和厂商运行库

**目的：** 确认用户态程序有路径访问 NPU，且厂商提供了所需的运行库。

```bash
ls -l /dev/aipu
```

**你的输出：**
```
crw-rw-rw- 1 root root 10, 117 ... /dev/aipu
```

- `c` 表示字符设备（Character device），这是 Linux 中驱动程序与用户程序交互的标准方式。
- `rw-rw-rw-` 表示所有用户都有读写权限，Python 程序无需 root 就能访问 NPU。

```bash
dpkg -L cix-noe-umd
dpkg -L cix-npu-onnxruntime
```

`dpkg -L` 列出某个已安装软件包包含的文件。

**找到的关键文件：**

| 文件 | 作用 |
|---|---|
| `/usr/share/cix/lib/libnoe.so` | CIX NOE 动态链接库，Python 通过它调用 NPU |
| `/usr/share/cix/lib/onnxruntime/libonnxruntime.so` | ONNX Runtime 底层动态库 |
| `/usr/share/cix/include/npu/cix_noe_standard_api.h` | C/C++ 接口声明（开发时参考用）|

`.so` 是 Linux 的共享库（Shared Object），相当于 Windows 的 DLL。程序运行时动态加载这些库，不需要把所有功能都编译进自身。

**找到的厂商 Python wheel：**

```
libnoe-2.0.0-py3-none-manylinux2014_aarch64.whl
NOE_Engine-2.0.0-py3-none-manylinux2014_aarch64.whl
onnxruntime_zhouyi-1.20.0-cp311-cp311-linux_aarch64.whl
ZhouyiOperators_x2-25.4.23-py3-none-any.whl
```

`.whl` 是 Python 软件包格式（Wheel）。文件名的规律：
- `cp311` → 只支持 Python 3.11
- `linux_aarch64` → 只支持 ARM64 Linux
- `py3-none-manylinux2014_aarch64` → 纯 Python 包，兼容 manylinux2014 标准

**结论：** 设备节点、动态库、Python wheel 都在，可以进行下一步安装。

---

## 六、Step 4：安装 NumPy 和 ONNX Runtime（当前阻塞点）

### 为什么需要虚拟环境？

之前已经创建了虚拟环境：

```bash
python3 -m venv --system-site-packages /opt/edge-ai-lab/venv
```

参数解释：
- `python3 -m venv`：用 Python 内置工具创建虚拟环境
- `--system-site-packages`：允许虚拟环境访问系统已安装的包（CIX 安装到系统目录的 NOE 包可以被复用）
- `/opt/edge-ai-lab/venv`：虚拟环境的存放路径

每次使用前激活：
```bash
source /opt/edge-ai-lab/venv/bin/activate
# 终端提示符变为 (venv)$，表示已进入虚拟环境
```

### 为什么先装 NumPy？

NumPy 是 Python 科学计算的基础库。在 AI 推理中，它负责：
- 创建输入数据（如 `np.zeros((1, 3, 224, 224), dtype=np.float32)` 创建一张空白图片）
- 接收模型输出结果（模型返回的张量就是 NumPy 数组）

厂商 ONNX Runtime 明确要求 `numpy>=1.21.6`，所以必须先装。

### 解决安装超时

pip 默认从国外服务器下载，在国内经常超时。解决方法：

**方法 1（推荐）：切换国内镜像**
```bash
source /opt/edge-ai-lab/venv/bin/activate
pip install -i https://mirrors.aliyun.com/pypi/simple/ "numpy>=1.21.6,<2"
```

**方法 2：用清华镜像**
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ "numpy>=1.21.6,<2"
```

**方法 3：Windows 机器下载后传入**
1. 在 Windows 浏览器访问 `https://pypi.org/project/numpy/#files`
2. 下载文件名含 `cp311` 和 `linux_aarch64` 的 `.whl` 文件
3. 用 scp 传到板子：
```bash
scp numpy-*-cp311-cp311-linux_aarch64.whl baai@<板端IP>:/opt/edge-ai-lab/
```
4. 在板子上安装：
```bash
pip install /opt/edge-ai-lab/numpy-*-cp311-cp311-linux_aarch64.whl
```

### NumPy 安装成功后，按顺序安装厂商 wheel

顺序很重要，后面的包依赖前面的包：

```bash
# 确认在虚拟环境内（提示符有 (venv)）
pip install libnoe-2.0.0-py3-none-manylinux2014_aarch64.whl
pip install NOE_Engine-2.0.0-py3-none-manylinux2014_aarch64.whl
pip install ZhouyiOperators_x2-25.4.23-py3-none-any.whl
pip install onnxruntime_zhouyi-1.20.0-cp311-cp311-linux_aarch64.whl
```

**验证安装成功：**
```bash
python -c "import onnxruntime; print(onnxruntime.__version__)"
# 预期输出：1.20.0（或类似版本号，不报错即可）
```

---

## 七、Step 5：验证 NPU Execution Provider

**目的：** 确认厂商 ONNX Runtime 注册了 CIX NPU 后端。

```python
import onnxruntime as ort
providers = ort.get_available_providers()
print(providers)
```

**预期输出（包含 NPU Provider）：**
```
['CIXExecutionProvider', 'CPUExecutionProvider']
```

如果只有 `['CPUExecutionProvider']`，说明 NPU Provider 未注册，需要检查 `libnoe.so` 是否在 `LD_LIBRARY_PATH` 中：

```bash
export LD_LIBRARY_PATH=/usr/share/cix/lib:$LD_LIBRARY_PATH
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

---

## 八、Step 6：加载 MobileNetV2 INT8 执行推理

### 为什么选 MobileNetV2？

MobileNetV2 是专为移动和边缘设备设计的轻量图像分类模型。选它作为第一个实验模型的原因：
- 体积小，输入输出格式清晰（224×224 RGB 图片）
- INT8 量化版本对 NPU 友好
- 推理结果容易验证（返回 1000 类的概率分布）

### 准备模型

> 注意：模型需要在 x86_64 PC 上用 NeuralONE SDK 转换，再复制到板子。这部分流程在 Phase 3A 详细展开。当前 Step 6 使用已转换好的 `.onnx` 文件。

模型存放位置：
```bash
/opt/edge-ai-lab/models/mobilenetv2/mobilenetv2_int8.onnx
```

### 推理脚本

```python
import onnxruntime as ort
import numpy as np

# 加载模型，指定使用 NPU
sess = ort.InferenceSession(
    "/opt/edge-ai-lab/models/mobilenetv2/mobilenetv2_int8.onnx",
    providers=["CIXExecutionProvider", "CPUExecutionProvider"]
)

# 查看输入节点信息
inp = sess.get_inputs()[0]
print(f"输入节点名：{inp.name}")
print(f"输入形状：{inp.shape}")   # 预期：[1, 3, 224, 224]
print(f"输入类型：{inp.type}")    # 预期：tensor(float)

# 构造随机输入（实际使用时替换为真实图片）
dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

# 执行推理
outputs = sess.run(None, {inp.name: dummy_input})
print(f"输出形状：{outputs[0].shape}")  # 预期：(1, 1000)
print(f"最高概率类别索引：{np.argmax(outputs[0])}")
```

---

## 九、Step 7：确认 NPU 实际执行（最重要的一步）

**这是整个实验的验收核心。** 程序返回结果不代表 NPU 工作了，可能存在以下三种情况：

```
情况 A（目标）：NPU 执行
Python → ONNX Runtime → CIXExecutionProvider → NPU ✅

情况 B（静默失败）：全部回退 CPU
Python → ONNX Runtime → CPUExecutionProvider → CPU ❌

情况 C（部分失败）：部分算子回退 CPU
某些节点在 NPU，不支持的节点回退 CPU（需要记录回退比例）
```

### 检查 Provider 分配情况

```python
# 在创建 Session 时开启 verbose 模式
options = ort.SessionOptions()
options.log_severity_level = 0  # 0 = 最详细的日志

sess = ort.InferenceSession(
    "mobilenetv2_int8.onnx",
    sess_options=options,
    providers=["CIXExecutionProvider", "CPUExecutionProvider"]
)
```

日志中查找类似以下内容（表示节点被 NPU 接管）：
```
Node xxx assigned to CIXExecutionProvider
```

如果看到：
```
Node xxx assigned to CPUExecutionProvider
```
说明该节点回退到了 CPU。

### 检查 NPU 设备日志

```bash
# 执行推理前清空旧日志，推理后查看是否有 NPU 活动
sudo dmesg -C
python run_inference.py
sudo dmesg | grep -i aipu
```

有 NPU 任务时，dmesg 应有类似输出：
```
aipu: job submitted
aipu: job done
```

### 验收清单

以下全部满足，才算本实验通过：

- [ ] `onnxruntime_zhouyi` 导入无报错
- [ ] `CIXExecutionProvider` 在 Provider 列表中
- [ ] Session 创建成功（无报错）
- [ ] 推理结果输出正确（`outputs[0].shape == (1, 1000)`）
- [ ] 至少大部分计算节点分配给 NPU（非全部 CPU 回退）
- [ ] `dmesg` 或 NOE 日志显示 NPU 收到了任务

---

## 十、实验记录模板

每次运行实验后，在 `results/` 目录保存以下信息：

```bash
# 保存系统快照
uname -a > results/env_snapshot.txt
pip list >> results/env_snapshot.txt
python -c "import onnxruntime as ort; print(ort.get_available_providers())" >> results/env_snapshot.txt

# 保存推理输出
python run_inference.py > results/inference_output.txt 2>&1

# 保存内核日志
sudo dmesg | grep -i aipu > results/dmesg_aipu.txt
```

---

## 十一、遇到问题怎么办

| 问题现象 | 可能原因 | 排查方向 |
|---|---|---|
| `ModuleNotFoundError: onnxruntime` | wheel 未安装，或虚拟环境未激活 | 确认 `(venv)` 提示符，重新安装 |
| Provider 列表无 `CIXExecutionProvider` | `libnoe.so` 路径未注册 | 设置 `LD_LIBRARY_PATH` |
| Session 创建报错 | 模型格式不兼容，或 NPU Runtime 版本不匹配 | 检查模型转换流程 |
| 推理结果全是 CPU 节点 | 模型算子 NPU 不支持 | 查阅 NeuralONE 支持的算子列表 |
| pip 下载超时 | 网络问题 | 切换国内镜像或离线安装 |

---

*本教程对应实验进度：Phase 1，Step 4/7。最后更新：2026-11。*

