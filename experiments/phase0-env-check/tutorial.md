# Phase 0 · 基础环境检查 小白教程

> 在开展任何 AI 实验之前，我们需要先摸清这台设备的"家底"：操作系统是哪个版本、CPU 是什么架构、内存有多少、NPU 驱动有没有加载。这一步不改变任何系统配置，只是观察和记录。
>
> **实验状态：已完成**

---

## 一、实验目标

通过一组只读命令，全面记录 Radxa Orion O6N 的系统与硬件状态，确认后续实验所需的基础条件已经满足，并建立可供后续对比的环境基准快照。

---

## 二、为什么要做这一步？

不同的 Linux 版本、内核版本和 CPU 架构，对应不同的驱动包、Python wheel 和工具链。如果跳过这一步直接安装软件，很容易遇到"明明下载了安装包，就是装不上"的情况。

举个例子：
- 厂商提供的 ONNX Runtime wheel 文件名包含 `linux_aarch64`，只能在 **ARM64** 架构的 Linux 上运行
- 如果你把它装到 x86_64 的机器上，会报 `cannot import name` 或 ABI 不兼容错误
- 只有先确认架构是 ARM64，才能放心下载对应的包

---

## 三、背景知识

### 3.1 Linux 基本概念

| 术语 | 小白理解 |
|---|---|
| **Kernel（内核）** | Linux 操作系统最核心的部分，管理硬件、内存和进程 |
| **ARM64 / aarch64** | CPU 指令集架构，Radxa Orion O6N 使用 ARM 64 位架构 |
| **Debian** | 一种 Linux 发行版，Ubuntu 也是基于 Debian 的 |
| **Docker** | 容器化平台，让应用运行在隔离的环境中 |
| **NVMe** | 一种高速固态硬盘接口协议，比传统 SATA 快很多 |
| **LPDDR5** | 低功耗 DDR5 内存，常用于移动和嵌入式设备 |

### 3.2 NPU 相关概念

| 术语 | 小白理解 |
|---|---|
| **NPU** | 专门做 AI 计算的处理器，本设备的 NPU 叫 Zhouyi V3 |
| **驱动（Driver）** | 让操作系统能管理硬件的程序 |
| **内核模块（Kernel Module）** | 可以动态加载到 Linux 内核的功能单元，驱动通常以模块形式存在 |
| **设备节点（/dev/aipu）** | Linux 暴露给应用程序的硬件访问入口，位于 `/dev/` 目录 |
| **用户态库（.so）** | 动态链接库，程序运行时调用，相当于 Windows 的 DLL |
| **dpkg** | Debian/Ubuntu 的软件包管理工具 |

---

## 四、执行命令与结果

> 以下所有命令均为只读操作，不修改任何系统文件。

### 4.1 系统版本

```bash
uname -a
```

查看 Linux 内核完整信息。

```bash
cat /etc/os-release
```

查看操作系统发行版信息。

**验证结果：**
```
Linux RadxaOrionO6N 6.6.89-3-sky1 ... aarch64 GNU/Linux
Debian GNU/Linux 12 (bookworm)
```

关键信息：内核 `6.6.89-3-sky1`（sky1 是 CIX P1 的平台代号），架构 `aarch64`（即 ARM64），操作系统 Debian 12。

---

### 4.2 CPU 信息

```bash
lscpu
```

查看 CPU 架构、核心数、频率等信息。

**验证结果摘要：**

| 字段 | 值 | 意义 |
|---|---|---|
| Architecture | aarch64 | ARM 64 位，确认 wheel 要选 aarch64 版本 |
| CPU(s) | 12 | 总核心数（8 个 A720 大核 + 4 个 A520 小核） |
| Model name | CIX P1 / Cortex-A720 | 确认硬件型号 |

---

### 4.3 内存

```bash
free -h
```

`-h` 表示 human-readable，用 GB/MB 显示而不是字节数。

**验证结果：**
```
              total    used    free
Mem:           30Gi    ...     ...
```

显示约 30 GiB，实际是 32 GB LPDDR5。操作系统会保留一部分内存给内核和硬件，所以显示比实际少一些，这是正常的。

---

### 4.4 磁盘

```bash
df -h
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS,MODEL
```

`df -h` 查看各挂载点的磁盘使用情况。`lsblk` 查看块设备（硬盘、分区）的树状结构。

**验证结果：** 1 TB NVMe SSD 已挂载，`/opt` 分区有足够空间存放模型和实验数据。

---

### 4.5 NPU 驱动

```bash
lsmod | grep aipu
```

`lsmod` 列出已加载的内核模块，`grep aipu` 过滤 NPU 相关模块。

**验证结果：**
```
aipu    98304    0
```

三列分别是：模块名、占用内存（字节）、被其他模块引用次数。`aipu 98304 0` 表示驱动已加载，未被其他模块依赖。

```bash
dmesg | grep -i aipu
```

`dmesg` 读取内核环形缓冲区日志，记录了系统启动以来的内核事件。`-i` 忽略大小写。

**验证结果（关键行）：**
```
AIPU detected: zhouyi-v3
ZHOUYI V3 AIPU
Enabled Partition Count: 1
Enabled Cluster Count: 1
Core Count per Cluster: 3
```

这段日志说明：驱动在系统启动时成功识别了 Zhouyi V3 NPU，启用了 1 个 Cluster，每个 Cluster 包含 3 个计算 Core。

---

### 4.6 NPU 设备节点

```bash
ls -l /dev/aipu
```

**验证结果：**
```
crw-rw-rw- 1 root root 10, 117 ... /dev/aipu
```

- `c` 表示字符设备（Character device）
- `rw-rw-rw-` 表示所有用户都有读写权限
- `10, 117` 是主次设备号，Linux 内核用来识别具体驱动

这说明 NPU 设备节点存在且可被普通用户访问（无需 sudo）。

```bash
cat /sys/class/misc/aipu/dev
```

**输出：** `10:117`，与上面的设备号一致。

---

### 4.7 CIX 厂商软件包

```bash
dpkg -l | grep -Ei 'cix|noe|npu'
```

列出所有已安装的 CIX / NOE / NPU 相关软件包。

```bash
dpkg -L cix-noe-umd
dpkg -L cix-npu-onnxruntime
```

`-L` 列出某个包安装的所有文件。

**找到的关键文件：**

| 文件路径 | 作用 |
|---|---|
| `/usr/share/cix/lib/libnoe.so` | CIX NOE 动态库，Python 通过它调用 NPU |
| `/usr/share/cix/lib/onnxruntime/libonnxruntime.so` | ONNX Runtime 动态库 |
| `/usr/share/cix/include/npu/cix_noe_standard_api.h` | C/C++ 接口头文件 |

**找到的厂商 Python wheel（待安装）：**
```
libnoe-2.0.0-py3-none-manylinux2014_aarch64.whl
NOE_Engine-2.0.0-py3-none-manylinux2014_aarch64.whl
onnxruntime_zhouyi-1.20.0-cp311-cp311-linux_aarch64.whl
ZhouyiOperators_x2-25.4.23-py3-none-any.whl
```

wheel 文件名的规律：
- `cp311` → 只支持 Python 3.11
- `linux_aarch64` → 只支持 ARM64 Linux
- `py3-none-manylinux2014_aarch64` → 纯 Python 包，兼容 manylinux2014 标准

---

### 4.8 Docker

```bash
docker info --format '{{.Architecture}} {{.ServerVersion}}'
```

**验证结果：** `linux/arm64 ...`，Docker 已安装，架构正确。

---

## 五、建立实验目录

环境检查完成后，建立统一的实验目录。所有模型、数据集、脚本和日志都存放在这里：

```bash
sudo mkdir -p /opt/edge-ai-lab/{models,datasets,experiments,logs,services,benchmarks}
sudo chown -R $USER:$USER /opt/edge-ai-lab

mkdir -p /opt/edge-ai-lab/models/{llm,vision,audio,converted}
mkdir -p /opt/edge-ai-lab/experiments/{npu,cpu,gpu,llm,vision}
mkdir -p /opt/edge-ai-lab/services/{open-webui,api,qdrant,monitoring}
```

**目录含义：**

| 目录 | 用途 |
|---|---|
| `models/` | 存放 AI 模型文件（.onnx、.gguf 等） |
| `datasets/` | 测试数据集 |
| `experiments/` | 各类实验脚本 |
| `benchmarks/` | 性能测试结果 |
| `logs/` | 实验日志 |
| `services/` | Docker 服务配置 |

---

## 六、创建 Python 虚拟环境

```bash
python3 -m venv --system-site-packages /opt/edge-ai-lab/venv
```

参数解释：
- `--system-site-packages`：允许虚拟环境访问系统已安装的包（CIX 安装到系统目录的 NOE 包可被复用，不用重装）

以后进入实验环境：
```bash
source /opt/edge-ai-lab/venv/bin/activate
# 终端提示符变为 (venv)$
```

---

## 七、验收标准（全部通过）

| 检查项 | 预期结果 | 状态 |
|---|---|---|
| OS 版本 | Debian 12 | ✅ |
| 内核版本 | 6.6.89-3-sky1 | ✅ |
| CPU 架构 | ARM64 / aarch64 | ✅ |
| CPU 核心 | 12 | ✅ |
| RAM | ≥ 30 GiB | ✅ |
| 存储 | ≥ 100 GB 可用 | ✅ |
| NPU 驱动模块 | `aipu 98304 0` | ✅ |
| NPU 识别 | Zhouyi V3，1×3 Core | ✅ |
| NPU 设备节点 | `/dev/aipu` 存在且可读写 | ✅ |
| CIX 动态库 | `libnoe.so` 存在 | ✅ |
| Docker | 已安装，架构 arm64 | ✅ |
| 实验目录 | `/opt/edge-ai-lab/` 创建完成 | ✅ |
| Python venv | `/opt/edge-ai-lab/venv` 创建完成 | ✅ |

---

## 八、完整验证脚本

把以下脚本保存为 `experiments/phase0-env-check/check_env.sh`，可以一键输出所有环境信息：

```bash
#!/bin/bash
echo "===== SYSTEM ====="
cat /etc/os-release | grep -E "^NAME|^VERSION="
uname -r

echo "===== CPU ====="
lscpu | grep -E "Architecture|CPU\(s\)|Model name"

echo "===== MEMORY ====="
free -h | grep Mem

echo "===== STORAGE ====="
df -h /opt

echo "===== NPU DRIVER ====="
lsmod | grep aipu
dmesg | grep -i "aipu detected"

echo "===== NPU DEVICE ====="
ls -l /dev/aipu 2>/dev/null || echo "/dev/aipu NOT FOUND"

echo "===== CIX PACKAGES ====="
dpkg -l | grep -Ei 'cix|noe|npu' | awk '{print $2, $3}'

echo "===== DOCKER ====="
docker info --format '{{.Architecture}} {{.ServerVersion}}' 2>/dev/null || echo "Docker not running"

echo "===== LAB DIR ====="
ls /opt/edge-ai-lab/ 2>/dev/null || echo "/opt/edge-ai-lab NOT FOUND"
```

---

*本教程对应实验：Phase 0，已完成。最后更新：2026-11。*
