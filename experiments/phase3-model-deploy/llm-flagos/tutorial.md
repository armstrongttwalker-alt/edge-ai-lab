# Phase 3C · FlagOS + vLLM 端侧大模型推理 小白教程

> 本教程面向第一次接触大模型推理优化的读者。你将理解：为什么在 ARM CPU 上跑大模型需要专门的算子优化，以及 FlagOS 是如何把 MiniCPM-V 4.7 的推理速度提升 2 倍以上的。
>
> **实验状态：已完成**（PP512 +223%，TG128 +87%）

---

## 一、实验目标

在 Radxa Orion O6N（CIX P1）上，使用 FlagOS 软件栈运行 MiniCPM-V 4.7 多模态大模型，对比三种推理路线（Stock / Minimal / FlagOS）的性能差异，验证底层算子优化对 ARM 平台推理速度的实际影响。

简单说：我们要证明，"同一个模型、同一块芯片，软件不同，速度可以差 3 倍"。

---

## 二、背景知识

### 2.1 模型与量化

**MiniCPM-V 4.7** 是一个多模态模型，能同时处理文字和图片。参数量 1B（10 亿），对边缘设备比较友好。

这个实验用了一个关键概念——**混合精度**：

| 模型部分 | 精度 | 为什么这样选 |
|---|---|---|
| 视觉编码器 | BF16（16 位浮点）| 保留高精度，图片理解和 OCR 效果好 |
| 语言骨干 + 输出头 | W8A8（8 位整数）| 降低计算和存储开销，提速明显 |

- **W8A8** 中的 W 表示 Weight（权重），A 表示 Activation（激活值）。W8A8 表示权重用 8 位整数存储，激活值在推理时动态量化为 8 位。
- 测试时上下文限制为 4096 token（约 3000 个汉字），KV-cache（保存历史 token 信息的缓存）设为 256 MiB。

### 2.2 软件栈版本锁定

为保证实验可复现，所有版本严格锁定：

| 组件 | 版本 | 作用 |
|---|---|---|
| vLLM | 0.24.0+CPU | 推理调度引擎，管理请求、采样、内存 |
| PyTorch | 2.11.0 | 深度学习框架底层 |
| Transformers | 5.15.1 | 模型加载与 Tokenizer |
| vLLM-plugin-FL | commit f0de70233590 | 把模型注册到 vLLM，接入量化 |
| FlagGems | commit b5f5e110c7c2 | 底层算子库 |
| FlagTree-CPU | commit 36133df9540c | MLIR/LLVM 编译器，把算子编译成机器码 |

### 2.3 关键术语

| 术语 | 小白理解 |
|---|---|
| **vLLM** | 高性能 LLM 推理引擎，负责接收请求、调度生成 |
| **PP512** | Prefill 512，预填充阶段，处理 512 个输入 token，测算力 |
| **TG128** | Token Generation 128，解码阶段，生成 128 个输出 token，测访存 |
| **Triton** | 一种中间语言，可编译成 GPU/CPU 指令 |
| **Neon SDOT** | ARM 处理器的向量指令集，专门加速 8 位整数矩阵乘法 |
| **GEMM** | 矩阵乘法，大模型计算的核心操作 |

### 2.4 CIX P1 的 CPU 结构

CIX P1 的 CPU 是大小核设计：

- **8 个 A720 大核**：性能强，大模型推理主要用这 8 个核心
- **4 个 A520 小核**：省电，处理轻量任务

FlagOS 默认把推理任务调度到 8 个 A720 核心上，并利用 A720 支持的 Neon SDOT 指令集做 8 位整数加速。

---

## 三、三条路线的本质区别

这次实验对比三种引擎配置，理解它们的区别是理解实验意义的关键。

### Stock（开箱即用）
官方 vLLM 0.24 没有注册 MiniCPM-V 4.7 的模型架构，直接运行会抛出 `ValidationError`——引擎根本不认识这个模型。

### Minimal（最小适配）
打上注册补丁，让引擎能加载模型。但底层算子仍然是 PyTorch 的通用实现，没有针对 ARM64 做任何优化。就像用通用工具完成一项专业工作——能用，但效率低。

### FlagOS（优化版）
在 Minimal 基础上，拦截并替换关键算子：
- 用 Triton 或 TLE Raw（自己写的 C++/Neon 汇编）代替通用算子
- 利用 ARM Neon SDOT 指令加速 W8 矩阵乘法
- 优化线程调度（Decode 阶段从 8 线程改为 4 线程，减少争抢）

类比：三辆相同的汽车，Stock 是拒绝出发，Minimal 是走普通路，FlagOS 是走高速公路。

---

## 四、推理架构图

```
你发送的请求（文字 / 图片）
  ↓
vLLM 0.24（请求调度、采样、内存管理）
  ↓
vLLM-plugin-FL（模型注册与量化接入）
  ↓
FlagGems（提供底层算子）
  ↓
FlagTree-CPU（MLIR/LLVM 编译器）
  ↓
CIX P1 · 8× A720 核心 · Neon SDOT
```

两个容器的分工：
- **开发容器**（`sleep infinity`）：安装 wheel、运行校验和跑分脚本
- **服务容器**（`vllm serve`）：接收 HTTP 请求，提供推理服务

---

## 五、执行步骤详解

### Step 01：设定工作目录

```bash
ls -d /home/baai/workshop1018     # 确认目录存在
export WORK_ROOT=/home/baai/workshop1018
cd $WORK_ROOT
source $WORK_ROOT/scripts/activate_host.env
```

`export` 把路径存入环境变量，当前 shell 的所有后续命令都能用 `$WORK_ROOT` 引用这个路径。`source` 执行脚本，设定 `CIX_IMAGE_REF`（基础镜像地址）等变量。

### Step 02：校验材料完整性

```bash
bash $WORK_ROOT/scripts/check_preloaded.sh
bash $WORK_ROOT/scripts/check_prebuilt.sh
```

这两个脚本用 SHA256 哈希值校验模型文件和 wheel 包是否与锁定清单一致。**必须看到 `PRELOADED_CHECK_PASS` 和 `PREBUILT_CHECK_PASS`**，不一致就停止，说明文件被意外修改或下载不完整。

### Step 03：准备测试目录

```bash
cp -r /home/baai/workshop1018 /home/baai/workshop1018-test
rm -rf /home/baai/workshop1018-test/envs
export WORK_ROOT=/home/baai/workshop1018-test
cd $WORK_ROOT
source $WORK_ROOT/scripts/activate_host.env
```

为什么要删 `envs/`？基础镜像自带了一个初始化的环境目录，如果不删除，后续 `prepare_env.py` 会因为目录已存在抛出 `FileExistsError`。

### Step 04：创建开发容器

```bash
CONTAINER=flags-workshop1018-test bash $WORK_ROOT/scripts/init_container.sh
sudo docker exec -it flags-workshop1018-test /bin/bash
```

执行第二条命令后，终端提示符变为 `root@...`，表示你已进入容器内部。**此后的命令都在容器内执行**，直到 Step 07 的 `exit`。

### Step 05：离线安装预编译 Wheel

```bash
/opt/flagos-runtime/bin/flagos-python --profile 1b /work/scripts/prepare_env.py
bash /work/scripts/dev_python.sh /work/scripts/prebuilt_tools.py preflight /work
bash /work/scripts/dev_python.sh -m pip install --no-index --no-deps --no-compile \
    --target /work/envs/dev/lib/python3.11/site-packages /work/prebuilt/wheels/*.whl
bash /work/scripts/dev_python.sh /work/scripts/prebuilt_tools.py installed /work
```

关键 pip 参数解释：
- `--no-index`：禁止联网，只用本地文件
- `--no-deps`：不自动替换镜像里锁定的 PyTorch 等依赖
- `--target`：安装到指定目录，与基础镜像环境隔离

**预期第四条命令输出 `PREBUILT_IMPORT_AND_RAW_LOAD_PASS`。**

### Step 06：正确性检查

```bash
bash /work/scripts/validate_source.sh /work/evidence/workshop-run01
```

这个脚本执行数值测试和功能测试（算术、中文问答、红色图片颜色识别、OCR），同时验证是否真实调用了优化算子（而非回退默认实现）。

**预期输出 `SOURCE_VALIDATION_PASS`。**

### Step 07：退容器，启动推理服务

```bash
exit   # 退出容器，回到主机

export CACHE_DIR="$WORK_ROOT/cache/http-run01"
export RUN_DIR="$WORK_ROOT/evidence/http-run01"
export CONTAINER=minicpm-1b-workshop1018-test
bash $WORK_ROOT/scripts/start_service.sh
```

`start_service.sh` 启动 `vllm serve`，挂载模型（只读挂载为 `/model`），分配 8 个 A720 核心。启动过程会刷屏显示 `curl: (7) Failed to connect`——这是正常轮询，服务还没就绪。**等到打印 `READY` 才表示可以接收请求。**

### Step 08：发送推理请求

```bash
# 文字请求
python3 $WORK_ROOT/scripts/http_request.py \
    --output "$RUN_DIR/text-response.json"

# 图片请求（识别图片主色）
python3 $WORK_ROOT/scripts/http_request.py \
    --image $WORK_ROOT/scripts/example.png \
    --text "这张图片的主要颜色是什么？" \
    --output "$RUN_DIR/image-response.json"

# 保存完整服务日志
sudo docker logs "$CONTAINER" > "$RUN_DIR/server-full.log" 2>&1
```

图片请求会把 PNG 转为 Base64 Data URL，通过 `/v1/chat/completions` 接口发送。

**预期：** 文字请求返回正确答案，图片请求回答"主要颜色是红色"，token usage 非零。

---

## 六、性能跑分

### Step 09：停服务，回开发容器

```bash
sudo docker stop minicpm-1b-workshop1018-test
sudo docker exec -it flags-workshop1018-test /bin/bash
```

停掉服务容器，释放 CPU 资源后再跑分，避免干扰。

### Step 10：三路对比测试

```bash
export E=/work/evidence/workshop-run01
export VLLM_PLUGINS=""

bash /work/scripts/run_bench.sh stock   "$E/stock.json"   --repeats 0
bash /work/scripts/run_bench.sh minimal "$E/minimal.json" --repeats 5
bash /work/scripts/run_bench.sh flagos  "$E/flags.json"   --repeats 5
```

`--repeats 5` 表示 1 次预热 + 5 次测量取中位数，消除偶然误差。

Stock 会直接报 `ValidationError`（符合预期），Minimal 和 FlagOS 会打印测量数据。

### Step 11：生成对比报告

```bash
bash /work/scripts/dev_python.sh /work/scripts/compare_bench.py \
    "$E/minimal.json" "$E/flags.json" --output "$E/comparison.json"
```

**预期输出 `COMPARISON_PASS`。**

---

## 七、实验结果

### 吞吐量对比

| 路线 | PP512（tok/s） | TG128（tok/s） |
|---|---|---|
| Stock | 失败（ValidationError）| 失败 |
| Minimal | ~183 | ~11.7 |
| FlagOS | ~589–600 | ~21.9 |
| **FlagOS 相对提升** | **+223%** | **+87%** |

### 为什么 PP512 提升幅度更大？

PP512 测的是**算力**（Compute-bound），核心是矩阵乘法速度。FlagOS 通过以下方式大幅优化：
- GDN 按 head 并行并融合状态更新，核心热点从 1823ms 降至 246ms
- W8 复用预打包权重，`aten::copy_`（内存复制操作）从 12263 次降至 5338 次

TG128 测的是**访存带宽**（Memory-bound），每生成一个 token 都要读一遍权重，改进空间相对有限，但通过以下方式仍提升了 87%：
- 新增 W8A8 输出头：+11.78%
- Decode 阶段线程从 8 → 4（减少线程切换和 cache 争抢）：+19.57%

### 推理正确性验证

- ✅ 文字推理结果正确
- ✅ 图片推理正确（红色图片 → 答"红色"）
- ✅ token usage 非零（确认推理真实发生）

---

## 八、边界检查

```bash
exit  # 退出容器

python3 $WORK_ROOT/scripts/test_workshop_boundaries.py \
    --output $WORK_ROOT/evidence/boundaries-run01.json
```

检查端口占用、文件权限等 17 项边界条件，**预期输出 `BOUNDARIES 17 / 17`**。

---

## 九、异常处理规范

| 情况 | 正确做法 |
|---|---|
| pip 安装中断或 ABI 不匹配 | 把 `/work/envs` 移动重命名备份，再从 Step 05 重试，绝不直接覆盖 |
| 服务启动超时或端口冲突 | 查 `server.log` 和 `docker logs`，确认是本项目旧服务后 `docker stop`，重试时更换容器名和输出目录 |
| 跑分脚本中途中断 | 重试时必须改名输出文件，禁止拼接不同轮次的最好成绩 |

---

## 十、实验产出文件

实验结束后，以下文件记录了完整证据：

```
/work/evidence/workshop-run01/
├── stock.json          # Stock 路线结果（ValidationError）
├── minimal.json        # Minimal 路线跑分数据
├── flags.json          # FlagOS 路线跑分数据
└── comparison.json     # 对比结果（COMPARISON_PASS）

/work/evidence/http-run01/
├── text-response.json  # 文字推理响应
├── image-response.json # 图片推理响应
└── server-full.log     # 完整服务日志

/work/evidence/
└── boundaries-run01.json  # 边界检查结果
```

---

## 十一、关键结论

1. **未优化的 PyTorch 算子无法发挥 ARM64 潜力**：Minimal 路线的 183 tok/s 是"能跑但没优化"的基线，不代表硬件极限。
2. **底层编译优化效果显著**：FlagTree-CPU (MLIR/LLVM) 将算子编译为原生 Neon 指令，配合算子融合和线程调度优化，吞吐量提升 2–3 倍。
3. **线程数不是越多越好**：Decode 阶段从 8 线程改为 4 线程反而更快，因为减少了线程切换开销和 L2 cache 竞争。
4. **这条路线是纯 CPU 推理**：本实验完全运行在 8 个 A720 大核上，NPU 未参与。与 Phase 1–2 的 NPU 路线是互补关系，不是替代关系。

---

*本教程对应实验：Phase 3C，已完成。最后更新：2026-11。*

