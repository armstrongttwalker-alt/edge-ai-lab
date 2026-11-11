收到！这次我将**保持最高级别的详细程度**，同时**完整融入 PPT 的宏观架构、核心参数以及底层技术细节**。

所有命令已严格处理为单行，方便直接复制。没有任何缩减，解释将深入到操作系统、AI 引擎和底层算子层面。

---

# 📘 端侧大模型推理适配与性能对比 实验手册（全量深度解析版）

## 🧭 第一章：核心架构与底层原理详解

在开始实操前，必须理解本节实验的软硬件架构。这不仅是一次简单的跑分，而是涉及 AI 引擎底层调度的深度适配。

### 1.1 模型与量化契约 (Model & Quantization Contract)
*   **MiniCPM-V 4.7**：这是一个包含视觉编码器和语言骨干的多模态模型，参数量为 1B。
*   **混合精度设计**：为了兼顾速度与功能，视觉模块使用 BF16（保留高精度的图片理解与 OCR 能力），而语言骨干层的 Linear 和输出头则被转换为 **W8A8**（INT8 权重，动态非对称逐 Token 激活）。测试上下文长度被限制为 4096 token，CPU KV-cache 为 256 MiB。

### 1.2 完整推理架构 (FlagOS on CIX P1)
*   **执行流**：客户端发送文字/图片请求 \(\rightarrow\) **vLLM 0.24** (负责请求调度、采样、内存管理) \(\rightarrow\) **vLLM-plugin-FL** (负责模型注册与量化接入) \(\rightarrow\) **FlagGems** (提供底层算子)。
*   **底层编译**：代码通过普通 CPU Triton 或 CPU TLE Raw（自有 C++/Neon 汇编例程），最终交由 **FlagTree-CPU** (MLIR/LLVM 编译器) 转换为机器码，在 CIX P1 芯片的 8 个 A720 核心上执行。默认利用 W8 的 Neon SDOT 指令集。

### 1.3 软硬件版本锁定 (Environment Lock)
*   **vLLM**: 0.24.0+CPU (调度与 Runner)
*   **PyTorch / Transformers**: 2.11.0 / 5.15.1
*   **vLLM-plugin-FL**: commit f0de70233590
*   **FlagGems**: commit b5f5e110c7c2
*   **FlagTree-CPU**: commit 36133df9540c

### 1.4 三种测试路线的本质区别
*   **Stock (纯开箱)**：官方 vLLM 0.24 未注册 MiniCPM-V 4.7 架构，直接抛出 `ValidationError`。
*   **Minimal (最小适配)**：打上注册补丁，引擎能加载模型，但底层仍走 PyTorch 原生的、未针对 ARM 优化的通用算子，吞吐量低。
*   **FlagOS (优化版)**：拦截并替换底层算子。使用 Triton 或 TLE Raw 汇编，利用 ARM Neon 指令集加速 W8 GEMM 运算，同时优化了线程调度（Decode 阶段调为 4 线程减少争抢）。

### 1.5 Docker 容器分工与性能指标
*   **开发容器**：主进程 `sleep infinity`，用于挂载代码、安装 wheel、跑跑分脚本。
*   **服务容器**：运行 `vllm serve`，用于接收 HTTP 请求。
*   **指标**：**PP512** (预填充，处理 512 输入 token，测算力)；**TG128** (解码，生成 128 输出 token，测访存)。

---

## 🟢 第二章：主机环境校验与准备

*当前在主机环境（`baai@RadxaOrionO6N:~$`）。*

### Step 01：设定工作目录与加载环境
*   **目的**：加载预置脚本的锁定清单与路径。
*   **命令**：
    ```bash
    ls -d /home/baai/workshop1018
    ```
    ```bash
    export WORK_ROOT=/home/baai/workshop1018
    ```
    ```bash
    cd $WORK_ROOT
    ```
    ```bash
    source $WORK_ROOT/scripts/activate_host.env
    ```
*   **原理解析**：`export` 将路径存入环境变量。`source` 执行脚本，该脚本内部设定了 `CIX_IMAGE_REF`（基础镜像地址）和 `WORK_ROOT`。
*   **预期输出**：执行 `source` 后，可能输出 `Host paths ready...`。

### Step 02：校验材料与镜像身份
*   **目的**：核验模型文件、wheel 包哈希以及 Docker 镜像 ID 是否与锁定清单一致。
*   **命令**：
    ```bash
    bash $WORK_ROOT/scripts/check_preloaded.sh
    ```
    ```bash
    bash $WORK_ROOT/scripts/check_prebuilt.sh
    ```
*   **原理解析**：执行校验脚本，比对文件的 SHA256 哈希值。如果不一致，说明环境被污染，必须停止。
*   **预期输出**：必须输出 **`PRELOADED_CHECK_PASS`** 和 **`PREBUILT_CHECK_PASS`**。

---

## 🔵 第三章：开发容器构建与算子安装

*为避免污染基础镜像，我们将克隆一份工作目录。*

### Step 03：准备专属测试目录
*   **目的**：清理镜像自带的旧 env 目录，防止环境冲突。
*   **命令**：
    ```bash
    cp -r /home/baai/workshop1018 /home/baai/workshop1018-test
    ```
    ```bash
    rm -rf /home/baai/workshop1018-test/envs
    ```
    ```bash
    export WORK_ROOT=/home/baai/workshop1018-test
    ```
    ```bash
    cd $WORK_ROOT
    ```
    ```bash
    source $WORK_ROOT/scripts/activate_host.env
    ```
*   **原理解析**：基础镜像内可能自带初始化的 `/work/envs/native` 前缀，导致后续执行 `prepare_env.py` 时抛出 `FileExistsError`。`rm -rf` 强制删除该目录，确保从零开始构建。

### Step 04：创建并进入开发容器
*   **目的**：启动开发容器（沙盒环境）。
*   **命令**：
    ```bash
    CONTAINER=flags-workshop1018-test bash $WORK_ROOT/scripts/init_container.sh
    ```
    ```bash
    sudo docker exec -it flags-workshop1018-test /bin/bash
    ```
*   **原理解析**：通过 `docker exec -it` 打开容器伪终端。此后的所有命令均在容器内部执行，提示符变为 `root@...`。

### Step 05：离线安装预编译 Wheel
*   **目的**：将 FlagTree (编译器)、FlagGems (算子库)、Plugin (接入层) 安装到隔离的 Python 环境。
*   **命令**：
    ```bash
    /opt/flagos-runtime/bin/flagos-python --profile 1b /work/scripts/prepare_env.py
    ```
    ```bash
    bash /work/scripts/dev_python.sh /work/scripts/prebuilt_tools.py preflight /work
    ```
    ```bash
    bash /work/scripts/dev_python.sh -m pip install --no-index --no-deps --no-compile --target /work/envs/dev/lib/python3.11/site-packages /work/prebuilt/wheels/*.whl
    ```
    ```bash
    bash /work/scripts/dev_python.sh /work/scripts/prebuilt_tools.py installed /work
    ```
*   **原理解析**：
    *   `--no-index`：禁止 pip 联网，只查本地。
    *   `--no-deps`：不自动替换镜像自带的锁定版 PyTorch 等依赖。
    *   `--target`：将包安装到 `/work/envs/dev`，与基础镜像环境隔离。
*   **预期输出**：第四步输出 **`PREBUILT_IMPORT_AND_RAW_LOAD_PASS`**。

### Step 06：执行正确性检查
*   **目的**：验证 CPU JIT 编译、8 组独立 W8 数值计算及多模态路由。
*   **命令**：
    ```bash
    bash /work/scripts/validate_source.sh /work/evidence/workshop-run01
    ```
*   **原理解析**：脚本会执行数值测试和功能测试（算术、中文、红色图片、OCR），并验证是否真实调用了 Raw/Triton，杜绝回退到默认算子。
*   **预期输出**：最后输出 **`SOURCE_VALIDATION_PASS`**。进度日志保存在 `/work/evidence/workshop-run01/`。

---

## 🚀 第四章：启动 HTTP 推理服务

### Step 07：退容器，启服务
*   **目的**：服务容器与开发容器分离，释放开发容器占用的资源。
*   **命令**：
    ```bash
    exit
    ```
    ```bash
    export CACHE_DIR="$WORK_ROOT/cache/http-run01"
    ```
    ```bash
    export RUN_DIR="$WORK_ROOT/evidence/http-run01"
    ```
    ```bash
    export CONTAINER=minicpm-1b-workshop1018-test
    ```
    ```bash
    bash $WORK_ROOT/scripts/start_service.sh
    ```
*   **原理解析**：启动 `vllm serve`，挂载模型（只读为 `/model`），分配 8 个 A720 核心，启用 `CPUPhaseWorker`。此过程极耗 CPU。
*   **预期输出**：不断刷 `curl: (7) Failed to connect` 是正常的轮询。直到打印 **`READY`**，说明 Uvicorn 服务已就绪。

### Step 08：发送 HTTP 请求与保存日志
*   **目的**：验证端到端的推理能力，并固化证据。
*   **命令**：
    ```bash
    python3 $WORK_ROOT/scripts/http_request.py --output "$RUN_DIR/text-response.json"
    ```
    ```bash
    python3 $WORK_ROOT/scripts/http_request.py --image $WORK_ROOT/scripts/example.png --text "这张图片的主要颜色是什么？" --output "$RUN_DIR/image-response.json"
    ```
    ```bash
    sudo docker logs "$CONTAINER" > "$RUN_DIR/server-full.log" 2>&1
    ```
*   **原理解析**：图片请求会将 PNG 转为 Base64 Data URL，发送至 `/v1/chat/completions`。最后一行命令将 Docker 内部的完整运行日志导出，用于事后分析。
*   **预期输出**：回答 `5` 及“主要颜色是红色”，并输出非零的 token usage。

---

## 📊 第五章：性能跑分与深度解析

### Step 09：停服务，重回开发容器
*   **目的**：确保跑分时 CPU 被测试独占，避免 I/O 和网络干扰。
*   **命令**：
    ```bash
    sudo docker stop minicpm-1b-workshop1018-test
    ```
    ```bash
    sudo docker exec -it flags-workshop1018-test /bin/bash
    ```
*   **原理解析**：服务容器停止后，释放 8000 端口和占用的 CPU 周期。

### Step 10：三路对比测试
*   **目的**：验证各版本引擎的执行效率。
*   **命令**：
    ```bash
    export E=/work/evidence/workshop-run01
    ```
    ```bash
    export VLLM_PLUGINS=""
    ```
    ```bash
    bash /work/scripts/run_bench.sh stock "$E/stock.json" --repeats 0
    ```
    ```bash
    bash /work/scripts/run_bench.sh minimal "$E/minimal.json" --repeats 5
    ```
    ```bash
    bash /work/scripts/run_bench.sh flagos "$E/flags.json" --repeats 5
    ```
*   **原理解析**：`--repeats 5` 表示进行 1 次预热、5 次测量取中位数，消除偶然误差。
*   **预期输出**：
    *   `stock` 抛出不支持该架构的 `ValidationError`。
    *   `minimal` 与 `flagos` 打印测量数据。

### Step 11：生成对比与深度解读
*   **目的**：计算性能提升幅度。
*   **命令**：
    ```bash
    bash /work/scripts/dev_python.sh /work/scripts/compare_bench.py "$E/minimal.json" "$E/flags.json" --output "$E/comparison.json"
    ```
*   **预期输出**：**`COMPARISON_PASS`**。
*   **技术深度解析 (参考 PPT 30-33)**：
    *   **PP512 飞跃**：最小适配基线约 183 tok/s，FlagOS 可达 589~600 tok/s。**原因**：FlagOS 通过 GDN 按 head 并行并融合状态更新，核心热点从 1823ms 降至 246ms；同时 W8 复用预打包权重，大幅降低了 `aten::copy_` 的调用次数（从 12263 次降至 5338 次）。
    *   **TG128 优化**：从约 11.7 tok/s 提升至 21.9 tok/s。**原因**：加入了 W8A8 输出头（带来 11.78% 增益），并且将 Decode 阶段的线程数从 8 调整为 4（减少线程切换开销，再获 19.57% 增益）。最终达成 PP 提升 223.76%，TG 提升 87.12%。

---

## 🏁 第六章：安全边界检查与异常恢复机制

### Step 12：执行边界检查
*   **目的**：确保实验结束前，系统端口、文件权限等 17 项边界条件处于安全状态。
*   **命令**：
    ```bash
    exit
    ```
    ```bash
    python3 $WORK_ROOT/scripts/test_workshop_boundaries.py --output $WORK_ROOT/evidence/boundaries-run01.json
    ```
*   **预期输出**：**`BOUNDARIES 17 / 17`**。

### 🛠️ 附录：异常排查与重试规范 (参考 PPT 附录 D1-D3)
如果实验中途失败，遵循以下原则重试：
1.  **材料安装失败 (D1)**：若 pip 安装中断或 ABI 不匹配，**绝对不能直接覆盖**。必须先将 `/work/envs` 移动重命名，归档为备份（如 `mv /work/envs /work/evidence/pip-retry-时间戳`），再重新执行 Step 05。
2.  **容器或端口冲突 (D2)**：如果服务启动超时或端口被占，先查 `server.log` 和 `docker logs`。确认属于本项目的旧服务后，使用 `sudo docker stop` 停掉。重试时，必须更换新的容器名和输出目录名。
3.  **性能测量中断 (D3)**：如果 HTTP 请求或跑分脚本中途中断，**拒绝同名重试**（防覆盖）。重试输出文件必须改名，且比较工具拒绝处理残缺数据，不能拼接不同轮次的最好成绩。