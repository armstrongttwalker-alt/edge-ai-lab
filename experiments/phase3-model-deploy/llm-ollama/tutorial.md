# Phase 3E · Ollama 大模型管理 小白教程

> Ollama 是一个大模型管理平台，让"下载模型、启动服务、发送请求"变成三条命令的事。本教程说明如何在 ARM64 板子上安装 Ollama、运行模型，并对接 Open WebUI。
>
> **实验状态：⏳ 待执行**

---

## 一、实验目标

在 Radxa Orion O6N 上安装 Ollama，拉取 Qwen2.5 系列模型，启动推理服务，并通过 HTTP API 和 Open WebUI 访问本地大模型。

---

## 二、背景知识

### 2.1 Ollama vs llama.cpp

| 对比 | llama.cpp | Ollama |
|---|---|---|
| 定位 | 推理引擎（底层） | 模型管理平台（上层） |
| 使用方式 | 命令行，需要手动指定模型路径 | 类似 Docker，`ollama run qwen2.5` |
| 底层 | C++ 自研 | 底层也用 llama.cpp |
| API | 有，但需要自己启动 | 内置 OpenAI 兼容 API |
| 模型库 | 手动下载 | 类似 Docker Hub，自动拉取 |

Ollama 本质是 llama.cpp 的封装，更适合作为服务层，对接 Open WebUI 等前端。

### 2.2 Ollama 支持 ARM64 吗？

是的。Ollama 官方提供 `linux/arm64` 二进制，支持 Cortex-A 系列 CPU 的 Neon 指令集加速。

---

## 三、安装 Ollama

### 方案一：官方脚本

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

如果网络慢，改用：

```bash
# 手动下载 ARM64 版本
wget https://github.com/ollama/ollama/releases/latest/download/ollama-linux-arm64 \
    -O /usr/local/bin/ollama
chmod +x /usr/local/bin/ollama
ollama --version
```

### 方案二：Docker

```bash
docker run -d \
    --name ollama \
    -p 11434:11434 \
    -v /opt/edge-ai-lab/models/ollama:/root/.ollama \
    ollama/ollama:latest
```

---

## 四、运行模型

### 4.1 启动服务

```bash
ollama serve &
# 或作为系统服务：sudo systemctl start ollama
```

### 4.2 拉取并运行模型

```bash
# 拉取 1.5B 模型（约 1.0 GB）
ollama pull qwen2.5:1.5b

# 交互式对话
ollama run qwen2.5:1.5b "你好，介绍一下自己"
```

Ollama 模型名格式：`<模型名>:<版本/尺寸>`，常用：
- `qwen2.5:1.5b` → Qwen2.5 1.5B（默认 Q4 量化）
- `qwen2.5:3b`
- `qwen2.5:7b`
- `llama3.2:3b`

### 4.3 通过 API 访问

```bash
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5:1.5b",
        "messages": [{"role": "user", "content": "你好"}]
    }'
```

Ollama 实现了 OpenAI 兼容 API，这意味着任何支持 OpenAI API 的客户端（包括 Open WebUI）都可以直接连接。

---

## 五、对接 Open WebUI

```bash
docker run -d \
    --name open-webui \
    -p 3000:8080 \
    -e OLLAMA_BASE_URL=http://host-gateway:11434 \
    --add-host=host-gateway:host-gateway \
    -v /opt/edge-ai-lab/services/open-webui/data:/app/backend/data \
    ghcr.io/open-webui/open-webui:main
```

然后访问 `http://<板端IP>:3000`，在设置中选择 Ollama 作为后端，即可通过浏览器与本地模型对话。

---

## 六、性能测量

```bash
# 测量推理速度
ollama run qwen2.5:1.5b --verbose "写一首关于边缘计算的诗" 2>&1 | grep -E "eval|prompt"
```

输出包含：
- `prompt eval rate`：Prefill 速度（tok/s）
- `eval rate`：Token Generation 速度（tok/s）

---

## 七、验收标准

- [ ] Ollama 安装成功（`ollama --version`）
- [ ] `qwen2.5:1.5b` 模型拉取完成
- [ ] `ollama run` 能正常对话
- [ ] HTTP API 返回正确响应
- [ ] Open WebUI 可通过 Ollama 访问本地模型

---

*本教程对应实验：Phase 3E，待执行。最后更新：2026-11-11。*
