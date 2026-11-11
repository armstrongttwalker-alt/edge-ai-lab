# Phase 6 · Docker 部署与实验室运维 小白教程

> 本教程说明如何把所有 AI 服务容器化，建立统一管理方案，并添加监控和自动启动。
>
> **实验状态：⏳ 待执行（依赖 Phase 4/5）**

---

## 一、实验目标

将 Open WebUI、FastAPI 网关、Qdrant 向量库统一用 Docker Compose 管理，添加 Prometheus + Grafana 监控，实现开机自动启动。

---

## 二、背景知识

### 2.1 为什么要容器化？

如果直接在宿主机运行所有服务：
- 重装系统或换板子时，需要重新配置所有环境
- 不同服务的 Python 依赖可能冲突
- 没有统一的日志管理

Docker 容器化的好处：
- **隔离**：每个服务有独立环境
- **可移植**：`docker compose up` 一键启动所有服务
- **可重复**：用 `compose.yaml` 记录所有配置，版本控制
- **可观测**：统一日志、监控

### 2.2 NPU 容器化的特殊注意

把 NPU 推理服务放进容器，需要额外配置：

```yaml
services:
  npu-service:
    devices:
      - /dev/aipu:/dev/aipu      # 映射 NPU 设备节点
    volumes:
      - /usr/share/cix/lib:/usr/share/cix/lib:ro  # 挂载 NOE 运行库
    environment:
      - LD_LIBRARY_PATH=/usr/share/cix/lib:/usr/share/cix/lib/onnxruntime
```

**重要：** NPU 容器化必须在宿主机验证通过 NPU 推理之后再尝试（已完成，Phase 1）。

---

## 三、服务规划

### 优先级一：核心服务

```yaml
# /opt/edge-ai-lab/services/compose.yaml
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports: ["3000:8080"]
    volumes:
      - ./open-webui/data:/app/backend/data
    restart: unless-stopped

  api-gateway:
    build: ./api
    ports: ["8000:8000"]
    network_mode: host
    restart: unless-stopped
```

### 优先级二：数据服务

```yaml
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes:
      - ./qdrant/storage:/qdrant/storage
    restart: unless-stopped
```

### 优先级三：监控

```yaml
  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports: ["3100:3000"]
    volumes:
      - ./monitoring/grafana:/var/lib/grafana
    restart: unless-stopped
```

---

## 四、执行步骤

### 4.1 检查镜像 ARM64 支持

```bash
# 检查每个镜像是否有 arm64 版本
docker buildx imagetools inspect ghcr.io/open-webui/open-webui:main | grep -i arm
docker buildx imagetools inspect qdrant/qdrant:latest | grep -i arm
docker buildx imagetools inspect prom/prometheus:latest | grep -i arm
```

### 4.2 启动所有服务

```bash
cd /opt/edge-ai-lab/services
docker compose up -d
docker compose ps
docker compose logs -f
```

### 4.3 验证服务可访问

```bash
# Open WebUI
curl http://localhost:3000

# API Gateway
curl http://localhost:8000/health

# Qdrant
curl http://localhost:6333/

# Prometheus
curl http://localhost:9090/-/healthy
```

### 4.4 配置开机自启

Docker Compose 服务已经有 `restart: unless-stopped`，只需确保 Docker 守护进程开机自启：

```bash
sudo systemctl enable docker
sudo systemctl status docker
```

---

## 五、监控指标

在 Grafana 中配置以下监控面板：

| 指标 | 来源 | 意义 |
|---|---|---|
| CPU 使用率 | Node Exporter | 推理时 CPU 负载 |
| 内存使用率 | Node Exporter | 防止 OOM |
| GPU/NPU 利用率 | 自定义 Exporter | NPU 是否被充分利用 |
| API 请求延迟 | FastAPI metrics | 服务响应速度 |
| API 错误率 | FastAPI metrics | 服务健康状态 |
| 板端温度 | Node Exporter | 防过热降频 |

```bash
# 安装 Node Exporter
docker run -d \
    --name node-exporter \
    --net=host \
    -v /:/host:ro \
    prom/node-exporter:latest \
    --path.rootfs=/host
```

---

## 六、验收标准

- [ ] 所有核心服务通过 Docker Compose 启动
- [ ] Open WebUI 可通过浏览器访问并对话
- [ ] API Gateway 健康检查通过
- [ ] Qdrant 向量库可访问
- [ ] Grafana 监控面板展示基础指标
- [ ] 重启板子后服务自动恢复

---

*本教程对应实验：Phase 6，待执行。最后更新：2026-11-11。*
