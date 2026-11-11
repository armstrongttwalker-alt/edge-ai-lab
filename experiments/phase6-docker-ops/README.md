# Phase 6 · Docker 部署与实验室运维 ⏳

**状态：未开始（依赖 Phase 4–5）**

## 实验目标

将核心 AI 服务容器化，建立统一的 Docker Compose 管理方案、监控体系与自动启动机制。

## 部署服务优先级

| 服务 | 镜像 | 端口 | 优先级 |
|---|---|---|---|
| Open WebUI | `ghcr.io/open-webui/open-webui:main` | 3000 | 高 |
| FastAPI 推理网关 | 本地构建 | 8000 | 高 |
| Qdrant 向量库 | `qdrant/qdrant` | 6333 | 中 |
| Prometheus | `prom/prometheus` | 9090 | 中 |
| Grafana | `grafana/grafana` | 3100 | 中 |
| n8n 工作流 | `n8nio/n8n` | 5678 | 低 |
| PostgreSQL | `postgres:16` | 5432 | 低 |

## 目录结构

```
/opt/edge-ai-lab/services/
├── open-webui/compose.yaml
├── api/
│   ├── Dockerfile
│   └── compose.yaml
├── qdrant/compose.yaml
└── monitoring/
    ├── compose.yaml
    ├── prometheus.yml
    └── grafana/
```

## Open WebUI 检查

在部署前确认 ARM64 支持：

```bash
docker buildx imagetools inspect ghcr.io/open-webui/open-webui:main | grep -i arm
```

## NPU 容器化注意事项

将 NPU 服务容器化时，须映射设备节点并挂载运行时库：

```yaml
devices:
  - /dev/aipu:/dev/aipu
volumes:
  - /usr/share/cix/lib:/usr/share/cix/lib:ro
```

具体映射方式须在宿主机验证通过 NPU 推理后再研究。

## 监控指标

- CPU / GPU / NPU 利用率
- 内存与存储占用
- 推理服务 QPS 与延迟
- 板端温度与 CPU 频率（throttling 检测）
