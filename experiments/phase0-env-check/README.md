# Phase 0 · 基础环境检查 ✅

**状态：已完成**

## 验证结果摘要

| 项目 | 结果 |
|---|---|
| OS | Debian 12 |
| Kernel | 6.6.89-3-sky1 |
| CPU | ARM64 · 12 核（8× A720 + 4× A520） |
| RAM | 32 GB |
| NVMe | 1 TB |
| Docker | 已安装 |
| NPU 驱动 | `aipu 98304 0`（已加载） |
| NPU 识别 | Zhouyi V3，1 Cluster × 3 Core |
| NPU 设备节点 | `/dev/aipu`（crw-rw-rw-） |
| CIX 用户态库 | `libnoe.so`、`libonnxruntime.so` |

## 验证命令

```bash
# 系统
uname -a && cat /etc/os-release

# 硬件
lscpu && free -h && df -h

# NPU 驱动
lsmod | grep aipu
dmesg | grep -i aipu

# 设备节点
ls -l /dev/aipu
cat /sys/class/misc/aipu/dev

# CIX 软件包
dpkg -l | grep -Ei 'cix|noe|npu'
dpkg -L cix-noe-umd
dpkg -L cix-npu-onnxruntime

# Docker
docker info --format '{{.Architecture}} {{.ServerVersion}}'
```

详细进度见 [PROGRESS.md](../../PROGRESS.md)。
