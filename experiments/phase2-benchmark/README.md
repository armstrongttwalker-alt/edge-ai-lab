# Phase 2 · CPU / GPU / NPU 基准测试 ⏳

**状态：未开始（依赖 Phase 1 完成）**

## 实验目标

在统一条件下对比三种算力的推理性能，输出可重复的性能报告。

## 基准配置

- **模型：** MobileNetV2 INT8，输入 224×224
- **重复次数：** 1 次预热 + 100 次测量取中位数
- **固定变量：** batch size=1，单线程（CPU 测试时对比多线程）

## 测试矩阵

| 精度 | CPU EP | Mali GPU | Zhouyi NPU |
|---|---|---|---|
| FP32 | ✅ | 视驱动 | — |
| FP16 | ✅ | 视驱动 | 视算子 |
| INT8 | ✅ | 视后端 | ✅ 优先 |

## 记录指标

| 指标 | 单位 |
|---|---|
| 平均延迟 | ms |
| P50 / P95 延迟 | ms |
| 吞吐量 | FPS |
| 内存占用 | MB |
| 温度（峰值） | °C |

## 注意事项

- GPU 路线需先单独验证 Mali G720 OpenCL 计算后端可用性
- 不能以"OpenCL 支持"推断 ONNX GPU EP 可直接调用
- 每次测试记录完整环境快照（驱动版本、CPU 频率、当前温度）

## 结果存放

```
experiments/phase2-benchmark/
├── results/
│   ├── cpu-fp32.json
│   ├── cpu-int8.json
│   ├── gpu-*.json
│   └── npu-int8.json
└── report.md
```
