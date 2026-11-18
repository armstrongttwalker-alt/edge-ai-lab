# NPU 跑嵌入模型的尝试，与 CIX P1 能力边界结论

**日期：** 2026-11-18
**背景：** 大语料 RAG（1287 篇真实文档）在纯 CPU 上用 bge-m3 建索引太慢
（1.6s/块、批处理无效、全量约 6 小时），尝试把嵌入放到 Zhouyi NPU 上加速。

## 结论先行

**CIX P1 / Zhouyi V3 适合做"边缘机头"——规整的 CNN 视觉负载——不适合跑大
Transformer 嵌入这类"重任务"。** 板子定位：NPU 视觉 + 轻量 LLM 服务 + 前端网关；
大语料嵌入建索引、大模型推理应放后端服务器 / GPU。本结论有下方实测数据支撑。

## 实测过程与数据

嵌入模型：bge-m3（BAAI，568M 参数 Transformer，ONNX，输出 sentence_embedding，1024 维）。

### 1. FP32 原图直接上 NPU → 整图被拒

`ZhouyiExecutionProvider` 加载 bge-m3 FP32，**1236 个节点被拒**，压倒性原因：
`GetOnnxTensorInfo Cannot get shape`。根因是模型用**动态形状**（`batch_size`、
`sequence_length` 是符号维），而 NPU 需要**静态形状**（对照：Phase 1 的 CNN 输入是
固定 `[1,3,224,224]` 才能上 NPU）。另有少量算子硬不支持：Gather（词嵌入查表）、
Shape、Where、Concat、动态 Slice。

### 2. 固定形状 [1,512] 后 → 只剩 FP32 MatMul 被拒

用 `onnxruntime.tools.symbolic_shape_infer` 把输入固定为 `[1,512]`：
- `Cannot get shape` 拒绝：**1236 → 0**（动态形状问题解决）
- 仅剩 **197 个节点**不支持，其中 **192 个是 MatMul**，原因全是
  `BATCH_MATMUL ... doesn't support operand type: FLOAT32`——**NPU 是 INT8 加速器，
  不吃 FP32 矩阵乘**（与 MobileNetV2 需 INT8 同理）。

### 3. INT8 QDQ 量化（per-tensor）后 → 算子过关，但质量崩塌

- per-channel 量化报错（bge-m3 图 opset < 13 不支持），改用 per-tensor。
- 量化耗时 951s，INT8 模型 1362 MB（FP32 2.2GB）。
- MatMul 的 FP32 拒绝消失（NPU 肯收了）。**但：**

| 指标 | 结果 | 含义 |
|---|---|---|
| **INT8 vs FP32 嵌入余弦相似度** | **0.51**（5 条文本均值） | ❌ 质量崩塌。嵌入需 ≥0.98 才能保检索；0.51 等于向量基本不相关 |
| 固定 512 后单次嵌入 | ~6000 ms | 比动态形状的 1.6s 更慢（NPU 要求每条按满 512 算） |
| INT8 @CPU vs FP32 @CPU | 6027 vs 6119 ms | INT8 在 CPU 上并不更快 |
| NPU 实际执行 | 崩溃 | mask/pooling 里的 INT32 Mul/Add/Abs 等小算子 NPU 编译不过 |

**质量崩塌根因（教科书级）：** per-tensor INT8 PTQ 扛不住 Transformer 注意力
激活的**离群值**。CNN 激活分布规整、per-tensor 量化无碍（故 MobileNetV2 成功）；
Transformer 激活动态范围极大，per-tensor 一刀切把信息削没。per-channel 只改善
**权重**量化，**激活**仍是 per-tensor，救不到根上，故未继续。

## 能力边界（本板实测汇总）

| 任务 | 本板表现 | 定位 |
|---|---|---|
| MobileNetV2 / YOLO（CNN 视觉） | NPU 8.6× / 17.8× 加速 | ✅ 机头核心负载 |
| 轻量 LLM 服务（≤3B） | 可交互 | ✅ 轻量前端 |
| 大 Transformer 嵌入（bge-m3） | 勉强上 NPU，INT8 余弦 0.51 | ❌ 重任务 |
| 大语料 RAG 建索引 | CPU 1.6s/块，全量约 6h | ❌ 重任务（应放后端） |
| 3B 长列表 / 长综合归纳 | 容量到顶，丢项 | ❌ 偏重任务 |

## 对 RAG 架构的建议

- **板上**：维持小语料的可用 RAG 演示（FlagOS overview/whats_new），或只做视觉 / 前端网关。
- **大语料（上千篇）**：在后端服务器 / GPU 上建好 Qdrant 索引，板子只做查询前端。
- 若坚持板上做嵌入：换**小 FP32 嵌入模型**（bge-small / e5-small，约 100M，避免量化损失）
  比 INT8 量化 bge-m3 更现实，质量用本目录评测工具量化对比。

## 复现要点

- 固定形状：`SymbolicShapeInference` + 把输入 dim 设为 `[1,512]`，2.2GB 模型须
  `save_as_external_data=True`（否则 protobuf 超 2GB 序列化失败）。
- 量化：`quantize_static(..., quant_format=QDQ, per_channel=False, op_types_to_quantize=["MatMul"], use_external_data_format=True)`。
- NPU 工作目录需 `operator` 软链 + `LD_LIBRARY_PATH=/usr/share/cix/lib:/usr/share/cix/lib/onnxruntime`。
