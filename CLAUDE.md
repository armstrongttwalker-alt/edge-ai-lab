# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo is a Windows-side workspace (`E:\BAAI\AI-LAB`) for the **CIX P1 Edge AI Lab** — a learning and research platform built on a **Radxa Orion O6N** single-board computer accessed over SSH. All code and experiments run on the remote device; this directory holds documentation, saved session notes, and scripts for the lab.

The project goal is to systematically validate and benchmark heterogeneous compute (CPU / GPU / NPU) on the CIX P1 chip, then build progressively higher-level AI services on top.

## Target Hardware (Remote Device)

| Item | Value |
|---|---|
| Board | Radxa Orion O6N |
| SoC | CIX P1 |
| OS | Debian 12 |
| Kernel | 6.6.89-3-sky1 |
| CPU | ARM64, 12 cores |
| RAM | 32 GB |
| Storage | 1 TB NVMe |
| NPU | Zhouyi V3 (1 cluster × 3 cores) |
| Python | 3.11 |
| Docker | Installed |

## Remote Python Environment

```bash
# Activate the lab venv (system-site-packages enabled)
source /opt/edge-ai-lab/venv/bin/activate

# Install NumPy (network may time out — use a mirror or offline wheel)
pip install "numpy>=1.21.6,<2"

# Install vendor wheels (order matters)
pip install libnoe-2.0.0-py3-none-manylinux2014_aarch64.whl
pip install NOE_Engine-2.0.0-py3-none-manylinux2014_aarch64.whl
pip install ZhouyiOperators_x2-25.4.23-py3-none-any.whl
pip install onnxruntime_zhouyi-1.20.0-cp311-cp311-linux_aarch64.whl
```

**Always use `onnxruntime_zhouyi` — not PyPI `onnxruntime`.** The vendor wheel includes the CIX NPU Execution Provider; the upstream package does not.

## NPU Software Stack

```
Python app
  └─ ONNX Runtime (onnxruntime_zhouyi)
       └─ CIX NPU Execution Provider
            └─ CIX NOE Runtime  (libnoe.so)
                 └─ Linux aipu driver  (/dev/aipu)
                      └─ Zhouyi V3 NPU
```

Key installed artifacts on the device:

```
/usr/share/cix/lib/libnoe.so
/usr/share/cix/lib/onnxruntime/libonnxruntime.so
/usr/share/cix/include/npu/cix_noe_standard_api.h
```

Verify driver and device node:

```bash
lsmod | grep aipu                  # expect: aipu 98304 0
ls -l /dev/aipu                    # expect: crw-rw-rw-
dmesg | grep -i aipu               # expect: AIPU detected: zhouyi-v3
```

## Project Directory on Device

```
/opt/edge-ai-lab/
├── venv/           # Python environment
├── models/
│   ├── mobilenetv2/
│   ├── yolo/
│   └── llm/
├── datasets/
├── experiments/
│   ├── npu/
│   ├── cpu/
│   └── gpu/
├── benchmarks/
├── logs/
└── services/
```

## Project Phases

| Phase | Goal | Status |
|---|---|---|
| 0 | Base environment check | ✅ Done |
| 1 | NPU driver + runtime validation | 🔄 In progress (Step 4/7) |
| 2 | CPU / GPU / NPU benchmark | Not started |
| 3 | Model deployment (MobileNetV2 → YOLO → LLM) | Not started |
| 4 | FastAPI inference service (OpenAI-compatible) | Not started |
| 5 | Document RAG (vector DB, local QA) | Not started |
| 6 | Docker deployment + monitoring | Not started |

## Current Blocker (Phase 1, Step 4/7)

NumPy download times out on the device. Resolution options:
- Use a domestic PyPI mirror: `pip install -i https://mirrors.aliyun.com/pypi/simple/ "numpy>=1.21.6,<2"`
- Download the ARM64 wheel on this Windows machine and `scp` it to the device

After NumPy installs, install the four vendor wheels above in order, then continue with Step 5 (verify NPU Execution Provider registers) and Step 6 (MobileNetV2 INT8 inference).

## NPU Inference Validation Checklist (Step 5–7)

A successful NPU inference requires all of the following, not just a result returned:

1. `onnxruntime_zhouyi` imported without error
2. `CIXExecutionProvider` appears in `ort.get_available_providers()`
3. Session created with `providers=["CIXExecutionProvider"]`
4. Model graph nodes assigned to NPU (not all falling back to CPU)
5. `dmesg` or NOE logs show NPU received work
6. Inference result is numerically correct

If ONNX Runtime silently falls back to `CPUExecutionProvider`, the NPU experiment has not passed — that must be explicitly ruled out.

## Experiment Protocol

Each experiment in this lab follows this structure:
1. Goal — what problem this solves
2. Background — hardware/software concepts involved
3. Commands — each flag explained
4. Expected output — what normal and abnormal results look like
5. Pass criteria — explicit evidence the hardware was used

Do not proceed to the next step until the current step's pass criteria are met.
