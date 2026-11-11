#!/bin/bash
# Phase 1 NPU Bring-up 完整验证脚本
# 用法：cd /opt/edge-ai-lab/npu-workspace && bash ../experiments/npu/validate_npu.sh
# 2026-11-11 验证通过

VENV=/opt/edge-ai-lab/venv
WORK=/opt/edge-ai-lab/npu-workspace
DATASETS=$VENV/lib/python3.11/site-packages/onnxruntime/datasets

echo "===== Phase 1: NPU Bring-up Validation ====="
echo ""

# Step 1: Driver
echo "[Step 2] NPU Driver"
lsmod | grep aipu && echo "PASS: aipu module loaded" || echo "FAIL: aipu not loaded"

# Step 3: Device node
echo ""
echo "[Step 3] Device node"
ls -l /dev/aipu && echo "PASS: /dev/aipu exists" || echo "FAIL: /dev/aipu missing"

# Step 4-5: Python packages
echo ""
echo "[Step 4-5] Python packages"
source $VENV/bin/activate
python3 -c "
import onnxruntime as ort
import numpy as np
print('numpy:', np.__version__)
print('onnxruntime:', ort.__version__)
providers = ort.get_available_providers()
print('providers:', providers)
assert 'ZhouyiExecutionProvider' in providers, 'FAIL: ZhouyiEP not registered'
print('PASS: ZhouyiExecutionProvider registered')
" 2>/dev/null

# Step 6-7: Inference with ZhouyiEP
echo ""
echo "[Step 6-7] NPU Inference"
mkdir -p $WORK
ln -sfn /usr/share/cix/lib/onnxruntime/operator $WORK/operator 2>/dev/null
cd $WORK

LD_LIBRARY_PATH=/usr/share/cix/lib:/usr/share/cix/lib/onnxruntime:$LD_LIBRARY_PATH \
python3 -c "
import onnxruntime as ort, numpy as np, os, time
os.chdir('/opt/edge-ai-lab/npu-workspace')
DATASETS='$DATASETS'
opts = ort.SessionOptions()
opts.log_severity_level = 3
for name, path, shape in [('sigmoid', f'{DATASETS}/sigmoid.onnx', [3,4,5]),
                            ('mul_1',  f'{DATASETS}/mul_1.onnx',   [3,2])]:
    sess = ort.InferenceSession(path, sess_options=opts,
           providers=['ZhouyiExecutionProvider','CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    dummy = np.random.rand(*shape).astype(np.float32)
    sess.run(None, {inp.name: dummy})  # warmup
    t0 = time.time()
    for _ in range(100): sess.run(None, {inp.name: dummy})
    lat = (time.time()-t0)/100*1000
    print(f'PASS: {name} on ZhouyiEP, avg={lat:.3f}ms')
" 2>/dev/null

echo ""
echo "===== Validation Complete ====="
