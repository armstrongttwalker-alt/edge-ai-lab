# -*- coding: utf-8 -*-
"""RAG 评测：每题同时测【检索召回】与【生成覆盖】，把失分定位到检索层还是模型层。

用法（在板端）：
    source /opt/edge-ai-lab/venv/bin/activate
    export LD_LIBRARY_PATH=/usr/share/cix/lib:/usr/share/cix/lib/onnxruntime:$LD_LIBRARY_PATH
    python3 -I eval_rag.py

前提：网关(:8000) 与 llama-server(:8081) 已启动，Qdrant(:6333) 已建好 collection。
设计说明见同目录 README.md。
"""
import json, time, urllib.request
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort
from qdrant_client import QdrantClient

EMBED_BASE = "/opt/edge-ai-lab/models/models/BAAI--bge-m3/snapshots/master/onnx"
COLL = "edge_ai_rag_bge"
GATEWAY = "http://localhost:8000/v1/chat/completions"

# ===== 评测集：q=问题  ents=检索应召回的实体  facts=回答必须命中的点(每点可多写法) =====
EVAL = [
 # --- A 结构/总览（难）---
 {"id":"A1","q":"FlagOS 有哪些核心库(core libraries)?","lang":"zh",
  "ents":["FlagGems","FlagTree","FlagScale","FlagCX","Operator librar"],
  "facts":[["FlagGems"],["FlagTree"],["FlagScale"],["FlagCX"],["operator","算子","操作"]]},
 {"id":"A2","q":"FlagOS 的多领域算子库(multi-domain operator libraries)包括哪些?","lang":"zh",
  "ents":["FlagDNN","FlagBLAS","FlagFFT","FlagSparse","FlagTensor","FlagAttention","FlagAudio"],
  "facts":[["FlagDNN"],["FlagBLAS"],["FlagFFT"],["FlagSparse"],["FlagTensor"],["FlagAttention"],["FlagAudio"]]},
 {"id":"A3","q":"FlagOS 2.2 由哪几大部分组成?","lang":"zh",
  "ents":["core librar","Plugins","Domain-specific","Developer tools","Platform services"],
  "facts":[["core librar","核心库"],["plugin","插件"],["domain-specific","领域"],
           ["developer tool","开发者工具"],["platform service","平台服务"]]},
 # --- B 分类清单（中）---
 {"id":"B1","q":"FlagOS 有哪些插件(plugins)?","lang":"zh",
  "ents":["vllm-plugin-FL","sglang-plugin-FL","Torch-FL","Megatron-LM-FL","TransformerEngine-FL","verl-FL"],
  "facts":[["vllm-plugin-FL"],["sglang-plugin-FL"],["Torch-FL"],["Megatron-LM-FL"],["TransformerEngine-FL"],["verl-FL"]]},
 {"id":"B2","q":"FlagOS 的平台服务(platform services)有哪些?","lang":"zh",
  "ents":["FlagRelease","FlagPerf","FlagCICD","KernelGenBench"],
  "facts":[["FlagRelease"],["FlagPerf"],["FlagCICD"],["KernelGenBench"]]},
 {"id":"B3","q":"FlagOS 的开发者工具(developer tools)有哪些?","lang":"zh",
  "ents":["KernelGen","FlagOS-Compressor","FlagOS Skills","Online Laboratory"],
  "facts":[["KernelGen"],["FlagOS-Compressor","Compressor"],["FlagOS Skills","Skills"],["Online Laborator"]]},
 {"id":"B4","q":"FlagOS 的领域专用项目(domain-specific projects)有哪些?","lang":"zh",
  "ents":["FlagOS-Robo","FlagQuantum"],
  "facts":[["FlagOS-Robo","Robo"],["FlagQuantum","Quantum"]]},
 # --- C 单实体细节（易，应当通过）---
 {"id":"C1","q":"FlagGems 是什么?","lang":"zh",
  "ents":["FlagGems"],
  "facts":[["FlagGems"],["operator","算子","操作"],["Triton"]]},
 {"id":"C2","q":"FlagCX 的作用是什么?","lang":"zh",
  "ents":["FlagCX"],
  "facts":[["FlagCX"],["communication","通信"],["cross-chip","跨芯片","multi-chip"]]},
 {"id":"C3","q":"Torch-FL 是干什么的?","lang":"zh",
  "ents":["Torch-FL"],
  "facts":[["Torch-FL"],["PyTorch"],["PrivateUse1","device","私有使用","设备"]]},
 # --- D 跨文档事实（whats_new）---
 {"id":"D1","q":"FlagTree-CPU 支持哪些低比特量化?基于哪个 triton 版本?","lang":"zh",
  "ents":["FlagTree-CPU","W8A8"],
  "facts":[["W4A8"],["W8A8"],["triton-cpu 3.7.2","3.7.2"]]},
 {"id":"D2","q":"FlagRelease 提供多少个模型镜像?","lang":"zh",
  "ents":["FlagRelease","300"],
  "facts":[["300"]]},
 # --- E 英文/空泛（复现之前翻车的问法）---
 {"id":"E1","q":"What are the core libraries of FlagOS?","lang":"en",
  "ents":["FlagGems","FlagTree","FlagScale","FlagCX","Operator librar"],
  "facts":[["FlagGems"],["FlagTree"],["FlagScale"],["FlagCX"],["operator","算子","操作"]]},
 {"id":"E2","q":"explain FlagOS in details","lang":"en",
  "ents":["core librar","Plugins","Platform services","FlagGems","FlagTree"],
  "facts":[["core librar"],["plugin"],["FlagGems"],["FlagTree"],["platform service"]]},
]

# ===== 向量化 + 检索（复用网关同一套 bge-m3 + Qdrant）=====
_tok = AutoTokenizer.from_pretrained(EMBED_BASE)
_sess = ort.InferenceSession(f"{EMBED_BASE}/model.onnx", providers=["CPUExecutionProvider"])
_inn = [i.name for i in _sess.get_inputs()]
_outn = [o.name for o in _sess.get_outputs()]
_cli = QdrantClient(url="http://localhost:6333")

def embed(text):
    enc = _tok(text, return_tensors="np", truncation=True, max_length=512, padding=True)
    feed = {n: enc[n].astype(np.int64) for n in _inn if n in enc}
    outs = _sess.run(None, feed)
    if "sentence_embedding" in _outn:
        emb = outs[_outn.index("sentence_embedding")][0]
    else:
        emb = outs[0][0, 0]
    return (emb / np.linalg.norm(emb)).astype(np.float32)

def retrieve(q, k=15):
    qv = embed(q).tolist()
    res = _cli.query_points(collection_name=COLL, query=qv, limit=k).points
    return "\n".join(p.payload["text"] for p in res)

def ask_gateway(q, max_tokens=512):
    body = {"model":"qwen2.5-rag","messages":[{"role":"user","content":q}],
            "max_tokens":max_tokens,"temperature":0.1}
    req = urllib.request.Request(GATEWAY, data=json.dumps(body).encode(),
                                 headers={"Content-Type":"application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return r["choices"][0]["message"]["content"]

def cover(text, points):
    """points: [[alt1,alt2],...]  命中=任一写法出现。返回(命中数,总数,缺失列表)"""
    low = text.lower()
    hit, miss = 0, []
    for alts in points:
        if any(a.lower() in low for a in alts):
            hit += 1
        else:
            miss.append(alts[0])
    return hit, len(points), miss

# ===== 跑评测 =====
rows, answers = [], []
print("="*72)
print(f"{'ID':4}{'类型':6}{'检索召回':>10}{'生成覆盖':>10}   诊断")
print("="*72)
for e in EVAL:
    ctx = retrieve(e["q"])
    r_hit, r_tot, r_miss = cover(ctx, [[x] for x in e["ents"]])
    t0 = time.time()
    try:
        ans = ask_gateway(e["q"])
        err = ""
    except Exception as ex:
        ans, err = "", f"ERR:{ex}"
    dt = time.time() - t0
    g_hit, g_tot, g_miss = cover(ans, e["facts"])
    # 诊断：检索缺 → 检索层问题；检索全有但生成缺 → 模型层问题
    if r_hit < r_tot:
        diag = f"检索漏:{','.join(r_miss)}"
    elif g_hit < g_tot:
        diag = f"模型漏:{','.join(g_miss)}"
    else:
        diag = "OK"
    print(f"{e['id']:4}{e['lang']:6}{f'{r_hit}/{r_tot}':>10}{f'{g_hit}/{g_tot}':>10}   {diag}  ({dt:.0f}s)")
    rows.append({"id":e["id"],"lang":e["lang"],"retrieval":[r_hit,r_tot],
                 "generation":[g_hit,g_tot],"r_miss":r_miss,"g_miss":g_miss,
                 "diag":diag,"sec":round(dt,1),"err":err})
    answers.append({"id":e["id"],"q":e["q"],"answer":ans})

# ===== 汇总 =====
rs = sum(x["retrieval"][0] for x in rows); rt = sum(x["retrieval"][1] for x in rows)
gs = sum(x["generation"][0] for x in rows); gt = sum(x["generation"][1] for x in rows)
n_ok = sum(1 for x in rows if x["diag"]=="OK")
n_ret = sum(1 for x in rows if x["diag"].startswith("检索漏"))
n_gen = sum(1 for x in rows if x["diag"].startswith("模型漏"))
print("="*72)
print(f"总检索召回率: {rs}/{rt} = {rs/rt*100:.0f}%")
print(f"总生成覆盖率: {gs}/{gt} = {gs/gt*100:.0f}%")
print(f"题目诊断分布: 全对 {n_ok} | 检索层失分 {n_ret} | 模型层失分 {n_gen} | 共 {len(rows)} 题")
print(f"平均单题生成耗时: {sum(x['sec'] for x in rows)/len(rows):.0f}s")

with open("/tmp/rag_baseline.json","w",encoding="utf-8") as f:
    json.dump({"rows":rows,"answers":answers},f,ensure_ascii=False,indent=2)
print("\n完整回答与明细已存 /tmp/rag_baseline.json")
