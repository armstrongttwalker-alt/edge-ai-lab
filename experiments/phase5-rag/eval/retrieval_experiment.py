# -*- coding: utf-8 -*-
"""检索对照实验：A/B/C/D/E 五组 × 12 题，算 Recall@10 / MRR@10，并逐层追踪正确块排名。
在板子上运行，结果写 /tmp/rag_exp_result.txt。不碰生产网关。"""
import json, time, urllib.request, psycopg2

ENV = {}
for line in open("/opt/edge-ai-lab/.env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); ENV[k] = v.strip()

EMBED_URL = ENV["EMBED_BASE_URL"].rstrip("/") + "/embeddings"
EMBED_KEY = ENV["EMBED_API_KEY"]; EMBED_MODEL = ENV["EMBED_MODEL"]
RERANK_URL = ENV.get("RERANK_BASE_URL", "").rstrip("/") + "/rerank"
RERANK_MODEL = ENV.get("RERANK_MODEL", "")
CHAT_URL = ENV["CHAT_BASE_URL"].rstrip("/") + "/chat/completions"
CHAT_KEY = ENV["CHAT_API_KEY"]; CHAT_MODEL = ENV["CHAT_MODEL"]
NOISE = "source NOT LIKE '%%/model_readmes/%%' AND source NOT LIKE '%%/chip_adaptation_guide/%%' AND source NOT LIKE '%%/references/test/%%'"

# ===== 评测集：每题的"正确块"用 (source LIKE, title LIKE) 判定；重复文档两套都算 =====
EVAL = [
 ("D1","What is FlagOS?","定义",[("flagos_homepage/overview.md","%What is FlagOS%")]),
 ("D2","What is FlagGems?","定义",[("%","%FlagGems (general-purpose%")]),
 ("D3","What is FlagCX?","定义",[("%","%FlagCX (communication%")]),
 ("E1","What are the core libraries of FlagOS?","枚举",[("flagos_homepage/overview.md","%Open-source core libraries%")]),
 ("E2","What plugins does FlagOS provide?","枚举",[("flagos_homepage/overview.md","%Plugins%"),("%","FlagOS > FlagOS Plugins%")]),
 ("E3","What platform services does FlagOS have?","枚举",[("flagos_homepage/overview.md","%Platform services%"),("%","FlagOS > FlagOS Platform%")]),
 ("O1","How to install or build FlagCX?","操作",[("flagcx_en/%","%Install%"),("flagcx_en/%","%Build%")]),
 ("O2","How to configure or install FlagScale?","操作",[("flagscale_en/%","%Install FlagScale%"),("flagscale_en/%","%getting%")]),
 ("O3","How to use FlagGems?","操作",[("%","Use FlagGems%"),("%","%Enabling FlagGems%")]),
 ("R1","How does FlagOS support heterogeneous AI chips?","综合",[("flagos_homepage/overview.md","%What is FlagOS%"),("%","%FlagCX (communication%")]),
 ("R2","What is the difference between FlagGems and FlagGems-vllm?","综合",[("%","%FlagGems-vLLM Overview%"),("%","%FlagGems (general-purpose%")]),
 ("R3","How does FlagTree relate to FlagGems?","综合",[("flagtree_en/%","%architecture%"),("flagos_homepage/overview.md","%FlagTree (compiler%")]),
]

conn = psycopg2.connect(ENV["PG_DSN"]); conn.autocommit=True

def is_correct(src, title, patterns):
    import fnmatch
    def like(val, pat):
        # SQL LIKE -> python: % -> *, 大小写不敏感
        return fnmatch.fnmatch((val or "").lower(), pat.replace("%","*").lower())
    return any(like(src,sp) and like(title,tp) for sp,tp in patterns)

def embed(text, retries=3):
    body=json.dumps({"model":EMBED_MODEL,"input":text,"dimensions":1024}).encode()
    last=None
    for i in range(retries):
        try:
            req=urllib.request.Request(EMBED_URL,data=body,headers={"Content-Type":"application/json","Authorization":f"Bearer {EMBED_KEY}"})
            return json.loads(urllib.request.urlopen(req,timeout=30).read())["data"][0]["embedding"]
        except Exception as e:
            last=e; time.sleep(3)
    raise last

def dense(qv, k=50):
    cur=conn.cursor(); cur.execute("SET hnsw.ef_search=200;")
    cur.execute(f"SELECT id,source,title FROM rag_chunks WHERE {NOISE} ORDER BY embedding <=> %s::vector LIMIT %s",(qv,k))
    r=cur.fetchall(); cur.close(); return r

def fts(query, k=50):
    cur=conn.cursor()
    cur.execute(f"SELECT id,source,title FROM rag_chunks WHERE {NOISE} AND content_tsv @@ websearch_to_tsquery('english',%s) ORDER BY ts_rank(content_tsv, websearch_to_tsquery('english',%s)) DESC LIMIT %s",(query,query,k))
    r=cur.fetchall(); cur.close(); return r

def rrf(lists, k=60):
    score={}
    meta={}
    for lst in lists:
        for rank,(rid,src,title) in enumerate(lst):
            score[rid]=score.get(rid,0)+1.0/(k+rank+1)
            meta[rid]=(src,title)
    order=sorted(score, key=lambda x:-score[x])
    return [(rid,meta[rid][0],meta[rid][1]) for rid in order]

def rerank(query, cands, top_k=10):
    if not RERANK_MODEL or not cands: return cands[:top_k]
    # 取内容
    ids=[x[0] for x in cands[:50]]
    cur=conn.cursor(); cur.execute("SELECT id,content FROM rag_chunks WHERE id = ANY(%s)",(ids,))
    cmap={rid:ct for rid,ct in cur.fetchall()}; cur.close()
    docs=[cmap.get(i,"") for i in ids]
    body=json.dumps({"model":RERANK_MODEL,"query":query,"documents":docs,"top_n":top_k}).encode()
    try:
        req=urllib.request.Request(RERANK_URL,data=body,headers={"Content-Type":"application/json","Authorization":f"Bearer {EMBED_KEY}"})
        r=json.loads(urllib.request.urlopen(req,timeout=30).read())
        out=[]
        for it in r["results"]:
            rid=ids[it["index"]];
            # 从 cands 找 meta
            m=next((x for x in cands if x[0]==rid),(rid,"",""))
            out.append((rid,m[1],m[2]))
        return out
    except Exception:
        return cands[:top_k]

def rewrite(query, retries=2):
    prompt=("Rewrite the user's question into 3 short statement-style search queries that match "
            "the wording of a documentation passage answering it. Declarative, keep proper names. "
            f"Output ONLY the 3 queries, one per line.\n\nUser question: {query}")
    body=json.dumps({"model":CHAT_MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":1024,"temperature":0.3}).encode()
    for i in range(retries):
        try:
            req=urllib.request.Request(CHAT_URL,data=body,headers={"Content-Type":"application/json","Authorization":f"Bearer {CHAT_KEY}"})
            txt=json.loads(urllib.request.urlopen(req,timeout=40).read())["choices"][0]["message"].get("content") or ""
            qs=[l.strip(" -*0123456789.").strip() for l in txt.splitlines() if l.strip()]
            return [q for q in qs if len(q)>3][:3]
        except Exception:
            time.sleep(3)
    return []

def first_hit_rank(lst, patterns):
    for rank,(rid,src,title) in enumerate(lst,1):
        if is_correct(src,title,patterns): return rank
    return None

def metrics(lst, patterns, k=10):
    r=first_hit_rank(lst[:k], patterns)
    recall = 1 if r else 0
    mrr = 1.0/r if r else 0.0
    return recall, mrr, r

out=open("/tmp/rag_exp_result.txt","w",encoding="utf-8")
def P(s): out.write(s+"\n"); out.flush()

groups=["B_Dense","C_FTS","D_RRF","E_RRF_Rerank","A_Rewrite_Dense_Rerank"]
agg={g:{"recall":0,"mrr":0.0,"n":0} for g in groups}

P("检索对照实验  (Recall@10 / MRR@10, 正确块首次命中排名)")
P("="*90)
for qid,question,qtype,patterns in EVAL:
    P(f"\n[{qid}] ({qtype}) {question}")
    try:
        qv=embed(question)
    except Exception as e:
        P(f"  嵌入失败(OpenRouter抖): {e}"); continue
    d=dense(qv); f=fts(question)
    # B Dense
    rc,mr,rk=metrics(d,patterns); agg["B_Dense"]["recall"]+=rc; agg["B_Dense"]["mrr"]+=mr; agg["B_Dense"]["n"]+=1
    P(f"  B 纯Dense      : Recall@10={rc} MRR={mr:.3f} 正确块排名={rk}  (Dense top50命中={first_hit_rank(d,patterns)})")
    # C FTS
    rc,mr,rk=metrics(f,patterns); agg["C_FTS"]["recall"]+=rc; agg["C_FTS"]["mrr"]+=mr; agg["C_FTS"]["n"]+=1
    P(f"  C 纯FTS        : Recall@10={rc} MRR={mr:.3f} 正确块排名={rk}  (FTS top50命中={first_hit_rank(f,patterns)})")
    # D RRF
    fused=rrf([d,f])
    rc,mr,rk=metrics(fused,patterns); agg["D_RRF"]["recall"]+=rc; agg["D_RRF"]["mrr"]+=mr; agg["D_RRF"]["n"]+=1
    P(f"  D Dense+FTS+RRF: Recall@10={rc} MRR={mr:.3f} 正确块排名={rk}")
    # E RRF + Rerank
    rr=rerank(question, fused, top_k=10)
    rc,mr,rk=metrics(rr,patterns); agg["E_RRF_Rerank"]["recall"]+=rc; agg["E_RRF_Rerank"]["mrr"]+=mr; agg["E_RRF_Rerank"]["n"]+=1
    P(f"  E +Rerank      : Recall@10={rc} MRR={mr:.3f} 正确块排名={rk}")
    # A 当前方案: rewrite->dense(多查询合并)->rerank
    rws=rewrite(question)
    allq=[question]+rws
    mergedids={}
    for q in allq:
        try: qvv=embed(q)
        except Exception: continue
        for rank,(rid,src,title) in enumerate(dense(qvv)):
            if rid not in mergedids: mergedids[rid]=(rid,src,title)
    amerged=list(mergedids.values())
    ar=rerank(question, amerged, top_k=10)
    rc,mr,rk=metrics(ar,patterns); agg["A_Rewrite_Dense_Rerank"]["recall"]+=rc; agg["A_Rewrite_Dense_Rerank"]["mrr"]+=mr; agg["A_Rewrite_Dense_Rerank"]["n"]+=1
    P(f"  A 当前(改写+Dense+Rerank): Recall@10={rc} MRR={mr:.3f} 正确块排名={rk}  改写数={len(rws)}")

P("\n"+"="*90)
P("汇总 (12题平均):")
for g in groups:
    a=agg[g]; n=a["n"] or 1
    P(f"  {g:26s}  Recall@10={a['recall']}/{a['n']}={a['recall']/n*100:.0f}%   MRR@10={a['mrr']/n:.3f}")
out.close()
print("DONE")
