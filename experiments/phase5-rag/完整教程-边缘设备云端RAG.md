# 在边缘设备上搭一套"云端 RAG 文档问答系统"——完整实战教程

> 面向零基础读者。从"为什么要做"讲到"怎么一步步做出来",每一步都说清**作用**，
> 并结合我们真实的调试过程举例。读完你能自己复现一套能问答的文档知识库。

---

## 目录

1. [背景：我们遇到的问题](#1-背景)
2. [需求：到底要做成什么样](#2-需求)
3. [先搞懂几个名词（小白必读）](#3-名词扫盲)
4. [解决方案总览](#4-解决方案总览)
5. [方案选型：为什么这么选](#5-方案选型)
6. [需要的环境与材料](#6-环境与材料)
7. [全流程架构：每个环节干什么](#7-全流程架构)
8. [一步步部署（结合实测举例）](#8-部署步骤)
9. [踩坑总结（我们真实踩过的）](#9-踩坑总结)
10. [最终效果与验收](#10-最终效果)

---

## 1. 背景

我们有一块边缘计算小板子（Radxa Orion O6N，CIX P1 芯片，Debian 12 系统）。
手头有一套技术文档（FlagOS 项目文档，1000 多篇 Markdown）。

**目标**：让人能用自然语言问这套文档，比如问"FlagOS 是什么？""FlagCX 怎么安装？"，
系统自动从文档里找答案并用人话回答。这就是 **RAG（检索增强生成）** 要解决的事。

**一开始的弯路**：我们先试过把 AI 模型（嵌入模型、大语言模型）直接跑在板子上。
结论是——**这块板子的 NPU 擅长跑视觉 CNN（MobileNetV2/YOLO），但不适合跑大语言模型
和大的嵌入模型**（实测 bge-m3 嵌入模型量化到 INT8 后余弦相似度只剩 0.51，质量崩了；
CPU 跑嵌入 1.77 秒一块，全量建索引要 6 小时）。

所以方向调整为：**板子当"服务器 + 编排层"，把重活（嵌入、生成）交给云端 API**。
这篇教程讲的就是这套"云端 RAG"方案。

---

## 2. 需求

把需求说清楚，才知道做成什么样算成功：

1. **能问答**：用户在网页界面输入问题，系统返回基于文档的答案；
2. **答得准**：答案必须来自文档，不能瞎编（AI 最怕"一本正经胡说八道"）；
3. **中英文都行**：文档是英文，但用户可能用中文问；
4. **能扩展**：现在 1000 多篇，以后可能上万篇；
5. **板子扛得住**：板子只做轻量的编排和存储，不干重活。

---

## 3. 名词扫盲（小白必读）

| 名词 | 一句话解释 |
|---|---|
| **RAG** | 检索增强生成。先从文档里"检索"相关段落，再让 AI"生成"答案。不是 AI 凭记忆答，是现查现答。 |
| **嵌入 / Embedding** | 把一段文字变成一串数字（向量），意思相近的文字，数字也相近。用来做"语义搜索"。 |
| **向量数据库** | 专门存这些数字向量、并能快速"找最相似"的数据库。我们用 PostgreSQL + pgvector。 |
| **切块 / Chunking** | 把长文档切成小段再存。因为检索和喂给 AI 都需要"小段"。 |
| **召回 / 检索** | 根据问题，从所有小段里挑出最相关的几段。 |
| **FTS / 全文检索** | 按"关键词"找（像传统搜索引擎）。和向量检索（按"语义"找）互补。 |
| **RRF** | 把多种检索方式的结果排名融合到一起的算法。 |
| **Rerank / 精排** | 对召回的候选段落重新打分排序，把最相关的提到最前。 |
| **大语言模型 / LLM** | 就是 ChatGPT 那类，负责"读懂检索到的段落 + 用人话组织答案"。 |
| **网关 / Gateway** | 我们自己写的中间服务，把上面这些环节串起来。 |

---

## 4. 解决方案总览

一句话：**网页问答界面 → 网关 → （检索：向量+关键词融合+精排）→ 云端大模型生成答案**。

```
用户（浏览器 Open WebUI :3000）
  └─ FastAPI 网关 (:8000，跑在板子上)
       ├─ 嵌入     → 云端 OpenRouter（qwen3-embedding-8b，1024维）
       ├─ 向量检索 → 板子本地 PostgreSQL + pgvector
       ├─ 关键词检索(FTS) → 板子本地 PostgreSQL 全文检索
       ├─ RRF 融合 → 把两路检索结果合并排序
       ├─ 精排     → 云端 OpenRouter（cohere/rerank-v3.5）
       └─ 生成答案 → 云端 DeepSeek（deepseek-flash）
```

**板子负责**：网关编排 + 向量/全文存储检索（轻活）。
**云端负责**：嵌入、精排、生成（重活）。

---

## 5. 方案选型：为什么这么选

每个选择都是有理由的，不是拍脑袋：

| 环节 | 选型 | 为什么 |
|---|---|---|
| 向量库 | PostgreSQL + pgvector | 一个库同时存向量+原文+元数据，运维主流；板子资源足够 |
| 嵌入模型 | 云端 qwen3-embedding-8b | 板子跑不动大嵌入模型（NPU 量化质量崩、CPU 太慢）；云端又快又好 |
| 向量维度 | 1024 | 该模型默认 4096 维，但 pgvector 的 HNSW 索引上限 2000 维；用它支持的 MRL 截断到 1024 |
| 生成模型 | 云端 DeepSeek deepseek-flash | 板子本地 3B 模型能力不足（长列表会截断、会漏项）；云端大模型稳 |
| 检索方式 | Hybrid（向量+关键词+RRF+精排） | **用数据对照实验选出来的**（见第 9 节），比任何单一方式都好 |
| 只建英文索引 | 是 | 文档有中英两版，英文是源语言更权威；实测跨语言对齐好，中文问也能命中英文块 |

---

## 6. 环境与材料

**硬件/系统**：
- 边缘板子：Radxa Orion O6N（CIX P1），Debian 12，12 核 ARM64，30GB 可用内存，1TB NVMe
- 你的访问方式：SSH 登录板子（示例 `ssh baai@192.168.15.195`）

**要装的软件（都在板子上）**：
- PostgreSQL 15 + pgvector 扩展（向量库）
- Python 3.11 venv（跑网关）+ FastAPI / uvicorn / psycopg2 / httpx
- llama-server（可选，本地小模型备用）
- Open WebUI（网页问答界面，Docker 跑，端口 3000）

**要申请的云端 API Key（3 个，都用 OpenAI 兼容接口）**：
- 嵌入 + 精排：OpenRouter（`sk-or-...`）
- 生成：DeepSeek（`sk-...`）

**语料**：你的文档（我们是 `/home/baai/rag-test/docs/docs` 下的 1287 篇 md）

---

## 7. 全流程架构：每个环节干什么

一次问答，数据这样流动（以"FlagCX 怎么安装"为例）：

1. **用户提问** → 浏览器 Open WebUI 发给网关；
2. **网关过滤** → 先踢掉 Open WebUI 的"自动生成标题"等系统请求，只处理真实提问；
3. **嵌入** → 把问题发给云端嵌入 API，得到 1024 维向量（代表问题的"语义"）；
4. **向量检索（Dense）** → 拿这个向量去 pgvector 找语义最近的 50 块 → 擅长"是什么/关系"类问题；
5. **关键词检索（FTS）** → 同时用 PostgreSQL 全文检索按 "FlagCX install" 关键词找 50 块 → 擅长"怎么装/怎么用"类问题；
6. **RRF 融合** → 把两路结果按排名融合（互补：向量漏的关键词补，反之亦然）；
7. **精排（Rerank）** → 把融合后的候选发给云端 rerank API，重新打分，挑最相关的 8 块；
8. **生成** → 把这 8 块 + 问题 + 规则（只据文档答、不许编、用提问语言回答）发给 DeepSeek，生成最终答案；
9. **返回** → 答案经网关回到浏览器。

**每个环节的"作用"一句话**：
- 嵌入=把文字变成可比较的语义向量；
- 向量检索=按意思找；关键词检索=按词找；RRF=两者取长补短；
- 精排=从"大致相关"里挑"最相关"；
- 生成=把检索到的素材组织成人话答案。

---

## 8. 部署步骤（结合我们的实测举例）

> 下面每一步都给命令和"为什么"。命令在板子上执行（SSH 进去）。

### 第 0 步：准备 Python 环境

```bash
source /opt/edge-ai-lab/venv/bin/activate   # 激活虚拟环境
pip install fastapi uvicorn psycopg2-binary httpx "numpy<2" \
    -i https://mirrors.aliyun.com/pypi/simple/   # 走国内镜像，否则超时
```
**作用**：装网关要用的 Python 库。
**实测踩坑**：板子直连 PyPI 官方源会超时，必须用阿里云镜像。

### 第 1 步：装 PostgreSQL + pgvector（向量库）

```bash
# 1.1 换 apt 源为华为云镜像（板子连不上官方 deb.debian.org）
sudo sed -i "s#https://deb.debian.org/debian-security#https://mirrors.huaweicloud.com/debian-security#g; \
             s#https://deb.debian.org/debian#https://mirrors.huaweicloud.com/debian#g" \
             /etc/apt/sources.list.d/*.list
sudo apt-get update

# 1.2 装 PostgreSQL 15 + 编译工具
sudo apt-get install -y --fix-missing postgresql postgresql-contrib \
     postgresql-server-dev-15 build-essential git make gcc

# 1.3 pgvector 源码编译（apt 源里没有现成包）
cd /tmp && git clone --depth 1 https://github.com/pgvector/pgvector.git
cd pgvector && sudo make && sudo make install
```
**作用**：PostgreSQL 是数据库本体；pgvector 是给它加"向量相似度搜索"能力的扩展。
**实测踩坑**：① 官方 apt 源连不上 → 换华为云；② 下载偶尔断 → 加 `--fix-missing` 重试；
③ pgvector 没有 apt 包 → 必须源码编译（幸好板子有 gcc/make）。

### 第 2 步：建数据库和表

```bash
# 建角色和库
sudo -u postgres psql -c "CREATE ROLE raguser LOGIN PASSWORD 'ragpass';"
sudo -u postgres createdb -O raguser ragdb

# 启用扩展 + 建表 + 建索引（以 postgres 身份，否则无权限）
sudo -u postgres psql -d ragdb <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE rag_chunks (
  id        bigserial PRIMARY KEY,
  source    text NOT NULL,          -- 来自哪篇文档
  title     text,                   -- 章节标题路径，如 "FlagOS Overview > What is FlagOS?"
  chunk_idx int,                    -- 块在文档中的序号
  content   text NOT NULL,          -- 块的正文
  embedding vector(1024)            -- 1024 维向量
);
CREATE INDEX rag_chunks_hnsw ON rag_chunks
  USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
ALTER TABLE rag_chunks OWNER TO raguser;
SQL
```
**作用**：建一张表存"文档块+向量"，HNSW 索引让"找最相似"变成毫秒级。
**实测踩坑**：表的 owner 必须对，否则后面 raguser 无权改表（我们就因为 owner 不对，
加全文检索列时报 `must be owner of table`）。

### 第 3 步：配置云端 API Key

在板子建一个 `.env` 文件 `/opt/edge-ai-lab/.env`（**Key 不要提交到 git**）：
```ini
EMBED_BASE_URL=https://openrouter.ai/api/v1
EMBED_API_KEY=你的OpenRouterKey
EMBED_MODEL=qwen/qwen3-embedding-8b
EMBED_DIM=1024
CHAT_BASE_URL=https://api.deepseek.com
CHAT_API_KEY=你的DeepSeekKey
CHAT_MODEL=deepseek-flash
RERANK_BASE_URL=https://openrouter.ai/api/v1
RERANK_MODEL=cohere/rerank-v3.5
PG_DSN=postgresql://raguser:ragpass@localhost:5432/ragdb
```
`chmod 600 /opt/edge-ai-lab/.env`（只有自己能读）。
**实测踩坑**：① qwen3-embedding 默认输出 4096 维，必须传 `dimensions:1024` 截断
（否则超 pgvector 索引上限）；② deepseek-flash 是带思考(reasoning)的模型，调用时
`max_tokens` 要给足（≥1024），否则思考就把额度吃光、正式回答为空。

### 第 4 步：建索引（把文档灌进向量库）

写一个脚本：遍历文档 → 切块 → 调嵌入 API → 存入 pgvector。核心逻辑：
```python
# 切块：按 Markdown 标题层级切，每块带标题路径
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
# 嵌入：批量调 OpenRouter，dimensions=1024，失败重试
# 入库：execute_values 批量 INSERT
```
运行（建议后台 + 断点续传）：
```bash
nohup python3 build_index.py > build.log 2>&1 &
```
**作用**：把 859 篇英文文档切成约 1.4 万块，每块算向量存库。
**实测数据**：全程约 1.5 小时（云端嵌入 API，比板子本地 CPU 的 6 小时快）。
**实测踩坑**：
- ① 网络抖动会让脚本崩 → 加"重试 8 次 + 单篇失败跳过 + 断点续传（已入库的跳过）"；
- ② 别把构建产物（`_build/`、`_sources/` 等 HTML 导出副本）也索引了 → 会重复污染；
- ③ 只建英文（排除 `_zh` 目录）→ 省一半时间，中文问也能命中（跨语言对齐好）。

### 第 5 步：建全文检索（FTS）能力

```sql
-- 给表加一列 tsvector（title+content 分词），建 GIN 索引
ALTER TABLE rag_chunks ADD COLUMN content_tsv tsvector;
UPDATE rag_chunks SET content_tsv =
   to_tsvector('english', coalesce(title,'') || ' ' || content);
CREATE INDEX rag_chunks_fts ON rag_chunks USING GIN (content_tsv);
```
**作用**：让数据库支持"按关键词找"（像搜索引擎），和向量检索互补。
**为什么要它**：实测发现"FlagCX install"这种**操作类问题，关键词检索比向量准得多**。

### 第 6 步：写网关（把流程串起来）

两个文件：
- `app/cloud_rag.py`：封装"嵌入 + 向量检索 + 全文检索 + RRF + 精排"（`CloudRAG.search()`）；
- `app/main.py`：FastAPI，提供 OpenAI 兼容的 `/v1/chat/completions`，`qwen-rag` 模型走 RAG。

启动：
```bash
cd /opt/edge-ai-lab/services/api
export LLAMA_SERVER_URL=http://localhost:8081
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 \
      > gateway.log 2>&1 &
```
**实测踩坑**：后台启动要用 `setsid nohup ... & disown` 彻底脱离，否则 SSH 一断网关被带走
（我们就遇到过网关"启动后又被关闭"，排查发现是进程没脱离终端）。

### 第 7 步：接 Open WebUI（网页界面）

Open WebUI（Docker，端口 3000）里添加一个"OpenAI 兼容"连接，地址指向网关：
```
http://<板子地址>:8000/v1
```
然后在模型下拉里选 `qwen-rag`，就能聊天问文档了。
**实测踩坑**：Open WebUI 会偷偷发"生成对话标题""生成标签"等系统请求给网关，
这些不该触发完整 RAG（白跑、还和真实提问抢云端配额）→ 网关里要识别并过滤掉。

---

## 9. 踩坑总结（我们真实踩过的，按重要性排序）

### 坑 1（最核心）：`what is flagos?` 一开始完全答不出

这是整个项目最硬的骨头，层层排查，每层都有真实数据：

| 排查层 | 发现（实测数据） | 解法 |
|---|---|---|
| 切块 | overview 被切成 19 碎块，核心库答案散在 8 块里 | 重组成标题层级（37 块，每实体独立） |
| 召回 | 定义块在全库**排第 176 名**，被 300+ 模型卡淹没 | SQL 层过滤噪声目录 |
| 再查 | 重组后定义块**仍只排第 70、相似度 0.634** | 定位真因 ↓ |
| 真因 | **问题"what is"和答案"X is a..."字面不匹配 + flagos 词太泛** | 这是向量检索的范式局限 |

**第一版解法（Query 改写）**：让 DeepSeek 把"what is flagos"改写成"FlagOS is an
open-source AI system stack"再检索 → 定义块命中分 **0.81→0.95 排第一，答对了**。
但慢（20-30 秒）。

**踩坑**：改写一度返回空——deepseek-flash 先思考 676 token，`max_tokens` 只给 200
被吃光 → 调到 1024 才正常。

### 坑 2：对照实验推翻了 Query 改写

不满足于"能用"，建了 12 题评测集做五组检索对照实验（指标 Recall@10 / MRR@10）：

| 方案 | Recall@10 | MRR@10 |
|---|---|---|
| 纯向量 | 83% | 0.384 |
| 纯全文检索 | 50% | 0.326 |
| 向量+全文+RRF | 92% | 0.519 |
| **+ 精排（最终方案）** | **92%** | **0.729** |
| 改写+向量+精排（旧） | 92% | 0.736 |

**结论**：Hybrid+精排 和 改写方案效果一样好，但**不需要那次又慢又不稳的改写**
（全文检索天然替代了改写的作用）。于是定稿 Hybrid，**速度从 20-30 秒降到 10-11 秒**。

### 坑 3：嵌入模型想跑在板子 NPU 上——此路不通

想用板子 NPU 加速嵌入，结论是负面的（但很有价值）：
- FP32 模型上 NPU：1236 个节点被拒（动态形状，NPU 要静态）；
- 固定形状后：仍有 192 个 MatMul 被拒（NPU 只吃 INT8）；
- INT8 量化后：**嵌入余弦相似度崩到 0.51**（per-tensor 量化扛不住 Transformer 激活离群值）。
- **教训**：这块 NPU 适合 CNN 视觉，不适合大 Transformer 嵌入。老实用云端 API。

### 坑 4：云端服务不稳会拖垮整个网关

OpenRouter 从板子连，**5 次有 3 次超时**。最初嵌入设了 `timeout=60s × 重试5次`，
OpenRouter 一断，一次请求死等近 5 分钟，把网关拖到不响应 → 界面报 "model not found"。
**解法**：所有云端调用改"快速失败"（超时 15s、重试 2 次），断网时几秒放弃而非拖死；
Dense 失败还能退化成纯 FTS，不致瘫痪。

### 坑 5：一堆环境/网络小坑

| 坑 | 解法 |
|---|---|
| PyPI / apt / 官方源连不上 | 换阿里云 / 华为云镜像 |
| pgvector 无 apt 包 | 源码编译 |
| 表 owner 不对无权改表 | `ALTER TABLE ... OWNER TO raguser` |
| 2GB+ 模型 protobuf 序列化失败 | `save_as_external_data=True` |
| paramiko 后台进程被 SSH 带走 | `setsid nohup ... & disown` |
| 文档中英双版 | 只建英文，跨语言对齐已验证 |

---

## 10. 最终效果

- `what is flagos?`：11 秒答出"FlagOS 是全开源、面向异构 AI 芯片的统一软件栈..."；
- `how to install FlagCX?`：直接给出 `git clone` + 安装步骤（全文检索命中安装文档）；
- `FlagOS 有哪些核心库?`（中文）：完整列出算子库/FlagTree/FlagScale/FlagCX。

**一句话收获**：RAG 的准确性首先取决于**检索**（能不能把对的段落找出来），
而不是反复调 AI 的提示词。用**数据（正确块的排名）**定位问题在哪一层，
用 **Hybrid（向量+关键词+RRF+精排）** 让不同类型的问题各得其所——这才是正道。

---

## 附：关键坐标速查

| 项 | 值 |
|---|---|
| 向量库 | PostgreSQL 15 + pgvector 0.8.7，库 `ragdb`，表 `rag_chunks` |
| 网关 | FastAPI `:8000`，`app/main.py` + `app/cloud_rag.py` |
| 界面 | Open WebUI `:3000`，模型选 `qwen-rag` |
| 配置 | `/opt/edge-ai-lab/.env` |
| 语料 | 英文 859 篇 → 13896 块 |
| 检索 | Dense + FTS + RRF + Rerank |
| 实验脚本 | `eval/retrieval_experiment.py` |
