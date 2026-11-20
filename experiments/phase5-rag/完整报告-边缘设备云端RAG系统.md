# 边缘设备上的云端 RAG 文档问答系统 —— 完整技术报告

> 一份从零到可用的完整实战报告。记录在一块边缘计算板子上，如何从"文档问答答不准"
> 一步步做到"中英文都能准确问答"的全过程。包含：背景动机、需求拆解、概念原理、
> 方案选型论证、环境搭建、架构设计、每个环节的作用、逐步部署（结合真实测试数据举例）、
> 以及我们真实踩过的每一个坑及其根因分析。
>
> 面向零基础读者，关键名词都有解释；同时保留了全部实测数据，供有经验的读者复现与参考。

---

## 第一部分：背景与动机

### 1.1 项目所处的大背景

本项目隶属于 **CIX P1 Edge AI Lab**——一个在国产边缘 AI 芯片（CIX P1）上做异构算力
验证与应用落地的实验平台。硬件是一块 **Radxa Orion O6N** 开发板，通过 SSH 远程访问，
所有实验都在这块板子上进行。

在做完 NPU 视觉推理（MobileNetV2、YOLO）、CPU 大模型推理（llama.cpp）、FastAPI 推理网关、
Docker 服务编排之后，项目进入"文档智能问答"阶段——也就是本报告的主题：**RAG**。

### 1.2 为什么要做 RAG

我们手上有一整套技术文档（FlagOS 开源 AI 系统软件栈的文档，1287 篇 Markdown，
覆盖几十个子项目）。这套文档的问题是：

- **太多、太散**：1000 多篇，人工找答案很慢；
- **新人看不懂全貌**：想问"FlagOS 到底是什么""它有哪些核心库""某个组件怎么装"，
  没有一个入口能直接回答。

**RAG（Retrieval-Augmented Generation，检索增强生成）** 正是为此而生：
让用户用大白话提问，系统自动从文档里**检索**出相关段落，再交给大语言模型**生成**
一段基于这些段落的答案。它的最大好处是——**答案有出处、不是 AI 凭空编的**。

### 1.3 一开始走的弯路（很重要，决定了最终架构）

最初的设想是"全部本地化"：嵌入模型、大语言模型、向量库都跑在板子上，不依赖云端。
我们真的试了，而且踩得很彻底。结论是一句话：

> **这块板子的 NPU 擅长跑规整的视觉 CNN，但不适合跑大语言模型和大的嵌入模型。**

具体证据（后文第九部分有完整复盘）：

- 把嵌入模型 bge-m3 放到 NPU 上：FP32 整图被拒（NPU 要静态形状），固定形状后又因
  "NPU 只吃 INT8"被拒，强行 INT8 量化后 **嵌入向量的余弦相似度崩到 0.51**（等于向量
  基本失真，检索会全乱）；
- 退回 CPU 跑嵌入：**1.77 秒才能处理一个文档块**，批处理也没用（12 核已被单次推理占满），
  全量建索引要 **6 小时**；
- 本地大模型（Qwen2.5-3B）：量化到 Q4 时会把专有名词写错（"FlagTree"写成"Tree"）、
  列长清单会中途截断、漏项；换 Q8 名字对了但长列表仍力不从心，要 7B 才够，
  而 7B 在纯 CPU 上每个问题要 2-3 分钟。

**于是方向调整为**：板子当"**服务器 + 编排层**"，只做它擅长的轻活（流程编排、数据存储、
向量检索）；把真正吃算力的重活（文本嵌入、答案生成、精排）**交给云端 API**。
这就是本报告讲的"**云端 RAG**"方案。

---

## 第二部分：需求拆解

把需求讲清楚，才知道"做成什么样算成功"。我们定了 5 条，并给出各自的验收标准：

| # | 需求 | 验收标准 |
|---|---|---|
| 1 | **能问答** | 用户在网页界面输入问题，系统返回基于文档的答案 |
| 2 | **答得准** | 答案必须来自文档，不许编造；文档里没有的要如实说"没有"，不能幻觉 |
| 3 | **中英文都行** | 文档是英文，用户用中文问也要能答对，且用提问的语言回答 |
| 4 | **可扩展** | 现在 1000 多篇，架构要能扩到上万篇 |
| 5 | **板子扛得住** | 板子只做轻量编排与存储，不跑大模型；重活走云端 |

需求 2（不许幻觉）和需求 3（跨语言）是后面大量调试的焦点，尤其需求 2——
AI 最危险的不是"答不出"，而是"一本正经地胡说八道"。

---

## 第三部分：概念原理（小白必读）

在讲方案之前，先把关键名词讲透。理解了这些，后面的设计和踩坑才看得懂。

### 3.1 RAG 到底怎么工作

传统的 ChatGPT 是"凭记忆答"——它把训练时见过的知识压进参数里，回答时全靠记忆，
所以会过时、会编。RAG 不一样，它是"**现查现答**"：

```
提问 → ①检索：从你的文档里找出最相关的几段 → ②生成：把这几段+问题交给大模型，
       要求它"只根据这几段回答" → 得到有出处的答案
```

**关键认知**：RAG 答得准不准，**第一取决于"检索"能不能把对的段落找出来**。
如果检索阶段就没找到正确段落，后面大模型再聪明也是巧妇难为无米之炊——
这是本项目最核心的教训，后面反复印证。

### 3.2 嵌入（Embedding）与向量

计算机不懂文字，但擅长算数字。**嵌入**就是把一段文字变成一串数字（叫"向量"，
我们用的是 1024 个数字）。神奇之处在于：**意思相近的两段文字，它们的向量也相近**。

于是"找语义最相关的段落"就变成了数学问题："找向量距离最近的段落"。
衡量距离常用**余弦相似度**（0 到 1，越接近 1 越相似）。

### 3.3 向量数据库与 HNSW 索引

有了向量，还要能**快速**从上万个向量里找最近的。逐个算太慢，所以用
**向量数据库 + 专门的索引（HNSW）**，把"找最近邻"变成毫秒级。
我们用 **PostgreSQL + pgvector 扩展**来做这件事。

### 3.4 切块（Chunking）

一篇长文档不能整篇存，要切成小段（"块"），原因有二：
1. 嵌入模型一次能处理的长度有限（我们用 512 token 上限）；
2. 检索要精确——用户问一个点，最好只召回讲那个点的那一小段，而不是整篇。

**切块的质量极大影响效果**——切得太碎会把一个完整答案劈成几块导致召回不全；
切得太大又会让一块里混进无关内容。后面第九部分有血的教训。

### 3.5 两种检索：向量检索 vs 全文检索

- **向量检索（Dense）**：按"语义"找。问"如何配置 FlagScale"，它能找到讲配置的段落，
  哪怕用词不同。擅长概念、定义、关系类问题。
- **全文检索（FTS，近似 BM25）**：按"关键词"找，像传统搜索引擎。问"FlagCX install"，
  它精确匹配含这些词的段落。擅长操作类、含具体专有名词的问题。

两者各有盲区，**合起来（Hybrid）互补**——这是本项目的核心结论之一。

### 3.6 RRF（排名融合）与 Rerank（精排）

- **RRF（Reciprocal Rank Fusion）**：把向量检索和全文检索各自的排名榜，用一个公式
  融合成一个总榜。好处是不用纠结两套检索的原始分数怎么比，只看排名。
- **Rerank（精排）**：召回的候选（比如 50 段）里，用一个更强的"交叉编码器"模型
  （cohere rerank）把问题和每一段**一起**重新打分，把最相关的提到最前。
  它比向量检索更准，但只能在"已召回的候选"里重排，捞不回没进候选的段落。

### 3.7 大语言模型与"网关"

- **大语言模型（LLM）**：负责"读懂检索到的段落 + 用人话组织答案"。我们用云端 DeepSeek。
- **网关（Gateway）**：我们自己用 FastAPI 写的中间服务，把上面所有环节串起来，
  对外提供一个标准的 OpenAI 兼容接口，让网页界面（Open WebUI）能直接连。

---

## 第四部分：方案选型论证

每个技术选择都是权衡后的结果，不是拍脑袋。下面逐一说清"选了什么、为什么、
排除了什么"。

### 4.1 整体架构：本地 vs 云端

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| 全本地（板子跑一切） | 不依赖网络、数据不外传 | 板子跑不动大模型/大嵌入（实测证伪） | ❌ 排除 |
| 全云端（含向量库） | 省心 | 数据全外传、依赖网络、成本高 | ❌ 过度 |
| **混合（板子存储+编排，云端算）** | 板子物尽其用、重活交云端 | 嵌入/生成依赖云端网络 | ✅ **采用** |

### 4.2 向量库：为什么选 PostgreSQL + pgvector

早期本地方案用的是 Qdrant（专门的向量数据库）。转云端方案时改用
**PostgreSQL + pgvector**，原因：

- **一库多用**：PostgreSQL 既能存向量，又能存原文、标题、来源、序号等结构化字段，
  还能顺便做全文检索（FTS）——一个库搞定 Hybrid 检索的两条腿，不用再维护第二个系统；
- **运维主流**：PostgreSQL 是最成熟的开源数据库，资料多、稳定；
- **板子扛得住**：向量存储检索是轻活，板子 12 核 30GB 绰绰有余。

### 4.3 嵌入模型：为什么是云端 qwen3-embedding-8b，维度为什么是 1024

- **为什么云端**：板子本地跑嵌入这条路已被证伪（NPU 量化质量崩、CPU 太慢，见 1.3）；
- **为什么这个模型**：qwen3-embedding-8b 是较强的多语言嵌入模型，中英文对齐好
  （后面验证了中文问能命中英文块）；
- **为什么 1024 维**：这个模型默认输出 **4096 维**，但 **pgvector 的 HNSW 索引上限是
  2000 维**，4096 维建不了索引。好在该模型支持 MRL（套娃表示学习）截断，
  调用时传 `dimensions:1024` 就能输出 1024 维——既能建索引，质量也够用。

### 4.4 生成模型：为什么云端 DeepSeek

本地 Qwen2.5-3B 的能力上限实测扛不住"列全核心库"这类需要完整长列表的问题
（Q4 会写错名字、Q8 仍会漏项），7B 又太慢。云端 DeepSeek deepseek-flash
能力足够，稳定。注意它是**带思考（reasoning）的模型**，调用要给足 token（见选型坑）。

### 4.5 检索方式：为什么是 Hybrid（这是用实验选出来的）

这是全项目最硬核的决策——**不是我们主观选的，是用 12 题评测集跑五组对照实验，
用数据选出来的**（完整实验见第八部分）。简要结论：

| 检索方式 | Recall@10 | MRR@10 |
|---|---|---|
| 纯向量 | 83% | 0.384 |
| 纯全文检索 | 50% | 0.326 |
| 向量+全文+RRF | 92% | 0.519 |
| **+ 精排（最终）** | **92%** | **0.729** |

Hybrid+精排 比任何单一方式都好，且用全文检索替代了早期"query 改写"的作用，
更快更稳。

### 4.6 只建英文索引：为什么敢这么做

文档每个子项目都有英文（源语言）和中文（翻译）两版。我们**只索引英文**，
依据是实测的跨语言对齐——**用中文问题去检索纯英文块**：

| 中文问题 | 命中的英文块 | 余弦相似度 |
|---|---|---|
| FlagCX 是什么？架构如何？ | flagcx-overview > Architecture | 0.79 |
| 如何编译安装 FlagCX？ | user-guide > Installation | 0.85 |
| （英文同问对照） | 同一块 | 0.87 |

中文问命中正确英文块，相似度 0.74~0.85，接近英文直问的 0.87。
**所以只建英文就够**：省一半嵌入时间和成本、消除中英重复块互相挤占，
中文问答靠大模型用中文组织英文内容即可。语料从 1287 篇降到 859 篇英文。

---

## 第五部分：环境与材料清单

### 5.1 硬件与系统

| 项 | 规格 |
|---|---|
| 板子 | Radxa Orion O6N（CIX P1 芯片） |
| 系统 | Debian 12 (bookworm) |
| CPU | ARM64，12 核 |
| 内存 | 30 GB 可用 |
| 存储 | 1 TB NVMe |
| 访问方式 | SSH（示例 `baai@192.168.15.195`） |

### 5.2 板子上要装的软件

| 软件 | 版本 | 作用 |
|---|---|---|
| PostgreSQL | 15 | 向量库 + 全文检索 |
| pgvector | 0.8.7（源码编译） | 给 PostgreSQL 加向量能力 |
| Python | 3.11（venv） | 跑网关 |
| FastAPI/uvicorn | — | 网关 Web 框架 |
| psycopg2 / httpx | — | 连数据库 / 调云端 API |
| langchain-text-splitters | — | 切块 |
| Open WebUI | Docker，`:3000` | 网页问答界面 |
| llama-server | 可选，`:8081` | 本地小模型备用后端 |

### 5.3 要申请的云端 API（3 个，都是 OpenAI 兼容接口）

| 用途 | 服务 | 模型 | Key 形如 |
|---|---|---|---|
| 嵌入 | OpenRouter | qwen/qwen3-embedding-8b | `sk-or-...` |
| 精排 | OpenRouter（同上 Key） | cohere/rerank-v3.5 | 同上 |
| 生成 | DeepSeek | deepseek-flash | `sk-...` |

### 5.4 语料

你的文档。我们的是 `/home/baai/rag-test/docs/docs` 下的文件：
- 总计 35369 个文件、2.2GB，但**绝大多数是 HTML 构建产物**（29497 个）；
- **真正有用的是 1287 篇 Markdown 源文档**；
- 只建英文后为 **859 篇**，切块后约 **13896 块 / 853 文档**。

---

## 第六部分：系统架构与各环节作用

### 6.1 架构全景图

```
┌─────────────────────────────────────────────────────────────┐
│ 用户浏览器：Open WebUI (:3000)                                │
└───────────────┬─────────────────────────────────────────────┘
                │ OpenAI 兼容请求 (model = qwen-rag)
                ▼
┌─────────────────────────────────────────────────────────────┐
│ FastAPI 网关 (:8000，跑在板子上)   app/main.py + cloud_rag.py │
│                                                               │
│  0. 过滤 Open WebUI 的系统请求(生成标题/标签)                 │
│  1. 嵌入问题 ───────────────────────► 云端 OpenRouter        │
│  2. 向量检索(Dense) ──┐                qwen3-embedding-8b     │
│  3. 全文检索(FTS) ────┤► 板子本地 PostgreSQL + pgvector       │
│  4. RRF 融合 ─────────┘                                       │
│  5. 精排(Rerank) ──────────────────► 云端 OpenRouter          │
│                                        cohere/rerank-v3.5     │
│  6. 拼上下文+规则，调生成 ──────────► 云端 DeepSeek           │
│                                        deepseek-flash         │
└─────────────────────────────────────────────────────────────┘
       ▲                                         │
       │                                         ▼
  配置 /opt/edge-ai-lab/.env            答案返回浏览器
```

**分工原则**：板子做轻活（编排 + 向量/全文存储检索），云端做重活（嵌入/精排/生成）。

### 6.2 一次完整问答的数据流（以"how to install FlagCX?"为例）

1. **用户提问**：浏览器把问题发给网关 `/v1/chat/completions`；
2. **网关过滤**：识别并放行真实提问（踢掉 Open WebUI 自动发的"生成标题"等请求）；
3. **嵌入**：把问题发给云端嵌入 API，得到 1024 维向量（作用：把问题变成可比较的语义）；
4. **向量检索**：用该向量在 pgvector 找语义最近的 50 块（作用：按意思召回）；
5. **全文检索**：同时用 "flagcx install" 在 PostgreSQL 全文检索找 50 块
   （作用：按关键词召回——这题正是靠它精准命中安装文档）；
6. **RRF 融合**：把两路结果按排名融合去重（作用：取长补短）；
7. **精排**：把融合后的候选发给云端 rerank，重新打分取最相关 8 块（作用：优中选优）；
8. **生成**：把这 8 块 + 问题 + 规则（只据文档、不编造、用提问语言）发 DeepSeek
   （作用：组织成人话答案）；
9. **返回**：答案经网关回浏览器。

---

## 第七部分：逐步部署全流程（结合真实测试举例）

> 每一步都给：**做什么命令 → 这步的作用 → 我们实测时遇到了什么**。
> 命令都在板子上执行（先 `ssh baai@192.168.15.195` 登录）。

### 步骤 0：准备 Python 环境

```bash
source /opt/edge-ai-lab/venv/bin/activate
pip install -i https://mirrors.aliyun.com/pypi/simple/ \
    fastapi uvicorn psycopg2-binary httpx "numpy<2" \
    langchain-text-splitters transformers
```
**作用**：装网关和建索引要用的 Python 库。
**实测举例**：直连 PyPI 官方源 **必然超时**（板子的网络环境只放行了部分国内镜像）。
必须加 `-i https://mirrors.aliyun.com/pypi/simple/` 走阿里云镜像才能装上。

### 步骤 1：安装 PostgreSQL

```bash
# 1.1 换 apt 源为华为云镜像
sudo cp /etc/apt/sources.list.d/50-bookworm.list{,.bak}
sudo sed -i "s#https://deb.debian.org/debian-security#https://mirrors.huaweicloud.com/debian-security#g; \
             s#https://deb.debian.org/debian#https://mirrors.huaweicloud.com/debian#g" \
             /etc/apt/sources.list.d/*.list
sudo apt-get update

# 1.2 安装 PostgreSQL 15 + 编译工具（pgvector 要编译）
sudo apt-get install -y --fix-missing \
     postgresql postgresql-contrib postgresql-server-dev-15 \
     build-essential git make gcc
```
**作用**：装数据库本体 + 后面编译 pgvector 需要的工具链。
**实测举例**：
- 一开始 `apt-get install` 报 `Failed to fetch ... deb.debian.org ... Network is unreachable`
  ——板子连不上 Debian 官方源。我们测了多个镜像，**华为云 https 最快（0.14 秒）**，
  清华/中科大连不上，于是换华为云。
- 换源后装 PostgreSQL 时**又偶尔断**（security 仓库的包），加 `--fix-missing` 重试一次就补全了
  ——这是板子网络间歇性抖动的典型表现，全程反复遇到。

### 步骤 2：编译安装 pgvector

```bash
cd /tmp
git clone --depth 1 https://github.com/pgvector/pgvector.git
cd pgvector
sudo make
sudo make install
# 验证
ls -l /usr/lib/postgresql/15/lib/vector.so   # 应存在
```
**作用**：pgvector 是给 PostgreSQL 增加"向量类型 + 向量相似度索引"的扩展。
**实测举例**：Debian 的 apt 源里**没有** pgvector 现成包（`apt-cache policy postgresql-15-pgvector`
为空），所以只能源码编译。幸好板子装了 gcc/make，GitHub 也能直连，编译一次成功，
装出 `vector.so` 和 0.8.7 版的扩展 SQL。

### 步骤 3：建数据库、表、向量索引

```bash
# 3.1 建登录角色和数据库
sudo -u postgres psql -c "CREATE ROLE raguser LOGIN PASSWORD 'ragpass';"
sudo -u postgres createdb -O raguser ragdb

# 3.2 启用扩展、建表、建 HNSW 索引（以 postgres 身份执行）
sudo -u postgres psql -d ragdb <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE rag_chunks (
  id        bigserial PRIMARY KEY,
  source    text NOT NULL,      -- 来自哪篇文档，如 flagos_homepage/overview.md
  title     text,               -- 章节标题路径，如 "FlagOS Overview > What is FlagOS?"
  chunk_idx int,                -- 块在文档内的序号
  content   text NOT NULL,      -- 块的正文
  embedding vector(1024)        -- 1024 维向量
);
CREATE INDEX rag_chunks_hnsw ON rag_chunks
  USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
ALTER TABLE rag_chunks OWNER TO raguser;
SQL
```
**作用**：建一张表存"文档块 + 向量 + 元数据"；HNSW 索引让"找最相似的 N 块"变毫秒级。
**字段设计用意**：`title` 存完整章节路径（不只末级标题），后面全文检索和理解归属都靠它。
**实测举例**：最后那句 `ALTER TABLE ... OWNER TO raguser` 很关键——我们一开始漏了，
后面用 raguser 给表加全文检索列时报 `must be owner of table rag_chunks`，
只好再用 postgres 身份补做 DDL。**建表时就把 owner 设对，省后患。**

### 步骤 4：配置云端 API Key

建 `/opt/edge-ai-lab/.env`：
```ini
# 嵌入：OpenRouter qwen3-embedding-8b
EMBED_BASE_URL=https://openrouter.ai/api/v1
EMBED_API_KEY=你的OpenRouterKey
EMBED_MODEL=qwen/qwen3-embedding-8b
EMBED_DIM=1024
# 生成：DeepSeek
CHAT_BASE_URL=https://api.deepseek.com
CHAT_API_KEY=你的DeepSeekKey
CHAT_MODEL=deepseek-flash
# 精排：OpenRouter cohere（复用 EMBED_API_KEY）
RERANK_BASE_URL=https://openrouter.ai/api/v1
RERANK_MODEL=cohere/rerank-v3.5
# 向量库连接串
PG_DSN=postgresql://raguser:ragpass@localhost:5432/ragdb
```
```bash
chmod 600 /opt/edge-ai-lab/.env   # 只有自己能读，保护 Key
```
**作用**：把所有云端配置和数据库连接集中管理，代码从这里读，Key 不写进代码。
**实测举例**（两个真实验证出来的坑）：
- **维度**：我们实测 qwen3-embedding-8b 默认返回 **4096 维**，传 `dimensions:1024` 返回 1024 维。
  必须用 1024，否则超 pgvector 的 HNSW 2000 维上限、建不了索引。
- **deepseek-flash 的 max_tokens**：这是个带 reasoning 的模型。实测它调用一次先花 ~676 token
  "思考"，如果 `max_tokens` 只给 200，思考就把额度吃光、**正式回答是空的**
  （`finish_reason=length`）。必须给足（≥1024）。

### 步骤 5：建索引（把文档灌进向量库）

写一个 `build_index.py`，核心三件事：切块 → 调嵌入 API → 批量入库。关键代码骨架：

```python
# 切块：先按 Markdown 标题层级切，每节再按 700 字切，每块前面带标题路径
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
hsplit = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#","H1"),("##","H2"),("###","H3"),("####","H4")],
    strip_headers=False)
csplit = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)

# 只建英文 + 排除构建产物
#   排除 /_build/ /_sources/ /_static/ 等（HTML 导出副本，内容重复）
#   排除 *_zh 目录（中文翻译版，只建英文源）

# 嵌入：批量调 OpenRouter，dimensions=1024，顽强重试
def embed_batch(texts, retries=8):
    for i in range(retries):
        try: ...                       # POST /embeddings
        except: time.sleep(min(5*(i+1),30))   # 指数退避抗网络抖动

# 入库：execute_values 批量 INSERT；每篇 commit 一次（断点续传）
```
运行（后台 + 日志）：
```bash
nohup python3 -I build_index.py > build.log 2>&1 &
```
**作用**：把 859 篇英文文档切成约 1.4 万块，逐块算向量、连同原文和标题存进 `rag_chunks`。
**实测数据**：全程约 **1.5 小时**（云端嵌入 API，比板子 CPU 的 6 小时快很多）。
**实测举例（这步踩了好几个坑）**：
- **脚本被网络抖动打崩**：最初重试写得弱，一次 `TimeoutError` 穿透重试把整个脚本干掉，
  当时已入库 599/859 篇。→ 改成"重试 8 次 + 退避 + 单篇失败只跳过不崩溃 + 断点续传
  （按 source 跳过已入库）"，重启后从第 600 篇接着跑，不白费。
- **误索引构建产物**：第一次把 `_build/`、`_sources/` 里的 HTML 导出副本也索引了
  （62 个重复块），这些和正文重复、会污染检索。→ 加目录排除。
- **只建英文**：把 `_zh` 目录排除，859 篇而非 1287 篇，省一半。

### 步骤 6：加全文检索（FTS）能力

```bash
sudo -u postgres psql -d ragdb <<'SQL'
ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector;
UPDATE rag_chunks SET content_tsv =
   to_tsvector('english', coalesce(title,'') || ' ' || content);
CREATE INDEX rag_chunks_fts ON rag_chunks USING GIN (content_tsv);
SQL
```
**作用**：给每块生成一个"分词后的全文索引（tsvector）"，让数据库支持按关键词检索。
标题也放进去（`coalesce(title,'')||content`），这样"install"能同时匹配正文和章节标题。
**实测举例**：加完立刻验证 `flagcx install` → FTS 精准命中
`Quick Setup > Install from Source > Install FlagCX`。而对照 `what is flagos`，
FTS 也只能召回一堆泛泛的 "FlagOS" 块（印证：FTS 对高频通用词无能，对具体关键词强）。
**注意**：用 postgres 身份执行（接步骤 3 的 owner 坑）。

### 步骤 7：部署网关

两个文件放到 `/opt/edge-ai-lab/services/api/app/`：
- **`cloud_rag.py`**：封装检索 `CloudRAG.search()` —— 嵌入 → 向量检索 + 全文检索
  → RRF 融合 → 精排，并带"云端断网快速失败"降级；
- **`main.py`**：FastAPI，提供 `/v1/chat/completions`，`qwen-rag` 模型走 RAG，
  并过滤 Open WebUI 的系统请求。

启动（彻底脱离终端，否则 SSH 一断就被带走）：
```bash
cat > /opt/edge-ai-lab/services/api/run_gw.sh <<'EOF'
#!/bin/bash
cd /opt/edge-ai-lab/services/api
source /opt/edge-ai-lab/venv/bin/activate
export LLAMA_SERVER_URL=http://localhost:8081
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
EOF
chmod +x /opt/edge-ai-lab/services/api/run_gw.sh
setsid nohup /opt/edge-ai-lab/services/api/run_gw.sh \
       > /opt/edge-ai-lab/logs/gateway.log 2>&1 </dev/null & disown
# 验证
curl -s http://localhost:8000/health        # {"status":"ok",...}
curl -s http://localhost:8000/v1/models      # 应含 qwen-rag
```
**作用**：把检索和生成串成一个标准 OpenAI 接口，供界面调用。
**实测举例**：我们一度遇到"网关启动后 health 正常，几秒后又变 DOWN、界面报 model not found"，
排查发现是后台进程没脱离干净，SSH 通道一关就把它带走了。改用
`setsid nohup ... & disown` + 独立启动脚本后稳住。

### 步骤 8：接 Open WebUI 网页界面

Open WebUI（Docker，端口 3000）里：
1. 设置 → 添加 "OpenAI 兼容" 连接，URL 填 `http://<板子地址>:8000/v1`（随便填个 API Key）；
2. 新建对话，模型下拉选 **`qwen-rag`**；
3. 提问即可。

**作用**：给用户一个聊天式的问答入口。
**实测举例**：Open WebUI 会**自动**发一些系统请求给网关——比如"给这段对话生成一个标题"、
"生成几个标签"、"生成追问建议"。这些**不是用户真实提问**，却也走了完整 RAG
（白跑嵌入+检索+精排，还和真实提问抢云端配额、加剧卡顿）。
→ 网关里识别这些请求的特征（如含 `### Task: Generate a concise title`）并直接转发普通
后端、不走 RAG。这是实测日志里发现的，不看日志根本想不到。

---

## 第八部分：准确性优化全过程（本报告的核心）

系统搭起来后，具体问题（FlagCX 怎么装、FlagGems 是什么）答得不错，但
**`what is flagos?` 这个最基础的问题，一开始完全答不出**。
下面是把它从"答不出"调到"11 秒准确答出"的完整过程——这也是整个项目最有价值的部分，
因为它展示了"**如何用数据定位 RAG 问题在哪一层**"。

### 8.1 现象

问 `what is flagos?`，模型回答："参考段落里没有 FlagOS 的定义，我无法回答"，
或者勉强拼凑成"FlagOS 是个文档中心"（错的）。

### 8.2 用"正确块排名"逐层诊断

RAG 答不准，无非三层出问题：**检索没找到 → 融合排序靠后 → 生成没用好**。
我们不猜，直接**追踪"那个正确的定义块，在每一层排第几"**。

**第一层发现——切块太碎**：
`overview.md`（约 2400 词）被"按标题切 + 每节再按 700 字切"两道工序切成了 **19 块**。
其中"核心库"那一节内容最长，被 700 字规则切成了 **8 块（标题路径完全相同）**。
后果：系统每次只取 top-8 喂模型，"核心库"答案散在 8 块里，不可能一次召回全
→ 问"有哪些核心库"必然漏项。
→ **解法**：把文档的深层缩进列表**重组成标题层级**，每个组件一个 `###`，自成一块。
重组后 **19 块变 37 块**，但每块干净自包含。

**第二层发现——定义块被海量噪声淹没**：
查 `what is flagos`，那个权威定义块在**全库排第 176 名**。top-20 全是
**300+ 张 flagrelease 模型卡**（每张都模板化复读"本模型由 FlagGems 加速…FlagOS…"）
和上百篇芯片适配指南。它们密集提及 "FlagOS"，语义上都沾边，把真正"定义 FlagOS"的
那块挤到了 176 名。
→ **解法**：SQL 层硬过滤噪声目录：
```sql
WHERE source NOT LIKE '%/model_readmes/%'
  AND source NOT LIKE '%/chip_adaptation_guide/%'
  AND source NOT LIKE '%/references/test/%'
```
过滤后，`what is flaggems` 立刻命中权威块（0.96 排第一），**耗时还从 75s 降到 22s**
（候选池干净了）。

**第三层发现——真正的根因：高频通用词导致区分度不足**：
重组 + 过滤后，`what is flagos` 的定义块**仍只排第 70 名、相似度 0.634**，
前面全是叫 "Documentation"、"FlagOS Core Libraries" 的块（0.70+）。为什么？

先排除一个常见但错误的直觉——"问句 what is 和答案 X is a 字面不像所以匹配不上"。
这不成立：所有 "what is X" 问题都这样，文档不可能把答案写成 "flagos is what"；
且嵌入模型本就是训练来把问句对齐到陈述式答案的，`what is flaggems` 能答对即是明证。

真正的原因是**高频通用词导致区分度不足**。铁证是对照：同样句式，
`what is flaggems` 定义块排第 26、能答对；`what is flagos` 排第 70、答不对。
差别只在词——"FlagGems" 相对集中、竞争者少，定义块能冒头；而 "FlagOS" 在 853 篇里
篇篇都有，数百个块都"与 FlagOS 相关"，那句真正的定义被淹没在大量"仅顺带提及"的块里。

→ **这是纯向量检索面对高频通用词的固有局限，不是文档或模型的 bug。** 改文档、改切块都绕不过去，
因为它们都依赖"问题和内容向量要接近"，而高频词恰恰让定义块在相似度上冒不出头。

### 8.3 第一版解法：Query 改写（HyDE 思路）

不碰文档，在检索侧解决：让大模型把"what is flagos"改写成**答案会用的陈述句**，
再用改写后的句子检索。DeepSeek 把它改写成：
```
FlagOS is an open-source AI system software stack
FlagOS provides a unified software stack for AI
FlagOS definition and overview
```
这些陈述句和定义块内容字面高度一致 → 检索必然命中。
**效果**：定义块命中分从 **0.81 → 0.95 排第一，答对了**。

**但踩了个坑**：改写一度返回空。实测 deepseek-flash 的 reasoning 行为：

| max_tokens | content | reasoning_tokens | finish_reason |
|---|---|---|---|
| 200 | 空 | 200（全被思考吃光） | length（截断） |
| 800 | 正常输出 3 条改写 | 676 | stop |

它先花 ~676 token 思考，改写只给 200 全被吃光 → 空改写 → 退回原问 → 还是答不出。
→ 把改写的 max_tokens 提到 1024。

代价：改写多一次 LLM 调用，单次问答变 **20-30 秒**，还多依赖一次爱抖的 OpenRouter。

### 8.4 第二版（最终）：用对照实验推翻改写，定稿 Hybrid

不满足于"能用"。我们建了 **12 题评测集**（定义/枚举/操作/综合 四类各 3 题，
每题标注语料里真实存在的"正确块"），跑 **五组检索对照实验**，指标 Recall@10 / MRR@10
（见第九部分完整数据）。核心结论：

- **纯向量 83%、纯全文 50%，RRF 融合后 92%**——两者互补（向量漏的全文补，反之亦然）；
- **精排把 MRR 从 0.519 提到 0.729**——把正确块顶到第 1-2 名；
- **Query 改写无额外价值**：Hybrid+精排(MRR 0.729) ≈ 改写方案(0.736)，
  但 Hybrid **不需要那次慢改写**——全文检索天然替代了改写的作用。

→ **定稿：Dense + FTS + RRF + Rerank（去掉 Query 改写）**。
效果持平、**速度从 20-30 秒降到 10-11 秒**、少依赖一次 OpenRouter。

### 8.5 优化全过程一览

| 阶段 | `what is flagos` 定义块状态 | 结果 |
|---|---|---|
| 起点 | 全库第 176 名 | ❌ 答不出 |
| 切块重组 | 仍排不上 | ❌ |
| 噪声过滤 | 第 70 名、0.634 | ❌ |
| Query 改写 | 0.95 排第 1 | ✅ 但慢(20-30s) |
| **Hybrid+精排（最终）** | **排第 1** | ✅ **11s** |

**一句话**：定义块从"第 176 名、模型够不着"→"第 1 名、11 秒答对"。
核心突破是**认清"不是模型/文档的错，是检索范式对泛问的固有局限"**，
再用 Hybrid 检索解决，而不是靠改文档或调提示词硬凑。

---

## 第九部分：检索对照实验（数据详情）

这是"为什么选 Hybrid"的完整证据。

### 9.1 实验设计

- **评测集**：12 题，四类各 3 题，每题标注语料里真实存在的"正确块"
  （用 source+title 模式匹配，重复文档两套都算命中）；
- **五组方案**：
  - B 纯 Dense（向量）
  - C 纯 FTS（全文检索）
  - D Dense+FTS+RRF
  - E D + 精排（候选最终方案）
  - A 改写+Dense+精排（旧生产方案，对照）
- **指标**：Recall@10（正确块是否进前 10）、MRR@10（第一个正确块的排名倒数）。

### 9.2 汇总结果（12 题平均）

| 组 | 检索方式 | Recall@10 | MRR@10 |
|---|---|---|---|
| B | 纯 Dense | 83% (10/12) | 0.384 |
| C | 纯 FTS | 50% (6/12) | 0.326 |
| D | Dense+FTS+RRF | 92% (11/12) | 0.519 |
| **E** | **D+Rerank** | **92%** | **0.729** |
| A | 改写+Dense+Rerank | 92% | 0.736 |

### 9.3 逐题排名（关键题，看"不同问题该用哪种检索"）

| 题 | 类型 | B 向量 | C 全文 | D RRF | E +精排 |
|---|---|---|---|---|---|
| what is FlagOS | 定义 | 3 | 漏 | 3 | **1** |
| what is FlagGems | 定义 | 漏 | 漏 | 5 | 4 |
| what is FlagCX | 定义 | 9 | 漏 | 4 | **2** |
| install FlagCX | 操作 | **1** | 3 | 1 | 1 |
| how to use FlagGems | 操作 | 8 | **1** | 1 | 2 |
| 支持异构芯片 | 综合 | 漏 | 漏 | 漏 | **1** |
| FlagTree vs FlagGems | 综合 | 5 | 漏 | 9 | 漏 |

### 9.4 从数据得出的四条结论

1. **不同问题该用不同检索**（核心假设被证实）：
   - 操作类（how to use FlagGems）：**全文检索排第 1，向量排第 8**——关键词碾压；
   - 定义/综合类（what is、关系）：**全文检索全漏**（词太泛），靠向量；
   - RRF 让两者互补：Recall 从 83%/50% 融合到 92%。
2. **精排显著提精度**：MRR 0.519→0.729，正确块从"在前 10"顶到"第 1-2"。
3. **Query 改写无额外增益**：E≈A，但 E 省掉那次慢且不稳的改写。
4. **唯一失败题（FlagTree vs FlagGems 关系）**：各法皆弱，属"跨块综合"难题，
   检索层解决不了——这类问题需要更复杂的手段（或语料本就没有直接讲这个关系的块）。

---

## 第十部分：踩坑总结（全部真实踩过，按类别归纳）

### 10.1 架构级大坑

**坑 A：想把嵌入模型跑到板子 NPU 上——此路不通（但结论很有价值）**
- FP32 嵌入模型（bge-m3）上 NPU：**1236 个节点被拒**，原因是动态形状（NPU 要静态形状）；
- 固定形状为 [1,512] 后：仍有 **192 个 MatMul 被拒**，原因是"NPU 只吃 INT8"；
- INT8 量化（per-tensor）后：MatMul 过关了，**但嵌入余弦相似度崩到 0.51**
  （per-tensor 量化扛不住 Transformer 的激活离群值；CNN 没这问题，所以视觉模型能量化）；
- **教训**：这块 NPU 适合 CNN 视觉，不适合大 Transformer 嵌入。老实用云端 API。

**坑 B：云端服务不稳会拖垮整个网关（必须做"快速失败"）**
- 实测 OpenRouter 从板子连，**5 次有 3 次超时**；
- 最初嵌入设 `timeout=60s × 重试 5 次`，OpenRouter 一断，一次请求**死等近 5 分钟**，
  把网关线程占满 → `/health` 不响应 → 界面报 "model not found"；
- **解法**：所有云端调用改"快速失败"——超时降到 15s、重试降到 2 次，断网几秒内放弃；
  且向量检索失败还能退化成纯全文检索，不致整个瘫痪。

### 10.2 准确性相关坑（第八部分已详述，这里归纳）

| 坑 | 根因 | 解法 |
|---|---|---|
| 核心库问题漏项 | 切块太碎，答案散在 8 块 | 文档重组成标题层级（19→37 块） |
| 定义块排 176 名 | 300+ 模型卡/适配指南刷屏"FlagOS" | SQL 过滤噪声目录 |
| 定义块仍排 70 名 | 高频通用词(flagos 篇篇都有)导致向量区分度不足 | Hybrid 检索（最终方案） |
| 改写返回空 | deepseek reasoning 吃光 token | max_tokens 200→1024 |
| 中英文双版互相挤占 | 重复索引 | 只建英文（跨语言对齐已验证） |

### 10.3 环境与工程坑

| 坑 | 现象 | 解法 |
|---|---|---|
| PyPI 超时 | pip 装不上 | 换阿里云镜像 `-i mirrors.aliyun.com` |
| apt 官方源连不上 | `deb.debian.org unreachable` | 换华为云镜像 |
| apt 下载偶尔断 | security 包 fetch 失败 | `--fix-missing` 重试 |
| pgvector 无 apt 包 | 装不了 | 源码编译 |
| 无权改表 | `must be owner of table` | 建表时 `OWNER TO raguser` |
| 2GB+ 模型序列化失败 | `protobuf ... > 2GB` | `save_as_external_data=True` |
| per-channel 量化报错 | `opset < 13` | 退回 per-tensor |
| 建索引脚本崩溃 | 一次超时穿透重试把脚本打死 | 重试 8 次+退避+单篇失败跳过+断点续传 |
| 误索引构建产物 | `_build/_sources` 重复块 | 目录排除 |
| 网关被 SSH 带走 | 启动后又 shutdown | `setsid nohup ... & disown` |
| Open WebUI 乱发请求 | 标题/标签生成也走 RAG | 网关识别并过滤系统请求 |
| 多轮对话污染检索 | 带整段历史、前话题带偏 | 只用最后一条 user 消息检索 |

### 10.4 方法论层面的总教训

1. **RAG 的准确性首先取决于检索，不是提示词**。定义块排 176 名时，提示词再怎么调都没用
   ——因为模型手里根本没有正确段落。**先保证检索把对的块找出来。**
2. **"提到 X" ≠ "定义 X"**。向量检索分不清这两者，通用词尤其严重。这是范式局限，不是 bug。
3. **不同问题该用不同检索**：具体关键词问题靠全文检索，语义/定义问题靠向量，Hybrid 兼得。
4. **用数据定位问题在哪一层**：追踪"正确块的排名"，就知道是检索漏了、融合排后了、
   还是生成没用好——比反复瞎调提示词高效十倍。
5. **不要为迁就检索去糟蹋正常文档**（往文档里塞问法、造冗余都是歪招）；
   该在检索侧解决（Hybrid / query 改写）。
6. **边缘设备要认清能力边界**：板子做编排和存储，重活交云端——物尽其用。

---

## 第十一部分：最终效果与验收

### 11.1 实测问答效果

| 问题（类型） | 回答 | 耗时 |
|---|---|---|
| what is flagos?（定义，中/英） | "FlagOS 是全开源、面向异构 AI 芯片的统一软件栈，一次开发、到处部署…" | 11s |
| how to install FlagCX?（操作） | 直接给出 `git clone … + 安装步骤`（全文检索命中安装文档） | 10s |
| FlagOS 有哪些核心库?（枚举，中文） | 完整列出算子库 / FlagTree / FlagScale / FlagCX 四类 | 10s |

全部达到需求：能问答、答得准（不编造）、中英文都行、板子只做编排。

### 11.2 关键指标

- 检索质量：Recall@10 = 92%，MRR@10 = 0.73（Hybrid+精排）；
- 单次问答延迟：10-11 秒（相比改写方案的 20-30 秒快一倍）；
- 语料规模：859 篇英文 → 13896 块，pgvector + HNSW 毫秒级检索。

### 11.3 已知局限（诚实记录）

- **嵌入依赖 OpenRouter，而它在板子网络下不稳**（5 次断 3 次）→ 问答偶发失败，
  已做快速失败降级（断网时快速返回而非卡死），但根本稳定性取决于网络；
- **"跨块综合"类问题**（如"X 和 Y 的关系"）检索层难解，需更复杂手段；
- **超通用词的宏观定义**（what is + 满篇都是的词）是 RAG 固有难点，靠 Hybrid 缓解但非银弹；
- 板子重启后网关/数据库需手动起（未配开机自启）；
- 增量索引（文档改/删的处理）尚未实现。

---

## 附录 A：关键坐标速查

| 项 | 值 |
|---|---|
| 向量库 | PostgreSQL 15 + pgvector 0.8.7，库 `ragdb`，表 `rag_chunks` |
| 向量索引 | HNSW，`vector_cosine_ops`，m=16, ef_construction=64 |
| 全文索引 | GIN on `content_tsv`（english 分词） |
| 网关 | FastAPI `:8000`，`app/main.py` + `app/cloud_rag.py` |
| 界面 | Open WebUI `:3000`，模型选 `qwen-rag` |
| 配置 | `/opt/edge-ai-lab/.env`（chmod 600） |
| 嵌入 | OpenRouter qwen3-embedding-8b，dim=1024 |
| 生成 | DeepSeek deepseek-flash（max_tokens≥1024） |
| 精排 | OpenRouter cohere/rerank-v3.5 |
| 检索管线 | Dense + FTS + RRF + Rerank |
| 语料 | 英文 859 篇 → 13896 块 / 853 文档 |

## 附录 B：检索参数

| 参数 | 值 | 说明 |
|---|---|---|
| coarse_k | 50 | 每路粗召回数量 |
| top_k | 8 | 精排后喂给生成的块数 |
| RRF k | 60 | 融合公式常数（标准值） |
| hnsw.ef_search | 200 | HNSW 查询期精度参数 |
| 嵌入超时/重试 | 15s / 2 次 | 快速失败 |
| 精排超时/重试 | 15s / 2 次 | 失败退回向量序 |

## 附录 C：相关文件

| 文件 | 内容 |
|---|---|
| `app/cloud_rag.py` | Hybrid 检索实现（`CloudRAG.search()`） |
| `app/main.py` | 网关，qwen-rag 路由 + 抗抖动 + 过滤系统请求 |
| `eval/retrieval_experiment.py` | 五组对照实验脚本 |
| `eval/retrieval_experiment_result.md` | 实验结果与逐题数据 |
| `cloud-rag-debugging.md` | 调试过程复盘 |
| 本文件 | 完整技术报告 |

---

*报告完。本项目全过程坚持"用数据决策、不靠猜"：每一个结论都有实测排名/分数支撑，
每一个坑都记录了现象、根因与解法，供后续复现与改进参考。*
