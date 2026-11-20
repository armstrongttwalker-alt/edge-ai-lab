# 云端 RAG 调试复盘：从"答不出"到"答得准"

**日期：** 2026-11-19 ~ 2026-11-20
**目标：** 用真实大语料（FlagOS 文档站 1287 篇 md）搭一套"板子当服务器"的 RAG，
并把问答准确度调到可用。
**架构：** FastAPI 网关 + PostgreSQL/pgvector + 云端 API
（嵌入 OpenRouter `qwen/qwen3-embedding-8b` 1024 维 / 生成 DeepSeek `deepseek-flash`
/ 精排 OpenRouter `cohere/rerank-v3.5`）。

本文记录从"`what is flagos?` 完全答不出"到"中英文都答准"的完整过程，每一步都附实测数据。

---

## 一、语料与建索引阶段的坑

### 1.1 语料构成：35369 文件，真正有用的只有 1287 篇 md

`/home/baai/rag-test/docs/docs` 共 2.2GB、35369 个文件，其中：

| 类型 | 数量 | 处理 |
|---|---|---|
| html（构建产物） | 29497 | 排除 |
| md（源文档） | 1287 | **只索引这些** |
| js/png/doctree/po/mo 等 | 其余 | 排除 |

**坑：** 首次建索引误把 `_build/` `_sources/`（Sphinx 构建产物，与正文重复）也索引了，
入库 62 个重复块。→ 排除 `/_build/` `/_sources/` `/_static/` 等目录。

### 1.2 只建英文：跨语言对齐实测可行

语料每篇有 `_en`（源）和 `_zh`（翻译）两版。决定**只索引英文**（源语言、最权威），
依据是实测跨语言对齐良好——**用中文问题检索纯英文块**：

| 中文问题 | 命中英文块 | 余弦 |
|---|---|---|
| FlagCX 是什么？架构如何？ | flagcx-overview > Architecture | 0.79 |
| 如何编译安装 FlagCX？ | user-guide > Installation | 0.85 |
| （对照）英文同问 | 同一块 | 0.87 |

中文问命中正确英文块，余弦 0.74~0.85，接近英文直问的 0.87。→ 只建 en，从 1287 降到 859 篇。

### 1.3 嵌入很慢，CPU 批处理无效

`qwen3-embedding` 走 API，但建索引仍要逐块调用。全量 859 篇 ≈ 1.4 万块。
（注：本地 bge-m3 CPU 实测 1.77s/块，batch=16 仍 1.58s/块——12 核已被单次推理占满，
批处理无加速，这也是最终放弃板上本地嵌入、改用云 API 的原因之一。）

### 1.4 建索引脚本被网络抖动打崩

**坑：** 板子网络间歇性抖动，一次 `TimeoutError` 穿透了 4 次重试把整个脚本干掉
（已入库 599/859 篇时崩溃）。
→ 修复：重试 8 次 + 指数退避（封顶 30s）+ 单篇失败不崩溃（记录跳过）+ 断点续传
（按 `source` 跳过已入库）。重启后从第 600 篇续跑，无损。

### 1.5 其它建索引坑

| 坑 | 原因 | 解法 |
|---|---|---|
| apt 装 PostgreSQL 失败 | `deb.debian.org` 连不上 | 换华为云镜像源 |
| security pool 下载偶尔断 | 网络抖动 | `apt-get install --fix-missing` 重试 |
| pgvector apt 无包 | Debian 源未收录 | 源码编译（pg_config + make） |
| paramiko 启动后台进程卡住 | `read()` 阻塞等 EOF | `setsid` 脱离 + 不阻塞读 |

---

## 二、核心难题：`what is flagos?` 为什么答不出

建好索引后，具体问题（FlagCX 怎么用、FlagGems 是什么）答得不错，但
**`what is flagos?` 和 `FlagOS 有哪些核心库?` 答不出来**。层层挖下去，是四个问题叠加。

### 2.1 第一层：切块太碎——答案被劈成碎片

`overview.md`（9708 字符 ≈ 2427 token）用的切法是
`MarkdownHeaderTextSplitter`（按 `#/##/###` 标题）**+** `RecursiveCharacterTextSplitter`
（每节再按 700 字切）。两道工序串联。

**实测：overview.md 被切成 19 块。** 其中"核心库"一节内容最长（十几个库），
被第二道 700 字规则切成 **8 块（第 3~10 块），标题路径完全相同**：

```
块3  ...> Open-source core libraries   (FlagGems)
块4  ...> Open-source core libraries   (FlagGems-vllm)
...
块10 ...> Open-source core libraries   (FlagCX)
```

**后果：** 系统每次只取 top-8 块喂模型。"核心库"答案散在 8 个块里，
不可能被一次全召回 → 问"有哪些核心库"必然漏项。
`what is flagos` 的定义只在第 1 块，没被召回就答不出。

**教训：** 不是"按 ### 切"，是"按 ### 切 + 每节再按 700 字切"。第二道对长手册是对的，
但对"一节=一个完整清单"的总览文档，把清单劈碎了。

### 2.2 第二层：定义块被海量边缘文档淹没

查 `what is flagos?`（纯向量，全库滤噪声后），overview 定义块的真实排名：

```
overview.md 定义块：第 176 名
top-20 全是：各组件 index.md + flagrelease model_readmes（模型卡）
```

**原因：** 语料里有 **300+ 张 flagrelease 模型卡**（每张模板复读"本模型由 FlagGems
加速...FlagOS..."）、**上百篇 chip_adaptation_guide**。它们密集提及 "FlagOS"，
语义上和"what is flagos"都沾边，把真正"定义 FlagOS"的那篇挤到 176 名。

**验证噪声危害**：`what is flaggems?` rerank 后 top-8 一度全是模型卡（rerank 分 0.95，
因为每张卡都复读 FlagGems 介绍）——答案靠"模型卡恰好有介绍"的运气蒙对。

**解法：SQL 层硬过滤噪声目录：**
```sql
WHERE source NOT LIKE '%/model_readmes/%'
  AND source NOT LIKE '%/chip_adaptation_guide/%'
  AND source NOT LIKE '%/references/test/%'
```
过滤后 `what is flaggems?` 命中 `flaggems overview(0.96)` 排第一，**耗时也从 75s 降到 22s**
（候选池干净了）。

### 2.3 第三层：重组文档结构（不破坏正常性）

把 `overview.md` 的**深层嵌套列表**改成**标题层级**——每个核心库一个 `###`，自成一块。

**实测：重组后从 19 块变 37 块**，但每块干净自包含：
```
块1   FlagOS Overview > What is FlagOS?                          (定义)
块4   ...> FlagGems (general-purpose operator library)
块10  ...> FlagCX (communication library)
...每库独立一块，标题点名归属
```

**效果：`FlagOS 有哪些核心库?` 彻底答对**——top-8 全是 overview 块（0.90/0.87...），
每个核心库独立成块都被召回，完整列出算子库/FlagTree/FlagScale/FlagCX。8 秒答全。

**走过的弯路（都被否掉）：**
- ❌ 给定义块"塞问法"（`What is FlagOS? FlagOS是什么? ...`）—— 糟蹋正常文档，弃用；
- ❌ 强调定义块"独特性" —— 没依据，弃用；
- ❌ 发现 `## What is FlagOS?` 节混了"组件清单"段 → 拆出到独立 `## FlagOS components` 节，
  让定义块纯净。但**纯净后纯向量检索定义块仍只排第 70 名、分 0.634**，证明"内容纯不纯"
  不是关键。

### 2.4 第四层（真正的根因）：高频通用词导致区分度不足

逐块追踪 `what is flagos?` 的全库召回，定义块 **第 70 名、0.634**，而前面是：

```
#1 0.749 Documentation
#2 0.720 FlagOS > FlagOS Core Libraries
#3 0.709 FlagScale Documentation
```

对比 `what is flaggems?`：flaggems 定义块 **第 26 名、0.704** → 能答对。

**先排除一个错误直觉**：有人会以为病根是"问句 what is 和答案 X is a 字面不像"。
这站不住——所有"what is X"问题都这样，文档不可能把答案写成"flagos is what"，
若此是病根则所有定义问题都该挂；且嵌入模型本就是训练来把问句对齐到陈述式答案的，
`what is flaggems` 能答对就是证明。

**真正的根因是高频通用词导致区分度不足。** 铁证是对照：同样句式，
`what is flaggems` 定义块排第 26（答对），`what is flagos` 排第 70（答不对）——
差别只在词。"FlagGems" 相对集中，只在讲它的少数文档里出现，定义块竞争者少；
而 "FlagOS" 在 853 篇里篇篇都有，数百个块都"与 FlagOS 相关"，那句真正的定义
被淹没在大量"仅顺带提及"的块里，相似度上冒不出头。

**这是纯向量 RAG 面对高频通用词的固有局限，不是文档或嵌入模型的 bug。** 改文档/切块都绕不过去，
因为它们都依赖"问题和内容向量要接近"，而恰恰这个"接近"不成立。

---

## 三、解法：Query 改写（HyDE 思路）

不碰文档，在**检索侧**解决：用户问 `what is flagos`，先让 chat 模型把问题**改写成
"答案会用的陈述句"**，再用改写后的查询检索。

DeepSeek 把 `what is flagos?` 改写成：
```
FlagOS is an open-source AI system software stack
FlagOS provides a unified software stack for AI
FlagOS definition and overview
```
这些陈述句和定义块内容字面高度一致 → 检索必然命中。然后多查询召回合并、按原问 rerank。

### 3.1 坑：改写一度返回空（reasoning 吃光 token）

**实测 `deepseek-flash` 的 reasoning 行为：**

| max_tokens | content | reasoning_tokens | finish_reason |
|---|---|---|---|
| 200 | `''`（空） | 200 | length（被截断） |
| 800 | 正常输出 3 条改写 | 676 | stop |

deepseek-flash 每次先花 ~676 token 思考，改写只给 200 token 全被思考吃光，
正式输出为 0 → 空改写 → 退回原问 → 还是答不出。
**（这个坑之前在生成端也踩过：max_tokens 太小导致答案被 reasoning 挤空。）**
→ 修复：改写调用 max_tokens 提到 1024。

### 3.2 效果：what is flagos 命中块从 0.81 → 0.95 排第一

| 阶段 | 英文 what is flagos 命中第一块 | 结果 |
|---|---|---|
| 改写前 | overview.md 定义块排不进前列（0.81 是别的块） | 答偏（讲成 platform services） |
| 改写后 | **overview.md 定义块 0.95 排第一** | ✅ 准确定义 |

中文 `FlagOS是什么?` 定义块 0.94 → 同样答准。

**代价：** query 改写多一次 LLM 调用，单次问答从 6-8s 增至 20-30s——检索精度换延迟，
正常权衡。

---

## 四、最终方案与完整根因链

`what is flagos?` 答不出 = **四层问题叠加，每层都真实**：

| 层 | 问题 | 解法 |
|---|---|---|
| 1 | 切块太碎（深层列表被 700 字规则劈成 19 块，核心库占 8 块） | 重组成标题层级（37 块，每实体独立） |
| 2 | 定义块被 300+ 模型卡/适配指南淹没（排 176 名） | SQL 层过滤噪声目录 |
| 3 | 定义与组件清单混在一节 | 拆成独立 `## What is FlagOS?` / `## FlagOS components` |
| 4 | **高频通用词(flagos 篇篇都有)导致向量区分度不足** | **Query 改写（HyDE）/ 最终用 Hybrid** |
| 附 | 改写被 deepseek reasoning 吃光 token 返回空 | max_tokens 200→1024 |

**最终检索管线：**
```
用户问题
 → chat 改写成 3 条陈述式查询（+ 原问）
 → 各自 pgvector 粗召回 50 块（SQL 过滤噪声目录）
 → 合并去重
 → cohere/rerank-v3.5 按原问精排 top-8
 → DeepSeek 生成（max_tokens 给足，避免 reasoning 挤空）
```

## 五、关键教训

1. **RAG 的上限是检索**：模型再强，检索没把对的块递进来就白搭（定义块排 176 名时，
   任何模型都答不出）。
2. **"提到 X" ≠ "定义 X"**：纯向量分不清，大语料里通用词尤其严重。这是 RAG 范式局限，
   不是 bug。
3. **RAG 擅长具体问题，不擅长超宏观定义**：`FlagCX 怎么装` 又快又准；
   `what is <满篇都是的词>` 天然吃力，要靠 query 改写补。
4. **不要为迁就检索糟蹋文档**：塞问法、造独特性都是歪招。正解在检索侧（query 改写），
   文档保持正常。
5. **reasoning 模型要给足 max_tokens**：deepseek-flash 先思考 ~676 token，
   token 不够则正式输出为空——改写、生成两处都踩过这个坑。
6. **诊断靠数据，不靠猜**：每一层都是实测排名/分数定位的（176 名、70 名、0.634、
   reasoning_tokens=200），不是推断。

---

## 六、检索对照实验：用数据推翻 Query 改写，定稿 Hybrid

Query 改写虽让 `what is flagos` 答对，但慢（20-30s）且多依赖一次爱抖的 OpenRouter。
为了用数据判断"到底该用什么检索"，建了 12 题评测集（定义/枚举/操作/综合 四类各 3 题，
每题标注语料里真实存在的"正确块"）+ 五组对照实验，指标 Recall@10 / MRR@10。

新增能力：给 `rag_chunks` 加 `content_tsv`（tsvector，title+content）+ GIN 索引，
支持 PostgreSQL 原生全文检索（FTS，近似 BM25 的关键词检索）。

### 五组结果（12 题平均）

| 组 | 检索方式 | Recall@10 | MRR@10 |
|---|---|---|---|
| B | 纯 Dense（向量） | 83% | 0.384 |
| C | 纯 FTS（关键词） | 50% | 0.326 |
| D | Dense+FTS+RRF | 92% | 0.519 |
| **E** | **D+Rerank** | **92%** | **0.729** |
| A | 改写+Dense+Rerank（旧方案） | 92% | 0.736 |

### 关键发现

1. **不同问题该用不同检索**（实测印证）：
   - 操作类（how to install/use）：**FTS 碾压**（O3 FTS 排 1、Dense 排 8）；
   - 定义/综合类（what is / 关系）：**FTS 全漏**（flagos/gems 词太泛），靠 Dense；
   - RRF 让两者**互补** → Recall 83%/50% 融合后升到 92%。
2. **Rerank 大幅提精度**：MRR 0.519→0.729，把正确块从 top10 顶到第 1-2 名。
3. **Query 改写无额外价值**：E（Hybrid+Rerank，MRR 0.729）≈ A（改写方案，0.736），
   但 E 不需要那次慢改写 → **FTS 替代了改写的作用**（关键词匹配本就是改写想达到的效果）。

### 定稿：用 E（Hybrid）替换 A（改写）

生产检索管线改为：
```
Dense（向量，擅长定义/综合） + FTS（关键词，擅长操作）
  → RRF 融合（互补）
  → cohere rerank 精排 top-8
```
- 效果持平（MRR 0.73），**去掉改写后快一倍（20-30s → 10-11s）**，少依赖一次 OpenRouter；
- OpenRouter 抖动时 Dense 失败，仍可退化为纯 FTS，不致瘫痪；
- 实测：`what is flagos`(11s)、`how to install FlagCX`(FTS 直接命中安装文档)、
  `FlagOS 有哪些核心库`(中文) 均答准。

实验脚本与完整逐题数据见 `eval/retrieval_experiment.py` 与 `eval/retrieval_experiment_result.md`。
