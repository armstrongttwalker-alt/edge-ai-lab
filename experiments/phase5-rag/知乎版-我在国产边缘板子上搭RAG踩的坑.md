# 我在一块国产边缘板子上搭了套 RAG，结果被"FlagOS 是什么"这个问题卡了整整两天

先说结论，省得你往下翻半天：

1. **RAG 答不准，90% 的锅在"检索"，不在那个大模型，也不在你提示词写得好不好。**
2. **纯向量检索有个天坑**：问"什么是 X"这种大白话，它经常把真正的定义段落排到一百多名开外，你信不信？我实测排到了第 176 名。
3. **救命的不是换更强的模型，是上"混合检索"**（向量 + 关键词 + 融合 + 重排）。我用 12 道题做了对照实验，数据实打实。
4. 还有一个反直觉的发现：**我吭哧吭哧写的"查询改写"，被实验证明基本没用**，一个全文检索就把它的活干了，还快一倍。

下面是完整的故事，多图多代码多数据，适合搬个小板凳慢慢看。

---

## 一、先交代背景：一块"小马拉大车"的板子

我手上有块国产边缘 AI 开发板（Radxa Orion O6N，CIX P1 芯片，Debian 12���12 核 ARM，30G 内存）。手上还有一坨技术文档——FlagOS 这个开源 AI 软件栈的文档，1000 多篇 Markdown。

需求很朴素：**让人能用大白话问这坨文档**。比如：

> "FlagOS 是什么？"
> "FlagCX 怎么装？"
> "它有哪些核心库？"

系统自动去文档里找答案、用人话回答。这就是 RAG（检索增强生成）。

### 一开始我飘了，想把 AI 全塞进板子里

这板子不是有 NPU 嘛，我寻思把嵌入模型、大模型全跑在本地，多酷。

结果被现实一巴掌拍醒。我把嵌入模型 bge-m3 往 NPU 上怼：

- FP32 模型？NPU 拒收 1236 个节点，嫌你是"动态形状"（它只吃定长的）。
- 固定成定长？又拒收 192 个矩阵乘法，因为"只支持 INT8，不要你浮点"。
- 那我量化成 INT8 总行了吧？**行是行了，嵌入向量的余弦相似度直接崩到 0.51。**

0.51 什么概念？就是量化完的向量跟原来基本不是一个东西了，检索全乱套。原因是 INT8 的逐张量量化扛不住 Transformer 里那些离群的激活值——CNN 没这毛病，所以视觉模型能量化，语言/嵌入模型不行。

退回 CPU 跑？**一个文档块嵌入要 1.77 秒**，一万多块算下来建个库得 6 小时。

于是我认清现实：**这板子的 NPU 是干视觉 CNN 的料，不是干大 Transformer 的料。** 老老实实让板子当个"调度中心 + 仓库"，嵌入、生成这些重活全甩给云端 API。

这就定了整个架构的调子：

```
浏览器(Open WebUI :3000)
   └─ 网关(FastAPI :8000，跑板子上)
        ├─ 嵌入     → 云端 OpenRouter (qwen3-embedding-8b)
        ├─ 向量检索 → 板子本地 PostgreSQL + pgvector
        ├─ 关键词检索 → 板子本地 PostgreSQL 全文检索
        ├─ 融合+重排 → RRF + 云端 cohere rerank
        └─ 生成     → 云端 DeepSeek (deepseek-flash)
```

板子干轻活（编排 + 存储检索），云端干重活（嵌入 + 重排 + 生成）。

---

## 二、系统搭起来了，然后我就懵了

库建好了（859 篇英文文档，切成 13896 块，灌进 pgvector）。我兴冲冲去测。

**"FlagCX 怎么装？"** —— 答得贼好，直接给我 `git clone` + 安装步骤。
**"FlagGems 是什么？"** —— 答得也不错。

然后我问了个最简单的：

**"what is flagos?"**

它回我：

> "参考段落里没有 FlagOS 的定义，我无法回答。"

？？？我这一坨文档全是讲 FlagOS 的，你跟我说没有定义？

更离谱的是换个问法它开始瞎编，说"FlagOS 是一个文档中心"——**错的**。

就这么个小学生都会答的问题，卡了我整整两天。下面是破案过程。

---

## 三、破案：别猜，去看"正确答案排第几名"

我一开始也想当然，觉得是提示词没写好，改了半天提示词，没用。

后来学乖了，**不猜，直接量一个数**：那个真正写着"FlagOS 是什么"的文档块，在检索结果里排第几名？

这一量，真相层层剥开。

### 线索 1：文档被切得稀碎

这个光说没用，我直接把切之前和切之后的真实数据摆出来，你一看就懂。

**先看原始文档长啥样。** 我那篇 overview.md 里，"核心库"这一节是用**深层缩进列表**写的，大概是这个结构：

```markdown
### Open-source core libraries
- Operator libraries
  - **General-purpose operator library**
    - **FlagGems** (v5.4.0)
      FlagGems is a high-performance general-purpose operator library...
  - **Fused operator libraries**
    - **FlagGems-vllm** (v0.2.0)
      A high-performance operator library designed for...
    - **FlagGems-sglang** (v0.1.0) ...
  - **Multi-domain operator libraries**
    - **FlagDNN** (v0.3.0) ...
    - **FlagBLAS** (v0.3.0) ...
    - **FlagFFT / FlagSparse / FlagTensor / FlagAttention / FlagAudio** ...
- **FlagTree** (编译器) ...
- **FlagScale** (工具包) ...
- **FlagCX** (通信库) ...
```

一整节十几个库，全挤在 `### Open-source core libraries` 这一个标题底下。

**问题来了**：切块工具先按标题切（`#` `##` `###`），但这十几个库的小标题是用**列表缩进**（`-`）写的，不是真正的 Markdown 标题。切块工具不认缩进，只认 `#`。所以它一看——好家伙，整节都在一个 `###` 下面，太长了（超 700 字），那就按 700 字硬切！

**切完的真实结果（我从数据库里一条条捞出来的）**：这一节被切成了 8 块，而且这 8 块的"标题路径"**一模一样**，全是 `FlagOS Overview > ... > Open-source core libraries`：

```
块3  标题: ...> Open-source core libraries   内容: ### Open-source... FlagGems...
块4  标题: ...> Open-source core libraries   内容: - Fused... FlagGems-vllm...
块5  标题: ...> Open-source core libraries   内容: - Multi-domain... FlagDNN...
块6  标题: ...> Open-source core libraries   内容: FlagSparse... FlagTensor...
块7  标题: ...> Open-source core libraries   内容: FlagAudio...
块8  标题: ...> Open-source core libraries   内容: FlagTree is an open-source compiler...
块9  标题: ...> Open-source core libraries   内容: FlagScale...
块10 标题: ...> Open-source core libraries   内容: FlagCX...
```

看出灾难了吗？**FlagGems 在块3、FlagTree 在块8、FlagCX 在块10**，一个完整的"核心库清单"被劈成了 8 片，散落在 8 个块里。

**后果**：系统每次只拿相关性前 8 名的块喂给大模型。你问"FlagOS 有哪些核心库"，就算这 8 块全都相关，也要和别的内容抢那 8 个名额，根本凑不齐。实测结果就是——模型只列出一部分核心库，或者干脆说"文档里核心库那节没有具体条目"（因为它可能只拿到了块3、块5，没拿到块8、块10）。

**怎么治**：把那些用缩进列表写的库，**改写成真正的 Markdown 标题**。改之后长这样：

```markdown
## Open-source core libraries
### FlagGems (general-purpose operator library)
**FlagGems** (v5.4.0) is a core library of FlagOS...
### FlagGems-vllm (fused operator library)
**FlagGems-vllm** (v0.2.0) is a core library of FlagOS...
### FlagCX (communication library)
**FlagCX** (v0.14.0) is a core library of FlagOS...
```

**改完再从数据库捞一次**，每个库都成了独立的一块，标题各不相同、还点名道姓：

```
块3  标题: ...> Open-source core libraries > FlagGems (general-purpose operator library)
块4  标题: ...> Open-source core libraries > FlagGems-vllm (fused operator library)
...
块17 标题: ...> Open-source core libraries > FlagCX (communication library)
```

整篇文档从 19 块变成 37 块。**块数反而变多了，但这是好事**——以前是"一节硬切成 8 片、标题全一样、谁也说不清自己是讲啥的"，现在是"一个库一块、自带身份、各自完整"。

重组之后，"FlagOS 有哪些核心库"这类问题立马答全了。

> 小结一句：**切块不是切得越小越好，而是要让"一个完整的意思"落在"一个块"里。** 用标题承载结构，别用缩进列表——因为切块工具只认标题。

### 线索 2：定义块被 300 张"小广告"淹了（排第 176 名！）

"what is flagos"还是不行。我把定义块的全库排名一拉——

**第 176 名。**

前面 175 名都是啥？是 300 多张"模型发布卡片"，每张都复读一遍"本模型由 FlagGems 加速、基于 FlagOS……"，还有上百篇芯片适配指南。这些玩意儿满嘴"FlagOS"，把真正讲"FlagOS 是什么"的那一块挤到了姥姥家。

这就像你在搜索框打"苹果是什么"，结果前 100 条全是苹果手机的广告，真正讲水果的那条排到了第二页。

**上个真实对比你就明白了。** 我拿"what is flaggems"做实验，看重排后最终进前 8 的是哪些块——

屏蔽噪声**之前**，前 8 名长这样（我从日志里扒的，分数是重排给的相关性）：

```
#1 0.95  flagrelease/model_readmes/FlagRelease_Moonlight-16B-A3B-hygon...
#2 0.95  flagrelease/model_readmes/FlagRelease_Moonlight-16B-A3B-ascend...
#3 0.95  flagrelease/model_readmes/FlagRelease_Phi-3.5-MoE-nvidia...
#4 0.95  flagrelease/model_readmes/FlagRelease_HY-MT2-30B-nvidia...
... 前 8 名全是模型卡片
```

**整整 8 个名额全被模型卡片占了！** 为啥它们分还都 0.95 这么高？因为每张卡片里都抄了一段模板化的 FlagGems 介绍，重排一看"诶这段跟问题挺像"，就全给了高分。模型最后是靠"卡片里顺带抄的那句介绍"蒙对的——**纯属运气**，哪天卡片模板一改就崩。

屏蔽噪声**之后**，同样问题，前几名变成了：

```
#1 0.96  flaggems_en/.../FlagGems-overview.md   ← 真正的权威介绍文档！
#2 0.94  flaggems_en/index.md
#3 ...   flaggems_sglang / flaggems_vllm overview
```

**怎么治**：SQL 里直接把这些噪声目录屏蔽掉，一行搞定：

```sql
WHERE source NOT LIKE '%/model_readmes/%'
  AND source NOT LIKE '%/chip_adaptation_guide/%'
```

屏蔽完，"FlagGems 是什么"从"靠运气蒙"变成"稳稳命中权威块"（0.96 排第一），而且因为候选池干净了，检索还从 **75 秒降到 22 秒**。一行 SQL，一箭双雕。

### 线索 3：真·元凶——你问的词和答案的词"长得不像"

屏蔽噪声后，我满怀期待再查"what is flagos"。定义块排名——

**第 70 名，相似度 0.634。**

还是不行！我把"what is flagos"纯向量检索的前 10 名拉出来看（都已经屏蔽噪声了），是这样的：

```
#1  0.749  Documentation
#2  0.720  FlagOS > FlagOS Core Libraries
#3  0.709  FlagScale Documentation
#4  0.703  Use FlagGems > ...
#5  0.697  ...> FlagTree (compiler)
...
#70 0.634  FlagOS Overview > What is FlagOS?   ← 真正的定义块在这儿
```

真正写着定义的块（"What is FlagOS?"）排在**第 70 名、相似度才 0.634**，前面全是些叫"Documentation""Core Libraries"的块。

这里我要先纠正一个很多人（包括一开始的我）会有的错误直觉。你可能会想：
"是不是因为我问的是 'what **is** flagos'，而答案写的是 'FlagOS **is** a stack...'，
问句和陈述句长得不一样，所以匹配不上？"

**这个解释是错的。** 理由很简单：凡是"what is X"的问题，文档里都不可能反过来写成"X is what"，
所有定义类问题都是这样。如果"问句和陈述句长得不像"真是病根，那**所有** "what is" 问题
都该失败——可我问"what is flaggems"明明答对了。而且更重要的是，**嵌入模型本来就是专门
训练来跨越这种问答差异的**，"what is X"去匹配"X is a..."对它是基本功，不是难点。

**真正的病根是：词频，或者说"区分度"。** 上对照组你就秒懂：

```
what is flaggems  → 定义块排第 26 名（能答对）
what is flagos    → 定义块排第 70 名（答不对）
```

同样的句式、同样的模型、同样是"what is X"，为啥一个 26 名一个 70 名？**差别只在那个词：**

- "FlagGems" 这词相对集中，只在讲 FlagGems 的那些文档里出现，竞争者少，定义块容易冒头；
- "FlagOS" 这词在 853 篇文档里**篇篇都有**。于是有几百个块都"跟 FlagOS 沾边"，
  那句真正的定义，要和这几百个"顺口提了一句 FlagOS"的块抢排名，就被稀释、淹没了。

打个比方：你搜"张伟是谁"，如果全公司就一个张伟，一下就找到了；可要是公司里几百个人
的简历里都写着"曾与张伟共事"，那张伟本人的主页反而被这几百份简历埋了。

**所以根子跟"what is 句式"一点关系没有，是"高频通用词区分度太低"。** 这也解释了为什么
具体问题（how to install FlagCX）没事——那些问题带着独特的关键词，区分度高。

**到这我才明白：这不是我文档的错，也不是模型笨，是"纯向量检索"这个范式本身的天坑。** 改文档、改切块都绕不过去，因为它们都依赖"问题和答案向量得接近"，可偏偏就是不接近。

---

## 四、第一版解法：把问题"翻译"成答案的样子

既然问题和答案长得不像，那我**在搜之前，先让 AI 把问题改写成答案的口吻**不就行了？

我让 DeepSeek 把 "what is flagos" 改写成：

```
FlagOS is an open-source AI system software stack
FlagOS provides a unified software stack for AI
FlagOS definition and overview
```

拿这些陈述句去搜，跟定义块一下就对上了。**定义块命中分从排不上 → 0.95 排第一，答对了！**

这里还踩了个贼坑：**改写一开始老返回空**。查了半天发现，deepseek-flash 是个"爱思考"的模型，它答之前先在脑子里盘算（reasoning）约 676 个 token。我给改写只分配了 200 个 token 的额度，**全被它用来思考了，正式答案一个字没剩**。把额度提到 1024 才正常。

> 划重点：用带 reasoning 的模型，max_tokens 千万别抠门，不然它"想太多"把话都憋回去了。

但这版有个毛病：多了一次 AI 调用，一个问题要答 **20~30 秒**，慢。而且多依赖了一次那个三天两头抽风的 OpenRouter。

---

## 五、第二版：我做了个实验，把自己的方案给否了

我没停在"能用就行"。我想知道：查询改写到底是不是最优解？

于是建了个 **12 道题的评测集**（定义/枚举/操作/关系 各 3 题），每题标好"正确答案块",然后拿 5 种检索方案跑对照，看两个指标：Recall@10（正确块进没进前 10）和 MRR@10（正确块排得够不够靠前）。

结果如下，我把它钉在这：

| 方案 | Recall@10 | MRR@10 |
|---|---|---|
| 纯向量 | 83% | 0.384 |
| 纯关键词(全文检索) | 50% | 0.326 |
| 向量+关键词+融合(RRF) | 92% | 0.519 |
| **向量+关键词+融合+重排** | **92%** | **0.729** |
| 查询改写+向量+重排(我的旧方案) | 92% | 0.736 |

看出名堂了吗？

**发现一：向量和关键词是绝配，互补得明明白白。**
逐题看特别有意思：
- 问"how to use FlagGems"（操作类）：**关键词检索排第 1，向量才排第 8**——关键词碾压。
- 问"what is FlagGems"（定义类）：**关键词检索直接漏了，向量才行**。

俩各有各的瞎点，合一块（RRF 融合）召回率从 83%/50% 一下干到 92%。

**发现二：重排是提精度的神器。**
融合后召回率 92% 不变，但 MRR 从 0.519 飙到 0.729——就是把已经捞上来的正确块，往第 1、2 名怼。

**发现三（最打脸）：我那个辛辛苦苦写的查询改写，基本没用。**
你看最后两行，"混合检索+重排"(0.729) 和 "查询改写方案"(0.736) 几乎一样。但混合检索**不需要那次又慢又抽风的改写调用**——全文检索天然就把"按字面匹配答案"这活干了。

于是我果断把查询改写砍了，换成混合检索。**速度从 20~30 秒降到 10~11 秒。**

---

## 六、最终效果

现在的检索管线长这样：

```
问题 → 向量检索(擅长定义/关系) + 全文检索(擅长操作)
     → RRF 融合(取长补短)
     → cohere 重排(优中选优，取前8)
     → DeepSeek 生成答案
```

实测三类问题：

- **"what is flagos?"**（纠缠我两天的）：11 秒答出"FlagOS 是全开源、面向异构 AI 芯片的统一软件栈，一次开发到处部署……"✅
- **"how to install FlagCX?"**：直接甩 `git clone` + 安装步骤（全文检索的功劳）✅
- **"FlagOS 有哪些核心库?"**（中文问）：完整列出算子库/FlagTree/FlagScale/FlagCX ✅

那个定义块的排名变迁，堪称一部奋斗史：

> 第 176 名（答不出）→ 切块重组后仍垫底 → 去噪后第 70 名 → 混合检索后**第 1 名**。

---

## 七、如果你也要搭，这几个坑提前帮你踩了

1. **RAG 准不准，先盯检索，别死磕提示词。** 定义块排 176 名的时候，我提示词写出花来也没用——模型压根没拿到正确素材。
2. **纯向量检索有天坑**：遇到满篇都是的高频词（比如你产品的名字），"定义它"的那段会被一堆"顺口提到它"的段落淹没，区分度不够。**上混合检索（向量+关键词+RRF+重排）**，这是正解。
3. **不同问题吃不同的检索**：操作类靠关键词，定义类靠向量。别指望一种打天下。
4. **带 reasoning 的模型，max_tokens 给足**，不然它想太多把答案憋没了。
5. **云端服务会抽风**：我那 OpenRouter 五次断三次。代码里一定要做"快速失败 + 降级"，不然一次超时能把整个网关拖死（我就这么被搞出过 "model not found"）。
6. **边缘板子别硬刚大模型**：NPU 是干视觉的，大 Transformer 量化质量崩给你看。该上云上云。
7. **零碎工程坑**：pip/apt 换国内镜像、pgvector 得源码编译、建表记得设 owner、后台进程用 `setsid nohup ... & disown` 别被 SSH 带走……这些单拎出来都不难，凑一起够喝一壶。

---

## 写在最后

整个过程我最大的感触是：**RAG 这东西，demo 很简单，调准很难，而难点几乎全在"检索"这一环。**

而且调试它最有效的方法不是凭感觉瞎试，是**量化**——盯住"正确答案排第几名"这一个数，你就能精确知道问题卡在切块、召回、融合还是生成哪一层，然后对症下药。我之所以能把"what is flagos"从第 176 名救到第 1 名，靠的全是这个笨办法。

共勉。有问题评论区见。

---

*本文所有数据均来自真实实测。相关代码与完整实验记录已开源，检索实现见 `cloud_rag.py`，对照实验脚本见 `eval/retrieval_experiment.py`。*