# 检索对照实验结果（2026-11-20）

**目的**：用数据判断大语料 RAG 该用哪种检索方式，而非凭感觉。
**方法**：12 题评测集（定义/枚举/操作/综合 四类各 3），每题标注语料里真实存在的"正确块"
（source+title 模式匹配，重复文档两套都算命中）。指标 Recall@10 / MRR@10。
**脚本**：`retrieval_experiment.py`。语料：FlagOS 文档英文部分，13896 块 / 853 文档。

## 五组方案

| 组 | 检索方式 |
|---|---|
| B | 纯 Dense（qwen3-embedding 向量） |
| C | 纯 FTS（PostgreSQL tsvector 全文检索，近似 BM25） |
| D | Dense + FTS + RRF 融合 |
| E | D + cohere rerank 精排 |
| A | 改写(DeepSeek) + Dense + rerank（旧生产方案，对照） |

## 汇总（12 题平均）

| 组 | Recall@10 | MRR@10 |
|---|---|---|
| B 纯 Dense | 83% (10/12) | 0.384 |
| C 纯 FTS | 50% (6/12) | 0.326 |
| D Dense+FTS+RRF | 92% (11/12) | 0.519 |
| **E D+Rerank** | **92%** | **0.729** |
| A 改写+Dense+Rerank | 92% | 0.736 |

## 逐题正确块首次命中排名（节选关键题）

| 题 | 类型 | B Dense | C FTS | D RRF | E +Rerank |
|---|---|---|---|---|---|
| D1 what is FlagOS | 定义 | 3 | 漏(50外) | 3 | **1** |
| D2 what is FlagGems | 定义 | 漏(30) | 漏(14) | 5 | 4 |
| D3 what is FlagCX | 定义 | 9 | 漏(44) | 4 | **2** |
| O1 install FlagCX | 操作 | **1** | 3 | 1 | 1 |
| O3 how to use FlagGems | 操作 | 8 | **1** | 1 | 2 |
| R1 支持异构芯片 | 综合 | 漏 | 漏 | 漏 | **1** |
| R3 FlagTree vs FlagGems | 综合 | 5 | 漏 | 9 | 漏 |

## 结论

1. **不同问题该用不同检索**：操作类 FTS 强（O3：FTS 排 1，Dense 排 8）；
   定义/综合类 FTS 全漏（词太泛），靠 Dense。RRF 让两者互补 → Recall 83%/50% → 92%。
2. **Rerank 提精度**：MRR 0.519 → 0.729，把正确块顶到第 1-2 名。
3. **Query 改写无额外增益**：E(0.729) ≈ A(0.736)，但 E 不需要那次慢改写调用。
4. **唯一失败题 R3**（FlagTree 与 FlagGems 关系）：各法皆弱，属跨块综合难题，检索层解决不了。

## 定稿

生产检索改为 **E 方案：Dense + FTS + RRF + Rerank**（去掉 Query 改写）。
效果持平、快一倍（20-30s → 10-11s）、少依赖一次 OpenRouter。
实现见 `../app/cloud_rag.py` 的 `CloudRAG.search()`。
