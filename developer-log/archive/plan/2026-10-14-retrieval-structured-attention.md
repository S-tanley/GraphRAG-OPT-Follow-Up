# Plan：用检索结构决定 attention 连接（GraphRAG-Opt follow-up）

*Bowen · 2026-10-14 创建 · 2026-10-15 修订 v2（读完 GraphRAG-Opt 代码后）、v3（实验 0–4 完成后）· 状态：实验 0–4 已完成；实验 5（系统端到端）等待确定方向*

依据：

- NSF proposal（`developer-log/context/NSF_confidential.pdf`）
- GraphRAG-Opt 代码（`graphrag-main/`，MICRO 2026 投稿版本）及其 `results/`
- GraphRAG-Opt 阅读笔记（Paper library / Hao Zheng）
- Paper library 中的相关论文

> **查新状态**：网络搜索仍被组织设置禁用。§11 中标注"未核实"的相关工作只凭记忆。

## 变更记录

**v3（2026-10-15，实验 0–4 完成后）**：各实验报告在 `developer-log/report/2026-10-15-*.md`。

1. **T（把 trie 写成文本，配普通 causal）成为第一个贡献。** 在 4 个模型、2 个数据集上都不掉点，路径 token 少 50–58%（实验 2）。cap 提到 30 后，token 只是 cap=8 基线的 1.13 倍，Hit 却高 2–6 个点，与 M0 cap=30 持平；后者要用 2.8–3.2 倍的 token（实验 4）。
2. **M2（trie mask）的免训练版本不够用。** 它相对 GraphRAG-Opt 的 reuse-all 能补回一半损失，但仍掉 2–9 个点（实验 2、4）。按层开放跨路径 attention 时，必须从第 8 层左右就放开，可复用的层只剩 12–25%；sibling 组有害（实验 3）。
3. **在 mask 下做 LoRA 后，M2 能和最佳全 attention 方案打平**：LoRA-M2 + M2-hyb 对 LoRA-M0 + M0，差距 −0.6 到 −1.0，在四组上都不显著，路径 token 只有对方的 31–44%。对照组证明这不是"多训练一轮"的效果（实验 3）。残余差距：与"同一个 LoRA-M2 模型用全 attention 推理"相比，cap30 CWQ 仍差 3.4 个点。
4. **系统层面要算清楚**："T + 现成前缀缓存"已经能省掉 50–74% 的路径 prefill；M2 节点级复用只再多省 5–8 个百分点（实验 4 结果三）。M2 相对 T 的独有价值在于路径间 prefill 并行、检索与 prefill 的流水线，以及在长上下文下 attention 平方项的节省，这些都需要实验 5 用端到端测量来证明。
5. **"graph"贡献**：在 CWQ/WEB 上，检索结构几乎就是一棵树，按实体分组没有作用对象（§1.3）；sibling 组还会伤害准确率。所以论文应该讲"检索 trie"，不讲"图拓扑"。
6. 实验 0 还发现 GraphRAG-Opt 的 R 和 Opt 之间存在 tokenization 不一致，Table 2 的复用代价可能被低估约 1 个点；缺换行的问题经验证无影响，已撤回。

**v2（2026-10-15）**：读代码和已有结果后，有五处判断要改。

1. **M1 的掉点已经测过，而且不小。** GraphRAG-Opt 的 reuse-all 本质上就是 block-diagonal（M1）。RoG/CWQ 掉 6.8 个点，Llama-2-7B/CWQ 掉 12.4 个点，CacheBlend 30% 只能补回 ≤2.6 个点（§1.2）。所以主线要回答的核心问题，从"mask 会不会伤精度"变成了"掉的点从哪里来、怎么补回来"。
2. **"精确复用"作为卖点不够。** GraphRAG-Opt 的 reuse-all 本身已经是确定性的、与 query 无关的。真正的区别在别处：去重、batch 无关的输出、以及能把差距补回来的 mask 设计（§3.5）。
3. **检索结构基本是一棵树，不是图。** 每个 query 平均只有 1.0–1.15 个 topic entity；在中间实体处交汇的路径每个 query 只有 ≤1.4 处；来自不同起点的路径在终点汇合，只出现在 CWQ 4–9% 的 query 里。原来设想的"按共享实体分组"（M3）在这些 benchmark 上几乎没有作用对象，改为按 trie 结构设计跨路径连接（§3.6）。
4. **query 内的 trie 去重是最主要的冗余来源。** 字符级估计能省 58–66%，而且不需要 batching。GraphRAG-Opt 的跨 query 路径级复用只有 3–39%（§1.3）。
5. **必须加一个"trie 写成文本 + 普通 causal mask"的 baseline。** 它不改 mask 也能去重，审稿人一定会问这一点（§3.3 T）。

另外修正了 v1 中的几处事实：reader 实际只吃 ≤8 条路径（final table 是 30 条），大多是 1–2 hop，"37 条、4 hop"出自另一个设置；评测口径沿用 RoG Hit；trace 直接使用 `precomputed_paths_v2/`，不再另定格式。

---

## 0. 一句话

GraphRAG 的检索结果是一棵以 topic entity 为根的前缀树。把这棵树直接变成 attention 的结构，就能在一个 query 内去掉约六成重复的路径 token，并让路径的 KV 与 query、与 batch 组成都无关，可以跨 query 复用。核心研究问题是：独立编码的路径相对完整 prefill 会掉 5–13 个点，这些点是从哪里掉的，又能不能用结构化的连接补回来。

---

## 1. 读代码后确认的事实

### 1.1 GraphRAG-Opt 的实际设置

| 项目 | 实际情况 | 出处 |
|---|---|---|
| Reader prompt | RoG 的 Llama-2-chat 模板：`[INST] <<SYS>>…Reasoning Paths:\n{paths}\nQuestion:\n{q} [/INST]`。**问题放在路径之后**，路径本来就看不到问题 | `faithful_pfkv.build_prompt` |
| 路径序列化 | `e0 -> r1 -> e1 -> …`，一行一条；每行单独 tokenize（`s + "\n"`） | 同上 |
| 每个 query 的路径数 | ablation 和 `table2_worker` 默认 `max_paths=8`；final table（`orchestrate.sh`）用 v2 路径、`max_paths=30` | 代码默认值 |
| 路径 hop 数 | CWQ v2 中 1 hop 占 40%，2 hop 占 57%，3 hop 占 3%；WEB 以 1 hop 为主 | 本地统计 |
| 检索 | RoG planner 规划关系路径（beam=4），再用 `bfs_with_rule` 在**无向图**上落地；不使用答案 | `METHODOLOGY.md` |
| 评测 | "Hits@1" 实际是 RoG 的 substring Hit（归一化后 gold ⊂ pred），另有 list-F1 | `metrics.py` |
| 500 题子集 | `load_dataset(…, split="test").shuffle(seed=42).select(range(500))`，可以直接复现 | `table2_worker.py` |
| 路径 KV 的编码 | `model(P ⊕ π)` 后只保留 π 的切片。P 是固定的 RoG 前导（到 `Reasoning Paths:\n` 为止）；复用时按 delta 重新旋转 key | `hf_reuse.HFPathCache` |
| Selective reuse | 全数据集出现 ≥2 次的路径算作 shared；每个 query 最多复用 `reuse_budget=4` 条，按 (hop, 字典序) 排在最前面，其余路径作为文本在上下文里现算；final 配置用 `recompute_k=2` | `table2_worker.py`、`orchestrate.sh` |
| 两套引擎 | `hf_reuse`（HF 通用，Table 2 用）和 `selective_recompute.LlamaManualPrefill`（从权重手写的 Llama 前向，position 显式给出，ablation 用） | — |

### 1.2 M1 的掉点（来自 `results/ablation_a1_template.txt`，500 题，encode_prefix=template）

| 模型 / 数据集 | 完整 prefill | Reuse-all（≈M1） | + CacheBlend 15% | + CacheBlend 30% |
|---|---|---|---|---|
| RoG / CWQ | 0.510 | 0.442（−6.8） | 0.452 | 0.468 |
| RoG / WEB | 0.760 | 0.716（−4.4） | 0.708 | 0.712 |
| Llama-2-7B / CWQ | 0.408 | 0.284（−12.4） | 0.316 | 0.328 |
| Llama-2-7B / WEB | 0.678 | 0.582（−9.6） | 0.604 | 0.618 |

不加前导 P 时更差（`ablation_a1_sweep.txt`：RoG/CWQ 0.356，Llama-2-7B/CWQ 0.230）。

从这些数字能得出三点：

- 按 token 重算（EPIC、CacheBlend）在短 KG 路径上基本无效，README 自己也写了"recovers ≤1 pt"。损失是结构性的，不在少数几个偏差大的 token 上。
- GraphRAG-Opt Table V 里的 28.4 就是这里的 Llama-2-7B reuse-all。
- 前导 P 的作用很大，因为它决定了 attention sink 和路径能看到的上下文。这说明"路径能看到什么"是关键变量，正是 mask 设计要控制的。

### 1.3 路径结构（本地统计，`precomputed_paths*/`，字符级估计）

| 指标 | CWQ cap=8 | CWQ v2 cap=30 | WEB v2 cap=30 |
|---|---|---|---|
| 每个 query 的路径数（均值） | 7.1 | 21.6 | 17.4 |
| topic entity 数 / q | 1.09 | 1.15 | 1.01 |
| 与其他路径共用 (head, rel1) 的路径比例 | 0.85 | 0.91 | 0.81 |
| query 内 trie 节省（字符） | 0.52–0.58 | 0.59–0.66 | 0.50 |
| 在中间实体处交汇的次数 / q | 0.62 | 1.40 | 0.91 |
| 来自 ≥2 个起点、在终点汇合的 query 比例 | 3.9% | 9.5% | 0.5% |

冗余对比（CWQ v2 cap=30，字符级）：

| batch 方式 | GraphRAG-Opt 跨 query 路径级 | 只做 query 内 trie | trie + 跨 query 节点级 |
|---|---|---|---|
| random, B=64 | 0.049 | 0.664 | 0.683 |
| topical, B=64 | 0.340 | 0.663 | 0.783 |
| topical, B=512 | 0.386 | 0.664 | 0.800 |

结论：最大的冗余来自**单个 query 内部**的前缀共享，不需要 batch，也不需要 topical 排序，这正是 GraphRAG-Opt 没有覆盖的部分。字符级数字需要用真实 tokenizer 复核（实验 1）。Freebase 的 relation 名很长（如 `location.location.containedby`），token 级的节省可能更高。

### 1.4 代码里需要注意的几个问题（尚未在 GPU 上验证）

- ~~Reuse 路径缺少换行分隔~~：token 层面属实，但实验 0 表明它对准确率没有可测影响（见 report 2026-10-15），撤回。
- **GraphRAG-R 与 Opt 的 tokenization 不一致**（实验 0 发现）：`build_prompt` 逐行 tokenize（`▁New`），`partial_reuse_generate` 把剩余路径整段 tokenize（`New`），500 题中有 483 题的 token 序列不同；budget=0 时 Opt 比 R 高 1.2 个点（p=0.07）。Table 2 的复用代价可能被低估约 1 个点。后续一律使用逐行 tokenization。
- **噪声底**：只改一个分隔符就有 3–5% 的题 Hit 翻转；n=500 时小于约 2 个点的差异不可信，后续报告配对 McNemar 检验。
- **Selective reuse 的输出依赖数据集组成。** 一条路径算不算 shared，取决于全数据集的频次，所以同一个 query 的答案会随"还有哪些其他 query"而变化。新方案没有这个问题，可以作为一个论点。
- **ablation 和 final table 的设置不同**：ablation 是 `max_new_tokens=16`、cap=8；final 是 64、v2、cap=30。对比时要用同一套设置。
- **环境**：代码要求 `transformers==5.2.0`，因为用到了 `DynamicCache.layers[i].keys`；RoG 模型的 tokenizer 要换成 Llama-2-7b-hf 的。

---

## 2. Prefill 之前已知的信息，以及打算怎么用（更新）

| 信息 | GraphRAG-Opt | 本计划 | 在 CWQ/WEB 上的实际分量 |
|---|---|---|---|
| prompt 组成和精确长度 | 只用来分配共享 KV | D：调度、预留 KV | — |
| 路径重复次数（跨 query） | r−1 优先级、query 重排 | 细化到 trie 节点 | 3–39% |
| **前缀共享（query 内）** | 未使用 | **A：trie mask 去重** | **≈60%（主要来源）** |
| 中间实体交汇、终点汇合 | 未使用 | A 中的跨路径连接 | 很少（≤1.4 处 / q；CWQ 中 4–9% 的 query） |
| sibling 关系：同一 (head, rel) 下的多个尾实体 | 未使用 | **A：sibling 组内互相可见** | 81–91% 的路径属于某个 sibling 组 |
| relation schema | 只用来预测 expert | A 中作为分组依据之一 | — |
| relevance / rank | 未使用 | C | precomputed 路径里**没有分数**，C 降级 |
| 候选答案（路径上的实体） | 未使用 | B | 召回上限 0.644 而 Hit 只有 0.51，见 §4 |

---

## 3. 方向 A：用检索 trie 构造 attention（主线）

### 3.1 Prompt 布局

沿用 RoG 模板，只改中间部分：

```
[INST] <<SYS>>…Reasoning Paths:\n  [路径 trie]  \nQuestion:\n{q} [/INST]  [answer …]
```

路径原本就在问题之前，所以不需要改动问题的位置。

### 3.2 Trie 的构建

- 按 hop 单元（实体、` -> `、relation）**分段 tokenize，再拼接 token ID**，保证同一前缀在不同路径里的 token 完全一致。叶子末尾保留 `"\n"`。
- 用分段拼接得到的 token 序列，和整行一次性 tokenize 得到的序列可能不一样。实验 1 要统计两者不一致的比例；完整 prefill 的 baseline 统一改用分段 tokenize，保证对照公平。
- 节点 key 为 `hash(model, P, 前缀 token IDs)`。

### 3.3 条件族

| 记号 | 规则 | 去重 | 可跨 query 复用 | 用途 |
|---|---|---|---|---|
| M0 | 平铺 + causal | 否 | 否 | 上界 |
| M0-shuf | M0，路径顺序随机打乱（3 个 seed） | 否 | 否 | **噪声底**：只是改变顺序会让 Hit 波动多少 |
| **T** | **trie 写成文本**（例如 `NY -> containedby -> {A; B; C}`）+ causal | 是（文本层面） | 否 | **最强的简单 baseline** |
| M1-seq | block-diagonal，路径按真实顺序编号位置 | 否 | 否（位置依赖偏移） | 诊断：只切断跨路径 attention 的代价 |
| M1-par | block-diagonal，所有路径都从 \|P\| 开始编号（PCW 式） | 否 | 是 | 诊断：并行位置的代价 |
| M1-hyb | GraphRAG-Opt reuse-all：按 M1-par 计算，读出时把 key 旋转到顺序位置 | 否 | 是 | 已有结果，复现用 |
| M2-par / M2-hyb | trie mask，位置方案同上 | 是 | 是 | 主方法的基础形态 |
| M2+S | M2 + **sibling 组**：同一 trie 父节点下的叶子互相可见 | 是 | 部分（组构成相同时可以） | 结构化连接 ① |
| M2+J | M2 + 每个内部节点一个**汇总 token**，它看自己的整棵子树；问题经过它读证据 | 是 | 路径部分可以；汇总 token 现算 | 结构化连接 ② |
| M2+top-ℓ | 下面 L−ℓ 层用 M2，最上面 ℓ 层路径 token 用 M0 | 是 | 下面 L−ℓ 层可以 | 结构化连接 ③ |

要回答的分解：

- M0 → M1-seq：切断跨路径 attention 的代价
- M1-seq → M1-par / M1-hyb：位置方案的代价
- M1 → M2：去重本身的代价（重复带来的 attention 权重被去掉）
- M2 → M2+S / J / top-ℓ：结构化连接能补回多少
- 与 T 对比：用 mask 去重相比在文本上去重，是否还有额外价值

### 3.4 位置编号

- 默认左对齐：节点位置 = |P| + 前缀 token 数。位置只由前缀决定，这是跨 query 复用的前提。
- 问题的位置 = |P| + 最长路径长度。也可以采用 hyb 方案，在读出时把 key 旋转到紧凑的顺序位置。旋转是确定性的，所以不影响复用。
- 路径数多时，并行位置可能出问题。按路径数分桶报告结果（cap=8 和 cap=30 两档）。

### 3.5 和 GraphRAG-Opt 相比，区别到底在哪（v2 重新表述）

GraphRAG-Opt 的 reuse-all 已经是确定性的、与 query 无关的，所以单说"精确"不足以区分。本方案的区别是：

1. **query 内去重**：节省约 60% 的路径 token，而且不需要 batching。GraphRAG-Opt 完全不做这件事。
2. **输出与 batch 组成无关**：缓存命中与否不改变结果。GraphRAG-Opt 的 selective reuse 中，一条路径是否被复用取决于数据集里的其他 query。
3. **用结构化连接补回掉点**：按 token 重算在 KG 路径上已经被证明无效（§1.2）。如果 M2+S/J/top-ℓ 能补回，就是一个新的、与 CacheBlend 和 EPIC 正交的结论。
4. **复用粒度更细**：按 trie 节点复用，前缀不同的路径之间也能共享。跨 query 冗余从 39% 提高到约 80%（§1.3）。

### 3.6 关于"graph"贡献的重新定位

在 CWQ 和 WEB 上，检索结果基本是一棵前缀树：在中间实体处的交汇和多起点的终点汇合都很少。因此：

- 主线讲 **trie 结构**：前缀共享 + sibling 组 + 内部节点汇总。这仍然是只有知道检索结构才能做出来的设计，普通 RAG 的平铺 chunk 没有这种结构。
- 多起点汇合（CWQ 4–9%）作为子集单独分析：在这些 query 上，让汇合的路径互相可见是否特别有用。
- 如果要讲更强的"graph"故事，需要多实体、汇合更多的工作负载，例如 v2 的 cap=30、更深的 hop，或者 GrailQA、proposal 中的生物医学 KG。这一部分放到后期，不作为第一篇的前提。

### 3.7 补救手段（按成本从低到高）

1. 位置方案（par / hyb / 顺序）
2. M2+S、M2+J
3. M2+top-ℓ，ℓ ∈ {1, 2, 4, 8}。代价约为 ℓ/L 的路径计算，可以看作"按层重算"，和 CacheBlend 的"按 token 重算"形成对比
4. LoRA：RoG 本身就是在这个模板上微调过的 Llama-2-7B，可以用 RoG 的训练集（CWQ 和 WebQSP 的 train split）在目标 mask 下继续做 LoRA，参照 Block-Attention 的做法（**未核实**）

### 3.8 收益模型

同 v1：线性部分 ∝ token 数，prefill attention 计算比为 Σ(S + depth) / Σ(S + i)，decode 读取量 ∝ N_trie。新增一点：**在相同 token 预算下可以放进更多路径**。cap 从 8 提高到 30，路径 token 约变为 3 倍，去重后约 1.2 倍。按 `METHODOLOGY.md`，CWQ 的召回上限是 0.644，而 Hit 只有 0.51，更多证据可能带来准确率的提升（实验 4）。

### 3.9 实现路线

- **诊断原型**：扩展 `selective_recompute.LlamaManualPrefill`。它从权重手写了 Llama 前向，position 显式给出，mask 写在 `_attention` 里，只要把 causal 条件换成任意布尔矩阵就能支持所有条件。适用于 RoG 和 Llama-2，`tests/test_prefill.py` 可以作为正确性测试。
- **通用模型**（Qwen3、Llama-3）：HF 自定义 4D mask + `position_ids`，或者 FlexAttention。Qwen3 有 q/k norm，不能套用手写的 Llama 前向。
- **系统**：trie 节点映射到 paged KV block。kernel 候选是 FlashInfer 的 custom mask / cascade（**未核实**）。

---

## 4. 方向 B：用候选答案约束 decode（配套，价值上调）

- 新信息：CWQ 的召回上限是 0.644，而 RoG 的 Hit 只有 0.51，说明 reader 在"把答案抽出来"这一步就丢了约 13 个点。把路径上的实体当作候选做 constrained decoding 或 reranking，**可能提高准确率**，而不只是加速。
- 第一步：统计 gold 落在末端实体中的比例、落在任意路径实体中的比例，以及 Hit 失败的题里有多少 gold 其实在候选里。用 500 题子集和现有的 `results/preds_full/` 就能离线算出来，不需要 GPU。
- speculative draft 的加速空间仍然有限（输出很短），作为次要部分。

## 5. 方向 C：按 relevance 分配精度（降级）

precomputed 路径里只有 triple，没有分数。RoG planner 的 beam 排名需要在 `precompute_retrieval.py` 中额外保存下来。C 暂缓，等 A 有结果后再决定；如果要做，先改用"按 hop 数、按 trie 深度"分级。

## 6. 方向 D：调度与内存（配套）

同 v1。新增一点：在 M2 下，路径的 KV 只依赖自己的前缀，所以**检索每落地一条路径就能开始它的 prefill**。这和 trie 的增量构建天然契合。

---

## 7. 实验计划

### 环境（mbzuai-masdar，可用 GPU0 和 GPU1）

- 两张卡上已经有别的进程：GPU0 上是你自己的 vLLM（约 50GB），GPU1 上是 Andrew.Yang 的 vLLM 和一个服务（约 61GB）。每张卡剩约 80–90GB，7–8B 的 fp16 模型各占一张卡没有问题；不要在 GPU1 上跑 32B。
- 新建 uv 环境，固定 `torch 2.10`、`transformers==5.2.0`。base-env 是 4.57.6，不要动它。
- 已有缓存：Llama-2-7b-hf、Meta-Llama-3-8B（base）、Qwen3-8B、Qwen2.5-7B-Instruct。需要下载：`rmanluo/RoG`、数据集 `rmanluo/RoG-cwq` 和 `rmanluo/RoG-webqsp`（HF 可访问）。
- 代码同步到 `~/mbzuai-projects/graphrag-followup/`，任务放在 tmux 里跑，输出写到 `developer-log/report/`。

### 实验 0：复现 gate（第 1 周前半）

在 RoG/CWQ 500 题上复现 ablation 的两个数：完整 prefill 0.510，reuse-all（template）0.442。复现出来才说明 harness 是一致的，后面的数字才能和 GraphRAG-Opt 对比。顺便验证 §1.4 的换行问题：用 `table2_worker` 跑 `reuse_budget=0`，结果应该和 R 完全一致；再看 budget>0 时，加上换行前后的差别。

### 实验 1：token 级结构统计（第 1 周，CPU）

- 用 Llama-2、Llama-3、Qwen3 三种 tokenizer，在 cap=8 和 cap=30（v2）下计算 N_flat、N_trie、节省比例、attention 计算比，以及跨 query 节点级命中率（random / topical × B）。
- 统计分段 tokenize 与整行 tokenize 不一致的比例、sibling 组大小的分布、最长路径长度。
- 统计方向 B 需要的答案覆盖率。
- **Go/no-go**：token 级 query 内节省 ≥40%（字符级估计约为 52–66%）。

### 实验 2：诊断掉的点从哪里来（第 1–2 周，核心）

- 条件：M0、M0-shuf、T、M1-seq、M1-par、M1-hyb、M2-par、M2-hyb。
- 模型：RoG 和 Llama-2-7B（用手写 Llama 引擎，可以和 §1.2 直接对比）；Qwen3-8B 和 Llama-3-8B 用 HF 4D mask 作为泛化。
- 数据：CWQ 和 WEB 各 500 题（seed 42 子集），cap=8；cap=30 作为第二档。
- 指标：RoG Hit、F1，并按路径数和是否多起点汇合分桶。

### 实验 3：结构化连接能补回多少（第 3–4 周）

M2+S、M2+J、M2+top-ℓ（ℓ 扫 1/2/4/8），再加 LoRA。对照组用 CacheBlend 15%/30%（已有数字）。产出两条 Pareto 曲线：Hit vs 可复用比例，Hit vs prefill FLOPs。

### 实验 4：同样代价下放进更多证据（第 4 周）

M0 cap=8 对比 M2(+最佳连接) cap=30：路径 token 预算大致相当，比较 Hit。如果去重后多放的证据能提高 Hit，就有一个"又快又准"的结果。

### 实验 5：系统端到端（v3：方向待定，见下）

v3 的建议设计：在 vLLM 或 SGLang 上，以 cap=30、真实到达顺序（random 和 topical）为负载，比较四种方案：M0、M0+前缀缓存、T+前缀缓存、M2（trie 节点 KV 池 + LoRA-M2，需要自定义 mask kernel，或者用 FlexAttention 做原型）。指标：TTFT、吞吐、KV 显存。重点看 M2 相对 T+前缀缓存能否在 TTFT 上胜出，因为它的路径 prefill 可以并行，也可以和检索流水线。若胜不出，M2 在论文中就作为分析和训练方面的结果，系统部分以 T 为主。

原设计（v2）：

对照 GraphRAG-R、GraphRAG-Opt（修复换行后）、vLLM/SGLang 前缀缓存、CacheBlend。指标：TTFT、吞吐、KV 显存、decode 延迟。在 H200 上测。GraphRAG-Opt 的数字是在 A100 上测的，需要在同一平台上重跑。

---

## 8. 决策树

```
实验 0 复现通过？ ── 否 → 先修 harness，不往下走
实验 2：
├─ M1-seq ≈ M0，但 M1-par / hyb 掉点
│    → 损失来自位置方案。找到不伤精度的、可复用的位置方案即可
│      （这一点本身就解释了 GraphRAG-Opt 为什么掉点）
├─ M1-seq 也掉点 → 需要跨路径连接 → 实验 3
│    ├─ M2+S / J / 小 ℓ 补回      → 主线最强版本：结构化连接 + 去重 + 复用
│    ├─ 只有 LoRA 能补回          → 主线成立，加上"训练用的 mask"作为贡献
│    └─ 都补不回                  → 转向 T（文本层面的 trie）+ B + D
├─ M2 明显低于 M1 → 去重本身有害 → 试着补偿重复的 attention 权重（共享前缀的 logit 加 log(重复次数)）
└─ T ≈ M0 且 M2 系列都明显低于 T
     → 文本去重已经够用；mask 的价值只剩跨 query 复用和流水线，需要用系统指标来证明
```

---

## 9. 风险

| 风险 | 可能性 | 影响 | 应对 |
|---|---|---|---|
| T（文本 trie）已经拿到大部分收益，mask 显得多余 | 中 | 高 | 实验 2 第一时间测 T；mask 的独有价值落在跨 query 复用、输出与 batch 无关、prefill 稀疏和流水线 |
| 结构化连接补不回掉点 | 中 | 高 | LoRA 兜底；再不行就退回到 T + B + D |
| "graph"的故事在 CWQ/WEB 上太弱 | 高 | 中 | 主线改讲 trie 结构；把多起点汇合的 query 单独分析；后期换用汇合更多的工作负载 |
| 查新发现已有工作做过结构化 mask | 中 | 高 | 优先查 Superposition Prompting、RAGCache、Prompt Cache 的 `<union>` |
| GPU1 和别人共用，显存紧张 | 中 | 低 | 7–8B 模型单卡；大模型只放 GPU0，或者另外申请卡 |
| transformers 版本冲突 | 低 | 中 | 独立 uv 环境 |

---

## 10. 与 NSF proposal 的对应关系

| 本计划 | Proposal | 说明 |
|---|---|---|
| trie 节点作为 IR 单位 | T1.1（composition relationships） | 第一个用到 IR 中组合关系的下游 |
| trie mask + 跨 query 节点级复用 | T2.3 Semantic Path KV Reuse | 复用粒度从整条路径细化到节点；去重 |
| M2+top-ℓ | T2.3 "path-based reuse across transformer layers" | 直接兑现这句话 |
| B | T2.3 "prior for speculative decoding" | 加上"提高准确率"的角度 |
| D | T2.3、T3.1 | 边检索边 prefill |
| 稀疏 block mask 的 kernel / 硬件 | T3 | 留作后续 |

注意：proposal §2 里"CWQ 每个 query 37 条路径、4 hop"与 reader 的实际输入（≤8 或 ≤30 条，以 1–2 hop 为主）不一致，写论文时要分清"检索候选"和"reader 实际吃进去的输入"。

---

## 11. 相关工作与查新

| 工作 | 在 library 中 | 与本计划的关系 |
|---|---|---|
| Prompt Cache | 有，已核对 | 按 module 独立编码，固定 position ID，`<union>` 共享位置；指出 module 之间有语义依赖时质量会下降。没有前缀树，也没有结构化连接 |
| SubGCache（AAAI 2026，arXiv 2505.10951） | **无**，`METHODOLOGY.md` 中引用 | 把整个共享子图作为前缀复用。需要精读：它是否按图结构组织 KV |
| CacheBlend / EPIC / KVLink | CacheBlend 有 | 按 token 重算。§1.2 表明在 KG 路径上基本无效，这是本计划的动机之一 |
| SpecInfer | 有 | 在 decode 阶段对候选 token 树用 tree mask |
| SGLang RadixAttention | 有 | 跨请求共享前缀，causal，位置固定 |
| FiD / FiD-Light | 有 | encoder-decoder 上的 M1 |
| SuffixDecoding | 有 | 与方向 B 相关 |
| Block-Attention、TurboRAG、APE、PCW、Superposition Prompting、RAGCache | **无，未核实** | 查新重点：Superposition 的 fork-join DAG，RAGCache 的 knowledge tree |

查新清单：

- [ ] Superposition Prompting
- [ ] RAGCache
- [ ] SubGCache
- [ ] Block-Attention 的微调配方
- [ ] APE 的位置和温度处理
- [ ] KGQA 方向的结构化 attention 工作
- [ ] FlashInfer / FlexAttention 对树形 mask 的支持

---

## 12. 时间线（从 2026-10-15 起）

| 周 | 内容 | 产出（`developer-log/report/`） |
|---|---|---|
| W1（10/15–10/21） | 环境、实验 0、实验 1、换行问题验证、方向 B 的离线覆盖率 | 复现报告、token 级结构统计 |
| W2（10/22–10/28） | 实验 2 | 诊断表，走决策树 |
| W3–W4（10/29–11/11） | 实验 3、实验 4 | Pareto 曲线 |
| W5–W8（11/12–12/09） | 实验 5 系统原型 | 系统性能报告 |

## 13. 待办

- [x] 在集群上建 uv 环境（transformers 5.2.0），同步代码，下载 RoG 模型和数据集
- [x] 实验 0 复现：0.510 / 0.442（逐位复现）
- [x] 验证 §1.4 的换行问题：对准确率无影响；改为告知作者 tokenization 不一致的问题
- [x] 实验 1（真实 tokenizer）：query 内节省 CWQ 0.60–0.69，WEB 0.55–0.65
- [x] 支持任意 mask 和 position 的引擎（改为通用的 `experiments/common/masked_engine.py`，Llama 系和 Qwen3 都可用）；三项 gate 全部通过
- [x] 实验 2、3（含 LoRA）、4，报告见 `developer-log/report/`
- [ ] 实验 5：先确定方向（T 为主，还是 T + M2）
- [ ] 重测 sibling 组（S）：改用按布局顺序的内部位置，排除兄弟分支位置冲突的混杂（见 exp3 报告结论 4）
- [x] 实现 T（trie 文本序列化）：`conditions.trie_text_lines`
- [ ] 方向 B 的离线覆盖率（`results/preds_full/` + 数据集答案）
- [ ] 查新（§11）
- [ ] 与导师确认目标会议
