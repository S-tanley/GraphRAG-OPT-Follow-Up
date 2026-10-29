# Report：KV prefetch K0 第二轮——更多数据集和 RAG 类型、只看生成侧、question 提前 prefill

日期：2026-10-21
plan：`archive/plan/2026-10-19-graphrag-kv-prefetch-plan.md`（K0；K1 的复用率预览）
上一轮：`report/2026-10-20-h200-quant-anchor-k0.md` §4（RoG KG 路径枚举：4 跳时检索占 78–94%，3 跳以下检索可忽略）

这一轮按讨论的三点调整：
1. 不只看检索占比，**单独看生成侧（检索结束之后）能省多少**；
2. 不局限于 CWQ / WebQSP 和 GraphRAG，看常用的多跳文本 RAG（HotpotQA、MuSiQue、2WikiMultihopQA）和其他检索方式，**找检索和生成接近五五开的场景**；
3. **question 在检索开始前就已知**，可以在检索期间先 prefill；要测换成 question 在前的格式会不会掉准确率。

端侧（Mac）的测量在 B 档采集结束后完成，见 §3.8。

---

## 0. 结论

1. **五五开的场景是存在的，而且是常见配置**（H200 + vLLM，Qwen3-8B，batch 1，短答案）：
   - 稠密检索 + 交叉编码器重排：检索占 54–63%；
   - 维基百科 2100 万段 + CPU 上的 IVF 索引：nprobe 32 时检索占 36–43%，nprobe 128 时占 67–72%。

   和文献一致：TeleRAG 测到 41–60%，Shen et al. 测到 41%。
2. **只看生成侧，上限很大**：假设段落的 KV 都已就绪、question 也已提前 prefill，检索结束后只剩约 8 ms（8B）或 12–15 ms（30B）要算：
   - 生成侧首 token 延迟可省 60–92%；
   - 生成侧总时间（首 token + 短答案 decode）可省 27–71%。

   top-k 越大（上下文越长），省得越多。
3. **question 放在前面基本不掉准确率**：
   - Qwen3-8B：稠密检索的 top-5 / top-10 在三个数据集上都更高（+1.0 到 +3.7 pt）；重排和 wiki 检索时有升有降（−2.6 到 +1.0 pt）；
   - Qwen3-30B-A3B 在 MuSiQue 上 +3.6 pt，在 HotpotQA / 2Wiki 上 −1.6 / −5.0 pt。

   而"前缀 + question"的 prefill 只要约 8 ms（8B）或 14–16 ms（30B），能完全藏在检索时间里。
4. **跨请求复用在开放域文本 RAG 里很低**：维基百科 2100 万段时，top-5 段落被之前的请求检索过的只有 0–7%。小语料上的 11–44% 是基准构造的结果（语料就是由这些题目的段落拼成的）。所以这里 KV prefetch 的价值主要来自"**检索进行期间就开始算**"，而不是跨请求的 KV 池。
5. **一个具体可做的设计**：重排期间预算候选段落的 KV。
   - 稠密检索在约 8 ms 时就给出 top-50 候选，重排还要 46–70 ms 才结束。
   - 这段时间里，GPU 可以先算稠密检索排名靠前的候选段落的 KV。
   - 稠密 top-10 能覆盖最终 top-5 的 55–74%，top-20 能覆盖 70–84%。
   - 文献里最接近的是 RAGCache 的推测流水（基于中间检索结果开始 prefill，结果变了就重算），但它针对的是向量检索本身的分段，不是"检索 + 重排"两阶段。
6. **端侧（Mac，MLX 4bit）是 KG 场景下最合适的落点**：
   - Mac CPU 上 KG 4 跳检索平均 0.4–0.5 s，Mac 上 8B / 30B-A3B 的 prefill 是 0.4–0.6 s，检索占端到端 **39–57%**；
   - 检索期间 GPU 完全空闲。假设路径 KV 全部就绪，首 token 延迟可降 34–52%；
   - 端侧 prefill 比 H200 慢约 10 倍，所以生成侧可省的绝对时间也大得多（文本 RAG top-5：235 → 27 ms）。见 §3.8。
7. **文献空白**：调研没有找到"把 KV 计算和图 / KG 遍历重叠"的工作。KG 路径是逐条产出、前缀共享的。上一轮已经测到，RoG 子图 4 跳时检索远长于生成，GPU 空闲时间足够。

---

## 1. 文献：各类 RAG 的检索耗时占比

（子任务调研，**[D]** = 论文正文直接给出的数字，**[S]** = 来自摘要或二手来源，**[E]** = 推算。完整列表见附录。）

| 来源 | 设置 | 检索占比 |
|---|---|---|
| TeleRAG（arXiv 2502.20969） | H100，Llama-3-8B，wiki_dpr 2100 万段，CPU IVF4096 nprobe 256，top-3 | **[D]** batch 1 时 41.1%，batch 4 时 60.5%；检索放到 GPU 后降到 10.5% / 28.3% |
| Shen et al.（arXiv 2412.11854） | A6000 Ada + 32 核 CPU，Gemma-2-9B，1 亿段，HNSW / IVF | **[D]** 端到端 41%，首 token 延迟里约 35–47% |
| VectorLiteRAG（arXiv 2504.08930） | 8×H100，88M 向量 IVF-PQ | **[D]** CPU 检索可达 LLM prefill 的 2 倍 |
| RAGO（ISCA 2025） | 超大规模检索库（640 亿段） | **[D]** 短序列 86%；2048 前缀 / 512 decode 时 31% |
| SearchAgent-X（arXiv 2505.12065） | Search-R1，2100 万段 | **[D]** 精确最近邻 > 60%；HNSW 时可忽略 |
| Graph RAG 统一对比（arXiv 2503.04338） | Llama-3-8B | **[D]** LLM 主导（ToG、LightRAG、GraphRAG global 都以 LLM 调用为主） |
| HippoRAG2（ISO-RAG，arXiv 2609.00513） | – | **[S]** 纯检索 49–117 ms |

调研结论：
- **五五开的场景**：CPU 上 2000 万到 1 亿段的 IVF 检索，配 3–9B 模型和短答案；或者带外部检索（搜索 API、BM25、精确稠密检索）的多步 RAG。
- **不适合的场景**：LLM 主导的图 RAG（ToG、LightRAG、GraphRAG global）、GPU 上的近似检索（毫秒级）、长上下文 RAG。
- **最接近的已有工作**：
  - RAGCache（推测流水 + KV 树）；
  - PipeRAG（检索和 decode 流水，不预算 KV）；
  - TeleRAG（预取 IVF 的桶，加速的是检索本身）；
  - CacheBlend / TurboRAG / EPIC / Block-attention（段落 KV 复用，都在检索结束后）。

---

## 2. 设置

- **数据**：HotpotQA（distractor）、MuSiQue、2WikiMultihopQA 的 dev 集，各取前 1000 题建语料，前 500 题测检索、前 300 题测生成。脚本 `experiments/kv_prefetch/rag_prep.py`。

| 语料 | 规模 |
|---|---|
| 小语料（HippoRAG 的做法：这些题目附带的全部段落） | 6.4k–12.5k 段 |
| 中等语料（三个数据集 train + dev 的全部段落） | 104 万段 |
| 大语料（facebook/wiki_dpr psgs_w100，DPR multiset 向量） | 2101.5 万段 |

- **检索方式**（每题、batch 1，取中位数；脚本 `rag_bench.py retrieve`、`wiki_index.py`）：

| 检索方式 | 组成 |
|---|---|
| dense-small | bge-large 编码 query（GPU）+ 小语料精确搜索（GPU） |
| dense+rerank | 上面 + bge-reranker-v2-m3 对 top-50 重排（GPU） |
| dense-1M | 104 万段：CPU 精确搜索（faiss flat，16 线程）或 IVF4096（nprobe 64） |
| wiki21M | DPR 编码 query + 2100 万段 IVF4096，CPU 32 线程，nprobe 32 / 128 / 256。和 TeleRAG 的设置相同 |

- **生成**：vLLM 0.11，batch 1，关闭前缀缓存，Qwen3-8B 和 Qwen3-30B-A3B（单卡）。
  - prompt：指令 + 段落（每段一行）+ 问题，chat 模板关闭 thinking，短答案。
  - 每题测四项：
    - 完整 prefill；
    - 完整生成；
    - rest：如果段落 KV 都已就绪，检索结束后还要 prefill 的部分。question 在后时是"问题 + 结尾"，question 在前时只有结尾；
    - early：检索期间就能 prefill 的"前缀（+ 问题）"。
  - 两种格式：qlast（question 在段落之后，原格式）、qfirst（question 在段落之前，结尾只有 "Answer:"）。

---

## 3. 结果

### 3.1 检索耗时（ms，每题中位数）

| 数据集 | dense-small | dense+rerank | dense-1M IVF | dense-1M 精确（CPU） | wiki21M nprobe 32 | wiki21M nprobe 128 | wiki21M nprobe 256 |
|---|---|---|---|---|---|---|---|
| HotpotQA | 7.3 | 55.8 | 16.9 | 270.5 | 33.1 | 109.7 | 207.0 |
| MuSiQue | 7.3 | 53.5 | 14.1 | 268.6 | 35.3 | 113.7 | 217.1 |
| 2Wiki | 7.3 | 76.9 | 14.1 | 269.7 | 27.8 | 100.4 | 196.7 |

- 其中 query 编码约 7 ms（bge-large）或 5 ms（DPR）。
- wiki21M 在 16 线程时更慢：nprobe 256 要 260–313 ms。
- 在 GPU 上对 2100 万段做精确搜索只要 13 ms。

### 3.2 生成耗时和检索占比（Qwen3-8B，top-5，question 在后）

| 数据集 | prompt 长度 | prefill | decode | dense-small | dense+rerank | 1M IVF | 1M 精确 | wiki21M np32 / 128 / 256 |
|---|---|---|---|---|---|---|---|---|
| HotpotQA | 721 | 20.8 | 18.5 | 16% | 59% | 30% | 87% | 43 / 72 / 83% |
| MuSiQue | 693 | 19.7 | 19.5 | 16% | 58% | 26% | 87% | 42 / 70 / 82% |
| 2Wiki | 948 | 22.1 | 24.0 | 14% | 63% | 23% | 85% | 36 / 67 / 80% |

（prefill、decode 单位 ms；后面各列是检索占端到端的比例。wiki21M 的生成时间用的是 wiki 检索结果的 prompt，长度 835–873 token。）

Qwen3-30B-A3B（top-5）：prefill 36–42 ms，decode 17–23 ms，dense+rerank 的检索占比 49–56%。

### 3.3 生成侧能省多少（上限：段落 KV 全部就绪 + question 提前 prefill）

| 模型 | top-k | prompt 长度 | prefill | rest（检索后剩下的） | early（检索期间先算的） | 生成侧首 token 延迟 | 生成侧总时间 |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | 5 | 690–950 | 20–24 ms | 7.4–7.8 ms | 7.6–7.8 ms | −60% 到 −69% | −27% 到 −38% |
| Qwen3-8B | 10 | 1340–1910 | 35–46 ms | 7.6–7.9 ms | 7.7 ms | −78% 到 −83% | −45% 到 −53% |
| Qwen3-8B | 20 | 2620–3900 | 68–101 ms | 7.9–8.0 ms | 7.7 ms | −88% 到 −92% | −62% 到 −71% |
| Qwen3-30B-A3B | 5 | 690–950 | 36–42 ms | 12.3–14.8 ms | 14–16 ms | −60% 到 −66% | −38% 到 −46% |
| Qwen3-30B-A3B | 20 | 2620–3900 | 75–104 ms | 14.7–15.2 ms | 14.2 ms | −80% 到 −86% | −61% 到 −70% |

- rest 基本就是 vLLM 一次调用的固定开销（8B 约 5.6 ms + 少量 token），所以 prompt 越长，省的比例越大。
- early 只有 8–16 ms，比所有"五五开"场景的检索时间都短，可以完全藏住。
- 这是上限，**没有计入**：
  - 段落 KV 的拼接和位置修正（RoPE 重新旋转）；
  - 为补全段落之间 attention 的部分重算（CacheBlend 式，约 15% 的 token）；
  - 段落没有提前算好的情况。

  这些在 K2 测。

### 3.4 question 放前面（qfirst）的准确率

EM，300 题：

| 模型 / 检索 | HotpotQA qlast → qfirst | MuSiQue | 2Wiki |
|---|---|---|---|
| Qwen3-8B / dense top-5 | .500 → .527 | .223 → .247 | .387 → .407 |
| Qwen3-8B / dense top-10 | .513 → .550 | .240 → .250 | .373 → .403 |
| Qwen3-8B / rerank top-5 | .543 → .543 | .233 → .207 | .407 → .417 |
| Qwen3-8B / wiki21M top-5 | .224 → .234 | .100 → .090 | .248 → .245 |
| Qwen3-30B-A3B / dense top-5 | .503 → .487 | .197 → .233 | .367 → .317 |

- 8B 上大多持平或更好；30B 在 2Wiki 上低 5 pt。差异有正有负，300 题的误差约 ±2.9 pt。
- 结论：question 在前的格式**总体可用**，但要按模型和数据检查。
- 另一种做法是 question 在前、末尾再重复一次 question。这样末尾的几 token 要在检索后 prefill，代价很小，准确率可能更稳，待测。

### 3.5 重排期间可以先算哪些段落

稠密检索在约 8 ms 时给出排名；重排结束（约 55–77 ms）后才知道最终的 top-k。最终 rerank top-5 中，有多少出现在稠密检索的前 m 名里：

| 数据集 | m = 5 | m = 10 | m = 20 |
|---|---|---|---|
| HotpotQA | 59% | 74% | 84% |
| MuSiQue | 43% | 55% | 70% |
| 2Wiki | 48% | 59% | 73% |

重排的 40–70 ms 里，8B 可以 prefill 约 1500–3000 个 token，也就是 10–20 段左右。所以重排结束时，最终 top-5 中 70–84% 的段落 KV 可以已经算好。

### 3.6 跨请求复用（K1 预览）

top-5 段落已被之前的请求检索过的比例（按题目顺序到达，池容量不限）：

| 数据集 | 小语料 dense | 小语料 rerank | wiki21M |
|---|---|---|---|
| HotpotQA | 11% | 16% | 0% |
| MuSiQue | 32% | 44% | 7% |
| 2Wiki | 25% | 39% | 2% |

开放域时几乎没有复用。真实流量里热门文档的复用（Zipf 分布）需要用真实查询日志才能评估。

### 3.7 IRCoT 式多步检索

Qwen3-8B 在第一步就直接给出"So the answer is ..."，平均 1.0 步，没有进行第二次检索，所以这个设置没有形成真正的多步检索。要测多步检索，需要用专门训练过的模型（如 Search-R1），或者强制分解问题。暂不展开。

### 3.8 端侧：Mac Studio（M5 Max，MLX 4bit）

prompt 和检索结果与 H200 完全一样（语料在本地按同样方式重建，段落数和题目逐条一致）。脚本 `experiments/kv_prefetch/rag_mac.py`。

**文本 RAG**（每题中位数，单位 ms；EM 为 100 题）：

| 模型 | 数据集 | top-k | prefill | decode | rest（qlast / qfirst） | early（qlast / qfirst） | EM qlast → qfirst |
|---|---|---|---|---|---|---|---|
| Qwen3-8B-4bit | HotpotQA | 5 | 235 | 36 | 50 / 27 | 33 / 53 | .51 → .51 |
| Qwen3-8B-4bit | HotpotQA | 10 / 20 | 489 / 1001 | 41 / 44 | 52 / 52 | 33 | .52 / .53 |
| Qwen3-8B-4bit | MuSiQue | 5 | 222 | 40 | 33 / 28 | 33 / 52 | .16 → .21 |
| Qwen3-8B-4bit | 2Wiki | 5 | 301 | 40 | 39 / 28 | 33 / 52 | .37 → .40 |
| Qwen3-30B-A3B-4bit | HotpotQA | 5 | 155 | 32 | 56 / 24 | 42 / 72 | .50 → .47 |
| Qwen3-30B-A3B-4bit | 2Wiki | 20 | 806 | 44 | 47 | 43 | .38 |

- 端侧 prefill 比 H200 慢 10 倍左右（8B：235 ms 对 21 ms），decode 只慢约 2 倍。所以端侧生成时间以 prefill 为主。
- 生成侧可省的**绝对时间**大得多。8B 在 top-5 时首 token 延迟从 235 ms 降到 27 ms（−88%），top-20 时从 1001 ms 降到 52 ms。
- 端侧的检索：
  - query 编码 12 ms，小语料搜索 < 1 ms；
  - **重排 50 段要 470 ms**（H200 上 48 ms）。"稠密 + 重排"的检索约占 60%，但重排和 LLM 共用同一块 GPU，这段时间没法同时做 KV 预算。
  - 只跑完了 HotpotQA，另外两个数据集遇到 MPS 内存不足，没有重跑。

**KG 路径检索（RoG 子图，cap 50）**：Mac CPU 比集群的 Xeon 快约 2.5 倍。4 跳平均 CWQ 490 ms、WebQSP 394 ms；3 跳 29–38 ms。

| 模型 | 数据集 | 跳数 | prompt 长度 | 检索（CPU，平均 / 中位数） | prefill | decode | 检索占比 | 首 token 延迟：基线 → 路径 KV 全部就绪 | 检索时间够藏住全部路径 prefill 的请求 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B-4bit | CWQ | 4 | 1683 | 514 / 156 | 610 | 32 | **46%** | 1066 → 569（−47%） | 30% |
| Qwen3-8B-4bit | WebQSP | 4 | 1835 | 437 / 156 | 592 | 43 | **39%** | 1039 → 496（−52%） | 25% |
| Qwen3-30B-A3B-4bit | CWQ | 4 | 1683 | 514 / 156 | 383 | 27 | **57%** | 868 → 574（−34%） | 40% |
| Qwen3-30B-A3B-4bit | WebQSP | 4 | 1835 | 437 / 156 | 400 | 35 | **48%** | 864 → 504（−42%） | 32% |
| Qwen3-8B-4bit | CWQ | 3 | 1106 | 33 / 5 | 419 | 31 | 8% | 390 → 88（−77%） | 15% |
| Qwen3-8B-4bit | WebQSP | 3 | 1226 | 43 / 30 | 446 | 42 | 8% | 444 → 99（−78%） | 11% |

（时间单位均为 ms。）

- **端侧 + KG 4 跳就是五五开的场景**：检索占 39–57%，而且检索全程在 CPU 上，GPU 空闲。这正是"边检索边预算路径 KV"的理想条件。
- 3 跳时检索太短，来不及边检索边算。但端侧 prefill 很重，只要路径 KV 能从池子里复用（上一轮测到 CWQ 的三元组复用率约 60%），首 token 延迟就能省 70% 以上。这时价值来自跨请求复用，而不是和检索重叠。
- 在 H200 上，4 跳时检索占 78–94%，prefill 只有 40–60 ms，生成侧能省的绝对时间很少。所以 KG 场景下，**KV 预算 / 预取在端侧比在服务器上更有意义**。

---

## 4. 对 KV prefetch 方向的建议

1. **场景定位**：
   - 主场景：带 CPU 大规模向量检索或重排的文本 RAG，以及 KG 多跳路径检索（上一轮：4 跳时 GPU 空闲远大于 prefill）；
   - 不适合：小语料 + GPU 检索。
2. **方法定位**：从"跨请求的 KV 池"转为"**检索期间的推测式预算**"：
   - 文本 RAG：重排期间预算稠密检索的前 10–20 个候选，question 提前 prefill；
   - KG：路径逐条产出时立即预算；路径之间前缀共享，可以和 GraphRAG-Opt 的 trie 结构结合。
3. **下一步实验**：
   - **K2 准确率**：段落 KV 分别预算后拼接（加 RoPE 修正、部分重算），在 qfirst 格式下比较 EM，和完整 prefill 对比。这是方法能否成立的关键；
   - ~~端侧 K0~~：已完成（§3.8）。端侧 + KG 4 跳是五五开的场景，可以作为原型的首选平台：路径逐条产出时，在 MLX 上逐段 prefill；
   - **批量服务**：多请求时 prefill 会相互排队，预算的价值可能更大（文献中 TeleRAG 的 batch 4 检索占比更高）；
   - **Llama-70B**：本轮没跑。大模型 prefill 更重，生成侧可省的绝对时间更多。

---

## 附录

**代码**：`experiments/kv_prefetch/{rag_prep,rag_bench,wiki_index,rag_summary}.py`

**数据**：`outputs/kv_prefetch/rag/`（集群上另有 104 万段和 2100 万段的向量与索引，约 70 GB）

**过程中的问题**：
- faiss-cpu 建 2100 万段的 IVF 时，训练阶段只用到一个核，47 分钟没有完成。改为在 GPU 上做 k-means 和分桶，faiss 只负责写入倒排表，34 秒完成。
- 在 vLLM 环境里装 `datasets` 时，huggingface-hub 被升到 2.x，和 transformers 4.57 不兼容。已回退，数据改为用主环境预处理成 jsonl。
