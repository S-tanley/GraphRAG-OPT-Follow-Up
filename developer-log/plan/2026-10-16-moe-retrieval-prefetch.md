# Plan：用检索信息指导 MoE 的 expert 搬运与调度（显存受限场景）

*Bowen · 2026-10-16 创建 · 2026-10-16 晚修订 v2 · 状态：计划阶段*

依据：

- 2026-10-16 的讨论，包括 Bowen 提出的三个问题：prefill 与 decode 的区别、检索信息到底能早多久、在 FFN 前加 buffer
- NSF proposal T2.1 / T2.2 / T3.1，以及其中的 Fig. 4、Fig. 7、Fig. 8
- trie / KV 复用实验（`developer-log/report/2026-10-15-*`、`2026-10-16-arrow-recompute.md`）

> 模型的参数量、结构和 checkpoint 名称都凭记忆填写，下载后用 `config.json` 核对。标注"未核实"的相关工作，需要等网络搜索可用后查新。

> **2026-10-18**：Phase 2–4 已细化为 `2026-10-18-moe-experiment-plan.md`，依据是 Phase 0.5 的结果和对 20 篇相关工作的调研（`report/2026-10-18-moe-related-work-survey.md`）。以新文件为准。

## 变更记录

**v3（2026-10-17，Phase 0.5 完成后）**：见 `developer-log/report/2026-10-17-moe-phase0.5-timeline.md`。

- RoG 检索约 0.6 秒，几乎全部花在规划（一次 LLM 生成）上；schema 已知后只需约 2ms 就能进入 prefill。所以 **Q0 中"基于检索语义的提前 prefetch"不成立**。
- 规划期间约 0.6 秒的窗口只知道问题文本；而 prefill 反正要用到全部 expert，按层顺序直接搬就行，不需要检索信息。只作为工程优化保留。
- 研究重心转到：**D 问题**（prefill 本身要 2–6 秒，可以在这段时间里按 schema 和候选实体提前搬好 decode 要用的 expert）、**并发场景下的前瞻**（Q4、Q5），以及 **KV 复用**（S3）。
- 带宽：pinned 内存约 54GB/s，pageable 只有 10–18GB/s，所以必须用 pinned。

**v2（2026-10-16 晚）**，相对 v1 的主要变化：

1. **硬件**：home-gpu-node1 重启后连不上，2080 Ti 的部分全部作废。改用 H200，并且选**单张 H200 装不下**的 MoE，把 expert offload 到主机内存。
2. **把问题拆成两个**：prefill 和 decode 在 offload 场景下是两种不同的问题，分别处理（§2）。
3. **重新界定"提前"**：检索结果出来时 prompt 也就确定了，不存在"检索结果比 prefill 早"。提前量只来自检索**进行期间**就能得到的信息。新增 Phase 0.5，先实测这个窗口有多大（§3）。
4. **新增并发场景**：多请求并发时，检索信号的价值在于对队列的"前瞻"：决定哪些 expert 常驻显存，以及按 expert 亲和性组 batch。后者是对"在 FFN 前加 buffer"这个想法的改进（§4）。
5. **吸收对 Fig. 4 / Fig. 7 的分析**：分析时必须加上"跨请求"对照组，按 decode 长度分组，并把 schema 的效果和"token 重叠"的效果分开；同时要考虑 schema 的长尾分布（§5、§7）。

---

## 0. 一句话

在显存装不下整个 MoE、expert 需要从主机内存搬运的场景下，利用 GraphRAG 检索过程中逐步得到的信息（问题文本、relation schema、候选实体），以及检索队列提供的"未来请求"信息，来决定 expert 的常驻、搬运顺序和请求的分组，从而降低 TTFT 和每 token 延迟，或者提高吞吐。

---

## 1. 场景与硬件

| 项目 | 规格 |
|---|---|
| GPU | H200 NVL（mbzuai-masdar 的 GPU0/1），每张 143GB HBM |
| 主机内存 | 1.5TB（约 1.3TB 可用），用来放 offload 的 expert |
| 主机到 GPU | PCIe 5.0 x16，理论约 64GB/s，需要实测 pinned memory 下的实际带宽 |
| 磁盘 | /data 约 11TB 可用 |

**主场景**：单张 H200，非 expert 部分（attention、embedding、共享 expert）常驻显存，routed expert 部分常驻、部分在主机内存。扫描"显存里能放下多少比例的 expert"（例如 30%、50%、70%）。

**模型**（都超过单张 H200 的容量）：

| 模型 | 总参数 / 激活 | expert 结构 | bf16 大小 | 用途 |
|---|---|---|---|---|
| **Mixtral-8x22B-Instruct** | 约 141B / 39B | 56 层 · 8 个选 2 | 约 282GB | 粗粒度代表：expert 少而大，搬一个约 0.6GB |
| **Qwen3-235B-A22B** | 约 235B / 22B | 94 层 · 128 个选 8 | 约 470GB（官方 FP8 版约 240GB） | 细粒度代表：expert 多而小；支持 thinking 模式，可以产生长输出 |
| DeepSeek-V2（备选） | 约 236B / 21B | 160 个路由 + 2 个共享 · 选 6 | 约 470GB | 共享 expert + 细粒度；需要自定义代码 |
| DBRX（备选） | 约 132B / 36B | 16 个选 4 | 约 264GB | 中等粒度 |

**核实结果（2026-10-17，HF API + config.json，脚本 `experiments/moe_survey.py`）**：

| 模型 | 大小 | 层 · expert | 共享 expert | 状态 |
|---|---|---|---|---|
| Mixtral-8x22B-Instruct-v0.1 | 281GB | 56 · 8 选 2 | 0 | 已确认，下载中 |
| Qwen3-235B-A22B / -FP8 | 470GB / 239GB | 94 · 128 选 8 | 0 | 已确认，下载中 |
| DeepSeek-V2.5 | 471GB | 60 · 160 选 6 | 2 | 建议加入，待确认 |
| GLM-4.5-Air | 221GB | 46 · 128 选 8 | 1 | 建议加入，待确认 |
| Hunyuan-A13B-Instruct | 161GB | 32 · 64 选 8 | 待核对 | 可选 |
| Llama-4-Scout-17B-16E | 217GB | 16 选 1 + 共享 | 1 | 需要申请权限（目前 403） |
| DBRX-instruct | – | – | – | 仓库已不存在（404） |

以上模型 transformers 5.2 都原生支持。下载位置是 `/data`（4×14TB NVMe RAID，剩余约 11TB），HF 缓存默认就在 `/data/home/bowenzheng/.cache/huggingface`；根分区只剩 69GB，不要往那里下载。

**工程注意**：

- **采集路由 trace 时**，可以用 FP8 版本放在 GPU0+GPU1 两张卡上跑（Qwen3-235B-FP8 约 240GB，放得下），速度快；但要在一部分题上和 bf16 offload 版对照，确认量化没有明显改变路由。
- **做 offload 实验时**，用 bf16 + 单卡，才是真实场景。
- transformers 5.2 对 FP8 checkpoint 的支持情况需要先确认；不行就用 bf16，靠 `device_map` 拆到两张卡加主机内存上采集 trace，慢一些但可以接受。

---

## 2. 问题拆分：prefill 和 decode

假设每个 token 从 E 个 expert 中选 k 个，且选择均匀独立，那么 N 个 token 之后，某个 expert 一次都没被选中的概率约为 (1 − k/E)^N：

- Mixtral（8 选 2），N = 20：约 0.3%；
- Qwen3（128 选 8），N = 30：约 14%；N = 100：约 0.2%。

KGQA 的 prompt 有 200–700 个 token，**prefill 必然会用到每一层的几乎全部 expert**。所以：

| | prefill（P 问题） | decode（D 问题） |
|---|---|---|
| 每层用到的 expert | 几乎全部 | 每个 token 只用 k 个 |
| 瓶颈 | 把缺的 expert 全部搬一遍，几乎纯粹受带宽限制。例如 Mixtral-8x22B 显存放下 100GB 时，每次 prefill 约要搬 170GB，约 3 秒以上 | 每步每层要等用到的 expert 搬到位 |
| 能优化什么 | ① 常驻哪些 expert（跨请求）；② 先搬哪些（按各 expert 分到的 token 数排序，或按层流水）；③ 和检索时间重叠；④ token 很少的冷门 expert 直接在 CPU 上算（Fiddler 思路）；⑤ KV 复用后需要计算的 token 变少，需要的 expert 可能不再是全部 | 预测用哪几个、提前搬好（经典的 prefetch 问题） |
| "预测具体用哪些 expert"的价值 | 低（反正几乎全都要用） | 高 |
| 主要相关工作 | Fiddler、MoE-Lightning、Klotski、按层流水的 offload | MoE-Infinity、ProMoE、Pre-gated MoE、Mixtral-offloading |

各篇工作对 prefill 和 decode 各自怎么处理，需要逐篇查新确认（**未核实**）。

**KGQA 的 decode 很短**（5–16 个 token），所以 D 问题在 KGQA 上的端到端收益有限，需要加入长输出的负载（W2，见 Phase 1）。

---

## 3. 时间线：检索信息到底能早多久

检索一结束，prompt 就确定了，prefill 也可以开始了，所以"检索结果"本身并不比 prefill 更早。**提前量只来自检索进行期间就能得到的信息：**

```
请求到达 ──→ RoG 规划关系路径 ──→ 在图上落地、枚举路径 ──→ 组装 prompt ──→ prefill ──→ decode
  │           （一次 LLM 生成）           （CPU）
  ├ 问题文本已知        ├ schema 已知              ├ 候选实体逐渐出现     ├ prompt 已知（普通引擎也知道）
```

| 信息 | 何时可得 | 相对 prefill 的提前量 |
|---|---|---|
| 问题文本 | 请求到达 | 整个检索过程 |
| relation schema（RoG 规划的输出） | 规划完成后 | 落地 + 组装的时间 |
| 候选实体 | 落地过程中逐渐出现 | 落地剩余的时间 |
| 完整 prompt | 检索结束 | 0 |

**单请求时的收益上限**：min(重叠窗口, 需要搬运的时间)。目前只有零散数据：proposal 说 CWQ 4 跳的路径枚举约 1.1 秒、1 跳约 0.2 毫秒；RoG 规划是一次 LLM 生成，耗时未知。如果窗口是 0.5 秒、带宽是 50GB/s，大约能提前搬 25GB，约等于 Mixtral-8x22B 的 4–5 层 expert。**这个量值不值得做，必须先测（Phase 0.5）。**

还有一个容易忽略的细节：RoG 规划本身就是一次 LLM 推理。如果规划模型和阅读器共用同一个 MoE，规划这一步自己就会用到 expert，这些 expert 正好可以在后续 prefill 中复用。这一点要在测时间线时一并记录。

---

## 4. 并发场景：对队列的前瞻，以及按 expert 亲和性组 batch

多个请求并发时，GPU 和 PCIe 本来就是忙着的，没有"空闲窗口"可用。这时检索信号的价值变成**前瞻**：每个请求在进入 prefill 之前，已经完成了检索，系统可以提前知道队列里接下来的请求大概会用哪些 expert。

**用法 1：expert 的常驻和换出。** 已知（或预测到）未来请求的 expert 需求，就可以做接近 Belady 的换出决策：把最晚才会再用到的 expert 换出去。proposal Fig. 8 在 KV 复用上已经显示，已知未来时缓存命中率可以从 LRU 的 18.7% 提高到 41%。expert 缓存可能有类似的空间。

**用法 2：按 expert 亲和性组 batch**（由"在 FFN 前加 buffer"的想法改进而来）：

- **原想法**：在 MoE 层前面等一等，攒够 token 再一起算。
- **需要纠正的前提**：vLLM、SGLang 这类引擎里，同一次迭代内所有请求是按层同步推进的，MoE 层本来就会把这次迭代所有请求的 token 按 expert 分组，一起计算。所以"层内攒 token"已经自动发生了。
- **真正的价值**：在 offload 场景下，搬一个 expert 代价很大，让它一次服务更多 token 才划算。这条思路已有的相关工作：吞吐优先的 offload（FlexGen、MoE-Lightning、Klotski 的"expert-aware multi-batch pipeline"），以及 attention 与 FFN 分离部署（如 MegaScale-Infer）。这些都是在用延迟换吞吐（**未核实**各自的细节）。
- **我们的角度**：不是被动地等待，而是利用检索信号，**主动把会用到相同 expert 的请求放进同一批**，比如 schema 相同、候选实体有重叠的请求。这样每个被搬进来的 expert 能服务更多 token，又不需要额外等待。这个角度是否新，需要查新。

---

## 5. 信号清单（v2）

| 信号 | 何时可得 | 主要用于 | 备注 |
|---|---|---|---|
| S1. relation schema | RoG 规划完成后 | P 问题的搬运顺序、并发时的前瞻和组批；D 问题的先验 | proposal Fig. 7。需要控制 token 重叠的影响；schema 长尾严重（CWQ 测试集按 schema 集合分组有 1764 组，其中 1373 组只有 1 道题；53% 的题有多种 schema），要评估覆盖率 |
| S2. 候选实体 token | 落地过程中 | D 问题：预测生成答案时用到的 expert | 依赖"低层路由主要由 token 本身决定"（**未核实**） |
| S3. 会被复用的 token | 检索完成后 | P 问题：KV 复用的 token 不需要再过 FFN，需要计算的 token 和 expert 都减少 | 和 trie / T 的工作结合 |
| S4. 问题文本 | 请求到达时 | 最早可用；可以先单独过一遍各层的 gate 来预判 | 问题很短，信息量有限 |
| S5. 按 token 序列缓存的路由 | 离线建立 | 和上下文无关的片段（关系名、实体名），路由可以直接查表 | 关系名在几百道题里反复出现，覆盖率远高于完整 schema |
| S6. prefill 自己观察到的 expert（P、W） | prefill 中或之后 | D 问题的**基线** | proposal Fig. 4 |

---

## 6. 研究问题（v2）

| 编号 | 问题 | 判定 |
|---|---|---|
| **Q0** | 检索各阶段的耗时，以及在这些时间里能搬多少 expert | 如果窗口 < 需要搬运时间的 10%，就放弃"利用窗口"这个卖点，转向并发场景（Q4、Q5） |
| Q1 | prefill 每层用到的 expert 是否接近全部？KV 复用后（只算需要计算的 token）是多少？各 expert 分到的 token 数偏斜到什么程度？ | 决定 P 问题的优化方向 |
| Q2 | decode 时，各预测器在缓存预算 B 下的命中率（recall@B） | 衡量 D 问题的可预测性 |
| Q3 | 检索信号在 S6 之上能否带来额外命中？需要控制两件事：跨请求对照组、token 重叠 | recall@B 提升 ≥ 5 个百分点，或搬运量减少 ≥ 15% |
| **Q4** | 并发场景下，有了对队列的前瞻，expert 缓存的命中率和搬运量能比 LRU / LFU 好多少？离 Belady 还有多远？ | 相对 LRU 搬运量减少 ≥ 20% |
| **Q5** | 按 expert 亲和性组 batch，在相同延迟约束下吞吐能提高多少？ | 吞吐提高 ≥ 15%，且 P99 延迟不变差 |
| Q6 | 原型上的端到端 TTFT、每 token 延迟和吞吐 | 和模拟结果对照 |

---

## 7. 分阶段计划

### Phase 0：环境、模型与阅读器检查（W1）

- 下载 Mixtral-8x22B-Instruct（bf16，约 282GB）和 Qwen3-235B-A22B（bf16，约 470GB；如果 FP8 可用，再加 FP8 版用于采集 trace）。
- 确认 transformers 5.2 能否加载这些模型，以及能否加载 FP8；准备好 `device_map` 跨两张卡加主机内存的加载方式。
- **阅读器检查**：在 CWQ/WEB 各 200 题上跑 Direct 和 M0（few-shot），确认读路径确实有用（M0 明显高于 Direct）。

### Phase 0.5：时间线与带宽实测（W1，可以和 Phase 0 并行）

- 把 RoG 检索拆开计时：规划（LLM 生成）、在图上落地 / 枚举、组装 prompt。分别在 cap 8、cap 30，hop 1–3 下测。
- 测 pinned 主机内存 → H200 的实际带宽，以及单个 expert 的搬运时间（Mixtral 和 Qwen3 各测一次）。
- 计算各阶段的窗口能搬多少 expert，回答 Q0。**这是第一个 go/no-go。**

### Phase 1：路由 trace 采集（W2）

记录每个 token 在每一层的 top-k expert 编号和 gate 权重，以及 token 类别（指令、实体、关系、箭头、换行、问题、生成的答案；复用 `conditions.line_units`）。

工作负载：

- **W1：KGQA 短输出**：CWQ/WEB 测试集 500 题，用 M0 和 T 两种 prompt。
- **W2：长输出**：同一批题加思维链提示；Qwen3 用 thinking 模式。
- **W3：并发请求流**：用 W1 的题目构造到达序列，三种顺序：随机、按主题聚类、按实体流行度的 Zipf 分布。
- **离线画像**：在 train split 上采集一批，用来建立 S1 和 S5 的查找表，避免用测试题建表造成泄漏。

### Phase 2：离线分析（W3，CPU）

- **Q1**：每层的 expert 覆盖率、各 expert 的 token 负载分布；比较 M0、T、KV 复用后三种情况。
- **Q2 / Q3**：比较各预测器在不同 B 下的 recall@B。预测器包括：全局最常用、S6（本次 prefill 观察到的）、前一层的 gate、S1、S2、S5，以及它们的组合；Oracle 作为上界。必须做以下几组控制：
  - **跨请求对照组**：用请求 A 的 prefill 去预测请求 B 的 decode，作为"只是整体分布相似"的基线；
  - **按 decode 长度分组**：排除样本量的影响；
  - **只用问题 token / decode token 统计 profile**：排除 schema 相同导致 token 相同的影响；
  - **schema 覆盖率**：在测试集里，有多大比例的题，其 schema 在 train split 里出现过。
- **Q4（模拟前的估算）**：在 W3 的请求流上，算各种 expert 缓存策略下的搬运量：LRU、LFU、基于前瞻、Belady。
- **第二个 go/no-go**：Q3 和 Q4 至少有一个达标，才进入 Phase 3。

### Phase 3：trace 驱动的模拟器（W4）

- 逐层的离散事件模拟。用实测数据设定每层 attention 和每个 expert 的计算时间、PCIe 传输队列、显存中 expert 缓存的容量 C。
- **单请求**：P 问题的各种搬运顺序；是否与检索重叠（用 Phase 0.5 实测的窗口）；冷门 expert 放 CPU 算。D 问题的各种 prefetch 策略。
- **并发**：各种常驻 / 换出策略；按 expert 亲和性组 batch 与按到达顺序组 batch 的对比（Q5）。
- 扫描参数：C（显存里能放下 expert 的比例）、请求到达率、batch 上限、延迟约束。
- 基线：按需加载 + LRU、MoE-Infinity 式、ProMoE / Pre-gated 式、Klotski / MoE-Lightning 式的大 batch 摊薄。

### Phase 4：H200 上的原型（W5–W7，过了 go/no-go 之后）

- 自己写一个最小的 offload 运行时：在 HF 的 MoE 模块上挂钩子，expert 权重放在 pinned 主机内存里，显存里维护一个 expert 缓存，用独立的 CUDA stream 异步搬运。可以参考 Mixtral-offloading 和 MoE-Infinity 的开源代码。
- 检索与搬运重叠：检索在 CPU 上真实运行，或者使用 Phase 0.5 实测的耗时回放。
- 测 Q6：TTFT、每 token 延迟、吞吐，和模拟器的结果对照。

### Phase 5：与 KV 复用结合（可选）

T 或 trie 去重后，prefill 需要计算的 token 少了 55–70%，测这对 P 问题（需要的 expert 是否不再是"全部"）和端到端的叠加效果，把 MoE 这条线和前面的 trie 工作串起来。

---

## 8. 基线与相关工作

| 工作 | 在 library 中 | 做法 | 与本计划的关系 |
|---|---|---|---|
| MoE-Infinity | 有（笔记在 System/Group for Wang/Notes） | 按请求追踪 expert 激活，做 offload 和缓存 | D 问题的主要基线 |
| ProMoE | 有（同上） | 主动缓存，提前一层预测；笔记提到 batch 越大越容易掩盖搬运 | D 问题的主要基线 |
| Pre-gated MoE（ISCA'24） | 无（proposal [61]） | 用前一层的输入算下一层的 gate | D 问题的基线 |
| Fiddler | 无（[56]） | 缺失的 expert 直接在 CPU 上算 | P 问题的对照 |
| MoE-Lightning、Klotski、FlexGen | 无（[55]、[62]） | 吞吐优先的 offload，用大 batch / 多 batch 流水摊薄 expert 加载 | Q5（按亲和性组 batch）的基线，**未核实** |
| MegaScale-Infer 等 attention 与 FFN 分离部署 | 无 | expert 端汇聚多个 micro-batch 的 token | 相关，**未核实** |
| Mixtral-offloading（Eliseev & Mazur） | 无（[60]） | LRU 缓存 + 推测性加载 | 实现参考 |
| HOBBIT、AdapMoE、MoE-APEX | 无（proposal 引用） | 混合精度 offload、自适应 gate | 相关，**未核实** |
| "低层路由主要由 token 本身决定"（OpenMoE 等） | 无 | – | S2 / S5 的依据，**未核实** |
| proposal Fig. 4 / Fig. 7 / Fig. 8 | proposal | prefill 与 decode 的相似度；schema 与 expert 分布；KV 复用的前瞻 | 已有的初步结果，先问清楚是谁做的、代码在哪 |

---

## 9. 风险

| 风险 | 可能性 | 影响 | 应对 |
|---|---|---|---|
| 检索窗口太短（Q0 不过关） | 中 | 中 | 转向并发场景（Q4、Q5），不依赖空闲窗口 |
| prefill 时 expert 全部用到，P 问题只剩"顺序 + 常驻" | 高 | 中 | 这本身就是一个明确的结论；P 问题重点做常驻策略、CPU 计算和 KV 复用 |
| KGQA 的 decode 太短 | 高 | 中 | 用 W2 长输出；报告里把 W1、W2 分开讲 |
| schema 的效果其实只是 token 重叠 | 中 | 高 | Phase 2 专门做控制；如果成立，就改用 S5（token 级查表），说法也随之改变 |
| schema 长尾，新请求查不到 | 高 | 中 | 用 train split 建表并统计覆盖率；查不到时退回到 S5 / S6 |
| 按亲和性组 batch 已经有人做过 | 中 | 高 | 查新；如果已有，就把重点放在"检索提供的前瞻"上 |
| 超大模型的加载、FP8 支持等工程问题 | 中 | 中 | Phase 0 先解决；必要时用两张卡加主机内存的 device_map 采集 trace |
| GPU0/1 和其他人共享，影响计时 | 中 | 中 | 计时类实验在机器空闲时跑并重复多次；采集 trace 不受影响 |

---

## 10. 时间线（从 2026-10-17 起）

| 周 | 内容 | 产出（`developer-log/report/`） |
|---|---|---|
| W1 | Phase 0 + Phase 0.5 | 阅读器准确率、检索时间线与带宽（Q0，第一个 go/no-go） |
| W2 | Phase 1 | trace 数据 |
| W3 | Phase 2 | 可预测性与前瞻分析（Q1–Q4，第二个 go/no-go） |
| W4 | Phase 3 | 模拟器报告（含 Q5） |
| W5–W7 | Phase 4 | H200 原型的端到端报告（Q6） |

---

## 11. 待办

- [ ] **你**：问清楚 proposal Fig. 4 / Fig. 7 / Fig. 8 是谁做的、代码和 trace 在哪
- [x] 确认模型：Mixtral-8x22B + Qwen3-235B-A22B（含 FP8 版），2026-10-17 开始下载
- [ ] 是否再加入 DeepSeek-V2.5、GLM-4.5-Air（另需约 700GB）
- [x] Phase 0.5：RoG 检索分阶段计时、主机到 GPU 带宽测试（结论见 v3 变更记录）
- [ ] Phase 1：路由 trace 采集脚本（在 router 上挂钩子，复用 `conditions` 做 token 分类）
- [ ] 查新：各 offload 工作对 prefill 的处理；按 expert 亲和性组 batch 是否已有工作；OpenMoE 的路由结论
