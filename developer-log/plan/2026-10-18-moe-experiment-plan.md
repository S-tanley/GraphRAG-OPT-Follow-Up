# Experiment Plan：检索信息能否指导 MoE 的 expert 调度（显存受限场景）

*Bowen · 2026-10-18 · 状态：GLM-4.5-Air 的 E0–E4 已完成（四个假设都未达到门槛，见 `report/2026-10-18-moe-e0-e4-glm-interim.md`）；其余 3 个模型的 E0 因 GPU0/1 硬件故障暂停，等待管理员 reset*

本计划细化 `2026-10-16-moe-retrieval-prefetch.md`（v3）的 Phase 2–4，依据是：

- `report/2026-10-17-moe-phase0.5-timeline.md`：拿到 schema 后，只剩约 2ms 就进入 prefill，没有可用的提前窗口；
- `report/2026-10-17-retrieval-guidance-concerns.md`：两个根本性疑问，以及待验证事项 V1–V6；
- `report/2026-10-18-moe-related-work-survey.md`：20 篇相关工作的调研。

---

## 0. 与之前计划相比，放弃了什么、保留了什么

| 方向 | 状态 | 原因 |
|---|---|---|
| 在 prefill 阶段预测 expert | **放弃** | prefill 时几乎用到全部 expert（ProMoE、HOBBIT 都写明了；我们的估算也一致） |
| 拿到 schema 后提前 prefetch | **放弃** | 窗口只有约 2ms（Phase 0.5） |
| 在 FFN 前设 buffer、跨请求攒 token | **不作为贡献** | MoE-Gen、Klotski 已经做了；只作为实验的底座 |
| decode 阶段的预测 | 保留，但必须超过强基线 | ProMoE 84.7%，HOBBIT 约 90–96%，ExpertFlow 召回率 > 90% |
| **请求准入时按 expert 亲和性组 batch** | **核心之一** | 没有人在准入时用请求级元数据挑选一起执行的请求（ExpertFlow 只在相邻两个 decode batch 之间、基于 token 重新分组） |
| **根据队列里的未来请求决定 expert 缓存** | **核心之一** | 没有人做过；现有系统都只用历史统计 |
| **决定 offload prefill 结束时保留哪些 expert** | 新增 | prefill 期间 PCIe 被搬运占满，没有余量做 decode 的 prefetch；真正的决策是"prefill 流过的 expert 里，留下哪些给 decode" |

---

## 1. 要验证的假设

| 编号 | 假设 | 对应的疑问 | 不成立时怎么办 |
|---|---|---|---|
| **H1** | 请求级的检索元数据（schema、候选实体、路径结构）对**单个请求**的 expert 使用有预测力，而且在控制住 token 重叠之后，**仍然超过**基于 token 的预测器 | 疑问 1，前提 A（V1、V2） | H2–H4 改用基于 token 的预测，贡献就不再是"检索特有的"，需要和导师讨论是否继续 |
| **H2** | 在 batch 大小受限（受 KV 内存或延迟约束）时，准入阶段按预测的 expert 集合组 batch，可以让每一步用到的 expert 并集更小，在相同延迟下提高吞吐 | V5 | 回到系统设计本身（逐层流水、attention / FFN 分离） |
| **H3** | 多请求并发时，利用队列中请求预测的 expert 集合来决定常驻和换出，可以比 LRU、LFU、MoE-Infinity 式的策略少搬运，接近 Belady | V4 | 放弃并发前瞻这个方向 |
| **H4** | 在 offload 的 prefill 结束时，按 decode 的先验决定保留哪些 expert，可以减少 decode 前若干步的等待 | 疑问 2 | decode 的优化只剩常规的 prefetch |

---

## 2. 实验设置

**硬件**：mbzuai-masdar 的 GPU0/1（H200 NVL，各 143GB），主机内存 1.5TB，pinned 内存到 GPU 约 54GB/s。采集 trace 时用两张卡加主机内存；E5 的原型用单卡加主机内存。

**模型**（都超过单卡容量，transformers 5.2 都原生支持）：

| 模型 | 层 · expert（选 k） | 共享 expert | 是否从零训练 | 状态 |
|---|---|---|---|---|
| GLM-4.5-Air | 45 · 128（8） | 1 | 是 | 已下载，E0 采集中 |
| Mixtral-8x22B | 56 · 8（2） | 0 | 否（由 dense 改造），按 OpenMoE 的结论，token 决定路由的程度可能较弱 | 下载中 |
| Qwen3-235B-A22B | 94 · 128（8） | 0 | 是 | 下载中（FP8 版已下载） |
| DeepSeek-V2.5 | 59 · 160（6） | 2 | 是 | 下载中 |

是否从零训练这一项，用来检验 OpenMoE 的结论（"从零训练的模型，路由更多由 token 决定"），并分析 H1 在不同类型模型上的差异。

**工作负载：**

| 编号 | 内容 | 用途 |
|---|---|---|
| W1 | CWQ/WEB 测试集各 500 题，M0 prompt，cap 8，短输出 | 主工作负载 |
| W2 | 同一批题加思维链提示（GLM、Qwen3 用 thinking 模式），最多 512 个 token | decode 占比更高的场景 |
| W3 | 用 W1 构造请求流：泊松到达；按随机、按主题聚类、按实体 Zipf 三种顺序 | E2、E3 的输入 |
| **Wsyn** | W1 中出现最多的 50 种关系，用**语义相同但 token 不同的改写**替换（例如 `location.location.containedby` → `is located within`），其他内容不变 | H1 的关键控制：expert 分布是跟着 schema 走，还是跟着 token 走 |
| Wtrain | CWQ/WebQSP train split 约 3000 题 | 建立 S1 / S5 的查找表，训练基于 token 的预测器；不用测试题，避免泄漏 |

**噪声上限**：同一请求在不同 batch 形状下，约 3% 的边界 expert 会变化（Phase 1 的冒烟测试），任何预测器的 recall 上限约为 97%。

---

## 3. 实验

### E0：trace 采集与阅读器检查（进行中，W1）

- 在已经实现的 `experiments/moe/collect_traces.py` 基础上，对 4 个模型采集 W1、W2、Wsyn、Wtrain，同时跑 Direct 作为准确率对照。
- **需要补充的采集项**：
  - 提前一层的 gate 预测：把第 l+1 层的 router 作用在第 l 层 MoE 的输入 hidden state 上，记录得到的 top-k。这是 Mixtral-offloading、HOBBIT 式基线所需要的，事后无法补算。实现方式是在 MoE 模块上挂前置钩子，按各模型自己的路由规则计算（softmax top-k；DeepSeek 用分组贪心；GLM 用 sigmoid 加偏置）。
  - 每题的 schema、候选实体，以及每个 token 属于哪个单元（已有 `line_units`）。
- **完成标准**：每个模型 M0 的 Hit 明显高于 Direct（否则这个模型不算在"读证据"，H1 的结论不适用于它）；并且通过 batch 与单题的一致性检查（prefill 的 expert 重合率 ≥ 95%）。

### E1：路由可预测性（H1，W2，CPU 离线分析）

**预测对象**：decode token（主要）；问题 token、路径 token（次要）。

**预测器：**

| 预测器 | 可用时间 | 来源 / 类别 |
|---|---|---|
| 全局最常用（Wtrain 统计） | 永远 | 基线 |
| **token-ID 直方图**：每个 token ID 在各层最常去的 expert（Wtrain 统计） | 准入时 | 基线（OpenMoE 的发现） |
| 时间局部性：沿用上一个 decode 步的路由 | decode 时 | 基线（ExpertFlow 的启发式） |
| 提前一层的 gate | decode 时，提前一层 | 基线（Mixtral-offloading、HOBBIT） |
| prefill 观察：本请求 prefill 时各层的 expert 频率 P、权重 W | prefill 后 | 基线（proposal Fig. 4） |
| **RPP-lite**：一个小模型，输入 prompt 的 token，输出各层 expert 的概率（参照 ExpertFlow，在 Wtrain 上训练） | 准入时 | **最强的基于 token 的基线** |
| S1：按 schema 查表（Wtrain 上 schema → 各层 expert 分布） | 准入时 | 我们的 |
| S2：候选实体名的 token 在各层的路由（来自 token-ID 表或 trie 缓存） | 准入时 | 我们的 |
| 组合：基线 + S1 / S2 | – | 我们的 |
| Oracle | – | 上界 |

**指标：**

- 每个 token、每层的 recall@B（B ∈ {k, 2k, 4k, E/8, E/4}）和精确率；
- 按 batch 统计的并集召回率（和 ExpertFlow 的口径对齐，便于比较）；
- 按照预测集合回放缓存时需要搬运的字节数。

**控制组：**

1. **跨请求**：用请求 A 的信号去预测请求 B，作为"只是整体分布相似"的下限；
2. **Wsyn 改写**：比较改写前后，schema 信号的预测力还剩多少；
3. **按 decode 长度分组**；
4. **schema 覆盖率**：测试题的 schema 在 Wtrain 里出现过的比例，以及没出现过时的退化情况。

**判定（第一个 go/no-go）**：

- 在 decode token 上，"最强基线 + S1 / S2"相比"最强基线"，recall@B 提升 ≥ 5 个百分点，并且在 4 个模型中至少有 2 个成立；
- 在 Wsyn 下，schema 带来的提升至少保留一半。

两条都满足，H1 才算成立。

### E2：准入时按亲和性组 batch（H2，W3，trace 驱动的模拟）

- **组 batch 的策略**：先来先服务；按长度均衡（MoE-Lightning）；相邻 batch 重新分组（ExpertFlow 式，用 RPP-lite 的预测）；**我们的**：在最多等待 τ 的前提下，按预测的 expert 集合（S1 / S2 或组合）做聚类后再组 batch；oracle（用实际的路由集合）。
- **扫描参数**：batch 上限 ∈ {4, 8, 16, 32, 64}（对应 KV 内存或延迟的限制）；最长等待 τ ∈ {0, 50, 200, 500 ms}；到达率。
- **代价模型**：expert 的大小来自各模型的 config；搬运带宽 54GB/s；GPU 计算时间按实测的 roofline（单个 expert 处理不同 token 数的耗时，等两张卡空闲时实测）。
- **指标**：每个 decode 步用到的 expert 并集、每个 expert 分到的 token 数、每个 token 平均搬运的字节数、模拟的每 token 延迟和吞吐、等待带来的额外延迟（P50 / P99）。
- **判定**：在相同 P99 延迟下，相对先来先服务和 ExpertFlow 式策略，吞吐提高或搬运字节数减少 ≥ 15%。

### E3：根据队列前瞻决定 expert 缓存（H3，W3，模拟）

- **缓存**：显存里能放下全部 expert 的 C ∈ {30%, 50%, 70%}。
- **策略**：LRU、LFU、MoE-Infinity 式（与历史请求的激活矩阵匹配）、HOBBIT 式（多项加权）、**前瞻**（用队列中接下来 Q 个请求的预测 expert 集合来决定换出谁，扫描 Q）、Belady（oracle）。
- **指标**：搬运字节数、命中率、因等待 expert 造成的停顿时间。
- **判定**：相对最好的非 oracle 基线，搬运字节数减少 ≥ 20%。

### E4：offload prefill 结束时保留哪些 expert（H4，模拟）

- **背景**：offload 时，prefill 会把各层不在显存里的 expert 全部流式搬一遍，PCIe 是满的，没有余量做 decode 的 prefetch。所以真正的决策是：每层算完后，在容量 C 内**留下哪些 expert** 给接下来的 decode。
- **策略**：保留最后搬入的（LRU）；保留 prefill 中用得最多的（频率 P）；保留 decode 先验预测的（S1 / S2）；组合；oracle（decode 实际要用的）。
- **指标**：decode 前 N 步（N = 1、8、32）的未命中数和停顿时间；W1、W2 分别统计。

### E5：H200 上的原型（只有 E2、E3、E4 中至少一个通过后才做）

- **底座**：吞吐场景用 MoE-Gen（开源，已有 module-based batching），延迟场景用 MoE-Infinity（开源）。在它们之上加入：准入时按亲和性组 batch、前瞻式 expert 缓存、prefill 结束时的保留策略。
- **测量**：在 Mixtral-8x22B 和 Qwen3-235B 上，单张 H200 加主机内存，测真实的 TTFT、每 token 延迟和吞吐。
- **对照**：原版 MoE-Gen、MoE-Infinity、ProMoE、Fiddler（都开源），以及 E2–E4 的模拟结果。

---

## 4. 执行顺序与决策点

```
E0（采集 trace）──→ E1（可预测性）──┬─ H1 成立 ──→ E2 / E3 / E4 用检索信号 ──→ 有一个通过 ──→ E5
                                    │
                                    └─ H1 不成立 ──→ E2 / E3 / E4 用基于 token 的预测
                                                     （不再是"检索特有的"贡献 → 和导师讨论是否继续）
```

E2、E3、E4 都是 trace 驱动的模拟，可以在 E1 进行的同时搭好框架。

---

## 5. 时间线（从 2026-10-19 起）

| 周 | 内容 | 产出（`developer-log/report/`） |
|---|---|---|
| W1（10/19–10/25） | E0：4 个模型 × W1 / W2 / Wsyn / Wtrain；补充提前一层的 gate 预测；实测单个 expert 计算的 roofline | 阅读器准确率、trace 质量 |
| W2（10/26–11/01） | E1；同时搭建模拟器 | 可预测性报告（第一个 go/no-go） |
| W3（11/02–11/08） | E2、E3、E4 | 模拟报告 |
| W4（11/09–11/15） | 决策；如果通过，开始 E5 | – |
| W5–W7 | E5 | 原型报告 |

---

## 6. 风险

| 风险 | 可能性 | 应对 |
|---|---|---|
| H1 不成立（路由主要由 token 决定） | **中到高**（OpenMoE、Fiddler、ExFlow 都指向这个方向） | 这本身就是一个明确的结论；E2 / E3 / E4 仍可以基于 token 的预测来做，但需要重新评估论文的定位 |
| batch 大时路由近似均匀（MoE-Gen），E2 没有收益 | 中 | 只在 batch 受限的场景下讨论；扫描 batch 上限，给出收益随 batch 增大而消失的拐点 |
| Mixtral（由 dense 改造）表现和其他模型很不一样 | 中 | 这正好可以检验 OpenMoE 的结论，单独分析 |
| Wsyn 改写后模型读不懂路径，准确率下降 | 中 | 只用于分析路由，同时报告准确率；改写尽量用自然语言短语 |
| 集群连接不稳定（VPN 断开） | 已发生 | 所有任务都在 tmux 中运行；结果写到磁盘；断线不影响运行 |
| GPU0 / 1 与其他用户共享，影响计时 | 中 | E2–E4 用代价模型和离线实测的 roofline；E5 的计时在机器空闲时进行，并重复多次 |

---

## 7. 待办

- [x] E0：在 `collect_traces.py` 中加入提前一层的 gate 预测（各模型使用自己的路由规则；GLM、Mixtral、Qwen3 的自洽性都是 1.000）
- [x] 构造 Wsyn（前 120 种关系，覆盖 80% 的出现次数；token 重叠 ≤0.2），以及 Wsyn-rand（每种关系约 3 种改写，每个请求随机选一种）
- [x] 采集 Wtrain（2000 题）
- [x] 写 E1 的分析脚本（各预测器 + 控制组 + bootstrap）；RPP（ridge）和 RPP_R（加检索特征）
- [x] 搭建 E2–E4 的 trace 驱动模拟器（另加 HYB、SW、MIX 策略和分组准入）
- [x] GLM-4.5-Air：E0–E4 全部完成
- [ ] E0：Mixtral（W1 已完成）、Qwen3-235B、DeepSeek-V2.5，以及 GLM 的 Wsyn-rand。等 GPU0/1 reset 后，由 tmux `e0` 里的 `run_e0_after_reset.sh` 自动续跑
- [ ] DeepSeek-V2.5 在 transformers 5.2 下的加载问题：已加入减小显存预算后重试的逻辑，待在 GPU 上验证
- [ ] 每个模型完成后：`e1_analysis.py --export`，然后 `run_sim2.sh <model>`，再更新报告
- [ ] 实测单个 expert 处理不同 token 数的耗时（两张卡空闲时）
- [ ] 和导师讨论：GLM 上 H1–H4 都不成立后，这个方向是否继续（见报告 §7）
- [ ] **你**：问清楚 proposal 的 Fig. 4 / Fig. 7 是谁做的，是否有现成的 trace
- [ ] **你**：请管理员 reset masdar 的 GPU0/1；确认 GPU2 能不能用
