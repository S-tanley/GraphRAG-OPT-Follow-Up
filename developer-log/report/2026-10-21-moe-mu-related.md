# Report：MoE 多用户调度——相关工作补查

日期：2026-10-21

对应 plan `2026-10-21-moe-multiuser-scheduling-plan.md` 的 0.1.1–0.1.3。每篇论文都核对过 arXiv 摘要页或会议页。表里写到的机制细节来自论文 HTML 正文，但正文是经网页抓取工具转述的，数字引用前请再对一遍 PDF。只看过摘要的条目会注明。

---

## 0. 结论

- **"常驻 + 流动"分区：最像的是 Diff-MoE（SC'25）。** 它在 GPU 里每层分三档：
  - 高优先级缓存，放"全局热门"expert（整个推理过程中都常用）；
  - 中优先级缓存，放"短期热门"expert（连续几步反复用），按优先级替换；
  - 冷 expert 临时放，按需逐出。

  它面向多请求 batch。但正文在 ACM 付费墙后，常驻部分怎么填、会不会更新，没能确认。
- **第二像的是 Mira（arXiv 2609.38090，2026-09）。** 它分 HOT（长期复用）和 STAGE（预测出来的、短期）两层。HOT 每隔固定数量的 token 做一次有上限的换入换出，和我们"常驻区按自己的周期刷新"最接近。只测了单请求。
- **字面上最一致的是 VisMMOE（arXiv 2605.05899）**，原文就叫 "static resident region + dynamic slab-based expert cache"。但它的常驻区按层深度选（前 L 层的全部 expert 加 shared expert），部署后不再变化。
- 神经元级的类比是 PowerInfer-2：固定区（attention + KV）、hot 区、cold 区，后两个都用 LRU。
- 用户记得的那篇：如果是"热门 expert 常驻，其余按优先级 / LRU 流动"，最可能是 Diff-MoE；如果是"常驻区周期刷新，另有预测区"，是 Mira。
- **四个组件都有很接近的已有工作。**
  - 组件 1（相似性组 batch）：arXiv 2608.07911（Yu Zhang，2026-08）做过几乎相同的实验。设置是 24 个会话、每步选 8 个、最多等 4 步；用上一 token 的路由做预测，贪心最小化并集。**miss 只相对降了 4.6–4.8%，oracle 也只有 5.9–7.9%。** 这是负面结果，我们必须正面回应。
  - 组件 3（顶替）：DAOP（DATE'25）的 graceful degradation 是"把低分、在 CPU 上的 expert 换成 GPU 上下一个最高分的 expert"，和我们的规则基本相同。SMoE、MoE-CORE 用的是带分数窗口的同类规则。
  - 组件 2（轨迹匹配）：FineMoE（EuroSys'26，v1 叫 fMoE）用已观察层的 gate 概率前缀，按 cosine 匹配历史 iteration。这和我们的 trajectory anchor 是同一思路。
- **没找到同时做"多用户 batch + offload + 顶替"的工作。** 已有工作最多覆盖其中两项：
  - Diff-MoE、两篇 ExpertFlow：batch + offload，但没有顶替；
  - DynaExq：batch 32 + offload，但兜底方式是降精度，不是换 expert；
  - MoE-CORE、SMoE：offload + 顶替，但 batch ≤ 3；
  - OEA、Lynx、SERE：在 batch 内改派 expert，但不涉及 offload；
  - 2608.07911：多用户 trace 模拟加亲和 batch，但明确排除了顶替。

  所以我们的卖点应该放在这个组合，以及"改动率预算"上。
- **建议的基线：**
  - Mira 式 HOT + STAGE 分层缓存，用来对照组件 4；
  - Cache-Prior（Skliar 等）的 logit 偏置路由，用来对照组件 3；
  - 另外用 2608.07911 的设置在我们的 trace 上复现一次组件 1，作为对照检查。

## 1. 常驻 + 流动分区

按和"常驻（自有刷新周期）+ 流动（LRU）"的吻合程度排序。

| # | 工作 | 出处 | 常驻部分怎么填 | 何时更新 | 流动部分怎么管 | 吻合程度 |
|---|---|---|---|---|---|---|
| 1 | **Diff-MoE**: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching（Kexin Li, Wenkan Huang, Qinggang Wang, Long Zheng, Xiaofei Liao, Hai Jin, Jingling Xue） | SC'25，pp.1951–1965，DOI 10.1145/3712285.3759903（无 arXiv 版） | 每层一个高优先级缓存，放"全局热门"expert | **未能确认** | 中优先级缓存放"短期热门"，按优先级替换（论文说 LRU 会把热门 expert 挤掉）；冷 expert 临时放、按需逐出；另有下一层预取 | 高，而且面向多请求 batch。只读到摘要 |
| 2 | **Mira**: Memory-Efficient MoE Inference Using Adaptive Caching and Predictive Expert Staging（Sanjali Yadav, Bahar Asgari） | arXiv 2609.38090 | HOT 区：按 token 级路由计数（路由量、由 CPU 代算的 token 数）挑选；初始怎么填没写 | 每处理固定数量的 token 做一次 rebalance，每次最多换 s 个（离线调参），计数器每轮清零；STAGE 里持续高频的才升进 HOT | STAGE 区：放预测器提前 2 层预测出的 expert；目标层过去就丢；满了就不再放 | 高，周期刷新最像；但流动区是"预测区"而不是 LRU；batch = 1 |
| 3 | **VisMMOE**: Exploiting Visual-Expert Affinity for Efficient Visual-Language MoE Offloading（Cheng Xu, Xiaofeng Hou, Jiacheng Liu, Chao Li） | arXiv 2605.05899 | shared expert，加上前 L 层的全部 expert；L 在部署时离线选定 | 部署后不变 | 剩余显存切成等大的 slab；按预测优先级装入；只逐出已过期的 expert（正在用和推测中的受保护） | 中：结构一致，但常驻区按层深度选，不按热度；针对视觉-语言 MoE |
| 4 | **PowerInfer-2**（Zhenliang Xue, …, Haibo Chen） | arXiv 2406.06282 | 固定区放 attention 和 KV；hot / cold 神经元由离线统计（1000 万以上 token 的激活频率）决定 | 固定区不换；hot / cold 两区的大小随 batch 大小变 | hot 区按神经元簇做 LRU，cold 区按单个神经元做 LRU | 中：神经元级的类比 |
| 5 | **LightMoE**: Task-Aware Expert Availability Management for Memory-Efficient MoE-LLM Inference（Puhan Luo, Yunhao Yao, Junyang Wang, Junyang Zhang, Xiangyang Li） | Findings of ACL 2026 | 按频率初始化一小组常驻 expert | 运行时按"任务"粒度整体替换 | 没有流动区：缺的 expert 直接重定向到相似的常驻 expert | 中：常驻加顶替。只读到摘要 |
| 6 | **DynaExq**（Kexin Chu 等） | arXiv 2511.15015 | 按路由 trace 估计长期热度，每层选 top-n 以高精度常驻 | 在线、异步地升级或降级 | 其余 expert 保留低精度版本 | 低到中：这是精度分层，不是驻留分层；batch 32 |
| 7 | **ExpertFlow**（Shen 等） | arXiv 2510.26730 | 两个 LRU 队列（高、低），优先从低队列逐出 | 分层随前瞻步长变化 | — | 低：没有固定常驻区 |
| 8 | **MoE-CORE**（Ke Yang 等） | arXiv 2610.01950 | 每层槽数按 trace 重放出的命中曲线分配；请求进入时，按领域 profile（各 expert 的平均 router 分数）填满 | 之后每个 token 按打分替换：窗口频率 + EMA + 当前分数 − 年龄 | 整个缓存都可替换 | 低：没有固定区，但填法（按负载类型、每层配额、滑动平均）和我们五种填法里的三种重合 |

**查过但没有分区的：**
- 只有一组常驻 expert：
  - Fiddler（ICLR'25）：按校准集上的热度一次性放到 GPU，之后不变；
  - PowerInfer（SOSP'24）：热神经元预加载到 GPU；
  - SwapMoE（ACL'24）：每隔若干输入按重要性更新常驻集合，miss 全部改路由到常驻 expert；
  - DAOP：按校准集初始放置，每条序列只在 prefill 时调整一次。
- 单一缓存加打分策略：
  - HybriMoE（DAC'25）：top-p 路由分数的 EMA；
  - FineMoE：LFU 类优先级；
  - MoE-Infinity：预测激活率乘以层位置衰减；
  - HOBBIT：LRU、LFU、高精度使用频率、层距离四项加权；
  - AdapMoE（ICCAD'24）：用 DP 分配每层缓存大小，替换用 LRU；
  - ProMoE：每层一个 LRU。
- LLM in a flash（ACL'24）：只有滑动窗口；attention 和 embedding 常驻 DRAM。
- EdgeMoE、Klotski、MoE-Lightning、MoE-Gen、FloE（ICML'25）、CoServe（ASPLOS'25）、eMoE：摘要里都没提分区，正文没读。

**对组件 4 有用的反面证据：**
- arXiv 2608.12103（*Who Should Own the Expert Cache?*）：按静态频率钉住的区域，在 trace 重放里比内核 LRU 快 1.09–1.11 倍。但领域一变就退化，按最近使用排序的方法则保持稳定。
- arXiv 2608.07911：continuous batching、B = 8 时，固定常驻集比 LFRU 多传约 7.5%。只有在按 deadline 轮转调度时，固定常驻集才赢。

两者都说明，常驻区必须刷新，也必须留出流动区。

## 2. 和各组件最接近的工作

| 工作 | 出处 | 做法 | 和我们的区别 |
|---|---|---|---|
| **组件 1：相似性组 batch** | | | |
| Reproducible Evaluation of MoE Expert Caching（v1 题为 *When Does Trace-Driven Evaluation Mislead MoE Expert Caching?*），Yu Zhang | arXiv 2608.07911 | 用 trace 驱动的模拟器。§8.3：最多 24 个会话，每步服务 8 个，最多等 4 步；用上一 token 的路由预测，贪心最小化并集。miss 相对降 4.6–4.8%，oracle 降 5.9–7.9% | **和组件 1 几乎相同，而且是负面结果。** 我们剩下的区别：anchor 预测、和预取 / 顶替联合、扫更大的 N、真机测量 |
| ExpertFlow（He 等；v1 题为 *Optimized Expert Activation and Token Allocation…*） | arXiv 2410.17954，DAC'26 | 一次前向预测所有层的路由路径。把相邻两个 batch 的 token 合在一起，按路径的 Hamming 相似度，用 K-means 分成两等份 | 按 token 分组，只在两批之间重组，没有公平性约束。我们从 N 个请求里选 B 个，并且有等待上限 |
| ExpertFlow（Shen 等） | arXiv 2510.26730 | "cache-aware routing"：所需 expert 已驻留的 token 先算，会引起加载的推后；没有等待上限 | 看的是"当前驻留了什么"，不是"彼此预测重叠多少"；没有公平性 |
| ELDR | arXiv 2607.00466 | PD 分离场景。用 prefill 阶段的 expert 签名，把请求分到不同的 decode worker（离线做均衡 K-means） | 请求级、跨 worker，没有 offload |
| Read-ME | NeurIPS'24，arXiv 2410.19123 | 把路由器移到骨干之外（pre-gating），于是可以提前排程、按 expert 组 batch 和缓存 | 要改模型并重新训练 |
| eMoE | arXiv 2503.06823 | 每隔几个 prompt 预测一次 expert；按 SLO、输出长度、加载延迟做任务感知调度 | 不按路由重叠组 batch |
| ExFlow / MoETuner | arXiv 2401.08383 / 2502.06643 | 利用层间 expert 亲和性，用整数规划做多卡 expert 放置 | 解决的是多卡通信，不是 offload；但它们证明了层间亲和稳定，这是 anchor 预测的依据 |
| **组件 2：多层预测 + 轨迹匹配** | | | |
| HOBBIT | arXiv 2411.01433 | 把当前 gate 的输入送进后面 p 层（p ≤ 3）的 gate；前一层预测的 expert 已在缓存时才继续往后预取；预测出的 expert 不许逐出。Mixtral 下一层 top-1 准确率 96%，后 2–3 层约 90% | 我们看到 8 层，有置信门限、deadline 队列和取消；HOBBIT 只测单请求 |
| AdapMoE | ICCAD'24，arXiv 2408.10284 | 用后 2–3 层的 gate 预测；第一层用一个训练出来的预测 gate | 同上 |
| Pre-gated MoE | ISCA'24，arXiv 2308.12066 | 微调时训练 pre-gate，让第 N 块选出第 N+1 块的 expert；batch = 1 | 要训练，只看 1 层；我们不需要训练 |
| ProMoE | arXiv 2410.22134 | 每层一个离线训练的 2 层 MLP 做预测。推测加载是低优先级任务；真实 gate 一出来，就丢掉该层剩下的推测任务。每层 LRU；batch 1–4 | "丢弃过时预测"和我们的取消相同；但预测器要训练 |
| **FineMoE**（v1 名 fMoE），Hanfei Yu 等 | EuroSys'26，arXiv 2502.05370 | 每个 iteration 存一张"层 × expert"的 gate 概率图（最多存 1000 张）。用已观察层的概率前缀按 cosine 找最像的一张历史图（trajectory），前 d 层用 prompt 语义匹配。累积概率门限 δ = clip(1 − 相似度)，越不像预取越多 | **最接近我们的 trajectory anchor。** 它只取最像的一张图，我们取 kNN 投票，给每个 expert 一个置信度；它的预取距离 d 固定为 3–6 层 |
| MoE-Infinity | arXiv 2401.14361 | 请求级的 expert 激活矩阵，和历史矩阵做 cosine 匹配，再乘层距离衰减；同一个分数也用来决定逐出；batch = 1 | 请求级，不是 token 级；不涉及多用户 |
| DoMoE | IJCAI'26 | 按领域建 expert 路由表，只在和领域相关的 token 上做语义匹配 | 领域级；我们是 token 级 kNN |
| ExpertFlow（Shen 等） | arXiv 2510.26730 | pre-gate 输出加随机森林修正；前瞻步长按带宽和"停顿 / 多取"计数自适应调整；按累积概率决定取几个 | 和"按置信度取"相近，但步长是全局的；我们按单个 expert 的置信度和 deadline 决定 |
| **组件 3：缓存感知路由 / 顶替** | | | |
| **DAOP**（Yujie Zhang, Shivam Aggarwal, Tulika Mitra） | DATE'25，arXiv 2501.10375 | 两个预测 expert 都在 CPU 时，把分数低的那个换成 GPU 上下一个最高分的 expert；找不到就按原路由 | **规则和我们几乎相同。** 但它是单请求，没有改动预算 |
| SMoE（v1 题为 *Enabling MoE on the Edge via Importance-Driven Expert Scheduling*） | arXiv 2508.18983 | 设 b 为第 k+1 名的分数。分数落在 [b, (1+α)b] 的未驻留 expert，可以换成分数在 [(1−α)b, b] 的驻留 expert。α 在 TPOT 预算下搜索 | 用分数窗口控制，而不是控制改动比例；batch 为 1 或 3 |
| MoE-CORE 的 AES | arXiv 2610.01950 | 和 SMoE 同样的窗口规则（α = 0.1），一对一配对，保留原 expert 的路由权重；默认关闭 | 同上；batch 1–2 |
| **Cache-Prior**（Skliar 等，*Mixture of Cache-Conditional Experts…*） | TMLR 06/2025，arXiv 2412.00099 | 给已缓存 expert 的 logit 加 λ·Δ_avg（Δ_avg 是该层 logit 极差的滑动平均）；前 J 名永远保留；加权仍用原 logit；缓存用 LRU。miss 降一半以上，ppl 增加 0.1–3% | 软偏置，没有显式预算；batch = 1 |
| BuddyMoE | arXiv 2511.10054 | 离线按共同激活频率找"伙伴"；预取没命中时换成已驻留的伙伴；每个 token 最多换 3–4 次 | 伙伴离线固定。缓存 75% 时准确率约降 4 个点（0.735→0.695） |
| LightMoE | Findings ACL'26 | 缺的 expert 重定向到相似的常驻 expert | 相似度怎么算没读到（只读到摘要）；面向端侧 |
| HOBBIT / AdapMoE / Not All Experts are Equal（ACL'24，arXiv 2402.14800） | 同上 / 同上 / 同左 | HOBBIT：不重要的 miss 用低精度版本顶替。AdapMoE：按 Fisher 敏感度，把 top-2 降成 top-1。Not All Experts：第二个 expert 的权重低于 β 乘以第一个时跳过（β 取各层校准中位数） | 换精度或直接跳过，不是换成别的 expert；与缓存状态无关（HOBBIT 除外） |
| SwapMoE | ACL'24，arXiv 2308.15030 | 只保留一组"虚拟 expert"，用 masked gating 把所有路由改到它们上；batch = 1 | 全部强制改路由，没有预算 |
| Lynx / OEA / SERE | arXiv 2411.08982 / 2511.02237 / ICLR'26 arXiv 2602.07616 | 在 batch 内，把低亲和的 token 改派到 batch 已经在用（或相似）的 expert，减少本步用到的 expert 数 | 多用户版的顶替，但目的是省 HBM 带宽，不涉及 offload |

## 3. 建议的基线

两个方案都只需要 trace：每个 token 每层的 top-k 编号、router 打分、GATE-h 预测。

1. **Mira 式 HOT + STAGE 分层缓存**，对照组件 4（也部分对照组件 2）。
   - 选它而不选 Diff-MoE，是因为 Diff-MoE 的细节读不到，没法忠实复现。
   - 模拟方法：
     - 总槽数 C 分成 HOT 和 STAGE 两部分。
     - 每 T 个 decode 步统计一次每个（层, expert）的路由次数和 miss 次数。HOT 里路由量最低的，最多换出 s 个；换入非 HOT 里路由量最高的。
     - STAGE 放 GATE-2 预测的 expert，目标层算完就丢，满了就不再放。
     - Mira 的预测器是训练出来的，这里用 GATE-2 代替，报告里要写明。
     - T、s、HOT 占比做网格搜索。在同样的总槽数下，和我们的 ρ 常驻 + LRU 比 miss 和传输字节。
2. **Cache-Prior logit 偏置路由**，对照组件 3。
   - 模拟方法：
     - 每个 token 每层，算 z′ = z + λ·Δ_avg·m，其中 m 是当前驻留掩码。
     - 原 top-J（J = 1 或 2）保留，其余名额按 z′ 取 top-k。
     - λ 从 0 扫到 1，记录 miss、改动的 expert 使用比例，以及丢掉的打分质量。
     - 在相同改动比例下，和我们的"下一个最高分驻留 expert + 改动预算 ρ_c"比较。
   - 注意：
     - 如果 trace 只存了前 16 名的打分，只能在前 16 名里选，λ 不宜太大。
     - GLM 用 sigmoid + bias 打分，Δ_avg 要按打分的极差来算。
     - 质量影响 trace 给不出来，最后要用模型真跑一次。
- **附带的对照检查**：在我们的 trace 上，用 2608.07911 的设置（N = 24、B = 8、W = 4、上一 token 路由、贪心最小并集）复现组件 1。如果我们的 miss 降幅也只有约 5%，就要靠 anchor 预测，或者靠和顶替、预取的联合收益，来拉开差距。

## 4. 未能确认的

- **Diff-MoE**：ACM 全文返回 403，只读到摘要（通过 OpenAlex、Semantic Scholar 和 DBLP 核对）。"全局热门"怎么识别（离线还是在线）、会不会刷新、各档容量占比，都没确认。
- ***Which Expert Misses Are Worth Waiting For? Substitution and Caching for Mixture-of-Experts Inference on a Consumer GPU***（Research Square rs-11268552 / ResearchGate）：页面打不开，只看到标题和搜索摘要（Qwen3-Next-80B、1/4 expert 常驻、研究哪些 miss 值得等）。作者、日期、是否经过同行评审都不知道。它和组件 3、4 都很相关，建议手动下载来看。
- Mira、VisMMOE、MoE-CORE、DynaExq、2608.07911、2608.12103、SMoE、BuddyMoE、ELDR 都是 arXiv 预印本，没有经过同行评审。其中 2026 年的几篇超出我的已有知识，只能靠这次抓取的内容。
- LightMoE、Read-ME、eMoE、Klotski、MoE-Lightning、MoE-Gen、FloE、CoServe、EdgeMoE、FluxMoE（arXiv 2604.02715）只读了摘要，没有确认正文里有没有常驻 / 流动分区。
- PowerInfer 的热神经元放置是不是完全离线、运行中不更新：摘要只说"预加载"。
- BuddyMoE 正文有前后矛盾：算法 1 里没有 gate 检查，第 3 节却有；"准确率损失可忽略"也和表中的数字不符。
- MoE-Infinity、Klotski、MoE-Lightning、ExFlow 的发表会议没确认，页面只列了 arXiv。
- 文中的数字都来自论文自报，我们没有复现过。
