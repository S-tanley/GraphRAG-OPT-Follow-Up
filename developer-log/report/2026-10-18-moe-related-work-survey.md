# Report：MoE 推理相关工作调研（20 篇）

*Bowen · 2026-10-18 · 论文 PDF 和全文在 `developer-log/context/papers/`（`txt/` 为抽取的文本，`ids.txt` 为 arXiv 编号）*

> 方法：网络搜索工具仍被组织设置禁用，所以论文是用 curl 从 arXiv 直接下载的，每篇都用首页标题核对过。全文由 4 个并行的子任务精读，要求结论都有原文依据；带"推算"的数字是子任务自己的计算，不是论文给出的。

## 一句话结论

1. **prefill 时几乎所有 expert 都会被用到**，这一点有论文明确写过（ProMoE、HOBBIT），所以在 prefill 阶段做预测几乎没有意义。
2. **"跨请求攒 token、再一次性交给 expert"已经有人做了**（MoE-Gen 的 module-based batching；Klotski 做的是规模小一些的版本），**不算新**。
3. **"在 prefill 之前预测路由"和"按预测的 expert 重新分组"也已经以基于 token 的形式出现过**（ExpertFlow）。
4. **没有人用请求级的元数据（例如检索得到的 schema、实体）来预测 expert，也没有人在请求准入时按 expert 亲和性组 batch，更没有人利用队列里未来的请求来决定 expert 缓存（类似 Belady）。** 这是剩下的空白。
5. **但有几项不利的证据**：路由很大程度上由 token 本身决定（OpenMoE），expert 流行度在不同领域之间几乎一样（Fiddler），batch 足够大时路由近似均匀（MoE-Gen），层与层之间的关联在不同语料上几乎不变（ExFlow）。所以首先要用实验证明：**检索元数据能提供 token 本身给不出的预测力**。

## 1. 逐篇要点

### A. 单卡 offload：预测与 prefetch（关注 decode 延迟，几乎都是 batch size = 1）

| 论文 | 预测信号 | 准确率 | prefill 的处理 | 代码 |
|---|---|---|---|---|
| MoE-Infinity | 当前请求的 expert 激活矩阵，与历史请求匹配（cosine） | 未给出 | 单独追踪；文中提到"同一上下文、同一 prompt 中 expert 会被重复激活"，但没有使用任务标签 | 有 |
| ProMoE | 每层一个学习出来的 2 层 MLP（输入 hidden state，约 200 万参数） | 平均 84.7%（token-ID 预测器 58.3%） | 写明 prefill 时"几乎所有 expert 都会被访问"；prefill 的收益主要来自重新排序执行顺序 | 有 |
| Pre-gated MoE | 修改模型结构：在第 N 层就算出第 N+1 层的 gate（需要微调） | 构造上就是精确的 | 未说明 | 有 |
| Mixtral-offloading | 把下一层的 gate 作用在当前层的 hidden state 上，推测性地加载 | 只在图中给出 | 写明 prefill"每层只需要加载一次" | 有 |
| HOBBIT | 把当前 gate 的输入送进接下来 p 层的 gate | Mixtral 下一层 top-1 准确率 96%，2–3 层后约 90% | 写明 prefill"大多数情况下用到每层的全部 expert" | 未说明 |
| AdapMoE | 后续层的 gate；第 1 层用一个训练出来的预测 gate | 约 90% | 未说明 | 有 |

### B. 单卡吞吐优先：攒 token / 攒 batch

| 论文 | 怎样摊薄 expert 的加载 | 每个 expert 实际分到的 token | 是否挑选请求 |
|---|---|---|---|
| FlexGen（dense） | 之字形顺序，一层的权重被多个 batch 共用 | – | 否 |
| MoE-Lightning | 按层分页加载权重，多个 micro-batch 共用；attention 放 CPU、MoE 放 GPU | 每层约 126，每次 kernel 约 9（推算） | 只按输入长度做均衡 |
| **MoE-Gen** | **module-based batching：在主机内存中跨多个 attention batch 攒 token，再以大 batch 交给 expert**；batch 上限由主机内存决定 | 例如 DeepSeek-V2：prefill 8192，decode 75（基线只有 0.3–0.4）；Mixtral decode 约 910（推算） | 否；**明确表示"大 batch 时预测已经没有必要"** |
| Klotski | 在一组 batch 内按 expert 而不是按 batch 组织计算；根据前一层的关联表预取热门 expert | 约 160（推算） | 否 |
| MoE-Lens | 尽量增加每次前向的 token 数；decode 的 attention 放 CPU | 约 4.8k（推算），达到约 90% 的计算上限 | 否（按先来先服务准入） |
| Fiddler | 按 expert 决定在 CPU 还是 GPU 上算 | – | 否；**发现 expert 流行度在不同领域之间"几乎通用"** |

剩下的瓶颈：**主机内存里 KV cache 的容量限制了并发的序列数**（MoE-Lens）；细粒度 MoE（例如 160 个 expert）每个 expert 分到的 token 仍然偏少。

### C. 大规模部署

| 论文 | 怎样让 expert 分到更多 token | 负载均衡 | 是否用请求元数据 |
|---|---|---|---|
| MegaScale-Infer | attention 与 FFN 分离部署：多个 attention 副本把 token 汇集到 expert 节点；2–4 个 micro-batch 交替执行；比例离线确定 | 根据"过去一段时间的流量"放置冗余 expert；**指出 prefill 的负载在 batch 之间波动大，decode 稳定** | 否（只用序列长度来平衡 attention） |
| DeepSeek-V3 | 大规模 expert 并行：prefill 32 卡，decode 320 卡；2 个 micro-batch 重叠计算和通信；decode 时每个 expert 通常不超过 256 个 token | 根据线上统计，**大约每 10 分钟**调整一次冗余 expert；计划中的"动态冗余"是按当前 batch 的路由临时决定的，属于事后反应，不是提前预测 | 否 |
| Lynx | 不是增加 token，而是**减少 batch 实际用到的 expert**：在 decode 阶段，把置信度低的 token 改派到 batch 已经在用的 expert 上（会改变路由，准确率变化在 1 个点以内） | – | **刻意不使用**任何工作负载先验 |
| Sarathi-Serve（dense） | chunked prefill + 按 token 预算组 batch；不等待 | – | 只用 prompt 长度 |

**没有任何一个系统**为每个 expert 设一个显式的、等待 token 到来的 buffer；"等待"都是在请求和 batch 层面完成的。

### D. 路由的可预测性

| 论文 | 发现 |
|---|---|
| OpenMoE | 从零训练的 MoE（OpenMoE、DeepSeek-MoE）里，**路由主要由 token 本身决定，而不是高层语义**；这种行为在训练早期就形成了。expert 并不按领域分工；看起来按任务分工的现象来自 token 构成的不同。Mixtral 是从 dense 模型改造来的，没有这个现象。没有给出各层的对比数据 |
| ExFlow | 层与层之间 expert 的关联性很强，**在不同语料（Pile、C4、Dolma、Yelp）上几乎不变**，属于模型固有的性质；用于多卡上的 expert 放置，以减少 all-to-all 通信 |
| ExpertFlow | 约 7MB 的 T5 式预测器，**从 batch 的输入 token 一次性预测所有层的路由**；in-domain 的 batch 级召回率 > 90%（这个指标比较宽松，没有报告精确率），跨领域会下降 5–10%。并且**把相邻两个 decode batch 的 token 按预测的路由重新分组**，让并集更小：吞吐 1.03–1.17 倍，缓存命中率最高 91.9%（LRU 为 36–77%） |
| SwapMoE | 只保留重要的"虚拟 expert"（按 token 的模长和 gate 值估计重要性），会改变路由；batch size = 1 |

## 2. 对我们几个想法的判断

| 想法 | 是否已有人做过 | 剩下的空间 |
|---|---|---|
| 在 FFN 前设 buffer，跨请求攒 token | **已有**（MoE-Gen、Klotski） | 本身不新；只能作为基础设施 |
| 在 prefill 时预测 expert 并 prefetch | 意义不大（几乎全部用到） | 只剩按层流式搬运和执行顺序的优化（ProMoE 的重新排序已经做了） |
| decode 时预测 expert 并 prefetch | **已有**多种（ProMoE 84.7%，HOBBIT 约 90–96%，ExpertFlow > 90%） | 检索信号必须超过这些基线；或者凭借"更早可用"拿到收益（offload 时 prefill 本身要几秒） |
| 在 prefill 之前从输入预测路由 | **已有基于 token 的版本**（ExpertFlow 的 RPP） | 用**请求级的检索元数据**来预测：没有人做过，但必须证明它超过基于 token 的预测器 |
| 按 expert 亲和性重新分组 | **部分已有**：ExpertFlow 只在相邻两个 decode batch 之间、基于 token 的预测重新分组；Lynx 是改变路由 | **在请求准入时**就按检索元数据挑选一起执行的请求，不改变路由：没有人做过 |
| 根据队列里未来的请求决定 expert 缓存（类似 Belady） | **没有人做过** | 空白；DeepSeek 和 MegaScale 都只用历史统计，而 MegaScale 指出 prefill 的负载波动最大 |

## 3. 必须正视的不利证据，以及对应的验证

1. **token 决定路由（OpenMoE）**：我们看到的"schema 相同 → expert 分布相似"可能完全来自 token 相同。验证办法有两个：(a) 只用 token→expert 的统计直方图来预测，和用 schema 预测比较；(b) **把关系名换成 token 不同的同义词**，看 expert 分布是跟着 schema 走还是跟着 token 走。
2. **流行度与领域无关（Fiddler）、关联性与语料无关（ExFlow）**：请求的语义可能只影响很小一部分路由。需要直接测**单个请求层面**的可预测性，而不是整体分布。
3. **大 batch 时路由近似均匀（MoE-Gen）**：在 batch 大到受内存上限约束的吞吐优先场景里，按亲和性组 batch 可能没有收益。只有当 batch 大小**受 KV cache 内存或延迟限制**时才有空间。所以实验要扫描 batch 上限。
4. **指标要统一**：各篇论文的准确率口径不同（top-1、batch 级召回率、全部 expert 的准确率），我们统一使用"每个 token 的 top-k recall@B + 精确率 + 搬运字节数"。
5. **我们的噪声上限**：batch 中的数值差异会让约 3% 的边界 expert 发生变化（Phase 1 的冒烟测试），任何预测器都不可能超过这个上限。

## 4. 可以复用的开源代码

MoE-Infinity、MoE-Gen（同一个团队 EfficientMoE）、ProMoE、Mixtral-offloading、Fiddler、AdapMoE、Pre-gated MoE 都开源了。MoE-Gen 的 module-based batching 可以直接作为吞吐场景的底座，在上面加我们的准入策略和缓存策略。
