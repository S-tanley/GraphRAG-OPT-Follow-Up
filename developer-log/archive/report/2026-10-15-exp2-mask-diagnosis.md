# Report：实验 2，独立编码掉的点从哪里来（mask 诊断）

*Bowen · 2026-10-15 · 对应 plan v2 §7 实验 2*

## 一句话结论

**把 trie 写成文本（T）再用普通 causal 读，在 4 个模型、2 个数据集上都不掉点，同时路径 token 减少 50–58%。** 用 mask 实现的 trie（M2）都比 block-diagonal（M1，即 GraphRAG-Opt 的 reuse-all）好，大约能补回 M1 一半的损失，但相对 M0 仍掉 2–7 个点。损失几乎全部来自路径数 ≥5 的 query，原因是缺少跨路径的 attention。位置方案的影响因模型而异，不存在统一的最优方案。

## 设置

- **引擎**：`experiments/common/masked_engine.py`。直接调用 HF 模型自带的各个子模块（投影、norm、rotary、MLP），上下文部分支持任意 bool mask，并把"内部计算用的位置 `pos_int`"和"被问题读取时用的位置 `pos_out`"分开设置。
- **条件构造**：`experiments/common/conditions.py`。tokenization 与 `build_prompt` 完全一致，trie 建在逐行 token 上。
- **数据**：CWQ 和 WEB，seed-42 子集各 500 题；cap=8；路径来自 `precomputed_paths_v2`（cap=8 时与 v1 相同）。
- **生成**：greedy，最多 16 个 token。fewshot 模型在第一行有效答案生成完毕后提前停止，该规则与 `extract_answer` 等价，已用边界用例验证。
- **模型**：RoG、Llama-2-7B（fewshot）在两个数据集上跑全部条件；Qwen3-8B、Llama-3-8B（base，fewshot）只在 CWQ 上跑核心条件。
- **统计**：Hit 为 RoG 口径；配对 exact McNemar 检验。

### 正确性 gate（`experiments/exp2/test_engine.py`，RoG/CWQ 前 100 题）

| gate | 结果 |
|---|---|
| 引擎跑 M0 vs HF 原生 `full_generate` | 预测 100/100 相同；首 token 概率最大差 7.5e-3（fp16） |
| 引擎跑 M1-hyb vs GraphRAG-Opt reuse-all（手写 Llama 引擎） | 98/100 相同（两套引擎的 fp16 数值差异） |
| 同一 trie 节点在不同 query 中的 K/V（486 对） | 最大差 1.56e-2，约为 fp16 在数值 ~16 处的 1 个 ulp，属于不同长度下 kernel 的舍入差异 |

## 主表（ΔHit 相对 M0，括号内为变好/变差题数；* 表示 p<0.05）

| 条件 | RoG CWQ | RoG WEB | Llama-2 CWQ | Llama-2 WEB | Llama-3 CWQ | Qwen3 CWQ | 路径区 token（Llama-2 tok，CWQ / WEB） |
|---|---|---|---|---|---|---|---|
| **M0**（Hit） | 0.508 | 0.754 | 0.406 | 0.678 | 0.438 | 0.404 | 222 / 166 |
| M0-shuf1/2/3 | +1.6 / +0.2 / +0.6 | −0.8 / −0.2 / +0.8 | −2.2 / −1.0 / −0.8 | −1.6 / +0.2 / −0.4 | – | – | 222 / 166 |
| **T**（文本 trie） | **+0.8** | **+0.6** | **−0.6** | **+0.4** | **−0.4** | **+0.2** | **100 / 83** |
| M1-seq | −3.6* | −5.6* | −19.0* | −25.8* | −9.0* | −6.0* | 222 / 166 |
| M1-par | −8.0* | −10.0* | −11.0* | −9.2* | – | – | 222 / 166 |
| M1-hyb（= GraphRAG-Opt reuse-all） | −6.8* | −4.4* | −12.4* | −9.8* | −9.6* | −8.6* | 222 / 166 |
| M2-hyb | −3.4* | −3.0* | −7.2* | −3.2 | −5.8* | −6.6* | **90 / 73** |
| M2-par | −2.2 | −3.4* | −7.4* | −4.2* | −4.2* | −4.4* | **90 / 73** |

（完整数字和分桶见 `outputs/exp2/summary.md`。）

按路径数分桶（以 RoG CWQ 为例）：

| 条件 | ≤4 条路径（n=70） | ≥5 条路径（n=430） |
|---|---|---|
| M0 | 0.600 | 0.493 |
| M1-hyb | 0.614 | 0.412 |
| M2-par | 0.571 | 0.472 |
| T | 0.614 | 0.500 |

Llama-2 CWQ 的规律一样：≤4 条路径时所有条件都在 0.47–0.51 之间；≥5 条路径时 M1-seq 只有 0.174，而 M0 是 0.388。

## 结论

1. **噪声底**：只打乱路径顺序，ΔHit 就在 −2.2 到 +1.6 之间，没有一组显著。后续 2 个点以内的差异都不作解读。
2. **文本层面的去重是"免费"的**：T 在全部 6 个组合上都和 M0 没有显著差异（|Δ| ≤ 0.8），路径 token 却少了 50–58%。T 不需要改模型，也不需要改 kernel，是一个很强、很简单的结果；而且对 prefill FLOPs、KV 显存和 decode 读取量都直接生效。示例：

   ```
   M0: Jackson Pollock -> influence.influence_node.influenced_by -> Carl Jung
       Jackson Pollock -> influence.influence_node.influenced_by -> Jan-Hein Arens
       …（共 8 行）
   T : Jackson Pollock -> influence.influence_node.influenced_by -> {Carl Jung; Jan-Hein Arens; …; Ho Ho Ying}
   ```

3. **trie mask 比 block-diagonal 好，去重有帮助**：在全部 6 组对比中，M2 都优于 M1-hyb，大约补回一半损失（例如 Llama-2 WEB −9.8 → −3.2，RoG CWQ −6.8 → −2.2，后者不显著）。v1 中担心的"去重会去掉重复带来的 attention 权重、从而伤害准确率"并没有发生；恰恰相反，在独立编码的设定下，重复的前缀副本越多越伤。
4. **剩下的损失来自缺少跨路径的 attention**，而且集中在路径多的 query 上：所有条件在 ≤4 条路径时都不掉点，损失全部在 ≥5 条路径的桶里。模型需要在路径之间做整合（比如在一串同关系的尾实体里挑出答案），独立编码让上层表示里缺了这一步。
5. **位置方案没有统一的赢家**：RoG 上 M1-seq 最好（−3.6），Llama-2 上 M1-seq 最差（−19 / −26）；M2 的 par 和 hyb 相差都在 2 个点以内。因此不值得在位置方案上多花工夫，后续默认用 hyb，因为它和 GraphRAG-Opt 对齐，而且读取时的位置是紧凑的。
6. **多起点汇合**的子集只有 21 题（CWQ），各条件的差异在这个样本量下无法解读。

## 对研究路线的影响

- **T 改变了 baseline 的高度。** "要不要用 mask"的论证必须建立在 T 做不到的事情上：跨 query 精确复用（节点级冗余 61–76%，见实验 1）、prefill 中路径之间可以并行、检索与 prefill 的流水线。如果 mask 方案在准确率上补不回来，它就只剩系统层面的价值。
- **实验 3 的目标就此明确**：在保留大部分可复用比例的前提下，让 M2 追上 M0 和 T。候选手段是 top-ℓ 层全连接，以及 sibling 组。
- **一个可以直接写进论文的观察**：GraphRAG-Opt 的 reuse-all 相当于 M1-hyb；同样的独立编码，按 trie 去重（M2）后损失减半。这对 GraphRAG-Opt 本身也是一个低成本的改进。

## 产出

- 代码：`experiments/common/{masked_engine,conditions}.py`，`experiments/exp2/{run,test_engine,analyze,launch}.py|sh`
- 结果：`outputs/exp2/*.json`（逐题预测），`outputs/exp2/summary.md`
