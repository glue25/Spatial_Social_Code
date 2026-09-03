# Spatial-Social Code：投稿结果与 baseline 口径

更新日期：2026-09-03。

## 结论

当前最稳妥的论文主张是：

> 在同模型、同视觉证据的 paired protocol 下，task/question-conditioned、
> agent-indexed Spatial-Social Code 在 MMToM 和 MA-EgoQA 上提高回答准确率，
> 在 EnactToM/Habitat 的冻结子集上提高平均目标完成度，并在 CoMind
> recording-heldout OOF 上显著改善真实多视角定位；真正有效的组件随 benchmark
> 而变化。

这里的 compiler 可以读取任务、问题和选项，但不能读取 gold label、正确性
反馈或 baseline answer。视觉主实验匹配模型和视觉输入，但不匹配调用数、token、
延迟或 FLOPs。

## 1. 同协议主结果

只有实际保存了逐题或逐任务配对结果的实验进入本表。

| 环境 | Direct | Spatial-Social | 差值 | 统计与边界 |
|---|---:|---:|---:|---|
| MMToM，600 题 | 406/600，67.67% | **470/600，78.33%** | **+10.67 pp** | McNemar `p=3.17e-05`；同模型/同帧，不同计算量 |
| MA-EgoQA，1,741 题 | 848/1741，48.71% | **904/1741，51.92%** | **+3.22 pp** | McNemar `p=0.00388`；同模型/同 clips，不同计算量 |
| Habitat frozen-12，36 run/条件 | success 2.78%，完成度 45.83% | success 11.11%，完成度 **53.47%** | 完成度 **+7.64 pp** | task-cluster bootstrap 95% CI `[+0.69,+15.74]`；仅 12 个任务 |

CoMind 没有可合并成单一准确率的官方总指标，因此作为同一主表的独立 grounding
面板报告。协议是 55 个 labeled train recording 上的五折 recording-heldout
OOF；每个 held-out recording 的标签不参与该 fold 的策略拟合或选择。

| CoMind 字段（6,673 条严格共同集） | Direct | 原 Spatial-Social | Grounding + OOF |
|---|---:|---:|---:|
| Joint left bbox IoU@0.5 | 0.17% | 0.19% | **20.33%** |
| Joint right bbox IoU@0.5 | 0.11% | 0.19% | **21.27%** |
| SCOIA bbox IoU@0.5 | 0.21% | 0.10% | **6.47%** |
| Handover bbox IoU@0.5 | 0.34% | 0.00% | **23.15%** |
| Handover category L1 | 22.82% | 24.16% | **31.88%** |

CoMind 支持的是“显式 detector/SAM grounding 修复文本式 Spatial bbox”的结论；
它不是官方 hidden-test 或 leaderboard 成绩，也不证明所有社会字段都提升。

MMToM 的 equal-call direct diagnostic 使用约 1,550 个逻辑调用，得到
372/600（62.00%），仍低于完整方法 470/600。它缓解了“只是因为调用次数更多”
这一解释，但不等于 token、延迟或 FLOP 完全匹配，而且是在已经迭代过的 benchmark
上运行，应作为诊断而非独立 held-out 证据。

## 2. 组件归因

| 环境 | 当前能支持的组件结论 | 当前不能支持的结论 |
|---|---|---|
| MMToM | `rules -> rules + Spatial bridge` 从 69.67% 到 75.33%，`p=1.16e-6`；dense/refined 完整路线达到 78.33% | sparse Social 从 75.33% 降到 74.17%，不能说任意 Social Code 都有增益 |
| MA-EgoQA | 冻结100题诊断中 Raw 44.44%、Spatial-only **51.00%**、Social-only 48.00%、P+S 50.00%；Spatial 有独立信号 | P+S 没超过 Spatial-only；自然语言 scratchpad 也是 51.00%，不能把完整增益只归因于代码格式 |
| Habitat | P+S 完成度比 Direct 高 7.64 pp；Social-only 达到最高的 59.95% | P+S 低于 Social-only 6.48 pp，不能声称 Habitat 上 P/S 互补或 Spatial 独立有效 |
| CoMind | recording-heldout OOF grounding 在三项 bbox IoU@0.5 上由近零提高到 6.47%--23.15%，handover category 达 31.88% | 这是 labeled train OOF，不是 hidden test；原始300题 paired SSC 也没有整体领先 |

另外两项已完成的 Spatial 诊断为：MMToM 冻结70题上 VLM-only 74.29%，
VLM+SAM 75.71%（+1.43 pp，6个正翻转、5个负翻转，McNemar `p=1.0`）；
Habitat Spatial v2 在同 seed 的12题上把 Spatial-only 从38.19%修复到44.44%，
但 P+S 为54.86%，仍低于 Social-only 63.89%。两者说明 Spatial 感知/约束有信号，
但当前融合仍不是稳定增益来源。

因此论文不应写“每个环境都由 Spatial 和 Social 同时贡献”。更准确的发现是：
结构化、主体化状态在不同任务上有效，但瓶颈不同；MMToM 有清晰 Spatial bridge
贡献，Habitat 的主要增益来自 Social 协调信息。

## 3. 官方公开结果：外部参考表

下面的数字可以直接引用进论文 baseline 表，但必须保留协议、backbone 和
`严格可比：否`。它们不能替代上面的 paired Direct，也不能用来计算我们的
方法增益。

| 环境/协议 | 官方或论文公开方法 | 公开结果 | 严格可比 | 主要差异 |
|---|---|---:|:---:|---|
| MMToM multimodal | AutoToM / GPT-4o | 75.5% | 否 | 我们 78.33%，高 2.83 pp；不同模型与实现 |
| MMToM multimodal | AutoToM + Model Specification / GPT-4o | 79.8% | 否 | 人工预先指定变量结构；不同模型与实现 |
| MMToM multimodal | BIP-ALM / LLaMA 2 | 76.7% | 否 | 不同模型与方法实现 |
| MMToM video-only | BIP-ALM / LLaMA 2 | 61.2% | 否 | 输入协议不同 |
| MMToM text-only 官方榜 | Muse-Spark-1.1 | 88.7% | 否 | 输入模态与模型不同 |
| MMToM verified text/state | PDDL-Mind / GPT-4o | 88.3% | 否 | 独立 verified-state 协议 |
| MMToM verified text/state | PDDL-Mind / Claude-Sonnet-4.5 | 90.9% | 否 | 独立 verified-state 协议 |
| MA-EgoQA | All-caption / Gemini-2.5-Flash | 36.9% | 否 | 约 1M context、不同 backbone |
| MA-EgoQA | BM25 / Qwen3-VL-8B-Instruct | 36.0% | 否 | 检索与 backbone 不同 |
| MA-EgoQA | EgoMAS / Gemini-2.5-Flash | 41.41% | 否 | shared-memory 官方协议，backbone/上下文不同 |
| MA-EgoQA | EgoMAS / Qwen3-VL-8B-Thinking | 40.26% | 否 | backbone/上下文不同 |
| MA-EgoQA oracle | Gemini-2.5-Flash | 83.8% | 否 | oracle context ceiling |
| EnactToM Standard functional | Gemini-Flash | Avg 42.5，Pass@3 70.0，Pass³ 22.5 | 否 | 官方全协议指标，不是 frozen-12 Habitat success |
| EnactToM Hard functional | DeepSeek-v3.2 | Avg 8.3，Pass@3 22.5，Pass³ 0.0 | 否 | 不同 split 与官方功能指标 |

CoMind 官方 leaderboard 目前没有公开 submission；论文内不同字段由不同模型
领先，不能拼成一个统一 SOTA 总分。完整字段、来源链接与模态拆分见
[`docs/benchmark_sota_zh.md`](../docs/benchmark_sota_zh.md)。

### 官方代码什么时候能算“我们的 baseline”

- 只引用作者论文/榜单数字：放外部参考表，注明协议差异即可。
- 用官方代码在官方 split 上复现，但模型或输入不同：可写 reproduced reference，
  仍不能计算因果增益。
- 用官方代码在与我们完全相同的样本、视觉输入和 backbone 上运行：可进入主表；
  还需报告调用数、token、延迟和失败率。

所以“官方代码能运行”并不自动等于公平，但确实能低成本扩充 related baseline。

AutoToM 的 `Model Specification` 条件通过 `assigned_model` 人工给定要使用的
状态、观察、信念、动作、目标等变量类型；它不读取正确答案，但减少了自动发现
推理结构的难度。因此自动化能力的主比较应使用标准 AutoToM 75.5%，而 79.8%
应标作“人工结构先验”的强参考。我们的 schema 也是一次性人工设计的，但每道题
的具体实体、事件和主体状态由系统从视频与问题自动编译，没有人工逐题指定模型。

## 4. 投稿前 baseline 优先级

优先补能回答审稿人因果问题的本地对照，而不是继续堆不同模型的公开数字：

1. MA-EgoQA 的100题强自然语言 baseline 和组件矩阵已经完成；下一步应在独立开发
   split 冻结融合规则，而不是继续用 test QA 调参。
2. MMToM 的 agent-indexed vs shared Social，以及 option decomposition without
   code。equal-call direct 已完成。
3. Habitat 扩大 task cluster 数；继续报告 success 和完成度，但不以 seed 数冒充
   独立任务数。
4. CoMind 若无法提交 hidden test，就把 OOF grounding 作为真实多视角机制主结果，
   但不声称 leaderboard/SOTA。

StoryVideoQA 与 SocialNav-SUB pilot 当前没有增益，不建议为首篇论文继续扩大主表。

## 5. Artifact 状态

公开仓库提供：

- 聚合机器可读结果：
  [`submission_results_summary.json`](submission_results_summary.json)；
- MMToM 完整组件聚合与统计；
- CoMind 300 题探索性结果和 recording-heldout OOF 聚合；
- Habitat frozen-12 三 seed 聚合；
- 无外部数据/API key 的安装、单元测试和结果一致性验证入口：
  `bash scripts/verify_artifact.sh`。

未随仓库分发 raw video、数据集、模型权重和大体积 prediction JSONL。MA-EgoQA
公开 artifact 因而只能核验聚合算术，不能由第三方离线重算逐题 McNemar；正式
匿名 artifact 若许可允许，应补充去除敏感字段后的逐题 `gold/direct/method` 三列。

## 6. 可写与不可写

可以写：

- “same-model, same-evidence paired improvement”；
- MMToM 的 Spatial bridge 有显著独立贡献；
- Habitat P+S 提高平均完成度，Social-only 是该环境最强组件；
- runtime、unknown semantics、provenance 和 agent-indexed state 是可审计设计。

暂时不要写：

- query-independent visual world representation；
- compute-matched，除非补齐 token/延迟/FLOP 对照；
- Spatial 与 Social 在所有环境都互补；
- CoMind hidden-test/SOTA（当前是 train recording-heldout OOF）；
- Habitat 超过官方 EnactToM SOTA，因为指标和任务协议不同。
