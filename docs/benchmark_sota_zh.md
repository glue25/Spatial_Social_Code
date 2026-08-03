# Spatial Social Code：相关模型与 SOTA 地图

更新日期：2026-08-03

## 1. 结论先行

我们需要对齐的不是一个笼统的“社会理解 SOTA”，而是六个环境中的不同任务协议。当前最重要的模型路线可以归为五类：

1. **直接 LLM/VLM**：直接读文本、caption、帧或视频，如 Gemini、GPT、Qwen-VL；
2. **显式 mental model / inverse planning**：BIP-ALM、AutoToM；
3. **显式 event/state tracking**：SymbolicToM、Thought-tracing、PDDL-Mind；
4. **长视频 memory/retrieval**：EgoMAS、BM25、VideoRAG、EgoRAG；
5. **任务专用视频模型**：TimeSformer、VideoDistill、DMC³、DR-MoE、UE-MCM。

与本项目最直接的竞争关系是第 2、3、4 类。我们的路线应固定为：

`annotations / canonical facts → deterministic compiler → Spatial–Social Code → Qwen3-4B`

因此外部 VLM 的结果用于说明任务难度，但不等于我们的主模型也要换成 VLM。第一优先级仍是：

- **MMToM-QA**：验证显式 code 是否优于 direct text、普通 memory 和已有显式 mental-state 方法；
- **MA-EgoQA**：验证 agent-indexed code 是否优于 shared memory；
- **EnactToM**：在前两项有效后，验证 code 是否真正改善行动，而不只是报告信念。

HoloAssist、CoMind 和 EgoTaskQA 仍很相关，但当前公开 SOTA 主要依赖原始视频和视觉定位，不适合直接拿来判断纯 code reasoner 的好坏。

## 2. SOTA 记录规则

后续所有实验表都必须同时记录：

`环境 / split / 输入模态 / 可用真值 / 方法 / backbone / 指标 / 数值 / 发表状态 / 日期`

以下数字不能直接混比：

- text-only、video-only、multimodal；
- oracle context、retrieved context、raw input；
- frozen prompting、task-specific fine-tuning；
- QA accuracy、episode success、Pass@3、Pass³；
- 同一数据集上的不同子任务或不同 leaderboard protocol。

“当前最好”在本文中表示截至 2026-08-03 能从官方页面、正式论文或作者技术报告核对的公开结果。若官方 leaderboard 无投稿，则只称为“论文内最好基线”，不称为社区 SOTA。

## 3. 六个环境总表

| 环境与协议 | 当前可核对最好结果 | 方法类型 | 对本项目的意义 |
|---|---:|---|---|
| MMToM-QA multimodal | AutoToM + Model Specification / GPT-4o：**79.8** | 自动 mental model + Bayesian inverse planning | 强结构化外部基线 |
| MMToM-QA text-only 官方榜 | Muse-Spark-1.1：**88.7** | 直接闭源模型 | 最高公开榜分，但方法可解释性弱 |
| MMToM-QA text/state 独立协议 | PDDL-Mind / GPT-4o：**88.3**；Claude-Sonnet-4.5：**90.9** | verified PDDL state tracking | 与 Spatial–Social Code 最接近的主要方法竞争者 |
| MMToM-QA video-only | BIP-ALM / LLaMA 2：**61.2** | 视频识别 + inverse planning | 未来 predicted compiler 参照 |
| MA-EgoQA | EgoMAS / Gemini-2.5-Flash：**41.41** | shared 4W1H memory + dynamic retrieval | agent-indexed code 的直接对手 |
| EnactToM Standard functional | Gemini-Flash：Avg **42.5**、Pass@3 **70.0**、Pass³ **22.5** | frontier agent prompting | functional ToM 当前最强公开基线 |
| EnactToM Hard functional | DeepSeek-v3.2：Avg **8.3**、Pass@3 **22.5**；所有模型 Pass³ **0.0** | frontier agent prompting | 尚无成熟任务专用方法，空间很大 |
| HoloAssist mistake detection | UE-MCM：F-score **0.60** | CLIP4CLIP + Qwen3-VL embedding + gated fusion | 2026 challenge winner；视觉任务，不与 code-only 直接比较 |
| CoMind 三项任务 | 官方榜尚无投稿；论文内各指标由不同模型领先 | VLM prompting / Qwen3-VL fine-tuning | 适合未来社会感知与 compiler 评测 |
| EgoTaskQA normal / indirect | DMC³：**52.51 / 46.04** | 双模态反事实样本 + 对比学习 | 状态、因果、反事实理解的重要后续环境 |

## 4. MMToM-QA

### 4.1 官方 leaderboard 必须按模态拆开

| 协议 | 方法 | Belief | Goal | All |
|---|---|---:|---:|---:|
| Multimodal | Human | 97.5 | 88.5 | 93.0 |
| Multimodal | AutoToM + Model Specification / GPT-4o | 94.0 | 65.7 | **79.8** |
| Multimodal | BIP-ALM / LLaMA 2 | 80.3 | 73.3 | 76.7 |
| Multimodal | AutoToM / GPT-4o | 88.7 | 62.3 | 75.5 |
| Text-only | Muse-Spark-1.1 | 87.7 | 89.7 | **88.7** |
| Text-only | BIP-ALM / LLaMA 2 | 82.3 | 58.7 | 70.5 |
| Text-only | SymbolicToM / GPT-4 | 78.3 | 47.7 | 63.0 |
| Text-only | Qwen3.5-4B direct | 79.0 | 34.7 | 56.8 |
| Video-only | BIP-ALM / LLaMA 2 | 64.0 | 58.3 | **61.2** |

来源：[MMToM-QA 官方 leaderboard](https://chuanyangjin.com/mmtom-qa-leaderboard.html)。带星号的部分 prompting 结果只评了 200 个样本，不能与完整 600 题结果等价处理。

### 4.2 相关模型链

| 模型 | 核心思想 | 我们应该怎样比较 |
|---|---|---|
| Direct LLM/VLM | 直接从文字、视频或多模态上下文回答 | 用同一个 Qwen3-4B 做 direct-text matched baseline |
| SimToM | 先只保留目标主体可观察到的事件，再回答 | 对比我们的 agent-indexed view 是否更完整 |
| SymbolicToM | 为角色构建显式 belief graph | 对比 code 的物理状态、证据和可执行更新 |
| BIP-ALM | 用动作似然进行 Bayesian inverse planning，推断目标与信念 | 对比准确率、计算成本和可解释变量 |
| AutoToM | 自动提出 agent model，并根据不确定性迭代加入变量或时间步 | 主要已发表结构化基线；NeurIPS 2025 Spotlight |
| Thought-tracing | 生成并加权 mental-state hypothesis | 作为 inference-time mental tracking 基线 |
| PDDL-Mind | 把叙事解析成 PDDL actions，用预定义 transition 验证状态，再回答 | 与我们的 deterministic compiler + code 最接近 |

[AutoToM](https://arxiv.org/abs/2502.15676) 已正式发表于 NeurIPS 2025。 [PDDL-Mind](https://arxiv.org/abs/2604.17819) 目前是 2026 预印本；它报告 GPT-4o 88.3、Claude-Sonnet-4.5 90.9，但明确是 text-only、verified-state 路线，而且论文自行汇总的 AutoToM 83.0 与官方 multimodal leaderboard 的 79.8 不是同一个协议表。

### 4.3 我们的正确目标

MMToM-QA 第一张主表应包含：

1. Qwen3-4B direct text；
2. Qwen3-4B + matched free-form memory；
3. Qwen3-4B + oracle Spatial Code；
4. Qwen3-4B + oracle Social Code；
5. Qwen3-4B + full Spatial–Social Code；
6. PDDL-Mind、BIP-ALM、AutoToM 的公开结果，作为外部参考。

论文的第一因果结论应是同一 Qwen3-4B 下 `code > matched memory > direct text`，而不是依靠不同闭源 backbone 拼一个最高数字。

## 5. MA-EgoQA

### 5.1 当前公开结果

| 方法 | Context | SI | TC | ToM | TR | EI | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|
| All-caption Gemini-2.5-Flash | 1M | 41.2 | 36.4 | 24.3 | 46.6 | 34.0 | 36.9 |
| BM25 + Qwen3VL-8B-Instruct | 8.1k | 44.7 | 37.6 | 30.2 | 33.5 | 30.6 | 36.0 |
| EgoMAS + Qwen3VL-8B-Thinking | 5.4k | 38.0 | 39.9 | 28.9 | 47.4 | 44.9 | 40.3 |
| EgoMAS + Gemini-2.5-Flash | 4.6k | 41.5 | 41.3 | 33.6 | 39.4 | 48.2 | **41.4** |
| Oracle + Qwen3VL-8B | — | 76.9 | 81.0 | 57.0 | 77.4 | 69.9 | 74.0 |
| Oracle + Gemini-2.5-Flash | — | 86.4 | 97.6 | 70.2 | 88.2 | 81.3 | **83.8** |

来源：[MA-EgoQA 官方项目页与结果表](https://ma-egoqa.github.io/)。截至当前未发现晚于 2026 年 3 月、在同一 1,741 题协议上超过 EgoMAS 的公开方法。

### 5.2 相关模型链

- **All-caption / all-frame concat**：上下文很大，但容易被无关信息淹没；
- **BM25 / VideoRAG / EgoRAG / WorldMM**：先检索再回答，但通常缺乏跨主体的统一事件表示；
- **EgoMAS**：每 10 分钟将六个主体信息合并成 shared 4W1H event memory，再做 system-level 与 agent-level 两级检索；
- **Oracle context**：说明主要瓶颈仍是找对证据，而不是最终多选题解码。

我们的主要假设是：EgoMAS 的 shared memory 虽然有效，但会弱化“谁亲眼观察、谁只是被告知、谁仍未知”。因此需要在相同 context budget 下比较：

`shared 4W1H memory` vs `agent-indexed Spatial–Social Code`。

MA-EgoQA 全部为 test QA，不能用于训练或调参。

## 6. EnactToM

EnactToM 是 300 个具备私有信息、受限通信和 PDDL 验证的具身任务。它区分：

- **Literal ToM**：能否说出另一主体相信什么；
- **Functional ToM**：能否真的使用该信念选择消息和行动。

| 协议 | 当前最好公开结果 | 说明 |
|---|---:|---|
| Standard functional | Gemini-Flash：Avg 42.5、Pass@3 70.0、Pass³ 22.5 | 三个功能指标均领先 |
| Standard literal | O3：Avg 34.2、Pass@3 55.0、Pass³ 15.0 | 主要指标最好 |
| Hard functional | DeepSeek-v3.2：Avg 8.3、Pass@3 22.5；Pass³ 0.0 | 所有七个模型 Pass³ 都是 0 |
| Hard literal | Gemini-Pro：Avg 63.3、Pass@3 87.5、Pass³ 37.5 | 明显存在 act-report gap |

来源：[EnactToM 官方项目页](https://enact-tom.github.io/)。目前这些是 frontier model baselines，不是经过 EnactToM 专门训练的成熟 SOTA 方法。

对本项目最有价值的实验不是只报一次 episode success，而是：

1. message selection accuracy；
2. next-action accuracy；
3. functional Pass@3 / Pass³；
4. 删除关键 message 后，code 与行为是否局部一致变化；
5. literal 正确但 functional 失败的比例。

## 7. HoloAssist

HoloAssist 没有单一总分。原论文同时定义动作识别、mistake detection、intervention prediction 和 3D hand forecasting；2024–2026 的持续竞赛主要集中在 mistake detection。

### 7.1 原始 benchmark 最好公开基线

| 子任务 | 方法 | 指标与结果 |
|---|---|---|
| Fine-grained action recognition | pretrained multimodal TimeSformer，RGB+Hands+Eyes | action Top-1 **35.27** |
| Coarse-grained action recognition | pretrained TimeSformer，RGB | action Top-1 **50.91** |
| Mistake detection，原论文 | TimeSformer，Hands only | F-score **40.19** |
| Intervention type prediction | TimeSformer，RGB+Hands+Eyes+Head/IMU | overall precision **48.75**、recall **47.75** |
| 3D hand forecasting | Seq2Seq，RGB+Hands+Eyes+Head/IMU | MPJPE：0.5/1.0/1.5s = **9.72/10.62/11.19 cm** |

来源：[HoloAssist ICCV 2023 论文](https://arxiv.org/abs/2309.17024)。动作识别等旧赛道的 Codabench 页面没有稳定公开的完整后续榜单，因此不能把原始论文结果误称为 2026 社区 SOTA。

### 7.2 Mistake detection 方法演进

| 年份 | 方法 | 输入与机制 | F-score |
|---|---|---|---:|
| 2023 | TimeSformer baseline | RGB / hand pose 多模态分类 | 0.35–0.40 |
| 2024 | Gaze Completion | 预测 gaze trajectory，以预测/实测 gaze 差异检测异常 | 0.51 |
| 2025 | MistSense | ViT + spatial Q-Former + Video Q-Former；同时生成错误解释 | 0.55 |
| 2025 | DR-MoE | frozen + LoRA ViViT experts；reweighted CE、AUC、label-aware loss | 0.57 |
| 2026 | UE-MCM | DCR-CLIP4CLIP 看 coarse+fine context；Qwen3-VL embedding 看 fine action；门控融合 | **0.60** |

[Gaze Completion](https://arxiv.org/abs/2406.08379) 在 one-class HoloAssist 上还报告 F1 0.26、AUC 0.63；这与官方 challenge F-score 0.51 是不同训练/评测设置。[MistSense 技术报告](https://arxiv.org/abs/2506.06174) 给出了 2025 榜结果，[UE-MCM 技术报告](https://arxiv.org/abs/2606.02120) 给出了 2026 冠军结果。

这里的启发是长尾优化、coarse workflow context 和 fine execution cue 必须同时建模。但 UE-MCM 使用 Qwen3-VL embedding 并不改变我们的主线：它是 raw-video mistake detector，而我们的第一阶段是 oracle/semi-oracle code understanding。

## 8. CoMind

CoMind 官方 leaderboard 的三个任务目前都显示 **No submissions yet**，因此当前只能记录 ECCV 2026 论文内最好基线，而且每个指标由不同模型领先。

### 8.1 Joint Attention Estimation

| 指标 | 论文内最好模型 | 结果 |
|---|---|---:|
| Left IoU@0.5 | Gemini 3 Flash | 0.2164 |
| Right IoU@0.5 | Gemini 3 Flash | 0.2071 |
| Cue Type | Qwen3-VL-8B FT | 0.4775 |
| Category L1 | GPT-5.2 | 0.4285 |

### 8.2 Socially Conditioned Object Interaction Anticipation

| 指标 | 论文内最好模型 | 结果 |
|---|---|---:|
| IoU@0.5 | Qwen3-VL-32B FT | 0.2513 |
| Cue Type | Qwen3-VL-32B FT | 0.5503 |
| Action Verb | Qwen3-VL-32B FT | 0.6508 |
| Action Noun L1 | GPT-5.2 | 0.4371 |

### 8.3 Collaborative Handover Prediction

| 指标 | 论文内最好 learned model | 结果 |
|---|---|---:|
| IoU@0.5 | Gemini 3 Flash | 0.0313 |
| Delivery Flow | Gemini 3 Flash | 0.6524 |
| Initiator | Qwen3-VL-32B FT | 0.7286 |
| Initiation Type | Qwen3-VL-32B FT | 0.3714 |
| Category L1 | GPT-5.1 | 0.4000 |
| TTH within 0.25s | Claude Opus 4.6 | 0.1317 |

来源：[CoMind ECCV 2026 论文](https://arxiv.org/abs/2607.06691) 与[官方 leaderboard](https://comind.ethz.ch/)。Initiator 和 Initiation Type 的 most-frequent baseline 分别为 0.7857 和 0.4667，高于 learned model，说明类别不平衡严重；论文内高亮的“最好模型”不应掩盖这个事实。

CoMind 的 transcript 和社会标注适合先做一个 text/oracle protocol，但它不能与官方 raw multimodal task 混报。要参加正式榜，未来仍需专用感知 compiler 输出 bbox、cue、agent 和时序事件。

## 9. EgoTaskQA

| 方法 | Normal all | Indirect all | 关键机制 |
|---|---:|---:|---|
| HCRN | 42.20 | 41.56 | hierarchical conditional relation network；原始强基线 |
| EgoVLP | 42.51 | 38.69 | 第一视角 video-language pretraining |
| EgoVLPv2 | 46.26 | 42.28 | 更强 egocentric video-language pretraining |
| VideoDistill | 45.02 | 44.77 | question-aware sparse frames + vision refinement，减少 language shortcut |
| MFAS | 48.69 | 45.40 | multi-factor adaptive visual selection |
| DMC³ | **52.51** | **46.04** | text event paraphrase + visual interaction mining + counterfactual contrastive learning |

来源：[EgoTaskQA 原论文](https://arxiv.org/abs/2210.03929)、[VideoDistill](https://arxiv.org/abs/2404.00973) 与 [DMC³](https://arxiv.org/abs/2510.20285)。DMC³ 同时报告 normal predictive 53.97、counterfactual 52.05、multi-agent 42.05；这些维度与我们的 physical-state、causal 和 counterfactual code 很接近。

EgoTaskQA 是很好的第三个 QA 环境，但当前本地只有仓库和处理代码，数据仍需单独许可。获得数据后，应优先对比 state、change、predictive、counterfactual 和 multi-agent 子类，而不仅看 overall。

## 10. 对我们实验路线的具体调整

### 第一阶段：MMToM-QA

- 主模型固定 Qwen3-4B；
- 先做 direct / memory / code 的 matched comparison；
- 将 PDDL-Mind 作为最重要的结构化方法竞争者；
- 报 belief 与 goal 分项，避免 overall 掩盖 goal failure；
- 外部表严格区分官方 multimodal、官方 text-only 和 PDDL text/state protocol。

### 第二阶段：MA-EgoQA

- 不训练 test QA；
- 复现 EgoMAS 的 shared-memory 输入预算；
- 比较 shared code 与 agent-indexed code；
- 重点看 ToM、task coordination 和跨主体 temporal reasoning；
- oracle gap 用于诊断 retrieval/compiler，而不是当作可达到的正常 baseline。

### 第三阶段：EnactToM

- 先训练 next-action 与 message-selection；
- 再测完整 functional Pass@3 / Pass³；
- hard split 的第一目标不是立即超过某个稳定 SOTA，而是从 0 Pass³ 建立可靠非零结果；
- 强制报告 literal/functional gap。

### 暂不作为首篇主表

- **HoloAssist**：需要大量 raw video，且 SOTA 主要是视觉长尾 mistake detection；
- **CoMind**：正式任务同时要求 bbox 和社会 cue，当前 pilot transcript 不足以公平参榜；
- **EgoTaskQA**：方法高度相关，但本地数据许可尚未完成。

## 11. 需要复现和需要引用的最小模型集合

首篇论文不必实现所有模型，但至少应维护下面两层：

**必须本地运行：**

1. Qwen3-4B direct text；
2. Qwen3-4B matched free-form memory；
3. Qwen3-4B shared Spatial–Social Code；
4. Qwen3-4B agent-indexed Spatial–Social Code；
5. oracle compiler 与各类 code ablation。

**必须作为外部结果引用：**

1. MMToM-QA：BIP-ALM、AutoToM、PDDL-Mind、官方 direct model leaderboard；
2. MA-EgoQA：EgoMAS、BM25+Qwen3VL、oracle；
3. EnactToM：七个 frontier agent baselines；
4. 后续环境：UE-MCM、CoMind FT Qwen3-VL、DMC³。

这套组织能把问题说清楚：我们不是再训练一个“看视频更强的 VLM”，而是在检验小型语言模型能否通过显式、可验证、带主体索引的 Spatial–Social Code，获得已有 memory、inverse-planning 和 symbolic-state 方法没有同时具备的稳定性与可迁移性。
