# MMToM、MA-EgoQA 与 CoMind：当前结果、案例与方法迭代

> **历史报告，已由 2026-09-14 审计取代。** 本文中的 MMToM 470/600 使用答案定向
> `query_evidence`；MA-EgoQA 904/1741 的 compiler 读取问题与选项。两者均不得作为
> 当前 clean SSC 主结果。当前口径见
> [投稿结果与 baseline 口径](submission_results_and_baselines_zh.md) 和
> [最新证据包](audit_latest_experiments/README_zh.md)。以下内容仅保留用于追踪历史迭代。

更新日期：2026-08-19

## 一句话结论

在同一模型、同一视觉输入的 paired evaluation 下，Spatial-Social Code 在两个完整测试集上都显著优于 direct baseline：

| Benchmark | Spatial-Social Code | Direct baseline | 绝对提升 | McNemar exact p |
|---|---:|---:|---:|---:|
| MMToM（600 题） | **470/600，78.33%** | 406/600，67.67% | **+10.67 pp** | 3.17e-05 |
| MA-EgoQA（1,741 题） | **904/1741，51.92%** | 848/1741，48.71% | **+3.22 pp** | 0.00388 |

这两个结果的意义不同：MMToM 的主要收益来自对行动轨迹进行逆向目标/信念推断；MA-EgoQA 的收益主要来自主体归属、Theory of Mind 和跨事件时序。当前结果是项目内的正式保存结果，但不等同于已经提交或确认的官方 leaderboard 成绩。

CoMind 仍在运行，不能与上面两个完整结果混为一谈。截至 2026-08-19，train split 前 14/55 个录像已完成 1,632 条 paired inference；在双方都成功的 1,577 条样本上，Spatial-Social 将 Joint Attention Cue 从 14.90% 提高到 21.44%（+6.55 pp），但 SCOIA 和 Handover 多数指标未提高，且所有 bbox IoU@0.5 仍为 0。CoMind 的完整阶段性结果见第 4 节。

主要结果文件：

- [MMToM final summary](../spatial_social_code/mmtom_dense_authoritative_gemini3pro_summary.json)
- [MMToM full predictions](../spatial_social_code/mmtom_dense_authoritative_gemini3pro.jsonl)
- [MA-EgoQA final summary](../MA-EgoQA/data/baselines/video_evidence_audit_full_summary.json)
- [MA-EgoQA full predictions](../MA-EgoQA/data/baselines/video_evidence_audit_full.jsonl)
- [CoMind 14-recording paired summary](../workdirs/comind_ready_evaluation/comparison_paired_partial.json)
- [CoMind 14-recording merged predictions](../workdirs/comind_ready_evaluation/predictions.jsonl)

## 1. 我们实际在做什么

核心问题不是让模型“多想几步”，而是把视觉证据、物理事实和心理推断分开：

```text
相同视觉输入
  -> option-aware but answer-agnostic evidence extraction
  -> Spatial Code：人物、物体、动作、时间、观察者、视频片段与证据区间
  -> restricted runtime：语法与证据引用校验，执行成 canonical trace
  -> Social Code：观察、信念、目标、意图、矛盾和 unknown
  -> 对两个或多个候选答案逐项检查
  -> MMToM：native posterior；MA-EgoQA：raw video + traces 的 final VLM answer
```

实验遵循以下约束：

1. 方法与 direct baseline 使用同一个模型和相同视觉输入。
2. baseline 只用于最终 paired comparison，不参与选择、切换、融合或覆盖方法答案。
3. `world fact` 不自动等于某个主体的 `belief`。
4. 没看到证据表示 `unknown`，不能直接当作反证。
5. “执行了动作”“观察到动作”不自动推出发起者、责任、意图或因果。
6. MA-EgoQA 的候选相关检索对每个选项对称进行；最终证据必须绑定 clip、时间段和 evidence ID。
7. MMToM 对方法自身不确定的样本使用独立、更密集的视觉 pass；它仍然不读取 direct baseline 的答案。

### 1.1 代码结构

| 层 | 主要文件 | 实际职责 |
|---|---|---|
| Spatial/Social runtime | [`runtime.py`](../spatial_social_code/runtime.py) | 定义 `S.fact/event/visible/unknown` 和 `M.observe/belief/goal/intent/...`；通过 AST 白名单执行模型生成的 literal calls |
| Social compiler 与推断 | [`social_code.py`](../spatial_social_code/social_code.py) | 融合 neural extraction、规则、Spatial bridge、候选假设和 native posterior；处理时间传播与显式矛盾 |
| MMToM 主评测 | [`evaluate_mmtom_jdcloud.py`](../spatial_social_code/evaluate_mmtom_jdcloud.py) | 相同 RGB frames 上运行 direct 与 Spatial-Social；方法答案直接来自 Social trace 中 A/B posterior |
| MMToM 不确定性 pass | [`refine_mmtom_uncertain_jdcloud.py`](../spatial_social_code/refine_mmtom_uncertain_jdcloud.py) | 当 `max(P(A),P(B)) < 0.70` 时增加 16-frame dense pass |
| MMToM final policy | [`rescore_mmtom_dense_authoritative.py`](../spatial_social_code/rescore_mmtom_dense_authoritative.py) | 对触发不确定性的样本采用 dense pass posterior，而不是读取 baseline 或 gold |
| MA-EgoQA paired retrieval | [`run_video_balanced_baseline_jdcloud.py`](../MA-EgoQA/egomas/src/run_video_balanced_baseline_jdcloud.py) | 对每个答案选项建立独立 BM25 clip pool，round-robin 保证每个选项都有检索机会；保存 baseline 与方法共用的 clips |
| MA-EgoQA provenance method | [`run_video_provenance_v3_jdcloud.py`](../MA-EgoQA/egomas/src/run_video_provenance_v3_jdcloud.py) | 从同一 raw clips 生成带 clip/time/observer/evidence ID 的 Spatial trace，再生成必须引用 Spatial ID 的 Social trace，最后结合 raw video 回答 |
| MA-EgoQA evidence audit | [`audit_video_evidence_sparse_jdcloud.py`](../MA-EgoQA/egomas/src/audit_video_evidence_sparse_jdcloud.py) | 仅对方法答案没有有效 citation 的样本做独立 raw-video audit；不向 audit 暴露原答案或 baseline |

### 1.2 两个 benchmark 的真实执行链

MMToM：

```text
8 个稀疏 RGB frames + 初始状态/行动文字 + A/B 选项
  ├─ direct VLM -> direct_prediction
  └─ Spatial VLM -> safe Spatial trace
       + neural Social evidence
       + deterministic action rules
       + Spatial-to-observation bridge
       + A/B query evidence
       -> native log-odds/softmax posterior -> method prediction
       -> 若 posterior confidence < 0.70：16-frame dense pass
       -> final 对不确定项采用 dense posterior
```

最终 600 题中有 175 题触发 dense pass，31 题改变答案，其中 28 题由错变对、3 题由对变错。

MA-EgoQA：

```text
caption BM25 只用于定位 clips
  -> 每个选项 round-robin + planner pools -> 最多 10 个相同 raw clips
  ├─ direct VLM -> baseline answer
  └─ Spatial VLM -> provenance-validated Spatial trace
       -> Social VLM -> citation-validated Social trace
       -> final VLM 同时查看 raw clips + traces -> method answer
       -> 若 final answer 没有有效 Spatial citation：独立 raw-video audit
```

最终 1,741 题中有 266 题进入 evidence audit，只有 6 个高置信且至少含两条有效视频证据的 flip 被接受：3 个有益、2 个有害、1 个新旧答案都错，净提升 1 题（903 -> 904）。

### 1.3 代码层面可以成立的创新点

以下更适合称为本项目的技术贡献。是否属于文献意义上的“首次”仍需经过完整 related-work 检索，当前不要直接写成 first-ever。

1. **可执行的双层中间表示。** Spatial Code 保存世界状态、动作、可见性与来源；Social Code 单独保存主体化观察、信念、目标和意图。生成结果不是自由文本，而是经过受限 AST runtime 执行成 canonical JSON trace，因此可以检查、过滤和重放。
2. **跨层 provenance 约束。** 在 MA-EgoQA 中，每条 grounded Spatial record 必须具有唯一 evidence ID、合法 clip ID、与 clip wearer 一致的 observer，以及局部时间段；非 `unknown` Social record 必须引用真实存在的 Spatial ID。这比普通 CoT 的“口头引用视频”更可验证。
3. **显式区分 world state、observation 与 belief。** 代码不允许把静态世界事实直接复制成主体信念；容器内容只有在主体完成 open/check/search 后才能桥接成 observation。`unknown` 保持中性，不自动变成反证。
4. **同输入的 answer-symmetric paired protocol。** MA-EgoQA 对每个选项建立独立检索池并轮转选 clip，避免某个看似合理答案独占上下文；baseline 和方法随后使用完全相同的 raw clips。它既是检索设计，也是减少实验混杂的评测设计。
5. **可解释的混合 Social inference。** MMToM 把 neural evidence、确定性动作规则、物理到社会状态的 bridge 和答案假设放入同一 trace，再用显式 likelihood/log-odds 生成归一化 posterior。每个答案概率都能追溯到 query evidence，而不是只保留最终字母。
6. **不读取 baseline 的 method-only uncertainty control。** MMToM 用方法自身 posterior 触发 dense pass；MA-EgoQA 用 citation 缺失触发 audit。选择函数不使用 baseline prediction 或 gold label，避免把 baseline 当 oracle 来路由方法答案。
7. **失败消融可审计。** 更硬的 Executable Hypothesis Layer 没有被选择性包装成主结果，而是保留完整失败结果。这不是模型结构创新，但增强了方法结论的可证伪性。

### 1.4 不能过度声称的部分

代码核对后，以下表述目前不够准确：

- **不是严格 choice-blind。** MMToM 的 Spatial prompt 和 neural Social prompt都能看到完整 A/B 选项；MA-EgoQA v4.1 的 Spatial、Social 和 final prompts 也能看到完整选项。代码保证的是“不把答案字母直接编码进 evidence calls”，而不是从证据抽取阶段完全隐藏候选。
- **不是 compute-matched。** 两边使用相同 backbone 和相同视觉输入，但 direct baseline 通常只有一次回答调用；方法包含 Spatial、Social、final answer，并可能增加 dense/audit pass。因此可靠主张是 same-model/same-evidence improvement，不能声称相同 token、延迟或计算预算。
- **final MA 方法不是纯符号 reasoner。** 最终答案仍由能看到 raw videos 的 VLM 生成，trace 是受约束的辅助信息，并且 prompt 明确允许 raw video 覆盖 trace。
- **native Bayesian 是可解释启发式，不是完整生成式规划模型。** 其 likelihood 由正则解析、实体匹配和手工标定值构成，当前尤其针对 MMToM 的容器搜索模板。
- **claim planner/coverage repair 没有接入最终 MA v4.1 主链。** 文件中虽然定义了 `CLAIM_PLANNER_PROMPT` 和 `SPATIAL_REPAIR_PROMPT`，但 `process_one` 将 claim plan 置空且不调用 repair；对应 summary 的 claim coverage 为 `null`。不能把 full claim coverage 当作当前 51.92% 的组成部分。
- **不能直接声称 causal reasoning。** 当前主要是限制无证据的责任、意图和因果归因，并没有实现可干预的因果图或反事实推理器。
- **测试集迭代风险需要披露。** MA-EgoQA 全部是 test split，而仓库保存了多轮 pilot/full 迭代。如果这些结果用于选择版本，51.92% 更适合作为 development-on-benchmark 结果；正式论文应使用预注册规则、额外 held-out set 或新 benchmark 复验。

## 2. MMToM

### 2.1 最终结果

MMToM 使用 Gemini 3-Pro-Preview，在相同 RGB frames 上进行完整 600 题 paired evaluation。方法净赢 64 题：方法独赢 148 题，baseline 独赢 84 题。

| Question type | 样本数 | Direct | Spatial-Social | 差值 |
|---|---:|---:|---:|---:|
| 1.1 | 100 | 95.00% | 80.00% | -15.00 pp |
| 1.2 | 100 | 93.00% | 66.00% | -27.00 pp |
| 1.3 | 100 | 85.00% | **97.00%** | +12.00 pp |
| 2.1 | 75 | 18.67% | **94.67%** | +76.00 pp |
| 2.2 | 75 | 92.00% | 81.33% | -10.67 pp |
| 2.3 | 75 | 4.00% | **70.67%** | +66.67 pp |
| 2.4 | 75 | 62.67% | 56.00% | -6.67 pp |

因此，78.33% 不能被概括为“每一类题都更好”。收益高度集中在 1.3、2.1 和 2.3，特别是 direct model 容易被表面物体位置或答案模板干扰的逆向推断题；1.1、1.2、2.2 和 2.4 仍有明显退化，需要继续审计。

公开结果只适合作为背景参考：项目文档记录的 multimodal AutoToM + Model Specification / GPT-4o 为 79.8%，BIP-ALM / LLaMA 2 为 76.7%。我们的 78.33% 位于二者之间，但模型、实现和评测入口不同，不能把这个横向数字当成严格的因果比较。

#### 2.1.1 MMToM 组件消融（2026-08-19 新增）

我们在完整 600 题上对冻结的 Sparse Spatial trace 做了离线重放。`query/rules/spatial` 三个条件不调用 neural Social extractor；后续条件逐步加入保存的 Social trace 与 dense refinement。整个消融不调用 API，也不读取 gold 或 baseline 来生成方法答案。

| 条件 | 正确数 | 准确率 | 相对前一条件 |
|---|---:|---:|---:|
| Direct VLM | 406/600 | 67.67% | — |
| Query candidates only | 410/600 | 68.33% | — |
| Query + deterministic rules | 418/600 | 69.67% | +1.33 pp |
| Query + rules + Spatial bridge，**无 neural Social** | **452/600** | **75.33%** | +5.67 pp |
| Full hybrid，sparse Social | 445/600 | 74.17% | -1.17 pp |
| Full hybrid，average dense refinement | 467/600 | 77.83% | +3.67 pp |
| Full hybrid，dense authoritative | **470/600** | **78.33%** | +0.50 pp |

配对检验给出更明确的组件结论：

- `rules -> rules + Spatial`：8 题由对变错、42 题由错变对，净 +34，McNemar exact `p=1.16e-6`。Spatial bridge 是稳定且显著的贡献。
- `Spatial-only -> sparse Social hybrid`：7 题由对变错、0 题由错变对，净 -7，`p=0.0156`。因此不能声称“加入 Social Code 本身必然增益”。
- `sparse -> average dense hybrid`：1 题由对变错、23 题由错变对，净 +22，`p=2.98e-6`。Social 路线的实际价值依赖 evidence coverage 与角色/时序质量。
- `average dense -> dense authoritative`：净 +3，`p=0.453`，单独看这一小步不显著。
- `Direct -> dense authoritative`：84 题只被 direct 答对、148 题只被完整方法答对，净 +64，`p=3.17e-5`。

这组结果修正了一个过强叙述：最终方法确实使用了 Social Code，但 10.67 pp 的总提升不能全部归因于 Social 模块。冻结消融中，Spatial + rules 已贡献到 75.33%；Sparse Social 会伤害结果，只有提高 Social evidence 的覆盖和可靠性后，完整路线才从 74.17% 上升到 78.33%。

**可直接引用的结论：最终 78.33% 路线确实启用了 Spatial Code、Spatial-to-Social bridge 和 Social Code。Spatial-to-Social bridge 的独立贡献最清楚：69.67% -> 75.33%，提升 5.67 pp，净多答对 34 题，`p=1.16e-6`。Social Code 的作用依赖证据质量：sparse Social 会退化到 74.17%，dense/refined Social 路线最终达到 78.33%。**

可复现文件：

- [消融脚本](../spatial_social_code/evaluate_mmtom_component_ablation_offline.py)
- [逐题预测](../spatial_social_code/mmtom_component_ablation_offline.jsonl)
- [汇总与配对检验](../spatial_social_code/mmtom_component_ablation_offline_summary.json)

### 2.2 成功案例：世界事实不等于主体信念

来源：[MMToM prediction index 0](../spatial_social_code/mmtom_dense_authoritative_gemini3pro.jsonl)，question type 1.3。

视觉输入：[模型实际看到的 8 帧 contact sheet](cases/mmtom_index_0_model_frames.png)；原始 PNG 位于 `MMToM-QA-data/required_frames/task_6/script/0/`。

题目中的相关状态与行动：冰箱和第二个橱柜里都有盘子，第三个橱柜里有酒杯；Jennifer 从卫生间进入厨房并走向酒杯。

> If Jennifer has been trying to get a plate, which statement is more likely?
>
> A. Jennifer thinks that the plate is inside the fridge.  
> B. Jennifer thinks that the plate is not inside the fridge.

| 项目 | 回答 |
|---|---|
| Gold | **B：不在冰箱里** |
| Direct baseline | A：在冰箱里（错误） |
| Spatial-Social | **B：不在冰箱里（正确）** |
| 方法 posterior | P(A)=0.183，P(B)=0.817 |

方法保留了两层不同信息：

```python
S.fact(0, "in", "plate", "fridge", ...)
S.fact(0, "in", "plate", "kitchen_cabinet_2", ...)
S.event(1, "Jennifer", "proceed_to", "kitchen", ...)
S.event(2, "Jennifer", "advance_towards", "wine_glass", ...)

M.goal("Jennifer", "get a plate", ...)
M.belief(3, "Jennifer", "the plate is not inside the fridge", ...)
```

Direct answer 容易被“盘子确实在冰箱”这个真实状态吸引；Spatial-Social 路线则用 Jennifer 的搜索行为反推她的主观信念。这正是我们想解决的 `world state / agent belief` 混淆。

### 2.3 失败案例：把“准备打开”误当成“已经检查”

来源：同一结果文件的 prediction index 42，question type 1.1。

视觉输入：[模型实际看到的 8 帧 contact sheet](cases/mmtom_index_42_model_frames.png)；原始 PNG 位于 `MMToM-QA-data/required_frames/task_143/script/0/`。

题目中的相关状态与行动：第一个厨房橱柜里有调味瓶；David 依次检查第四、第三个橱柜、炉灶和微波炉，最后正准备打开第一个橱柜。

> If David has been trying to get a condiment bottle, which statement is more likely?
>
> A. David thinks that the condiment bottle is not inside the 1st kitchen cabinet.  
> B. David thinks that the condiment bottle is inside the 1st kitchen cabinet.

| 项目 | 回答 |
|---|---|
| Gold | **B：在第一个橱柜里** |
| Direct baseline | **B：在第一个橱柜里（正确）** |
| Spatial-Social | A：不在第一个橱柜里（错误） |
| 方法 posterior | P(A)=0.999，P(B)=0.001 |

错误 trace 把 `preparing to open the first kitchen cabinet` 同时解释成了 `completed search` 和 `opened-container observation`，进而产生“已经检查且不存在”的强反证。这是典型的事件阶段错误：

```text
approach / preparing_to_open != open != inspect != observe_absence
```

这个案例说明 runtime 只保证代码合法，并不保证语义判断正确。下一步必须为动作阶段增加更严格的 precondition，尤其禁止从“靠近/准备打开”推出“观察到容器内部”。

### 2.4 MMToM 的迭代过程

| 版本 | 方法 | 当轮 direct | 观察 |
|---|---:|---:|---|
| Contrastive full | 72.00% | 63.67% | 对两个选项显式比较后首次形成稳定正收益 |
| Robust provenance | 74.17% | 67.67% | 更严格的物理证据和引用 |
| Uncertainty refinement | 77.50% | 67.67% | 对方法自身不确定项增加独立 dense pass |
| Role rerun | 77.83% | 67.67% | 修正部分 actor/role 判断 |
| Dense authoritative，正式采用 | **78.33%** | 67.67% | 方法独赢 148、baseline 独赢 84 |
| Executable Hypothesis ablation | 53.00% | 67.67% | 过度确定性的离线规则层严重损失信息，未采用 |

早期版本的 direct baseline 实现和重跑结果不同，因此上表适合说明研发轨迹，不应被当成严格单变量 ablation。正式结论只使用最终同轮 paired 数字。

## 3. MA-EgoQA

### 3.1 最终结果

MA-EgoQA 使用 Gemini 3-Pro-Preview 和 answer-symmetric raw-video retrieval，在完整 1,741 题上无错误完成评测。方法独赢 210 题，baseline 独赢 154 题，净赢 56 题。

| 类别 | 样本数 | Direct | Spatial-Social | 差值 |
|---|---:|---:|---:|---:|
| Environmental interaction | 359 | 44.01% | 44.57% | +0.56 pp |
| Social interaction | 376 | 50.80% | **54.52%** | +3.72 pp |
| Task coordination | 484 | 52.27% | **53.93%** | +1.65 pp |
| Temporal reasoning | 287 | 51.92% | **57.84%** | +5.92 pp |
| Theory of Mind | 235 | 41.28% | **47.66%** | +6.38 pp |

方法的主要价值出现在 ToM 和 temporal reasoning。按子类看，`multi_span` 从 baseline 的 42.04% 降到 40.76%，说明跨片段事件对齐仍是明确短板。

仓库中还有 EgoMAS + Gemini 3-Pro 的 shared-memory/BM25 结果 55.37%。它的绝对分数高于我们的 51.92%，但不是相同 raw-video paired 协议，也不代表 Spatial-Social 方法增益。对外表述时应同时保留这一区分。

#### 3.1.1 Social Code 是否真的在 MA-EgoQA 主链中使用

是。`video_balanced_role_full.jsonl` 的 1,741 条记录全部含有执行后的 Social trace，平均每题 6.30 条 Social records（范围 1--10）。最终 answer VLM 同时接收相同 raw clips、Spatial trace 和 Social trace，而不是只生成 Social Code 后丢弃它。

作为非因果诊断，159/1741（9.13%）题的 Social trace 只有 `unknown`；这组上 full method 为 28.93%，低于 direct 的 32.70%，净输 6 题。其余 1,582 题具有非 unknown Social records，full method 为 54.17%，高于 direct 的 50.32%，净赢 61 题。这个分组受题目难度与证据可得性混杂，不能替代组件消融，但与 MMToM 的发现一致：Social trace 的覆盖和质量比“模块是否存在”更关键。

#### 3.1.2 MA-EgoQA 严格组件消融：已有正式结果，但新组件对照待运行

这里必须区分两件事：**MA-EgoQA 正式完整结果已经跑完**，是 1,741 题上的 Direct `848/1741=48.71%` 对最终 Spatial+Social `904/1741=51.92%`，提升 3.22 pp，McNemar exact `p=0.00388`。目前没有结果的是本次新增的单变量组件对照，而不是 MA-EgoQA 主实验。

历史 MA 文件比较的是 `Direct` 与 `Spatial+Social` 整体，没有在相同题目、相同 clips、相同 final prompt 和同轮模型调用下保存 `Spatial-only`、`Social-only`、`trace-only` 三个条件。因此历史 51.92% 能证明完整系统优于 Direct，但不能把 3.22 pp 拆分成 Spatial 与 Social 各自贡献。早期 `Grounded v2` 等结果也同时包含 Social inference，且检索、prompt、路由策略不同，不能充当这个单变量消融。

已新增 [MA component-ablation runner](../MA-EgoQA/egomas/src/evaluate_video_component_ablation_jdcloud.py) 和 [对应测试](../MA-EgoQA/egomas/src/test_video_component_ablation_jdcloud.py)。固定随机种子 `20260819`，从五个 category 各取 20 题，共 100 题；每题冻结相同问题、相同 raw clips、相同保存的 Spatial/Social traces、同一模型和同一 final-answer prompt，只切换 final VLM 能看到的输入：

| 条件 | Raw video | Spatial trace | Social trace | 用途 |
|---|:---:|:---:|:---:|---|
| `raw_only_rerun` | ✓ |  |  | 同轮 direct control，测服务漂移 |
| `raw_spatial` | ✓ | ✓ |  | Spatial-only |
| `raw_social` | ✓ |  | ✓ | Social-only |
| `raw_spatial_social` | ✓ | ✓ | ✓ | 完整输入，同轮重跑 |
| `trace_only_spatial_social` |  | ✓ | ✓ | 测试 raw video 的不可替代性 |
| `saved_direct` | ✓ |  |  | 历史 paired baseline，不新增调用 |
| `saved_full_primary` | ✓ | ✓ | ✓ | 历史完整方法，不新增调用 |

runner 可断点续跑，逐题 checkpoint，并自动报告整体/分类准确率以及 `raw_spatial -> raw_spatial_social` 等关键对照的 exact McNemar 检验。dry-run 已核对为五类各 20 题，相关测试 14/14 通过。

截至 2026-08-19，本次严格组件消融仍**没有 MA 组件消融准确率**。进一步核对发现，历史运行使用的是 `AutoToM/model/.env` 中的 JDCloud OpenAI-compatible 配置，而不是当前 runner 首先查找的 `GEMINI_*` 配置；旧 endpoint 仍可访问，但最小 `/models` 与 `/responses` smoke 分别返回 `Account invalid` 和“异常apikey”，说明保存的旧 key 已失效。不能用历史版本差异伪装成单变量消融；更新 JDCloud key 后即可直接运行：

```bash
cd MA-EgoQA
GEMINI_CREDENTIALS_FILE=/path/to/credentials.env \
  .venv/bin/python egomas/src/evaluate_video_component_ablation_jdcloud.py
```

### 3.2 成功案例：从“惊讶”到“警告”的主体化因果链

来源：[MA-EgoQA prediction index 863](../MA-EgoQA/data/baselines/video_evidence_audit_full.jsonl)，Theory of Mind。

> 为什么 Tasha 强调大家用刀时要小心？
>
> A. 她之前用过它，并发现它出乎意料地锋利。  
> B. Lucia 使用时差点割到手。  
> C. Alice 特别提到这是一个安全隐患。  
> D. Tasha 自己被它的锋利程度吓了一跳。  
> E. 之前有其他人在厨房错误使用过它。

| 项目 | 回答 |
|---|---|
| Gold | **D：Tasha 自己被刀的锋利程度吓了一跳** |
| Direct baseline | A：她之前使用时发现刀很锋利（错误） |
| Spatial-Social | **D：Tasha 自己被吓了一跳（正确）** |

抽取到的关键证据是连续且带主体归属的：

```python
S.event(2, "Tasha", "express surprise", "blade",
        evidence_id="S2", evidence_span="00:21-00:22", quote="我靠")
S.event(3, "Tasha", "warn about", "blade",
        evidence_id="S3", evidence_span="00:22-00:24",
        quote="啊小心手，这个挺利的")

M.belief(2, "Tasha", "the blade is surprisingly sharp", evidence_ids="S2")
M.intent(3, "Tasha", "warn others to be careful with the blade", evidence_ids="S3")
```

这里有效的不只是识别出“刀很锋利”，而是保留了 `Tasha 感到惊讶 -> Tasha 发出警告` 的同一主体、相邻时间和证据链，因而能够排除“Lucia 差点割伤”“Alice 提醒过”等无证据解释。

### 3.3 失败案例：多片段中重复事件的时间对齐

来源：同一结果文件的 prediction index 335，`multi_span`。

> 两段场景中的镜子旁分别发生了什么？
>
> A. Katrina 在镜前整理头发；更早时 Lucia 也在镜前整理头发。  
> B. Katrina 在镜前整理头发；更早时 Tasha 也在整理头发。  
> C. Katrina 在镜前整理头发；之后 Lucia 也在镜前整理头发。  
> D. Alice 在镜前整理头发；更早时 Tasha 也在整理头发。  
> E. Alice 在镜前整理头发；之后 Lucia 也在整理头发。

| 项目 | 回答 |
|---|---|
| Gold | **C：Katrina -> later Lucia** |
| Direct baseline | **C：Katrina -> later Lucia（正确）** |
| Spatial-Social | A：earlier Lucia -> Katrina（错误） |

证据中同时出现了：

```text
DAY5 11:48  Lucia fixes hair
DAY5 15:21  Katrina fixes hair
DAY6 11:35  Lucia fixes hair
```

原始题目标注的场景来自 DAY2 与 DAY6，但检索结果混入了多个 DAY5/DAY6 的相似“整理头发”事件。方法选择 DAY5 的 Lucia 事件与 Katrina 比较，导致事件集合和先后关系同时漂移。物理动作识别本身正确，失败发生在检索后的场景归属以及“哪个重复事件属于当前组合命题”的跨片段绑定上。这也是 `multi_span` 整体略低于 baseline 的代表性原因。

### 3.4 MA-EgoQA 的迭代过程

| 版本 | Spatial-Social | 当轮 baseline | 差值 | 结论 |
|---|---:|---:|---:|---|
| Shared-memory/BM25 code | 53.02% | 55.38% | -2.36 pp | 结构化压缩损失了原始 memory 信息 |
| 初始 raw-video | 45.20% | 47.39% | -2.18 pp | 仅加入代码并不会自动提高准确率 |
| Provenance v3 | 47.44% | 47.39% | +0.06 pp | 严格引用后基本追平 |
| Standalone v3.1 | 45.49% | 47.39% | -1.90 pp | 过度依赖结构 trace 仍会退化 |
| Contrastive provenance v4 | 49.22% | 47.39% | +1.84 pp | 对候选解释进行对比开始有效 |
| Balanced v4.1 | 51.87% | 48.71% | +3.16 pp | 对称检索改善公平性和证据覆盖 |
| Evidence audit，正式采用 | **51.92%** | 48.71% | **+3.22 pp** | 显著正收益，p=0.00388 |

后续 Executable Hypothesis Layer 在固定 67 题 pilot 上只有 41.79%，低于 saved primary 的 44.78%，因此没有扩展到全量，也没有替换正式结果。

## 4. CoMind 阶段性实验（14/55 个 train recordings）

### 4.1 结果状态与评测协议

本节是 **2026-08-19 的运行中快照**，不是完整 train 结果，也不是官方 test leaderboard 结果。已完成的 14 个录像来自 train split 当前执行顺序，没有做随机抽样或预注册 held-out 选择，因此只能用于检查趋势。CoMind 完整 train 含 55 个录像、6,877 个有效标注样本；当前快照覆盖 1,632/6,877（23.73%）。

| 任务 | 已看到样本 | Direct 成功 | Spatial-Social 成功 | 公平配对样本 |
|---|---:|---:|---:|---:|
| Joint Attention | 1,380 | 1,380 | 1,329 | 1,329 |
| SCOIA | 180 | 180 | 178 | 178 |
| Handover | 72 | 72 | 70 | 70 |
| **合计** | **1,632** | **1,632** | **1,577** | **1,577** |

Spatial-Social 共 55 条结构化输出失败：Joint Attention 51 条、SCOIA 2 条、Handover 2 条，失败率为 55/1,632=3.37%；Direct 在这 1,632 条中无失败。为避免通过失败样本改变两边分母，下面所有 Direct/Spatial-Social 数字都只在双方同时成功的完全相同样本交集上计算。

实验使用本地 `Qwen3-VL-8B-Instruct`，两边接收相同的官方时间协议帧与 transcript：预测时刻前 10 秒均匀取 5 个 context frames，再加 prediction frame；Joint Attention/Handover 使用 leader 与 helper 双视角，SCOIA 只使用 helper 视角；图像最长边缩放到 1408。执行链为：

```text
相同 frames + transcript
  ├─ Direct：一次 VLM inference -> structured task answer
  └─ Spatial-Social：Spatial compiler -> executable Spatial trace
                       -> Social compiler -> executable Social trace
                       -> raw evidence + traces -> structured task answer
```

因此这是 same-model、same-evidence paired comparison，但不是 compute-matched comparison：Direct 每题一次模型调用，Spatial-Social 每题三次模型调用。

### 4.2 14 录像公平配对结果

Joint Attention 的唯一明确正收益是 cue type；类别基本持平但略降，两个视角的定位均完全失败。

| Joint Attention 指标（n=1,329） | Direct | Spatial-Social | 差值 | 论文 Qwen3-VL-32B-FT 参考 |
|---|---:|---:|---:|---:|
| Leader bbox IoU@0.5 | 0.00% | 0.00% | 0.00 pp | 15.62% |
| Helper bbox IoU@0.5 | 0.00% | 0.00% | 0.00 pp | 14.96% |
| Cue type | 14.90% | **21.44%** | **+6.55 pp** | 45.72% |
| Category L1 | **27.99%** | 27.69% | -0.30 pp | 33.23% |

SCOIA 没有形成正收益。Cue type 小幅下降，未来 action verb 与 noun 的绝对准确率都很低，Spatial-Social 进一步退化。

| SCOIA 指标（n=178） | Direct | Spatial-Social | 差值 | 论文 Qwen3-VL-32B-FT 参考 |
|---|---:|---:|---:|---:|
| Object bbox IoU@0.5 | 0.00% | 0.00% | 0.00 pp | 25.13% |
| Cue type | **23.60%** | 21.91% | -1.69 pp | 55.03% |
| Action verb | **2.81%** | 0.56% | -2.25 pp | 65.08% |
| Action noun L1 | **18.54%** | 14.04% | -4.49 pp | 43.12% |

Handover 的 delivery flow、initiation type、category 和 time-to-handover 与 Direct 完全相同；initiator 下降 5.71 pp，bbox 仍为 0。

| Handover 指标（n=70） | Direct | Spatial-Social | 差值 | 论文 Qwen3-VL-32B-FT 参考 |
|---|---:|---:|---:|---:|
| Object bbox IoU@0.5 | 0.00% | 0.00% | 0.00 pp | 1.30% |
| Delivery flow | 51.43% | 51.43% | 0.00 pp | 61.43% |
| Initiator | **47.14%** | 41.43% | -5.71 pp | 72.86% |
| Initiation type | 20.00% | 20.00% | 0.00 pp | 37.14% |
| Category L1 | 25.71% | 25.71% | 0.00 pp | 36.19% |
| TTH within 0.25 s | 5.71% | 5.71% | 0.00 pp | 9.05% |

表中的论文值仅用于显示量级，不能当作严格 baseline 差值：论文参考使用 Qwen3-VL-32B fine-tuned 模型和官方评测协议，而本地 paired 实验使用 8B instruct 模型、公开 train 标注与当前 14 录像子集。严格的因果比较只应看本地 Direct 与 Spatial-Social 两列。

### 4.3 当前可以成立的结论

1. **Social cue 理解存在可重复的正信号。** 单录像 pilot 中 Joint Cue 为 11.43% -> 14.29%；扩展到 14 个录像后为 14.90% -> 21.44%，正收益没有消失，且扩大到 +6.55 pp。
2. **当前方法不是整体优于 Direct。** Joint category 基本持平，SCOIA 的未来动作预测更差，Handover 只有若干指标持平且 initiator 更差。不能把 Joint Cue 的改善概括为 CoMind 全任务提升。
3. **最大瓶颈是定位。** 三个任务的所有 bbox IoU@0.5 都为 0。当前让 VLM 直接生成 `[x,y,width,height]` 数字坐标不可用；更合理的通用方案是由 detector/tracker 提供 proposal，Spatial-Social 只选择 proposal ID。
4. **结构化稳定性仍不足。** 3.37% 的 Spatial-Social 输出失败会降低实际可用覆盖率；需要继续做 schema-constrained decoding、错误重试和 confidence gating。
5. **未来动作不能只靠社会解释。** SCOIA 需要物体 affordance、动作动力学和更强的 temporal anticipation。当前 Social trace 容易生成合理叙事，但没有改善 verb/noun 的精确匹配。

### 4.4 运行状态与可复现文件

截至本快照，50/55 个 train 录像已完整下载。这 50 个录像覆盖 6,012/6,877 条标注（87.4%），已冻结为独立的阶段性集合；剩余下载与推理继续在后台进行。50 录像结果只有在全部 6,012 条完成后才会单独生成，不能用当前 1,632 条结果冒充“87.4% 数据结果”。

- [14-recording paired summary](../workdirs/comind_ready_evaluation/comparison_paired_partial.json)
- [14-recording merged predictions](../workdirs/comind_ready_evaluation/predictions.jsonl)
- [冻结的 50-recording ID 列表](../experiments/configs/comind_downloaded_50_ids.txt)
- 50-recording 最终 paired report（完成后生成）：`workdirs/comind_downloaded_50/comparison_paired.json`
- [CoMind 指标实现](../spatial_social_code/comind.py)
- [官方帧协议抽取](../spatial_social_code/prepare_comind_pilot.py)
- [Direct/Spatial-Social paired runner](../spatial_social_code/run_comind_codex_pilot.py)

## 5. 从三套实验得到的核心认识

### 5.1 “结构化”本身不是贡献

早期 MA-EgoQA text/code 和 raw-video/code 都低于 direct baseline。代码如果只是把原始证据再总结一次，会成为有损瓶颈。真正有效的是结构中保留了 direct answer 容易混淆的变量：谁观察、谁行动、事件阶段、时间、证据出处，以及未知状态。

### 5.2 Provenance 是必要条件，但不是充分条件

clip ID、时间段、observer 和 evidence ID 能减少无依据推断，却不能修复错误的事件语义。MMToM index 42 具有完整 trace，但仍把“准备打开”误判成“检查完成”。所以必须同时验证引用正确性和状态转换正确性。

### 5.3 对比推断比单路解释更稳定

MMToM 与 MA-EgoQA 都是在显式比较候选解释后才形成稳定正收益。模型需要回答“支持 A 的证据是什么、反驳 A 的证据是什么、B 是否只是未知”，而不是生成一段看似连贯的单路故事。CoMind 当前没有候选答案对比，但同样显示自由叙事不能替代精确的类别、时序和坐标输出。

### 5.4 Unknown 必须保持中性

没有看到某个主体、对话或动作，不能推出它没有发生。这个原则在长视频尤其重要，因为检索只返回有限片段。将 missing evidence 当成 negative evidence 会制造非常自信但错误的答案。

### 5.5 不应继续堆更硬的规则

Executable Hypothesis Layer 在 MMToM 上从正式结果的 78.33% 降到 53.00%，在 MA-EgoQA pilot 上也没有超过 primary。当前更合理的方向是修复事件阶段、跨片段绑定和证据覆盖，而不是继续增加会放大 compiler error 的确定性规则。

## 6. 当前最值得做的后续工作

1. 修复 MMToM 的动作阶段类型：严格区分 approach、prepare、open、inspect 和 observe。
2. 专项检查 1.1/1.2 的负迁移，确认是否来自否定语义、选项方向或过强的 inverse-planning likelihood。
3. 为 MA-EgoQA multi-span 建立 `(agent, action, object, day, timestamp, episode)` 级事件绑定，再执行 ALL/ORDER 组合判断。
4. 报告 evidence coverage 与 accuracy 的关系。目前 MA-EgoQA 有 1,591/1,741 题包含 Spatial evidence，1,475 题 citation valid，但仍需分析无证据和无效引用子集。
5. 用同一个开放模型、相同 token budget 比较 direct、matched free-form memory 与 full code，隔离 Gemini 3-Pro backbone 带来的能力增益。
6. 增加多随机种子或重复推理，确认总体提升和各 subtype 差异的稳定性。
7. CoMind 将数值 bbox 生成替换为通用 detector/tracker proposals，并在独立 recording-level held-out 子集上验证 Joint Cue 增益是否保持。
8. CoMind 对无效 code、`unknown` 过多或 Spatial/Social 矛盾的样本加入 method-only confidence gating；回退策略必须在 held-out 前冻结，不能读取 Direct 或 gold。

## 7. 对外汇报时建议使用的表述

> 在严格 paired 的同模型、同视觉输入条件下，Spatial-Social Code 在 MMToM 上达到 78.33%，相对 direct baseline 提升 10.67 个百分点；在 MA-EgoQA 上达到 51.92%，提升 3.22 个百分点，两者均通过 paired McNemar 显著性检验。分析显示，收益主要来自逆向目标/信念推断、Theory of Mind 和时序推理，而简单 belief subtype 与 multi-span 对齐仍存在明确退化。早期结构化版本和后续 deterministic executable ablation 均曾失败，说明收益来自带 provenance 的主体化证据、候选对比和不确定性处理，而不是“输出代码”本身。

不要直接声称“全面超过 SOTA”：MMToM 的不同公开结果存在 multimodal、text-only、verified-state 等协议差异；MA-EgoQA 的项目内 EgoMAS + Gemini 3 shared-memory 绝对分数也高于当前 raw-video Spatial-Social 结果。最可靠的主张始终是同轮、同模型、同输入的 paired improvement。

CoMind 目前只能表述为阶段性发现：在 14 个 train 录像、1,577 个公平配对样本上，Joint Attention Cue 提升 6.55 pp，但定位为 0，SCOIA 退化，尚无全任务总体优势。不得将它描述为完整 train、官方 test 或 leaderboard 结果。

## 8. 本次代码核对记录

本次核对同时检查了 source、保存结果的一致性和现有测试：

| 检查项 | 结果 |
|---|---|
| Spatial/Social runtime 与 compiler 单元测试 | 在 `spatial_social_code/` 目录运行，**37/37 通过** |
| MA paired clips 一致性 | 1,741/1,741 records 均能对齐；method 与 paired baseline 的 clip ID 列表 **0 mismatch** |
| MA 保存的 baseline prediction 一致性 | 1,741 records **0 mismatch** |
| MMToM final JSONL 重算 | 600 条完整记录，方法 470、direct 406，与 summary 一致 |
| MA-EgoQA final JSONL 重算 | 1,741 条完整记录，方法 904、direct 848，与 summary 一致 |
| CoMind paired metrics tests | `tests/test_comind.py` **10/10 通过**；14 录像重算为 1,632 seen、1,577 paired、55 method failures |
| MA evaluator tests | 当前可用 Python 环境缺少 `rank_bm25`，测试在 import/collection 阶段被依赖阻塞；该依赖已声明在根目录 `requirements.txt`，未观察到具体断言失败 |

代码层面的主要待修复项：

1. 真正隐藏 options 后再抽取证据，或将文档统一改为 option-aware/answer-agnostic。
2. 把 claim planner/coverage repair 接入主链后重新做独立评测；若不采用，应删除或明确标记 dead experimental path。
3. 修复 approach/preparing/open/inspect 的阶段解析，并为该失败 case 增加回归测试。
4. 为 runtime 的 confidence 增加 `[0,1]` 范围验证，并避免对字符串使用宽松 `bool(value)`。
5. 建立完整可复现环境并运行 MA 全套测试，避免 evaluator 因可选检索依赖而无法 import。
