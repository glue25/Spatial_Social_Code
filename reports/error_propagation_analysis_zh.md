# Spatial–Social Code 误差传播与不确定性分析

> 本报告由 `scripts/analyze_error_propagation.py` 从冻结结果自动生成。中间层没有人工真值时只报告代理指标，不把模型自报 confidence 写成真实确定性。

## 一页结论

- **MMToM**：Direct 67.67% → primary 74.17% → final 78.33%。不确定性门修复 28 题、破坏 3 题，净增 25；但仍有 27 个置信度≥0.9的错误。
- **MA-EgoQA**：Direct 48.71% → structured 51.87% → audit final 51.92%。audit 只净增 1 题，说明主要增益来自结构化回答之前，而不是事后审计。
- **Habitat**：P+S 的成功率 11.11%、平均完成度 53.47%；相对 Direct 的配对平均完成度变化 +7.64 pp。这里可直接定位首个 simulator action failure。
- **CoMind**：SAM grounding 的 12075 个 bbox/view 判断中 IoU@0.5 为 19.70%。这是最清楚的真值定位：原始文本 bbox 几乎全错，SAM 大幅修复，但 grounding 仍是主要瓶颈。
- **SocialNav-SUB / StoryVideoQA**：仍是负向或中性 pilot。SocialNav P+S agreement 51.92% vs Direct 57.77%；Story Code 90.42% vs Direct 93.33%。

## MMToM：答案阶段传播和校准

|阶段|正确/总数|准确率|
|---|---:|---:|
|Direct|406/600|67.67%|
|Code primary|445/600|74.17%|
|Uncertainty final|470/600|78.33%|

传播拆解：Code primary 修复 Direct 错误 131/194（67.53%），同时把原本正确的答案改错 92/406（22.66%），净增 39。随后 uncertainty 阶段修复 28/155，又破坏 3 个正确答案。

不确定性 gate 触发 175 题；错误检测 precision=55.43%，recall=62.58%。ECE=0.0686，Brier=0.1548。这里最重要的风险是 **58 个 confidence≥0.8 的错误**。

Spatial Code 中 literal unknown 为 0；Social Code 保留 unknown 的 298 题准确率为 84.23%，不含 unknown 的准确率为 72.52%。这支持“允许 unknown 比强行填值更安全”，也暴露 Spatial 层过度声明。

## MA-EgoQA：证据覆盖、引用和审计

|有效 evidence 数|题数|准确率|
|---|---:|---:|
|0|150|28.00%|
|1|408|43.63%|
|2|232|60.78%|
|>=3|951|57.10%|

传播拆解：structured stage 修复 Direct 错误 210/893，但引入 155/848 个新错，净增 55。audit 只修复 3 个 primary 错误并破坏 2 个正确答案；其错误检测 precision=66.92%、recall=21.24%。

citation valid 时准确率 55.25%，invalid 时 33.46%。但“引用有效”仅表示引用 ID 合法，不表示证据语义正确。强证据代理集合有 382 题，其中 149 题最终仍错；不能把它直接叫作“模型已经知道答案”。

Instrumentation 缺口：159 题 Social trace 没有 confidence；repair 实际触发 0 次；claim coverage 非空 0 题。

## Habitat：动作级错误传播

|条件|成功率|平均完成度|有动作失败的 run|失败动作数|timeout|
|---|---:|---:|---:|---:|---:|
|direct|2.78%|45.83%|32/36|128|34|
|physical_only|2.78%|43.06%|25/36|77|34|
|social_only|19.44%|59.95%|32/36|107|27|
|physical_social|11.11%|53.47%|33/36|109|32|

Direct→P+S success 配对转移：`{'F→F': 32, 'F→T': 3, 'T→T': 1}`；完成度逐 run 比较为 `{'degraded': 6, 'improved': 13, 'unchanged': 17}`。P+S 首个失败动作类型：`{'Pick': 9, 'Place': 7, 'Open': 10, 'Navigate': 7}`。P+S 中执行了含 UNKNOWN 的动作 8 次，说明表示中的未决变量确实会传播到 action 层。

## CoMind：可被真值定位的 grounding 误差

|任务/视角|样本|IoU@0.5|
|---|---:|---:|
|handover.helper|159|22.01%|
|handover.leader|139|24.46%|
|joint_attention.helper|5402|21.27%|
|joint_attention.leader|5402|20.33%|
|scoia.helper|973|6.47%|

SAM proposal score 对最终 bbox hit 的 ECE=0.6508。这个 score 明显不是校准后的正确概率；高 score 错框案例已写入 cases 文件。完整的 raw SSC→SAM 配对 contingency（含所有 Social 字段）保存在汇总 JSON 的 `published_oof_gate_vs_spatial_social`。

## 两个扩展 pilot

SocialNav-SUB 的 Direct→P+S human-mode 转移为 `{'F→F': 35, 'F→T': 8, 'T→F': 20, 'T→T': 57}`。由于只有问题级人类分布，没有 code 字段真值，只能定位“用了 code 后答案变坏”，不能断言究竟是哪条 Spatial/Social fact 错。

StoryVideoQA 的 Direct→Story Code 转移为 `{'F→F': 11, 'F→T': 5, 'T→F': 12, 'T→T': 212}`。检索 event 数与准确率是覆盖代理，不是证据正确率。

## 论文中建议采用的错误分类

1. **Representation omission/unknown**：应有事实缺失或保留 unknown。
2. **Representation hallucination**：需要 fact-level 人工标注才能严格确认；现阶段不要用最终答错反推 code 错。
3. **Grounding error**：CoMind bbox 有真值，可以严格定位。
4. **Reasoning/fusion error**：中间答案正确而后处理改错，或强证据代理下仍答错。
5. **Execution error**：Habitat simulator 明确返回失败。
6. **Calibration failure**：高置信错误、gate 对错误的 precision/recall 不足。

## 科学边界

- MMToM/MA-EgoQA 没有逐条 Spatial/Social fact 的人工真值；报告中的 unknown、citation 和 evidence coverage 都是代理。
- unknown 与准确率的关系是相关性分析，不能解释成 unknown 本身造成提升或下降。
- CoMind 分析是 labeled-train recording-heldout OOF，不是官方 hidden test。
- SocialNav-SUB 与 StoryVideoQA 是 pilot，不进入主结果主张。
- Habitat 的首个失败动作不一定是最早的认知错误；计划错误可能在此前已经发生，只是在执行时才可观察。
- 若论文要正式声称“知道但答不出”，下一步需要抽样人工标注 code 是否包含足够且正确的 gold facts。
