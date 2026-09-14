# Spatial–Social Code 完整补充材料（证据约束版）

本文件由机器可读结果自动生成。不同协议分表报告，不把历史诊断、开发集、单例或外部论文数字混成同一个主结果。

## A. 表征定义

Spatial Code 保存对象、房间、支撑/容器、抓取、关节、时间和证据来源；Social Code 保存主体、观察/消息归属、访问权限、通信预算、reported/observed/unknown。物理事实只有在主体观察到或收到确认送达消息时，才进入该主体状态。

## B. 主要实证结果

### MMToM：选项盲、同模型同帧

Gemini 3-Pro-Preview 的 204 个成功配对样本中，Direct 为 121/204（59.3%），SSC 为 142/204（69.6%），exact McNemar p=0.00145049。该结果是当前最强干净问答证据。

question-independent GPT-5.6-sol compile-once 开发 pilot 为 Direct 67.4%、SSC 74.4%，p=0.453125；样本只有 43 题，不显著。

### MA-EgoQA：冻结感知的强推理基线

| 条件 | 准确率 | calls | input tokens |
|---|---:|---:|---:|
| cot | 10/20 (50.0%) | 20 | 178925 |
| natural_language_scratchpad | 9/20 (45.0%) | 20 | 179265 |
| pddl_mind_adapted | 11/20 (55.0%) | 20 | 179745 |
| spatial_only | 10/20 (50.0%) | 20 | 178945 |
| ssc | 10/20 (50.0%) | 20 | 182724 |

这 20 题中 PDDL-Mind adapted 为 55%，SSC 为 50%；不能声称 SSC 胜过所有强基线。

### Habitat：三 seed 严格闭环

| 条件 | 严格成功 | 平均完成度 | benchmark成功 | 平均token |
|---|---:|---:|---:|---:|
| direct | 3/36 (8.3%) | 53.0% | 8.3% | 147432 |
| static_ssc | 3/36 (8.3%) | 47.2% | 8.3% | 168007 |
| dynamic_ssc | 4/36 (11.1%) | 41.7% | 13.9% | 221910 |

Dynamic 对 Direct 的配对 McNemar p=1.0，且平均完成度更低、token 更高。闭环实现成立，但总体效果主张不成立。

### CoMind：recording-heldout OOF grounding

CoMind 的可支持结论是显式 grounding 改善 bbox/category 字段；它不是官方 hidden-test。完整数字见源文件 `reports/comind_grounding_oof_summary.json`。

## C. SSC → 资产化环境与 Physical AI

资产编译状态：`accepted=true`；真实 HSSD stage、5 个刚体配置、agent 出生点、权限、通信图、目标与反事实规范均有来源哈希。它复用现有资产，不是新网格生成。

已执行 8 个 Direct/Dynamic 配对运行；它们是固定场景机制案例，不作为总体准确率估计。

未来更强的多模态世界模型可以执行 `视频/传感器 → SSC（逆图形学） → 资产检索与布局 → 可交互模拟器 → agent 尝试 → 新观察更新 SSC`。这里应写作未来 GPT-6 级模型潜力，不能声称 GPT-6 已完成或已验证。

## D. 人工核验

人工包目标答案是否出现：`False`；当前状态：`awaiting_two_independent_reviewers`。未填写的人工结果不能报告为完成。

## E. 泄漏与复现边界

- 干净 MMToM 主结果不含选项、query_evidence 或手工答案权重。
- Habitat planner 不接收 scorer-only 标签；失败动作不会写成完成。
- 生成环境 cases 删除 golden trajectory、目标答案和持久化 expected response。
- 开发 pilot、单例和合成干预只作机制证据。
- 每个源文件的 SHA-256 记录在配套 manifest。

## F. 当前可写与不可写

可写：MMToM 选项盲配对提升；SSC 是带主体归属和 provenance 的中间表征；Habitat 在线闭环可审计；SSC 可编译为资产化场景规范。

不可写：SSC 在所有环境显著提升；Dynamic SSC 在 Habitat 总体胜出；已经从任意视频生成全新3D世界；GPT-6 已用于本实验；人工核验已经完成。
