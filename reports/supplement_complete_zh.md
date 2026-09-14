# Spatial–Social Code 补充材料证据汇编（进行中）

> 本文件名为兼容旧链接而保留 `supplement_complete_zh.md`；它不表示实验已经完成。人工核验、冻结确认集和若干同内容控制仍待完成。

本文件由机器可读结果自动生成。不同协议分表报告，不把历史诊断、开发集、单例或外部论文数字混成同一个主结果。逐样本证据见 `reports/audit_latest_experiments/`。

## A. 表征定义

Spatial Code 保存对象、房间、支撑/容器、抓取、关节、时间和证据来源；Social Code 保存主体、观察/消息归属、访问权限、通信预算、reported/observed/unknown。物理事实只有在主体观察到或收到确认送达消息时，才进入该主体状态。

## B. 主要实证结果

### MMToM：选项盲、同模型同帧

Gemini 3-Pro-Preview 的 204 个成功配对样本中，Direct 为 121/204（59.3%），SSC 为 142/204（69.6%），exact McNemar p=0.00145049。这是 API 中断前的部分运行积极信号，不是完整 600 题结果；保存记录没有 provider token usage。

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

任务异质性很强：episode 1166 中 Dynamic 为 3/3 seed 全成功而 Direct/Static 均为 0/3；episode 1566 中 Direct 为 3/3 而 Dynamic 为 0/3。两种方向都必须报告。

### CoMind：recording-heldout OOF grounding

CoMind 的可支持结论是显式 grounding 改善 bbox/category 字段；它不是官方 hidden-test。完整数字见源文件 `reports/comind_grounding_oof_summary.json`。

## C. SSC → 资产化环境与 Physical AI

资产编译状态：`accepted=true`；真实 HSSD stage、5 个刚体配置、agent 出生点、权限、通信图、目标与反事实规范均有来源哈希。它复用现有资产，不是新网格生成。

已执行 8 个 Direct/Dynamic 配对运行；它们是固定场景机制案例，不作为总体准确率估计。

未来更强的多模态世界模型可以执行 `视频/传感器 → SSC（逆图形学） → 资产检索与布局 → 可交互模拟器 → agent 尝试 → 新观察更新 SSC`。这里应写作未来 GPT-6 级模型潜力，不能声称 GPT-6 已完成或已验证。

## D. 人工核验

人工包目标答案是否出现：`False`；当前状态：`awaiting_two_independent_reviewers`。未填写的人工结果不能报告为完成。

## E. 泄漏、状态与复现边界

- MMToM 保存的 204 个有效样本的 compiler Code 不含选项、query_evidence 或手工答案权重；但运行不完整，且缺 provider token usage。
- Habitat planner 不接收 scorer-only 标签；失败动作不会写成完成。
- strict scorer v2 在稳定采样不能推进时输出 invalid/null，不再把瞬时完成度纳入汇总。历史 108 个 summary 的 `stability_advance_ok` 均为 true，因此不因该修复而改分。
- 动态状态 v2 将成功移动后失效的旧 on/in/inside 关系移入 history，不再同时作为当前事实。历史 rollout 保持原样，不能假装由新 runtime 生成。
- 生成环境 cases 删除 golden trajectory、目标答案和持久化 expected response。
- 开发 pilot、单例和合成干预只作机制证据。
- 每个源文件的 SHA-256 记录在配套 manifest。

## F. 当前可写与不可写

可写：MMToM 204 个有效配对的选项盲部分运行显示积极信号；SSC 是带主体归属和 provenance 的中间表征；Habitat 在线闭环可审计；SSC 可编译为资产化场景规范。

不可写：SSC 在所有环境显著提升；Dynamic SSC 在 Habitat 总体胜出；已经从任意视频生成全新3D世界；GPT-6 已用于本实验；人工核验已经完成。
