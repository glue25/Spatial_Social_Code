# Habitat Spatial-Social Code 实验状态

更新时间：2026-09-03。

## 论文主实验

Habitat 主实验只比较四个直接相关的条件：

1. Direct：原始任务与每个 agent 自己的私有信息；
2. Spatial-only：增加任务相关目标、已知空间关系、对象类型和交互前置条件；
3. Social-only：增加 agent-indexed 私有知识、房间访问能力、通信图、消息预算和中继角色；
4. Spatial+Social：同时使用两类代码。

四组使用相同的冻结任务、Qwen3-VL-8B、私有视角、动作空间和 20-turn
预算。编译器可以读取当前任务，但不读取 `golden_trajectory`、标签、正确性
反馈或 baseline 输出；生成内容不包含物理动作序列。Qwen 仍通过 Habitat
标准运行时自主选择并执行每一步动作。

冻结 12 题使用三个 Habitat seed 重复评测，共 36 个 run/条件、144 个配对
run。模型保持 greedy decoding；seed 只改变 Habitat 运行状态，四个条件在每轮
使用相同 seed。

## 三轮合并结果

| 条件 | Task success | 平均目标完成度 |
|---|---:|---:|
| Direct | 1/36（2.78%） | 45.83% |
| Spatial-only | 1/36（2.78%） | 43.06% |
| Social-only | **7/36（19.44%）** | **59.95%** |
| Spatial+Social | 4/36（11.11%） | 53.47% |

Spatial+Social 相对 Direct 的平均完成度提升 **7.64 pp**，36 个 paired run
为 13 胜、17 平、6 负。按任务聚类、保留三个 seed 的 bootstrap 95% CI 为
`[+0.69, +15.74]` pp，因此支持“P+S 改善部分目标完成度”。严格 task
success 提升 8.33 pp，但样本仍小，只作为辅助结果。

组件归因并不支持互补性：P+S 比 Spatial-only 高 10.42 pp，但比 Social-only
低 6.48 pp；Spatial-only 还比 Direct 低 2.78 pp。当前最强 Habitat 条件是
Social-only，而不是 P+S。论文可以报告 P+S 相对 Direct 的稳定完成度增益，
但不能声称 Spatial Code 在 Habitat 中带来额外增益。

## Spatial v2 诊断

根据旧版 trace 中重复执行同一失败 Navigate/Place 的错误，v2 加入 UNKNOWN
证据门控、房间权限检查、禁止原样重复失败动作和有限重试后 Find/委派。该版本
在 seed 891003 的同一12题上完成24个新 run：

| 条件 | Task success | 平均目标完成度 |
|---|---:|---:|
| Direct（同 seed 既有对照） | 0/12 | 44.44% |
| Spatial-only v1 | 0/12 | 38.19% |
| Spatial-only v2 | 0/12 | **44.44%** |
| Social-only（同 seed 既有对照） | **3/12** | **63.89%** |
| P+S v1 | 2/12 | 56.94% |
| P+S v2 | 1/12 | 54.86% |

因此 v2 修复了 Spatial-only 的平均退化，但没有改善 P+S 融合。它是开发诊断，
不替代三 seed 主结果，也不能据此选择后再在同一12题上宣称泛化提升。

## 已有对照与边界

冻结 12 题的一次运行：

| 条件 | Task success |
|---|---:|
| Direct / private standard | 1/12（8.33%） |
| Shared-information baseline | 4/12（33.33%） |
| Oracle action schedule ceiling | 11/12（91.67%） |

Paper-80 三次重复尚未全部完成；暂停时的部分结果为：普通 standard
3/109（2.75%），oracle schedule 153/225（68.00%）。这些结果只说明规划与
协调存在很大可改进空间。Oracle schedule 来自黄金轨迹，不能作为
Spatial-Social Code 的主结果。

## 当前实现

- 编译器：`experiments/habitat_enacttom/build_task_conditioned_code.py`
- 后台入口：`experiments/habitat_enacttom/run_task_conditioned_frozen12.sh`
- 顺序：Spatial+Social → Spatial-only → Social-only，优先尽快检验完整方法。
- 代码通过 EnactToM 原有的 per-agent private context 输入，无需修改 vendored
  EnactToM；不同 agent 不会看到彼此私有事实的具体值。
- Social Code 显式编码受限通信的有向路径和单消息中继策略；Spatial Code
  显式编码交互前置条件，避免把家具当作可拿物、未导航就 Open/Place 等错误。

首轮输出位于 `workdirs/habitat_enacttom/frozen12/benchmark_qwen8b_task_conditioned_v2/`，
新增 seed 位于 `workdirs/habitat_enacttom/frozen12/benchmark_qwen8b_task_conditioned_repeats/`。
这些目录是本地运行产物，不提交 GitHub。
