# Habitat Spatial-Social Code 实验状态

更新时间：2026-09-01。

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

冻结 12 题用于开发和消融选择。只有 Spatial+Social 明确超过 Direct、
Spatial-only 和 Social-only 后，才扩展到更大的正式集合。

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

实验输出位于 `workdirs/habitat_enacttom/frozen12/benchmark_qwen8b_task_conditioned_v2/`，
该目录是本地运行产物，不提交 GitHub。最终完成后应在本报告加入逐条件成功率、
平均目标完成度、paired win/loss 和代表性 case。
