# Habitat hard_20 案例：标准规划与 Oracle 调度上界

本页公开一个可逐步核查的 Habitat/EnactToM 案例。它用于展示规划与多智能体
协调失败如何影响任务完成，不作为 Spatial-Social Code 相对 Direct 的主实验结果。

## 画面核对结论

该次历史运行**没有保存 RGB 截图或视频**：两组结果的
`benchmark_observation_mode` 都是 `text`，标准运行的 55 条动作记录和 Oracle
运行的 18 条动作记录中，`selected_frames`、`selected_frame_paths` 和
`selected_frame_handles` 均为空；结果目录也没有 PNG、JPEG 或 MP4 文件。

因此，下面展示的是 Habitat 运行时返回的动作、成功/失败反馈和文本视觉摘要，
不是事后生成的模拟截图。若要展示真实第一/第三人称画面，需要使用相同 scene、
episode 和动作计划，以 `--video` 重新回放；回放只补可视化，不应改写本页历史指标。

## 任务

- Task ID：`emtom-104862639_172226823-1013-mixed-quiet-museum-reset-with-a-relayed-drawer-5f2c5fa7`
- Scene / episode：`104862639_172226823` / `1013`
- 标题：Quiet museum reset with a relayed drawer check
- 模型：`qwen3-vl-8b-instruct`
- 轮数上限：20
- 限制：房间访问限制、每人一次消息、受限有向通信

团队需要同时完成四项功能目标：

1. 把 `plate_0` 放到客厅边桌 `table_19`；
2. 把 `vase_9` 放到卧室床 `bed_18`；
3. 最终保持正确的卧室抽屉柜 `chest_of_drawers_34` 打开；
4. 最终保持厨房冰箱 `fridge_31` 打开。

关键社会信息被分散在不同 agent：agent_1 知道正确抽屉柜的 ID，agent_0 知道
物体目标和冰箱终态，agent_2 能进入卧室与厨房；通信方向和次数又受到限制。

## 结果边界

| 条件 | 成功 | 记录进度 | 轮数 | 模型 API 调用 |
|---|---:|---:|---:|---:|
| Standard agent planning | 否 | 75% | 20 | Qwen 65；评估器 1 |
| Oracle action schedule | 是 | 100% | 6 | Qwen 3 |

这里的 `75% → 100%` 只表示同一任务的标准规划轨迹与确定性 Oracle 动作调度
上界。Oracle 每轮直接给出一个精确动作，计算量也不匹配，因而不能写成“SSC 将
准确率从 75% 提升到 100%”。SSC 的正式 Habitat 结果仍应引用冻结 12 任务、
三 seed 的配对汇总。

## Oracle 成功轨迹

| 轮次 | agent_0 | agent_1 | agent_2 | 新完成目标 |
|---:|---|---|---|---|
| 1 | `Navigate[fridge_31]` | `Navigate[vase_9]` | `Navigate[chest_of_drawers_34]` | 定位 |
| 2 | `Open[fridge_31]` | `Pick[vase_9]` | `Open[chest_of_drawers_34]` | 冰箱、正确抽屉柜打开 |
| 3 | `Navigate[living_room_1]` | `Navigate[bed_18]` | `Navigate[plate_0]` | 搬运中 |
| 4 | `Wait[None]` | `Place[vase_9, on, bed_18, None, None]` | `Pick[plate_0]` | 花瓶到正确床 |
| 5 | `Wait[None]` | `Wait[None]` | `Navigate[table_19]` | 搬运中 |
| 6 | `Wait[None]` | `Wait[None]` | `Place[plate_0, on, table_19, None, None]` | 全部功能目标完成 |

18 个 agent-action 全部收到 `Successful execution!`。评估记录为 `done=true`、
`functional_success=true`、`percent_complete=1.0`。

## Standard 失败轨迹的关键节点

Standard 轨迹并非完全不会执行物理动作：agent_0 最终把盘子放到 `table_19`，
agent_1 也打开了 `chest_of_drawers_34`。失败主要来自错误目标、冲突终态和无效重试：

| 轮次 | 记录到的行为 | 后果 |
|---:|---|---|
| 2 | agent_0 将唯一消息用于泛化请求“找花瓶” | 没有传递 `bed_18`、`fridge_31` 等关键私有事实 |
| 6 | agent_1 在未靠近时 `Pick[vase_9]` | 执行失败 |
| 8 | agent_0 未靠近 `table_19` 就放盘子 | 执行失败，之后通过 Navigate 修复 |
| 10 | agent_1 将花瓶放到 `bed_14` | 选择了错误卧室床，未满足 `bed_18` 目标 |
| 11 | agent_0 再次通信 | 已耗尽消息预算，发送失败 |
| 13–14 | agent_0 先打开、随后关闭 `fridge_31` | 破坏团队要求的最终打开状态 |
| 14–15 | agent_1 尝试进入受限厨房并远程开冰箱 | 分别因访问限制、距离/遮挡失败 |
| 12–20 | agent_2 反复在 `counter_25` 附近 Navigate/Pick | 多次未靠近或无进展，直到 20 轮超时 |

最终记录为 `done=false`、`functional_success=false`、`percent_complete=0.75`。
这说明该 case 的主要瓶颈确实与“谁知道哪个精确目标、谁能进入哪个房间、有限
消息应传什么、如何维持最终状态”有关；但单个 Oracle 案例本身不能估计 SSC 的
平均因果增益。

## 可审计数据

精简机器可读记录见
[`cases/habitat_hard20_case_summary.json`](cases/habitat_hard20_case_summary.json)。
原始媒体、完整数据集和本地运行目录不随仓库发布。
