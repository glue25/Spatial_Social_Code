# Habitat episode 772：真实 Physical+Social 成功案例

本案例来自 `repeat_3 / physical_social` 的实际 learned-agent 运行，不是 episode
1013，也不是 Oracle action schedule。

## 运行身份与结果

| 字段 | 值 |
|---|---|
| Task ID | `emtom-108294897_176710602-772-cooperative-exhibit-reset-with-split-object-target-k-41a5f8ab` |
| Scene / episode | `108294897_176710602` / `772` |
| 条件 | `physical_social` |
| Repeat / 实际 seed | `3` / `891003` |
| 模型 | `qwen3-vl-8b-instruct` |
| 运行模式 | `standard`，非 Oracle |
| Physical+Social | **100%**，4/4 子任务，14 turns，success=true |
| 同 seed Direct | 75%，3/4 子任务，20 turns 超时，success=false |

任务要求把正确玻璃杯放到洗衣房桌子、花瓶放到客厅桌子、相框放到餐厅
counter，并让 bedroom-1 中正确的抽屉柜保持打开。对象 ID、目的地、房间权限和
通信路径被拆分给三个 agent。

## 实际 Spatial / Social Code

运行时使用的 Code 已从编译后的任务文件逐字导出，三名 agent 各有一份
Spatial Code 和 Social Code：

- [Spatial / Social Code JSON](cases/habitat_episode_772_repeat3_spatial_social_code.json)

编译记录为 `task_conditioned_spatial_social_v1`，`uses_golden_trajectory=false`。
Code 通过每个 agent 的 private context 注入；它给出当前 agent 已知的目标、空间
关系、房间权限和通信角色，但不包含 Oracle 动作序列。

## 动作与通信

完整 42 条 agent-action、4 条 subtask-completion 事件、动作结果、文本视觉摘要、
模型 thought 和 4 次通信尝试均保存在案例 JSON：

- [完整案例 JSON](cases/habitat_episode_772_repeat3_physical_social.json)

成功过程中的关键节点：

| 动作步 | 关键动作或事件 |
|---:|---|
| 1 | agent_0 将 `chest_of_drawers_30`、`vase_1→table_12`、`glass_2→table_22` 等精确信息发给 agent_1 |
| 4 | agent_1 将厨房访问限制及取相框任务发给 agent_2 |
| 8 | agent_1 成功 `Open[chest_of_drawers_30]`，记录第一个子任务完成 |
| 10 | agent_2 成功将 `glass_2` 放到 `table_22` |
| 11 | agent_0 成功将 `vase_1` 放到 `table_12` |
| 13 | agent_2 成功拿起 `picture_frame_0` |
| 14 | agent_2 `Navigate[counter_25]`；同一步评估记录相框目标完成，4/4 完成 |

原始高层日志在第 14 步没有单独记录 `Place[picture_frame_0, ...]`，但同一日志的
subtask-completion 事件和最终评估都判定 `(is_on_top picture_frame_0 counter_25)`
完成。导出文件忠实保留这一现象；在有画面的回放完成前，不额外推断未记录动作。

## 画面状态

该历史运行使用 `benchmark_observation_mode=text`：42 条动作的
`selected_frame_paths` 全为空，运行目录中也没有 PNG、JPEG、MP4 或 WebM。

已检查回放环境，但当前主机 `nvidia-smi` 无法连接 NVIDIA driver，Habitat 渲染
无法启动。因此本次没有上传图片，也没有用示意图冒充模拟器画面。恢复 GPU 后应
按原动作轨迹补以下三张，并标注为“回放画面”：

1. 动作步 8：`chest_of_drawers_30` 打开；
2. 动作步 11：`vase_1` 已放到 `table_12`；
3. 动作步 14：agent_2 到达 `counter_25`，重点核对相框最终关系。

案例 JSON 的 `visual_artifacts.replay` 记录了当前阻断状态。图片缺失不影响
100% 指标是历史结果文件中的真实记录，但意味着目前还不能用视觉证据独立核查
最后一步的状态变化。
