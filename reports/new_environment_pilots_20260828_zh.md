# 新环境 pilot、held-out 扩展与 Habitat 修复结果

更新时间：2026-09-03。

## 结论摘要

- **SocialNav-SUB**：18题 pilot 后又完成60题开发切片和120题 held-out。最终 held-out 官方 human-probability agreement 为 Direct 57.77%、Visual P+S 51.92%、冻结 router 57.39%；没有增益。
- **StoryVideoQA**：query-independent indexed story code 在12个 held-out episode、240题上为217/240（90.42%），Direct caption 为224/240（93.33%）；没有增益。
- **Habitat/EnactToM**：最终统一的私有视角、目标最小 grounding、确定性 AST runtime 条件在冻结 5 题上 **5/5 完成任务**。这证明 oracle code 与执行器链路可行，但不是 learned planner 的成绩。

## SocialNav-SUB

设置：每个场景在不知道题目时只编译一次 visual code，随后复用回答同场景的 3 个问题；共 6 个场景、18 题。

| 条件 | 官方 human probability agreement | Human-mode accuracy |
|---|---:|---:|
| Direct | 60.16% | 66.67% |
| Oracle code | 59.44% | 61.11% |
| Visual predicted P+S | 59.28% | 72.22% |

Human-mode accuracy 对 Visual P+S 较高，但样本只有 18 题，且官方连续概率指标没有提升，因此只应视为需要扩样的信号。

后续在官方 metadata `[0:20]` 的60题开发切片上，Direct agreement 为59.89%，
Visual P+S 为55.73%。根据开发切片只冻结“Social reasoning 用P+S，其余用
Direct”的题型 router，然后在从未用于选择的 `[20:60]`、120题上评测：

| held-out 条件 | Human probability agreement | Human-mode accuracy |
|---|---:|---:|
| Direct | **57.77%** | **64.17%** |
| Visual P+S | 51.92% | 54.17% |
| 冻结题型 router | 57.39% | 63.33% |

router 接近 Direct 但没有超过。后续优化必须回到开发切片或新增独立开发场景，
不能继续读取这120题的标签调 prompt。

材料：

- `workdirs/socialnav_sub/pilot_v1/jdcloud_pilot/summary.json`
- `workdirs/socialnav_sub/pilot_v1/jdcloud_pilot/predictions.jsonl`
- `workdirs/socialnav_sub/pilot_v1/jdcloud_pilot/visual_compiled.jsonl`
- `workdirs/socialnav_sub/pilot_v1/jdcloud_pilot/cases.md`
- `workdirs/socialnav_sub/pilot_v1/official/`（60 张官方图片）

## StoryVideoQA

设置：使用官方 BigBang caption，将完整 episode 先固定压缩为 10 秒 bins；在不读取题目和选项时编译一次 Social Story Code，然后复用回答前 12 题。Direct 与 code 使用相同的完整 caption timeline。

| 条件 | 正确数 | 准确率 |
|---|---:|---:|
| Direct caption | 12/12 | 100.00% |
| Frozen Social Story Code | 11/12 | 91.67% |

唯一错误是问题 45：code 在 episode 压缩时保留了 pantsing 主事件，但丢失了 Leonard 提到母亲在哪里买裤子的细粒度地点事实。

材料：

- `workdirs/storyvideoqa/pilot_v1/jdcloud/summary.json`
- `workdirs/storyvideoqa/pilot_v1/jdcloud/compiled_episode.json`
- `workdirs/storyvideoqa/pilot_v1/jdcloud/predictions.jsonl`

这是 caption-only feasibility pilot，不是视频视觉结果。

后续 indexed story code 在 E01--E05 开发后，冻结版本依次评测 E06--E17，
共12个 held-out episode、240题：Direct caption 224/240（93.33%），indexed
code 217/240（90.42%）。因此扩展后仍是负结果，不进入正向投稿主表。

## Habitat/EnactToM

诊断过程：

1. 普通 prompt/action-card 在私有 partial graph 下失败；模型会因未见对象缺少可执行 node handle 而自行搜索或替换目标。
2. full/shared graph + 动态 schedule 在 5 个冻结任务上 5/5，确认模拟器、目标定义和动作序列没有坏。
3. 仅加强文字指令仍会失败：模型会把 `couch_25` 改成 `couch_26`，或忽略 `Navigate[bottle_2]` 去搜索。
4. 最终修复为：只向被分配的 agent 注入**当前 action 中出现的实体及其必要位置路径**，并让 AST runtime 直接执行冻结 action，不再让模型复述 action。

最终统一条件：

- private/partial per-agent graph：开启；
- full shared graph：关闭；
- grounding：仅当前 action 的实体；
- 动作：确定性 AST runtime；
- 冻结任务完成率：**5/5（100%）**；
- literal-ToM root probes：**2/5（40%）**。

因此，5/5 支持“oracle agent-indexed executable code 能可靠驱动任务执行”，但 40% 的 literal-ToM probe 说明它没有证明模型获得了普遍的心智推理能力。

最终结果目录：

- `workdirs/habitat_enacttom/frozen12/benchmark_qwen8b_oracle_schedule_v8_private_runtime_full5/`
- `workdirs/habitat_enacttom/frozen12/benchmark_qwen8b_oracle_schedule_v8_private_runtime_full5/status.jsonl`

实现与验证：

- `external/EnactTom/enacttom/runner/benchmark.py`
- `external/EnactTom/enacttom/task_gen/task_generator.py`
- `experiments/habitat_enacttom/build_oracle_schedule_tasks.py`
- `experiments/habitat_enacttom/run_oracle_schedule_v3.sh`
- targeted tests：5 passed；修改文件通过 `py_compile`。

## 论文口径

目前最稳妥的表述是：

> 在 Habitat 冻结任务上，agent-indexed oracle code 经目标最小 grounding 和确定性 runtime 可实现 100% functional completion；该结果验证的是表示到执行的可行性与上限，而不是 learned ToM 或自动 code compiler 的性能。

SocialNav-SUB 和 StoryVideoQA 当前都应报告为 pilot/negative-or-neutral evidence，不应写成新增 SOTA 或稳定增益。
