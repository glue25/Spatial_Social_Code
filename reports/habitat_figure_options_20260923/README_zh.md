# Habitat 逐轮完成度图：完整候选与核验总结

## 结论与推荐

**如果 panel (d) 可以明确标为一个子类，推荐原始 `cooperative` 类的全部 5 个任务。**
这组不仅最终完成度较高，20 轮时间平均完成度也较高，比 hard 组更适合展示交互过程。
但它是事后探索性展示，不应替代全体统计，也不证明整体更快或跨 seed 稳定。
建议附录同时放 cooperative/mixed 两类和完整 12 任务曲线。

**如果 panel (d) 必须代表全部任务，用完整 12 任务曲线，不换成有利子集。**
另一个保留全部任务的补充视角是“物体放置目标完成度”，但纵轴必须相应改名，
同时披露打开容器目标的结果。不能把只算放置目标的曲线标成总目标完成度。

本目录所有分析只读已有日志，没有新调用 API、没有重启或修改后台实验。
所有统计为真实 post-action 物理评分；终止后保持最终状态；失败保留。

## 1. 可用统计视角（完整第一 seed 891011）

所有百分数均以任务为等权统计单位。± 为跨任务样本标准差，ddof=1。
“时间平均”是第 1–20 轮完成度的算术平均，不是最终成功率或成功样本的完成时间。

| 范围 | 任务数 | NL 最终均值±SD | SSC 最终均值±SD | 最终差值 pp | NL 时间平均 | SSC 时间平均 | 时间平均差值 pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全部任务 | 12 | 84.03±22.04 | 89.58±15.54 | +5.56 | 50.45 | 49.17 | -1.28 |
| 原始 cooperative | 5 | 86.67±29.81 | 93.33±14.91 | +6.67 | 54.25 | 62.42 | +8.17 |
| 原始 mixed | 7 | 82.14±16.96 | 86.90±16.57 | +4.76 | 47.74 | 39.70 | -8.04 |
| 原始 hard | 6 | 79.17±26.74 | 90.28±15.29 | +11.11 | 42.85 | 47.50 | +4.65 |
| 原始 standard | 6 | 88.89±17.21 | 88.89±17.21 | 0 | 58.06 | 50.83 | -7.22 |
| 物体放置目标，全部任务 | 12 | 75.00±33.71 | 83.33±24.62 | +8.33 | 41.11 | 46.60 | +5.49 |
| 打开容器目标，全部任务 | 12 | 100±0 | 100±0 | 0 | 68.33 | 54.79 | -13.54 |

分组依据都是冻结任务文件中的字段，或原始物理评分的 predicate。
没有基于输赢组合任务，没有只保留成功轨迹。原始任务类别、难度、人数、ToM
标签的完整探索结果还保存在 `../habitat_memory_subgroups_20260923/`。
这些视角相互重叠，不能当成多份独立证据。

### 推荐 panel (d)：cooperative，n=5

完整成员：1781（茶具）、772（展品）、588（检查布置）、618（巡视简报）、619（画廊）。
其中也包括 SSC 输给 NL 的 619，没有删除负例。两组最终严格成功都为 4/5；
这里展示的是**目标完成度和进度优势，不是严格成功率优势**。

| 交互轮次 | NL 平均完成度 | SSC 平均完成度 |
|---|---:|---:|
| 5 | 20.00% | 26.67% |
| 8 | 45.00% | 61.67% |
| 10 | 66.67% | 80.00% |
| 12 | 73.33% | 80.00% |
| 15 | 80.00% | 80.00% |
| 17 | 80.00% | 93.33% |
| 20 | 86.67% | 93.33% |

20 个轮次中 14 个 SSC 领先、5 个相同、1 个（第 3 轮）落后。
不是所有轮次都领先，也不是差距随轮次单调扩大。
最后一轮跨任务样本方差：NL **888.89 pp²**，SSC **222.22 pp²**。
最终差值的探索性配对 task-bootstrap 95% 区间为 **[-20.00, +40.00] pp**，
不能据此声称显著优势或已建立机制因果关系。

文件：`panel_d_cooperative_mean_sd.{png,pdf,svg}`。
成对展示全部类别：`category_mean_sd.{png,pdf,svg}`。

### 补充解释图：目标类型，n=12

每个任务内，先对该 predicate 的所有目标取平均，再对 12 个任务等权平均。
不把目标当作独立样本，不删除没有完成的目标。每个任务都同时有 placement/opening。
原始 predicate 只有 `is_on_top` 和 `is_open`，两类都完整导出。

最终所有 opening 目标均已完成，剩余未完成目标是物体放置目标。
这解释了**最终增益出现在放置方面**；NL 对 opening 更早完成，则解释了
整体曲线为何在中段领先。它不是“证据链接导致放置能力提升”的因果证明。

文件：`goal_type_mean_sd.{png,pdf,svg}`。标题或纵轴必须明确是目标类型分解。

## 2. 可用的定性案例

案例均为事后选择；单条轨迹不能加伪造的方差带。第一 seed 的有利案例：

| Episode | 案例 | NL vs SSC | 合适表达 | 限制 |
|---|---|---|---|---|
| 1166 | Guest reset | 都成功，13 vs 8 轮 | 此次配对中提前 5 轮完成 | 没有追加 seed 的配对结果 |
| 1781 | Tea reset | 最终 1/3 vs 3/3，SSC 第17轮完成 | 有实际错误放置后的纠正链 | 追加 seed 结果反转，见下 |
| 623 | Laundry reset | 最终 2/3 vs 3/3 | 完成了对照没完成的目标 | 追加 seed 两组均仅1/3 |
| 772 | Exhibit reset | 都在第10轮成功，SSC中途进度较高 | 较早完成部分目标 | 不能说总完成时间更短 |
| 588 | Inspection | 都成功，7 vs 6轮 | 提前1轮完成 | 差异很小 |
| 618 | Briefing | 都成功，19 vs 17轮 | 此次配对提前2轮 | 中间存在交叉 |
| 1526 | Courier | 最终2/3 vs3/3 | 最终完成更充分 | SSC时间平均反而更低，不适合“更快”故事 |

**推荐定性小图用 1166；如果要解释纠错行为，使用 1781 并标明 seed。**
1781 的实际事件链已核实：agent_1 送达 chair_11 目标信息；agent_2 误放水壶至
chair_10；agent_0 在第6–7轮重新拿起并放到 chair_11；第9轮茶壶归位，
第17轮抽屉打开。相关 SSC 输入保留说话人、时间、消息来源及物理状态。
NL 也收到并保留了正确目标信息，不能写成“NL遗忘/没有获取证据”。

现成案例图：`../habitat_memory_subgroups_20260923/illustrative_cases.{png,pdf,svg}`。
所有 12 个第一 seed 配对和 4 个追加 seed 配对（包括反例）在 `all_paired_cases.csv`。

## 3. 多 seed 核验：不能把缺失补成重复实验的方差

本次重新检查发现 seed 891013 已完成前 **4任务×2条件=8次**，然后按用户要求暂停。
这4个任务为1781、1013、623、744。相同4任务在两个 seed 的结果如下：

| Seed | NL最终完成度 | SSC最终完成度 | NL严格成功 | SSC严格成功 |
|---|---:|---:|---:|---:|
| 891011 | 68.75% | 85.42% | 1/4 | 2/4 |
| 891013 | 58.33% | 58.33% | 1/4 | 2/4 |

不能把第二行与第一 seed 全部12题比较。也不应把不完整追加 seed 拼进
原始12任务图，导致有些任务2个seed、有些任务1个seed。

| Episode | seed891011：NL / SSC完成度 | seed891013：NL / SSC完成度 |
|---|---|---|
| 1781 | 33.33% / 100% | 66.67% / 0% |
| 1013 | 75% / 75% | 100% / 100%（NL17轮，SSC10轮） |
| 623 | 66.67% / 100% | 33.33% / 33.33% |
| 744 | 100% / 66.67% | 33.33% / 100% |

这说明行为存在明显运行间波动。第一seed正向案例不是已证实的稳定机制证据。
另外 seed891012 的首个 NL 任务在第9轮规划时中断，物理评分只完整保存至第8轮，
没有最终结果；不补成20轮最终状态，
不把seed891013当作它的替代。其余未开始任务也不作为模型失败计分。
完整覆盖及中断记录在 `additional_seed_coverage.json`；额外8次运行逐轮评分
在 `additional_seed_measurements.json`，均核对了冻结配置哈希及日志一致性。

先前完成的108次 Direct/Static/Dynamic实验有3seed，但**没有匹配的动态NL条件**。
它可以画另一个三条件图，不能拿其中的Dynamic SSC与本轮NL拼成多seed公平对照。

## 4. 画图规范与建议图注

- 横轴第1–20轮；逐轮评分发生于所有主体行动之后。
- 纵轴实际目标完成度，不能累积最大化或把真实回退抹平。
- 同一图两个条件的任务集合完全一致，失败保留，真正提前终止才保持最终值。
- 主线是均值；误差带为跨任务样本标准差。方差数值单独导出，不把方差当作误差条。
- SD不是置信区间、不是多seed方差；图形在0–100%裁切误差带，原始上下界未改动。
- NL是同一可修订状态存储的自然语言读取形式：不能笼统称为所有自然语言memory。
- 共享更新器、planner、信息访问规则和配置预算；轨迹分岔后实际观察、消息及总成本
  可以不同。不是逐轮强制相同输入，也不是已经证明等总token成本。
- 所有子组均为事后探索；bootstrap区间未校正多重探索，不能用来宣布显著。

推荐英文图注：

> **Goal completion over interaction on cooperative tasks.** Curves show mean
> post-action physical goal completion across all five tasks originally labeled
> cooperative (seed 891011); shaded regions denote between-task sample standard
> deviation. Failed rollouts are retained, and terminal states are carried
> forward to round 20. Dynamic SSC is compared with a natural-language readout
> of the same revisable state store. This is an exploratory subgroup analysis;
> full-cohort and complementary-category results are reported separately.

## 5. 可交给画图同学的文件

- `panel_d_cooperative_mean_sd.{png,pdf,svg}`：可直接用的推荐子类panel。
- `category_mean_sd.{png,pdf,svg}`：cooperative与mixed并列，便于透明展示。
- `goal_type_mean_sd.{png,pdf,svg}`：全12任务放置/打开两类目标分解。
- `all_views_by_turn.csv`：每轮每条件的n、均值、方差、SD及原始误差带边界。
- `task_level_plot_values.csv`：每个任务每轮的画图值，可独立重算统计。
- `figure_options.csv`：所有视角的终值、时间平均、领先/落后轮次及配对区间。
- `all_paired_cases.csv`：所有完成配对的案例清单。
- `additional_seed_*.json`、`partial_seed_summary.json`：追加seed覆盖审计，不与主图混合。
- `provenance.json`：输入、冻结配置和脚本哈希。
- 原始全体图：`../habitat_memory_progress_stats_20260923/panel_d_mean_sd.*`。
- hard/standard图：`../habitat_memory_subgroups_20260923/difficulty_mean_sd.*`。

复现：

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/physical_code_mpl \
  MA-EgoQA/.venv/bin/python -m experiments.habitat_figure_options
PYTHONDONTWRITEBYTECODE=1 MA-EgoQA/.venv/bin/python -m unittest tests.test_habitat_figure_options
```

## GitHub 发布与离线复现

仓库：<https://github.com/glue25/Spatial_Social_Code>。
本目录以及同日期的 `habitat_memory_progress`、`habitat_memory_progress_stats`、
`habitat_memory_subgroups` 目录共同构成本次发布包；包含 PNG/PDF/SVG、逐轮数据、
正反案例和不完整追加 seed 的覆盖审计。

上面的原始采集/核验入口需要本地冻结实验目录。只克隆公开仓库时，用下面的
入口即可从公开的 reduced checkpoints 独立重算统计及重画所有图，无需原始
workdirs、API、密钥或模拟器（Python 依赖为 numpy；重画另需 matplotlib）：

```bash
python -m experiments.verify_habitat_figure_exports
MPLCONFIGDIR=/tmp/ssc-mpl python -m experiments.verify_habitat_figure_exports --render
```

这是公开评分记录的算术核验，不是重新运行模拟器验证物理真值。
