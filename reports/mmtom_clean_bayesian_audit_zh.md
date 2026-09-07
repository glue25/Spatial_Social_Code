# MMToM：删除 `query_evidence` 后的清理与重算

更新日期：2026-09-07

## 结论

`query_evidence` 已从正式 SSC 运行路径移除。正常 Social runtime 会拒绝该调用；旧结果只能通过显式的 `legacy_*` 路径复现。

在 600 题冻结 Physical trace 上，结果如下：

| 条件 | 正确数 | 准确率 | 解释 |
|---|---:|---:|---|
| Direct | 406/600 | 67.67% | 原配对基线 |
| Legacy 保存结果 | 470/600 | 78.33% | 含答案定向启发式，撤出 clean SSC 主张 |
| 只删除 `query_evidence`、沿用旧 Bayes | 379/600 | 63.17% | 说明旧分数严重依赖该字段 |
| 新固定 Bayes + 旧 Social state | 525/600 | 87.50% | 旧 Social compiler 看过选项，只能作诊断 |
| 新固定 Bayes + 选项隐藏的规则 Social state | **508/600** | **84.67%** | 当前最严格的开发重算 |

最后一项相对 Direct 净增 102 题；paired McNemar exact `p=8.11e-12`。相对 legacy 保存结果净增 38 题，`p=6.27e-05`。

## 改了什么

- Social neural compiler 不再看到答案选项。
- 正式 `hybrid` 只生成 observation、belief、goal、intent 等状态；不再生成候选定向证据。
- 修复“准备打开”被误判成“已经打开”，并补齐第五至第八个柜子的序数归一化。
- 修复普通客厅 cabinet 与 numbered kitchen cabinet 被错误视为同一位置的问题。
- 新的固定答案层对所有选项使用同一组可解释项：显式 belief/goal 一致性、检查后未取走、下一待检查位置、belief 与行为计划冲突。
- 推理不读取 gold、question type、旧 prediction 或 baseline prediction。

## 不能过度声称的部分

508/600 是在已经查看过该 benchmark 的情况下开发出来的诊断结果，而且复用了旧实验的冻结 Physical trace，因此不能直接作为新的正式测试成绩。下一步应冻结代码和参数，在未参与开发的数据或预先固定的新划分上重跑；525/600 因旧 Social trace 当初看过选项，更不能作为正式结果。

机器可读结果见 `mmtom_clean_bayesian_rescore_summary.json`，逐题诊断由 `experiments/priority_ssc_v1/rescore_clean_bayesian.py` 生成。
