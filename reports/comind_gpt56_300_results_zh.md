# CoMind `gpt-5.6-sol` 300 例成对实验

## 结论

同一 `gpt-5.6-sol` 模型、同一组输入下，Direct 和 Spatial-Social 在冻结的 CoMind train 子集上完成了 300 例成对比较，每任务 100 例。两种条件各有 300 条有效预测，无失败或缺失。

Spatial-Social 在 14 个字段指标中 4 项更高、7 项更低、3 项持平，不能宣称整体优于 Direct。逐字段精确 McNemar 检验只有 handover category L1 显著，而且是 Spatial-Social 回归：0/100 对 Direct 14/100，`p=0.000122`。

## 结果

| 任务/字段 | Direct | Spatial-Social | 差值 | McNemar p |
|---|---:|---:|---:|---:|
| Joint attention left IoU@.5 | 0% | 0% | 0 pp | 1.000 |
| Joint attention right IoU@.5 | 0% | 0% | 0 pp | 1.000 |
| Joint attention cue type | 31% | 33% | +2 pp | 0.824 |
| Joint attention category L1 | 19% | 14% | -5 pp | 0.332 |
| SCOIA IoU@.5 | 13% | 12% | -1 pp | 1.000 |
| SCOIA cue type | 32% | 29% | -3 pp | 0.701 |
| SCOIA action verb | 10% | 9% | -1 pp | 1.000 |
| SCOIA action noun L1 | 30% | 26% | -4 pp | 0.388 |
| Handover IoU@.5 | 0% | 0% | 0 pp | 1.000 |
| Handover flow | 40% | 45% | +5 pp | 0.332 |
| Handover initiator | 38% | 43% | +5 pp | 0.473 |
| Handover initiation type | 36% | 29% | -7 pp | 0.281 |
| Handover category L1 | 14% | 0% | -14 pp | **0.000122** |
| Handover TTH ±0.25s | 8% | 11% | +3 pp | 0.607 |

这说明显式 Spatial/Social 中间表示在部分交互字段上有互补性，但没有解决定位、细粒度物体类别和未来动作识别。特别是 handover category L1 的显著退化值得优先检查：中间表示可能在从视觉证据到最终答案时丢失或改写了物体身份。

## 协议与边界

- 300 例通过 SHA-256 稳定排序与 recording round-robin 从 55 个 CoMind train recording 中确定性抽取，joint attention、SCOIA、handover 各 100 例。
- Direct 和 Spatial-Social 使用相同模型及相同图像、任务说明和预测帧前 10 秒转录。
- 抽样和生成提示不读取标签；预测全部生成后才读取 train 标签评分。
- 早期运行曾遇到 HTTP 429、CLI 状态目录和视觉流断连；可恢复 runner 最终于 2026-08-24 13:13:39 UTC 完成全部样本。
- 这是探索性 train 子集结果，不是 CoMind 隐藏 test 成绩；没有外部提交，也不替换仓库中的冻结主结果。
- 原始图像、数据和逐样本预测不公开；本仓库只发布聚合指标。

## 完整性

- Manifest SHA-256：`542c2f28230d417d1885789d953c781aab54d489f55599f4fe44405628d077ba`
- Prediction JSONL SHA-256：`a97ddb3293a50d0dd5051b2782a2f4fbfa59d5ccee7ab36f488b2ae5df3a5a14`
- 2026-08-25 独立重评分与保存的聚合结果完全一致。

机器可读版本见 [`comind_gpt56_300_summary.json`](comind_gpt56_300_summary.json)。
