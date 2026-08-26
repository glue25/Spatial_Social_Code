# CoMind grounding 与 recording-heldout OOF 结果

## 结论

在 CoMind 标注 train split 的 55 个 recording 上，两阶段视觉 grounding
完成 6,877/6,877 条预测，0 个生成错误。与 Direct 和原 Spatial-Social
严格共同有效的比较集为 6,673 条。所有策略选择均按 recording 做五折
out-of-fold，held-out recording 的标签不参与该 fold 的拟合或选择。

| 字段 | Direct | 原 SSC | Grounding + OOF | 相对 SSC |
|---|---:|---:|---:|---:|
| Joint left bbox IoU@0.5 | 0.17% | 0.19% | **20.33%** | +20.14 pp |
| Joint right bbox IoU@0.5 | 0.11% | 0.19% | **21.27%** | +21.08 pp |
| SCOIA bbox IoU@0.5 | 0.21% | 0.10% | **6.47%** | +6.37 pp |
| Handover bbox IoU@0.5 | 0.34% | 0.00% | **23.15%** | +23.15 pp |
| Handover category L1 | 22.82% | 24.16% | **31.88%** | +7.72 pp |

最新 nested recording-CV 进一步为非 bbox 字段选择冻结 SSC、provenance
router 或训练折 prior。代表性结果为 Joint cue type 58.4%、SCOIA action
verb 39.2%、Handover initiator 69.5%。这些提升中有一部分来自 CoMind
训练集的类别不平衡，因此必须与视觉 reasoning 增益分开表述。

## 完整性

- Grounding manifest 与输出均为 6,877 个唯一 key。
- 76,608 张输入图像全部存在。
- 严格共同集：Joint Attention 5,402、SCOIA 973、Handover 298。
- 每个 recording 只属于一个 fold，fold 与固定 SHA-256 分配一致。
- manifest 不含标签字段，所有预测完成后才读取标签评分。

## 边界

这是 labeled-train recording-heldout 机制实验，不是官方 CoMind hidden-test
成绩，也没有进行 leaderboard 提交。它支持“显式 grounding 修复文本式 bbox
生成”的结论，但不能当作未知 recording 上的最终泛化结果。

机器可读汇总见 `comind_grounding_oof_summary.json`。
