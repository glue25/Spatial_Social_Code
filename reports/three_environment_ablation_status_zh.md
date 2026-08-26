# 三环境消融状态

## MMToM-QA

- 600 题离线组件消融已完成：Spatial-only 74.0%，Social-only 75.0%，
  Spatial+Social executed 75.0%。
- 同一冻结 code 的 raw-code-only 与 executed-trace 对照已完成：71.33%
  对 73.00%，差值 +1.67 pp，McNemar `p=0.121`。
- 1,550-call 等调用预算 Direct 协议已经冻结；需使用主实验相同的 Gemini
  3-Pro endpoint，凭据可用后由后台队列自动续跑。

## MA-EgoQA

- 100 题、五条件 component ablation manifest 已冻结；903 个唯一视频 clip
  均存在，dry-run 与单元测试通过。
- 五个条件为 raw-only、raw+Spatial、raw+Social、raw+Spatial+Social、
  trace-only Spatial+Social，共 500 次终答调用。
- 该实验同样等待主实验的 Gemini endpoint 凭据。由于全部样本来自 test QA，
  结果只作为诊断消融，不用于调参。

## CoMind

- Direct/SSC 全量 train 推理、6,877 条两阶段 grounding、identity OOF gate、
  weak-field nested recording-CV 均已完成。
- 后台本地刷新会重新执行完整性审计、nested OOF 汇总和相关单元测试；不会
  重复消耗已经完成的 6,877 条 GPU grounding。
- 结果边界为 labeled-train recording-heldout，不是官方 hidden-test。

## 后台入口

```bash
bash scripts/run_three_environment_ablations.sh
bash scripts/run_pending_gemini_ablations.sh
```

第一个入口重算所有可离线复现的结果；第二个入口在 `credentials.env` 或
`GEMINI_CREDENTIALS_FILE` 指向的凭据可用后，自动续跑 MMToM 和 MA-EgoQA
的远端模型消融。
