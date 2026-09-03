# 三环境消融状态

## MMToM-QA

- 旧离线组件表受到缓存 posterior 污染，已作废。同口径重算（每组都有相同的
  读题/规则底座）为：底座 69.67%，+Physical 75.33%，+Social 69.17%，
  直接合并 P+S 75.00%。因此先前约 52% 的下降来自误删公共底座，不是修复后
  系统性能突然下降。
- 直接合并仍比 Physical 低 0.33 pp。五折 episode-heldout 的题型路由达到
  77.33%，比 Physical 高 2.00 pp；episode-cluster bootstrap 95% CI
  `[0.36, 3.72]` pp，McNemar `p=0.0169`。这是 labeled-data OOF 诊断，仍需在
  冻结的独立测试上确认。
- 去掉公共底座的裸证据诊断保留为次要结果：Physical 52.17%，Social
  49.67%，P+S 53.17%；不可与上述主消融直接比较。
- 同一冻结 code 的 raw-code-only 与 executed-trace 对照已完成：71.33%
  对 73.00%，差值 +1.67 pp，McNemar `p=0.121`。
- 1,550-call 等调用预算 Direct 协议已经冻结；需使用主实验相同的 Gemini
  3-Pro endpoint，凭据可用后由后台队列自动续跑。

## MA-EgoQA

- 100题强无代码 baseline 与组件矩阵均已完成。Raw-only 44.44%（99条可解析），
  option decomposition 43.00%，自然语言 scratchpad 51.00%，三调用自然语言
  pipeline 43.00%。
- 同一100题组件结果为 Spatial-only 51.00%、Social-only 48.00%、P+S 50.00%、
  executed trace-only 46.46%（99条可解析）。P+S 高于 Raw，但没有超过
  Spatial-only 或 scratchpad；差异均不显著。
- 全部样本来自 test QA，因此这些结果只作为诊断消融，不用于继续调参。

## MMToM SAM

- 同题、同8帧、同 Gemini 的冻结70题对照已完成：VLM-only 74.29%，VLM+SAM
  75.71%，净增1题；6个 SAM-only correct、5个 VLM-only correct，McNemar
  `p=1.0`。
- SAM terms 从所有选项对称抽取，覆盖率95.33%，overlay只提供给 Spatial compiler。
  结果支持“可插拔定位有小幅信号”，不支持“当前增益由SAM驱动”。

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
