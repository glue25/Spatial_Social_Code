# 最新实验证据包（2026-09-14）

本目录是对**已有运行**的离线导出与复算；生成过程没有调用模型、没有启动 GPU，
也没有用新 scorer 改写历史结果。运行：

```bash
.venv-habitat/bin/python scripts/audit_latest_experiments.py
```

## 已能独立复算的内容

- `audit.json`：MMToM 204 个有效配对样本、MMToM compile-once 43 题、
  MA-EgoQA 强基线 20 题以及 Habitat 108 行指标的独立复算。
- `mmtom_option_blind/`：205 行原始记录（204 行有效、index 75 为 HTTP 429）和原 summary。
  保存的 Spatial/Social Code 中 `query_evidence`、`M.candidate`、`M.posterior`
  出现次数均为 0。当前文件没有 provider token usage，因此只能报告准确率与重试标志，
  不能报告真实 token 成本。
- `cross_model_pilots/`：GPT-5.6-sol 的 43 题 compile-once 开发 pilot 和
  MA-EgoQA 20 题冻结感知推理 baseline，含逐题结果、manifest、调用与 token 字段。
- `habitat_per_task.csv`：由 108 行原始 CSV 按任务和条件重新聚合。
- `habitat_1166_1566/`：两个异质性案例、三个条件、三个 seed，共 18 次历史运行。
  planner 可见日志放在 `agent_visible/`；`physical_checks` 和 scorer labels 单独放在
  `scorer_only/`，防止误当作 planner 输入。机器绝对路径已替换为 `$WORKSPACE`。
- `habitat_all_108/`：全部 108 次历史运行的同类证据。manifest/summary 可直接阅读，
  JSONL 使用确定性 gzip 压缩；目录名只编码行号、条件和 seed，完整 task ID 在 manifest 中。
- `manifest.json`：每个源文件与导出文件的 SHA-256。导出文件因为路径去标识化，
  hash 可以与源文件不同。

## EnactToM 接入

`enacttom_integration/ssc_closed_loop.patch` 是本地实际 EnactToM 工作树相对于官方
commit `093c4a11b622c8f0b50070e51ae9461b2b1db2f8` 的差异。它包含闭环 SSC hook，
也包含同一工作树中用于协调回放的工程改动，因此不是“只含论文方法”的最小 patch。
应用前应人工审查，并在干净 checkout 上运行 `apply_patch.sh /path/to/EnactTom`。

## 仍不能独立确认的内容

- MMToM 目前是 API 中断后的 204 个有效配对，不是完整 600 题，也没有 provider
  token usage；不能把它写成完整 benchmark。
- Habitat 108 行沿用历史 `strict_physical_v1` 汇总，没有静默用 v2 重评分。已找到
  108/108 个源 summary，`stability_advance_ok=false` 为 0 次，因此本次发现的 invalid
  语义缺陷不会改变这 108 行；未来运行将由 v2 明确拒绝无效测量。
- 人工核验仍未完成；runner 和空白表不等于人工实验结果。
- 环境实例化已有单例执行摘要和画面，但完整三次 rollout 原始包仍未纳入本目录。

## 本轮代码修复

- `StateStore` 现在把被成功 Pick/Place 失效的旧 `on/in/inside` 关系移入独立 history，
  当前 planner state 不再同时保留互斥位置。
- `strict_physical_v2` 在物理稳定窗口无法推进时输出 `measurement_valid=false`，并把
  `strict_success/completed_goals/strict_percent_complete` 置为 null；聚合器拒绝这类行。
- 108 个历史 summary 的稳定推进标志全部为 true，所以没有回写或重算历史 CSV。
- 本仓库没有 `.tex`/论文 PDF 源文件；README、中文方法稿、结果口径和补充材料已更新，
  外部论文工程仍需用同样口径同步。
