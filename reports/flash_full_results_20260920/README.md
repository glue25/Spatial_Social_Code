# 完整结果与 SSC code：MMToM Pro/Flash + Habitat Flash

实验完成：2026-09-20 UTC；归档复核：2026-09-20 UTC。
这是**历史协议的开发集复现及完整文本产物归档**，不是官方隐藏测试榜单，也不是严格只改变模型、其余预算和服务商完全相同的实验。
本包覆盖完成的 MMToM 600 题和 Habitat 108 个 rollout；不将其他环境的未完成/旧诊断运行混入这两个最终结果。

## 1. MMToM 最终结果（失败保留在分母）

| 模型 | N | Direct | SSC | SSC − Direct |
|---|---:|---:|---:|---:|
| 历史 Gemini 3 Pro | 600 | 406/600 = 67.67% | 470/600 = 78.33% | +64 题，+10.67 pp |
| Gemini 3 Flash | 600 | 375/600 = 62.50% | 439/600 = 73.17% | +64 题，+10.67 pp |

Flash 请求及所有被接受的响应报告 `gemini-3-flash-preview`。模型身份依据服务商返回字段，而非独立验证的权重。
两个模型的样本 index、题目和答案逐题一致。Flash 从自己的新回复重新生成 Spatial/Social code 并执行历史推理流程，不是沿用 Pro 的生成结果。

### 为什么恰好都净增64题？

| 模型 | 两者都对 | 仅 SSC 对 | 仅 Direct 对 | 两者都错 | 净增 |
|---|---:|---:|---:|---:|---:|
| Pro | 322 | 148 | 84 | 46 | 148 − 84 = 64 |
| Flash | 301 | 138 | 74 | 87 | 138 − 74 = 64 |

这是不同配对结果的差值相同，不是相同预测表。599 个完成的 Flash 样本中，对应 Pro 的完整 `spatial_raw`、`spatial_code`、`social_code` **各自均有0个完全相同**。共享的确定性规则组件仍然是同一方法的一部分；不能以零完整字符串相同声称所有代码组件都独立不同。

额外进行了离线重放：从每题实际保存的回复出发，用冻结的原始实现重新生成/执行 code、规则/桥接/贝叶斯融合及条件密集帧细化，**600/600 与保存结果一致**（排除墙钟耗时字段）；其中 index44 重现原有 SSC 请求失败。检查了完整文本 prompt、调用顺序、历史阶段请求上限、实际16384上限和冻结帧哈希。该检查不是重新调用模型，也不是重新认证供应商权重。

### 分题型（分子均为答对数）

| 题型 | N | Pro Direct | Pro SSC | Flash Direct | Flash SSC |
|---|---:|---:|---:|---:|---:|
| 1.1 | 100 | 95 | 80 | 79 | 72 |
| 1.2 | 100 | 93 | 66 | 83 | 65 |
| 1.3 | 100 | 85 | 97 | 71 | 96 |
| 2.1 | 75 | 14 | 71 | 40 | 71 |
| 2.2 | 75 | 69 | 61 | 37 | 52 |
| 2.3 | 75 | 3 | 53 | 26 | 42 |
| 2.4 | 75 | 47 | 42 | 39 | 41 |

总体增益相同不代表每种题型趋势一致：1.1/1.2均下降，1.3/2.1/2.3均上升，2.2/2.4方向不同。Pro 最终绝对 SSC 分数仍高于 Flash。

### 失败、输入和比较边界

- 输入为**采样 RGB 帧 + 初始状态/动作/问题/选项文本**，不是纯视频或纯文本。初次8帧，原方法按不确定性触发16帧细化；本轮触发204题。
- index44 的 SSC 传输失败计错，不能丢弃。该题 Direct 回复在 SSC 失败前已经单独完成，故从自己的已保存 Direct 回复恢复得分；后来孤立重试的成功结果**没有替换**此题。
- 600题覆盖无重复、无缺号；599题完成SSC。2206份被接受的回复均为STOP，没有接受截断结果。
- 记录的传输异常为 ReadTimeout64、ConnectionError31、SSLError1；这是请求尝试次数，不是失败题数。
- 保留历史 question/option-conditioned SSC、规则证据、答案候选、query evidence、Bayesian posterior、原有语义重试及条件细化；**不是 question-independent clean SSC**。执行成功也不意味着所有生成事实/解释都语义正确。
- Pro 使用 JDCloud 和历史各阶段上限；Flash 使用 QDD、统一16384上限。前45题最多5次传输尝试，后555题最多10次，4个固定 key 分配互不重叠题目。扩大上限及传输策略对同轮 Direct/SSC 同样适用，但不同方法的总调用数不同，**不等总计算预算**。
- 先观察过 pilot/中间得分再扩展，不是 untouched confirmatory test。不能把结果写成严格控制服务商、预算、重试次数的“只换模型”因果证明。

## 2. Habitat Flash：完整108个rollout

12个任务 × 3个seed（891011/891012/891013）× 3种条件；每种条件36次。

| 条件 | 严格物理成功率 | 平均物理完成度 |
|---|---:|---:|
| Direct | 8/36 = 22.22% | 57.18% |
| Static SSC | 10/36 = 27.78% | 65.51% |
| Dynamic SSC | **24/36 = 66.67%** | **83.10%** |

Dynamic 相对 Direct：严格成功率 **+44.44 pp**，完成度 **+25.93 pp**。逐个核对108份底层 `summary.json → historical_benchmark_evaluation → strict_physical`，均为有效测量且与已有发布表一致。不是只依据顶层汇总表判断优势。

- 输入为**模拟器提供的私有文本观察**；v2状态表示，20环境turn，单调用8192上限，low reasoning。同一模型/服务的三个条件；不等总token预算。
- 这是本地开发套件的**严格物理评分优势**，不是声称所有字面ToM探针都正确。物理任务成功可能与字面心理探针成败不同。
- 12任务重复3seed不等于36个独立任务。没有确认过同协议的 Gemini3Pro Habitat 对照，故**不称为 Habitat 上 Pro/Flash 的互相印证**。
- 不改写或混同仓库已有 Qwen3-VL-8B 不同协议的历史结果。

## 3. 实际 SSC code 长什么样

### MMToM：生成的 Spatial/Social DSL

下面为 Flash index0（题型1.3）的**原样节选**，不是重新写出的理想示例。该题 Direct 错、SSC 对；选择它是为讲解，不是评测样本筛选。完整程序及正/负例结果见 [`examples/`](examples/)。负例为按索引选择的首个 Direct 对、SSC 错且非传输失败的题目。

```python
S.fact(0, "at", "jennifer", "bathroom", source="text")
S.fact(0, "inside", "plate", "cabinet_2", source="text")
S.fact(0, "inside", "plate", "fridge", source="text")
S.visible(1, "system", "jennifer", True, source="video")
```

```python
M.observe(1, 'Jennifer', 'is in the bathroom', source='video_or_action', confidence=1.0)
M.goal('Jennifer', 'get a plate', 0.9, 'Question context')
M.observe(2, 'Jennifer', 'is in the kitchen', source='video_or_action', confidence=1.0)
M.intent(2, 'Jennifer', 'search for a plate in the kitchen', 0.7, 'Jennifer enters the kitchen with the goal of getting a plate')
M.candidate('Jennifer', 'answer_belief', 'A', 'Jennifer thinks that the plate is inside the fridge', 0.5, 'MMToM answer option')
M.candidate('Jennifer', 'answer_belief', 'B', 'Jennifer thinks that the plate is not inside the fridge', 0.5, 'MMToM answer option')
```

完整 Social code 还包含 `M.query_evidence(...)`、规则/桥接证据和 `M.posterior(...)`。不能隐去这些部分，声称这里只是无选项条件的观察转录。可见性也不等于主体注意或信念的证明。

### Habitat：执行 Python updater + 持久化 JSON 状态

Habitat 不使用同一套模型生成的 `S/M` DSL；模型输出动作/计划，冻结 Python updater 根据各主体可用的事件和观察维护 Spatial/Social 状态。实际状态记录包括 key/value、主体来源、时间、状态、置信度和证据。例如：

```json
{
  "key": "access.denied_rooms",
  "value": ["kitchen_1"],
  "source_agent": "agent_0",
  "source_type": "task_private",
  "turn": 0,
  "status": "task_specified",
  "confidence": 1.0,
  "evidence": "task mechanic visible to self"
}
```

实际执行实现：[`updaters.py`](code/habitat_frozen_source/experiments/habitat_closed_loop/updaters.py)、[`state_store.py`](code/habitat_frozen_source/experiments/habitat_closed_loop/state_store.py)、[`runtime.py`](code/habitat_frozen_source/experiments/habitat_closed_loop/runtime.py)。每个rollout的完整 before/after 状态、事件、计划、预测和最终状态均已归档，而不只有一张汇总CSV。

## 4. 文件导航

| 路径 | 内容 |
|---|---|
| `mmtom/scores.csv` | 全600题Flash/Pro配对分数，失败标记、题型及来源 |
| `mmtom/{flash,pro}_results.jsonl.gz` | 两个模型各600条完整保存结果：生成code、trace、组件、细化结果及失败 |
| `mmtom/calls.jsonl.gz` | Flash每次逻辑调用的文本prompt、帧描述符、可见回复、尝试/异常日志 |
| `mmtom/run_metadata/` | 各批次/分片冻结配置、输入、媒体哈希 |
| `mmtom/original_frozen_inputs/` | 原始600题冻结输入及媒体哈希 |
| `habitat/rollouts.jsonl.gz` | 全108 rollout的summary、manifest、events、plans、usage、state_before_after、predictions |
| `habitat/scorer_only/checks.jsonl.gz` | 仅评分使用的物理检查和标签，明确与策略输入分开 |
| `habitat/api_responses.jsonl.gz` | Habitat保存的模型回复及usage |
| `habitat/published_summary/` | 原已公开108-rollout汇总、CSV及其验证器 |
| `code/workdirs/` | 冻结历史MMToM/相关依赖源码、实际Flash预算/传输/分片wrapper |
| `code/habitat_frozen_source/` | 本次Habitat真实冻结运行代码、任务、EnactToM依赖代码（无模拟器资产） |
| `audit/` | 600题离线重放报告、独立配对计数、原历史协议复放报告 |
| `provenance/source_files.json` | 原始文件SHA256与公开容器映射；汇集文件有多条来源映射 |
| `SHA256SUMS.json` | 发布文件哈希（不含哈希表自身） |

JSONL.GZ每行是一个独立JSON对象。MMToM calls按 `(index, call)`，Habitat rollouts按 `(task, condition, seed)` 索引。可用 Python `gzip.open(..., 'rt')` 逐行读取，无需下载图片或启动模拟器。

### 完整性与公开边界

“完整”指上述两套最终评估的**所有样本及保存的可见文本/code/状态产物**，不是包含API凭据、隐藏服务商续写令牌或原始媒体的机器镜像：

- 不上传密钥/私钥、图片/视频/frames、Habitat场景网格和模拟器数据；不跟随冻结source里的data符号链接。
- JSON内机器项目绝对路径替换为 `${PROJECT_ROOT}`，临时凭据位置替换为 `${CREDENTIAL_FILE}`；移除 `thoughtSignature`/续写token等opaque字段及认证字段，保留可见回复、finish/model/usage。源文件原哈希与导出文件哈希分开记录。
- 冻结Python/shell/YAML源码按原字节保存（含历史机器路径/环境变量名，**不含凭据值**）；它们是可审阅的原实现，不承诺所有旧启动器换目录即能运行。没有追认原哈希为脱敏后文件哈希。
- MMToM-QA基准文本/来源依照随附Apache2.0说明；EnactToM依赖保留MIT版权声明。真实重新采样/闭环重跑需要另行获得合法数据、模型服务、模拟器资产及原依赖环境。

## 5. 离线验证与重放

在本目录运行，基础验证仅依赖Python标准库：

```bash
python verify.py
```

它独立从每题答案/预测重算得分、配对净增、题型、code身份检查；验证600题/108rollout覆盖、底层严格物理得分、响应finish/model和发布哈希。它不是对历史方法的无偏性证明。

可移植的深层重放（不联网、不需要key/GPU/图片）：

```bash
# 在单独环境安装历史导入图用到的依赖后：
python -m pip install requests pillow numpy pandas rank-bm25 tqdm
PYTHONDONTWRITEBYTECODE=1 python replay_mmtom.py
```

`replay_mmtom.py`执行冻结方法，验证保存的完整prompt及回复到最终预测。为了不分发媒体，它替换**帧读取**为冻结帧选择/哈希，既不重新核对实际图像字节，也不独立重算官方采样。原本在有数据的本地完成的重放脚本另存于 `audit/original_local_replay.py`（历史路径，不是可移植入口）。两种重放均不是重新做模型推理。
