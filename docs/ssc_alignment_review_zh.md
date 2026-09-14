# Spatial–Social Code 主线对齐审查

更新日期：2026-09-09。

## 结论

当前仓库已经具备 SSC 的三个必要部件：显式 Spatial/Social 状态、受限执行器、
主体索引的信息记录。但现有端到端证据尚不能完整支持“联合表示、可执行运算、
一次编译多问题复用”三项主张。最需要修正的不是再强化通信策略，而是把接口收紧为：

> `C_t = Compile(O_<=t)`，`V_q = View(C_t, q)`，`y/a = Execute(V_q)`。

问题和选项不得进入 `Compile`；通信只是可观察事件的一种，不能在 Social Code 中
直接写入任务特定的行动脚本。

## 对齐矩阵

| 核心主张 | 已有支持 | 当前不符合/缺口 | 判定 |
|---|---|---|---|
| 联合表示 | runtime 区分空间事实、观察、信念、目标、角色、未知与 provenance；EnactToM reducer 能从动作效果更新主体知识；本次已把方法主公式统一为 `Spatial + Social` | 旧实现字段和部分实验仍叫 `physical_code` / `interaction_code`；部分视觉 Social 状态由问题引导生成；Habitat Social 输入包含明确通信策略 | 部分支持 |
| 从空间到社会再到行动 | EnactToM 的同室观察投影、消息路由、权限检查和动作 replay 可执行；raw-code vs executed-trace 已有 +1.67 pp 但不显著 | 视觉任务主要依靠最终 LLM 阅读 trace；“主体观察投影”和“行动条件组合”此前没有统一公开接口；现有 intervention 只问显式给出的 knows/unknown | 部分支持 |
| 一次编译、多问题复用 | EnactToM oracle compiler 是 query-independent；新受控实验可从一个状态回答五类问题 | MMToM/MA-EgoQA 主视觉 pipeline 均为逐题编译；现有保存 trace 不能直接作为 episode-level reusable code | 主视觉结果不支持 |
| Spatial × Social 互补 | Qwen32B MMToM reasoning-only 100题中 SSC 56%，Spatial-only 35%；但输入 token 不匹配且属于已见开发数据 | MA-EgoQA 100题：P+S 50%，Spatial-only 51%；Habitat 新 12-task×3-seed 在线闭环中 Dynamic 完成度 41.67%，低于 Direct 53.01% | 尚未稳定成立 |
| 对自然语言结构基线的优势 | Qwen32B MMToM fair pilot：SSC 68.33%，structured prose 51.67%，120题，paired `p=0.00366` | Gemini 未完成 pilot 的前31题二者近乎持平（93.33% vs 93.55%）；该批已用于开发，且调用/token 不完全匹配 | 有开发信号，未独立确认 |

## 具体协议问题

### 1. 视觉 compiler 仍然看问题

- MMToM 的 `process_one` 先构造 `option_blind_question`，随后把它送入 Spatial prompt
  和 Social compiler。它是 option-blind，但不是 query-independent。
- MA-EgoQA 的 `build_question_prompt` 明确拼入 A–E 选项，而 Spatial/Social prompt
  都接收这一完整字符串。904/1741 因而不能被描述为 answer-choice-blind 编译结果。
- `docs/method_zh.md` 则要求 `Gamma(O,q)=Gamma(O)`。实现与方法定义不一致。

### 2. 保存 trace 没有通过复用审计

对 MMToM 保存结果按 episode 和 evidence manifest 分组：

| 结果文件 | 相同 episode 的多题组 | Spatial trace 变化 | Social trace 变化 | 完全相同 evidence 的多题组 | 两类 trace 均变化 |
|---|---:|---:|---:|---:|---:|
| dense_authoritative 600 | 110 | 110 | 110 | 7 | 7 |
| option_blind 当前196 | 42 | 42 | 42 | 3 | 3 |

同 episode 的不同题可能拥有不同时间截断，因此第一组统计不能单独证明违规；但在
完全相同 evidence manifest 的重复组中 trace 仍全部变化，结合源码中的问题输入，
说明当前 artifact 不能支持 strict compile-once reuse。该审计只识别不变量失败，
不把随机采样造成的变化与 query conditioning 混为一谈。

### 3. Social Code 与策略耦合过强

Habitat `build_task_conditioned_code.py` 的 Social 输出包含 `FIRST DECISION`、
`Wait`、`send ... directly`、`after your required message` 等策略指令。它们不是主体
状态，而是针对任务拓扑手工生成的通信/控制 policy。现有 Habitat 最强条件也是
Social-only，而非 Full，因此结果更像“协调提示有效”，尚不能证明 Spatial–Social
联合代码有效。

建议把它拆成：

- `SocialState`：证据、信息状态、目标、角色、承诺、消息预算；
- `ActionConstraints`：可达、权限、动作前置条件；
- `Planner`：读取两者后选择问、等、通知、导航或操作。Planner 的策略文本不得写回
  SocialState。

### 4. 现有最小对过于直接

`priority_ssc_v1/run_attribution_intervention.py` 把 `observation` 或
`social_unknown` 直接放进输入，再问对应主体是否知道，100% 主要验证模型能读取
字段。它没有让系统从位置、朝向、遮挡、消息接收者推导信息更新，也没有联合空间
权限决定行动。

## 本次补跑：受控 2×2×2 机制实验

新增 `spatial_social_code/ssc_operations.py`，显式实现三类操作：

1. `project_observation`：由主体位置、朝向和遮挡投影事件证据；
2. `apply_move` / `inform`：由观察或有效消息更新主体信息，未获新证据时保留旧位置；
3. `can_retrieve`：联合当前物体位置、主体信息、空间权限、目标与角色判断行动。

实验对 100 组实体重命名实例穷举 `可见 × 收到消息 × 可达` 八个条件，共 800 个
编译状态。每个状态只编译一次，再回答世界位置、B 的最后已知位置、B 的下一搜索
位置、B 是否可取物、合格执行者五类问题。

| B 可见移动 | B 收到消息 | B 可达目标房间 | 信息更新率 | B 可执行取物率 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0/1 | 0% | 0% |
| 0 | 1 | 0 | 100% | 0% |
| 0 | 1 | 1 | 100% | 100% |
| 1 | 0/1 | 0 | 100% | 0% |
| 1 | 0/1 | 1 | 100% | 100% |

完整运行得到：4000/4000 查询符合预期；查询前后 state digest 100% 不变；对移除
遮挡、补发消息、授予权限三类共300个反事实，target sensitivity 与 non-target
locality 同时满足率为100%。

这只是 executable semantics 的 conformance 结果，证明代码接口能够表达并执行
目标因果链；它不能证明视觉 compiler 能正确恢复这些变量，也不能证明模型使用
SSC 优于 matched natural language。

## 下一轮实验优先级

### P0：真正的 compile-once 多问题实验

冻结同一 episode 的时间窗 `O_<=t`，只调用一次 Spatial compiler 和一次 Social
compiler，保存不可变 `C_t`。随后针对同一 `C_t` 生成五类只读 query view。比较：

- Direct；
- Spatial-only；
- Social-only；
- Full SSC；
- 由同一 canonical facts 自动渲染的 matched natural language。

主要指标不是只有总准确率，还应报告 multi-query all-correct、每次编译支持的问题数、
状态一致性和 amortized calls/tokens per question。

### P0：学习模型上的 2×2×2 交叉实验

使用独立于当前 prompt 开发的新实体、新布局、新事件组合。先从结构化/渲染场景做
oracle-code 对照，再加入视频。至少报告三项交互：

- 可见或消息是否正确更新主体状态；
- 已知但不可达时是否拒绝直接执行；
- Full 是否在“必须同时用 Social 信息和 Spatial 权限”的 action query 上超过
  Spatial-only、Social-only 和 matched prose。

单位应是 minimal pair，指标用 pair-both-correct、target update、non-target
invariance；不要把单题 accuracy 当作组合性证据。

### P1：把通信从表示中降级为普通事件

删掉 Social Code 中的 `FIRST DECISION` 等动作建议。消息只产生
`inform(sender, recipient, proposition, evidence_ref)` 和预算/路由状态；下一动作由
统一 Planner 通过 `ActionConstraints` 推出。重新跑 Habitat 四条件，成功标准是
Full 超过两种单组件，而不只是超过 Direct。

### P1：真实 trace 的 counterfactual replay

在人工核验的真实片段上各选一条遮挡、消息接收者或权限事实，修改后从干预点重放，
同时评价 state-diff F1 和下游答案/动作。当前 human audit 尚未完成，不能用自动
生成的自洽 trace 代替真实 state correctness。

## 可报告边界

现在可以主张：项目有可执行的 Spatial→Social→Action 核心语义；EnactToM oracle
reducer 和本次 factorial conformance 支持确定性更新、局部反事实与复用能力。

现在不能主张：现有视觉主结果已经证明 compile-once；Spatial 与 Social 在所有环境
都有协同增益；或当前 Habitat 提升来自 SSC 联合表示而不是通信策略提示。
