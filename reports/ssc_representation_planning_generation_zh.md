# Spatial–Social Code：作用、闭环规划与环境生成潜力

更新日期：2026-09-14（UTC）

## 一句话结论

Spatial–Social Code（SSC）最合理的定位不是“另一种更长的提示词”，而是视觉、
语言、模拟器和 planner 之间的**带主体归属、证据来源和可更新语义的中间状态**。
Spatial Code 表示世界中什么对象在哪里、关系是否成立以及动作是否真正生效；
Social Code 表示谁能到达、谁观察到、谁收到消息、谁只是在转述以及谁负责执行。

已有 Habitat 实现证明这个表征可以进入真实闭环；在当前 12 任务、3 seed 开发套件
中，Dynamic SSC 的严格成功为 4/36、Direct 为 3/36，但 Dynamic 的平均完成度反而
低 11.34 个百分点且 token 多约 50.5%。因此，目前可以主张“闭环机制已经实现并可
审计”，不能主张“动态 SSC 已在 Habitat 上稳定提高总体性能”。

## 1. Spatial Code 与 Social Code 分别在哪里有用

### Spatial Code

Spatial Code 主要解决以下问题：

1. **对象身份与关系持久化**：记录对象 ID、房间、容器、支撑、抓取和开关状态，
   避免模型每轮重新从长上下文猜测。
2. **动作前提与效果**：区分 `attempted`、`failed` 和 `succeeded`。只有环境确认成功，
   才把 `object.on`、`object.in` 或 `is_open` 写成完成事实。
3. **时序与冲突覆盖**：新观察覆盖旧关系，同时保留 `turn`、`source`、`supersedes`
   和证据，便于回放状态为何变化。
4. **严格验收接口**：同一个关系可以被 planner 使用，也可以交给模拟器 scorer 检查，
   从而避免“语言上说完成了”等于“物理上完成了”。
5. **场景生成约束**：对象类型、房间、支撑、容器和关节状态可以映射为资产选择、
   布局约束和初始物理状态。

### Social Code

Social Code 主要解决以下问题：

1. **信息归属**：同一个事实对不同 agent 可以是 observed、reported 或 unknown；
   消息内容不自动升级为世界真值。
2. **可见性与可达性**：记录谁在场、谁能进入哪个房间、谁能直接操作哪个目标。
3. **通信拓扑与预算**：只有环境确认送达的消息才更新接收者；被阻止的消息不会
   凭空进入另一主体状态。
4. **任务分工**：把物理目标分给有权限、有信息且仍有通信能力的主体。
5. **反事实干预**：保持世界终态不变，仅改变观察者、接收者或在场性，即可检验
   模型是否真的使用“谁知道什么”。

### 二者的本质连接

一个物理事件不会无条件成为所有人的社会知识。对 agent `i`，更新必须经过：

```text
真实物理事件 e_t
  -> i 是否亲自观察到 / 听到 / 收到已确认消息？
  -> 是：写入 i 的私有 Spatial/Social 状态，并标注来源
  -> 否：i 保持 unknown，而不是共享全知状态
```

因此，Spatial Code 描述“世界是什么”，Social Code 描述“这个主体凭什么知道”。
这正是 SSC 与普通 scene graph、自然语言 scratchpad 的关键区别。

## 2. Habitat 闭环 planning 的真实完成状态

完整在线循环已经实现：

```text
主体私有观察 o_t^i
  -> SpatialUpdater / SocialUpdater
  -> 版本化私有状态 C_t^i
  -> 注入该主体下一轮 planner prompt
  -> 执行动作 a_t^i
  -> 环境返回成功、失败及已确认消息送达
  -> 更新 C_(t+1)^i 后重新规划
```

工程上已经具备：

- 每个 agent 隔离的 Spatial / Social 状态与版本号；
- 逐轮真实 prompt 注入，日志记录 planner 实际读到的版本；
- 成功/失败动作分开更新，失败不会写成完成；
- 发送、送达和转述分开记录；
- 真实 RGB 帧、动作、thought、状态前后快照和 token/call 使用量；
- `strict_physical_v1` 三采样稳定窗口，放置还要求对象已脱手；
- 验收器检查 Dynamic 是否在线更新、Static 是否未修改、Direct 是否未收到 SSC，
  以及 scorer-only 标签是否泄漏。

修复后的 v3 工程回归结果如下：

| 条件 | 严格成功 | benchmark 成功 | 平均 API calls | 平均总 token |
|---|---:|---:|---:|---:|
| Direct | 1/12 | 1/12 | 57.67 | 150,273 |
| Static SSC | 1/12 | 1/12 | 56.50 | 166,481 |
| Dynamic SSC | 1/12 | 2/12 | 55.08 | 221,279 |

Dynamic 的第二个 benchmark success 没通过严格物理 scorer，所以正文应采用 1/12。
这组结果的状态是 `engineering_regression_one_seed`，不是冻结后的正式多 seed 结论。

随后完成的三 seed 扩展包含 `891011/891012/891013`，共 12 任务 × 3 条件 ×
3 seeds = 108 个真实 rollout；每个运行均通过工程验收：

| 条件 | 严格成功 | 平均严格完成度 | benchmark 成功 | 平均总 token |
|---|---:|---:|---:|---:|
| Direct | 3/36（8.3%） | 53.0% | 8.3% | 147,432 |
| Static SSC | 3/36（8.3%） | 47.2% | 8.3% | 168,007 |
| Dynamic SSC | 4/36（11.1%） | 41.7% | 13.9% | 221,910 |

Dynamic SSC 与 Direct 的配对结果为 Dynamic-only 4、Direct-only 3，exact McNemar
`p=1.0`。因此这组扩展仍不能证明 Dynamic SSC 提升 Habitat 总体成功率；同时其平均
完成度更低、token 更高。可信结论仍然是“闭环机制已完成并存在成功案例”，而不是
“动态 Code 已经带来稳定总体收益”。完整结果见
[三 seed 汇总](habitat_closed_loop_multiseed/habitat_multiseed_results.md)。

### 可审计机制案例：episode 1166

同一模型 `qwen3-vl-8b-instruct`、同一 seed `891011` 下：

| 条件 | 严格物理完成 | 结束轮数 |
|---|---:|---:|
| Direct | 1/3 | 20 |
| Static SSC | 1/3 | 20 |
| Dynamic SSC | 3/3 | 7 |

Dynamic SSC 的 `vase_0 ON table_15`、`box_3 ON table_12` 和 `fridge_34 OPEN`
均在最终连续三个物理采样点成立。该例能展示状态更新和重规划如何工作，但单例不能
证明平均性能或效率提升。

- [机制可视化 PNG](cases/habitat_ep1166_dynamic_ssc_visualization.png)
- [机器可读证据包 JSON](cases/habitat_ep1166_dynamic_ssc_visualization.json)
- [可复现生成脚本](../experiments/habitat_closed_loop/visualize_episode1166.py)
- [SSC 编译出的资产化环境规范](cases/habitat_ep1166_ssc_generated_environment.json)
- [环境编译与泄漏检查脚本](../experiments/habitat_closed_loop/compile_ssc_environment.py)

环境规范已经解析并校验真实 HSSD stage GLB、5 个刚体资产配置、agent 出生点、
房间权限、通信图、初始物理关系、功能目标和三组反事实 case。它复用已有资产，
不声称生成了新的 3D 网格；语义对象与底层 asset instance 的最终绑定仍交给 Habitat
runtime，编译器没有猜测隐藏 handle。

## 3. Code 作为环境生成与 Physical AI 表征

SSC 可以形成一个双向接口：

```text
视频 / 图像 / 传感器
        | 逆图形学：识别资产、几何、关系、状态和主体观测
        v
Spatial–Social Code
        | 前向生成：资产检索、布局、权限、私有知识、通信图、目标与 scorer
        v
Habitat / Isaac / 其他具身模拟环境
        | agent 规划、尝试、失败恢复与反事实干预
        v
新观察 -> 更新 SSC -> 再规划
```

具体映射可以定义为：

| SSC 字段 | 环境生成作用 |
|---|---|
| object / type / asset hint | 从资产库检索带碰撞体和语义标签的 3D 资产 |
| room / pose / on / in | 生成房间、位姿、支撑和容器约束 |
| open / held / movable | 初始化关节、抓取和可操作属性 |
| agents / visibility | 生成 agent、相机视角和私有观测 |
| access / presence | 配置房间权限、出生点和可达区域 |
| knows / reported / unknown | 生成每个 agent 的私有上下文，而非全知状态 |
| communication graph / budget | 配置消息边、接收者和成本 |
| goals / invariants | 生成任务、反事实变体和严格 scorer |

这可以称为“逆图形学 + 前向世界生成”的统一中间表征。现阶段不能写“GPT-6 已经
能做到”，因为没有可核验的 GPT-6 系统或实验。更严谨的表述是：**未来更强的多模态
生成模型可以把 SSC 作为结构化接口，用于生成带资产、物理约束和多主体信息边界的
可交互环境。**

## 4. 环境实例化已经完成什么、还缺什么

当前已经完成一个 `SSC-conditioned executable environment instantiation`：从
episode 1166 的资产链接 SSC 生成新 episode `911166`，在已有 HSSD 房间壳中实例化
5 个刚体资产，并对一个资产施加显式位姿扰动。三条件执行摘要为 Direct 0/3、Static
SSC 1/3、Dynamic SSC 3/3；这些是上游保存的单例执行证据，本轮只复核了编译输出和
文件哈希，没有重新进行 GPU 物理验收。

因此“SSC 到可执行环境的前向实例化”已经完成，不再只是未来工作；尚未完成的是
“从任意视频自动恢复三维资产和几何并生成新世界”的逆图形学链路。

扩展成多场景验证时应形成下面的闭环：

1. 从一个真实 Habitat episode 导出 answer-blind SSC；
2. 用 Spatial 字段检索资产并重建一个小型房间，使用物理约束而非固定像素位置；
3. 用 Social 字段生成两到三个 agent 的权限、私有事实和通信拓扑；
4. 自动生成三类问题：物理关系、信息归属、联合规划；
5. 对观察者/接收者做最小干预，同时保持物理终态不变；
6. 在重建环境中执行 planner，使用模拟器状态而非语言 judge 验收；
7. 把新观测重新编译为 SSC，检查 round-trip consistency。

至少应报告：资产召回、关系满足率、可达率、状态 round-trip F1、规划成功率，及
信息归属干预前后的答案翻转率。当前单例已经完成前向编译、实例化、可视化和三条件
执行摘要；但尚未完成“从任意视频自动生成一个新资产化 3D 环境”的端到端实验，
论文中必须把后者写成未来方向或待验证扩展。

## 5. 补充材料的完整自洽清单

补充材料不能只放几个成功案例，应按一条可审计证据链组织：

1. **定义**：完整 DSL/schema、unknown 语义、source/status/confidence 和覆盖规则；
2. **输入边界**：视觉证据、帧选择、是否冻结感知、禁止目标答案和候选泄漏；
3. **编译**：完整 prompt、模型版本、temperature、seed、重试规则和输出解析；
4. **闭环算法**：状态—动作—反馈—消息送达—状态更新—重规划伪代码；
5. **基线公平性**：同输入、同 backbone、同调用上限；明确 Static 与 Dynamic；
6. **严格评分**：物理关系、释放条件、稳定窗口以及 benchmark/scorer 分歧；
7. **完整结果**：成功率、置信区间、配对检验、token、calls、重试和延迟；
8. **案例**：至少一个成功、一个失败、一个 scorer 分歧，均含原帧、Code、动作和状态；
9. **泄漏审计**：Code 是否含答案、选项、query evidence 或 scorer-only 标签；
10. **复现**：命令、依赖、资产来源、文件 SHA-256、原始输出与导出脚本。

### 论文主张边界

- 可以写：MMToM 204 个有效配对的部分运行显示积极信号；Habitat 在线状态已经真实
  接入 planner，且存在严格成功的机制案例。
- 暂时不能写：Dynamic SSC 已在 Habitat 总体显著胜出，或 SSC 已自动从视频生成
  新的资产化 3D 世界。
- “GPT-6/未来模型”只能出现在 potential / future work 中，不能作为实验事实。
