# Thinking with Spatial–Social Code

## 摘要

现有多模态模型可以描述多主体视频中“发生了什么”，却常常不能稳定回答“谁获得了哪条信息”“谁应当执行下一步”“如果某条消息未被接收，后续会如何改变”。我们认为，这类失败不应被笼统归因于模型缺少社会常识，而应被表述为一个更基础的表示问题：**多主体物理世界中的推理是否缺少一种显式、主体索引的中间状态？**

受到 Thinking with Spatial Code [1] 的启发，本文提出 Spatial–Social Code。Spatial Code 将视频编译为显式空间变量，使语言模型能够在结构化空间上推理；Spatial–Social Code 将同一思想扩展到多主体场景，把中间表示定义为共享 Spatial Code 与 agent-indexed Social Code 的组合：

> Cₜ = (Cₜsp, Cₜsoc).

其中，Cₜsp 描述实体、物理属性、空间关系与信息接入条件；Cₜsoc 描述主体获得的证据、操作性信息状态、目标、角色、承诺与近期交互。核心不是在 Spatial 后附加心理描述，而是执行 `空间事件 → 主体证据 → 主体状态 → 可行动作`。论文不试图建立完整 Social World Model，也不声称读取人的真实心理，而是检验一个可证伪的 intermediate representation hypothesis：在相同信息源、相同推理器和近似 token budget 下，两类代码的显式组合能否优于视频、caption、自由文本 memory、Spatial-only 和 Social-only 表示。

我们进一步提出三类互补实验：在 MA-EgoQA 上检验主体索引是否减少 perspective leakage；在 HoloAssist 上检验代码是否改善真实协作中的 instruction、intervention 与 next-action reasoning；在 EnactToM 上检验局部信息、角色或承诺干预能否产生正确且局部的行为变化。Oracle、semi-oracle 与 predicted code 分开报告，从而区分表示上限、compiler 误差与 reasoner 误差。

**关键词：** 多主体视频推理；中间表示；Spatial–Social Code；主体索引状态；协作推理；反事实

<!-- FIGURE:pipeline -->

<!-- PAGEBREAK -->

## 1. 背后的科学问题

### 1.1 Spatial Code 真正提出了什么问题

Thinking with Spatial Code [1] 的贡献不只是使用三维检测器，也不是简单地把视频转成 JSON。它提出了一个具有一般性的研究判断：

> 模型能够看到空间，不等于模型拥有适合空间推理的变量。

直接视频问答通常近似条件分布：

> p(y | V, q),

其中 V 是视频，q 是问题，y 是答案。像素同时包含外观、背景、视角、遮挡与运动等信息，但距离、方向、朝向和持续物体身份并没有以稳定变量显式出现。因此，模型可能依赖二维外观或 viewer-centric shortcut，而不是恢复任务真正需要的三维结构。

Spatial Code 引入中间变量 c，将问题重写为：

> p(y | V, q) = ∫ pθ(y | c, q) pφ(c | V) dc.

实际系统采用 perception-then-reasoning 近似，先产生最可能的 code：

> c* ≈ fφ(V),

再让 reasoner 计算：

> ŷ ≈ Rθ(q, c*).

这一分解有三个科学价值。第一，它明确规定了任务所需的中间变量；第二，它把 perception error 与 reasoning error 分离；第三，它允许使用 ground-truth code 测量表示上限。参考论文中，同一推理器使用 GT spatial code 与 predicted spatial code 时存在明显差距，这说明 representation quality 会成为推理性能的上界之一 [1]。

需要准确说明：参考论文的主要系统包含经过监督训练的 Spatial Encoder，并进一步研究了带 spatial-rubric reward 的强化学习；它不是一个所有组件均“从未训练”的纯 training-free 系统。本文因此使用“主实验不进行目标任务参数更新”这一更严格表述，而不把 training-free 当作标题性贡献。

### 1.2 多主体世界的缺失变量

多主体视频包含 Spatial Code 所处理的物体、位置、方向与运动，但还存在另一类不能被单一共享几何状态表达的变量：

- 不同主体可能接触到不同证据；
- 不同主体承担不同角色与子目标；
- 请求、确认、承诺和纠错会约束后续行动；
- 同一物理动作在不同交互上下文中具有不同功能；
- 一个主体缺席时，世界已经改变，而其可用信息没有同步更新。

考虑两个情境：

| 情境 | 当前物理状态 | 先前交互 | Bob 的合理下一步 |
|---|---|---|---|
| A | Alice、Bob 同时握住桌子 | Alice 请求帮助，Bob 已确认目标位置 | 与 Alice 协同搬运 |
| B | Alice、Bob 同时握住桌子 | 没有请求，Bob 未获得目标位置 | 询问或等待 |

两个情境在当前帧上可以近似相同，但未来行为不同。差异不来自物体位置，而来自 interaction state。

我们把这一现象称为 **social-state aliasing**：

> 存在两个 episode e 与 e′，其当前共享物理投影相同 P(e) = P(e′)，但由于主体的目标、角色、信息或承诺不同 I(e) ≠ I(e′)，正确答案或行动也不同 y*(e) ≠ y*(e′)。

如果输入表示只保留 Pₜ，那么任何只依赖该表示的策略在这种 minimal pair 上都无法同时正确。由此得到本文的核心科学问题：

> **多主体推理是否需要把物理世界及其对不同主体的影响，编译成可执行、主体索引的 Spatial–Social Code？**

### 1.3 为什么问题不是“模型有没有社会常识”

“缺少社会常识”很难证伪，因为任何错误都可以在事后被解释为缺少某种常识。本文把问题缩小到可观察、可干预的结构：

1. 谁在场、谁能看到、谁收到定向消息；
2. 当前共同任务是什么，谁承担哪个角色；
3. 哪条信息由谁观察或被告知；
4. 谁对哪个动作作出可观察承诺；
5. 哪个 interaction event 刚刚发生。

这些变量不覆盖人格、情绪、长期关系或无限阶共同知识。它们的价值可以通过受控删除、共享化或反事实替换直接检验。

### 1.4 中心命题

本文的中心命题不是“我们构建了完整 Social World Model”，而是：

> **Physical-social reasoning requires an explicit representation of not only what exists in the world, but also how agents are situated within it.**

中文表述为：

> **物理-社会推理需要一种显式表示，不仅描述世界中存在什么，还描述主体如何通过位置、信息接入、目标、角色与承诺嵌入这个世界。**

<!-- PAGEBREAK -->

## 2. 问题定义

### 2.1 输入与任务

场景包含主体集合 A = {1, …, N}。主体 i 在时间 t 前的观察为：

> O≤tⁱ = (V≤tⁱ, A≤tⁱ, Z≤tⁱ),

其中 V 表示视频，A 表示音频，Z 表示可选 gaze、深度、位姿、手部或消息通道。外部固定摄像机和公共通道记为 U≤t，总观察为：

> O≤t = {O≤t¹, …, O≤tᴺ, U≤t}.

本文考虑三类输出。

1. **Perspective reasoning：** 谁接触了某条信息，谁能够回答某个任务相关问题？
2. **Coordination reasoning：** 在当前物理状态、目标、角色和承诺下，谁应执行下一动作或发出哪条消息？
3. **Counterfactual reasoning：** 如果某个主体未观察、未收到、未确认或角色被改变，后续状态与行动应如何变化？

问答输出记为 y，功能行动记为 aₜ₊₁ⁱ。

### 2.2 环境真值、模型代码与主体状态

必须区分三个概念：

- 环境真值 Wₜ*：评价阶段可用的真实物理与任务状态；
- 编译器代码 Cₜ：系统从观察中估计的中间表示；
- 主体相关信息 Iₜⁱ：代码中由该主体可接入证据支持的操作性状态。

因此：

> Wₜ* ≠ Cₜ ≠ Iₜⁱ.

模型不能在推理时读取 Wₜ*。当数据集不提供真实心智标签时，系统只能表示“观察到”“被告知”“没有观察证据”或“系统未知”，不能把这些字段升级为对人类内心的事实断言。

### 2.3 直接推理与代码推理

直接多模态推理近似：

> p(y | O≤t, q).

本文引入中间代码 C：

> p(y | O≤t, q) ≈ ∫ pθ(y | C, q) pφ(C | O≤t) dC.

实际采用 MAP 风格的两阶段近似：

> Cₜ* = arg maxC pφ(C | O≤t) ≈ Γφ(O≤t),

> ŷ = Rθ(q, Viewq(Cₜ*)).

对于行动：

> âₜ₊₁ⁱ = πθ(gⁱ, oₜⁱ, Viewi(Cₜ*)).

其中 Γφ 是 compiler，View 是只读视图构造器，Rθ 或 πθ 是冻结或单独报告训练状态的 reasoner。

### 2.4 Query independence

如果 code 在看到问题后才生成，它可能只是针对答案临时编写的解释，而不是世界状态。主实验要求：

> Γφ(O≤t, q) = Γφ(O≤t).

问题 q 只能影响从完整代码中读取哪些字段，不能改变历史事实、主体信息归属或状态更新。该约束也是 Code 与 query-conditioned caption 的关键区别。

<!-- FIGURE:aliasing -->

<!-- PAGEBREAK -->

## 3. Spatial Social Code

### 3.1 二分表示

本文把论文表面的代码压缩为两个部分：

> **Cₜ = (Cₜsp, Cₜsoc).**

Cₜsp 是共享 Spatial Code；Cₜsoc = {Cₜsoc,1, …, Cₜsoc,N, Cₜsoc,group} 是主体索引与群体级 Social Code。旧实现中的 `physical_code` / `interaction_code` 仅视为兼容字段名，论文、图表与新实验统一使用 Spatial / Social。二分表示控制了方法粒度：它足以检验缺失中间表示假设，但不会膨胀为完整认知架构。

<!-- FIGURE:code -->

### 3.2 Spatial Code

定义：

> Pₜ = (Entₜ, Attrₜphy, Relₜspatial, Relₜaccess).

各部分含义如下。

| 组成 | 内容 | 示例 |
|---|---|---|
| Entₜ | 持久主体与物体身份 | Alice, Bob, knife_3 |
| Attrₜphy | 位置、朝向、持有、开闭等属性 | location=drawer, open=true |
| Relₜspatial | inside、near、holding、contact 等关系 | inside(knife, drawer) |
| Relₜaccess | 由在场、视角、遮挡、音频或消息路由支持的接入关系 | cannot_observe(Bob, drawer) |

Pₜ 不要求每个数据集都有完整三维重建。具有相机位姿与深度时可使用 3D position、size 与 orientation；只有 caption 或二维标注时，可以退化为离散位置和关系。2D、3D 与 annotation-only 设置必须分开报告。

### 3.3 Social Code

定义：

> Iₜ = (Gₜ, Roleₜ, Kₜ¹:ᴺ, Mₜ, Eₜ₋w:ₜint).

| 组成 | 内容 | 论文中回答的问题 |
|---|---|---|
| Gₜ | 个体与群体目标、当前任务步骤 | 正在完成什么？ |
| Roleₜ | leader、helper、performer、instructor 等任务角色 | 谁负责什么？ |
| Kₜⁱ | 由证据支持的主体特定任务信息 | 谁获得了什么信息？ |
| Mₜ | request、promise、assignment 的可观察承诺状态 | 谁已同意做什么？ |
| Eₜ₋w:ₜint | 近期 request、inform、acknowledge、correct 等事件 | 刚才发生了什么交互？ |

这里 K 只是 operational information state。一个命题 φ 的条目可以写为：

> Kₜⁱ(φ) = (value, support, evidence_ref, confidence),

其中 support ∈ {observed, informed, inferred, no-observed-access, unknown}。

`no-observed-access` 只表示系统没有发现主体接触该更新的路径，不等价于“主体确定不知道”。只有 benchmark 提供 oracle 标签时，才可评价真正的 false belief。

### 3.4 为什么保留 Commitment

Goal 和 Role 说明主体可能做什么；Commitment 说明某个可观察交互是否已经对后续协调形成约束。Gift 与 Loan、请求已确认与请求未接收、任务已分配与尚未分配，都可能有相同物理状态和不同正确行动。

因此，Commitment 是构造 social-state aliasing minimal pair 的最小必要字段之一。第一版只建模可观察的生命周期：

> proposed → received → acknowledged → fulfilled / violated / cancelled.

这不是无限阶 common ground，也不是法律或道德判断引擎。

### 3.5 物理 grounding

Interaction Code 不能脱离物理与通信通道凭空生成。对事件 eₖ、主体 i 和模态 m，定义接入分数：

> xᵢₖᵐ = gₘ(poseᵢ, field-of-view, occlusion, distance, noise, addressee, route).

它表示主体 i 有机会接触事件证据的程度，而不是已经理解或相信事件。信息链条应保持：

> signal transmission ≠ sensor reception ≠ attention ≠ interpretation ≠ belief.

本文直接建模前两到三层可观察环节，并将高层心理解释限制为数据集可支持的 operational label。

### 3.6 代码实例

```yaml
Physical:
  objects:
    knife_3:
      location: drawer_1
      state: clean
  agents:
    Alice:
      location: kitchen
    Bob:
      location: living_room
  relations:
    - inside(knife_3, drawer_1)
    - near(Alice, drawer_1)
    - cannot_observe(Bob, drawer_1)

Interaction:
  goals:
    group: make_salad
  roles:
    Alice: cook
    Bob: helper
  information:
    Alice:
      knife_location:
        value: drawer_1
        support: observed
        evidence_ref: e_018
    Bob:
      knife_location:
        support: no-observed-access
  commitments:
    - agent: Bob
      content: bring(knife_3, Alice)
      status: acknowledged
      evidence_ref: e_027
  recent_events:
    - e_026: Alice requested knife_3 from Bob
    - e_027: Bob acknowledged the request
```

<!-- PAGEBREAK -->

## 4. Social Code Compiler

### 4.1 论文接口与内部实现

论文接口只输出 Cₜ = (Pₜ, Iₜ)。为保证代码可审计、可增量更新和可做反事实，compiler 内部可以维护不可变事件日志 H≤t 与物化状态 Zₜ：

> H≤t = (e₁, e₂, …, eK),

> Zₜ = U(Zₜ₋, eK),

> Cₜ = Π(Zₜ) = (Pₜ, Iₜ).

Zₜ 可以包含 registry、access score、provenance ledger、task graph 与 commitment lifecycle。它们是 compiler 的工程分解，不是七个并列的论文贡献。

### 4.2 Grounded event

每个事件使用固定 schema：

> eₖ = (time, type, actor, recipient, object, content, channel, evidence, confidence).

第一版事件 ontology 保持小而可验证：

| 类别 | 事件 |
|---|---|
| 物理动作 | pick, place, move, open, close, handover, enter, leave |
| 注意与指示 | gaze_at, point_at, show |
| 沟通行为 | inform, request, instruct, ask, acknowledge, deny, correct |
| 任务进度 | assign, start, complete, fail, block, intervene |

无法可靠 grounding 的槽位必须输出 `unknown`，不得根据剧情合理性自动补全。

### 4.3 三个可执行核心运算

1. **主体观察投影** `ProjectObserve(Csp, event, agent)`：联合位置、朝向、遮挡、距离与通道，产生该主体可获得的 evidence；
2. **主体状态更新** `UpdateSocial(Csoc,i, evidence/message)`：只更新有有效接入路径的主体，未收到新证据时保留旧值或 unknown；
3. **行动条件组合** `ComposeAction(Csp, Csoc,i, goal, role)`：联合当前位置、可达性、主体信息、目标与角色生成或验证动作。

通信只是第二个运算可接收的一种事件，不是独立的策略模块。`Social Code` 只保存状态与证据，`Planner` 才负责决定发送、等待或执行什么动作。

确定性 reducer 采用以下更新顺序：

建议采用以下更新顺序：

1. 验证事件 schema、实体 ID、时间与通道；
2. 更新物理属性和空间关系；
3. 根据在场、视角和消息路由计算 access；
4. 只给具有证据接入的主体写入 information entry；
5. 更新 goal、role、commitment 和 recent events；
6. 投影为紧凑 Cₜ；
7. 保存 state diff 与 provenance。

同一事件序列必须产生相同状态；新证据不得静默删除旧证据；冲突信息必须保留来源；无关事件不得改写无关主体。

### 4.4 Counterfactual replay

对局部干预 do(eⱼ ← eⱼ′)，从干预前状态重放：

> Z′T = Replay(Zⱼ₋₁, eⱼ′, eⱼ₊₁, …, eT).

评价需要同时满足：

- Target sensitivity：干预目标字段按预期改变；
- Non-target invariance：无关主体、物体和任务分支保持不变。

只做到“答案改变”是不够的；系统必须改变正确的状态，并保持不应变化的部分。

### 4.5 Oracle、semi-oracle 与 predicted code

| 轨道 | Code 来源 | 回答的问题 |
|---|---|---|
| Oracle | gold annotation、PDDL、gold trajectory | 表示上限是否存在？ |
| Semi-oracle | 部分 gold identity/event，部分自动预测 | 哪个 compiler 部分是瓶颈？ |
| Predicted | 从允许的原始视频、音频或 caption 自动编译 | 完整系统是否可用？ |

参考 Spatial Code 的关键实验逻辑，必须比较 GT code 与 predicted code。二者差距不是失败，而是对 representation ceiling 与 perception/compiler gap 的定量分解。

## 5. 可证伪假设

令 Acc(X) 表示同一冻结 reasoner 在输入表示 X 下的任务性能，所有比较使用相同原始信息和近似 token budget。

### H1：显式状态优于叙事压缩

> Acc(Pₜ, Iₜ) > Acc(free-form memory).

如果信息匹配的 memory 与 code 表现相同，则不能主张结构化中间状态具有独立价值。

### H2：Physical 与 Interaction 具有互补性

> Acc(Pₜ, Iₜ) > max{Acc(Pₜ), Acc(Iₜ)}.

如果组合不能超过最佳单一代码，论文只能主张其中一个表示有用，不能主张 physical-social integration。

### H3：Agent indexing 在信息不对称条件下必要

> Acc(Iₜ¹:ᴺ) > Acc(Iₜshared),

且差异应集中在 private message、缺席、遮挡和跨主体引用样本，而不是平均分布在所有问题上。

### H4：Compiler quality 约束 reasoning

> Acc(Coracle) ≥ Acc(Cpredicted),

并且 code quality 与 reasoning accuracy 呈正相关。该假设继承 Spatial Code 对 GT/predicted code 的分解，但把几何感知质量扩展为实体、access、event 与 interaction-state 编译质量。

### H5：局部社会干预产生局部行为变化

删除消息、改变角色或取消承诺后，目标主体的状态与行动应改变，而无关状态应保持不变：

> ΔCtarget ≠ ∅,  ΔCnon-target ≈ ∅.

如果模型只是在反事实提示下任意改变答案，而没有保持非目标状态，不能视为掌握了交互结构。

### 5.1 Go / No-Go

本项目在投入大规模原始视频处理前应通过三个门槛：

1. Oracle 或 semi-oracle Spatial–Social Code 相对信息匹配 memory 有稳定增益；
2. Pₜ + Iₜ 在至少一个 information-asymmetry 子集上超过 Pₜ-only 和 Iₜ-only；
3. Agent-indexed code 明显降低 perspective leakage。

如果上述条件不成立，应修改 schema、任务构造或核心主张，而不是继续堆叠视觉模块。

<!-- PAGEBREAK -->

## 6. 实验设计

### 6.1 主表示条件

所有条件使用同一 reasoner、同一问题、相同解码设置和尽可能接近的输入长度。

| 条件 | Physical | Interaction | Agent-specific | 目的 |
|---|---:|---:|---:|---|
| Direct video / Caption | - | - | - | 直接推理基线 |
| Free-form Memory | 部分 | 部分 | 通常否 | 叙事压缩基线 |
| Physical Code | 是 | 否 | 仅 access | 检验物理状态 |
| Shared Interaction Code | 否 | 是 | 否 | 检验共享状态局限 |
| Agent-indexed Interaction Code | 否 | 是 | 是 | 检验主体索引 |
| Spatial–Social Code | 是 | 是 | 是 | 完整方法 |
| Oracle Spatial–Social Code | 是 | 是 | 是 | 表示上限 |

Memory 必须由与 Code 相同的 canonical facts 生成，只移除类型、主体分区和结构，而不能少给事实。否则提升可能来自额外信息，而不是表示形式。

### 6.2 实验 A：MA-EgoQA - Perspective

**科学问题：** Agent-indexed state 是否比 shared memory 更能保持多主体信息边界？

现有本地数据包括 1,741 道问题、五类问题、127 个 caption JSON 和官方 shared memory。先在约 200 道按类别与信息不对称程度分层的开发样本上冻结 schema，再运行完整测试集。

任务分析：

- 谁最早观察或得知某事件；
- 谁未接收到世界更新；
- 同一物体跨 agent、跨时间的使用；
- 私有对话与跨主体引用；
- shared memory 是否把一个主体的信息错误赋给另一个主体。

主指标：

- overall 与 category accuracy；
- Perspective Leakage Rate；
- information-asymmetry subset accuracy；
- 随主体数、时间跨度和跨主体引用数量变化的性能曲线。

最关键比较：

> Shared Interaction Code vs. Agent-indexed Interaction Code.

### 6.3 实验 B：HoloAssist - Coordination

**科学问题：** 在真实 instructor-performer 协作中，Pₜ 与 Iₜ 的组合是否帮助理解干预与下一动作？

利用已有 action、conversation 和 task annotations 构造：

- instruction / correction event；
- object 与 task-step grounding；
- acknowledgement 与 physical execution 的区分；
- mistake / intervention prediction；
- next-action 和 intervention-target prediction。

设计 social-state aliasing minimal pairs：

| 相同部分 | 改变部分 | 预期变化 |
|---|---|---|
| 当前工具和手部状态 | 请求已确认 vs. 未接收 | 是否立即执行 |
| 当前任务步骤 | performer 已完成 vs. 只口头确认 | 是否继续纠错 |
| 同一句 instruction | 指向对象可见 vs. 被遮挡 | 是否需要澄清 |

第一阶段使用 annotation-only oracle/semi-oracle code。确认任务和表示有效后，冻结一组视频 ID，只下载对应 RGB/音频小子集，构建 predicted compiler；不先下载数百 GB 全量原始数据。

主指标：

- intervention timing AP；
- intervention type accuracy；
- next-action accuracy；
- request/acknowledgement/execution lifecycle accuracy；
- object grounding accuracy。

### 6.4 实验 C：EnactToM - Counterfactual 与功能行动

**科学问题：** 显式 Interaction Code 是否会导致正确行动变化，而不只是提高问答分数？

现有 300 个任务包含 PDDL、private information、message targets、literal ToM probes 与 golden trajectories。当前已完成 Habitat 在线闭环工程实现，并在 12 个开发任务、3 个 seed 上得到 Direct / Static SSC / Dynamic SSC 共 108 次运行。Dynamic 只在部分任务（尤其 episode 1166）稳定获益，在总体平均完成度上低于 Direct；因此这里研究的是适用条件和失效模式，不预设总体提升。

三类干预：

1. **信息干预：** 删除或重定向一条消息；
2. **角色干预：** 改变负责某个子任务的主体；
3. **承诺干预：** 将 acknowledged request 改为未接收、取消或违反。

评价：

- next-action accuracy；
- message recipient/content accuracy；
- valid action rate；
- Target Sensitivity；
- Non-target Invariance；
- communication cost；
- 完整 Linux GPU 环境可用后的 task success / Passᵏ。

### 6.5 补充环境

- **EgoTaskQA：** 获得许可后用于真实视频物理因果与 oracle-code upper bound；
- **CoMind：** 下载恢复后用于 gaze、joint attention、handover 和多视角 grounding。

二者不阻塞 MA-EgoQA、HoloAssist 与 EnactToM 的主结论。

<!-- FIGURE:experiments -->

## 7. 公平性、指标与统计

### 7.1 必须控制的混杂因素

1. **信息匹配：** Code 与 memory 来自相同事实集合；
2. **推理器匹配：** 使用同一模型、prompt family、temperature、重试次数和输出格式；
3. **长度匹配：** 报告 token 数、截断率，并使用固定预算和完整视图两种设置；
4. **检索匹配：** full-state 与 retrieved-view 分开报告；
5. **问题独立：** code 在问题出现前生成；
6. **训练状态透明：** frozen compiler、trained compiler、frozen reasoner 与 trained reasoner 分表报告；
7. **模态透明：** annotation、caption、video、audio、gaze 与 3D 输入分别注明。

### 7.2 表示与 compiler 指标

- schema-valid rate；
- entity/event slot F1；
- access relation F1 与 calibration；
- information attribution F1；
- commitment lifecycle accuracy；
- state-diff F1；
- deterministic replay pass rate。

### 7.3 下游指标

- QA accuracy / macro F1；
- perspective leakage rate；
- next-action accuracy；
- intervention timing AP；
- message selection accuracy；
- valid action rate；
- task success；
- communication token/message cost。

### 7.4 反事实局部性

定义：

> TargetSensitivity = 正确发生的目标状态变化 / gold 目标变化，

> NonTargetInvariance = 1 - 错误发生的非目标变化 / 可保持的非目标状态。

使用调和平均：

> CounterfactualLocality = 2TS·NTI / (TS + NTI).

该指标惩罚“为了让答案变化而重写整个世界”的模型。

### 7.5 统计协议

- episode 或 minimal pair 是主要统计单位；
- 对模型差异使用 paired bootstrap 95% confidence interval；
- 对二元成对结果使用 exact McNemar test；
- 同一视频的多个问题不能作为完全独立样本；
- 主指标与排除规则在测试集运行前冻结；
- 同时报告绝对差、置信区间和效应方向，不只报告 p-value。

## 8. 实施路线

### Stage 0：冻结 schema 与审计规则

完成 Pₜ/Iₜ JSON schema、event ontology、未知值语义、provenance 字段和 deterministic reducer 单元测试。SocialWorld-Micro 继续用于 gift/loan、private/public message、present/absent minimal pairs。

### Stage 1：MA-EgoQA 200 题 representation pilot

生成 Caption、Memory、Physical-only、Shared Interaction、Agent-indexed Interaction 和完整 Spatial–Social 六种输入。人工审核至少 30 个 code，确认主体信息没有被错误合并。

### Stage 2：完整 MA-EgoQA 与 EnactToM oracle

MA-EgoQA 跑完整 1,741 题；EnactToM 跑 300 个任务的 replay、信息干预、角色干预和承诺干预。此阶段给出 H1-H5 的第一轮判定。

### Stage 3：HoloAssist annotation-only coordination

从 action 与 conversation annotations 构造任务步骤、请求、确认、纠错和执行事件；完成 main representation table。

### Stage 4：Predicted compiler

按以下顺序替换 oracle：

> oracle entity → predicted entity → predicted access → predicted event → full predicted code.

绘制 code quality 与 reasoning accuracy 曲线，定位主要瓶颈。只有在 representation pilot 通过后，才下载冻结的 HoloAssist 视频子集并运行视觉/音频模型。

### Stage 5：功能环境与扩展（已进入实证阶段）

已在 Linux GPU 环境实现 EnactToM/Habitat 闭环 planning，并完成一个
`SSC-conditioned executable environment instantiation`：把包含资产、位姿、权限、
通信与目标的 SSC 规范编译成可运行 Habitat episode。该实例复用 HSSD 房间壳与已有
刚体资产，不是从视频恢复三维世界，也不是从零生成 mesh。下一步是扩展到多场景，
报告可达性、碰撞、约束满足率与额外 rollout 收益。

### 8.1 推荐的第一轮结果表

| Representation | MA Perspective | Holo Coordination | Enact Counterfactual | Leakage ↓ |
|---|---:|---:|---:|---:|
| Caption |  |  |  |  |
| Free-form Memory |  |  |  |  |
| Physical Code |  |  |  |  |
| Shared Interaction |  |  |  |  |
| Agent-indexed Interaction |  |  |  |  |
| Spatial–Social Code |  |  |  |  |
| Oracle Spatial–Social |  |  |  |  |

### 8.2 最重要的误差分解

每个错误应归入：

1. **Perception error：** 物体、动作、语音或主体身份未识别；
2. **Attribution error：** 事件正确，但分给了错误主体；
3. **State-update error：** 事件正确，状态迁移错误；
4. **Retrieval error：** 完整代码正确，query view 缺失相关字段；
5. **Reasoning error：** 正确 view 下仍回答或行动错误。

这一误差分解是 Spatial Code 的“perception bounds reasoning”在多主体场景中的直接推广。

## 9. 预期贡献与边界

### 9.1 论文可以主张什么

1. 提出一个面向多主体视频的 intermediate representation hypothesis；
2. 定义紧凑的 Cₜ = (Pₜ, Iₜ)，将共享物理世界与主体索引交互状态结合；
3. 通过信息匹配的 memory、Physical-only、Interaction-only 与 agent-index ablation 检验表示必要性；
4. 通过 oracle/predicted code 分解表示上限与 compiler gap；
5. 通过局部反事实检验代码是否支持功能性行动变化。

### 9.2 论文不能提前主张什么

1. 系统读取了真实人类内心；
2. `no-observed-access` 等价于“不知道”；
3. 系统覆盖人格、情绪、社会规范和所有 ToM；
4. 系统是完整的生成式 Social World Model；
5. JSON 或代码格式本身自动带来推理能力；
6. training-free 一定优于任务特定训练；
7. oracle-code 提升等于端到端视频系统已经解决。

### 9.3 最大研究风险

最大的风险不是 predicted compiler 不够准，而是 **oracle Spatial–Social Code 在信息匹配控制下并不优于 memory**。如果发生这种情况，可能意味着：

- schema 没有捕捉真正必要的交互变量；
- benchmark 问题可以靠局部文本匹配解决；
- code 与 memory 的信息控制不公平；
- 当前 reasoner 不善于读取这种 serialization；
- physical 与 interaction 的组合并无额外协同。

这些结果都具有科学价值，但要求收缩或修改论文主张。

## 10. 结论

Spatial Code 研究“视频模型为什么看见空间却不能稳定进行空间推理”；Spatial Social Code 研究“视频模型为什么看见多人互动却不能稳定区分主体视角、协调责任和反事实后果”。

两者共享同一个方法论：

> 不把所有能力缺口都归因于模型规模，而是寻找任务真正需要、但像素和自由文本没有稳定暴露的中间变量。

本文提出的最小答案是：

> Cₜ = (Pₜ, Iₜ),

即一个共享 Physical Code 与一组 agent-indexed Interaction Codes。第一篇论文的成功标准不是构建一个包罗万象的 Social World Model，而是通过严格的信息匹配、主体索引消融、oracle/predicted 分解和局部反事实实验，证明或否证这一表示假设。

## 参考文献

[1] Jieneng Chen, Wenxin Ma, Ruisheng Yuan, Yunzhi Zhang, Jiajun Wu, Alan Yuille. Thinking with Spatial Code for Physical-World Video Reasoning. arXiv:2603.05591, 2026. https://arxiv.org/abs/2603.05591

[2] Xin Wang et al. HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World. ICCV, 2023. arXiv:2309.17024.

[3] Kangsan Kim et al. MA-EgoQA: Question Answering over Egocentric Videos from Multiple Embodied Agents. arXiv:2603.09827, 2026.

[4] Gurusha Juneja et al. EnactToM: An Evolving Benchmark for Functional Theory of Mind in Embodied Agents. arXiv:2605.09826, 2026.

[5] EgoTaskQA: Understanding Human Tasks in Egocentric Videos. arXiv:2210.03929.

[6] CoMind: Understanding Collaborative Human Activity from Multiple Minds and Views. arXiv:2607.06691.
