# Habitat 中 Spatial–Social Code 的四层可执行链路

## 1. 环境表征导出

输入是真实 Habitat episode，输出是带来源信息的 Spatial–Social Code：

- Spatial Code：scene、房间、对象、承载/容器关系、关节状态、资产配置和 4×4 位姿；
- Social Code：agent 出生点、访问限制、通信图、消息预算和私有事实；
- provenance：任务、episode、stage 与资产配置的 SHA-256。

这一层证明 Code 能完整描述规划所需的物理和社会状态，但本身还不是环境生成。

## 2. 固定物理环境中的社会信息干预

在 episode 1566 上保持 scene、初始物理状态、目标和 seed 不变，只改变
`spray_bottle_3` 精确位置的知识归属，并加入无关事实负控制。四组干预均实际运行
Direct 与 Dynamic SSC，共 8 个 rollout。

这层验证的是 Code 的因果可干预性：谁拥有事实、谁能接收消息，以及信息缺失会如何
影响规划。它不属于新物理环境生成。

## 3. 在线闭环规划

Dynamic SSC 在每轮执行后根据私有观察、动作结果和已确认送达消息更新状态，再将最新
状态提供给下一轮规划器。Direct 不接收 SSC 状态，Static SSC 只接收初始状态。

最清晰的机制案例是冻结 episode 1166：三个 seed 中 Dynamic SSC 都严格完成 3/3
物理目标，Direct 和 Static SSC 都没有完成；Dynamic 分别在 7、10、7 轮结束，Direct
均运行到 20 轮。该结果证明动态状态更新在这一信息接力任务中可重复有效，但三个 seed
属于同一任务，不能替代跨任务显著性。

## 4. Code → Environment 前向实例化

前向编译器只读取 SSC JSON，不读取原始 episode。它输出：

1. 独立的 Habitat `episodes.json.gz`；
2. 与生成 episode 绑定的 EnactToM `task.json`；
3. 修改后的资产位姿与完整生成 Code；
4. 编译 manifest、运行证据和真实渲染帧。

当前实现生成新的 episode/layout instance，同时复用已有 HSSD 房间壳和资产 mesh。
因此可以称为 **SSC-conditioned executable environment instantiation**，不能称为从零生成
全新 3D 网格。未来生成式世界模型可替换资产检索与布局模块，而继续使用同一 SSC 接口
表达物理关系、主体视角和信息边界。

## 证据边界

- golden trajectory、Oracle schedule 和 QA 目标答案不会进入生成任务。
- 生成 episode 必须由 Habitat 实际加载并产生 RGB 帧，不能只通过 JSON schema 检查。
- 物理完成由 `strict_physical_v1` 连续三次物理采样判定，RGB 只作展示。
- 固定场景干预和单个生成环境属于机制证据，不报告成总体 benchmark 提升。

