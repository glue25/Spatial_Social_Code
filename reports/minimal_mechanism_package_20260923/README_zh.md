# 最小机制补实验包：范围、后台入口和已完成的离线核验

**最终状态：正式640/640已于2026-09-23 12:39 UTC完成（629格式有效，失败仍计入）。**
下面保留启动和工程修订的历史说明；本轮后台队列已结束。
正式投稿表格、全量原始回答、案例和公开复核入口见
[最终投稿资料包](../information_history_v1_20260923/README_zh.md)。

## 只新增一项付费实验

第一优先级是固定世界终态的信息历史干预，不同时铺开新环境、更多seed或大规模QA。
工程10个基础情节与正式40个基础情节互不重叠，每个四种历史、四种条件：

| 条件 | 改变什么 |
|---|---|
| linked SSC diagnostic | 完整物理事件、主体经历和跨层对应关系 |
| deterministic prose | 逐字段无损转写，全部语义相同，不是独立摘要器 |
| no_links | 保留两层全部记录、主体、时间、内容，只删除对应表 |
| no_attribution | 保留记录和链接，将持有者/来源主体改成null，不交换身份 |

正式640次最终答题，工程160次，另有四个可用key的服务预检。没有重新编译视频。
模型固定为服务端报告的`gemini-3-flash-preview`；各臂同一输出上限、温度和reasoning设置。
从125个合格且不在所列历史本地物化任务中的episode中，按固定哈希选择前50个：
前10工程、后40正式；正式基础词汇来自20个scene。选择不读取历史预测或正确率。

**重要范围：这是新构造的符号文本记录诊断。** 原EnactToM任务只提供实体词汇及来源ID，
没有执行原轨迹、模拟器物理、实际视线或相应视频。20个scene是来源标签，不能被描述成
已在20个真实场景完成物理反事实。共享事件模板也限制泛化。新适配器使用现有StateStore，
但不是原Habitat planner输入的逐字复现。因此本实验只检验受控文本状态中的信息使用，
不能单独补足视觉编译或独立在线任务泛化证据。

删除链接后，部分具有不同真值的历史会产生相同输入。下降只能说明被删关联承载的信息
必要，不代表同信息量条件下code更优。无归属条件也有信息损失。SSC与prose的比较独立解释。

## 正在运行的位置

主运行目录（评分修订2）：

```text
workdirs/information_history_v1_20260923_scoring2/
  frozen_config.json       # 冻结协议、模型参数、源文件哈希
  dataset.json             # 50基础情节、200变体、事件真值和原始状态快照
  selection_audit.json     # 全候选、排除证据、数据来源与预选规则
  source/                  # 运行真正导入的冻结代码
  preflight.json           # 四key均通过同模型预检，不含key本身
  status.json              # 当前阶段与覆盖量
  workers/                 # 各基础情节当前工作
  calls/{split}/{base}/{history}/{condition}/
    request.json           # 原始请求，不含Authorization
    attempt_*.json         # 每次实际尝试、耗时、tokens、错误
    response.json          # 原始HTTP200响应，先缓存再评分
    result.json            # 按字段评分、配置/源哈希、artifact路径
  engineering_gate.json   # 技术检查门槛，不按正确率筛选
  engineering_summary.json
  formal_summary.json
  COMPLETE.json           # 仅在正式640次都有终态记录后写入
```

tmux会话：`ssc-info-history-20260923-s2`。

启动核验：2026-09-23 11:54 UTC，工程160/160完成，158份格式有效，技术门槛通过；
后台已自动进入正式640次答题。以上是带时间的启动记录，最新覆盖以`status.json`为准。

工程门槛是结构测试全部通过且>=95%格式有效，不使用准确率门槛。通过后自动跑正式集。
固定样本、失败和超时都保留；未完成单列为pending，不假装是有效配对。模型有效输出不按
正确性或引用质量重采样。网络/HTTP临时错误最多12次尝试；模型身份不符会停止。

### 工程期评分修订的审计

初版将错误evidence ID当成整份输出无效，可能抹去三个正确的状态答案。工程阶段发现后，
暂停初版；**正式尚无任何请求**。修订2仅把值错误归入相应语义指标，JSON结构/类型/截断
检查不变。88份已经生成的回答逐字复用、离线重评，没有重新付费调用、没有修改提示词、
实体、金标或正式样本。完整初版保留在`workdirs/information_history_v1_20260923/`，
通过`SUPERSEDED_BY.json`和新配置中的`engineering_cache_parent`建立关联。

## 自动报告

报告输出到`reports/information_history_v1_20260923/`：
按基础情节聚类的配对区间、逐条件指标、实际调用/tokens和source/config/artifact映射。
调用成本表按条件列出，四次共同预检的成本保留在`probes/`中，不混入某一实验臂。
原始响应复用不作为新增API调用重复相加。

两项共同主要比较：SSC-no_links、SSC-no_attribution的成对联合正确率；10,000次
基础情节cluster bootstrap，报告Bonferroni97.5%边际区间。prose比较为次要95%区间。
同时报告世界状态正确不变率、目标主体正确变化率、无关主体正确保持率、证据支持正确率。
40个情节不是统计功效保证；不会根据前几题结果筛选子集或只画有利类别。

完整实现与限制：[`../../experiments/information_history_v1/README.md`](../../experiments/information_history_v1/README.md)。

## 已有结果接入：不重复调用模型

### Habitat：补齐108个历史运行的文件级映射

`habitat108_provenance_manifest.json`逐项连接：结果块行号→task/episode/seed→
运行manifest中的config hash→归档updater/scorer source hash→artifact路径。
108个任务/seed/条件与原计划完全对齐；353个归档源文件哈希匹配，1,885条评分记录
已定位；最终measurement无效标记0个，结果块与归档最终summary差异0个。
这次不是模拟器重跑，也不是重新判定全部逐轮物理事实。

逐运行git/source-commit identity仍缺失，已显式置null。已补的是可核验的归档文件/
配置对应关系，不用当前代码或一个新commit冒充旧运行的身份。

### CoMind：300题四臂逐项核验已完成

核对同批split/sample/recording IDs、原始Direct/SSC快照、四臂预测、冻结源代码、
SAM候选选择与坐标变换。300/300通过离线重放，没有新SAM或VLM调用。

四项定位指标（JA左右、SCOIA、Handover，每项100题）的等权均值：

| Direct | Direct+SAM | SSC | SSC+SAM |
|---:|---:|---:|---:|
| 0.75% | 13.75% | 0.25% | 16.00% |

SSC+SAM减Direct+SAM为**+2.25个百分点**；55个recording聚类、10,000次bootstrap的
95%区间约为**[-2.48,+6.99]个百分点**。跨零，只能作为描述性开发集差异。
阈值严格跟随冻结scorer的`IoU > 0.5`。没有将旧批次Direct与新批次SAM拼接。

- `comind_four_arm_audit.json`：全部字段、定位均值、配对区间。
- `comind_pairing_manifest.json`：全部300个样本到输入/结果/原始预测/source hash的映射。

### MMToM：旧协议错误分组只作保守核验

历史600题中，保存的Direct预测与原文末尾**独立明确A/B**有166处不一致（不是中间推理
里的字母）；原始输出、stored prediction和片段全部列在`mmtom_parser_inconsistencies.json`。
这些记录的`direct_retry_used`均为false。这里只确认档案不一致，尚未将其自动归因到某个
具体解析函数或改写旧分数。该历史Direct/SSC比较不能直接作为干净协议证据。

历史belief中Direct对/SSC错的48题全部收入`mmtom_belief_review_packet.json`。
它们的Spatial/Social trace可解析，但可解析不等于语义正确。没有足够逐项人工证据将其
判成“编译错误”或“规则覆盖错误”，因此全部保留为待语义/规则核查，不能凭最终错答猜原因。
600题中的一条终态失败另行记录，没有从原始范围中删除。

## 本轮没有启动的项目

- 独立动态NL摘要器的72个新任务rollout：有价值，但暂不挤占优先级1预算。
- MMToM option-blind600：没有拼接历史204对；本轮未扩跑。
- MA六臂或新recording扩展、新backbone、新环境、更多精选案例：未启动。
- 旧Habitat新seed：未擅自恢复已暂停队列，也不把旧状态文件当作仍活跃的证据。

本包不承诺效果正向，也不承诺审稿评分。当前目标是用最少新增调用，得到范围明确、
可复核而不是归因过度的机制诊断，并把已有协议问题显式隔离。
