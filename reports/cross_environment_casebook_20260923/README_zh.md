# 跨环境真实案例包：13例，含代码、证据、图和边界

**目的不是增加实验或拼出一个全部正向的故事，而是给Figure 2准备可追溯的运行过程材料。**
此次只读既有结果，没有调用LLM/SAM、启动模拟器或更改后台实验。

## 正文优先级

1. **Habitat 588：首选“在线状态更新”子图。** 有真正的before/after、送达消息、
   下一次实际API输入、动作、工具反馈和原始帧，最符合三格过程图。
2. **MA-EgoQA 1038：首选“证据引用与主体归属”静态案例。** 有视频片段、
   Tasha的提问/Jake的回答记录、social引用和最终答案。但说话人/音频内容仍需
   投稿前人工核对，不能称为已独立语义验证，也不能画成在线状态更新。
3. **CoMind 66：首选“目标识别接入相同SAM”的视觉例子。** 两视角框与金标
   重合直观，适合展示感知路径；其social code是unknown，**不能**用来证明双层联结。
4. **MMToM 84：历史补充例，不宜承担干净机制证据。** 有检查容器前后两条
   生成信念和完整代码，但答案条件化规则与融合也参与了最终判断。

不要四个环境强行都叫“state update”：只有本包Habitat有真实在线状态存储更新；
另三种分别是静态视频编译/证据链接/视觉定位的处理过程。

## 全部案例索引

| 环境 | 案例 | 适合展示什么 | 关键结果或限制 | 目录 |
|---|---|---|---|---|
| Habitat | **588 Inspection** | 报告新增→带来源状态→下一步导航 | t2消息，t3导航成功；不是消息立即导致t6放置 | [inspection_588](habitat/inspection_588/) |
| Habitat | 618 Briefing | 他人给出瓶子位置→接收者定向导航 | t1消息，t2导航；后续交接不能压缩成一步 | [episode_618](habitat/episode_618/) |
| Habitat | 983 Staging | 搜索后收到壶的位置→改为导航 | t5消息，t6导航；最终仅2/3，不是完整成功案例 | [episode_983](habitat/episode_983/) |
| MA-EgoQA | **1038** | 猜测、纠正、主体归属和S2/S3引用 | 三个条件都答C且正确，不是消融优势 | [index_1038](ma_egoqa/index_1038/) |
| MA-EgoQA | 950 | 自述饱腹感→belief记录→引用答题 | 三个条件都答D且正确 | [index_0950](ma_egoqa/index_0950/) |
| MA-EgoQA | 1337 | **反例：同一天不等于同时发生** | Full选E错误；去归属选D正确；实际引用时间11:54和16:25 | [index_1337](ma_egoqa/index_1337/) |
| MA-EgoQA | 1597 | 按日期/时间筛选证据 | Full/去归属选C正确；去链接无效，不能当作有效错误答案对比 | [index_1597](ma_egoqa/index_1597/) |
| MMToM | **84** | 检查容器与生成信念变化 | Direct A错，SSC B对；包含历史答案条件化融合 | [index_0084](mmtom/index_0084/) |
| MMToM | 318 | 神经目标假设与后续规则融合 | 神经social偏向salmon，历史融合最终选apple；不能把成功只归给神经code | [index_0318](mmtom/index_0318/) |
| MMToM | 6 | **反例：中间信念正确不保证最终答对** | Direct A对，SSC B错 | [index_0006](mmtom/index_0006/) |
| CoMind | **66 / 016248** | dough描述→相同SAM两视角定位 | IoU左0.003→0.925，右0→0.873；social=unknown | [index_0066](comind/index_0066/) |
| CoMind | 226 | 对象描述与视角选择影响定位 | IoU0→0.926，但其他社会字段更差，且选择视角不同 | [index_0226](comind/index_0226/) |
| CoMind | 5 | **反例：错误对象语义导致SAM错定位** | 左IoU0.962→0，Direct+SAM更好 | [index_0005](comind/index_0005/) |

所有案例都是事后选取的定性材料，不是新评测子集，不合并成“13例胜率”。

## 已完成的核验

### Habitat：实际更新、不是故事补写

- 核对接收主体、发信主体、消息轮次和真实`before/after`。
- 确认关键claim此前不存在、更新后为`reported`；不是直接写成已观察到的物理事实。
- 核对下一次API request确实包含完整状态块，实际下一动作及执行反馈一致。
- 图中场景帧来自对应原始运行，不是回放、重建或生成的画面。
- 消息也可出现在原生观察通道：可证明状态被维护并送入规划器，不能隔离其因果作用。
- 983的NL配对曾受服务问题影响，不能包装成干净的SSC胜过NL案例。

### MA：同一编译记录、真实引用、真实输入片段

- 保存问题、选项、独立scorer金标、完整Spatial/Social code及执行trace、三种实际view与答案。
- 核对三个最终答题条件接收的视频SHA256序列相同；保存已有的脱敏请求记录。
- 核对答案引用存在，social→evidence链接能解析，引用的clip属于实际输入列表。
- 预览图从输入MP4按引用时间抽取，保存clip哈希、预览时间及帧号；**不是声称
  模型内部恰好看了这些抽帧**。模型输入是包含音频的视频片段。
- Worker `row.answer`是刻意设置的option0占位值，不是金标；真实标签只取
  `SCORER_ONLY_gold.json`，没有把占位值误报为答案。
- 原始quote和身份仍是模型生成的解释；画面预览不能独立证实对白内容与说话人。
  **人工音频/身份核验是投稿前待办，不因程序检查通过而省略。**
- Full100分数46/46/49（Full/去链接/去归属）。1038和950展示记录形式，不能
  作为机制增益证据；1597的差异来自去链接无效输出。

### MMToM：拒绝把解析错误当推理优势

- 从已发表的历史600结果、冻结输入和请求归档抽取，保留完整代码与调用记录。
- 3个选中案例的Direct原文最后独立A/B与保存解析一致；真实8帧逐字节复制，
  不含未触发的16帧细化输入。
- 原来展示的index0不再推荐：原始Direct结尾是B，而存档解析为A。
  这是展示时必须披露的解析不一致，**此次没有修改旧分数或重算成新结果**。
- MMToM协议包含候选答案、query evidence、确定性规则、Spatial桥接和posterior；
  不能称为纯神经SSC或option-blind协议。生成观察/信念不等于独立语义真值。
- 案例6保留融合失败，案例318明确标出神经假设与最终融合结果不同。

### CoMind：不仅比SSC+SAM与没有工具的Direct

- 四条件都保存：Direct、Direct+SAM、SSC、SSC+SAM；独立重算所有字段分数。
- 从同一固定最终字段生成SAM检索词，使用同一候选排序、视角选择及回退规则。
- 已从保存proposals离线重放SAM选框与坐标转换，未重新调用SAM或使用金标选框。
- 原帧1408×1408；Joint Attention/Handover组合画布中每视角缩半，helper横向加704；
  红色预测框和绿色评估金标都使用相同xywh坐标。
- 66中SSC社会记录是unknown，定位增益不能解释为social层有效。
- 226的Direct+SAM选helper，SSC+SAM选leader；SSC仅定位改善，其他社会字段
  不能据此称为成功。5是对象描述误导定位的反例。
- 阈值遵循冻结代码`IoU > 0.5`，不是擅自改为另一套坐标或打分协议。

## 文件使用方式

每个目录都有README和机器可读证据包。

- **画论文图：**优先取`state_update_three_panel.*`、`evidence_flow.*`、
  `input_preview.jpg`和CoMind的`localization_overlay.*`。
- **核实原文：**`case.json`或`evidence_chain.json`保存原始记录与来源SHA256。
- **展示code：**`spatial_code.py`、`social_code.py`是原始生成文本，不自动执行。
- **实际输入：**MMToM/CoMind有相应全部采样输入帧；MA有原MP4哈希与引用预览，
  不重新发布完整视频；Habitat有对应动作后的原始帧。
- **透明选择：**`selection_ledger_all.json`包含MMToM600、MA100、CoMind300的
  逐样本筛选记录；未筛成只含成功样本。Habitat来自此前完整12任务审计。
- **文件完整性：**`SHA256SUMS.json`；`index_all.json`为13例机器可读索引。

等级说明：A表示相应记录/几何流程可追溯，不表示方法普遍有效；B表示还有协议混杂
或人工语义核验缺口。没有把所有案例标成“完全证实机制”。

## 复现与代码

公开仓库无需原始数据集即可进行离线一致性核验：

```bash
python -m experiments.verify_cross_environment_casebook
python -m unittest discover -s tests -p 'test_cross_environment_casebook.py'
```

从公开快照重画（需matplotlib和Pillow；不会调用模型、SAM或模拟器）：

```bash
MPLCONFIGDIR=/tmp/ssc-mpl python -m experiments.build_cross_environment_casebook --from-export
```

原始抽取模式需要本地冻结workdirs和视频文件，MA预览解码使用OpenCV。
公开核验只能验证记录、引用、几何和算术一致性，不能代替独立视频语义标注。

源脚本：`experiments/build_cross_environment_casebook.py`、
`experiments/verify_cross_environment_casebook.py`；Habitat详细脚本为
`experiments/build_habitat_update_case.py`。测试随包提交。

素材来自各数据集的原始研究输入，未由模型合成；原数据集许可仍适用，不把原视频帧
重新宣称为本项目独占作品。此包为少量案例预览，不是完整数据集再发布。
