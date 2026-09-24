# Figure 2局部修订：Pro MMToM替代案例与面板核验清单

## 交付范围

保留Figure 2原布局，保留(c)(d)、state/code→Blender图和附录过程图。本次只新建(a)替换素材和核验说明，**没有重画整张图，也没有修改上述现有材料**。

工作区未找到论文Figure 2总图、可编辑排版源或主稿TeX，所以这里是可直接交给排版的替换包，**不是已经修改了论文总图**。提供源文件后只需替换(a)局部及对应caption；(b)也需要原面板/样本ID才能完成针对性核验。

## 推荐MMToM替代：历史Pro index302，episode14，Goal题型2.1

这是从Pro档案挑出的定性案例，不是从Flash档案或当前3.1Pro补跑中取的。

- 原Direct回答完整且只有 `B`，保存预测也为B。
- 保存SSC读出为A；冻结benchmark标签为A。
- 不在69条无明确末行答案集合内。
- 没触发16帧细化，展示的8帧与该Pro记录的帧编号一致；所有8张图像SHA256匹配冻结档案。
- 选择规则：从全部600条中选择“明确最终回答与保存预测一致、在保存标签下Direct错/SSC对、未触发密集帧”的最小index。是事后选择的案例，不可拿来估计总体优势。
- 保留600条筛选账本，未隐藏其他记录或把案例计为新评测集。

### 为什么适合展示

1. **物理事实**：场景文本说，微波炉里有cupcake，冰箱里有dish bowl。
2. **主体前提**：问题明确给定，John认为微波炉里没有cupcake。
3. **动作历史**：供给文本记载John开关第4个厨柜，然后走向微波炉。
4. **保存表示**：`S.fact`保留物理位置，`M.belief(..., 'given query premise')`保留主体前提。
5. **历史读出**：候选条件化query规则与posterior给A（dish bowl）0.8775，B（cupcake）0.1225。

**不能把这讲成“模型从画面识别出John的错误信念”。** 主体前提直接来自问题；画面只展示对应输入环境。最终结果有历史规则/候选评分参与，也不能把该案例当成独立证据链接消融或在线更新案例。

### Figure2(a)建议呈现

使用三格“输入 → 保存表示 → 历史读出”，不使用Direct✗/SSC✓胜负徽标。
可以原样标出Direct B、SSC A、benchmark A，明确是档案标签。尚未找到该Pro调用的实际请求/完整响应，不能将保存标签映射称为完成逐请求验证。

图中对隐藏物体位置标 `supplied scene text`，对John信念标 `query premise`；不画成摄像头直接看见或主体新收到的消息。
帧96/224是**帧索引，不是秒数**。人物名John来自题目文本，不是人脸识别。

## 文件

主素材均位于 `mmtom_pro_0302/`：

- `panel_a_input_state_readout.svg`：可编辑文本/矢量排版，含实际输入帧。
- `panel_a_input_state_readout.pdf`：论文插图版本。
- `panel_a_input_state_readout.png`：预览。
- `case.json`：完整题目、保存Pro记录、code、trace、posterior、证据来源及限制。
- `direct_raw.txt`：真实Direct原文，只有B。
- `spatial_code.py` / `social_code.py`：原样保存的代码。
- `frames/` / `frames_contact_sheet.png`：8张真实输入帧与联系表。
- `runtime_schema_audit.json`：执行校验细节。

根目录另有 `selection_ledger_600.json`、`selection.json`、`panel_b_review.json` 和文件校验和。

### 关于代码核验的准确表述

Spatial code执行结果与保存trace逐字一致。
可用的归档runtime在Social `query_candidate` 中额外加入 `constraint=ALL` 和 `variable_type=unknown` 两个默认字段；逐项移除这四个新增默认字段后，其余所有记录/字段与旧Pro trace完全相同，posterior一致。
这不是逐字相同的Social序列化，也不是建立了Pro每次运行的source commit身份；完整差异已单独保存，未修改原记录。

## (a)旧index0怎么处理

历史**Pro** index0的Direct保存原文结束于 `However, Jennifer goes`，没有明确最终选项。不要依据保存的A标Direct✗。
注意不要混淆：历史**Flash** index0则有明确末尾B却被存A，是另一份模型回答。本次替换只使用Pro302。

## (b)核验状态：不能靠caption为未核实的对白兜底

本地已有候选MA-EgoQA1038被额外检查，但不能断言它就是你现在的(b)：

- 输入10个clip哈希、3张引用预览图哈希及social→evidence引用可对应。
- Tasha提问/Jake回答的姓名和对白来自生成记录，不是已独立核验的transcript。
- 相机佩戴者Tasha不等于每句话的说话人；预览帧不能证明对白。
- 三个保存答案都为C，只能说明档案读出一致，不能证明这段说话人归属正确。
- **这个本地候选属于Flash，不应未经说明替代Pro论文面板。** 本次没有把它放入新图。

在确认实际(b)的源样本之前：移除未核实的人物对白气泡、姓名与脸部连线、胜负徽标；若保留，只展示真实输入clip ID → 生成记录及引用结构 → 保存读出，不把生成quote画成实际对白事实。
若连实际面板对应的样本/clip也无法建立，就应暂停该面板的实证身份，不能凭caption继续保留断言。

## 建议英文caption片段

> **(a) MMToM, historical Pro example (index 302).** The supplied scene places a cupcake in the microwave, whereas the question explicitly stipulates that John believes no cupcake is there. The archived representation keeps the physical fact and the attributed premise separate. The historical candidate-conditioned readout assigns probability 0.8775 to option A (dish bowl). The archived Direct response is B and the saved SSC readout is A; the benchmark label is A. This illustrates recorded input-to-state-to-readout processing, not visually inferred belief, an online update, or an isolated causal effect of cross-layer links. Displayed images are source frames 96 and 224, not timestamps in seconds.

此caption只针对(a)，不重写(c)(d)或Blender/附录说明。

## 复现与后台

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/ssc-figure2-mpl \
  MA-EgoQA/.venv/bin/python experiments/build_figure2_pro_replacement_20260923.py
```

公开资料包无需下载原始benchmark即可检查档案对应关系：

```bash
python3 experiments/verify_figure2_pro_replacement_20260923.py
# 仅重画，需要matplotlib和Pillow：
MPLCONFIGDIR=/tmp/ssc-figure2-mpl python3 experiments/build_figure2_pro_replacement_20260923.py --from-export
```

纯离线读取/执行检查与绘图，没有调用模型、SAM或模拟器；没有停止或修改69题后台实验。
本次发布范围仅为案例替换包及其生成/核验脚本，不包含69题补跑数据，也不改历史总分。
