# Figure 2 Habitat 子图候选：真实 SSC 更新过程

## 推荐与运行身份

推荐用 **Inspection，episode 588** 的消息更新链替代 Tea 的“立即纠错”画法。
这是对已有运行的新增审计/可视化，不是新跑出的实验结果。

- 任务：`emtom-104348082_171512994-588-cooperative-inspection-staging-with-a-hidden-correct-8487f942`
- 条件：Dynamic SSC，job 0032，seed 891011，模型 `gemini-3-flash-preview`。
- 接收主体：agent_1；消息来源：agent_0。
- 同 seed 的自然语言状态读取对照为 job 0030，两组最终均完成3/3目标（SSC第6轮，NL第7轮）。
- 本例来自匹配动态自然语言对照的后续实验，不是另一个108次Direct/Static/Dynamic
  汇总中的同一条轨迹；不要把它标成该表原始运行的截图。

主图：`state_update_three_panel.png`，同名PDF和SVG可编辑排版。
第三格使用的是此次原始运行保存的帧，没有生成场景、重新模拟或更换视角。

## 三格内容（严格时间顺序）

| 格子 | 真实记录 | 图中简写 |
|---|---|---|
| ① 更新前 | 第2轮，agent_1的状态v7；目标是把花瓶放到卧室桌子，尚无花瓶位置报告。规划器发出`FindObjectTool[vase]`，搜索反馈没有找到对象 | “目标：移动花瓶；花瓶位置尚未记录” |
| ② 新消息与状态更新 | 第2轮，agent_0→agent_1的消息确认送达；状态v7→v8，新增social记录`claim.agent_0.cabinet_20`，保留原消息、来源、轮次、reported状态 | 突出“新增他人报告：花瓶在cabinet_20；来自agent_0；t=2” |
| ③ 更新后下一步 | 第3轮，规划器读取v11，其中仍有该报告；实际选择`Navigate[cabinet_20]`；工具反馈导航成功，并有第3轮原始帧 | “下一动作：导航至cabinet_20；导航成功” |

图里只显示一条消息的相关摘录和一个新增报告摘要。第2轮搜索动作在消息前已选定，
但该动作的反馈行排在消息送达行之后；因此左格只写“当前计划：搜索花瓶”，
不把稍后的搜索失败反馈提前当成更新前已经收到的信息。
**真实状态并没有一个叫`vase.location`的新增字段**：原实现是按提及实体索引
完整消息的social claim。图中的“Reported vase location”是可读摘要，不是逐字代码。
此次消息更新同时新增了若干其他索引和送达记录，均保留在证据JSON；主图只聚焦一个。

**“尚未记录”不等于错误信念被纠正。** `before`中该claim不存在；这次是新增而不是覆盖。
`supersedes=null`。也不能把`status=reported`改画成`observed`，消息送达证明接收了
报告，不证明所报告的位置已被该主体亲眼验证。此次更新前后spatial层保持不变。

v8到下一次读入v11之间还有动作反馈和私有观察更新，claim仍然保留。不能把版本号
画成“v8紧接着被planner读取”，实际读取的是v11。

## 后续物理结果，以及为什么不放到“下一步”格

第4轮实际Pick，第5轮Navigate至table_22，第6轮
`Place[vase_4, on, table_22, None, None]`成功。
第6轮严格物理评分确认`is_on_top:vase_4:table_22=true`。
`agent_1_turn_06.png`保留了该轮原始帧，可在扩展案例中使用。

但第6轮放置不是消息后的下一动作。若要放最终结果，必须标“后来，第6轮”，
并说明省略了第3–5轮导航/抓取；不能剪接成第2轮消息直接触发第6轮放置。

## 和自然语言对照的真实区别

NL也在第2轮收到正确位置，且其第3轮prose状态仍保存这条信息。
第3轮动作是`FindObjectTool[vase on cabinet_20]`，SSC则是`Navigate[cabinet_20]`。
两组都成功，分别第7/6轮结束。这可以在文字里描述，但不能把NL写成忘记消息或无法更新。
本主图只展示SSC的一条执行链，不另外添加误导性的NL失败路径。

还有一个重要限制：核对实际API request后，发现这条消息**也出现在原生观察文本里**。
因此能确认“更新被存储、状态被实际送入planner、下一动作与消息一致”，
不能断言“动作一定由SSC这条claim而非原生消息渠道造成”。模型输出的自述也不是因果证据。

## 对其他候选的检查结论

| 候选 | 是否适合紧凑三格图 | 原因 |
|---|---|---|
| Inspection 588 / job32 | **首选** | 第2轮消息→新增来源标注→第3轮明确定位导航，前后状态及实际request齐全 |
| Briefing 618 / job42 | 备选 | 第1轮报告bottle_2在chest_of_drawers_33→第2轮Navigate；但后续涉及跨主体交接，整体更复杂 |
| Staging 983 / job37 | 备选 | 第5轮jug_3在bed_9的消息→第6轮Navigate；最终仅2/3且NL job35有服务问题，不适合包装为干净的结果优势案例 |
| Tea 1781 / job2 | 不适合“立即重放”三格 | 第2轮消息后第3轮是Wait，第5轮才发生误放，第7轮纠正；原消息不是看到误放后发来的纠错通知 |
| Guest 1166 / job22 | 更适合速度案例 | 8轮vs13轮已核实，但消息所说box位置与紧接着的导航目标不构成最简直接链 |
| Laundry 623 / job12 | 不宜归因给第4轮消息 | 第3轮已拿毛巾，第4轮在消息到来前已计划Navigate至shelves_20；第5轮Place不能简单归因于新增消息 |
| Exhibit 772 / job27 | 更适合较长目标链 | 第2轮目标消息，第3轮先打开源容器，第6轮才放置相框；压缩成一步会省略关键操作 |

这里只引用当前Flash严格物理评分运行，不使用历史Qwen episode772中曾出现
“物体仍被抓取却判放置成功”的旧案例。

## 图注建议

> **A delivered location report updates agent-specific SSC before the next
> action.** In Habitat episode 588 (seed 891011), agent_1 initially has no
> recorded vase-location report. A delivered message from agent_0 adds a
> source- and time-attributed claim at turn 2. The next planner input retains
> this claim, and agent_1 navigates to cabinet_20 at turn 3, with successful
> execution feedback. Highlighted text is a compact rendering of one newly
> added claim; the image is an original saved frame. This trace illustrates
> state maintenance and subsequent behavior, not a causal ablation; the message
> is also available through the native observation channel.

图用于**解释系统如何运行**，而不是代替总体统计、表示/链接消融或多seed验证。
建议替换Figure 2的Habitat子图；不要再把它当作一张新的结果曲线。

## 文件与可追溯性

- `evidence_chain.json`：原消息、完整before/after、下一次状态输入与输出、反馈、
  NL对应记录、后续物理评分。每条带原日志路径、行号、SHA256。
- 实际API request已核实包含完整记录状态；这里只导出请求路径、哈希和核验结果，
  不复制整份请求或认证信息。
- `agent_1_turn_02.png`、`agent_1_turn_03.png`、`agent_1_turn_06.png`：未经改动的原始帧及哈希。
- `state_update_three_panel.{png,pdf,svg}`：三格草图。
- 脚本：`experiments/build_habitat_update_case.py`。
- 测试：`tests/test_habitat_update_case.py`，拒绝伪造物理更新、把reported改成observed、
  把后续放置替换为下一步动作等不正确的表达。

从公开导出重画（无需LLM、模拟器或私有workdirs）：

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/physical_code_mpl \
  MA-EgoQA/.venv/bin/python -m experiments.build_habitat_update_case --from-export
PYTHONDONTWRITEBYTECODE=1 MA-EgoQA/.venv/bin/python -m unittest tests.test_habitat_update_case
```

本案例随跨环境案例包发布：`reports/cross_environment_casebook_20260923/`。
