# 投稿资料包：受控信息历史诊断（正式40情节、640次回答）

**状态：全部完成。** 2026-09-23 12:39 UTC结束。工程160次单列，正式640次全部纳入；
629次格式有效，11次无效仍计入分母。不再有本轮后台队列待跑。

## 建议提交的核心内容

1. **正文/补充表：**[`submission_table.tex`](submission_table.tex)、
   [`submission_table.csv`](submission_table.csv)；全部指标及置信区间在
   [`formal_summary.json`](formal_summary.json)。
2. **完整范围统计图：**[`mechanism_diagnostic.png`](mechanism_diagnostic.png)，
   同名PDF/SVG可用于排版。所有40个基础情节、四个条件均展示，不按表现选子组。
3. **首选机制案例：**[`formal_000`](cases/formal_000/README.md)。保持世界终态相同，
   四种信息历史对应不同主体状态；SSC与无损prose均答对，不伪装成SSC独有优势。
4. **可直接改写的正文与图注：**[`manuscript_text.md`](manuscript_text.md)。
5. **复核材料：**全部800个工程/正式回答的原始请求、响应、评分、调用记录，
   输入/真值、冻结代码与哈希均在本目录；见下方复核方式。

## 正式结果（百分比）

| 指标 | Linked SSC | 无损prose | 去链接 | 去主体归属 |
|---|---:|---:|---:|---:|
| 世界位置正确 | 100.00 | 99.38 | 94.38 | 98.75 |
| 目标主体状态正确 | 98.75 | 96.88 | 38.75 | 3.13 |
| 无关主体状态正确 | 97.50 | 98.75 | 10.63 | 0.63 |
| 三项状态同时正确 | 96.88 | 96.25 | 5.00 | 0.63 |
| 世界状态成对正确保持 | 100.00 | 98.75 | 90.42 | 97.50 |
| 目标主体随历史正确变化 | 97.50 | 93.75 | 1.88 | 0.00 |
| 无关主体成对正确保持 | 95.83 | 97.50 | 5.00 | 0.00 |
| **成对联合正确率（预设主指标）** | **95.00** | **92.50** | **0.00** | **0.00** |
| 目标状态+指定物理证据ID同时正确 | 83.13 | 90.00 | 18.75 | 3.13 |

每个条件160次回答。同一基础情节包含四种历史；成对联合指标要求预先指定的两个
历史中，世界、目标主体、无关主体三项状态都正确。10,000次按基础情节配对bootstrap。

- SSC−去链接：**+95.0 pp，97.5%区间[87.5,100.0]**。
- SSC−去归属：**+95.0 pp，97.5%区间[87.5,100.0]**。
- SSC−prose：**+2.5 pp，95%区间[-5.0,10.0]**，跨零，未建立准确率优势。

前两个是共同主要比较，采用Bonferroni97.5%边际区间；图和逐指标表中误差条是95%
边际区间，不能与比较差值区间混淆。11个无效输出分布：SSC0、prose1、去链接9、去归属1。

**重要：去链接/去归属本身删除必要信息。** 部分不同历史在去链接后产生相同输入，
主体状态不再可唯一判断；`unknown`可能是合理答案，虽然不匹配完整信息金标。
这些大幅下降说明关联信息在该诊断中的作用，**不是**同信息量下SSC格式优于其他表示。
证据指标严格要求指定的物理事件ID，不只检查任意ID是否存在；使用social ID的回答
即使可沿链接追溯，也不满足此精确输出要求。该项并不对SSC有利，予以完整保留。

## 成本：描述性收益，不是新的显著性结论

| 条件 | 正式调用 | 输入tokens | 输出tokens（含服务端计入的推理） | 总tokens |
|---|---:|---:|---:|---:|
| SSC | 160 | 250,184 | 104,314 | 354,498 |
| prose | 160 | 311,817 | 143,914 | 455,731 |
| 去链接 | 160 | 191,067 | 249,342 | 440,409 |
| 去归属 | 160 | 243,397 | 159,158 | 402,555 |

SSC比prose少22.21%总tokens，这是同输出上限/同模型设置下的本次实测，非精确等输入
token实验。没有填充token。没有美元账单，不能把token降幅直接称为美元节省。
工程/预检另列；[`costs.json`](costs.json)和[`cost_table.csv`](cost_table.csv)保留全部条件。
elapsed_seconds是请求耗时之和，不是并行运行的墙钟时间。预检成本在原始archive中，
不分摊到某个实验臂；复用的88份工程回答不作为新增调用重复计费。

## 四个可用案例：选择透明、完整保留四种历史

| 案例 | 展示重点 | 使用边界 |
|---|---|---|
| [formal_000](cases/formal_000/README.md) | vase_1：世界始终bench_18；目标在bench_18/bed_11之间随信息历史变化；无关主体保持bed_11 | **首选正文过程示例**；SSC和prose都完全正确 |
| [formal_003](cases/formal_003/README.md) | candle_4：存在未观察中间移动，正确区分世界轨迹和主体获得的信息 | 备选；两种完整表示都正确 |
| [formal_007](cases/formal_007/README.md) | 未送达消息时，SSC保留table_30，prose输出未观察中间位置couch_61 | 对SSC有利的真实局部差异，**事后选例，不代表总体优势** |
| [formal_019](cases/formal_019/README.md) | SSC在目睹/未目睹变体中出错，prose四种历史均正确 | 反向案例，避免只保留有利结果 |

每例提供`history_panel.png/pdf/svg`、完整`case.json`以及16组原始请求/响应。
绿/红字只标目标状态正确性；完整世界/无关主体/引用评分在JSON中。
[`case_selection.json`](case_selection.json)列出全部40例结果与事后选择说明。
**这些图是构造文本实验的记录可视化，不是原Habitat物理轨迹，也不是视频画面。**

## 必须随结果保留的范围说明

- 原EnactToM任务只提供实体词汇及来源ID，未执行原任务、原轨迹、真实视线或视频。
  “20个scene”是正式样本的来源标签，**不是20个物理场景的反事实实验**。
- 使用现有StateStore和新factorized readout适配器；不是历史Habitat planner输入的逐字复现。
- 四种历史的事件状态机提供真值；观察表示确认获取信息，消息假设真实可信。
  这不是对真实人类注意、信任或信念的独立标注。
- prose是同语义确定性序列化，不是自行修订的动态NL摘要器。不能据此宣称超过一般memory。
- 来源episode局部排重不等于全局从未见过；共享事件模板限制泛化，40例不是功效保证。
- 工程期只修正了语义错误与格式无效混淆，88份回答直接复用；正式开始前完成冻结。
  [修订审计](../minimal_mechanism_package_20260923/engineering_cache_reuse_audit.json)。

## 公开快照与复核

```text
evidence/dataset.json.gz             50基础情节、200变体、输入与状态机真值
evidence/selection_audit.json.gz     全候选、排除来源、预先固定的工程/正式划分
evidence/all_calls.jsonl.gz          全800次工程/正式答题及4个预检的原始文件文本和SHA256
evidence/frozen_config.public.json   公开配置，移除凭据文件/锁路径，不包含key值
evidence/provenance.json             原配置hash、数据hash、精确冻结源码hash
frozen_source/                      真正用于本轮评分和表示构造的代码快照
formal_summary.json                 正式全指标、配对区间和40例逐例指标
engineering_summary.json            工程单列，不混入正式结果
SHA256SUMS.json                     此目录公开文件完整性清单
```

原始私有配置的hash被保留用于映射；公开配置经路径脱敏，不能假称两者字节相同。
源身份使用本轮冻结源文件SHA，而不是为历史运行补造git commit。

无需原数据、私有workdirs或API key即可核验全部输入、800个答案、评分、配对统计、成本：

```bash
python -m experiments.verify_information_history_publication
python -m unittest discover -s tests -p 'test_information_history_publication.py'
```

依赖：Python、numpy；绘图另需matplotlib。仅重画，不调用任何模型或模拟器：

```bash
MPLCONFIGDIR=/tmp/ssc-mpl python -m experiments.publish_information_history_20260923 --from-export
```

已有Habitat/CoMind/MMToM可信度核验位于
[最小机制资料包](../minimal_mechanism_package_20260923/README_zh.md)。
特别是历史MMToM的166处Direct原文—保存预测不一致必须单列，不能拿旧比较替代本轮证据。
