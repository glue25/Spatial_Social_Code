# CoMind Gemini 3 Flash：完整300题 Direct / SSC＋SAM 与独立失败恢复

主实验完成：2026-09-21 07:29 UTC。每任务100题，共300题，全部保留在分母。
这是 CoMind **train 开发子集**的历史方法复现及感知增强，不是官方隐藏测试成绩；下述14指标宏平均也不是官方综合分。

## 主结果：冻结300题，不拼接恢复结果

| 汇总 | Direct | SSC＋SAM | 差值 |
|---|---:|---:|---:|
| **14字段等权宏平均** | **17.21%** | **22.21%** | **+5.00 pp** |
| Joint Attention（4字段平均） | 9.25% | 26.50% | +17.25 pp |
| SCOIA（4字段平均） | 17.00% | 16.25% | −0.75 pp |
| Handover（6字段平均） | 22.67% | 23.33% | +0.67 pp |

14指标平均等于241/1400与311/1400个字段命中（每任务100题）。这些相关字段不是1400个独立试验，不将其用于独立伯努利显著性检验。

| 任务／字段 | Direct | SSC＋SAM |
|---|---:|---:|
| Joint Attention left IoU@.5 | 1% | 26% |
| Joint Attention right IoU@.5 | 1% | 26% |
| Joint Attention cue type | 35% | 41% |
| Joint Attention category L1 | 0% | 13% |
| SCOIA IoU@.5 | 1% | 9% |
| SCOIA cue type | 30% | 21% |
| SCOIA action verb | 8% | 8% |
| SCOIA action noun L1 | 29% | 27% |
| Handover IoU@.5 | 0% | 11% |
| Handover delivery flow | 54% | 48% |
| Handover initiator | 52% | 48% |
| Handover initiation type | 25% | 26% |
| Handover category L1 | 0% | 0% |
| Handover time within0.25s | 5% | 7% |

定位提升明显，但SCOIA的cue/noun、Handover的flow/initiator回归，不宣称全面优势。沿用冻结评分器：IoU使用严格`>0.5`；JA缺少right gold时仍计未命中，不删除该样本。

## 方法与比较边界

- 模型请求及接受响应报告`gemini-3-flash-preview`；这是服务商返回身份，不是独立权重认证。历史对应的300题闭源CoMind是GPT-5.6-Sol，**不是Gemini3Pro**；其基础版与SAM增强版也不混称。
- Direct与SSC看相同冻结多视角RGB帧、任务文本及预测时刻前10秒转录。SSC按原任务条件生成Spatial code、Social code，执行为trace，再解码字段。原schema通过原生Gemini JSON接口传递；normalizer、解析与执行规则不放宽。
- 主实验所有Direct/Spatial/Social/final阶段统一16384输出上限，temperature0；两个固定key各150题，最多10次传输尝试、600秒超时、每key一个在途请求。额度耗尽后暂停、充值后从确切请求断点继续；不重试已完成的语义失败。没有更换模型或删除额度失败尝试记录。
- SAM3.1从**本轮Flash自己的SSC trace**提取最高置信对象候选，按固定规则回退到SSC最终类别，不读取gold、不使用历史GPT答案或训练后的路由器。独立处理各预测视角并选择最高置信的合法框，只替换bbox。Depth没有参与。
- JA/Handover的独立视角1408×1408像素框转为官方半尺度水平拼接坐标，helper侧x加704；SCOIA保留独立helper坐标。Handover依据SAM候选分数选视角，不根据gold或得分选择。
- **4题内部SSC截断无效，7题至少一个视角未检出**，全部保留。SAM缺框计该bbox未命中，其他有效SSC字段仍计分；不回退为Direct或旧框。原始SSC中间结果保留用于审计，但主表按用户选择只展示Direct和SSC＋SAM。
- 看过中间结果后才添加SAM、选择双条件主表并做失败恢复，是开发性分析，不是未触碰的预注册确认集。SSC＋SAM额外使用感知工具/计算量；没有Direct＋SAM控制，不能把全部增益解释成SSC表征本身的因果效果。

## 独立恢复实验

全部4道SSC截断题：**94、113、236、245**，按技术无效选择，不按预测对错选择。
每题重新生成Direct与SSC，再运行SAM。恢复所有生成阶段统一提高为32768，其他协议不变；只做一轮，不反复采样直到成功。原300题的回答、code、失败及主分数全部不替换。

恢复结果见[`recovery/summary.json`](recovery/summary.json)与[`RECOVERY.md`](RECOVERY.md)。它是失败选择、提高预算的小队列诊断，不能作为新的无偏300题总分。

## 公开内容与验证

| 路径 | 内容 |
|---|---|
| `primary/results.jsonl.gz` | 全300题Direct、SSC完整code/trace、SSC＋SAM预测、SAM候选与失败 |
| `primary/calls.jsonl.gz` | 全部逻辑调用的实际文本prompt、schema、帧描述符、可见回复、模型/usage及尝试记录 |
| `primary/artifacts.jsonl.gz` | 分阶段生成JSON和执行结果 |
| `primary/sam_calls.jsonl.gz` | SAM原始框/分数/标签、请求身份、视角、mask哈希和失败尝试 |
| `primary/inputs.jsonl.gz` | 冻结300题清单与帧路径/顺序；原始媒体需自行取得 |
| `primary/scores.jsonl.gz` | 每题每字段的评分布尔值，失败保留；可重算全部表格 |
| `recovery/` | 单独4题恢复的同类完整产物，不混入primary |
| `code/` | 实际冻结编译器/runtime/schema/transport、SAM sidecar、恢复程序及相关历史依赖源码 |
| `provenance/`、`SHA256SUMS.json` | 原始源文件哈希、导出映射与发布内容哈希 |

不上传凭据/私钥、图像视频、原始像素mask、供应商opaque thoughtSignature，或整套数据集标注。保留mask哈希/shape及完整框候选。JSON机器项目路径替换为`${PROJECT_ROOT}`，凭据路径替换为占位符。冻结源码按原字节保留：其中历史机器路径仅供审计，旧启动器不保证换目录直接运行；真实重跑需合法数据、模型服务和原依赖环境。

独立可移植的离线入口（仅Python标准库，无需key/GPU/图片）：

```bash
python verify.py
python replay.py
# 如具备原始scorer-only gold JSON，可进一步独立核对全部字段分数：
python verify.py --gold /path/to/SCORER_ONLY_comind_gold.json
# 或从单独取得的CoMind训练标注构建同内容的scorer-only文件：
python prepare_gold.py --annotations /path/to/comind/annotations --output /tmp/comind_gold.json
python verify.py --gold /tmp/comind_gold.json
```

`verify.py`核对哈希/覆盖/失败/code执行/SAM候选与坐标/字段不变性，从保存的逐字段评分重算主表。默认不重新认证gold；`--gold`先检查原标签文件或规范化JSON内容哈希，再逐字段独立评分。`prepare_gold.py`只读训练标注，不下载数据，也不将标签传给推理。
`replay.py`调用冻结原编译与解码代码，逐字核对实际prompt、schema、帧描述符顺序、输出code/trace和预测；图像字节读取以存档描述符代替，因此不是媒体或模型重推理认证。
原数据权利及许可仍归CoMind数据发布方；本包不授予底层媒体/标注的额外权利。
