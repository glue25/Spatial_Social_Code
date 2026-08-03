# Spatial Social Code 实验环境准备状态

更新日期：2026-07-31

## 1. 当前结论

第一轮实验所需的三类互补环境已经准备好：

| 环境 | 本地状态 | 可立即进行的实验 | 暂未准备的部分 |
|---|---|---|---|
| HoloAssist | 标注、划分、官方代码已就绪 | 事件账本、物理痕迹、对话—动作对齐、oracle state | 原始视频、深度、手势与眼动流 |
| MA-EgoQA | 1,741 道问题、字幕、shared memory、代码已就绪 | caption/memory/state-code QA；多主体证据归属 | 约 512 GB 的 EgoLife 原始视频 |
| EnactToM | 300 个任务定义、PDDL、golden trajectory、代码已就绪 | oracle reducer、功能性 ToM、干预与反事实重放 | macOS 本机不能完整运行的 Habitat/HSSD 环境 |
| EgoTaskQA | 代码已就绪 | 先做 schema 与 evaluator 对接 | 数据需人工签署许可协议 |
| CoMind | 全局 annotations 与 1 个 test recording 双人 transcript pilot 已就绪 | joint attention、handover、socially conditioned anticipation；双主体信息归属 | 单条完整 recording 约 180.65 GiB，视频/MPS/SLAM/scan 未下载 |

这里的“已就绪”只表示对应层级的数据可用于实验，不等同于所有原始模态与完整模拟器都已安装。

### E0 更新（2026-07-31）

EnactToM oracle-state E0 已完成：

- 300 个 task entries、289 个唯一 task ID；
- 1,013 个 planner-selected physical literals；
- 317 个一阶 literal-ToM probes；
- query-independent Physical Code + agent-indexed Interaction Code；
- `Navigate/Pick/Open/Place/Wait` golden trajectory reducer；
- 消息删除与重定向的 Target Sensitivity / Non-target Invariance。

291/300 个任务完成全部物理目标重放。其余 9 个公开任务的保存轨迹只有 Navigate/Wait，没有实现目标所需的 effectful action。另有 83 个 golden navigation steps 的目标房间与静态 room restriction 元数据冲突；这项结果需要在 Linux Habitat runtime 中复核，当前只作为数据一致性诊断。

复现入口：

```bash
python3 scripts/run_enacttom_e0.py
```

## 2. 已验证的数据

### HoloAssist

- 标注 JSON：1,758 个 recording records。
- 嵌套事件：195,945 条，其中包含 fine/coarse action、conversation 与 narration。
- 官方划分：train 1,466、validation 207、test 438。
- 标注与划分压缩包均通过 SHA-256 校验。

建议先用标注时间线构建：

\[
\text{event}
\rightarrow
\text{evidence ledger}
\rightarrow
\text{physical/task state}
\rightarrow
\text{intervention prediction}.
\]

完成表示层消融后，再冻结一小批视频 ID 下载对应 RGB、hand pose 或 gaze，进行 predicted-state 实验。

### MA-EgoQA

- 1,741 道多主体长视频问题。
- 五类问题：social interaction、task coordination、theory of mind、temporal reasoning、environmental interaction。
- 127 个不同时间粒度的 caption JSON。
- 官方 `10min_shared_memory.json` 可作为 Markdown/shared-memory baseline。

这套数据最适合先比较：

1. question only；
2. captions；
3. shared memory；
4. Spatial Social Code；
5. oracle evidence assignment。

### EnactToM

- Standard 150 个任务，Hard 150 个任务。
- 每个任务均含 private information、message targets、PDDL、golden trajectory 与 literal ToM probes。
- 当前可以直接做文本/PDDL 状态解析、hidden unit tests、counterfactual replay 与 action selection。
- 完整 Habitat/HSSD 执行依赖 Linux、CUDA 与额外场景资产；应放到单独的 Linux GPU 环境，通过官方入口 `./enacttom/run.sh` 运行。

## 3. 受限环境与新下载

### EgoTaskQA

官方页面要求先填写 license agreement。代码已经下载到本地，但在许可完成前不绕过访问控制抓取数据。完成申请后，把文件放到 `external_datasets/egotaskqa/`，再补充校验项。

### CoMind

官方服务器已于 2026-07-31 恢复访问，现已冻结官方 test split 的第一个 recording：

```text
f0b49334-565a-4d30-a010-15d92e511c9a
```

已下载并通过官方 SHA-256 manifest 校验：

- 三套全局 annotations：handover 77、joint attention 78、SCOIA 79 条；
- helper transcript：439 segments / 2,644 words；
- leader transcript：345 segments / 2,769 words；
- transcript 的带 profanity 版本与 ASS 字幕，共 8 个 transcript files；
- 实际选择的数据约 8.72 MiB。

该 recording 的完整 manifest 大小为 180.65 GiB，其中 VRS 97.96 GiB、MP4 29.96 GiB、GoPro 21.83 GiB、MPS 15.11 GiB、Multi-SLAM 14.94 GiB。因此当前只保留文本和标注；先验证双主体 Interaction Code，再决定是否增加视频或 gaze。

复现命令：

```bash
python3 external/CoMind/comind_download.py \
  experiments/configs/comind_pilot_ids.txt \
  --target external_datasets/comind \
  --no-meshes \
  --parts transcripts,annotations \
  --yes
```

## 4. 推荐的实验启动顺序

1. **E0：EnactToM oracle state。** 不依赖视频感知，先验证 ledger、reducer、belief staleness、common-ground 与反事实语义。
2. **E1：MA-EgoQA caption/memory。** 固定同一推理器，比较 raw captions、官方 shared memory、Spatial Code 与 Spatial Social Code。
3. **E2：HoloAssist annotation-only。** 检验物理痕迹、信号可达性、任务进度和干预类型。
4. **E3：冻结小规模 raw-video pilot。** 根据 E1/E2 的错误类型选视频，而不是先下载数百 GB 全量数据。
5. **E4：Linux GPU 上的 EnactToM functional evaluation。**
6. **E5：CoMind transcript/annotation pilot。** 先做双主体信息边界、joint attention 与 handover schema；通过后再选择性补视频/MPS。

## 5. 复现入口

环境清单：

```text
experiments/configs/datasets.yaml
```

补齐可匿名下载的资源：

```bash
bash scripts/prepare_datasets.sh
```

验证本地文件、哈希和样本数：

```bash
python3 scripts/validate_datasets.py
```
