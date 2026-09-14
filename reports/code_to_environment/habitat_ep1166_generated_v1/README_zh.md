# SSC → Habitat 环境实例化与闭环执行

本实验不是只把现有环境转写成 JSON。编译器仅以 Spatial–Social Code 为输入，
生成独立的 Habitat episode 与任务文件，再由 Habitat 实际加载、渲染和执行。

## 生成边界

- 生成了新的 episode ID、资产实例位姿、任务绑定、agent 配置与社会信息边界。
- 使用已有 HSSD 房间壳和资产网格；没有声称生成全新 mesh。
- 输入中不含 golden trajectory、Oracle schedule 或目标答案。

## 执行结果

| 条件 | 生成 episode | 严格完成 | 轮数 | 实际帧数 |
|---|---:|---:|---:|---:|
| direct | 911166 | 0/3 | 20 | 306 |
| dynamic_ssc | 911166 | 3/3 | 10 | 174 |
| static_ssc | 911166 | 1/3 | 20 | 324 |

编译与执行验收：`accepted=true`。
