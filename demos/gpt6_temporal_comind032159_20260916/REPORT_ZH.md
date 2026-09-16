# CoMind 032159 连续 Spatial–Social Code 与 Blender 时间序列

## 交付入口

- [同步预览网页](index.html) / [三栏同步视频](comparison_preview.mp4)：左原始视频，中随观测更新的 Code，右 Blender 代理动画。
- [可编辑动画场景](blender/temporal_scene.blend)，301 个渲染帧 / 30 fps。
- [temporal_code.json](temporal_code.json)、[可读且可执行的 temporal_code.py](temporal_code.py)。
- [原帧/PTS 清单](observations_manifest.json)、[证据索引](evidence_index.json)、`frames/` 内完整分辨率 PNG、`annotated_observations/` 内带模型估计框的审阅图。
- [观测轨道](blender_animation_tracks.json) 与 [独立渲染插值](render_interpolation.json) 分开保存。

## 原始观测：不是六张图片插值

找到并使用了原录制的 `helper_trimmed_sync.mp4`，文件 SHA-256 为
`402e4ae5b5624384ecb6c271f809ff6047b49270be46397d81f594e36f55f42a`，与本地 CoMind healthcheck 清单一致。
原视频 1408×1408、30 fps。核对全部 45,504 个视频 packet PTS 为唯一连续 CFR，
并核对所选解码帧 showinfo PTS；帧号不是仅凭时间戳猜测。

请求区间 1059.067–1069.067 秒按最近原帧对齐为
**1059.066667–1069.066667 秒**，两端均包含。采样原帧（零基）
**31772, 31778, …, 32072，共 51 张**，每隔 6 原帧，即 5 fps。
清单同时给出一基帧号、PTS、time_base 和时间戳。原片段预览保留原始 30 fps，
包含末帧的一个显示帧，因此文件时长约 10.033 秒。预览重编码用于播放，不作为新观测。
模型分析使用各原帧的 704×704 缩小副本，原始 PNG 保留；没有生成/插值任何观测帧。
对照预览未包含音轨；没有从音轨臆测说话者或逐词时间。

## 模型与可追溯性

六次新视觉分析调用请求 `gt-6-as-a`，服务端均返回 `gpt-6-astra`。
使用固定实体名册、相邻上下文帧及首尾身份参考图；这是**离线回顾式开发案例标注**，
不是因果在线跟踪或盲测。身份参考图可能来自当前时刻之后，因此不能据此声称在线能力。
原始响应、模型输出、提示词、输入图哈希及 usage 保存在 `model_calls/`；没有用人工夹具替代模型输出。
六次调用 usage 合计：`{"input_tokens": 57565, "output_tokens": 58254, "total_tokens": 115819}`。没有额外模型调用来反复挑结果。
标签 observed 表示模型判断有直接图像支持，**不是独立人工真值认证**；图像框是近似视觉估计。
保守编译门控（例如不可见状态归 unknown、未来参考不能标当前 observed）全部记入
[audit_corrections.json](audit_corrections.json)，原始模型结果不改写。

## 身份、运动和 Social 边界

固定 12 个实体 ID，所有 51 个时刻均保留它们。尤其区分 `red_colander_counter`
（台面带孔滤篮）、`red_bowl_drawer`（抽屉实心红碗）、`nested_metal_bowl`，以及
`vegetable_scrap`（小残片）与 `cabbage_head`（整颗卷心菜）。手部按相机佩戴者左右手和
另一人的腕饰线索区分；遮挡后的连续身份仍可能不确定。

不可见时 bbox=null、状态 unknown（除有明确可见支持）；不输出“物体被删除”。
观测状态、图像位置、关系、手部交互和抽屉状态变化均带来源或证据引用。
改变只落在采样证据支持的时间区间，**没有亚帧精确动作起止时间**。

相机运动单独记录。`camera_motion_diagnostics.json` 提供背景特征 RANSAC 的二维
变换诊断，仅是 inferred 图像变换，存在视差、遮挡和鱼眼误差；不当作米制相机姿态，
不直接拿它计算真实物体速度。Blender 采用保守 **2.5D 图像平面代理**：
坐标单位任意，深度/朝向/网格均为渲染默认，屏幕位移（包含相机效应）不等于物体世界位移。
显示相机固定且明确不是恢复的原相机；没有用不可靠的三维重建制造动作。

Social Code 只保留已有、无逐词/说话人时间对齐的转录上下文。
`speaker=null`、`timestamp_seconds=null`，**每帧 social_updates 均为空**。
没有把需要大碗的话绑定到任何可见人物，没有编造信念变化、递交意图或对话轮次。

## 重点交互的采样证据

下表是模型在实际采样帧中报告的交互证据段，**不是精确动作起止标注**。
重复/短暂间断按采样证据保留；未覆盖的区间不自动填充。详见
[interaction_evidence.json](interaction_evidence.json)。

| 有证据的采样时段（秒） | 主体→对象 | 交互 | 来源 | 证据 |
|---|---|---|---|---|
| 1059.067–1061.867 | wearer_left_hand → vegetable_scrap | holding | observed | F031772–F031856 |
| 1063.267–1067.467 | wearer_left_hand → vegetable_scrap | holding | observed | F031898–F032024 |
| 1068.867–1069.067 | wearer_left_hand → vegetable_scrap | holding | observed | F032066–F032072 |
| 1060.467–1060.667 | wearer_left_hand → vegetable_scrap | moving_with_hand | observed | F031814–F031820 |
| 1063.067–1063.067 | wearer_left_hand → vegetable_scrap | holding | inferred | F031892–F031892 |
| 1064.067–1064.067 | wearer_left_hand → red_colander_counter | reaching_toward | inferred | F031922–F031922 |
| 1064.267–1067.267 | wearer_left_hand → red_colander_counter | holding | observed | F031928–F032018 |
| 1065.067–1065.467 | wearer_left_hand → red_colander_counter | moving_with_hand | inferred | F031952–F031964 |
| 1067.267–1067.267 | wearer_left_hand → red_colander_counter | moving_with_hand | inferred | F032018–F032018 |
| 1065.467–1065.467 | wearer_right_hand → red_colander_counter | reaching_toward | inferred | F031964–F031964 |
| 1065.667–1065.667 | wearer_right_hand → red_colander_counter | holding | observed | F031970–F031970 |
| 1067.067–1067.267 | wearer_right_hand → red_colander_counter | holding | observed | F032012–F032018 |
| 1065.667–1065.667 | wearer_left_hand → red_colander_counter | moving_with_hand | observed | F031970–F031970 |
| 1067.467–1067.467 | wearer_left_hand → red_colander_counter | moving_with_hand | observed | F032024–F032024 |
| 1065.867–1065.867 | wearer_right_hand → red_colander_counter | touching | observed | F031976–F031976 |
| 1066.867–1066.867 | wearer_right_hand → red_colander_counter | touching | observed | F032006–F032006 |
| 1066.067–1066.067 | wearer_right_hand → red_colander_counter | releasing | observed | F031982–F031982 |
| 1067.467–1067.467 | wearer_right_hand → red_colander_counter | releasing | observed | F032024–F032024 |
| 1066.267–1066.667 | wearer_right_hand → red_colander_counter | reaching_toward | observed | F031988–F032000 |
| 1067.267–1067.267 | wearer_right_hand → red_colander_counter | moving_with_hand | observed | F032018–F032018 |
| 1067.867–1068.067 | other_person_bracelet_hand → drawer_lower | reaching_toward | inferred | F032036–F032042 |
| 1068.267–1068.267 | other_person_bracelet_hand → drawer_lower | touching | observed | F032048–F032048 |
| 1068.467–1068.467 | other_person_bracelet_hand → drawer_lower | holding | inferred | F032054–F032054 |
| 1068.667–1069.067 | other_person_bracelet_hand → drawer_lower | moving_with_hand | observed | F032060–F032072 |
| 1069.067–1069.067 | other_person_other_hand → drawer_lower | reaching_toward | observed | F032072–F032072 |

抽屉的可判定采样状态：1067.867s: closed; 1068.067s: closed; 1068.267s: closed; 1068.467s: closed; 1068.667s: partially_open; 1068.867s: partially_open; 1069.067s: open。
变化区间在 `temporal_code.json` 的 `state_changes` 中，精确变化时刻仍为 null。

## 渲染插值不冒充观测

仅在相邻且同 ID 两端都可定位时做渲染用线性插值；跨遮挡/缺失框不插值。
灰色代理仅表示“最后看见的位置”，不是当前隐蔽位置；从未定位的实体保留 ID 和
Blender 对象，但不显示任意猜测的位置。抽屉开闭用单独、阶梯式代理状态驱动图示，
不是估计出的真实滑轨行程。

## 验证与限制

- 6 项时序单元测试通过：插值不修改观测、遮挡不删除实体、不跨缺失框插值、
  Python 与 JSON 一致、身份/隐藏坐标门控、不同红色容器 ID 区分。
- 所有原始模型输出通过本地 JSON Schema 校验；51 个时刻及实体集合完整。
- Python 可读代码执行结果与 JSON 完全一致。
- [Blender 独立轨道检查](blender/independent_track_validation.json)逐帧核对保存场景的位置/尺寸、
  可见性持久化和抽屉代理状态；这仅验证驱动链路，不是模型标注准确率。
- [同步检查](preview_validation.json)确认原视频、Blender 和合成对照均为 301 帧、30 fps。

无法可靠恢复：被遮挡时的具体位置/姿态、所有手指的接触与抓握力度、残片是否在遮挡中发生
分裂/换手、滤篮与台面首次接触及松手的亚采样精确时刻、抽屉隐藏期间的状态与真实开度、
真实相机内外参和米制三维轨迹、人物真实身份、转录说话人与逐词时间、心理状态或交流过程。
单目强相机运动下的三维运动不可唯一恢复；本结果没有宣称完成这件事。

论文正文未修改。原始观测/模型响应保留，旧静态 demo 和 Flash 后台评测不受替换。

## 人工预览复核与工程修复记录

已审阅全部 51 帧联系表，以及起点、中点、终点三栏预览。这里的名册仅覆盖 12 个任务相关实体，并非全场景对象清单。
滤篮在后段移到台面/水槽附近，但左手完整松开及篮底最终受支撑情况受裁切遮挡影响；没有把右手释放自动当成完成放置。
`temporal_code.json` 在 1067.467–1067.867 秒等时刻保留相关 uncertainties，不能从动画补出这些事实。

独立 Blender 检查发现初版自定义开度属性被整数类型四舍五入，导致 partially_open=0.5 读成 1。
已按相同导出轨道重建浮点、阶梯式属性通道，未修改模型输出、物体位置、尺寸、已烘焙的前面板可见性或渲染图像。
修复前场景及说明保存在 `history/`，原检查阈值不变；修复后 301 帧全部通过。
新版完整生成器为 `source_v3/blender_temporal.py`；其余流水线源码在 `source_v2/`。

打开 `blender/temporal_scene.blend`，时间轴范围为 1–301，按空格播放或拖动时间轴；进入相机视图可看到代理布局。
每个实体保留具名根对象，可检查图像平面位置、尺寸、visibility_code 和 ghost_not_current_position；
抽屉还有 drawer_open_state_proxy（-1未知、0关闭、0.5部分打开、1打开），这是图示状态编码，不是物理开度测量。
Blender Text 数据块中嵌入观测 Code、JSON 和独立插值轨道，便于继续编辑。
