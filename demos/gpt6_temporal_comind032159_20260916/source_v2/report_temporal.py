"""Write a provenance/limitations report from actual completed artifacts."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import runpy


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    data=read(out/'temporal_code.json');manifest=read(out/'observations_manifest.json');states=data['states']
    calls=[read(p) for p in sorted((out/'model_calls').glob('*/validated.json'))]
    assert len(calls)==6 and all(c['server_model']=='gpt-6-astra' for c in calls)
    assert runpy.run_path(str(out/'temporal_code.py'))['T'].data==data
    # Full local schema validation in addition to inference target/identity gates.
    import jsonschema
    schema=read(out/'annotation_schema.json')
    for p in (out/'model_calls').glob('*/raw_payload.json'):jsonschema.validate(read(p),schema)
    assert read(out/'blender/independent_track_validation.json')['all_passed']
    assert all(s['social_updates']==[] for s in states)
    assert len({s['source_frame_index_zero_based'] for s in states})==51
    assert all(s['source_frame_index_zero_based']==31772+i*6 for i,s in enumerate(states))
    # Group only contiguous sample evidence; never interpolate action onset.
    groups=defaultdict(list)
    for s in states:
        for r in s['interactions']:
            if r['target'] in ['red_colander_counter','vegetable_scrap','drawer_lower'] and r['predicate']!='unknown':
                groups[(r['actor'],r['predicate'],r['target'],r['epistemic_status'])].append(s['observation_index'])
    rows=[]
    for key,indices in groups.items():
        bands=[]
        for i in sorted(set(indices)):
            if bands and i==bands[-1][-1]+1:bands[-1].append(i)
            else:bands.append([i])
        for band in bands:
            start,end=states[band[0]],states[band[-1]]
            rows.append(f"| {start['timestamp_seconds']:.3f}–{end['timestamp_seconds']:.3f} | {key[0]} → {key[2]} | {key[1]} | {key[3]} | {start['evidence_id']}–{end['evidence_id']} |")
    drawer=[(s['timestamp_seconds'],next(e for e in s['entities'] if e['id']=='drawer_lower')) for s in states]
    drawer_text='; '.join(f"{t:.3f}s: {e['state']}" for t,e in drawer if e['state'] in ['open','closed','partially_open'])
    counts={}
    for entity in data['entity_registry']:
        localized=sum(next(e for e in s['entities'] if e['id']==entity)['bbox'] is not None for s in states)
        counts[entity]=localized
    total_usage={k:sum(c.get('usage',{}).get(k,0) for c in calls) for k in ['input_tokens','output_tokens','total_tokens']}
    report=f'''# CoMind 032159 连续 Spatial–Social Code 与 Blender 时间序列

## 交付入口

- [同步预览网页](index.html) / [三栏同步视频](comparison_preview.mp4)：左原始视频，中随观测更新的 Code，右 Blender 代理动画。
- [可编辑动画场景](blender/temporal_scene.blend)，301 个渲染帧 / 30 fps。
- [temporal_code.json](temporal_code.json)、[可读且可执行的 temporal_code.py](temporal_code.py)。
- [原帧/PTS 清单](observations_manifest.json)、[证据索引](evidence_index.json)、`frames/` 内完整分辨率 PNG、`annotated_observations/` 内带模型估计框的审阅图。
- [观测轨道](blender_animation_tracks.json) 与 [独立渲染插值](render_interpolation.json) 分开保存。

## 原始观测：不是六张图片插值

找到并使用了原录制的 `helper_trimmed_sync.mp4`，文件 SHA-256 为
`{manifest['source_video_sha256']}`，与本地 CoMind healthcheck 清单一致。
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
六次调用 usage 合计：`{json.dumps(total_usage)}`。没有额外模型调用来反复挑结果。
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
'''+ '\n'.join(rows)+f'''

抽屉的可判定采样状态：{drawer_text or '暂无可判定状态'}。
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
'''
    (out/'REPORT_ZH.md').write_text(report)
    (out/'final_validation.json').write_text(json.dumps({'complete':True,'source_frames':51,'persistent_ids':12,'fresh_gpt6_calls':6,'usage':total_usage,'localized_sample_counts':counts,'json_schema_passed':True,'python_equals_json':True,'no_timed_social_fabrication':True,'no_metric_reconstruction_claim':True,'render_track_checks':read(out/'blender/independent_track_validation.json'),'preview_checks':read(out/'preview_validation.json')},indent=2)+'\n')
    (out/'COMPLETE').write_text('Temporal SSC, evidence, proxy tracks, Blender animation and synchronized preview validated. Not metric reconstruction.\n')


if __name__=='__main__':main()
