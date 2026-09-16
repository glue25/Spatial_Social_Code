"""Synchronize original 30fps clip, held 5fps code states, and proxy animation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

import cv2
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    data=json.loads((out/'temporal_code.json').read_text()); states=data['states']; ff=imageio_ffmpeg.get_ffmpeg_exe()
    cmd=[ff,'-hide_banner','-framerate','30','-i',str(out/'blender/frames/frame_%04d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(out/'blender_animation.mp4')]
    with (out/'encode_blender.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True)
    mono='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'; regular='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    f=ImageFont.truetype(mono,13); small=ImageFont.truetype(regular,14); title=ImageFont.truetype(regular,20)
    panels=[]
    for s in states:
        im=Image.new('RGB',(640,640),'#0c1421');d=ImageDraw.Draw(im)
        d.text((18,12),f"t={s['timestamp_seconds']:.3f}s   frame={s['source_frame_index_zero_based']}",font=title,fill='white')
        d.text((18,44),f"REAL sample {s['observation_index']:02d}/50 | {s['evidence_id']} | 5 fps",font=small,fill='#66d9f2')
        y=76
        for e in s['entities']:
            status=e['epistemic_status'][0].upper(); b=e['bbox']; box='?' if b is None else ','.join(f'{x:.2f}' for x in b)
            color={'O':'#9cecbc','I':'#ffcc88','U':'#8fa3bc'}[status]
            d.text((18,y),f"{status} {e['id']:<27} {e['visibility']}",font=f,fill=color); y+=14
            d.text((45,y),f"bbox=[{box}]"+(f" state={e['state']}" if e['id']=='drawer_lower' else ''),font=f,fill='#9baac0');y+=15
        d.text((18,y+2),'S.interactions (sample evidence, not exact onset):',font=f,fill='#66d9f2');y+=22
        for r in s['interactions'][:3]:
            text=f"{r['epistemic_status'][0].upper()} {r['actor']} {r['predicate']} {r['target']}"
            for line in textwrap.wrap(text,width=70)[:2]:d.text((18,y),line,font=f,fill='#dce4ee');y+=16
        d.text((18,598),f"camera={s['camera_motion']['status']} | world displacement=UNKNOWN",font=f,fill='#ffcc88')
        d.text((18,620),'Social: no timed update; speaker/time alignment UNKNOWN',font=f,fill='#99b8d0')
        panels.append(im)
    source=cv2.VideoCapture(str(out/'original_clip.mp4')); assert source.isOpened()
    command=[ff,'-hide_banner','-f','rawvideo','-pix_fmt','rgb24','-s','1920x736','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(out/'comparison_preview.mp4')]
    with (out/'encode_comparison.log').open('w') as log:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=log)
        for k in range(301):
            ok,bgr=source.read(); assert ok,(k,'original clip ended early')
            canvas=Image.new('RGB',(1920,736),'#070e19');d=ImageDraw.Draw(canvas)
            for x,heading in [(18,'ORIGINAL VIDEO | 30 fps'),(658,'SPATIAL-SOCIAL CODE | real 5 fps'),(1298,'BLENDER | proxy + rendering interpolation')]:d.text((x,12),heading,font=title,fill='white')
            canvas.paste(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((640,640)),(0,48))
            canvas.paste(panels[min(k//6,50)],(640,48))
            with Image.open(out/f'blender/frames/frame_{k+1:04d}.png') as im:canvas.paste(im.convert('RGB'),(1280,48))
            t=states[0]['timestamp_seconds']+k/30
            d.text((18,703),f"source t={t:.3f}s | O observed / I inferred / U unknown | NO metric 3D claim | interpolation adds ZERO observations",font=title,fill='#b9cadc')
            proc.stdin.write(canvas.tobytes())
            if k in [0,150,300]:canvas.save(out/f'comparison_frame_{k:03d}.jpg',quality=92)
        proc.stdin.close(); assert proc.wait()==0
    source.release()
    (out/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CoMind 032159 temporal SSC</title><style>body{background:#0b1320;color:#e4edf8;font:17px system-ui;margin:30px auto;max-width:1600px}a{color:#79d4ff}video{width:100%;border-radius:12px}p{line-height:1.65}</style><h1>CoMind 032159 · 连续 Spatial–Social Code</h1><p>51 张真实观测 / 5 fps；原视频与代理动画 30 fps。左：原视频；中：按观测更新的 Code；右：Blender 图像平面代理动画。插值不增加观测，画面位移不等于真实物体运动，灰色物体仅为最后可见位置的占位。</p><video controls preload="metadata" src="comparison_preview.mp4" poster="comparison_frame_150.jpg"></video><p><a href="blender/temporal_scene.blend">下载可编辑动画 .blend</a> · <a href="temporal_code.py">可读 Code</a> · <a href="temporal_code.json">时序 JSON</a> · <a href="observations_manifest.json">原帧清单</a> · <a href="evidence_index.json">证据索引</a> · <a href="blender_animation_tracks.json">观测动画轨道</a> · <a href="render_interpolation.json">独立渲染插值</a> · <a href="REPORT_ZH.md">限制与验证报告</a></p><p>这是回顾式开发案例分析，不是米制重建或在线盲测。没有时间对齐的说话人证据，Social Code 不编造时序交流。</p></html>''')
    checks={}
    for name in ['original_clip.mp4','blender_animation.mp4','comparison_preview.mp4']:
        cap=cv2.VideoCapture(str(out/name)); checks[name]={'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'fps':cap.get(cv2.CAP_PROP_FPS),'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))};cap.release()
        assert checks[name]['frames']==301 and checks[name]['fps']==30
    (out/'preview_validation.json').write_text(json.dumps({'synchronized_streams':checks,'observation_count':51,'timed_social_updates':0,'frame_zero_source_time':states[0]['timestamp_seconds']},indent=2)+'\n')
    print(json.dumps(checks))


if __name__=='__main__':main()
