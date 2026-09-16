"""Extract actual CoMind source frames, preserving source PTS and frame indices."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageDraw


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    video = a.video.resolve()
    env = dict(os.environ, LD_LIBRARY_PATH='/opt/conda/pkgs/openh264-2.1.1-h4ff587b_0/lib')
    probe = ['/opt/conda/bin/ffprobe', '-v', 'error']
    def query(args):
        return json.loads(subprocess.check_output(probe + args + ['-of', 'json', str(video)], env=env))
    meta = query(['-show_streams', '-show_format'])
    stream = next(s for s in meta['streams'] if s['codec_type'] == 'video')
    assert stream['avg_frame_rate'] == stream['r_frame_rate'] == '30/1'
    packets = query(['-select_streams', 'v:0', '-show_packets', '-show_entries', 'packet=pts,duration'])['packets']
    pts = sorted(int(p['pts']) for p in packets)
    tb = Fraction(stream['time_base']); fps = Fraction(stream['avg_frame_rate'])
    step = int(1 / fps / tb)
    assert len(pts) == int(stream['nb_frames']) and pts[0] == 0
    assert all(t == i * step for i, t in enumerate(pts)), 'non-CFR/non-contiguous PTS: cannot assign exact indices'
    start, end = 31772, 32072
    assert abs(float(start / fps) - 1059.067) < 0.001
    selected = list(range(start, end + 1, 6))
    frames_dir = out / 'frames'; frames_dir.mkdir()
    model_dir = out / 'model_frames'; model_dir.mkdir()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    filt = f"select='between(pts,{pts[start]},{pts[end]})*not(mod(pts-{pts[start]},{6*step}))',showinfo"
    command = [ffmpeg, '-hide_banner', '-threads', '2', '-ss', '1058', '-copyts', '-i', str(video), '-an', '-vf', filt, '-fps_mode', 'passthrough', '-frames:v', str(len(selected)), str(frames_dir / 'observation_%03d.png')]
    with (out / 'extract.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=log, check=True)
    files = sorted(frames_dir.glob('*.png'))
    assert len(files) == 51, len(files)
    # showinfo is checked against packet PTS, rather than trusting output numbering.
    import re
    shown = [int(x) for x in re.findall(r'\bn:\s*\d+\s+pts:\s*(\d+)', (out / 'extract.log').read_text())]
    assert shown == [pts[n] for n in selected], (shown[:2], shown[-2:])
    rows = []
    for j, (path, index) in enumerate(zip(files, selected)):
        target = path.with_name(f'frame_{index:06d}.png'); path.rename(target)
        model = model_dir / f'frame_{index:06d}.jpg'
        with Image.open(target) as im:
            im.resize((704, 704), Image.Resampling.LANCZOS).save(model, quality=94)
        rows.append({'observation_index': j, 'evidence_id': f'F{index:06d}',
                     'source_frame_index_zero_based': index, 'source_frame_number_one_based': index + 1,
                     'pts': pts[index], 'time_base': str(tb), 'timestamp_seconds': float(pts[index] * tb),
                     'relative_seconds': (index - start) / 30, 'path': str(target.relative_to(out)),
                     'sha256': sha(target), 'model_image_path': str(model.relative_to(out)), 'model_image_sha256': sha(model),
                     'observation_kind': 'decoded_original_frame', 'interpolated': False})
    for page, offset in enumerate(range(0, 51, 9)):
        sheet = Image.new('RGB', (1080, 1152), '#171b22'); draw = ImageDraw.Draw(sheet)
        for k, row in enumerate(rows[offset:offset+9]):
            x, y = (k % 3) * 360, (k // 3) * 384
            with Image.open(out / row['model_image_path']) as im:
                sheet.paste(im.resize((360,360)), (x,y+24))
            draw.text((x+5,y+5), f"#{row['observation_index']:02d} f{row['source_frame_index_zero_based']}  {row['timestamp_seconds']:.3f}s", fill='white')
        sheet.save(out / f'contact_{page:02d}.jpg', quality=94)
    # 301 original frames: inclusive endpoint lasts one display frame, no interpolation.
    preview = [ffmpeg, '-hide_banner', '-threads', '2', '-ss', str(float(start / fps)), '-i', str(video), '-frames:v', '301', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', str(out / 'original_clip.mp4')]
    with (out / 'clip.log').open('w') as log:
        subprocess.run(preview, stdout=log, stderr=log, check=True)
    manifest = {'schema_version':'temporal_observation_manifest_v1', 'dataset':'CoMind', 'sample_id':'032159',
                'source_video':str(video), 'source_video_sha256':sha(video), 'source_video_size':video.stat().st_size,
                'video_stream':stream, 'audio_streams_present':any(s['codec_type']=='audio' for s in meta['streams']),
                'requested_interval_seconds':[1059.067,1069.067], 'actual_interval_seconds':[float(start/fps),float(end/fps)],
                'endpoint_policy':'nearest source frame to rounded millisecond request; inclusive both endpoints',
                'sampling_fps':5,'observation_count':51,'only_six_images':False,
                'frame_index_verification':{'method':'all video packet PTS sorted; unique contiguous CFR, one packet per frame; selected decoded showinfo PTS checked', 'packets':len(pts),'pts_step':step},
                'original_clip':{'path':'original_clip.mp4','sha256':sha(out/'original_clip.mp4'),'frames':301,'fps':30,'audio':'omitted; no speaker or word timing inferred'},
                'frames':rows, 'extraction_command':command,'clip_command':preview}
    (out/'observations_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'frames':len(rows),'source_hash':manifest['source_video_sha256'],'output':str(out)}))


if __name__ == '__main__':
    main()
