"""Verify exported MA886 archival artifacts, not speaker or semantic truth.

Standard library only. No inference, network, generated-code execution or writes.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/maego_pro_case886_candidate_20260924'


def read(name):
    return json.loads((OUT / name).read_text())


def main():
    manifest = read('SHA256SUMS.json')
    assert set(manifest) == {p.name for p in OUT.iterdir() if p.is_file() and p.name != 'SHA256SUMS.json'}
    for name, digest in manifest.items():
        assert Path(name).name == name
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest, name
    q, direct, ssc = read('question.json'), read('historical_direct.json'), read('historical_ssc.json')
    assert direct['index'] == ssc['index'] == 886
    assert direct['baseline_prediction_raw'] == 'C'
    assert direct['baseline_prediction_index'] == 2
    assert ssc['structured_data']['answer'] == 'B'
    raw = ssc['structured_raw'].strip()
    if raw.startswith('```json') and raw.endswith('```'):
        raw = raw[len('```json'):-3].strip()
    assert json.loads(raw) == ssc['structured_data']
    assert ssc['prediction_index'] == ssc['answer_index'] == 1
    assert q['answer'] == q['options'][1]
    assert q['question'] == "What was Lucia's misunderstanding about the sesame sauce?"
    clips = [r['id'] for r in direct['video_clips']]
    assert clips == [r['id'] for r in ssc['video_clips']] and len(clips) == 10
    assert (OUT / 'spatial_code.py').read_text() == ssc['spatial_code'] + '\n'
    assert (OUT / 'social_code.py').read_text() == ssc['social_code'] + '\n'
    assert ssc['structured_data']['evidence_ids'] == ['S1']
    assert 'S1' in ssc['valid_evidence_ids']
    for media in read('media_manifest.json'):
        assert media['original_model_input'] == (media['clip_id'] in clips)
        assert len(media['frames']) == 4
        for frame in media['frames']:
            assert frame['frame_index'] == round(frame['nominal_seconds'] * media['fps'])
            assert manifest[frame['file']] == frame['sha256']
    status = read('candidate_status.json')
    assert status['speaker_ground_truth_verified'] is False
    assert status['utterance_timing_verified'] is False
    assert status['direct'] == 'C' and status['ssc'] == status['gold'] == 'B'
    print(f'PASS: {len(manifest)} artifact hashes, original C/B/B answers, matching clip lists, code and frame provenance. Speaker/timing remain unverified.')


if __name__ == '__main__':
    main()
