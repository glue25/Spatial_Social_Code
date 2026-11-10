"""No API calls: archived MA pipeline replay and native-envelope equivalence."""
import copy
import json
from pathlib import Path
import sys

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[1]
sys.path.insert(0, str(HOME))
from legacy_loader import load
load()
from native_transport import native_body
from egomas.src import run_video_provenance_v3_jdcloud as ma
from egomas.src import audit_video_evidence_sparse_jdcloud as audit


def main():
    request_checks = 0
    for cap in (256, 1024, 2048, 4096):
        parts = [{'text': 'Clip 1: fixture.mp4 (includes audio)'},
                 {'inline_data': {'mimeType': 'video/mp4', 'data': 'Zml4dHVyZQ=='}},
                 {'text': 'original prompt, no suffix'}]
        before = copy.deepcopy(parts)
        legacy, native = native_body(parts, cap, 'gemini-3-flash-preview')
        restored = copy.deepcopy(native)
        restored['model'] = 'gemini-3-flash-preview'
        for part in restored['contents'][0]['parts']:
            if 'inlineData' in part:
                part['inline_data'] = part.pop('inlineData')
        assert restored == legacy and parts == before
        assert native['generationConfig'] == {'temperature': 0.0, 'maxOutputTokens': cap}
        request_checks += 1
    benchmark = json.loads((ROOT / 'MA-EgoQA/data/MA-EgoQA.json').read_text())
    rows = [json.loads(line) for line in (ROOT / 'MA-EgoQA/data/baselines/video_evidence_audit_full.jsonl').read_text().splitlines() if line.strip()]
    # Only the test harness replaces video loading; production uses the archived
    # function and frozen historical clip order. No fake response enters runs/.
    ma.video_parts = audit.video_parts = lambda *args: []
    mismatch = []
    for saved in rows:
        replies = [saved['spatial_raw'], saved['social_raw']]
        if saved.get('answer_retry_used'):
            replies.append(saved['structured_initial_raw'])
        replies.append(saved['structured_raw'])
        if saved['audit_triggered']:
            replies.append(saved['audit_raw'])
        class Replay:
            config = type('Config', (), {'model': saved['model']})()
            def generate_parts(self, parts, max_output_tokens=4096):
                return replies.pop(0)
            def generate(self, prompt, max_output_tokens=4096):
                return self.generate_parts([{'text': prompt}], max_output_tokens)
        client = Replay()
        row = benchmark[saved['index']]
        paired = {'video_clips': saved['video_clips'], 'baseline_prediction_index': saved['baseline_prediction_index']}
        primary = ma.process_one(saved['index'], row, paired, client, Path('.'), Path('.'), Path('.'))
        result = audit.audit_one(primary, row, client, Path('.'), Path('.'), Path('.')) if audit.needs_audit(primary) else audit.unchanged(primary)
        assert not replies
        for field in ('spatial_code', 'spatial_trace', 'social_code', 'social_trace',
                      'structured_prediction_index', 'prediction_index', 'audit_triggered', 'audit_flip_accepted'):
            if result.get(field) != saved.get(field):
                mismatch.append({'index': saved['index'], 'field': field})
    report = dict(native_envelope_roundtrips=request_checks, ma_rows=len(rows),
        ma_pipeline_mismatches=mismatch, no_api_calls=True,
        note='Replays historical responses, including parse retries and audit. Does not independently prove original prompt/media byte identity.')
    (HOME / 'OFFLINE_CHECKS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    assert not mismatch


if __name__ == '__main__':
    main()
