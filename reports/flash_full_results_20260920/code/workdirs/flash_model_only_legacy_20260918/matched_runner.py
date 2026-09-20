"""Historical-code model substitution: fresh Direct + original SSC, not typed SSC.

Restored Git source is never edited. Historical retrieval outputs are frozen as
inputs; model outputs, code, and model-dependent audit/refinement are regenerated.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import fcntl
import importlib.util

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[1]
sys.path.insert(0, str(HOME))
from legacy_loader import load
load()
from native_transport import Client, Paused, ModelIdentityError, write
from egomas.src import run_video_provenance_v3_jdcloud as ma
from egomas.src import audit_video_evidence_sparse_jdcloud as ma_audit
from egomas.src import run_video_spatial_social_jdcloud as video
from egomas.utils.parsing import get_prediction_index
from egomas.utils.prompt_helpers import build_question_prompt
import evaluate_mmtom_jdcloud as mm
import refine_mmtom_uncertain_jdcloud as dense
from mmtom_visual import load_questions, sample_rgb_frames

SPEC = importlib.util.spec_from_file_location('model_only_service_gate', HOME / 'service_gate.py')
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)

REFERENCES = {
    'mmtom': ROOT / 'spatial_social_code/mmtom_dense_authoritative_gemini3pro.jsonl',
    'maegoqa': ROOT / 'MA-EgoQA/data/baselines/video_evidence_audit_full.jsonl',
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    result = {str(p.relative_to(HOME)): sha(p) for p in (HOME / 'historical_source').rglob('*.py')}
    for name in ('matched_runner.py', 'native_transport.py', 'legacy_loader.py', 'service_gate.py'):
        result[name] = sha(HOME / name)
    return result


def prepare(dataset):
    out = HOME / 'runs' / dataset
    assert not out.exists(), 'never overwrite a model-substitution run'
    replay = json.loads((HOME / ('MM_REPLAY_AUDIT.json' if dataset == 'mmtom' else 'MA_REPLAY_AUDIT.json')).read_text())
    assert not any(n for k, n in replay['counts'].items() if 'mismatch' in k)
    references = [json.loads(line) for line in REFERENCES[dataset].read_text().splitlines() if line.strip()]
    historical = {r['index']: r for r in references}
    source_rows = load_questions(ROOT / 'MMToM-QA/Benchmark/questions.json') if dataset == 'mmtom' else json.loads((ROOT / 'MA-EgoQA/data/MA-EgoQA.json').read_text())
    rows = []
    hashes = {}
    for i in sorted(historical):
        row = dict(index=i, row=source_rows[i])
        media = []
        if dataset == 'mmtom':
            for count in (8, 16):
                paths, details = sample_rgb_frames(ROOT / 'MMToM-QA-data/required_frames', source_rows[i], num_frames=count)
                if count == 8:
                    assert details == historical[i]['image_details']
                elif historical[i].get('uncertainty_triggered'):
                    assert details == historical[i]['second_image_details']
                row[f'frames_{count}'] = details
                media.extend(paths)
        else:
            # Whitelist only retrieved clip descriptors, not old code/answers.
            row['video_clips'] = historical[i]['video_clips']
            for clip in row['video_clips']:
                video.source_video_path(ROOT / 'MA-EgoQA/data/egolife_raw', clip['id'])
                proxy = video.proxy_video_path(ROOT / 'MA-EgoQA/data/egolife_proxy_384p', clip['id'])
                assert proxy.is_file() and proxy.stat().st_size > 0, f'missing historical proxy: {proxy}'
                media.append(proxy)
        row['media_files'] = sorted({str(p.resolve()) for p in media})
        for p in row['media_files']:
            if p not in hashes:
                hashes[p] = sha(p)
        rows.append(row)
    out.mkdir(parents=True)
    write(out / 'inputs.json', rows)
    write(out / 'media_hashes.json', hashes)
    config = dict(dataset=dataset, created_utc=utc(), model='gemini-3-flash-preview',
        endpoint='https://api2.aigcbest.top/v1beta/models/gemini-3-flash-preview:generateContent',
        key_file='/tmp/physical_code_qdd_credentials/habitat_key',
        shared_gate=str(ROOT / 'workdirs/qdd_typed_ssc_v3_fair_20260917/shared_habitat_key.lock'),
        input_hash=sha(out / 'inputs.json'), media_hash=sha(out / 'media_hashes.json'),
        source_hashes=source_hashes(), reference=str(REFERENCES[dataset]), reference_hash=sha(REFERENCES[dataset]),
        conditions=['direct', 'historical_ssc'], samples=len(rows),
        historical_source_commit='b4039ee', source_status='restored historical implementation with offline replay checks; historical request-body archive unavailable',
        provider_difference='Historical Pro used JDCloud; provisioned Flash uses QDD native envelope. Not an isolated provider-controlled weights comparison.',
        media_limitation='Historical clip/frame identities matched. Current bytes frozen; historical Pro did not save media hashes, so historical byte identity is not independently certifiable.',
        scientific_changes=['model'], necessary_transport_changes=['provider endpoint', 'model moved to URL', 'inline_data -> inlineData', 'identity guard and credential-safe logs'],
        no_prompt_or_schema_changes=True, model_responses_reused=False,
        fresh_baseline=True, fixed_retrieval_inputs=True, output_caps='legacy per-stage caps unchanged',
        thinking='unspecified, as in archived client; no injected low setting',
        retry_policy='archived 5-attempt transport/empty-text policy; archived method-level parse retries unchanged',
        refinement='fresh model confidence threshold 0.70, 16 frames, dense-authoritative' if dataset == 'mmtom' else 'fresh citation-based sparse-evidence audit with original acceptance rule',
        labels='original functions compute scoring fields; answer labels never enter request construction',
        scope='Historical-protocol development replication, including historical answer-conditioned components. Not a clean option-blind SSC claim.')
    write(out / 'frozen_config.json', config)
    print(json.dumps(dict(dataset=dataset, prepared=len(rows), unique_media=len(hashes), config_sha256=sha(out / 'frozen_config.json'))), flush=True)


def verify(out, cfg):
    assert json.loads((out / 'frozen_config.json').read_text()) == cfg
    assert sha(out / 'inputs.json') == cfg['input_hash']
    assert sha(out / 'media_hashes.json') == cfg['media_hash']
    for rel, value in cfg['source_hashes'].items():
        assert sha(HOME / rel) == value, rel


def process_mm(item, client):
    index, row = item['index'], item['row']
    primary = mm.process_one(index, row, ROOT / 'MMToM-QA-data/required_frames', 8, client)
    if dense.confidence(primary) < dense.CONFIDENCE_THRESHOLD:
        result = dense.refine_one(primary, row, ROOT / 'MMToM-QA-data/required_frames', 16, client)
        a, b = result['second_prob_a'], result['second_prob_b']
        prediction = 'a' if a >= b else 'b'
        result.update(prediction=prediction, correct=prediction == str(row['answer']).lower(),
                      prob_a=a, prob_b=b, fusion_policy='dense_authoritative')
    else:
        result = dense.unchanged(primary)
    result['method'] = 'Dense-authoritative uncertainty-refined Spatial-Social Code'
    return result


def process_ma(item, client):
    index, row, clips = item['index'], item['row'], item['video_clips']
    video_root = ROOT / 'MA-EgoQA/data/egolife_raw'
    proxy_root = ROOT / 'MA-EgoQA/data/egolife_proxy_384p'
    ffmpeg = Path('/unused-proxies-must-already-exist')
    media = video.video_parts(clips, video_root, proxy_root, ffmpeg)
    prompt = video.BASELINE_PROMPT.format(question=build_question_prompt(row['question'], row['options']))
    baseline_raw = video.ask(client, media, prompt, max_tokens=256)
    baseline = get_prediction_index(baseline_raw)
    write(client.out / 'baseline' / f'{index:04d}.json', dict(index=index, baseline_prediction_raw=baseline_raw,
                                                           baseline_prediction_index=baseline))
    paired = dict(video_clips=clips, baseline_prediction_index=baseline)
    primary = ma.process_one(index, row, paired, client, video_root, proxy_root, ffmpeg)
    result = ma_audit.audit_one(primary, row, client, video_root, proxy_root, ffmpeg) if ma_audit.needs_audit(primary) else ma_audit.unchanged(primary)
    result['baseline_prediction_raw'] = baseline_raw
    return result


def run(dataset):
    out = HOME / 'runs' / dataset
    cfg = json.loads((out / 'frozen_config.json').read_text())
    with (out / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((out / n).exists() for n in ('OPERATOR_PAUSED.json', 'COMPLETE', 'GATE_FAILED.json', 'FAILED.json')):
            return
        verify(out, cfg)
        media_hashes = json.loads((out / 'media_hashes.json').read_text())
        rows = json.loads((out / 'inputs.json').read_text())
        client = Client(out, cfg, GATE.service_gate)
        for number, item in enumerate(rows, 1):
            if (out / 'OPERATOR_PAUSED.json').exists():
                raise Paused('operator pause')
            verify(out, cfg)
            for path in item['media_files']:
                assert sha(path) == media_hashes[path], 'frozen input media changed'
            target = out / 'results' / f'{item["index"]:04d}.json'
            write(out / 'status.json', dict(utc=utc(), state='sample_running', completed=number-1, total=len(rows), index=item['index'], pid=os.getpid()))
            if target.exists():
                result = json.loads(target.read_text())
            else:
                client.begin(item['index'])
                try:
                    result = process_mm(item, client) if dataset == 'mmtom' else process_ma(item, client)
                except (Paused, ModelIdentityError):
                    raise
                except Exception as exc:
                    # Never select a retry by correctness; preserve and count errors.
                    result = dict(index=item['index'], error_type=type(exc).__name__, prediction=None)
                write(target, result)
            write(out / 'progress.json', dict(utc=utc(), completed=number, total=len(rows)))
            if number == 1 and ('error_type' in result or (dataset == 'maegoqa' and (result.get('prediction_index', -1) < 0 or result.get('baseline_prediction_index', -1) < 0))):
                write(out / 'GATE_FAILED.json', dict(index=item['index'], reason='first-pair execution/parse gate; correctness not consulted'))
                write(out / 'status.json', dict(utc=utc(), state='first_pair_gate_failed', completed=number, total=len(rows)))
                return
        (out / 'COMPLETE').write_text('All historical selected samples processed; errors retained.\n')
        write(out / 'status.json', dict(utc=utc(), state='complete', completed=len(rows), total=len(rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['mmtom', 'maegoqa'])
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.dataset)
    else:
        try:
            run(args.dataset)
        except Paused:
            write(HOME / 'runs' / args.dataset / 'status.json', dict(utc=utc(), state='operator_paused'))
        except Exception as exc:
            write(HOME / 'runs' / args.dataset / 'FAILED.json', dict(utc=utc(), error_type=type(exc).__name__))
            raise
