"""Portable offline replay of archived inference (no keys, API, images, or GPU).

Uses frozen frame selections and hashes, NOT a fresh image-byte/sampling audit.
Dependencies: requests pillow numpy pandas rank-bm25 tqdm (legacy import graph).
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'code/workdirs/flash_model_only_legacy_20260918'))
import matched_runner as legacy
from egomas.src.jdcloud_gemini import _extract_text, GeminiServiceError

def read(name):
    return json.loads((ROOT/name).read_text())
def rows(name):
    with gzip.open(ROOT/name, 'rt') as f:
        return [json.loads(x) for x in f]
inputs = {r['index']:r for r in read('mmtom/original_frozen_inputs/inputs.json')}
hashes = read('mmtom/original_frozen_inputs/media_hashes.json')
calls = defaultdict(dict)
for r in rows('mmtom/calls.jsonl.gz'):
    calls[r['index']][r['call']] = r['files']

class Replay:
    def __init__(self, index):
        self.index = index
        self.counter = 0
        self.config = SimpleNamespace(model='gemini-3-flash-preview')
    def generate_parts(self, parts, max_output_tokens=4096):
        self.counter += 1
        files = calls[self.index][self.counter]
        q = files['request.json']
        assert q['generationConfig'] == {'temperature':0.0,'maxOutputTokens':16384}
        assert q['legacy_requested_cap'] == max_output_tokens
        assert len(parts) == len(q['parts'])
        for fresh,saved in zip(parts,q['parts']):
            if '_path' in fresh:
                assert hashes[fresh['_path']] == saved['sha256']
            else:
                assert fresh == saved
        if 'response.json' not in files:
            raise GeminiServiceError('Preserved missing transport response')
        raw = files['response.json']
        assert raw['modelVersion'] == self.config.model
        assert all(c['finishReason']=='STOP' for c in raw['candidates'])
        return _extract_text(raw)
    def generate(self,prompt,max_output_tokens=4096):
        return self.generate_parts([{'text':prompt}],max_output_tokens)

def main():
    results = []
    for old in rows('mmtom/flash_results.jsonl.gz'):
        item = inputs[old['index']]
        def sample(_root,_row,num_frames=8):
            details = item[f'frames_{num_frames}']
            paths = [next(p for p in item['media_files'] if p.endswith(f'/Action_{n:04d}_0_normal.png')) for n in details['frame_numbers']]
            return paths,details
        media = lambda paths: [{'_path':str(p)} for p in paths]
        client = Replay(old['index'])
        with patch.object(legacy.mm,'sample_rgb_frames',sample), patch.object(legacy.dense,'sample_rgb_frames',sample), patch.object(legacy.mm,'image_parts',media), patch.object(legacy.dense,'image_parts',media), patch('requests.sessions.Session.request',side_effect=AssertionError('Network forbidden')):
            try:
                fresh = legacy.process_mm(item,client)
                diff = [k for k,v in old.items() if k not in ['elapsed_seconds','refinement_seconds'] and fresh.get(k)!=v]
                assert not diff, (old['index'],diff)
                assert old['index'] != 44
            except GeminiServiceError:
                assert old['index']==44 and old.get('error_type')=='GeminiServiceError'
        assert client.counter == len(calls[old['index']])
        results.append(old['index'])
        if len(results)%100==0:
            print(f'Replayed {len(results)}/600, no differences',flush=True)
    assert results == list(range(600))
    print('PASS: all600 archived outcomes reproduced, including preserved SSC transport failure44. No network/media reads.')

if __name__ == '__main__':
    main()
