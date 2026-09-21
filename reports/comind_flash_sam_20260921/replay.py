#!/usr/bin/env python3
"""Replay original prompt/schema/compiler/answer path using archived replies only.
No model API, labels, GPU, or media. Frame byte reads become archived descriptors;
therefore this validates request order/text/schema, not original image bytes.
"""
import base64
import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
BASE=HERE/'code/workdirs/comind_native_schema_20260920'
sys.path.insert(0,str(BASE));import comind_adapter as adapter

class OutputTruncated(RuntimeError):pass
class NoNetwork(RuntimeError):pass

def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(l) for l in f if l.strip()]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=['all','primary','recovery'],default='all');args=parser.parse_args()
    for cohort in ['primary','recovery']:
        if args.cohort!='all' and cohort!=args.cohort:continue
        results={r['index']:r for r in rows(HERE/cohort/'results.jsonl.gz')}
        calls={(c['index'],c['call']):c['files'] for c in rows(HERE/cohort/'calls.jsonl.gz')}
        count=0
        for item in rows(HERE/cohort/'inputs.jsonl.gz'):
            idx=item['index'];first=calls[(idx,1)]['request.json'];descriptors=[p for p in first['parts'] if 'sha256' in p]
            assert len(descriptors)==len(item['images'])
            placeholders={str(Path(im['path'])):json.dumps(d,sort_keys=True).encode() for im,d in zip(item['images'],descriptors)}
            real_read=Path.read_bytes
            def fake_read(path):return placeholders[str(path)] if str(path) in placeholders else real_read(path)
            class Client:
                config=SimpleNamespace(model='gemini-3-flash-preview')
                out=Path('/no-real-output')
                counter=0
                def generate_parts(self,parts,max_output_tokens=16384,response_schema=None):
                    self.counter+=1;entry=calls[(idx,self.counter)];request=entry['request.json']
                    actual=[]
                    for p in parts:
                        if 'inline_data' in p:
                            desc=json.loads(base64.b64decode(p['inline_data']['data']));assert desc['mimeType']==p['inline_data']['mimeType'];actual.append(desc)
                        else:actual.append(p)
                    assert actual==request['parts'],(cohort,idx,self.counter,'prompt/media order')
                    assert response_schema==request['generationConfig']['responseJsonSchema']
                    assert max_output_tokens==request['legacy_requested_cap']
                    if 'incomplete.json' in entry:raise OutputTruncated('Incomplete output; no partial-code execution or cap escalation')
                    raw=entry['response.json']
                    return '\n'.join(str(p['text']) for c in raw.get('candidates',[]) for p in c.get('content',{}).get('parts',[]) if p.get('text')).strip()
            client=Client()
            with patch.object(Path,'read_bytes',fake_read):
                replay=adapter.infer(item,client,lambda *a:None,(NoNetwork,))
            assert replay['conditions']==results[idx]['base_conditions'],(cohort,idx,'predictions/code/traces mismatch')
            assert client.counter==len([k for k in calls if k[0]==idx]);count+=1
        print(f'PASS {cohort}: {count} exact prompt/schema/code/trace/prediction replays; no labels or model calls')

if __name__=='__main__':main()
