"""Replay original300 saved response tuples, without network or image reads."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from run import HERE,ROOT,read,write
from comind_adapter import infer

def main():
    w=ROOT/'pipelines/spatial_social_optimization_v1'
    rows=[json.loads(x) for x in (w/'comind_codex_300_manifest.jsonl').read_text().splitlines()]
    key=lambda x:(x['task'],str(x['recording_id']),str(x['sample_id']))
    old={(r['condition'],key(r)):r for line in (w/'comind_codex_300_predictions.jsonl').read_text().splitlines() for r in [json.loads(line)]}
    mismatch=[]
    for index,row in enumerate(rows):
        stem=f"{row['task']}_{row['recording_id']}_{row['sample_id']}"
        replies=[read(w/'codex_work/raw'/f'{stem}_{stage}.json') for stage in ['direct','spatial','social','answer']]
        class Replay:
            out=Path('/not-written');config=SimpleNamespace(model='historical-replay')
            def generate_parts(self,parts,max_output_tokens=16384,response_schema=None):
                assert response_schema is not None and 'responseJsonSchema' not in parts[-1]['text']
                return json.dumps(replies.pop(0))
        with patch.object(Path,'read_bytes',return_value=b'fixture'),patch('requests.sessions.Session.request',side_effect=AssertionError('No network')):
            fresh=infer(dict(row,index=index,id=str(index)),Replay(),lambda *a:None,())
        for c,oc in [('direct','direct'),('ssc','spatial_social')]:
            fields=['prediction'] if c=='direct' else ['prediction','spatial_code','spatial_trace','social_code','social_trace']
            for field in fields:
                if fresh['conditions'][c].get(field)!=old[(oc,key(row))].get(field):mismatch.append({'index':index,'condition':c,'field':field,'error_type':fresh['conditions'][c].get('error_type')})
    report={'rows':len(rows),'mismatches':mismatch,'passed':not mismatch,'no_network':True,'no_image_reads':True}
    write(HERE/'REPLAY.json',report)
    print(json.dumps({'rows':len(rows),'mismatches':len(mismatch),'passed':not mismatch}),flush=True)
    assert not mismatch

if __name__=='__main__':main()
