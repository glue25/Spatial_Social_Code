#!/usr/bin/env python3
"""Build scorer-only targets from separately obtained CoMind train annotations.
Does not call models, modify predictions, download data, or read hidden test.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code/workdirs/comind_flash_sam_20260920'))
import frozen_v2 as V2

def main():
    p=argparse.ArgumentParser();p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists(),'Refuse overwrite'
    annotations=V2.load_annotations(args.annotations)
    items=[json.loads(line) for line in gzip.open(HERE/'primary/inputs.jsonl.gz','rt')]
    gold={r['id']:{'task':r['task'],'gold':annotations[r['task']][str(r['recording_id'])][str(r['sample_id'])]} for r in items}
    digest=hashlib.sha256(json.dumps(gold,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    info=json.loads((HERE/'audit/label_boundary.json').read_text())
    assert digest==info['gold_canonical_sha256'],'Annotation version/content mismatch'
    args.output.write_text(json.dumps(gold,indent=2,ensure_ascii=False)+'\n')
    print('Verified300 scorer-only targets; never pass this file to inference.')

if __name__=='__main__':main()
