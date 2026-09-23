"""Offline full300 pairing/source/geometry audit; no SAM or model calls."""
from pathlib import Path
import json
import numpy as np

from experiments.mechanism_followups_v1 import comind_sam as c


def main():
    out=c.ROOT/'reports/minimal_mechanism_package_20260923'
    out.mkdir(parents=True,exist_ok=True)
    cfg=c.verify(c.OUT);c.audit()
    for rel,h in cfg['source_hashes'].items():
        assert c.sha(c.OUT/'source'/rel)==h,rel
    goldpath=c.ROOT/cfg['scorer_only_gold_path'];assert c.sha(goldpath)==cfg['scorer_only_gold_sha256']
    gold=c.read(goldpath);items=c.read(c.OUT/'inputs.json')
    paths=sorted((c.OUT/'results').glob('*.json'));rows=[c.read(p) for p in paths]
    assert len(items)==len(rows)==300
    by={r['index']:r for r in rows};assert set(by)=={x['index'] for x in items}
    arms=('direct','direct_sam','ssc','ssc_sam')
    fields=(('joint_attention','left_iou_at_0_5'),('joint_attention','right_iou_at_0_5'),
            ('scoia','iou_at_0_5'),('handover','iou_at_0_5'))
    aggregate={};manifest=[];values=[]
    for item in items:
        row=by[item['index']];assert all(row[k]==item[k] for k in ('id','task','recording_id','sample_id'))
        source=Path(item['upstream_snapshot']);assert c.sha(source)==item['upstream_snapshot_hash']
        upstream=c.read(source);assert upstream['index']==item['index']
        for name in ('direct','ssc'):
            before=upstream['conditions'].get(name,{})
            if not c.sam.valid_ssc(before):
                assert row['conditions'][name]['prediction'] is None
                assert row['conditions'][name+'_sam']['prediction'] is None
                continue
            base=c.coordinate_correct_original(item['task'],before['prediction'])
            assert row['conditions'][name]['prediction']==base
            enhanced=row['conditions'][name+'_sam']
            assert enhanced['identity']==c.identity(before['prediction'],item['task'])
            for proposals in enhanced['sam_proposals'].values():
                assert all(proposals[i]['score']>=proposals[i+1]['score'] for i in range(len(proposals)-1))
            pred,view=c.sam.augment(base,item['task'],enhanced['sam_proposals'])
            assert enhanced['prediction']==pred and enhanced['selected_view']==view
        scores={}
        for arm in arms:
            pred=row['conditions'][arm]['prediction']
            scores[arm]=(c.sam.V2.score_prediction(item['task'],pred,gold[item['id']]['gold'])
                         if pred is not None else dict.fromkeys(c.sam.V2.FIELD_TO_PREDICTION[item['task']],False))
            for field,value in scores[arm].items():
                cell=aggregate.setdefault(item['task']+'/'+field,{}).setdefault(arm,{'n':0,'correct':0})
                cell['n']+=1;cell['correct']+=int(value)
        for j,(task,field) in enumerate(fields):
            if task==item['task']:values.append((str(item['recording_id']),j,[int(scores[a][field]) for a in arms]))
        path=c.OUT/'results'/f"{item['index']:04d}.json"
        manifest.append({k:item[k] for k in ('index','id','task','recording_id','sample_id')})
        manifest[-1].update(result_path=str(path.relative_to(c.ROOT)),result_sha256=c.sha(path),
            upstream_path=str(source.relative_to(c.ROOT)),upstream_sha256=item['upstream_snapshot_hash'],
            frame_hashes=item['frame_hashes'])
    ids=sorted({r for r,j,v in values});pos={r:i for i,r in enumerate(ids)}
    counts=np.zeros((len(ids),4));success=np.zeros((len(ids),4,4))
    for r,j,v in values:counts[pos[r],j]+=1;success[pos[r],j]+=v
    assert list(counts.sum(axis=0))==[100]*4
    point=(success.sum(axis=0)/counts.sum(axis=0)[:,None]).mean(axis=0)
    rng=np.random.default_rng(20260923);boot=[]
    for _ in range(10000):
        sel=rng.integers(0,len(ids),len(ids));den=counts[sel].sum(axis=0)
        if np.any(den==0):continue
        boot.append((success[sel].sum(axis=0)/den[:,None]).mean(axis=0))
    boot=np.array(boot);diff=boot[:,3]-boot[:,1]
    result={'utc':c.utc(),'examples':300,'localization_fields':400,'recording_clusters':len(ids),
        'four_arm_pairing_verified':True,'sam_selection_and_coordinate_replay':'300/300 passed',
        'all_official_fields':aggregate,'localization_macro':{a:float(point[i]) for i,a in enumerate(arms)},
        'ssc_sam_minus_direct_sam':{'difference':float(point[3]-point[1]),
            'recording_cluster_bootstrap_ci95':np.quantile(diff,[.025,.975]).tolist(),
            'replicates':len(boot),'seed':20260923},
        'threshold':'Frozen scorer uses IoU > 0.5',
        'scope':cfg['scope'],'new_model_calls':0,'new_sam_calls':0,
        'interpretation':'Exploratory development300 macro localization difference, not overall social '
            'understanding or a confirmatory mechanism result. No cross-batch arm stitching.'}
    c.write(out/'comind_four_arm_audit.json',result)
    c.write(out/'comind_pairing_manifest.json',{'protocol_sha256':c.sha(c.OUT/'PROTOCOL.json'),
        'input_sha256':c.sha(c.OUT/'inputs.json'),'gold_sha256':cfg['scorer_only_gold_sha256'],
        'scorer_source_sha256':cfg['source_hashes']['workdirs/comind_flash_sam_20260920/frozen_v2.py'],
        'source_hashes':cfg['source_hashes'],'rows':manifest,
        'source_commit_identity':None,'note':'Exact archived source hashes verified; original git commit not inferred.'})
    print(json.dumps({k:result[k] for k in ('examples','recording_clusters','localization_macro','ssc_sam_minus_direct_sam')},indent=2))


if __name__=='__main__':main()
