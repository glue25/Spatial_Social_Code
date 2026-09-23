"""Map archived result blocks to frozen configuration/source and run artifacts.

No simulator replay. Does not invent missing per-run source-commit identities.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=ROOT/'reports/minimal_mechanism_package_20260923'
    run=ROOT/'workdirs/qdd_four_env_20260917/habitat108_v2'
    cfg=read(run/'frozen_config.json');cfgsha=sha(run/'frozen_config.json')
    source=Path(cfg['source'])
    checks={rel:sha(source/rel)==value for rel,value in cfg['source_hashes'].items()}
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    results=read(run/'partial_results.json');assert len(results)==108
    cells={(r['task'],r['seed'],r['condition']) for r in results}
    assert cells=={(r['task'],r['seed'],r['condition']) for r in cfg['jobs']}
    updater_scorer={k:v for k,v in cfg['source_hashes'].items() if
        k.startswith('experiments/habitat_closed_loop/') and k.endswith('.py')}
    records=[];final_differences=0;measurements=0
    for i,row in enumerate(results):
        task=read(Path(row['task']));art=Path(row['artifact'])
        manifest=read(art/'manifest.json');summary=read(art/'summary.json')
        assert manifest['config_sha256']==cfgsha
        assert manifest['condition']==row['condition'] and str(manifest['seed'])==str(row['seed'])
        assert manifest['task_id']==task['task_id'] and str(manifest['episode_id'])==str(task['episode_id'])
        strict=summary['historical_benchmark_evaluation']['strict_physical']
        agrees=(row['strict_success']==strict['strict_success'] and
                row['strict_percent_complete']==strict['strict_percent_complete'])
        final_differences+=int(not agrees)
        n=sum(bool(line.strip()) for line in (art/'physical_checks.jsonl').read_text().splitlines());measurements+=n
        records.append({'result_block':'partial_results.json','result_row_index':i,
            'task_id':task['task_id'],'episode_id':task['episode_id'],'scene_id':task['scene_id'],
            'seed':row['seed'],'condition':row['condition'],'config_sha256':cfgsha,
            'updater_scorer_source_hashes_ref':'root.updater_scorer_source_hashes',
            'artifact_path':str(art.relative_to(ROOT)),
            'source_file_hashes':{name:sha(art/name) for name in ('manifest.json','summary.json','physical_checks.jsonl')},
            'task_path':str(Path(row['task']).relative_to(ROOT)),'task_sha256':sha(Path(row['task'])),
            'strict_success':row['strict_success'],'strict_percent_complete':row['strict_percent_complete'],
            'measurement_valid':strict['measurement_valid'],'physical_score_records':n,
            'source_commit_identity':None})
    result={'run_root':str(run.relative_to(ROOT)),'result_block_sha256':sha(run/'partial_results.json'),
        'config_sha256':cfgsha,'source_snapshot':str(source.relative_to(ROOT)),
        'all_archived_source_hashes':cfg['source_hashes'],'updater_scorer_source_hashes':updater_scorer,
        'archived_sources_checked':len(checks),'rollouts':108,'physical_score_records':measurements,
        'invalid_final_measurement_flags':sum(not r['measurement_valid'] for r in records),
        'result_block_vs_archived_summary_differences':final_differences,
        'source_commit_identity_complete':False,
        'scope':'Offline archive mapping and hash verification only, NOT simulator replay or proof of '
            'per-run git revision. Runtime manifests name the frozen config; associated archived source '
            'files match that config now. No current/latest code is substituted for missing commit evidence.',
        'records':records}
    out.mkdir(parents=True,exist_ok=True)
    (out/'habitat108_provenance_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('rollouts','physical_score_records','archived_sources_checked',
        'invalid_final_measurement_flags','result_block_vs_archived_summary_differences','source_commit_identity_complete')},indent=2))


if __name__=='__main__':main()
