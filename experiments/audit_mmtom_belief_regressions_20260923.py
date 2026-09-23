"""Conservative offline triage. No causal labels inferred from final errors."""
import gzip
import hashlib
import json
from pathlib import Path
import re
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    archive=ROOT/'reports/flash_full_results_20260920/mmtom/flash_results.jsonl.gz'
    inputs=ROOT/'reports/flash_full_results_20260920/mmtom/original_frozen_inputs/inputs.json'
    rows=[json.loads(s) for s in gzip.open(archive,'rt') if s.strip()]
    items={r['index']:r for r in json.loads(inputs.read_text())}
    assert len(rows)==len(items)==600
    packet=[];parse_mismatches=[];groups=Counter()
    for row in rows:
        tail=re.findall(r'(?i)(?:^|\n)\s*([AB])[.\s]*\Z',row.get('direct_raw',''))
        mismatch=bool(tail and tail[-1].lower()!=row.get('direct_prediction'))
        if mismatch:parse_mismatches.append({'index':row['index'],'terminal_standalone_answer':tail[-1].lower(),
            'stored_prediction':row['direct_prediction'],'raw_suffix':row['direct_raw'][-350:]})
        question_type=row.get('question_type',items[row['index']]['row']['question_type'])
        if not (str(question_type).startswith('1.') and row.get('direct_correct') and not row.get('correct')):continue
        checks={}
        for layer in ('spatial','social'):
            trace=row.get(layer+'_trace','')
            try:
                records=[json.loads(s) for s in trace.splitlines() if s.strip()]
                assert all(isinstance(r,dict) and 'kind' in r for r in records)
                checks[layer]={'parseable':True,'records':len(records),
                    'kinds':dict(Counter(r['kind'] for r in records))}
            except Exception as exc:
                checks[layer]={'parseable':False,'exception':type(exc).__name__}
        if mismatch:group='format_direct_terminal_parser_inconsistency'
        elif not all(c['parseable'] for c in checks.values()):group='format_trace_not_parseable'
        else:group='undetermined_requires_semantic_and_rule_audit'
        groups[group]+=1
        packet.append({'index':row['index'],'episode':row['episode'],'question_type':row['question_type'],
            'group':group,'question_and_options':items[row['index']]['row'],
            'direct_prediction':row['direct_prediction'],'ssc_prediction':row['prediction'],
            'gold':row['answer'],'direct_raw':row['direct_raw'],'checks':checks,
            'component_presence':row.get('social_components'),
            'component_diagnostics':row.get('social_reasoning'),
            'spatial_code':row.get('spatial_code'),'social_code':row.get('social_code'),
            'spatial_trace':row.get('spatial_trace'),'social_trace':row.get('social_trace'),
            'classification_note':'Component presence/zero bridge count is NOT a proven cause. '
                'No compilation-semantic-error or rule-coverage label without manual trace evidence.',
            'manual_review':{'compilation_error':None,'candidate_or_rule_coverage':None,'supporting_trace_lines':[]}})
    out=ROOT/'reports/minimal_mechanism_package_20260923';out.mkdir(parents=True,exist_ok=True)
    summary={'protocol':'Historical candidate-conditioned Flash600; NOT option-blind',
        'archive_sha256':sha(archive),'inputs_sha256':sha(inputs),'rows':600,
        'historical_belief_regressions':len(packet),'groups':dict(groups),
        'terminal_failure_rows':sum('error_type' in r for r in rows),
        'terminal_parser_inconsistencies_all600':len(parse_mismatches),
        'old_scores_rewritten':False,'new_model_calls':0,
        'limitations':'Grouping is conservative mechanical triage, not completed human semantic error attribution. '
            'Review packet includes every historical Direct-correct/SSC-wrong belief row, with failures retained.'}
    (out/'mmtom_belief_regression_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'mmtom_belief_review_packet.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
    (out/'mmtom_parser_inconsistencies.json').write_text(json.dumps(parse_mismatches,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
