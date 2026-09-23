"""Export complete saved diagnostic evidence and draw submission assets offline."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from decimal import Decimal, ROUND_HALF_UP

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'workdirs/information_history_v1_20260923_scoring2'
OUT=ROOT/'reports/information_history_v1_20260923'
CONDITIONS=('ssc','prose','no_links','no_attribution')
LABELS=('Linked SSC','Lossless prose','No links','No attribution')
HISTORIES=('witness','unseen','delivered','undelivered')
SELECTED={'formal_000':'Main: both SSC and prose track all four histories correctly.',
          'formal_003':'Alternative: unobserved intermediate move; both complete representations succeed.',
          'formal_007':'Post-hoc favorable contrast: prose selects an unobserved intermediate location after nondelivery.',
          'formal_019':'Counterexample: SSC makes state errors while prose answers all four histories correctly.'}


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def gzwrite(p,data):p.write_bytes(gzip.compress(data,mtime=0))


def export():
    cfg=read(RUN/'frozen_config.json'); assert read(RUN/'status.json')['state']=='complete'
    evidence=OUT/'evidence';evidence.mkdir(exist_ok=True)
    for name in ('dataset.json','selection_audit.json'):
        gzwrite(evidence/(name+'.gz'),(RUN/name).read_bytes())
    public={k:v for k,v in cfg.items() if k not in ('credentials','source_root','publication_dir')}
    public['credential_aliases']=[c['alias'] for c in cfg['credentials']]
    public['public_redaction']='Credential paths/locks, runtime source/output paths omitted; no key values were stored.'
    write(evidence/'frozen_config.public.json',public)
    for rel,h in cfg['source_hashes'].items():
        src=RUN/'source'/rel;assert sha(src)==h
        dest=OUT/'frozen_source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    # Exact UTF-8 bytes, including original whitespace, remain hash-verifiable.
    archive=[]
    for p in sorted((RUN/'calls').rglob('*.json')):
        if p.name=='retry_status.json':continue
        archive.append({'path':str(p.relative_to(RUN)), 'sha256':sha(p),'text':p.read_text()})
    for p in sorted((RUN/'probes').rglob('*.json')):
        archive.append({'path':str(p.relative_to(RUN)), 'sha256':sha(p),'text':p.read_text()})
    gzwrite(evidence/'all_calls.jsonl.gz',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in archive).encode())
    for name in ('COMPLETE.json','engineering_gate.json','preflight.json'):
        shutil.copy2(RUN/name,evidence/name)
    write(evidence/'provenance.json',{'original_config_sha256':sha(RUN/'frozen_config.json'),
        'public_config_is_redacted':True,'data_sha256':cfg['data_sha256'],
        'selection_sha256':cfg['selection_sha256'],'source_hashes':cfg['source_hashes'],
        'source_commit_identity':cfg['source_commit_identity'],
        'archive_files':len(archive),'formal_cells':640,'engineering_cells':160,
        'notes':'All responses, including invalid outputs, retained. No model/SAM/simulator calls made for export.'})
    data=read(RUN/'dataset.json');results={}
    for p in (RUN/'calls/formal').glob('*/*/*/result.json'):
        r=read(p);results[r['base_id'],r['history'],r['condition']]=r
    ledger=[]
    for b in data['bases']:
        if b['split']!='formal':continue
        ledger.append({'base_id':b['base_id'],'selected':b['base_id'] in SELECTED,
            'per_condition':{c:{'joint_correct':sum(results[b['base_id'],h,c]['scores']['joint'] for h in HISTORIES),
                               'supported_target':sum(results[b['base_id'],h,c]['scores']['supported_target'] for h in HISTORIES)}
                             for c in CONDITIONS}})
    write(OUT/'case_selection.json',{'post_hoc':True,'selection_used_for_aggregate':False,
        'selected_cases':SELECTED,'all40':ledger,
        'rule':'Editorial selection after completed evaluation: two readable full-history successes, '
               'one favorable SSC/prose contrast and one reverse contrast. Not a new evaluation subset.'})
    for bid,note in SELECTED.items():
        folder=OUT/'cases'/bid;folder.mkdir(parents=True,exist_ok=True)
        b=next(b for b in data['bases'] if b['base_id']==bid)
        case={'base':b,'selection_note':note,'post_hoc':True,
              'histories':[c for c in data['cases'] if c['base_id']==bid],
              'outputs':{h:{c:results[bid,h,c] for c in CONDITIONS} for h in HISTORIES}}
        write(folder/'case.json',case)
        for h in HISTORIES:
            for c in CONDITIONS:
                source=RUN/'calls/formal'/bid/h/c
                for filename in ('request.json','response.json'):
                    dest=folder/'saved_calls'/h/c/filename;dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source/filename,dest)
        lines=[f'# {bid}: {b["object"]}', '', note,'',
            '**Constructed symbolic history; not a physical Habitat rollout or a video.**',
            'All four histories and all four arms are included. Editorial/post-hoc selection; not aggregate evidence.',
            '',f'World: `{b["old"]}` → `{b["new"]}`. Target: `{b["target"]}`; unrelated: `{b["unrelated"]}`.',
            '', '| History | Gold target | SSC target | Prose target | No links | No attribution |',
            '|---|---|---|---|---|---|']
        for item in case['histories']:
            h=item['history'];pred=[case['outputs'][h][c]['prediction'].get('target_location','INVALID') for c in CONDITIONS]
            lines.append('| '+' | '.join([h,item['gold']['target_location']]+pred)+' |')
        lines+=['','`case.json` includes full physical/social records, joins, StateStore snapshots, oracle access logs and scores.',
                '`saved_calls/` contains every original request/response for the selected base.',
                'See `history_panel.png` (also PDF/SVG); no intermediate trend was inferred from endpoint scores.',
                '', 'The ablated input can be underdetermined: `unknown` may be rational even when it fails full-information gold.',
                'The favorable and reverse contrasts do not establish overall superiority of either serialization.']
        (folder/'README.md').write_text('\n'.join(lines)+'\n')


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    summary=read(OUT/'formal_summary.json');costs=read(OUT/'costs.json')
    metrics=['world_accuracy','target_accuracy','unrelated_accuracy','joint_accuracy',
             'world_pair_invariant_correct','target_correct_change','unrelated_pair_invariant_correct',
             'paired_joint_correct','supported_target_accuracy','all_four_joint_correct']
    with (OUT/'submission_table.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['metric','condition','percent','ci95_low','ci95_high'])
        for m in metrics:
            for c in CONDITIONS:
                v=summary['conditions'][c][m];w.writerow([m,c,100*v['mean'],*[100*x for x in v['ci95']]])
    tex=['% Requires booktabs. Constructed text diagnostic; all failures retained.',
         r'\begin{tabular}{lrrrr}',r'\toprule',r'Metric (\%) & SSC & Prose & No links & No attribution \\',r'\midrule']
    names=['World location','Target state','Unrelated state','Joint state',
           'World invariant and correct','Target correct change','Unrelated invariant and correct',
           'Paired joint correctness','State + exact physical citation','All four histories jointly correct']
    for m,name in zip(metrics,names):
        values=[str(Decimal(str(100*summary['conditions'][c][m]['mean'])).quantize(
            Decimal('.01'),rounding=ROUND_HALF_UP)) for c in CONDITIONS]
        tex.append(name+' & '+' & '.join(values)+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}',
        '% SSC-Prose paired-joint delta +2.50 pp, paired base-cluster 95% CI [-5.00, 10.00].',
        '% Deletion arms remove information. Not equal-information representation comparisons.'])
    (OUT/'submission_table.tex').write_text('\n'.join(tex)+'\n')
    with (OUT/'cost_table.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['condition','calls','input_tokens','output_tokens','total_tokens','sum_request_seconds'])
        for c in CONDITIONS:
            v=costs['formal/'+c];w.writerow([c,v['actual_request_attempts'],v['input_tokens'],v['output_tokens'],v['total_tokens'],v['elapsed_seconds']])
    colors=['#2463A0','#D48730','#7E8793','#97679B']
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharey=True)
    for ax,m,title in zip(axes,['world_pair_invariant_correct','target_correct_change','paired_joint_correct'],
                         ['World remains correct','Target changes correctly','Paired joint correctness']):
        for i,c in enumerate(CONDITIONS):
            v=summary['conditions'][c][m];y=v['mean']*100;lo,hi=[x*100 for x in v['ci95']]
            ax.bar(i,y,color=colors[i],width=.65)
            ax.errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='none',color='#243343',capsize=4,lw=1.4)
            ax.text(i,min(hi+3,105),f'{y:.1f}',ha='center',fontsize=10)
        ax.set_xticks(range(4),['SSC','Prose','No\nlinks','No\nattribution'])
        ax.set_title(title,fontsize=11);ax.set_ylim(0,112);ax.set_yticks([0,25,50,75,100])
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Correct (%)')
    fig.suptitle('Controlled information histories: fixed world, different acquired evidence',fontsize=13,y=.98)
    fig.text(.5,.03,'40 constructed bases; 640 calls; failures included. Error bars: paired-base bootstrap 95% marginal CIs.\n'
             'Deletion arms remove information. SSC vs lossless prose: +2.5 pp paired-joint difference, 95% CI [-5.0, 10.0].',
             ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.15,1,.94])
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'mechanism_diagnostic.{ext}',dpi=180)
    plt.close(fig)
    for bid in SELECTED:
        folder=OUT/'cases'/bid;case=read(folder/'case.json');b=case['base']
        fig,axes=plt.subplots(1,4,figsize=(14,5))
        for ax,h,title in zip(axes,HISTORIES,['Witnessed move','Unseen / no relevant report','Relevant report delivered','Relevant report NOT delivered']):
            item=next(x for x in case['histories'] if x['history']==h);ax.axis('off')
            ax.set_title(title,fontsize=10,pad=14)
            ax.text(.03,.92,f"World: {item['gold']['world_location']}\nGold target: {item['gold']['target_location']}\n"
                    f"Gold unrelated: {item['gold']['unrelated_location']}",fontsize=10,va='top',linespacing=1.6)
            for j,c in enumerate(CONDITIONS):
                row=case['outputs'][h][c];p=row['prediction'];location=p.get('target_location','INVALID')
                ax.text(.03,.56-j*.12,LABELS[j]+': '+location,fontsize=9,
                        color='#206B47' if row['scores']['target_location'] else '#A43C3C')
            ax.text(.03,.03,'Exact target physical evidence:\n'+item['gold']['target_evidence'],fontsize=8,color='#59616C')
        fig.suptitle(f"{bid}: {b['object']} | target={b['target']}, unrelated={b['unrelated']}",fontsize=14,y=.98)
        fig.text(.5,.08,'Constructed symbolic history, NOT a physical rollout/video. Values are saved model outputs, not a recreated simulation.\n'
                 'Post-hoc illustration; full four-arm records included. Green/red mark target-state correctness only; see case.json for all fields.',
                 ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.16,1,.93])
        for ext in ('png','pdf','svg'):fig.savefig(folder/f'history_panel.{ext}',dpi=160)
        plt.close(fig)


def seal():
    write(OUT/'SHA256SUMS.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
          if p.is_file() and p.name!='SHA256SUMS.json' and '__pycache__' not in p.parts})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-export',action='store_true')
    args=parser.parse_args()
    if not args.from_export:export()
    render();seal()
    print('Export/render complete; no inference. Directory:',OUT)
