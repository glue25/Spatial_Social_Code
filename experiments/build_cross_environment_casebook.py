"""Offline casebook: audited saved artifacts, not new experiments or causal tests.

Each case retains original outputs, input manifests, source hashes and limitations.
Static video compilation is never described as online state updating.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import re
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/cross_environment_casebook_20260923'


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source(p):
    return {'path':str(p.relative_to(ROOT)), 'sha256':sha(p)}


def copy(p, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)
    return {**source(p), 'export':str(dest.relative_to(OUT)), 'export_sha256':sha(dest)}


def lines(p):
    return [(i,json.loads(s)) for i,s in enumerate(p.read_text().splitlines(),1)]


def one(p, pred):
    (n,r), = [(n,r) for n,r in lines(p) if pred(r)]
    return {**source(p), 'line':n, 'record':r}


def contact_sheet(items, output):
    from PIL import Image, ImageDraw, ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    w,h=330,270;cols=min(4,len(items));nrows=(len(items)+cols-1)//cols
    canvas=Image.new('RGB',(cols*w,nrows*h),'white');draw=ImageDraw.Draw(canvas)
    for i,(p,label) in enumerate(items):
        im=Image.open(p).convert('RGB');im.thumbnail((w-12,h-42))
        x=(i%cols)*w;y=(i//cols)*h
        canvas.paste(im,(x+(w-im.width)//2,y+6));draw.text((x+7,y+h-31),label,font=font,fill='#26354a')
    canvas.save(output)


def render_card(case, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,3,figsize=(13,4.2))
    fig.suptitle(case['title'],fontsize=14,fontweight='bold',y=.98)
    for ax,title,body in zip(axs,case['panel_titles'],case['panels']):
        ax.set_facecolor('#f5f8fc');ax.set(xticks=[],yticks=[])
        for s in ax.spines.values():s.set_color('#dce4ed')
        ax.text(.05,.92,title,va='top',fontsize=12,weight='bold',color='#235b89',transform=ax.transAxes)
        wrapped='\n\n'.join(textwrap.fill(t,width=39) for t in body.split('\n\n'))
        ax.text(.05,.78,wrapped,va='top',fontsize=10.5,linespacing=1.35,transform=ax.transAxes,color='#27354b')
    fig.text(.035,.05,textwrap.fill(case['figure_caveat'],width=145),fontsize=8.5,color='#59677a')
    fig.tight_layout(rect=[.015,.12,.985,.88])
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'evidence_flow.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def finish_case(case,folder):
    write(folder/'case.json',case)
    render_card(case,folder)
    text = f"# {case['title']}\n\n**Environment:** {case['environment']}  \n**Scope:** {case['scope']}\n\n"
    for heading,body in zip(case['panel_titles'],case['panels']):text+=f'## {heading}\n\n{body}\n\n'
    text+='## Limits / interpretation\n\n'+'\n'.join('- '+x for x in case['limitations'])+'\n\n'
    text+='## Files\n\n- `case.json`: complete selected saved records, labels and provenance.\n- `spatial_code.py` / `social_code.py` where present: literal generated code, not automatically executed.\n- `input_preview.jpg` where present: labeled visual preview; see media manifest for exact scope.\n- `evidence_flow.png/pdf/svg`: compact editorial summary, not new model output.\n'
    (folder/'README.md').write_text(text)
    return {'id':case['id'],'environment':case['environment'],'title':case['title'],
            'scope':case['scope'],'folder':str(folder.relative_to(OUT)),'grade':case['grade']}


def habitat_cases():
    base=ROOT/'workdirs/habitat_memory_controls_20260922_v1'
    rows=[]
    # Keep the fully audited primary three-panel case intact, including raw frames.
    src=ROOT/'reports/habitat_state_update_case_20260923';dest=OUT/'habitat/inspection_588'
    dest.mkdir(parents=True,exist_ok=True)
    for p in src.iterdir():
        if p.is_file():copy(p,dest/p.name)
    rows.append({'id':'habitat_588','environment':'Habitat','title':'Inspection: attributed report -> next navigation',
                 'scope':'Online update; Flash seed891011; job32','folder':'habitat/inspection_588','grade':'A: actual request/state/action chain'})
    for idx,episode,turn,sender,receiver,target,title in [
        (42,618,1,'agent_2','agent_0','chest_of_drawers_33','Bottle location report -> targeted navigation'),
        (37,983,5,'agent_0','agent_2','bed_9','Unsuccessful search -> attributed jug report -> navigation')]:
        folder=OUT/'habitat'/f'episode_{episode}';folder.mkdir(parents=True,exist_ok=True)
        job=base/'rollouts'/f'{idx:04d}';r=read(job/'result.json');p=Path(r['artifact'])
        ev=p/'events.jsonl';st=p/'state_before_after.jsonl';pl=p/'plans.jsonl'
        msg=one(ev,lambda x:x['event_type']=='message_delivered' and x['turn']==turn and x['sender']==sender)
        update=one(st,lambda x:x['reason']=='delivered_message' and x['turn']==turn and x['agent_id']==receiver)
        inp=one(pl,lambda x:x['phase']=='planner_input' and x['turn']==turn+1 and x['agent_id']==receiver)
        plan=one(pl,lambda x:x['phase']=='planner_output' and x['turn']==turn+1 and x['agent_id']==receiver)
        action=one(ev,lambda x:x['event_type']=='action_feedback' and x['turn']==turn+1 and x['agent_id']==receiver)
        key=f'claim.{sender}.{target}';before=update['record']['before'];after=update['record']['after']
        assert key not in before['social'] and after['social'][key]['status']=='reported'
        assert msg['record']['message']==after['social'][key]['value']
        assert key in inp['record']['context'] and plan['record']['action']==f'Navigate[{target}]'
        assert action['record']['outcome']['success'] and plan['record']['state_marker_found_in_actual_prompt']
        requests=[]
        for q in sorted((job/'calls').glob('*/request.json')):
            for m in read(q)['options'].get('messages',[]):
                content=m.get('content','')
                if isinstance(content,str) and inp['record']['context'] in content:
                    requests.append({**source(q),'exact_state_present':True,
                                     'native_message_also_present':msg['record']['message'] in content.split('END_SSC_RUNTIME_STATE',1)[1]})
        assert len(requests)==1
        fi=read(p/'frames/index.json')['turns'][str(turn+1)][receiver][-1]
        media=copy(Path(fi['path']),folder/'next_action_frame.png')
        count=r['strict_physical']['completed_goals'];total=r['strict_physical']['total_goals']
        case={'id':f'habitat_{episode}','environment':'Habitat','title':title,'grade':'A: actual request/state/action chain',
          'scope':f'Online Dynamic SSC; Flash; seed891011; episode{episode}; job{idx}',
          'panel_titles':['1  Before delivery','2  New attributed report','3  Next actual action'],
          'panels':[f'{receiver}: no {sender}-attributed report for {target} in the pre-update state.',
                    f'Turn {turn}: {sender} -> {receiver}.\n\n'+msg['record']['message'],
                    f'Turn {turn+1}: Navigate[{target}].\n\nNavigation feedback: succeeded.\n\nFinal shared goals: {count}/{total} (not the immediate navigation result).'],
          'figure_caveat':'One illustrative trace; not a causal ablation. The message may also appear in the native observation channel.',
          'limitations':['New reported claim, not verified object-location truth; do not label as a corrected false belief.',
                          'Actual next request, delta and action verified; planner causality is not isolated.',
                          'Episode983 ends at2/3; its NL pair had a service problem. Do not present as a clean superiority comparison.' if episode==983 else
                          'Bottle placement later requires inter-agent transfer. Do not compress it into this next navigation.'],
          'evidence':{'message':msg,'update':update,'next_input':inp,'next_plan':plan,'feedback':action,'actual_request':requests[0]},
          'result':r,'media':[media]}
        rows.append(finish_case(case,folder))
    return rows


def mmtom_cases(ledger):
    base=ROOT/'reports/flash_full_results_20260920/mmtom'
    rp=base/'flash_results.jsonl.gz';ip=base/'original_frozen_inputs/inputs.json'
    results={r['index']:r for r in map(json.loads,gzip.open(rp,'rt'))}
    inputs={r['index']:r for r in read(ip)}
    cp=base/'calls.jsonl.gz';calls={i:[] for i in (84,318,6)}
    for r in map(json.loads,gzip.open(cp,'rt')):
        if r['index'] in calls:calls[r['index']].append(r)
    for i,r in sorted(results.items()):
        ls=re.findall(r'(?im)^\s*([AB])[.\s]*$',r.get('direct_raw',''))
        ledger.append({'environment':'MMToM','index':i,'direct_correct':r.get('direct_correct'),
                       'ssc_correct':r.get('correct'),'last_standalone_letter':ls[-1].lower() if ls else None,
                       'saved_direct_prediction':r.get('direct_prediction'),'dense':r.get('uncertainty_triggered'),
                       'selected':i in calls})
    titles={84:'Container inspection and a recorded belief change',318:'Goal inference with a historical answer-conditioned scorer',6:'Counterexample: recorded belief versus fused final answer'}
    panels={
      84:["Robert opens and closes the third kitchen cabinet while looking for a dish bowl.",
          "Generated records include an earlier inside belief and a later not-inside belief.\n\nThe final answer also uses deterministic inspection evidence and an option-conditioned posterior.",
          "Direct: A (wrong).\n\nSSC: B (correct).\n\nThis is a historical pipeline example, not an option-blind update experiment."],
      318:["Jessica inspects kitchen appliances; the question supplies a belief premise and asks which object she is seeking.",
           "The neural social code favors salmon.\n\nHistorical query evidence and Bayesian fusion change the final answer toward apple.",
           "Direct: A (wrong).\n\nSSC final: B (correct).\n\nThe successful endpoint cannot be attributed to the neural code alone."],
      6:["Elizabeth searches kitchen containers and then approaches a living-room cabinet while seeking a wine glass.",
         "Generated social code records an inside-cabinet belief.\n\nThe fused historical answer nevertheless selects the opposite option.",
         "Direct: A (correct).\n\nSSC final: B (wrong).\n\nUseful for showing that intermediate records do not guarantee a correct final answer."]}
    rows=[]
    for idx in (84,318,6):
        folder=OUT/'mmtom'/f'index_{idx:04d}';folder.mkdir(parents=True,exist_ok=True)
        r=results[idx];inp=inputs[idx];assert not r['uncertainty_triggered']
        assert r['prediction']== 'b' and r['direct_prediction']=='a'
        terminal=re.findall(r'(?im)^\s*([AB])[.\s]*$',r['direct_raw'])
        assert terminal[-1].lower()==r['direct_prediction']
        media=[];previews=[]
        for n in r['image_details']['frame_numbers']:
            matches=[Path(s.replace('${PROJECT_ROOT}',str(ROOT))) for s in inp['media_files'] if f'Action_{n:04d}_' in s]
            path,=matches
            m=copy(path,folder/'frames'/path.name);m['frame_number']=n;media.append(m)
            previews.append((folder/'frames'/path.name,f'input frame {n}'))
        assert len(media)==8
        contact_sheet(previews,folder/'input_preview.jpg')
        for layer in ['spatial','social']:(folder/f'{layer}_code.py').write_text(r[f'{layer}_code']+'\n')
        write(folder/'saved_calls.json',calls[idx])
        case={'id':f'mmtom_{idx}','environment':'MMToM','title':titles[idx],
              'grade':'B: historical pipeline; not clean option-blind evidence','scope':'Flash historical600; static frame+text compilation; no dense refinement in selected cases',
              'panel_titles':['1  Observed task sequence','2  Actual intermediate code','3  Saved prediction'],
              'panels':panels[idx],'figure_caveat':'Post-hoc historical example. Query/options, deterministic rules and answer-conditioned scoring are part of this protocol.',
              'limitations':['Static video compilation, not an online state-store before/after trace.',
                             'Generated belief records are hypotheses, not direct proof of attention or belief.',
                             'The full historical code includes option-conditioned query evidence, rule bridges and posteriors.',
                             'Case0 was excluded: raw Direct ends in B but archived parsed prediction is A; original aggregate scores were not modified.'],
              'input':inp,'result':r,'media':media,'source_files':[source(rp),source(ip),source(cp)]}
        rows.append(finish_case(case,folder))
    return rows


def ma_cases(ledger):
    base=ROOT/'workdirs/ma_rich_ablation100_20260922_v1';gold=read(base/'SCORER_ONLY_gold.json')
    rows=[];selected=(1038,950,1337,1597)
    for p in sorted(base.glob('worker*/results/*.json')):
        r=read(p);i=r['index'];g=gold[str(i)]['answer_index']
        ledger.append({'environment':'MA-EgoQA','index':i,'selected':i in selected,
                       'outcomes':{k:{'valid':v.get('valid'),'prediction':v.get('prediction_index'),
                                      'correct':bool(v.get('valid') and v.get('prediction_index')==g)} for k,v in r['conditions'].items()}})
        if i not in selected:continue
        worker=p.parents[1];inp=next(x for x in read(worker/'inputs.json') if x['index']==i)
        compiled=read(worker/'compiled'/p.name);assert compiled['valid']
        folder=OUT/'ma_egoqa'/f'index_{i:04d}';folder.mkdir(parents=True,exist_ok=True)
        views={k:read(worker/'views'/f'{i:04d}'/f'{k}.json') for k in r['conditions']}
        requests={};media_sequences=[]
        for c in r['conditions']:
            paths=list((worker/'calls'/f'{i:04d}'/f'answer_{c}').glob('*/request.json'))
            assert paths
            request=read(sorted(paths)[0]);assert 'parts' in request
            requests[c]=request
            media_sequences.append([part['sha256'] for part in request['parts'] if part.get('mimeType')=='video/mp4'])
            text='\n'.join(part.get('text','') for part in request['parts'])
            assert inp['row']['question'] in text
        assert all(v==media_sequences[0] for v in media_sequences), 'Unequal answer-stage clips'
        write(folder/'answer_requests.json',requests)
        write(folder/'ablation_views.json',views)
        media=[{**source(Path(s)),'clip_id':Path(s).name,'kind':'input audiovisual clip; not redistributed'} for s in inp['media_files']]
        assert sorted(m['sha256'] for m in media)==sorted(media_sequences[0])
        evid={}
        for line in compiled['spatial_trace'].splitlines():
            x=json.loads(line);v=x['values']
            if 'evidence_id' in v:evid[v['evidence_id']]=x
        cited=r['conditions']['ssc'].get('parsed',{}).get('evidence_ids',[])
        assert set(cited)<=set(compiled['valid_evidence_ids'])
        chosen={1038:['S1','S2','S3'],950:['S1','S2','S3'],1337:['S4','S7'],1597:['S1']}[i]
        previews=[];frame_meta=[]
        for eid in chosen:
            v=evid[eid]['values'];clip=next(Path(s) for s in inp['media_files'] if Path(s).name==v['clip_id'])
            times=re.findall(r'(\d+):(\d+)',v['evidence_span']);assert len(times)==2
            sec=sum(int(m)*60+int(s) for m,s in times)/2
            target=folder/f'preview_{eid}.jpg'
            import cv2
            cap=cv2.VideoCapture(str(clip));assert cap.isOpened(),clip
            cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000);ok,frame=cap.read()
            assert ok, (clip,sec)
            frame_number=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
            assert cv2.imwrite(str(target),frame)
            frame_meta.append({'evidence_id':eid,'clip_id':clip.name,'clip_sha256':sha(clip),
                               'preview_time_seconds':sec,'decoded_frame_number':frame_number,'fps':fps,
                               'preview_file':target.name,'preview_sha256':sha(target),
                               'kind':'post-hoc preview from actual input MP4; not a logged model-sampled frame'})
            previews.append((target,f'{eid}: clip +{sec:.1f}s'))
        contact_sheet(previews,folder/'input_preview.jpg')
        for layer in ['spatial','social']:(folder/f'{layer}_code.py').write_text(compiled[f'{layer}_code']+'\n')
        title={1038:'A guess, a correction, and explicit evidence references',950:'Self-report linked to a belief record and cited answer',1337:'Counterexample: same day is not simultaneous',1597:'Time-filtered answer; invalid ablation is not mechanism evidence'}[i]
        interpretation={1038:'S2 records Tasha asking about a boyfriend; S3 records Jake replying no. Social records retain the attributed guess and correction.',
                        950:'S2/S3 record Alice saying the smell makes her feel full. The belief record and final answer refer to those same evidence IDs.',
                        1337:'The full answer cites washing at11:54 and dog-walking at16:25 on DAY4. Those records do not establish concurrency.',
                        1597:'S1 records Katrina using scissors before13:00 on DAY4. Full SSC cites S1; no-attribution also answers correctly.'}[i]
        scores={c:('invalid' if not v.get('valid') else chr(65+v['prediction_index'])) for c,v in r['conditions'].items()}
        case={'id':f'ma_{i}','environment':'MA-EgoQA','title':title,
              'grade':'B: actual clips/code/citations; speech and identities need human sign-off',
              'scope':'Flash rich100; same compiled record across three arms; static audiovisual QA',
              'panel_titles':['1  Actual question / evidence','2  Compiled links','3  Saved answers'],
              'panels':[inp['row']['question'],interpretation,'\n\n'.join(f'{k}: {v}' for k,v in scores.items())+f'\n\nGold: {chr(65+g)}.'],
              'figure_caveat':'Static compilation, not online memory. Quotes/identities are model-generated annotations pending human audiovisual verification.',
              'limitations':['All three arms see the same clips and derive views from one compiled record; the final answer still sees audiovisual clips.',
                             'The input row.answer is an intentional option0 dummy, not ground truth. This export uses SCORER_ONLY_gold.json.',
                             'Quotes, speaker identities and inferred beliefs have not received independent full audio/identity annotation review.',
                             'Full100 outcomes:46/46/49 correct for full/no-references/no-attribution. Individual examples do not establish linkage benefit.',
                             'Case1597 no-references is invalid, not a valid wrong answer; do not claim an evidence-link mechanism success from it.'],
              'input':inp,'gold':gold[str(i)],'compiled':compiled,'results':r,'media':media,'preview_frames':frame_meta,
              'cited_evidence':{eid:evid[eid] for eid in cited},'answer_media_identical':True,
              'source_files':[source(p),source(worker/'compiled'/p.name),source(worker/'inputs.json'),source(base/'SCORER_ONLY_gold.json')]}
        rows.append(finish_case(case,folder))
    return rows


def comind_cases(ledger):
    base=ROOT/'workdirs/comind_symmetric_sam_20260922_v1'
    scorer_path=ROOT/'workdirs/comind_flash_sam_20260920/frozen_v2.py'
    spec=importlib.util.spec_from_file_location('casebook_comind_scorer',scorer_path)
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    copy(scorer_path,OUT/'code/comind_frozen_scorer.py')
    inputs={r['index']:r for r in read(base/'inputs.json')}
    gp=ROOT/'workdirs/comind_native_schema_20260920/SCORER_ONLY_comind_gold.json';gold=read(gp)
    rows=[]
    for p in sorted((base/'results').glob('*.json')):
        r=read(p);idx=r['index'];task=r['task'];gt=gold[r['id']]['gold'];scores={};ious={}
        for c,v in r['conditions'].items():
            pred=v.get('prediction') or {};scores[c]=scorer.score_prediction(task,pred,gt)
            ious[c]={field:scorer.iou(pred.get(field),scorer.ground_truth_value(task,metric,gt))
                     for metric,field in scorer.FIELD_TO_PREDICTION[task].items() if 'iou' in metric}
        ledger.append({'environment':'CoMind','index':idx,'selected':idx in (66,226,5),'scores':scores,'ious':ious})
        if idx not in (66,226,5):continue
        folder=OUT/'comind'/f'index_{idx:04d}';folder.mkdir(parents=True,exist_ok=True)
        inp=inputs[idx];upstream=read(base/'upstream'/p.name);media=[];previews=[]
        assert max(x['time'] for x in inp['images'])<=inp['prediction_time']+1e-6
        for n,img in enumerate(inp['images']):
            path=Path(img['path']);target=folder/'frames'/f'{n:02d}_{path.name}'
            m=copy(path,target);m.update({k:v for k,v in img.items() if k!='path'});media.append(m)
            previews.append((target,f"{img['role']} {img['time']:.2f}s"))
        contact_sheet(previews,folder/'input_preview.jpg')
        ssc=upstream['conditions']['ssc']
        for layer in ['spatial','social']:(folder/f'{layer}_code.py').write_text(ssc.get(f'{layer}_code','')+'\n')
        title={66:'Same SAM, different identity: dough localization',226:'Same SAM, different identity and selected view',5:'Counterexample: a pan description misses the gold object'}[idx]
        d=r['conditions']['direct_sam'];s=r['conditions']['ssc_sam']
        case={'id':f'comind_{idx}','environment':'CoMind','title':title,'grade':'A for saved geometry; B for semantic interpretation',
              'scope':'Flash symmetric SAM300 development set; static visual grounding; no new VLM/SAM calls',
              'panel_titles':['1  Frozen inputs','2  Same SAM rule','3  Localization result'],
              'panels':[f'Task: {task}.\n\nTranscript: {inp["transcript_10s"]}',
                        f'Direct identity: {d["identity"]}.\n\nSSC identity: {s["identity"]}.\n\nSame final-field selection rule and SAM backend.',
                        'Direct+SAM IoU: '+', '.join(f'{k}={v:.3f}' for k,v in ious['direct_sam'].items())+'\n\nSSC+SAM IoU: '+', '.join(f'{k}={v:.3f}' for k,v in ious['ssc_sam'].items())],
              'figure_caveat':'Post-hoc grounding example; all field scores retained. Improvements may come from identity/view selection, not evidence-link reasoning.',
              'limitations':['Same candidate-selection rule does not imply the same generated identity or chosen view.',
                             'Case66 social_code is unknown; it is not evidence for a social-layer contribution.',
                             'Case226 selects different views (Direct helper, SSC leader); do not attribute the difference solely to noun specificity.',
                             'Gold overlays are evaluation-only. Raw SAM boxes are xyxy in1408-square views; scores use converted xywh on the combined canvas.',
                             'Frozen scorer uses IoU>0.5. No selected case lies exactly at0.5. All non-localization field scores are preserved.'],
              'input':inp,'gold':gt,'result':r,'upstream':upstream,'scores':scores,'ious':ious,'media':media,
              'source_files':[source(p),source(base/'upstream'/p.name),source(gp),source(scorer_path)],
              'protocol':read(base/'PROTOCOL.json')}
        rows.append(finish_case(case,folder));plot_boxes(case,folder)
    return rows


def plot_boxes(case,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    canvas=Image.new('RGB',(1408,704),'white')
    for role,x in [('leader',0),('helper',704)]:
        m,=[m for m in case['media'] if m['role']==role and m['kind']=='prediction']
        im=Image.open(OUT/m['export']).convert('RGB');assert im.size==(1408,1408)
        canvas.paste(im.resize((704,704)),(x,0))
    fig,axs=plt.subplots(2,1,figsize=(11,10))
    gt=case['gold'];boxes=gt.get('bounding_boxes') if case['input']['task']=='joint_attention' else [gt['bbox']]
    for ax,c in zip(axs,('direct_sam','ssc_sam')):
        ax.imshow(canvas);ax.axis('off');ax.set_title(c+' | identity: '+case['result']['conditions'][c]['identity'])
        for b in boxes:
            if b:ax.add_patch(Rectangle(b[:2],b[2],b[3],fill=False,edgecolor='#31e55f',linewidth=2.5))
        pred=case['result']['conditions'][c]['prediction']
        for field in case['ious'][c]:
            b=pred.get(field)
            if b:ax.add_patch(Rectangle(b[:2],b[2],b[3],fill=False,edgecolor='#ff6355',linewidth=2.5,linestyle='--'))
    fig.suptitle('Green: evaluation gold | Red dashed: prediction\nSame original prediction frames; coordinates converted by the frozen protocol.',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(folder/'localization_overlay.png',dpi=160,bbox_inches='tight');fig.savefig(folder/'localization_overlay.pdf',bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--only',choices=['habitat','mmtom','ma','comind','all'],default='all')
    parser.add_argument('--from-export',action='store_true',help='Re-render published snapshots without private datasets/workdirs')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);ledger=[];index=[]
    if args.from_export:
        from experiments.build_habitat_update_case import plot as plot_inspection
        for item in read(OUT/'index_all.json'):
            folder=OUT/item['folder']
            if item['id']=='habitat_588':plot_inspection(read(folder/'evidence_chain.json'),folder)
            else:
                case=read(folder/'case.json');render_card(case,folder)
                if case['environment']=='CoMind':plot_boxes(case,folder)
        write(OUT/'SHA256SUMS.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'})
        print('Re-rendered all13 published cases; no inference or dataset access.')
        return
    for name,func in [('habitat',habitat_cases),('mmtom',mmtom_cases),('ma',ma_cases),('comind',comind_cases)]:
        if args.only in ('all',name):
            index.extend(func() if name=='habitat' else func(ledger));print('Exported',name,flush=True)
    write(OUT/f'index_{args.only}.json',index);write(OUT/f'selection_ledger_{args.only}.json',ledger)
    if args.only=='all':
        write(OUT/'SHA256SUMS.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'})
    print('Cases:',len(index))


if __name__=='__main__':
    main()
