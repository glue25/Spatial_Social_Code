"""Audit one real message -> SSC delta -> next action; draw an offline 3-panel case.

Extraction needs frozen local artifacts. --from-export re-renders without them.
No simulator, LLM call, or synthetic scene imagery is used.
"""
import argparse
import json
import hashlib
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'workdirs/habitat_memory_controls_20260922_v1'
OUT = ROOT / 'reports/habitat_state_update_case_20260923'
KEY = 'claim.agent_0.cabinet_20'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def records(p):
    return [(i, json.loads(line)) for i, line in enumerate(p.read_text().splitlines(), 1)]


def select(path, predicate):
    (line, row), = [(i, r) for i, r in records(path) if predicate(r)]
    return {'source': str(path.relative_to(ROOT)), 'source_sha256': sha(path),
            'line': line, 'record': row}


def extract(out):
    job = SOURCE / 'rollouts/0032'
    result = read(job / 'result.json')
    artifact = Path(result['artifact'])
    manifest = read(artifact / 'manifest.json')
    assert manifest['config_sha256'] == sha(SOURCE / 'frozen_config.json')
    assert manifest['status'] == 'completed' and result['condition'] == 'dynamic_ssc'
    assert int(result['seed']) == 891011 and str(result['episode_id']) == '588'
    agent = lambda r: r['agent_id'] == 'agent_1'
    ev = artifact / 'events.jsonl'
    st = artifact / 'state_before_after.jsonl'
    pl = artifact / 'plans.jsonl'
    evidence = {
        'message': select(ev, lambda r: r['event_type'] == 'message_delivered' and
                          r['turn'] == 2 and r['sender'] == 'agent_0'),
        'update': select(st, lambda r: agent(r) and r['turn'] == 2 and r['reason'] == 'delivered_message'),
        'before_plan': select(pl, lambda r: agent(r) and r['turn'] == 2 and r['phase'] == 'planner_output'),
        'prior_search_feedback': select(ev, lambda r: agent(r) and r['turn'] == 2 and r['event_type'] == 'action_feedback'),
        'next_input': select(pl, lambda r: agent(r) and r['turn'] == 3 and r['phase'] == 'planner_input'),
        'next_plan': select(pl, lambda r: agent(r) and r['turn'] == 3 and r['phase'] == 'planner_output'),
        'next_action_feedback': select(ev, lambda r: agent(r) and r['turn'] == 3 and r['event_type'] == 'action_feedback'),
        'later_goal': select(artifact / 'physical_checks.jsonl', lambda r: r['turn'] == 6),
        'later_place': select(ev, lambda r: agent(r) and r['turn'] == 6 and r['event_type'] == 'action_feedback'),
    }
    # Actual API request contains the exact recorded state context.
    matches = []
    context = evidence['next_input']['record']['context']
    for path in sorted((job / 'calls').glob('*/request.json')):
        for m in read(path)['options'].get('messages', []):
            content = m.get('content')
            if isinstance(content, str) and context in content:
                after_state = content.split('END_SSC_RUNTIME_STATE', 1)[1]
                matches.append({'source':str(path.relative_to(ROOT)), 'source_sha256':sha(path),
                                'role':m['role'], 'contains_exact_state_context':True,
                                'message_also_in_native_observation':
                                    evidence['message']['record']['message'] in after_state,
                                'user_content_sha256':hashlib.sha256(content.encode()).hexdigest()})
    assert len(matches) == 1
    evidence['actual_request'] = matches[0]
    frame_index = read(artifact / 'frames/index.json')
    frames = []
    for turn in (2, 3, 6):
        f = frame_index['turns'][str(turn)]['agent_1'][-1]
        path = Path(f['path'])
        name = f'agent_1_turn_{turn:02d}.png'
        shutil.copy2(path, out / name)
        frames.append({**{k:v for k,v in f.items() if k != 'path'}, 'file':name,
                       'source':str(path.relative_to(ROOT)), 'sha256':sha(path)})
    # Matched NL is contextual evidence, not a counterfactual intervention.
    nl_job = SOURCE / 'rollouts/0030'
    nl_result = read(nl_job / 'result.json')
    assert nl_result['task_id'] == result['task_id'] and nl_result['seed'] == result['seed']
    nl = Path(nl_result['artifact'])
    evidence['nl_message'] = select(nl / 'events.jsonl', lambda r:
        r['event_type'] == 'message_delivered' and r['turn'] == 2 and r['sender'] == 'agent_0')
    evidence['nl_next_plan'] = select(nl / 'plans.jsonl', lambda r:
        agent(r) and r['turn'] == 3 and r['phase'] == 'planner_output')
    evidence['nl_next_input'] = select(nl / 'plans.jsonl', lambda r:
        agent(r) and r['turn'] == 3 and r['phase'] == 'planner_input')
    evidence['nl_later_goal'] = select(nl / 'physical_checks.jsonl', lambda r: r['turn'] == 7)
    data = {'task_id':result['task_id'], 'episode_id':'588', 'job_index':32,
            'seed':891011, 'model':manifest['model'], 'condition':'dynamic_ssc',
            'selection':'Post-hoc illustrative process case; not a causal or robustness test.',
            'highlighted_key':KEY, 'frames':frames, 'evidence':evidence,
            'figure_text':'Compact semantic summaries; full original records retained in evidence.'}
    validate(data)
    (out / 'evidence_chain.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    return data


def validate(data):
    e = data['evidence']
    msg = e['message']['record']
    update = e['update']['record']
    assert msg['receivers'] == ['agent_1'] and msg['sender'] == 'agent_0' and msg['turn'] == 2
    assert update['before']['version'] == 7 and update['after']['version'] == 8
    assert KEY not in update['before']['social']
    assert update['before']['spatial'] == update['after']['spatial'], 'Do not depict a direct physical update from this message'
    fact = update['after']['social'][KEY]
    assert fact['value'] == msg['message'] and fact['evidence'] == msg['message']
    assert fact['source_type'] == 'delivered_message' and fact['source_agent'] == 'agent_0'
    assert fact['status'] == 'reported' and fact['turn'] == 2 and fact['supersedes'] is None
    assert e['before_plan']['record']['action'] == 'FindObjectTool[vase]'
    assert 'No objects are found yet' in e['prior_search_feedback']['record']['outcome']['feedback']
    inp, plan = e['next_input']['record'], e['next_plan']['record']
    assert inp['state_version_read'] == plan['state_version_read'] == 11
    assert KEY in inp['context'] and msg['message'] in inp['context']
    assert 'status=reported; source=delivered_message; source_agent=agent_0; turn=2' in inp['context']
    assert plan['action'] == 'Navigate[cabinet_20]' and plan['state_marker_found_in_actual_prompt']
    feedback = e['next_action_feedback']['record']
    assert feedback['action']['serialized'] == plan['action'] and feedback['outcome']['success']
    assert feedback['outcome']['had_physical_effect'] is True
    assert e['actual_request']['contains_exact_state_context']
    assert e['actual_request']['message_also_in_native_observation'] is True
    assert e['later_place']['record']['action']['serialized'] == 'Place[vase_4, on, table_22, None, None]'
    goal, = [g for g in e['later_goal']['record']['checks'] if g['goal_id'] == 'is_on_top:vase_4:table_22']
    assert goal['satisfied'] and e['later_goal']['record']['measurement_valid']
    assert e['nl_next_plan']['record']['action'] == 'FindObjectTool[vase on cabinet_20]'
    assert e['nl_message']['record']['message'] in e['nl_next_input']['record']['context']
    return True


def plot(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from PIL import Image
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig = plt.figure(figsize=(13.2, 4.65), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    dark, blue, green, grey = '#253349', '#2166ac', '#16836b', '#64748b'
    ax.text(.035, .94, 'From a delivered report to the next action', fontsize=17, weight='bold', color=dark)
    ax.text(.035, .88, 'Habitat inspection · episode 588 · agent_1 · seed 891011 · Dynamic SSC', fontsize=11, color=grey)
    xs = [.035, .365, .695]; width = .275
    for x, title, subtitle in zip(xs, ['1  Before update', '2  Message + state update', '3  Next planner read'],
                                 ['Turn 2 · state v7', 'Turn 2 · v7 → v8', 'Turn 3 · state v11']):
        ax.add_patch(FancyBboxPatch((x, .15), width, .65, boxstyle='round,pad=0.012',
                                   facecolor='#f8fafc', edgecolor='#dbe3ed', linewidth=1.1))
        ax.text(x+.012, .745, title, fontsize=12, weight='bold', color=dark)
        ax.text(x+.012, .695, subtitle, fontsize=10, color=grey)
    for x in [.321, .651]:
        ax.add_patch(FancyArrowPatch((x, .49), (x+.035, .49), arrowstyle='-|>',
                                    mutation_scale=20, linewidth=1.7, color=grey))
    x=xs[0]
    ax.text(x+.012, .585, 'Goal: move the vase\nto the bedroom table', fontsize=12, color=dark, linespacing=1.5)
    ax.text(x+.012, .445, 'Vase location:\nnot recorded', fontsize=13, color=grey, linespacing=1.5)
    ax.text(x+.012, .27, 'Current plan: search for the vase.', fontsize=10.5, color=grey)
    x=xs[1]
    ax.text(x+.012, .625, 'agent_0 → agent_1', fontsize=12, weight='bold', color=blue)
    ax.text(x+.012, .575, '“The vase is on cabinet_20\nin kitchen_1 …”', fontsize=12, color=dark, linespacing=1.5, va='top')
    ax.add_patch(FancyBboxPatch((x+.007,.285),width-.014,.145,boxstyle='round,pad=0.009',
                               facecolor='#e8f2fc',edgecolor='none'))
    ax.text(x+.02,.38,'+ Reported vase location',fontsize=12,weight='bold',color=blue)
    ax.text(x+.02,.323,'cabinet_20  ·  from agent_0  ·  t=2',fontsize=10.3,color=blue)
    ax.text(x+.012,.215,'New claim; not a verified observation.',fontsize=9.5,color=grey)
    x=xs[2]
    ax.text(x+.012,.595,'Navigate[cabinet_20]',fontsize=12,weight='bold',color=green)
    ax.text(x+.012,.54,'Tool feedback: navigation succeeded',fontsize=9.5,color=green)
    photo = fig.add_axes([x+.072,.19,.13,.32]); photo.axis('off')
    photo.imshow(Image.open(out/'agent_1_turn_03.png'))
    ax.text(x+.1375,.165,'Actual saved frame · turn 3',ha='center',fontsize=8.5,color=grey)
    ax.text(.035,.067,'Highlighted: one newly attributed report. Intermediate state revisions v8→v11 retain this claim.',
            fontsize=9,color=grey)
    ax.text(.035,.024,'Observed execution trace, not a causal ablation. The message is also present in the native observation channel.',
            fontsize=9,color=grey)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'state_update_three_panel.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-export',action='store_true')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    data=read(out/'evidence_chain.json') if args.from_export else extract(out)
    validate(data)
    for frame in data['frames']:
        assert sha(out/frame['file']) == frame['sha256']
    plot(data,out)
    print('PASS: delivered message, actual state delta, next API input/action, feedback and saved frame.',out)


if __name__=='__main__':
    main()
