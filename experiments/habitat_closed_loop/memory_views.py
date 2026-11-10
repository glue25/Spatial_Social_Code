"""Fact-matched v2 memory renderers; never read another agent's state.

Prose uses the same deterministic updater (a strong rendering control, NOT a
learned summarizer). Local-flat removes explicit layer/source-agent metadata at
readout, not all implicit attribution and not private-state access boundaries.
"""
import json


def records(store, agent_id):
    state = store.snapshot(agent_id)
    rows = []
    for layer in ('spatial', 'social'):
        for key, fact in sorted(state[layer].items()):
            rows.append({'layer': layer, 'key': key, 'value': fact['value'],
                'status': fact['status'], 'source': fact['source_type'],
                'source_agent': fact['source_agent'], 'turn': fact['turn'],
                'confidence': round(fact['confidence'], 2)})
    return state['version'], rows


def render_memory(store, agent_id, view='ssc'):
    if view == 'ssc':
        return store.render(agent_id)
    if view not in ('prose', 'local_flat'):
        raise ValueError('Unsupported memory view')
    version, rows = records(store, agent_id)
    lines = [f'SSC_RUNTIME_STATE agent={agent_id} version={version}']
    for row in rows:
        if view == 'prose':
            lines.append(f"The {row['layer']} record {row['key']} has value "
                f"{json.dumps(row['value'], ensure_ascii=False, sort_keys=True)}. "
                f"Its status is {row['status']}; its source is {row['source']} from "
                f"{row['source_agent']} at turn {row['turn']}, with confidence {row['confidence']:.2f}.")
        else:
            lines.append(json.dumps({k: v for k, v in row.items() if k not in ('layer', 'source_agent')},
                                   ensure_ascii=False, sort_keys=True))
    if not rows:
        lines.append('Unknown: no supported records.')
    lines.append('END_SSC_RUNTIME_STATE')
    return '\n'.join(lines)
