"""Scoring only; same completed IDs for historical Pro and fresh Flash."""
import json
from pathlib import Path
from datetime import datetime, timezone
import sys

HOME = Path(__file__).resolve().parent
sys.path.insert(0, str(HOME))
from legacy_loader import load
load()
import evaluate_mmtom_jdcloud as mm
from egomas.src.jdcloud_gemini import _extract_text


def main():
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'datasets': {},
              'scope': 'Historical-protocol development replication; provider and historical byte-identity limitations in frozen configs. Invalids retained, incomplete samples not scored.'}
    for dataset in ('mmtom', 'maegoqa'):
        out = HOME / 'runs' / dataset
        if not (out / 'frozen_config.json').exists():
            continue
        cfg = json.loads((out / 'frozen_config.json').read_text())
        old = {r['index']: r for line in Path(cfg['reference']).read_text().splitlines()
               if line.strip() for r in [json.loads(line)]}
        inputs = {r['index']: r for r in json.loads((out / 'inputs.json').read_text())}
        new = {r['index']: r for p in (out / 'results').glob('*.json') for r in [json.loads(p.read_text())]}
        n = len(new)
        values = {model: {method: {'correct': 0, 'invalid': 0, 'n': n}
                          for method in ('direct', 'ssc')} for model in ('historical_pro', 'fresh_flash')}
        for index, result in new.items():
            row = inputs[index]['row']
            truth = str(row['answer']).lower() if dataset == 'mmtom' else row['options'].index(row['answer'])
            historical = old[index]
            if dataset == 'mmtom':
                predictions = [historical.get('direct_prediction'), historical.get('prediction'),
                               result.get('direct_prediction'), result.get('prediction')]
                if predictions[2] is None:
                    # Recover a baseline from its own saved reply if a later SSC
                    # stage failed; never replace the SSC answer with it.
                    for request in sorted((out / 'calls' / f'{index:04d}').glob('*/request.json')):
                        req = json.loads(request.read_text())
                        prompt = req['parts'][-1].get('text')
                        if prompt not in [template.format(question=row['question']) for template in (mm.DIRECT_PROMPT, mm.DIRECT_REPAIR_PROMPT)]:
                            continue
                        response = request.parent / 'response.json'
                        if response.exists():
                            predictions[2] = mm.parse_ab(_extract_text(json.loads(response.read_text())))
                valid = lambda value: value in ('a', 'b')
            else:
                baseline = out / 'baseline' / f'{index:04d}.json'
                fresh_direct = result.get('baseline_prediction_index')
                if fresh_direct is None and baseline.exists():
                    fresh_direct = json.loads(baseline.read_text())['baseline_prediction_index']
                predictions = [historical.get('baseline_prediction_index'), historical.get('prediction_index'),
                               fresh_direct, result.get('prediction_index')]
                valid = lambda value: type(value) is int and 0 <= value < 5
            for (model, method), prediction in zip(
                    [('historical_pro', 'direct'), ('historical_pro', 'ssc'),
                     ('fresh_flash', 'direct'), ('fresh_flash', 'ssc')], predictions):
                cell = values[model][method]
                cell['invalid'] += not valid(prediction)
                cell['correct'] += valid(prediction) and prediction == truth
        for methods in values.values():
            for cell in methods.values():
                cell['accuracy'] = cell['correct'] / n if n else None
        report['datasets'][dataset] = {'selected': len(inputs), 'processed': n,
            'paired_indices': sorted(new), 'comparison': values}
    (HOME / 'comparison_latest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
