"""Archived executable CoMind pipeline, not typed SSC or an OOF router.

Prompts are extracted verbatim from the frozen original main() AST. The original
output schema is supplied through the native API, NOT appended to prompt text.
No CLI is invoked, and no generated fields/code are repaired after validation.
"""
import ast
import base64
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from spatial_social_code import run_comind_codex_pilot as original
from spatial_social_code.runtime import execute_spatial, execute_social_legacy as execute_social
assert Path(original.__file__).resolve() == ROOT/'spatial_social_code/run_comind_codex_pilot.py'
source = Path(original.__file__).read_text()
main_ast = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
invocations = sorted([n for n in ast.walk(main_ast) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'invoke'], key=lambda n:n.lineno)
assert len(invocations) == 4
PROMPTS = dict(zip(('direct','spatial','social','answer'),
                   [compile(ast.Expression(n.args[0]), '<frozen-original-prompt>', 'eval') for n in invocations]))
SUFFIX = '\n\nReturn only one JSON object matching this schema; no markdown or prose:\n'
assert SUFFIX in [n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and isinstance(n.value,str)]

def build_prompt(stage, context):
    return eval(PROMPTS[stage], {'__builtins__':{}}, context)

def validate_schema(value, schema):
    """Validate precisely the subset used by the archived CoMind schemas."""
    kind=schema['type']
    if kind=='object':
        if not isinstance(value,dict) or set(value)!=set(schema['required']):
            raise ValueError('Original required-key contract mismatch')
        for k,v in value.items():validate_schema(v,schema['properties'][k])
    elif kind=='array':
        if not isinstance(value,list) or not schema.get('minItems',0)<=len(value)<=schema.get('maxItems',float('inf')):
            raise ValueError('Original array type/length contract mismatch')
        for v in value:validate_schema(v,schema['items'])
        if schema.get('uniqueItems') and len({json.dumps(v,sort_keys=True) for v in value})!=len(value):
            raise ValueError('Original uniqueItems contract mismatch')
    elif kind=='number':
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Expected finite number')
    elif kind=='string':
        if not isinstance(value,str):raise ValueError('Expected string')
    else:raise ValueError('Unsupported original schema type: '+kind)
    return value

def infer(row, client, write, fatal):
    context = {'evidence':original.evidence_prompt(row)}
    images = [{'inline_data': {'mimeType':'image/jpeg' if Path(i['path']).suffix.lower() in ('.jpg','.jpeg') else 'image/png',
               'data':base64.b64encode(Path(i['path']).read_bytes()).decode()}} for i in row['images']]
    folder = client.out/'artifacts'/f"{row['index']:04d}"
    schemas = {'direct':original.prediction_schema(row['task']), 'answer':original.prediction_schema(row['task']),
               'spatial':original.code_schema('spatial_code'), 'social':original.code_schema('social_code')}
    def invoke(stage):
        schema = schemas[stage]
        prompt = build_prompt(stage,context)
        raw = client.generate_parts(images+[{'text':prompt}],max_output_tokens=16384,response_schema=schema)
        payload = original._json_object(raw)
        validate_schema(payload,schema)
        write(folder/(stage+'.json'),{'payload':payload,'raw':raw})
        return payload
    result = {k:row[k] for k in ('index','id','task','recording_id','sample_id')}
    result.update(model=client.config.model, conditions={})
    try:
        result['conditions']['direct'] = {'prediction':invoke('direct')}
    except fatal:
        raise
    except Exception as exc:
        result['conditions']['direct'] = {'prediction':None,'error_type':type(exc).__name__,'error_message':str(exc)[:500]}
    write(folder/'direct_result.json',result['conditions']['direct'])
    try:
        spatial = invoke('spatial')
        context['spatial_code'] = original.normalize_code(spatial.get('spatial_code'),'S')
        context['spatial_trace'] = execute_spatial(context['spatial_code'])
        write(folder/'spatial_executed.json',{k:context[k] for k in ('spatial_code','spatial_trace')})
        social = invoke('social')
        context['social_code'] = original.normalize_code(social.get('social_code'),'M')
        context['social_trace'] = execute_social(context['social_code'])
        prediction = invoke('answer')
        result['conditions']['ssc'] = {'prediction':prediction, **{k:v for k,v in context.items() if k!='evidence'}}
    except fatal:
        raise
    except Exception as exc:
        result['conditions']['ssc'] = {'prediction':None,'error_type':type(exc).__name__,'error_message':str(exc)[:500],
            **{k:v for k,v in context.items() if k!='evidence'}}
    return result
