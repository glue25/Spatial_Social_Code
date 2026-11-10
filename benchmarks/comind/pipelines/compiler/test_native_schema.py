"""Offline request, parser, safety and restart regression tests; no real API."""
import copy
import json
from contextlib import nullcontext
from pathlib import Path
import tempfile
from unittest.mock import patch
import run
from budget_transport import Client,native_body,OutputTruncated,ModelIdentityError,GeminiServiceError,PermanentServiceError
from comind_adapter import original,validate_schema,build_prompt,SUFFIX

def raises(error,fn):
    try:fn()
    except error:return
    raise AssertionError('Expected '+error.__name__)

schema=original.code_schema('spatial_code')
parts=[{'text':'Original stage prompt unchanged'},{'inline_data':{'mimeType':'image/jpeg','data':'Zml4dHVyZQ=='}}]
before=copy.deepcopy(parts)
legacy,body=native_body(parts,16384,'gemini-3-flash-preview',schema)
assert parts==before
assert body['generationConfig']=={'temperature':0.0,'maxOutputTokens':16384,'responseMimeType':'application/json','responseJsonSchema':schema}
assert body['contents'][0]['parts'][0]==parts[0]
assert body['contents'][0]['parts'][1]=={'inlineData':parts[1]['inline_data']}
context={'evidence':'FIXTURE','spatial_trace':'trace','social_trace':'social','spatial_code':'S.unknown(0,"r","s","o")','social_code':'M.unknown("a","x")'}
for stage in ['direct','spatial','social','answer']:
    assert SUFFIX not in build_prompt(stage,context)
validate_schema({'spatial_code':['S.unknown(0,"r","s","o")']},schema)
for invalid in [{'type':'object','spatial_code':[]},{'spatial_code':'not-an-array'},{'spatial_code':[]},{'spatial_code':[123]}]:
    raises(ValueError,lambda:validate_schema(invalid,schema))
answer=original.prediction_schema('joint_attention')
valid={'left_bbox':[1,2,3,4],'right_bbox':[5,6,7,8],'cue_type':['gaze'],'category_l1':'x'}
validate_schema(valid,answer)
for invalid in [dict(valid,left_bbox=[1,2]),dict(valid,left_bbox=[True,2,3,4]),dict(valid,cue_type=['gaze','gaze'])]:
    raises(ValueError,lambda:validate_schema(invalid,answer))

class Response:
    def __init__(self,status=200,finish='STOP',model='gemini-3-flash-preview'):
        self.status_code=status;self.finish=finish;self.model=model
    def json(self):return {'modelVersion':self.model,'candidates':[{'finishReason':self.finish,'content':{'parts':[{'text':'{"spatial_code":["S.unknown(0, \'r\', \'s\', \'o\')"]}' }]}}]}

with tempfile.TemporaryDirectory() as td:
    out=Path(td);key=out/'secret';key.write_text('synthetic-test-credential')
    cfg={'key_file':str(key),'model':'gemini-3-flash-preview','uniform_output_cap':16384,'transport_max_attempts':10,'shared_gate':None,'endpoint':'http://localhost'}
    client=Client(out,cfg,lambda *a:nullcontext());client.begin(0)
    with patch('requests.post',return_value=Response()) as post:
        text=client.generate_parts(parts,256,response_schema=schema)
        assert post.call_args.kwargs['json']==body
    request=json.loads((out/'calls/0000/01/request.json').read_text())
    assert request['generationConfig']['responseJsonSchema']==schema
    assert request['parts'][1]['sha256'] and 'data' not in request['parts'][1]
    client.begin(0)
    with patch('requests.post',side_effect=AssertionError('No requery')):
        assert client.generate_parts(parts,256,response_schema=schema)==text
    client.begin(0)
    raises(AssertionError,lambda:client.generate_parts(parts,256,response_schema=answer))
    for index,response,error in [(1,Response(finish='MAX_TOKENS'),OutputTruncated),(2,Response(model='wrong'),ModelIdentityError),(3,Response(status=400),PermanentServiceError),(4,Response(status=429),GeminiServiceError)]:
        client.begin(index)
        with patch('requests.post',return_value=response) as post,patch('budget_transport.time.sleep'):
            raises(error,lambda:client.generate_parts(parts,response_schema=schema))
            assert post.call_count==(10 if index==4 else 1)
    for index,error in [(1,OutputTruncated),(4,GeminiServiceError)]:
        client.begin(index)
        with patch('requests.post',side_effect=AssertionError('No retry after terminal failure')):
            raises(error,lambda:client.generate_parts(parts,response_schema=schema))
print('PASS: exact native schema/body, no schema text suffix, original schema validation, credential-safe media descriptors, schema-aware cache, truncation/model/permanent guards, bounded and persistent retries')
