import json,sys,tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run
from budget_transport import Client,OutputTruncated,ModelIdentityError
with tempfile.TemporaryDirectory() as td:
 root=Path(td);key=root/'key';key.write_text('test-credential-not-real')
 cfg={'key_file':str(key),'model':'gemini-3-flash-preview','uniform_output_cap':16384,'shared_gate':None,'endpoint':'https://example.invalid'}
 client=Client(root,cfg,lambda *a:nullcontext())
 class Response:
  status_code=200
  def __init__(self,finish='STOP',model='gemini-3-flash-preview'):self.finish,self.model=finish,model
  def json(self):return {'modelVersion':self.model,'candidates':[{'finishReason':self.finish,'content':{'parts':[{'text':'S.fact(0, "r", "s", "o")'}]}}]}
 for i,cap in enumerate([256,1024,2048,4096]):
  client.begin(i)
  with patch('requests.post',return_value=Response()) as post:
   assert client.generate_parts([{'text':'unchanged'}],cap)
   assert post.call_args.kwargs['json']['generationConfig']=={'temperature':0.0,'maxOutputTokens':16384}
   assert post.call_args.kwargs['json']['contents'][0]['parts']==[{'text':'unchanged'}]
   req=json.loads((root/'calls'/f'{i:04d}'/'01/request.json').read_text());assert req['legacy_requested_cap']==cap and req['prompt_or_generation_settings_changed']
 for i,response,error in [(10,Response('MAX_TOKENS'),OutputTruncated),(11,Response(model='other-model'),ModelIdentityError)]:
  client.begin(i)
  with patch('requests.post',return_value=response) as post:
   try:client.generate_parts([{'text':'unchanged'}])
   except error:pass
   else:raise AssertionError('Expected fail-closed error')
   assert post.call_count==1
   assert not (root/'calls'/f'{i:04d}'/'01/response.json').exists()
 print('PASS: 4 original caps uniformly overridden; prompts unchanged; truncation and model drift fail closed; no API used')
