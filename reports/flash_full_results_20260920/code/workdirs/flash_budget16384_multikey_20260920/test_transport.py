import json,sys,tempfile
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch
import requests
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run
from budget_transport import Client,OutputTruncated,ModelIdentityError,GeminiServiceError
class Response:
 def __init__(self,status=200,finish='STOP',model='gemini-3-flash-preview',text='S.fact(0, "r", "s", "o")'):
  self.status_code,self.finish,self.model,self.text=status,finish,model,text
 def json(self):return {'modelVersion':self.model,'candidates':[{'finishReason':self.finish,'content':{'parts':[{'text':self.text}]}}]}
with tempfile.TemporaryDirectory() as td:
 root=Path(td);key=root/'key';key.write_text('fake-test-secret')
 cfg={'key_file':str(key),'model':'gemini-3-flash-preview','uniform_output_cap':16384,'shared_gate':None,'endpoint':'https://example.invalid','transport_max_attempts':10}
 c=Client(root,cfg,lambda *a:nullcontext())
 for i,cap in enumerate([256,1024,2048,4096]):
  c.begin(i)
  with patch('requests.post',return_value=Response()) as post:
   c.generate_parts([{'text':'unchanged'}],cap)
   body=post.call_args.kwargs['json'];assert body['generationConfig']=={'temperature':0.0,'maxOutputTokens':16384}
   assert body['contents'][0]['parts']==[{'text':'unchanged'}]
   assert post.call_args.kwargs['timeout']==600
 c.begin(10)
 with patch('requests.post',side_effect=[requests.ConnectionError('fake-test-secret connection reset')]*9+[Response()]) as post,patch('budget_transport.time.sleep') as sleep:
  assert c.generate_parts([{'text':'unchanged'}]);assert post.call_count==10
  assert [x.args[0] for x in sleep.call_args_list]==[5,10,20,40,80,120,120,120,120]
 assert 'fake-test-secret' not in '\n'.join(p.read_text() for p in (root/'calls/0010').rglob('*.json'))
 c.begin(11)
 with patch('requests.post',return_value=Response(status=429)) as post,patch('budget_transport.time.sleep') as sleep:
  try:c.generate_parts([{'text':'unchanged'}])
  except GeminiServiceError:pass
  else:raise AssertionError('Expected exhausted retry failure')
  assert post.call_count==10 and sleep.call_count==9
 for i,response,error in [(12,Response(finish='MAX_TOKENS'),OutputTruncated),(13,Response(model='other-model'),ModelIdentityError),(14,Response(status=400),GeminiServiceError)]:
  c.begin(i)
  with patch('requests.post',return_value=response) as post,patch('budget_transport.time.sleep') as sleep:
   try:c.generate_parts([{'text':'unchanged'}])
   except error:pass
   else:raise AssertionError('Expected fail-closed error')
   assert post.call_count==1 and not sleep.called
   assert not (root/'calls'/f'{i:04d}'/'01/response.json').exists()
 print('PASS: unchanged prompts/caps/timeouts; recovery on tenth try; ten-try429limit; exact backoff; credential-redacted exception details; no extra retries for400/truncation/model drift; no API calls')
