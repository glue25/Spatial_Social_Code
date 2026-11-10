import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.information_history.protocol import (
    CONDITIONS,HISTORIES,make_base,build,audit,transform,prose,unprose,prompt,score,canonical,oracle)
from benchmarks.information_history.metrics import per_base,summarize
from benchmarks.information_history.run import Transport,parse,MODEL,FIELDS,costs,write


def fixture():
    task={'task_id':'unit-test-only-not-formal','scene_id':'fixture','episode_id':'test',
          'problem_pddl':'(:objects\nagent_0 agent_1 agent_2 - agent\n'
              'cup_0 plate_1 - object\ntable_1 table_2 cabinet_3 - furniture\n)\n(:init)'}
    return make_base(task,'unit',0)


def response(prediction,finish='stop'):
    return {'model':MODEL,'choices':[{'finish_reason':finish,'message':{'content':json.dumps(prediction)}}],
            'usage':{'prompt_tokens':5,'completion_tokens':7,'total_tokens':12}}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.b=fixture();self.cases=[build(self.b,h) for h in HISTORIES]

    def test_all_invariants_and_gold(self):
        self.assertTrue(audit(self.b,self.cases))
        for c in self.cases:
            self.assertEqual(c['gold']['world_location'],self.b['new'])
            self.assertEqual(c['gold']['unrelated_location'],self.b['old'])
            self.assertEqual(c['gold']['target_location'],self.b['new'] if c['history'] in ('witness','delivered') else self.b['old'])

    def test_prose_lossless(self):
        for c in self.cases:self.assertEqual(unprose(prose(c['view'])),c['view'])

    def test_no_links_nonidentifiable(self):
        selected=[c for c in self.cases if c['history']!='undelivered']
        self.assertEqual(len({prompt(c,'no_links') for c in selected}),1)
        self.assertEqual(len({c['gold']['target_location'] for c in selected}),2)

    def test_no_attribution_does_not_leave_names(self):
        text=canonical(transform(self.cases[0]['view'],'no_attribution'))
        for a in self.b['agents']:self.assertNotIn(a,text)

    def test_failed_delivery_never_enters_private_store(self):
        c=self.cases[-1]
        failed=[r for r in c['oracle_access_log'] if not r['available']]
        self.assertEqual(len(failed),1)
        self.assertNotIn(failed[0]['id'],c['state_store_snapshots'][self.b['target']]['social'])
        bad=copy.deepcopy(c['oracle_access_log'])
        for r in bad:
            if not r['available']:r['available']=True;r['turn']=4
        modified=oracle(c['view']['physical_events'],bad,c['query'])
        self.assertEqual(modified['target_location'],self.b['new'])

    def test_reordering_does_not_change_oracle(self):
        c=self.cases[0]
        self.assertEqual(oracle(list(reversed(c['view']['physical_events'])),
            list(reversed(c['oracle_access_log'])),c['query']),c['gold'])

    def test_wrong_evidence_is_not_supported(self):
        c=self.cases[0];p=copy.deepcopy(c['gold']);p['target_evidence']='unknown'
        result=score(p,c)
        self.assertTrue(result['joint']);self.assertFalse(result['supported_target'])

    def test_foreign_evidence_does_not_zero_state_scores(self):
        c=self.cases[0];p=copy.deepcopy(c['gold']);p['target_evidence']='wrong-or-social-record-id'
        parsed=parse(response(p),c)
        self.assertTrue(score(parsed,c)['joint'])
        self.assertFalse(score(parsed,c)['supported_target'])

    def test_wrong_location_is_field_error_not_schema_failure(self):
        c=self.cases[0];p=copy.deepcopy(c['gold']);p['target_location']='nonexistent_location'
        parsed=parse(response(p),c)
        self.assertFalse(score(parsed,c)['target_location'])
        self.assertTrue(score(parsed,c)['world_location'])

    def test_missing_and_invalid_stay_in_denominator(self):
        rows=[]
        for c in self.cases:
            rows.append({'base_id':self.b['base_id'],'history':c['history'],'condition':'ssc',
                         'prediction':c['gold'],'valid':True,'scores':score(c['gold'],c)})
        self.assertEqual(per_base(rows)['ssc']['paired_joint_correct'],1)
        self.assertEqual(per_base(rows[:-1])['ssc']['joint_accuracy'],.75)
        self.assertEqual(per_base(rows)['prose']['joint_accuracy'],0)
        summary=summarize([self.b],rows,bootstrap=20)
        self.assertEqual(summary['planned_final_calls'],16)
        self.assertEqual(summary['pending_final_calls'],12)

    def test_unknown_is_valid_not_success(self):
        c=self.cases[0];p={k:'unknown' for k in FIELDS}
        self.assertEqual(parse(response(p),c),p)
        self.assertFalse(score(p,c)['joint'])

    def test_foreign_model_and_truncation_rejected(self):
        p=self.cases[0]['gold']
        with self.assertRaises(AssertionError):parse(response(p,'length'),self.cases[0])
        r=response(p);r['model']='some-other-model'
        with self.assertRaises(AssertionError):parse(r,self.cases[0])

    def test_no_gold_or_history_labels_in_final_request(self):
        for c in self.cases:
            for cond in CONDITIONS:
                text=prompt(c,cond)
                self.assertNotIn('"gold"',text)
                self.assertNotIn('"history"',text)
                self.assertNotIn('unit_000',text)


class TransportTests(unittest.TestCase):
    def test_http200_invalid_cached_no_resampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);key=root/'key';key.write_text('fake-unit-test-credential')
            cfg={'credentials':[{'alias':'test','file':str(key),'gate':str(root/'gate')}],
                 'temperature':0,'max_output_tokens':2048,'reasoning_effort':'low',
                 'transport_attempt_limit':3,'minimum_key_interval':0,
                 'request_timeout_seconds':1,'retry_backoff_seconds':[0]}
            client=Transport(root,cfg)
            class Resp:
                status_code=200
                def json(self):return response({k:'unknown' for k in FIELDS},'length')
            with patch('requests.post',return_value=Resp()) as post:
                a,_=client.request(root/'call','same prompt')
                b,_=client.request(root/'call','same prompt')
                self.assertEqual(a,b);self.assertEqual(post.call_count,1)
                with self.assertRaises(AssertionError):parse(a)
                with self.assertRaises(AssertionError):client.request(root/'call','changed prompt')

    def test_costs_include_failed_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);folder=out/'calls/formal/formal_000/witness/ssc'
            write(folder/'attempt_000.json',{'http_status':429,'elapsed_seconds':2})
            write(folder/'attempt_001.json',{'http_status':200,'elapsed_seconds':3,
                'response':response({k:'unknown' for k in FIELDS})})
            c=costs(out)['formal/ssc']
            self.assertEqual(c['actual_request_attempts'],2)
            self.assertEqual(c['input_tokens'],5);self.assertEqual(c['output_tokens'],7)
            self.assertEqual(c['usage_missing_attempts'],1);self.assertEqual(c['elapsed_seconds'],5)


if __name__=='__main__':unittest.main()
