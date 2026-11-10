import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
from PIL import Image
import run
from frozen_coordinates import coordinate_correct_original

class Tests(unittest.TestCase):
    def test_identity_without_gold_or_direct(self):
        ssc={'prediction':{'category_l1':'food'},'spatial_trace':json.dumps({'kind':'spatial_fact','values':{'relation':'category','subject':'cabbage','object':'food','confidence':1}})}
        with patch.object(run.V2,'load_annotations',side_effect=AssertionError('label read')):
            self.assertEqual(run.identity(ssc,'joint_attention')[0],'cabbage')
        self.assertEqual(run.identity({'prediction':{'action_noun_l1':'cup'}},'scoia')[0],'cup')
        self.assertEqual(run.identity({'prediction':{}},'handover')[0],'task relevant object')

    def test_bad_boxes_rejected(self):
        raw={'results':[{'box':[0,0,2000,3],'score':1},{'box':[0,0,10,10],'score':float('nan')},{'box':[2,3,12,23],'score':.8}]}
        self.assertEqual(len(run.proposals(raw)),1)

    def test_coordinates_and_field_invariance(self):
        views={'leader':[{'box_xyxy':[100,200,300,400],'score':.8}], 'helper':[{'box_xyxy':[200,400,600,800],'score':.9}]}
        for task in ['joint_attention','scoia','handover']:
            base={'category_l1':'food','cue_type':['gaze'],'delivery_flow':'ltr','time_to_handover':2,'bbox':[1,2,3,4],'left_bbox':[1,2,3,4],'right_bbox':[1,2,3,4]}
            before=copy.deepcopy(base);out,role=run.augment(base,task,views)
            self.assertEqual(base,before)
            excluded=set(run.V2.BBOX_FIELDS[task])
            self.assertEqual({k:v for k,v in out.items() if k not in excluded},{k:v for k,v in base.items() if k not in excluded})
            if task=='joint_attention':self.assertEqual(out['right_bbox'],[804,200,200,200])
            if task=='scoia':self.assertEqual(out['bbox'],[200,400,400,400])
            if task=='handover':self.assertEqual((role,out['bbox']),('helper',[804,200,200,200]))
        original=coordinate_correct_original('joint_attention',{'left_bbox':[100,200,200,200],'right_bbox':[200,400,400,400]})
        self.assertEqual(original['right_bbox'],[804,200,200,200])

    def test_failure_retains_semantics_not_old_bbox(self):
        out,_=run.augment({'bbox':[1,2,3,4],'category_l1':'cup'},'handover',{'helper':[],'leader':[]})
        self.assertIsNone(out['bbox']);self.assertEqual(out['category_l1'],'cup')
        self.assertFalse(run.valid_ssc({'prediction':{},'error_type':'bad'}))

    def test_resume_transport_and_raw_response(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);im=folder/'image.png';Image.new('RGB',(1408,1408)).save(im)
            cfg={'sam_model':'SAM 3.1 Multiplex','sam_url':'http://localhost:8114','max_attempts_per_view':3,'request_timeout_seconds':1,'retry_delay_seconds':0,'min_interval_seconds':0}
            with patch.object(run.urllib.request,'urlopen',side_effect=urllib.error.URLError('down')) as request:
                for _ in range(2):
                    with self.assertRaises(RuntimeError):run.segment(im,'cup',folder/'failed',cfg,run.sha(im))
                self.assertEqual(request.call_count,3)
            done=folder/'done';done.mkdir();raw={'results':[{'box':[1,2,30,40],'score':.8}]}
            (done/'response.json.gz').write_bytes(gzip.compress(json.dumps(raw).encode()))
            with patch.object(run.urllib.request,'urlopen',side_effect=AssertionError('resampled')):
                self.assertEqual(run.segment(im,'cup',done,cfg,run.sha(im))[0]['score'],.8)

if __name__=='__main__':unittest.main()
