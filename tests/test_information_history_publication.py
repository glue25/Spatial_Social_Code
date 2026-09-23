import gzip
import json
import unittest
from experiments.verify_information_history_publication import OUT,verify


class PublicationTests(unittest.TestCase):
    def test_entire_public_snapshot(self):
        result=json.loads(verify())
        self.assertEqual(result['formal'],640)
        self.assertEqual(result['formal_invalid'],11)

    def test_prose_contrast_not_claimed_significant(self):
        s=json.loads((OUT/'formal_summary.json').read_text())
        ci=s['paired_differences']['ssc_minus_prose']['ci']
        self.assertLess(ci[0],0);self.assertGreater(ci[1],0)

    def test_favorable_and_reverse_cases_present(self):
        pos=json.loads((OUT/'cases/formal_007/case.json').read_text())
        neg=json.loads((OUT/'cases/formal_019/case.json').read_text())
        self.assertTrue(pos['outputs']['undelivered']['ssc']['scores']['joint'])
        self.assertFalse(pos['outputs']['undelivered']['prose']['scores']['joint'])
        self.assertFalse(neg['outputs']['witness']['ssc']['scores']['joint'])
        self.assertTrue(neg['outputs']['witness']['prose']['scores']['joint'])

    def test_all_800_outputs_preserved(self):
        rows=[json.loads(x) for x in gzip.decompress((OUT/'evidence/all_calls.jsonl.gz').read_bytes()).decode().splitlines()]
        self.assertEqual(sum(x['path'].endswith('/result.json') for x in rows),800)


if __name__=='__main__':unittest.main()
