import unittest

from experiments.verify_cross_environment_casebook import OUT, read, verify, sam_replay


class CasebookTests(unittest.TestCase):
    def test_all_exports(self):
        self.assertEqual(sum(verify().values()),13)

    def test_negative_case_preserved(self):
        d=read(OUT/'mmtom/index_0006/case.json')
        self.assertTrue(d['result']['direct_correct']);self.assertFalse(d['result']['correct'])

    def test_dummy_label_not_gold(self):
        d=read(OUT/'ma_egoqa/index_1038/case.json')
        self.assertEqual(d['gold']['answer_index'],2)
        self.assertNotEqual(d['input']['row']['answer'],d['input']['row']['options'][2])

    def test_invalid_ablation_is_not_valid_error(self):
        d=read(OUT/'ma_egoqa/index_1597/case.json')
        self.assertFalse(d['results']['conditions']['ssc_no_references']['valid'])

    def test_sam_view_difference_preserved(self):
        d=read(OUT/'comind/index_0226/case.json')
        self.assertEqual(d['result']['conditions']['direct_sam']['selected_view'],'helper')
        self.assertEqual(d['result']['conditions']['ssc_sam']['selected_view'],'leader')
        sam_replay(d,'ssc_sam')

    def test_wrong_coordinate_conversion_rejected(self):
        d=read(OUT/'comind/index_0066/case.json')
        d['result']['conditions']['ssc_sam']['prediction']['right_bbox'][0]-=704
        with self.assertRaises(AssertionError):sam_replay(d,'ssc_sam')


if __name__=='__main__':unittest.main()
