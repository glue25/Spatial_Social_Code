import unittest

from experiments.build_habitat_update_case import OUT, KEY, read, sha, validate


class UpdateCaseTests(unittest.TestCase):
    def setUp(self):
        self.data = read(OUT / 'evidence_chain.json')

    def test_audited_chain(self):
        self.assertTrue(validate(self.data))

    def test_report_cannot_be_relabeled_as_observation(self):
        self.data['evidence']['update']['record']['after']['social'][KEY]['status'] = 'observed'
        with self.assertRaises(AssertionError):
            validate(self.data)

    def test_no_fabricated_physical_location_update(self):
        after = self.data['evidence']['update']['record']['after']
        after['spatial']['object.vase_4.on'] = {'value': 'cabinet_20'}
        with self.assertRaises(AssertionError):
            validate(self.data)

    def test_next_action_cannot_be_replaced_by_later_placement(self):
        self.data['evidence']['next_plan']['record']['action'] = 'Place[vase_4, on, table_22, None, None]'
        with self.assertRaises(AssertionError):
            validate(self.data)

    def test_no_exclusive_state_channel_claim(self):
        self.assertTrue(self.data['evidence']['actual_request']['message_also_in_native_observation'])
        msg = self.data['evidence']['nl_message']['record']['message']
        self.assertIn(msg, self.data['evidence']['nl_next_input']['record']['context'])

    def test_saved_frames_unchanged(self):
        for frame in self.data['frames']:
            self.assertEqual(frame['sha256'], sha(OUT / frame['file']))


if __name__ == '__main__':
    unittest.main()
