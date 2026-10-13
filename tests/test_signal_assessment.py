from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction as F
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.events.assessment import assess, from_report
from scripts.evaluate_events import main
from tests.test_events_evaluate import truth, prediction


def definition():
    gt, pred = asdict(truth()), asdict(prediction())
    gt.update(earliest='1', latest='1', signal_state='right_blink')
    pred.update(time='1', signal_state='right_blink')
    return {'schema': 'experimental-assessment-inputs-v1', 'truths': [gt], 'predictions': [pred],
            'correspondence': [{'source_id': 'source', 'segment_id': 'segment',
                                'prediction_vehicle_id': 'track', 'truth_vehicle_id': 'gt-car'}],
            'tolerance': '1/2', 'max_gt_width': '1'}


class Assessment(unittest.TestCase):
    def test_unknown_truth_dangerous_no_blink_and_missed_denominator(self):
        result = assess([truth('a'), truth('b', F(3))], [prediction()],
                        {('source', 'segment', 'track'): 'gt-car'},
                        truth_signals={'a': 'unknown', 'b': 'right_blink'}, prediction_signals={'pred': 'no_blink_observed'},
                        tolerance=F(1, 2), max_gt_width=F(1))
        self.assertEqual((result['events']['tp'], result['events']['fn']), (1, 1))
        self.assertEqual(result['end_to_end_known_prediction_coverage'], '1/2')
        self.assertEqual(result['truth_observability_ratio'], '1/2')
        self.assertEqual(result['unknown_truth_to_no_blink'], 1)
        self.assertEqual(result['end_to_end_correct_signal_coverage'], '0')
        self.assertEqual(result['observable_truth_known_prediction_coverage'], '0')
        self.assertEqual(result['observable_truth_correct_signal_coverage'], '0')

    def test_abstained_unknown_duplicate_and_empty(self):
        value = definition()
        value['predictions'][0]['abstained'] = True
        result = from_report(value)
        self.assertEqual(result['events']['fn'], 1)
        self.assertEqual(result['end_to_end_known_prediction_coverage'], '0')
        value['predictions'][0]['abstained'] = False
        value['predictions'][0]['signal_state'] = 'unknown'
        self.assertEqual(from_report(value)['end_to_end_known_prediction_coverage'], '0')
        value['predictions'].append({**value['predictions'][0], 'event_id': 'second'})
        self.assertEqual(from_report(value)['events']['fp'], 1)
        value['truths'] = value['predictions'] = []
        result = from_report(value)
        self.assertIsNone(result['end_to_end_known_prediction_coverage'])
        self.assertIsNone(result['truth_observability_ratio'])
        self.assertIsNone(result['observable_truth_known_prediction_coverage'])

    def test_strict_fields_signals_correspondence_bounds(self):
        original = definition()
        changes = [lambda v: v.update(extra=True), lambda v: v['truths'][0].update(signal_state='off'),
                   lambda v: v.update(truths=v['truths']*65), lambda v: v['predictions'][0].update(time='1.0'),
                   lambda v: v['correspondence'].append(deepcopy(v['correspondence'][0])),
                   lambda v: v['truths'].append(deepcopy(v['truths'][0]))]
        for change in changes:
            value = deepcopy(original)
            change(value)
            with self.assertRaises(ValueError):
                from_report(value)

    def test_success_owned_result_does_not_mutate_labels(self):
        value = definition()
        before = deepcopy(value)
        result = from_report(value)
        self.assertEqual(result['end_to_end_correct_signal_coverage'], '1')
        result['signal_confusion_on_matches']['right_blink']['right_blink'] = 999
        self.assertEqual(from_report(value)['signal_confusion_on_matches']['right_blink']['right_blink'], 1)
        self.assertEqual(value, before)

    def test_cli_success_malformed_duplicate_nonfinite_size_and_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'labels.json'
            for data, expected in ((json.dumps(definition()).encode(), 0), (b'{"x":1,"x":2}', 1),
                                   (b'{"x":NaN}', 1), (b'{"x":1e999}', 1), (b' '*1_048_577, 1)):
                path.write_bytes(data)
                with patch('builtins.print'):
                    self.assertEqual(main(['--labels', str(path)]), expected)
            with patch('scripts.evaluate_events.from_report', side_effect=KeyboardInterrupt), patch('builtins.print'):
                path.write_text(json.dumps(definition()))
                self.assertEqual(main(['--labels', str(path)]), 130)

    def test_actual_demo_and_labels_cli_match_illustrative_coverage(self):
        from scripts.demo_assessment import definition as example
        result = from_report(example())
        self.assertEqual((result['events']['tp'], result['events']['fn']), (3, 1))
        self.assertEqual(result['end_to_end_known_prediction_coverage'], '1/2')
        self.assertEqual(result['end_to_end_correct_signal_coverage'], '1/4')
        self.assertEqual(result['unknown_truth_to_no_blink'], 1)
        demo = subprocess.run([sys.executable, '-m', 'scripts.demo_assessment'], capture_output=True, text=True, timeout=10)
        self.assertEqual(demo.returncode, 0, demo.stderr)
        self.assertEqual(json.loads(demo.stdout), result)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'labels.json'
            path.write_text(json.dumps(example()))
            actual = subprocess.run([sys.executable, '-m', 'scripts.evaluate_events', '--labels', str(path)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(actual.returncode, 0, actual.stderr)
            self.assertEqual(json.loads(actual.stdout), result)
            self.assertNotIn(str(path), actual.stdout)


if __name__ == '__main__':
    unittest.main()
