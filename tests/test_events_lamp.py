from dataclasses import replace
from fractions import Fraction as F
import unittest

from cheating_racer_detect.events.lamp import LampConfig, LampEngine, LampROI
from cheating_racer_detect.tracking import Box
from tests.test_events_lane import frame, vehicle


def config():
    return LampConfig('synthetic-v1', 50., 180., 4, F(3, 5), F(3, 10), F(1, 10), F(1), 1)


def rois():
    return LampROI(Box(20, 12, 22, 14), 'usable'), LampROI(Box(26, 12, 28, 14), 'usable')


def pixels(f, left=20, right=20):
    data = bytearray(f.width*f.height*3)
    for roi, value in zip(rois(), (left, right)):
        for y in range(int(roi.box.y1), int(roi.box.y2)):
            for x in range(int(roi.box.x1), int(roi.box.x2)):
                data[(y*f.width+x)*3:(y*f.width+x+1)*3] = bytes([value])*3
    return bytes(data)


class Lamps(unittest.TestCase):
    def sequence(self, values, right=None, times=None, changes=None):
        engine = LampEngine(config())
        for n, value in enumerate(values):
            f = frame(n, None if times is None else times[n])
            a, b = rois()
            if changes and n in changes:
                a = changes[n]
            engine.update(f, vehicle(f), pixels(f, value, 20 if right is None else right[n]), a, b)
        return engine

    def test_full_blink_exact_cycle_and_no_blink(self):
        values = [20, 220, 220, 20, 20, 220, 220, 20, 20]
        result = self.sequence(values).report(F(0), F(4, 5))
        self.assertEqual(result['signal_state'], 'left_blink')
        self.assertEqual(result['left']['complete_cycles'], [['1/10', '1/2']])
        self.assertEqual(result['right']['state'], 'no_blink_observed')
        self.assertEqual(result['samples'][1]['left']['brightness'], '220')
        self.assertEqual(self.sequence([20]*9).report(F(0), F(4, 5))['signal_state'], 'no_blink_observed')
        # Steady on does not mean blinking; do not label it "off".
        self.assertEqual(self.sequence([220]*9).report(F(0), F(4, 5))['left']['state'], 'no_blink_observed')

    def test_both_right_and_partial_unknown_preserve_facts(self):
        values = [20, 220, 220, 20, 20, 220, 220, 20, 20]
        self.assertEqual(self.sequence(values, values).report(F(0), F(4, 5))['signal_state'], 'both_blink')
        self.assertEqual(self.sequence([20]*9, values).report(F(0), F(4, 5))['signal_state'], 'right_blink')
        result = self.sequence(values, changes={8: LampROI(None, 'occluded')}).report(F(0), F(4, 5))
        self.assertEqual(result['signal_state'], 'unknown')
        self.assertTrue(result['left']['blink_evidence'])
        self.assertIn('occluded', result['left']['reasons'])

    def test_cycles_outside_requested_window_do_not_count(self):
        engine = self.sequence([20, 220, 20, 220, 20, 20, 20, 20, 20])
        result = engine.report(F(1, 5), F(4, 5))
        self.assertEqual(result['left']['state'], 'no_blink_observed')
        self.assertFalse(result['left']['blink_evidence'])

    def test_short_uncovered_empty_and_ambiguous_brightness(self):
        engine = self.sequence([20]*9)
        for start, end in ((F(0), F(1, 5)), (F(0), F(1)), (F(1), F(2))):
            self.assertEqual(engine.report(start, end)['signal_state'], 'unknown')
        self.assertEqual(LampEngine(config()).report(F(0), F(1))['signal_state'], 'unknown')
        result = self.sequence([100]*9).report(F(0), F(4, 5))
        self.assertEqual(result['signal_state'], 'unknown')
        self.assertIn('model_uncertain', result['left']['reasons'])

    def test_visibility_and_pixel_size_never_become_no_blink(self):
        for status in ('occluded', 'out_of_view', 'too_small', 'identity_uncertain', 'model_uncertain'):
            result = self.sequence([20]*9, changes={3: LampROI(None, status)}).report(F(0), F(4, 5))
            self.assertEqual(result['left']['state'], 'unknown')
        result = self.sequence([20]*9, changes={3: LampROI(Box(20, 12, 21, 13), 'usable')}).report(F(0), F(4, 5))
        self.assertIn('too_small', result['left']['reasons'])

    def test_gap_vfr_identity_geometry_and_clock(self):
        values = [20, 220, 220, 20, 20, 220, 20]
        engine = self.sequence(values, times=[0, 1, 2, 3, 4, 9, 10])
        self.assertEqual(engine.report(F(0), F(1))['signal_state'], 'unknown')
        engine = self.sequence([20]*9, times=[0, 1, 2, 4, 5, 6, 7, 9, 10])
        self.assertEqual(engine.report(F(0), F(1))['signal_state'], 'no_blink_observed')
        for change in ({'source_id': 'other'}, {'width': 100}):
            f = frame(9, **change)
            engine.update(f, vehicle(f), pixels(f), *rois())
            self.assertEqual(engine.report(F(0), F(1))['signal_state'], 'unknown')
        engine = self.sequence([20]*9)
        f = frame(9)
        engine.update(f, vehicle(f, key=(2, 2, 1)), pixels(f), *rois())
        self.assertEqual(engine.report(F(0), F(1))['signal_state'], 'unknown')
        bad = frame(10, time_base=F(1, 5))
        with self.assertRaises(ValueError):
            engine.update(bad, vehicle(bad), pixels(bad), *rois())
        self.assertEqual(engine.report(F(0), F(1))['samples'], [])

    def test_prediction_wrong_roi_pixels_and_error_reset_retry(self):
        engine = self.sequence([20]*9)
        f = frame(9)
        self.assertEqual(engine.update(f, None, pixels(f), *rois())['reason'], 'no_actual_observation')
        for bad_pixels, roi in ((bytearray(pixels(f)), rois()[0]), (b'', rois()[0]),
                                (pixels(f), LampROI(Box(0, 0, 2, 2), 'usable'))):
            with self.assertRaises(ValueError):
                engine.update(f, vehicle(f), bad_pixels, roi, rois()[1])
            self.assertEqual(engine.report(F(0), F(1))['samples'], [])
        f = frame(0)
        self.assertEqual(engine.update(f, vehicle(f), pixels(f), *rois())['sample_count'], 1)

    def test_policy_bounds_cycle_durations_hysteresis_and_sample_bound(self):
        for change in ({'off_threshold': 180}, {'on_threshold': float('nan')}, {'min_cycles': True},
                       {'min_pixels': 65537}, {'min_phase': F(2)}, {'min_window': .3}):
            with self.assertRaises(ValueError):
                replace(config(), **change)
        result = self.sequence([20, 220, 20, 220, 20, 220, 20, 20, 20]).report(F(0), F(4, 5))
        self.assertTrue(result['left']['blink_evidence'])
        # Threshold deadband inherits only a previously known phase.
        result = self.sequence([20, 100, 220, 100, 20, 100, 220, 100, 20]).report(F(0), F(4, 5))
        self.assertEqual(result['signal_state'], 'left_blink')
        engine = self.sequence([20]*125)
        self.assertEqual(len(engine.report(F(0), F(62, 5))['samples']), 120)
        self.assertEqual(engine.report(F(0), F(62, 5))['signal_state'], 'unknown')
        with self.assertRaises(ValueError):
            engine.report(F(1), F(1))


if __name__ == '__main__':
    unittest.main()
