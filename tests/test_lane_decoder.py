"""Own numeric lane fixtures. No trained weights, datasets or upstream imports."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import importlib.util
import json
import math
import subprocess
import sys
import unittest
from unittest.mock import patch

from cheating_racer_detect.analysis import LaneBatch, LaneSlot, UFLDOutput, decode_ufld, to_coded_point
from tests.test_tracking import frame

AVAILABLE = importlib.util.find_spec('numpy') is not None
SHAPES = {'loc_row': (1, 200, 72, 4), 'loc_col': (1, 100, 81, 4),
          'exist_row': (1, 2, 72, 4), 'exist_col': (1, 2, 81, 4)}


def logits():
    import numpy as np
    values = {name: np.full(shape, -100., dtype=np.float32) for name, shape in SHAPES.items()}
    values['loc_row'][:, 40, :, :] = 100.
    values['loc_col'][:, 30, :, :] = 100.
    values['exist_row'][:, 1, :, :] = 100.
    values['exist_col'][:, 1, :, :] = 100.
    return values


def output(current=None, values=None, rotation=0):
    return UFLDOutput(current or frame(), 'scripted-lanes-v1', rotation, logits() if values is None else values)


def empty_slots():
    return tuple(LaneSlot(slot, axis, (False,)*count, (None,)*count)
                 for slot, axis, count in ((1, 'row', 72), (2, 'row', 72), (0, 'column', 81), (3, 'column', 81)))


class Contracts(unittest.TestCase):
    def test_empty_batch_retains_frame_clock_slots_and_report(self):
        source = frame(3, pts=509, display_rotation=270)
        batch = LaneBatch(source, 'fixture:v1', 0, empty_slots())
        self.assertIs(batch.frame, source)
        report = batch.report()
        self.assertEqual(report['frame']['normalized_time'], '9/10')
        self.assertEqual(report['applied_rotation_cw'], 0)
        self.assertEqual([v['slot'] for v in report['slots']], [1, 2, 0, 3])
        self.assertEqual(json.loads(json.dumps(report)), report)
        report['slots'][0]['exists'][0] = True
        self.assertFalse(batch.slots[0].exists[0])
        with self.assertRaises(FrozenInstanceError):
            batch.model_id = 'changed'

    def test_literal_rotation_inverses_use_extents_not_pixel_indices(self):
        source = frame(width=100, height=60)
        for rotation, point in ((0, (20., 30.)), (90, (30., 20.)),
                                (180, (80., 30.)), (270, (30., 80.))):
            self.assertEqual(to_coded_point(source, rotation, point), (20., 30.))
        self.assertEqual(to_coded_point(source, 90, (0., 0.)), (0., 60.))
        self.assertEqual(to_coded_point(source, 270, (60., 100.)), (0., 60.))

    def test_explicit_rotation_not_display_metadata(self):
        for metadata in (0, 90, 180, 270):
            source = frame(width=100, height=60, display_rotation=metadata)
            self.assertEqual(to_coded_point(source, 0, (20., 30.)), (20., 30.))
            self.assertEqual(to_coded_point(source, 90, (30., 20.)), (20., 30.))

    def test_invalid_points_and_rotation(self):
        for point in (None, [1, 2], (1,), (float('nan'), 2), (1, float('inf')), (True, 1), (-1, 0), (97, 0)):
            with self.subTest(point=point), self.assertRaises(ValueError):
                to_coded_point(frame(), 0, point)
        for rotation in (True, 1, -90, 360, 90., None):
            with self.assertRaises(ValueError):
                to_coded_point(frame(), rotation, (1, 2))

    def test_invalid_frame_and_metadata(self):
        for source in (None, frame(width=1025)):
            with self.assertRaises(ValueError):
                UFLDOutput(source, 'fixture', 0, {k: None for k in SHAPES})
        for model_id in ('../model', '', 'x'*161):
            with self.assertRaises(ValueError):
                UFLDOutput(frame(), model_id, 0, {k: None for k in SHAPES})
        for value in ({}, [], {**{k: None for k in SHAPES}, 'extra': None}):
            with self.assertRaises(ValueError):
                UFLDOutput(frame(), 'fixture', 0, value)

    def test_slot_invariants_and_no_point_without_support(self):
        base = empty_slots()[0]
        for changes in ({'slot': True}, {'slot': 4}, {'axis': 'column'}, {'exists': (False,)*71},
                        {'exists': (0,)*72}, {'points': [None]*72}, {'points': ((1, 2),)+(None,)*71}):
            with self.assertRaises(ValueError):
                replace(base, **changes)
        valid = replace(base, exists=(True,)*72, points=((1, 2),)*72)
        self.assertTrue(valid.support_passed)
        for point in ((float('nan'), 2), [1, 2], (1,)):
            with self.assertRaises(ValueError):
                replace(valid, points=(point,)*72)

    def test_batch_rejects_missing_reordered_and_outside_points(self):
        batch = LaneBatch(frame(), 'fixture', 0, empty_slots())
        for slots in (batch.slots[:3], batch.slots[::-1], list(batch.slots), (None,)*4):
            with self.assertRaises(ValueError):
                replace(batch, slots=slots)
        outside = replace(batch.slots[0], exists=(True,)*72, points=((97., 3.),)*72)
        with self.assertRaises(ValueError):
            replace(batch, slots=(outside, *batch.slots[1:]))

    def test_same_frame_rejected_before_optional_import(self):
        original = UFLDOutput(frame(), 'fixture', 0, {k: None for k in SHAPES})
        for current in (frame(1), frame(pts=501), frame(time_base=Fraction(1, 20)),
                        frame(first_pts=499), frame(width=97), frame(display_rotation=90), frame(source_id='other')):
            with self.assertRaisesRegex(ValueError, 'same source Frame'):
                decode_ufld(current, original)

    def test_stdlib_import_report_and_invalid_decode_need_no_numpy_torch_cv(self):
        code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'torch', 'torchvision', 'cv2', 'nvidia'):
            raise AssertionError('Forbidden dependency')
sys.meta_path.insert(0, Block())
from cheating_racer_detect.analysis import LaneBatch, LaneSlot, decode_ufld
from cheating_racer_detect.tracking import Frame
from fractions import Fraction
f = Frame('fixture', 0, 0, 500, Fraction(1,10), 500, 96, 64)
slots = tuple(LaneSlot(s,a,(False,)*n,(None,)*n) for s,a,n in
              ((1,'row',72),(2,'row',72),(0,'column',81),(3,'column',81)))
assert LaneBatch(f,'fixture',0,slots).report()['frame']['pts'] == 500
try: decode_ufld(f, None)
except ValueError: pass
else: raise AssertionError('Expected error')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))


@unittest.skipUnless(AVAILABLE, 'NOT_RUN: optional NumPy required')
class Decode(unittest.TestCase):
    def test_fixed_numeric_anchor_geometry_no_crop_inverse(self):
        source = frame(width=1000, height=600, display_rotation=90)
        result = decode_ufld(source, output(source))
        self.assertIs(result.frame, source)
        self.assertAlmostEqual(result.slots[0].points[0][0], 40.5/199*1000)
        self.assertAlmostEqual(result.slots[0].points[0][1], .42*600)
        self.assertEqual(result.slots[0].points[-1][1], 600.)
        self.assertAlmostEqual(result.slots[2].points[0][1], 30.5/99*600)
        self.assertEqual(result.slots[2].points[-1][0], 1000.)

    def test_all_four_masks_and_missing_slot_retained(self):
        values = logits()
        values['exist_row'][0, 0, :, 1] = 200.
        values['exist_row'][0, 0, 3, 2] = 200.
        result = decode_ufld(frame(), output(values=values))
        self.assertEqual([slot.slot for slot in result.slots], [1, 2, 0, 3])
        self.assertFalse(result.slots[0].support_passed)
        self.assertTrue(all(p is None for p in result.slots[0].points))
        self.assertFalse(result.slots[1].exists[3])
        self.assertIsNone(result.slots[1].points[3])
        self.assertIsNotNone(result.slots[1].points[4])

    def test_strict_anchor_count_gate_and_ties(self):
        for count, key, slot, total in ((36, 'exist_row', 1, 72), (20, 'exist_col', 0, 81)):
            values = logits()
            values[key][:, 0, :, slot] = 200.
            values[key][:, 0, :count, slot] = -100.
            decoded = decode_ufld(frame(), output(values=values))
            selected = next(s for s in decoded.slots if s.slot == slot)
            self.assertFalse(selected.support_passed)
            values[key][:, 0, count, slot] = -100.
            selected = next(s for s in decode_ufld(frame(), output(values=values)).slots if s.slot == slot)
            self.assertTrue(selected.support_passed)
        values = logits()
        values['exist_row'].fill(0)
        self.assertFalse(decode_ufld(frame(), output(values=values)).slots[0].support_passed)

    def test_local_window_against_scalar_softmax_oracle(self):
        values = logits()
        for index, value in ((39, 1.), (40, 2.), (41, 0.)):
            values['loc_row'][0, index, :, 1] = value
        expected = (sum(i*math.exp(v) for i, v in ((39,1.),(40,2.),(41,0.))) /
                    sum(math.exp(v) for v in (1.,2.,0.))+.5)/199*96
        self.assertAlmostEqual(decode_ufld(frame(), output(values=values)).slots[0].points[9][0], expected)

    def test_grid_edge_window_and_outside_not_clamped(self):
        values = logits()
        values['loc_row'][:, 40, :, :] = -100.
        values['loc_row'][:, 0, :, :] = 100.
        self.assertAlmostEqual(decode_ufld(frame(), output(values=values)).slots[0].points[0][0], .5/199*96)
        values['loc_row'][:, 0, :, :] = -100.
        values['loc_row'][:, 199, :, :] = 100.
        result = decode_ufld(frame(), output(values=values)).slots[0]
        self.assertTrue(all(result.exists))
        self.assertTrue(all(point is None for point in result.points))

    def test_both_axes_and_all_explicit_rotations(self):
        for rotation in (0, 90, 180, 270):
            source = frame(display_rotation=270)
            result = decode_ufld(source, output(source, rotation=rotation))
            width, height = (64, 96) if rotation in (90, 270) else (96, 64)
            row_view = (40.5/199*width, .42*height)
            column_view = (0., 30.5/99*height)
            self.assertEqual(result.slots[0].points[0], to_coded_point(source, rotation, row_view))
            self.assertEqual(result.slots[2].points[0], to_coded_point(source, rotation, column_view))

    def test_rotation_on_independent_actual_pixel_pattern(self):
        import numpy as np
        image = np.arange(60*100).reshape(60, 100)
        source = frame(width=100, height=60)
        # Pixel centers in view map to continuous coded centers, then floor index.
        for rotation in (0, 90, 180, 270):
            view = np.rot90(image, k=-rotation//90)
            for v, u in ((0, 0), (3, 5), (view.shape[0]-1, view.shape[1]-1)):
                x, y = to_coded_point(source, rotation, (u+.5, v+.5))
                self.assertEqual(image[int(y), int(x)], view[v, u])

    def test_invalid_array_shape_dtype_and_nonfinite(self):
        import numpy as np
        for bad in (np.zeros((2, 200, 72, 4)), np.zeros((1, 200, 71, 4)),
                    np.zeros((1, 200, 72, 4), dtype=np.int32),
                    np.zeros((1, 200, 72, 4), dtype=np.float16),
                    np.full((1, 200, 72, 4), float('nan')), np.full((1, 200, 72, 4), float('inf')),
                    [[1]], object()):
            values = logits()
            values['loc_row'] = bad
            with self.assertRaises(ValueError):
                decode_ufld(frame(), output(values=values))

    def test_extreme_finite_logits_stable_without_global_error_changes(self):
        import numpy as np
        values = {k: v.astype(np.float64) for k, v in logits().items()}
        values['loc_row'].fill(-1e308)
        values['loc_row'][:, 40, :, :] = 1e308
        before = np.geterr()
        result = decode_ufld(frame(), output(values=values))
        self.assertAlmostEqual(result.slots[0].points[0][0], 40.5/199*96)
        self.assertEqual(np.geterr(), before)

    def test_input_and_output_immutability_and_fresh_report(self):
        import numpy as np
        values = logits()
        before = {k: v.copy() for k, v in values.items()}
        result = decode_ufld(frame(), output(values=values))
        for k, v in values.items():
            np.testing.assert_array_equal(v, before[k])
            v.fill(0)
        self.assertTrue(result.slots[0].support_passed)
        report = result.report()
        report['slots'][0]['points'][0][0] = 999
        self.assertNotEqual(result.slots[0].points[0][0], 999)

    def test_fail_cancel_then_retry_is_stateless(self):
        import numpy as np
        source = frame()
        item = output(source)
        expected = decode_ufld(source, item)
        with patch.object(np, 'arange', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                decode_ufld(source, item)
        with patch.object(np, 'arange', side_effect=RuntimeError('fixture failure')):
            with self.assertRaises(RuntimeError):
                decode_ufld(source, item)
        self.assertEqual(decode_ufld(source, item), expected)

    def test_exact_vfr_offset_time_is_never_derived_from_fps(self):
        for index, pts in enumerate((150000, 150030, 150099, 150180)):
            source = frame(index, pts=pts, time_base=Fraction(1, 30000), first_pts=150000)
            batch = decode_ufld(source, output(source))
            self.assertIs(batch.frame, source)
            self.assertEqual(batch.frame.time, Fraction(pts-150000, 30000))
            self.assertEqual(batch.report()['frame']['pts'], pts)


if __name__ == '__main__':
    unittest.main()
