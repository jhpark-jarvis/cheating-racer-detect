"""Native detector bridge -> current actual replay inputs; no model accuracy."""

from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.analysis import DetectorBatch, DetectorObservation, DetectorTrackerBridge, Letterbox
from cheating_racer_detect.events.capture import capture_inputs
from cheating_racer_detect.events.inputs import load_inputs
from cheating_racer_detect.service import _sha256
from cheating_racer_detect.tools import discover_tools
from cheating_racer_detect.tracking import Box, Tracker
from scripts.demo_events import event_configs, generate
from tests.test_tracking import policy

try:
    TOOLS = discover_tools()
except Exception:
    TOOLS = None
AVAILABLE = all(importlib.util.find_spec(n) for n in ('numpy', 'cv2'))


class Detector:
    model_id = 'scripted-capture-v1'

    def detect(self, frame, image):
        if image.flags.writeable:
            raise AssertionError('Pixels not readonly')
        boxes = () if frame.ordinal == 3 else (DetectorObservation(0, 2, Box(10, 10, 30, 30), .9, self.model_id),)
        return DetectorBatch(frame, boxes, Letterbox(frame.width, frame.height), self.model_id)


@unittest.skipUnless(AVAILABLE and TOOLS, 'NOT_RUN: FFmpeg/NumPy/OpenCV required')
class Capture(unittest.TestCase):
    def run_capture(self, source, output, detector=None, **changes):
        model = detector or Detector()
        bridge = DetectorTrackerBridge(model, Tracker(policy()), model_alias='capture', expected_detector_id=model.model_id)
        lane, lamp = event_configs()
        options = dict(selected_identity=(1, 2, 1), lane_config=lane, lamp_config=lamp, pre=F(1, 5), post=F(2, 5), toolchain=TOOLS)
        options.update(changes)
        return bridge, lambda: capture_inputs(source, '0', '2.4', output, bridge, **options)

    def test_actual_source_capture_missing_prediction_not_observation_unknown_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = generate(TOOLS, root)
            before = _sha256(source)
            _bridge, run = self.run_capture(source, root/'saved')
            result = run()
            self.assertEqual(result.to_bytes(), load_inputs(root/'saved/inputs.json').to_bytes())
            self.assertEqual(len(result.rows), 24)
            self.assertIsNone(result.rows[3].vehicle)
            self.assertTrue(all(r.vehicle is not None for i, r in enumerate(result.rows) if i != 3))
            self.assertTrue(all(r.lanes is None and r.left.status == 'model_uncertain' for r in result.rows))
            self.assertEqual(_sha256(source), before)
            receipt = json.loads((root/'saved/capture.json').read_text())
            self.assertEqual(receipt['inputs_sha256'], result.sha256)
            self.assertEqual(receipt['model']['external_identifier'], Detector.model_id)
            self.assertEqual(receipt['actual_selected_observations'], 23)

    def test_explicit_nonexistent_identity_never_remaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = generate(TOOLS, root)
            _bridge, run = self.run_capture(source, root/'saved', selected_identity=(1, 2, 999))
            self.assertTrue(all(r.vehicle is None for r in run().rows))

    def test_failure_cancel_no_partial_tracker_reset_and_fresh_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = generate(TOOLS, root)
            for error in (RuntimeError('injected'), KeyboardInterrupt()):
                bridge, run = self.run_capture(source, root/'saved')
                with patch.object(bridge.detector, 'detect', side_effect=error), self.assertRaises(type(error)):
                    run()
                self.assertIsNone(bridge.tracker._previous)
                self.assertFalse((root/'saved').exists())
            _bridge, run = self.run_capture(source, root/'saved')
            self.assertEqual(len(run().rows), 24)
            before = (root/'saved/inputs.json').read_bytes()
            with self.assertRaises(Exception):
                run()
            self.assertEqual((root/'saved/inputs.json').read_bytes(), before)
            self.assertEqual(list(root.glob('*.partial')), [])

    def test_bad_config_gate_and_late_write_reset_preserve_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = generate(TOOLS, root)
            for options in ({'pre': .1}, {'lane_config': None}, {'gates': lambda *args: (None, None, None)}):
                bridge, run = self.run_capture(source, root/'saved', **options)
                with self.assertRaises(ValueError):
                    run()
                self.assertIsNone(bridge.tracker._previous)
            bridge, run = self.run_capture(source, root/'saved')
            with patch('cheating_racer_detect.events.capture.write_bytes', side_effect=OSError('disk')), self.assertRaises(Exception):
                run()
            self.assertFalse((root/'saved').exists())
            self.assertEqual(list(root.glob('*.partial')), [])

    def test_second_receipt_failure_and_pixel_cleanup_precede_publish(self):
        from cheating_racer_detect.events.inputs import write_bytes
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = generate(TOOLS, root)
            for target in ('receipt', 'pixels'):
                bridge, run = self.run_capture(source, root/'saved')
                def write(path, data):
                    if path.name == 'capture.json':
                        raise OSError('late disk')
                    write_bytes(path, data)
                context = patch('cheating_racer_detect.events.capture.write_bytes', side_effect=write) if target == 'receipt' else \
                    patch('cheating_racer_detect.events.capture.release_pixels', side_effect=OSError('cleanup'))
                with context, self.assertRaises(Exception):
                    run()
                self.assertFalse((root/'saved').exists())
                self.assertIsNone(bridge.tracker._previous)
                self.assertEqual(list(root.glob('*.partial')), [])


if __name__ == '__main__':
    unittest.main()
