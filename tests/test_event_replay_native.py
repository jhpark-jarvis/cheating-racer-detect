"""Actual encoded source/PTS/pixels -> saved inputs -> fresh event engines."""

from dataclasses import replace
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events.inputs import EventInputs, load_inputs, save_inputs
from cheating_racer_detect.events.lamp import LampROI
from cheating_racer_detect.events.replay import analyze_inputs, export_replay
from cheating_racer_detect.tools import discover_tools
from scripts.demo_events import analyze, capture_inputs, generate
from tests.fixtures import sha256

try:
    TOOLS = discover_tools()
except AppError:
    TOOLS = None
AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('numpy', 'cv2'))


@unittest.skipUnless(AVAILABLE and TOOLS, 'NOT_RUN: existing FFmpeg/NumPy/OpenCV required')
class Native(unittest.TestCase):
    def test_saved_inputs_replay_same_candidate_lamp_and_exact_media(self):
        with tempfile.TemporaryDirectory(prefix='crd-replay-native-') as directory:
            root = Path(directory)
            source = generate(TOOLS, root)
            before = sha256(source)
            inputs = capture_inputs(source, TOOLS, root)
            candidate, lamps = analyze(source, TOOLS, root)
            save_inputs(inputs, root/'saved')
            restored = load_inputs(root/'saved'/'inputs.json')
            replay = analyze_inputs(source, restored, TOOLS)
            self.assertEqual(replay.candidates, (candidate,))
            self.assertEqual(replay.lamps.report(F(3, 10), F(8, 5)), lamps.report(F(3, 10), F(8, 5)))
            output = root/'result'
            result = export_replay(source, restored, output, candidate_index=0, toolchain=TOOLS)
            self.assertEqual(result['signal_state'], 'right_blink')
            self.assertEqual(result['source_sha256'], before)
            self.assertEqual(result['inputs_sha256'], sha256(output/'inputs.json'))
            receipt = json.loads((output/'candidate'/'candidate.json').read_text(encoding='utf-8'))
            mapping = json.loads((output/'candidate'/'media'/'review.json').read_text(encoding='utf-8'))
            self.assertEqual(receipt['candidate'], candidate.report())
            self.assertEqual(mapping['review']['frame_count'], 14)
            self.assertEqual(receipt['context'], ['3/10', '17/10'])
            self.assertEqual(load_inputs(output/'inputs.json'), restored)
            for sample in receipt['lamp']['samples']:
                self.assertIn(sample['vehicle']['frame'], [f['source_frame'] for f in mapping['frames']])
            self.assertEqual(sha256(source), before)
            self.assertEqual(list(root.glob('.crd-inputs-*.partial')), [])
            self.assertEqual(list(output.rglob('*.bgr')), [])

    def test_source_hash_and_actual_timeline_reject_before_pixel_decode(self):
        with tempfile.TemporaryDirectory(prefix='crd-replay-bad-') as directory:
            root = Path(directory)
            source = generate(TOOLS, root)
            inputs = capture_inputs(source, TOOLS, root)
            # A well-shaped but invented PTS must not become pixel evidence.
            row = inputs.rows[5]
            frame = replace(row.frame, pts=row.frame.pts+1)
            vehicle = replace(row.vehicle, frame=frame, detection=replace(row.vehicle.detection, frame=frame))
            bad = replace(inputs, rows=inputs.rows[:5]+(replace(row, frame=frame, vehicle=vehicle,
                          lanes=replace(row.lanes, frame=frame)),)+inputs.rows[6:])
            with patch('cheating_racer_detect.events.replay.run_tool') as decode:
                with self.assertRaises(AppError):
                    analyze_inputs(source, bad, TOOLS)
                decode.assert_not_called()
            source.write_bytes(b'changed owned source')
            with self.assertRaises(AppError) as error:
                analyze_inputs(source, inputs, TOOLS)
            self.assertEqual(error.exception.code, 'INVALID_INPUTS')

    def test_occlusion_and_missing_recorded_frames_remain_unknown(self):
        with tempfile.TemporaryDirectory(prefix='crd-replay-unknown-') as directory:
            root = Path(directory)
            source = generate(TOOLS, root)
            inputs = capture_inputs(source, TOOLS, root)
            rows = list(inputs.rows)
            rows[8] = replace(rows[8], right=LampROI(None, 'occluded'))
            occluded = EventInputs.from_bytes(replace(inputs, rows=tuple(rows)).to_bytes())
            report = analyze_inputs(source, occluded, TOOLS).lamps.report(F(3, 10), F(8, 5))
            self.assertEqual(report['signal_state'], 'unknown')
            self.assertIn('occluded', report['right']['reasons'])
            sparse = replace(inputs, rows=inputs.rows[:8]+inputs.rows[9:])
            sparse_result = analyze_inputs(source, sparse, TOOLS)
            self.assertEqual(sparse_result.lamps.report(F(3, 10), F(8, 5))['signal_state'], 'unknown')


if __name__ == '__main__':
    unittest.main()
