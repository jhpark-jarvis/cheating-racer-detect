"""Actual encoded synthetic pixels/PTS -> native Tracker -> event media bundle."""

from dataclasses import replace
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events.export import export_candidate
from cheating_racer_detect.tools import discover_tools
from scripts.demo_events import analyze, generate, observation
from tests.fixtures import probe, sha256

try:
    TOOLS = discover_tools()
except AppError:
    TOOLS = None
AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('cv2', 'numpy'))


@unittest.skipUnless(AVAILABLE and TOOLS is not None, 'NOT_RUN: existing FFmpeg/OpenCV/NumPy required')
class Native(unittest.TestCase):
    def test_synthetic_real_pixels_native_tracking_and_exact_candidate_mapping(self):
        with tempfile.TemporaryDirectory(prefix='crd-event-native-') as directory:
            root = Path(directory)
            source = generate(TOOLS, root)
            before = sha256(source)
            candidate, lamps = analyze(source, TOOLS, root)
            self.assertEqual(candidate.direction, 'right')
            self.assertEqual(candidate.identity, (1, 2, 1))
            output = root/'result'
            result = export_candidate(source, candidate, output, observation(), pre=F(1, 5), post=F(2, 5), lamps=lamps, toolchain=TOOLS)
            self.assertEqual(result['signal_state'], 'right_blink')
            self.assertEqual(result['lamp']['left']['state'], 'no_blink_observed')
            self.assertEqual(json.loads((output/'candidate.json').read_text(encoding='utf-8')), result)
            media = json.loads((output/'media'/'review.json').read_text(encoding='utf-8'))
            actual = [f['source_frame'] for f in media['frames']]
            for pair in (candidate.start, candidate.crossing, candidate.completion):
                for frame in pair:
                    self.assertIn(frame.report(), actual)
            self.assertEqual([s['codec_type'] for s in probe(TOOLS, output/'media'/'review.mp4')['streams']], ['video'])
            self.assertEqual(sha256(source), before)
            self.assertEqual(list(root.glob('.crd-candidate-*.partial')), [])
            self.assertEqual(list(output.rglob('*.bgr')), [])
            # Identity mismatch and wrong source reject before publishing.
            for value in (replace(candidate, identity=(2, 2, 1)),
                          replace(candidate, start=tuple(replace(f, source_id='other') for f in candidate.start),
                              crossing=tuple(replace(f, source_id='other') for f in candidate.crossing),
                              completion=tuple(replace(f, source_id='other') for f in candidate.completion))):
                with self.assertRaises(AppError):
                    export_candidate(source, value, root/'bad', observation(), pre=F(1, 5), post=F(2, 5), lamps=lamps, toolchain=TOOLS)
                self.assertFalse((root/'bad').exists())
            # Context is clipped at true source duration, not guessed FPS.
            end = export_candidate(source, candidate, root/'clamped', observation(), pre=F(10), post=F(10), lamps=lamps, toolchain=TOOLS)
            self.assertEqual(end['context'], ['0', '12/5'])
            self.assertEqual(end['signal_state'], 'unknown')  # no sample at inclusive source end


if __name__ == '__main__':
    unittest.main()
