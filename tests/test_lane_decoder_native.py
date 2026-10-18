"""Actual FFmpeg source clocks/pixels + own scripted logits, not lane inference."""

from dataclasses import replace
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.analysis import decode_ufld
from cheating_racer_detect.pixels import pixel_window
from cheating_racer_detect.review.video import prepare
from cheating_racer_detect.tools import discover_tools
from cheating_racer_detect.tracking import Frame
from tests import fixtures
from tests.test_lane_decoder import output

try:
    TOOLS = discover_tools()
except Exception:
    TOOLS = None
AVAILABLE = importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(TOOLS and AVAILABLE, 'NOT_RUN: FFmpeg/NumPy required')
class NativeFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='crd-lane-native-')
        cls.root = Path(cls.temporary.name)
        try:
            cls.media = fixtures.generate(TOOLS, cls.root)
        except BaseException:
            cls.temporary.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        if cls.root.exists():
            raise AssertionError('Owned native fixture directory remains')

    def run_decode(self, name, fail=None):
        source = self.media[name]
        before = fixtures.sha256(source)
        video, timeline, selected, first, rotation = prepare(source, '0', '.6', TOOLS)
        children = set(self.root.iterdir())
        results = []
        try:
            # Own raw window inside this fixture root makes cleanup observable.
            with patch('cheating_racer_detect.pixels.tempfile.gettempdir', return_value=str(self.root)):
                with pixel_window(source, TOOLS, video, timeline, first, len(selected), complete=True) as stream:
                    for index, item in enumerate(selected):
                        current = Frame(before, video['index'], first+index, item.pts,
                            timeline.time_base, timeline.frames[0].pts,
                            video['width'], video['height'], rotation)
                        pixels = stream.read(current.width*current.height*3)
                        self.assertEqual(len(pixels), current.width*current.height*3)
                        if index == 1 and fail is not None:
                            raise fail
                        # Coded raw decoding did NOT autorotate. Do not infer a
                        # preprocessing turn from the source display metadata.
                        decoded = decode_ufld(current, output(current, rotation=0))
                        self.assertIs(decoded.frame, current)
                        self.assertEqual(decoded.report()['frame'], current.report())
                        self.assertEqual(decoded.rotation_cw, 0)
                        results.append(decoded)
        finally:
            self.assertEqual(set(self.root.iterdir()), children)
            self.assertEqual(fixtures.sha256(source), before)
        return results, timeline

    def test_six_profiles_exact_pts_timebase_and_original_coded_geometry(self):
        for name in ('cfr', 'vfr', 'offset', 'rotation90', 'rotation180', 'rotation270'):
            with self.subTest(profile=name):
                decoded, timeline = self.run_decode(name)
                self.assertGreaterEqual(len(decoded), 2)
                independent = json.loads(fixtures.run([TOOLS.ffprobe, '-v', 'error',
                    '-select_streams', 'v:0', '-show_frames', '-show_entries', 'frame=pts',
                    '-of', 'json', self.media[name]]))['frames']
                for batch in decoded:
                    current = batch.frame
                    self.assertEqual(current.pts, independent[current.ordinal]['pts'])
                    self.assertEqual(current.time, (current.pts-current.first_pts)*timeline.time_base)
                    self.assertEqual((current.width, current.height), (fixtures.WIDTH, fixtures.HEIGHT))
                if name == 'offset':
                    self.assertGreater(decoded[0].frame.first_pts, 0)
                    self.assertEqual(decoded[0].frame.time, 0)
                if name == 'vfr':
                    deltas = {b.frame.time-a.frame.time for a, b in zip(decoded, decoded[1:])}
                    self.assertGreater(len(deltas), 1)
                if name.startswith('rotation'):
                    self.assertEqual(decoded[0].frame.display_rotation, int(name[8:]))

    def test_clock_and_ordinal_mismatch_refused_after_real_decode(self):
        decoded, _timeline = self.run_decode('offset')
        source = decoded[0].frame
        raw = output(source)
        for changed in (replace(source, pts=source.pts+1), replace(source, ordinal=1),
                        replace(source, time_base=Fraction(1, 12345)),
                        replace(source, display_rotation=90)):
            with self.assertRaises(ValueError):
                decode_ufld(changed, raw)

    def test_failure_cancel_cleanup_source_unchanged_and_fresh_retry(self):
        for failure in (RuntimeError('scripted failure'), KeyboardInterrupt()):
            with self.assertRaises(type(failure)):
                self.run_decode('vfr', fail=failure)
            retried, _timeline = self.run_decode('vfr')
            self.assertTrue(all(batch.slots[0].support_passed for batch in retried))


if __name__ == '__main__':
    unittest.main()
