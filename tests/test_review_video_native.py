"""Actual FFmpeg/OpenCV generated media -> review MP4 and original clip."""

from decimal import Decimal
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.review import export_review, render
from cheating_racer_detect.review import video
from cheating_racer_detect.tracking import Box, Detection, Tracker
from cheating_racer_detect.tools import discover_tools
from tests import fixtures
from tests.test_tracking import policy

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('cv2', 'numpy'))
try:
    TOOLS = discover_tools()
except AppError:
    TOOLS = None


class ScriptedObserver:
    """Known synthetic boxes only, never model accuracy or calibrated policy."""
    def __init__(self):
        self.tracker = Tracker(policy())
        self.frames = []
        self.pixels = []
        self.reports = []

    def __call__(self, frame, pixels):
        self.frames.append(frame)
        self.pixels.append(pixels)
        detections = [Detection(2, 5, Box(60, 35, 80, 55), .9, 'scripted-v1', frame)]
        if frame.ordinal != 3:
            detections.append(Detection(1, 2, Box(10, 10, 30, 30), .9, 'scripted-v1', frame))
        report = self.tracker.update(frame, detections)
        self.reports.append(report)
        return report


@unittest.skipUnless(AVAILABLE and TOOLS is not None, 'NOT_RUN: FFmpeg and NumPy/OpenCV are required')
class ActualMedia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='crd-review-native-')
        cls.root = Path(cls.directory.name)
        cls.paths = fixtures.generate(TOOLS, cls.root/'fixtures')

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_six_profiles_exact_pts_pixels_and_original_clip(self):
        import numpy as np
        for name in ('cfr', 'vfr', 'offset', 'rotation90', 'rotation180', 'rotation270'):
            with self.subTest(name=name):
                source = self.paths[name]
                before = fixtures.sha256(source)
                observer = ScriptedObserver()
                output = self.root/('result-'+name)
                result = export_review(source, '.05', '.81', output, observer, TOOLS)
                ids = [i for i in (fixtures.VFR_IDS if name == 'vfr' else range(30)) if .05 <= i/10 < .81]
                ordinals = list(range(1, 1+len(ids)))
                self.assertEqual([f.ordinal for f in observer.frames], ordinals)
                self.assertEqual([f.time for f in observer.frames], [Fraction(i, 10) for i in ids])
                self.assertEqual([f['review_time'] for f in result['frames']], [str(Fraction(i-ids[0], 10)) for i in ids])
                self.assertEqual(result['review']['source_time_at_review_zero'], str(Fraction(ids[0], 10)))
                self.assertEqual(fixtures.sha256(source), before)
                self.assertEqual(json.loads((output/'review.json').read_text(encoding='utf-8')), result)
                source_rgb = fixtures.rgb_frames(TOOLS, source)
                clip_rgb = fixtures.rgb_frames(TOOLS, output/'original'/'clip.mp4')
                self.assertEqual(len(clip_rgb), len(ids))
                for ordinal, actual in zip(ordinals, clip_rgb):
                    delta = np.frombuffer(actual, np.uint8).astype(float)-np.frombuffer(source_rgb[ordinal], np.uint8)
                    self.assertLess(float(np.abs(delta).mean()), 12)
                info = fixtures.probe(TOOLS, output/'review.mp4')
                self.assertEqual(len(info['streams']), 1)
                self.assertEqual(info['streams'][0]['codec_name'], 'h264')
                self.assertEqual(result['review']['audio'], 'none')
                raw = fixtures.run([str(TOOLS.ffmpeg), '-v', 'error', '-i', str(output/'review.mp4'),
                                    '-fps_mode', 'passthrough', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])
                width, height = result['review']['canvas_size']
                size = width*height*3
                self.assertEqual(len(raw), size*len(ids))
                for index, (frame, pixels, report) in enumerate(zip(observer.frames, observer.pixels, observer.reports)):
                    self.assertLess(abs(pixels[0]-(24+ids[index]*6)), 5)
                    expected, _ = render(frame, pixels, report)
                    actual = np.frombuffer(raw[index*size:(index+1)*size], np.uint8).reshape(height, width, 3)
                    # Lossy H.264, not a pixel-exact original; inspect raster/timing via distinct markers.
                    self.assertLess(float(np.abs(actual[138:394, :384].astype(float)-expected[138:394, :384]).mean()), 8)
                    self.assertEqual(frame.display_rotation, int(name[8:]) if name.startswith('rotation') else 0)
                probe = json.loads(fixtures.run([str(TOOLS.ffprobe), '-v', 'error', '-select_streams', 'v:0',
                    '-show_frames', '-show_streams', '-show_entries', 'stream=time_base:frame=pts,duration',
                    '-of', 'json', str(output/'review.mp4')]))
                base = Fraction(probe['streams'][0]['time_base'])
                self.assertEqual([f['pts']*base for f in probe['frames']], [Fraction(i-ids[0], 10) for i in ids])
                self.assertEqual(probe['frames'][-1]['duration']*base, Fraction(result['review']['last_frame_duration']))
                self.assertEqual(result['review']['sha256'], fixtures.sha256(output/'review.mp4'))
                self.assertEqual(list(output.rglob('*.bgr')), [])
        self.assertEqual(list(self.root.glob('.crd-review-*.partial')), [])

    def test_single_frame_and_callback_stale_cancel_retry(self):
        source = self.paths['cfr']
        single = export_review(source, '.05', '.11', self.root/'single', ScriptedObserver(), TOOLS)
        self.assertEqual(single['review']['frame_count'], 1)
        self.assertEqual(single['frames'][0]['review_time'], '0')
        for error in (ValueError, KeyboardInterrupt):
            def fail(frame, pixels):
                raise error('callback failure')
            output = self.root/('fail-'+error.__name__)
            with self.assertRaises((AppError, KeyboardInterrupt)):
                export_review(source, 0, '.2', output, fail, TOOLS)
            self.assertFalse(output.exists())
            self.assertEqual(list(self.root.glob('.crd-review-*.partial')), [])
            export_review(source, 0, '.2', output, ScriptedObserver(), TOOLS)
        old = None
        def stale(frame, pixels):
            nonlocal old
            if old is None:
                old = ScriptedObserver()(frame, pixels)
            return old
        with self.assertRaises(AppError) as caught:
            export_review(source, 0, '.2', self.root/'stale', stale, TOOLS)
        self.assertEqual(caught.exception.code, 'INVALID_REVIEW')
        self.assertFalse((self.root/'stale').exists())

    def test_bounds_and_existing_result(self):
        source = self.paths['cfr']
        with patch.object(video, 'MAX_FRAMES', 1):
            with self.assertRaises(AppError) as caught:
                export_review(source, 0, '.3', self.root/'too-many', ScriptedObserver(), TOOLS)
        self.assertEqual(caught.exception.code, 'REVIEW_LIMIT')
        output = self.root/'preserved'
        export_review(source, 0, '.2', output, ScriptedObserver(), TOOLS)
        before = fixtures.sha256(output/'review.json')
        with self.assertRaises(AppError):
            export_review(source, 0, '.2', output, ScriptedObserver(), TOOLS)
        self.assertEqual(fixtures.sha256(output/'review.json'), before)

    def test_audio_is_only_in_original_and_failures_leave_no_pair(self):
        source = self.paths['audio']
        output = self.root/'audio'
        result = export_review(source, '1.35', '1.71', output, ScriptedObserver(), TOOLS)
        self.assertEqual(result['review']['audio'], 'none')
        self.assertEqual([s['codec_type'] for s in fixtures.probe(TOOLS, output/'original'/'clip.mp4')['streams']], ['video', 'audio'])
        self.assertGreater(len(fixtures.pcm_samples(TOOLS, output/'original'/'clip.mp4')), 1000)
        actual_tool = video.run_tool
        for phase in ('decode', 'encode'):
            failed = self.root/('tool-failed-'+phase)
            def execute(arguments, **kwargs):
                if ((phase == 'decode' and arguments[-1].name == 'source.bgr')
                        or (phase == 'encode' and arguments[-1].name == 'review.mp4')):
                    raise AppError('EXTRACTION_FAILED', 'Injected media failure')
                return actual_tool(arguments, **kwargs)
            with patch.object(video, 'run_tool', side_effect=execute), self.assertRaises(AppError):
                export_review(source, 0, '.2', failed, ScriptedObserver(), TOOLS)
            self.assertFalse(failed.exists())
            self.assertEqual(list(self.root.glob('.crd-review-*.partial')), [])
            export_review(source, 0, '.2', failed, ScriptedObserver(), TOOLS)


if __name__ == '__main__':
    unittest.main()
