"""Independent native coded pixels/AAC oracles, real child cancel and retry."""

from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect import media, pixels, tools
from cheating_racer_detect.errors import AppError
from cheating_racer_detect.review.video import prepare
from cheating_racer_detect.source import source_session
from tests import fixtures

try:
    TOOLS = tools.discover_tools()
except AppError:
    TOOLS = None


@unittest.skipUnless(TOOLS, 'NOT_RUN: existing FFmpeg required')
class Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='crd-window-native-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.paths = fixtures.generate(TOOLS, cls.root)

    def test_complete_and_bounded_pixels_equal_independent_full_coded_decode(self):
        for profile in ('cfr', 'vfr', 'offset', 'audio', 'audio_delayed', 'rotation90', 'rotation180', 'rotation270'):
            with self.subTest(profile=profile):
                source = self.paths[profile]
                before = fixtures.sha256(source)
                video, timeline, frames, first, rotation = prepare(source, F(1, 20), F(281, 100), TOOLS)
                raw = fixtures.run([TOOLS.ffmpeg, '-v', 'error', '-noautorotate', '-i', source,
                    '-map', '0:v:0', '-fps_mode', 'passthrough', '-an', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])
                size = video['width']*video['height']*3
                self.assertEqual(len(raw), size*len(timeline.frames))
                for complete in (False, True):
                    with source_session(pixel_parent=self.root) as job:
                        with pixels.pixel_window(source, TOOLS, video, timeline, first, len(frames), complete=complete) as stream:
                            self.assertEqual(stream.read(size*len(frames)), raw[first*size:(first+len(frames))*size])
                        snapshot = job.pixels
                        with patch.object(pixels, 'run_tool') as decode:
                            with pixels.pixel_window(source, TOOLS, video, timeline, first+1, 1) as stream:
                                self.assertEqual(stream.read(size), raw[(first+1)*size:(first+2)*size])
                            decode.assert_not_called()
                        self.assertEqual(snapshot.raw.stat().st_size, size*len(frames))
                    self.assertFalse(snapshot.directory.exists())
                self.assertEqual(fixtures.sha256(source), before)

    def test_audio_selected_summaries_equal_independent_json_list(self):
        for profile in ('audio', 'audio_delayed', 'offset', 'rotation90', 'rotation180', 'rotation270'):
            with self.subTest(profile=profile):
                source = self.paths[profile]
                video, audio = media._profile(media._probe(source, TOOLS, 'UNSUPPORTED_MEDIA'))
                records = json.loads(fixtures.run([TOOLS.ffprobe, '-v', 'error', '-select_streams', 'a:0',
                    '-show_frames', '-show_entries', 'frame=pts,nb_samples', '-of', 'json', source]))['frames']
                base, rate = F(audio['time_base']), int(audio['sample_rate'])
                intervals = [(int(r['pts'])*base, int(r['pts'])*base+F(int(r['nb_samples']), rate)) for r in records]
                for window in (None, (F(0), F(1, 4)), (F(1, 2), F(12001, 24000)),
                               (F(13, 10), F(17, 10)), (F(5), F(6)), (F(100), F(101))):
                    selected = intervals if window is None else [(max(a, window[0]), min(b, window[1]))
                        for a, b in intervals if a < window[1] and b > window[0]]
                    expected = media.AudioWindow(selected[0][0] if selected else None,
                                                 selected[-1][1] if selected else None, len(selected))
                    self.assertEqual(media._audio_window(source, audio, TOOLS, 'INVALID_TIMELINE', window), expected)

    def test_actual_raw_child_cancel_reaped_window_removed_then_retry(self):
        source = self.paths['cfr']
        video, timeline, frames, first, rotation = prepare(source, F(1, 20), F(1), TOOLS)
        processes, directories = [], []
        actual_popen, actual_mkdtemp = subprocess.Popen, tempfile.mkdtemp
        def mkdtemp(*args, **kwargs):
            result = actual_mkdtemp(*args, **kwargs)
            directories.append(Path(result))
            return result
        def popen(*args, **kwargs):
            process = actual_popen(*args, **kwargs)
            processes.append(process)
            original, first_call = process.communicate, True
            def communicate(*args, **kwargs):
                nonlocal first_call
                if first_call:
                    first_call = False
                    raise KeyboardInterrupt()
                return original(*args, **kwargs)
            process.communicate = communicate
            return process
        with patch.object(pixels.tempfile, 'mkdtemp', side_effect=mkdtemp), \
                source_session(pixel_parent=self.root) as job:
            with patch.object(subprocess, 'Popen', side_effect=popen), self.assertRaises(KeyboardInterrupt):
                with pixels.pixel_window(source, TOOLS, video, timeline, first, len(frames), complete=True):
                    self.fail('cancel must not return pixels')
            self.assertIsNone(job.pixels)
            self.assertTrue(all(p.poll() is not None for p in processes))
            self.assertTrue(all(p.stdout.closed and p.stderr.closed for p in processes))
            self.assertTrue(all(not d.exists() for d in directories))
            with pixels.pixel_window(source, TOOLS, video, timeline, first, len(frames), complete=True) as stream:
                self.assertEqual(len(stream.read()), len(frames)*96*64*3)
        self.assertTrue(all(not d.exists() for d in directories))


if __name__ == '__main__':
    unittest.main()
