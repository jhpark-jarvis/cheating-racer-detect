"""Actual ffprobe JSON oracle and cancellation-to-retry source CLI workflow."""

from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect import media, service, tools
from cheating_racer_detect.errors import AppError
from cheating_racer_detect.source import source_session
from tests.fixtures import generate, run, sha256

try:
    TOOLS = tools.discover_tools()
except AppError:
    TOOLS = None


@unittest.skipUnless(TOOLS, 'NOT_RUN: existing FFmpeg required')
class Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='crd-stream-native-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.paths = generate(TOOLS, cls.root)

    def test_all_video_and_audio_records_equal_independent_json_oracle(self):
        fields = 'frame=pts,duration,pkt_duration,width,height,pix_fmt,color_transfer,color_primaries'
        for name in ('cfr', 'vfr', 'offset', 'audio', 'audio_delayed', 'rotation90', 'rotation180', 'rotation270'):
            with self.subTest(profile=name):
                source = self.paths[name]
                before = sha256(source)
                video, audio = media._profile(media._probe(source, TOOLS, 'UNSUPPORTED_MEDIA'))
                reference = json.loads(run([TOOLS.ffprobe, '-v', 'error', '-select_streams', 'v:0',
                    '-show_frames', '-show_entries', fields, '-of', 'json', source]))['frames']
                expected = [media.Frame(int(row['pts']), int(row.get('duration', row.get('pkt_duration', 0)))) for row in reference]
                for index in range(len(expected)-1):
                    if expected[index].duration == 0:
                        expected[index] = media.Frame(expected[index].pts, expected[index+1].pts-expected[index].pts)
                if expected[-1].duration == 0:
                    expected[-1] = media.Frame(expected[-1].pts, int(video['start_pts'])+int(video['duration_ts'])-expected[-1].pts)
                timeline = media._timeline(source, video, TOOLS, 'INVALID_TIMELINE')
                self.assertEqual(timeline.frames, tuple(expected))
                self.assertEqual(timeline.storage_bytes, 16*len(expected))
                self.assertEqual(timeline.origin, expected[0].pts*F(video['time_base']))
                for start, end in ((F(0), timeline.end), (F(1, 20), F(281, 100)), (F(1), F(3, 2))):
                    selected = tuple(row for row in expected if start <= row.pts*timeline.time_base-timeline.origin < end)
                    self.assertEqual(media.select_frames(timeline, start, end), selected)
                if audio:
                    records = json.loads(run([TOOLS.ffprobe, '-v', 'error', '-select_streams', 'a:0',
                        '-show_frames', '-show_entries', 'frame=pts,nb_samples', '-of', 'json', source]))['frames']
                    base, sample_rate = F(audio['time_base']), int(audio['sample_rate'])
                    expected_audio = [(int(row['pts'])*base, int(row['pts'])*base+F(int(row['nb_samples']), sample_rate)) for row in records]
                    self.assertEqual(media._audio_intervals(source, audio, TOOLS, 'INVALID_TIMELINE'), expected_audio)
                self.assertEqual(sha256(source), before)

    def test_cancel_actual_probe_reaps_child_clears_job_and_retries_clip(self):
        source, output = self.paths['audio'], self.root/'retry-clip'
        before = sha256(source)
        actual_lines, actual_popen = media.run_tool_lines, subprocess.Popen
        children = []
        def spawn(*args, **kwargs):
            child = actual_popen(*args, **kwargs)
            children.append(child)
            return child
        def cancelled(args, consume, **kwargs):
            def interrupt(line):
                raise KeyboardInterrupt()
            return actual_lines(args, interrupt, **kwargs)
        with source_session() as job:
            with patch.object(media, 'run_tool_lines', side_effect=cancelled), \
                    patch('cheating_racer_detect.tools.subprocess.Popen', side_effect=spawn), \
                    self.assertRaises(KeyboardInterrupt):
                service.create_clip(source, '0.05', '1.8', output, TOOLS)
            self.assertIsNone(job.snapshot)
            self.assertFalse(output.exists())
            self.assertEqual(list(self.root.glob('*.partial')), [])
            self.assertTrue(children)
            self.assertTrue(all(child.poll() is not None for child in children))
            streaming = [child for child in children if '-show_frames' in child.args]
            self.assertEqual(len(streaming), 1)
            self.assertTrue(streaming[0].stdout.closed)
            result = service.create_clip(source, '0.05', '1.8', output, TOOLS)
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['media']['verification']['source_full_decode'])
            self.assertTrue(result['media']['verification']['output_full_decode'])
        self.assertIsNone(job.snapshot)
        self.assertEqual(sha256(source), before)
        self.assertEqual(list(self.root.glob('*.partial')), [])


if __name__ == '__main__':
    unittest.main()
