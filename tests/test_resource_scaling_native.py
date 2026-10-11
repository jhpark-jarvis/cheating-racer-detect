import importlib.util
from fractions import Fraction as F
from pathlib import Path
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.tools import discover_tools
from scripts import benchmark_scaling as scaling

try:
    TOOLS = discover_tools()
except AppError:
    TOOLS = None
AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('numpy', 'cv2'))


@unittest.skipUnless(AVAILABLE and TOOLS, 'NOT_RUN: existing FFmpeg/NumPy/OpenCV required')
class Native(unittest.TestCase):
    def test_head_tail_reuse_keeps_actual_ordinals_pixels_and_validation(self):
        reports = {}
        for window in ('head', 'tail'):
            for reuse in (False, True):
                with self.subTest(window=window, reuse=reuse):
                    result = scaling.benchmark(seconds=3, window=window, reuse=reuse)
                    reports[window, reuse] = result
                    self.assertEqual((result['source_frames'], result['analysis_frames'], result['review_frames']), (30, 24, 14))
                    first = 0 if window == 'head' else 6
                    self.assertEqual(result['analysis_first_ordinal'], first)
                    self.assertEqual(result['review_first_ordinal'], first+3)
                    self.assertEqual(result['review_last_ordinal'], first+16)
                    self.assertEqual(F(result['review_source_time_at_zero']), F(first+3, 10))
                    self.assertEqual(result['review_relative_times'], [str(F(n, 10)) for n in range(14)])
                    self.assertEqual(result['analysis_raw_bytes'], 24*96*64*3)
                    self.assertEqual((result['status'], result['signal_state'], result['source_unchanged']), ('complete', 'right_blink', True))
                    self.assertEqual(result['source_sha256_reads'], 7)
                    self.assertEqual(result['source_sha256_logical_bytes'], 7*result['source_file_bytes'])
                    calls = result['tool_calls']
                    self.assertEqual(calls['source_metadata_probe'], 1 if reuse else 5)
                    self.assertEqual(calls['source_timeline_probe'], 1 if reuse else 5)
                    for key, count in (('source_full_decode', 1), ('source_raw_decode', 2),
                            ('derived_metadata_probe', 2), ('derived_timeline_probe', 2), ('derived_full_decode', 2)):
                        self.assertEqual(calls[key], count)
                    self.assertGreater(result['source_timeline_stdout_characters'], 0)
                    self.assertGreaterEqual(result['python_traced_peak_bytes'], result['inspection_python_traced_peak_bytes'])
                    self.assertFalse(tracemalloc.is_tracing())
                    self.assertTrue(result['disk_sampling_serialized_with_publication'])
            for key in ('analysis_first_ordinal', 'review_first_ordinal', 'review_last_ordinal',
                        'review_source_time_at_zero', 'review_relative_times', 'signal_state'):
                self.assertEqual(reports[window, False][key], reports[window, True][key])
            self.assertEqual(reports[window, False]['source_timeline_stdout_characters'],
                             5*reports[window, True]['source_timeline_stdout_characters'])

    def test_ten_second_fixture_and_success_owned_cleanup(self):
        directories = []
        actual = tempfile.TemporaryDirectory
        def temporary(*args, **kwargs):
            directory = actual(*args, **kwargs)
            directories.append(Path(directory.name))
            return directory
        with patch.object(scaling.tempfile, 'TemporaryDirectory', side_effect=temporary):
            result = scaling.benchmark(seconds=10, window='tail')
        self.assertEqual((result['source_frames'], result['analysis_first_ordinal'], result['review_first_ordinal']), (100, 76, 79))
        self.assertEqual(result['signal_state'], 'right_blink')
        self.assertTrue(directories)
        self.assertTrue(all(not directory.exists() for directory in directories))


if __name__ == '__main__':
    unittest.main()
