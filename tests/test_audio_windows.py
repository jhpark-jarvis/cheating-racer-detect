"""Full AAC validation with constant-count endpoint retention, exact legacy oracle."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from pathlib import Path
import random
import tracemalloc
import unittest
from unittest.mock import patch

from cheating_racer_detect import media
from cheating_racer_detect.errors import AppError
from cheating_racer_detect.tools import Toolchain

TOOLS = Toolchain(Path('ffmpeg'), Path('ffprobe'))
AUDIO = {'time_base': '1/48000', 'sample_rate': 48000}


def summarize(rows, window=None):
    def records(source, stream, fields, tools, code, accept):
        for row in rows:
            accept(row)
    with patch.object(media, '_frame_records', side_effect=records):
        return media._audio_window(Path('mock.mp4'), AUDIO, TOOLS, 'INVALID_TIMELINE', window)


def intervals(rows):
    return [(F(row['pts'], 48000), F(row['pts']+row['nb_samples'], 48000)) for row in rows]


def oracle(rows, window=None):
    selected = intervals(rows)
    if window is not None:
        selected = [(max(a, window[0]), min(b, window[1])) for a, b in selected if a < window[1] and b > window[0]]
    return media.AudioWindow(selected[0][0] if selected else None, selected[-1][1] if selected else None, len(selected))


class Audio(unittest.TestCase):
    def test_exact_random_boundary_negative_delayed_gapped_and_single_sample(self):
        rows = [{'pts': n*1024-1024+(4800 if n > 12 else 0), 'nb_samples': 1024} for n in range(40)]
        rows[-1]['nb_samples'] = 1
        rng = random.Random(35)
        for window in [None, (F(-1), F(0)), (F(100), F(101)), (F(0), F(1, 48000)),
                       (F(2, 100000), F(2, 100000))]+[
                tuple(sorted((F(rng.randrange(-5000, 60000), 48000), F(rng.randrange(-5000, 60000), 48000))))
                for _ in range(150)]:
            with self.subTest(window=window):
                self.assertEqual(summarize(rows, window), oracle(rows, window))

    def test_empty_and_immutable_summary(self):
        value = summarize([])
        self.assertEqual(value, media.AudioWindow(None, None, 0))
        with self.assertRaises(FrozenInstanceError):
            value.count = 1

    def test_one_sample_overlap_tolerance_preserved(self):
        rows = [{'pts': 0, 'nb_samples': 1024}, {'pts': 1023, 'nb_samples': 1024}]
        self.assertEqual(summarize(rows), oracle(rows))
        rows[1]['pts'] = 1022
        with self.assertRaises(AppError):
            summarize(rows)

    def test_late_errors_outside_selected_window_not_skipped(self):
        for bad in ({'pts': 0, 'nb_samples': 1024}, {'pts': 2048, 'nb_samples': 0},
                    {'pts': 2048, 'nb_samples': -1}, {'pts': 'N/A', 'nb_samples': 1024}, {'pts': 2048}):
            rows = [{'pts': 0, 'nb_samples': 1024}, {'pts': 1024, 'nb_samples': 1024}, bad]
            with self.subTest(bad=bad), self.assertRaises(AppError) as caught:
                summarize(rows, (F(0), F(1, 100)))
            self.assertEqual(caught.exception.code, 'INVALID_TIMELINE')

    def test_invalid_audio_clock_and_rate(self):
        for audio in ({'time_base': '0', 'sample_rate': 48000}, {'time_base': '1/48000', 'sample_rate': 0}, {}):
            with patch.object(media, '_frame_records') as run, self.assertRaises(AppError):
                media._audio_window(Path('mock.mp4'), audio, TOOLS, 'OUTPUT_INVALID')
            run.assert_not_called()

    def test_cancel_and_failed_eof_do_not_return_partial_success(self):
        for error in (KeyboardInterrupt(), AppError('INVALID_TIMELINE', 'late child failure')):
            def records(source, stream, fields, tools, code, accept):
                accept({'pts': 0, 'nb_samples': 1024})
                raise error
            with patch.object(media, '_frame_records', side_effect=records), self.assertRaises(type(error)):
                media._audio_window(Path('mock.mp4'), AUDIO, TOOLS, 'INVALID_TIMELINE')
        self.assertEqual(summarize([{'pts': 0, 'nb_samples': 1024}]).count, 1)

    def test_long_list_reference_allocation_decreases_without_skipping_records(self):
        rows = [{'pts': n*1024, 'nb_samples': 1024} for n in range(10000)]
        expected = oracle(rows, (F(1), F(2)))
        def peak(function):
            tracemalloc.start()
            try:
                result = function()
                return result, tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
        old, old_peak = peak(lambda: intervals(rows))
        new, new_peak = peak(lambda: summarize(rows, (F(1), F(2))))
        self.assertEqual(len(old), len(rows))
        self.assertEqual(new, expected)
        self.assertLess(new_peak, old_peak)


if __name__ == '__main__':
    unittest.main()
