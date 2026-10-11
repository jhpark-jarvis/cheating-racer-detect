"""Strict incremental protocol, immutable storage and exact selection oracles."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
import json
from pathlib import Path
import random
import tracemalloc
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect import media
from cheating_racer_detect.tools import Toolchain

TOOLS = Toolchain(Path('ffmpeg'), Path('ffprobe'))
VIDEO = dict(time_base='1/1000', width=320, height=180, pix_fmt='yuv420p',
             start_pts=0, duration_ts=1000)


def frame(pts=0, duration=100, **changes):
    return dict(pts=pts, duration=duration, width=320, height=180, pix_fmt='yuv420p', **changes)


def wire(rows):
    return ''.join('[FRAME]\n' + ''.join(f'{key}={value}\n' for key, value in row.items())
                   + '[/FRAME]\n' for row in rows)


def decode(text, video=None, code='INVALID_TIMELINE'):
    def run(_args, consume, **_kwargs):
        for line in text.splitlines(keepends=True):
            consume(line)
    with patch.object(media, 'run_tool_lines', side_effect=run):
        return media._timeline(Path('input.mp4'), video or VIDEO, TOOLS, code)


class FrameProtocolTests(unittest.TestCase):
    def test_reordered_fields_empty_side_data_and_optional_absence(self):
        text = '[FRAME]\npix_fmt=yuv420p\nheight=180\nwidth=320\npts=-5000\nduration=N/A\npkt_duration=100\n[SIDE_DATA]\n[/SIDE_DATA]\n[/FRAME]\n'
        timeline = decode(text)
        self.assertEqual(timeline.frames, (media.Frame(-5000, 100),))
        self.assertEqual(timeline.origin, -5)

    def test_zero_duration_inferred_online_and_last_from_stream_end(self):
        timeline = decode(wire([frame(0, 0), frame(400, 0), frame(900, 0)]))
        self.assertEqual(timeline.frames, (media.Frame(0, 400), media.Frame(400, 500), media.Frame(900, 100)))
        self.assertEqual(timeline.end, 1)

    def test_missing_duration_uses_zero_fallback_not_average_fps(self):
        rows = [frame(0), frame(900)]
        for row in rows:
            del row['duration']
        self.assertEqual(decode(wire(rows)).frames, (media.Frame(0, 900), media.Frame(900, 100)))

    def test_invalid_clock_duration_and_missing_fields(self):
        bad = [wire([]), wire([frame(2**52)]), wire([frame(-2**52)]), wire([frame(0, -1)]),
               wire([frame(0), frame(0)]), wire([frame(10), frame(0)]),
               wire([{k: v for k, v in frame().items() if k != 'pts'}]),
               wire([frame('N/A')]), wire([frame(0, 'bad')]), wire([frame(1000, 0)])]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(AppError):
                decode(text)
        for base in ('0', '-1/1000', '1/2147483648', 'bad'):
            with self.subTest(base=base), self.assertRaises(AppError):
                decode(wire([frame()]), {**VIDEO, 'time_base': base})

    def test_late_format_or_hdr_change_not_hidden_by_short_selection(self):
        for changes in ({'width': 640}, {'height': 360}, {'pix_fmt': 'yuv420p10le'},
                        {'color_transfer': 'smpte2084'}, {'color_transfer': 'arib-std-b67'},
                        {'color_primaries': 'bt2020'}):
            for code in ('INVALID_TIMELINE', 'OUTPUT_INVALID'):
                rows = [frame(), {**frame(900), **changes}]
                with self.subTest(changes=changes, code=code), self.assertRaises(AppError) as ctx:
                    decode(wire(rows), code=code)
                self.assertEqual(ctx.exception.code, 'OUTPUT_INVALID' if code == 'OUTPUT_INVALID' else 'UNSUPPORTED_MEDIA')

    def test_strict_sections_keys_and_complete_eof(self):
        valid = wire([frame()])
        bad = ['pts=0\n', '[FRAME]\n[FRAME]\n', '[/FRAME]\n', '[ERROR]\n',
               valid.replace('pts=0\n', 'pts=0\npts=1\n'),
               valid.replace('pts=0\n', 'extra=1\npts=0\n'),
               valid.replace('pts=0\n', 'pts=\n'), valid.replace('pix_fmt=yuv420p', 'pix_fmt=yuv\\n420p'),
               valid.replace('[/FRAME]\n', ''), valid + '[FRAME]\n',
               valid.replace('[/FRAME]', '[SIDE_DATA]\nfield=x\n[/SIDE_DATA]\n[/FRAME]'),
               valid.replace('[/FRAME]', '[SIDE_DATA]\n[SIDE_DATA]\n[/SIDE_DATA]\n[/FRAME]'),
               valid.replace('[/FRAME]', '[SIDE_DATA]\n[/FRAME]'), valid + '\n']
        for text in bad:
            with self.subTest(text=text), self.assertRaises(AppError):
                decode(text)

    def test_native_error_after_valid_frame_cannot_return_partial_timeline(self):
        def fail(args, consume, **kwargs):
            for line in wire([frame()]).splitlines(keepends=True):
                consume(line)
            raise AppError(kwargs['error_code'], 'fixture')
        with patch.object(media, 'run_tool_lines', side_effect=fail), self.assertRaises(AppError):
            media._timeline(Path('input.mp4'), VIDEO, TOOLS, 'INVALID_TIMELINE')

    def test_probe_keeps_full_input_security_options_and_scalar_protocol(self):
        def run(args, consume, **kwargs):
            for option in ('-protocol_whitelist', '-enable_drefs', '-use_absolute_path', '-ignore_chapters', '-err_detect'):
                self.assertIn(option, args)
            self.assertEqual(args[args.index('-protocol_whitelist') + 1], 'file')
            self.assertEqual(args[args.index('-of') + 1], 'default')
            self.assertIn(':frame_side_data=', args[args.index('-show_entries') + 1])
            self.assertNotIn('-read_intervals', args)
            for line in wire([frame()]).splitlines(keepends=True):
                consume(line)
        with patch.object(media, 'run_tool_lines', side_effect=run):
            media._timeline(Path('input.mp4'), VIDEO, TOOLS, 'INVALID_TIMELINE')

    def test_audio_samples_and_overlap_still_validated(self):
        def audio(text):
            def run(args, consume, **kwargs):
                self.assertEqual(args[args.index('-select_streams') + 1], 'a:0')
                for line in text.splitlines(keepends=True):
                    consume(line)
            with patch.object(media, 'run_tool_lines', side_effect=run):
                return media._audio_window(Path('input.mp4'), {'time_base': '1/48000', 'sample_rate': 48000}, TOOLS, 'OUTPUT_INVALID')
        self.assertEqual(audio(wire([{'pts': 24000, 'nb_samples': 1024}, {'pts': 25024, 'nb_samples': 1024}])),
                         media.AudioWindow(F(1, 2), F(407, 750), 2))
        for rows in ([{'pts': 0, 'nb_samples': 0}], [{'pts': 0, 'nb_samples': -1}],
                     [{'pts': 'N/A', 'nb_samples': 1024}],
                     [{'pts': 0, 'nb_samples': 1024}, {'pts': 100, 'nb_samples': 1024}]):
            with self.subTest(rows=rows), self.assertRaises(AppError):
                audio(wire(rows))


class PackedTimelineTests(unittest.TestCase):
    def test_index_slice_iteration_equality_hash_and_immutability(self):
        timeline = decode(wire([frame(0), frame(100), frame(300), frame(700)]))
        expected = tuple(media.Frame(n, 100) for n in (0, 100, 300, 700))
        self.assertEqual(tuple(timeline.frames), expected)
        self.assertEqual(timeline.frames, expected)
        self.assertEqual(expected, timeline.frames)
        self.assertEqual(hash(timeline.frames), hash(expected))
        self.assertEqual(timeline.frames[::-1], expected[::-1])
        self.assertEqual(timeline.frames[1:3], expected[1:3])
        self.assertEqual(timeline.frames[-1], expected[-1])
        self.assertEqual(timeline.storage_bytes, 16 * len(expected))
        for index in (-5, 4):
            with self.assertRaises(IndexError):
                timeline.frames[index]
        with self.assertRaises(TypeError):
            timeline.frames[1.0]
        with self.assertRaises(FrozenInstanceError):
            timeline.frames.data = b'bad'
        with self.assertRaises(FrozenInstanceError):
            timeline.frames[0].pts = 999
        with self.assertRaises(TypeError):
            timeline.frames[0] = expected[1]

    def test_wide_duration_fallback_does_not_narrow_integer_contract(self):
        huge = 2**65
        timeline = decode(wire([frame(0, 100), frame(100, huge), frame(200, 0)]))
        self.assertIsInstance(timeline.frames, tuple)
        self.assertIsNone(timeline.storage_bytes)
        self.assertEqual(timeline.frames, (media.Frame(0, 100), media.Frame(100, huge), media.Frame(200, 800)))
        timeline = decode(wire([frame(0, 0)]), {**VIDEO, 'duration_ts': huge})
        self.assertEqual(timeline.frames[0].duration, huge)

    def test_exact_fraction_binary_search_matches_linear_vfr_oracle(self):
        randomizer = random.Random(34)
        pts = [-5000]
        for _ in range(299):
            pts.append(pts[-1] + randomizer.randint(1, 900))
        timeline = decode(wire([frame(p, 100) for p in pts]))
        for _ in range(250):
            start = F(randomizer.randrange(0, int(timeline.end * 10**6)), 10**6)
            end = min(timeline.end, start + F(randomizer.randint(1, 100000), 10**6))
            expected = tuple(f for f in timeline.frames if start <= (f.pts - pts[0])*timeline.time_base < end)
            if expected:
                self.assertEqual(media.select_frames(timeline, start, end), expected)
            else:
                with self.assertRaises(AppError) as ctx:
                    media.select_frames(timeline, start, end)
                self.assertEqual(ctx.exception.code, 'EMPTY_INTERVAL')
        boundary = F(pts[99] - pts[0], 1000)
        tiny = F(1, 10**30)
        self.assertEqual(media.select_frames(timeline, boundary, boundary + tiny), (timeline.frames[99],))
        self.assertEqual(timeline.lower_bound(F(pts[99]) + tiny), 100)

    def test_selection_uses_logarithmic_lookup_not_whole_frame_iteration(self):
        class Indexed:
            def __init__(self):
                self.reads = 0
            def __len__(self):
                return 100000
            def __getitem__(self, index):
                self.reads += 1
                return media.Frame(index if index >= 0 else 100000 + index, 1)
            def __iter__(self):
                raise AssertionError('Whole timeline must not be scanned')
        frames = Indexed()
        timeline = media.Timeline(F(1, 1000), frames)
        self.assertEqual(media.select_frames(timeline, F(99998, 1000), F(100)),
                         (media.Frame(99998, 1), media.Frame(99999, 1)))
        self.assertLess(frames.reads, 50)

    def test_same_records_lower_python_peak_than_legacy_json_tuple(self):
        # No arbitrary product gate: only compare two representations of the
        # same 3000 records, exclude pre-built wire strings from both traces.
        rows = [frame(n * 100) for n in range(3000)]
        text, legacy = wire(rows), json.dumps({'frames': rows})
        import io
        lines = io.StringIO(text)
        def reference():
            data = json.loads(legacy)
            return media.Timeline(F(1, 1000), tuple(media.Frame(row['pts'], row['duration']) for row in data['frames']))
        def streamed():
            # Generate bounded lines lazily; splitlines() here would defeat the comparison.
            def run(args, consume, **kwargs):
                lines.seek(0)
                for line in lines:
                    consume(line)
            with patch.object(media, 'run_tool_lines', side_effect=run):
                return media._timeline(Path('input.mp4'), VIDEO, TOOLS, 'INVALID_TIMELINE')
        def measured(function):
            tracemalloc.start()
            try:
                result = function()
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            return result, peak
        old, old_peak = measured(reference)
        new, new_peak = measured(streamed)
        self.assertEqual(new, old)
        self.assertEqual(new.storage_bytes, 48000)
        self.assertLess(new_peak, old_peak)


if __name__ == '__main__':
    unittest.main()
