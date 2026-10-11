from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import threading
import tracemalloc
import unittest
from unittest.mock import Mock, patch

from cheating_racer_detect.tools import Toolchain
from cheating_racer_detect.media import Frame, Timeline
from fractions import Fraction as F
from scripts import benchmark_scaling as scaling


class Diagnostic(unittest.TestCase):
    def test_bad_input_rejected_before_discovery_or_generation(self):
        cases = [(n, 'tail') for n in (True, 2, 601, 3., '3', None)]+[(3, w) for w in ('', None, 'middle')]
        for seconds, window in cases:
            with self.subTest(seconds=seconds, window=window), patch.object(scaling.tools, 'discover_tools') as discover, \
                    patch.object(scaling, 'generate_padded') as generate:
                with self.assertRaises(ValueError):
                    scaling.benchmark(seconds=seconds, window=window)
                discover.assert_not_called()
                generate.assert_not_called()
        for seconds in (3, 600):
            for window in ('head', 'tail'):
                scaling.validate(seconds, window)

    def test_existing_trace_preserved_and_cli_bad_arguments(self):
        tracemalloc.start()
        try:
            with patch.object(scaling.tools, 'discover_tools') as discover:
                with self.assertRaises(RuntimeError):
                    scaling.benchmark(seconds=3)
                discover.assert_not_called()
            with self.assertRaises(RuntimeError), scaling.allocation_trace():
                self.fail('Existing trace cannot be claimed')
            self.assertTrue(tracemalloc.is_tracing())
        finally:
            tracemalloc.stop()
        with redirect_stderr(StringIO()), patch.object(scaling, 'benchmark') as run:
            for argv in (['--seconds', '2'], ['--seconds', '3.5'], ['--window', 'other']):
                with self.assertRaises(SystemExit) as error:
                    scaling.main(argv)
                self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_measurements_count_logical_reads_characters_and_failed_elapsed(self):
        with tempfile.TemporaryDirectory(prefix='crd-scaling-unit-') as directory:
            source = Path(directory)/'source.mp4'
            source.write_bytes(b'1234')
            tools = Toolchain(Path('ffmpeg'), Path('ffprobe'))
            metrics = scaling.Measurements(source, tools)
            metrics.actual_tool = Mock(return_value='{"frames": []}')
            self.assertEqual(metrics.run([tools.ffprobe, '-show_frames', source]), '{"frames": []}')
            self.assertEqual(metrics.timeline_stdout_characters, 14)
            metrics.actual_hash = Mock(return_value='fake')
            self.assertEqual(metrics.hash_file(source), 'fake')
            metrics.hash_file(Path('derived'))
            self.assertEqual((metrics.hash_reads, metrics.hash_bytes), (1, 4))
            self.assertGreaterEqual(metrics.hash_seconds, 0)
            metrics.actual_tool.side_effect = KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                metrics.run([tools.ffprobe, '-show_frames', source])
            self.assertEqual(metrics.tool_calls['source_timeline_probe'], 2)
            self.assertGreaterEqual(metrics.tool_seconds['source_timeline_probe'], 0)
            self.assertEqual(metrics.timeline_stdout_characters, 14)

    def test_generation_uses_owned_padding_and_explicit_timeout(self):
        tools = Toolchain(Path('ffmpeg'), Path('ffprobe'))
        root = Path('owned')
        for window, expected in (('tail', 'tpad=start=6:start_mode=clone'), ('head', 'tpad=stop=6:stop_mode=clone')):
            with patch.object(scaling.demo_events, 'generate', return_value=root/'synthetic.mp4'), \
                    patch.object(scaling.tools, 'run_tool') as run:
                self.assertEqual(scaling.generate_padded(tools, root, 3, window), root/'padded.mp4')
                self.assertIn(expected, run.call_args.args[0])
                self.assertIn('-n', run.call_args.args[0])
                self.assertEqual(run.call_args.kwargs['timeout'], 120)

    def test_error_and_cancel_close_trace_sampler_and_temp_then_retry(self):
        tools = Toolchain(Path('ffmpeg'), Path('ffprobe'))
        owned = []
        def generate(toolchain, root, seconds, window):
            owned.append(root)
            source = root/'padded.mp4'
            source.write_bytes(b'fixture')
            return source
        for failure in (ValueError('invalid fixture'), KeyboardInterrupt(), OSError('disk')):
            with patch.object(scaling.tools, 'discover_tools', return_value=tools), \
                    patch.object(scaling, 'generate_padded', side_effect=generate), \
                    patch.object(scaling, 'inspect_source', side_effect=failure):
                with self.assertRaises(type(failure)):
                    scaling.benchmark(seconds=3)
            self.assertFalse(owned[-1].exists())
            self.assertFalse(tracemalloc.is_tracing())
            self.assertFalse(any(t.name == 'crd-owned-disk-sampler' for t in threading.enumerate()))
        with scaling.allocation_trace():
            self.assertTrue(tracemalloc.is_tracing())
        self.assertFalse(tracemalloc.is_tracing())

    def test_generation_failure_cleans_owned_temp_without_starting_trace(self):
        owned = []
        def fail(toolchain, root, seconds, window):
            owned.append(root)
            (root/'partial-fixture').write_bytes(b'owned')
            raise KeyboardInterrupt()
        with patch.object(scaling.tools, 'discover_tools'), patch.object(scaling, 'generate_padded', side_effect=fail):
            with self.assertRaises(KeyboardInterrupt):
                scaling.benchmark(seconds=3)
        self.assertFalse(owned[0].exists())
        self.assertFalse(tracemalloc.is_tracing())

    def test_sampler_cannot_scan_during_publish_and_never_retries_failures(self):
        with tempfile.TemporaryDirectory(prefix='crd-scaling-lock-unit-') as directory:
            sampler = scaling.DiskSampler(Path(directory))
            sampled = threading.Event()
            thread = threading.Thread(target=sampler.sample)
            def publish():
                thread.start()
                self.assertFalse(sampled.wait(.02))
                return 'published'
            with patch.object(sampler, '_sample', side_effect=sampled.set):
                try:
                    self.assertEqual(sampler.publish(publish), 'published')
                finally:
                    thread.join(timeout=1)
                self.assertFalse(thread.is_alive())
                self.assertTrue(sampled.is_set())
                failing = Mock(side_effect=OSError('locked destination'))
                with self.assertRaises(OSError):
                    sampler.publish(failing)
                failing.assert_called_once_with()
                sampled.clear()
                retry = threading.Thread(target=sampler.sample)
                retry.start()
                retry.join(timeout=1)
                self.assertFalse(retry.is_alive())
                self.assertTrue(sampled.is_set())

    def test_cli_success_emits_only_measurement_json(self):
        output = StringIO()
        with patch.object(scaling, 'benchmark', return_value={'status': 'complete'}) as run, \
                patch('sys.stdout', output):
            scaling.main(['--seconds', '10', '--window', 'head', '--no-reuse'])
        run.assert_called_once_with(seconds=10, window='head', reuse=False)
        self.assertEqual(output.getvalue(), '{"status": "complete"}\n')

    def test_demo_rejects_wrong_selection_count_and_pixel_count_before_observation(self):
        tools = Toolchain(Path('ffmpeg'), Path('ffprobe'))
        video = {'index': 0, 'width': 96, 'height': 64}
        frames = tuple(Frame(n, 1) for n in range(24))
        timeline = Timeline(F(1, 10), frames)
        with tempfile.TemporaryDirectory(prefix='crd-scaling-demo-unit-') as directory:
            root = Path(directory)
            with patch.object(scaling.demo_events, 'prepare', return_value=(video, timeline, frames[:-1], 0, 0)), \
                    patch.object(scaling.demo_events, 'run_tool') as run:
                with self.assertRaises(ValueError):
                    scaling.demo_events.analyze(root/'source', tools, root)
                run.assert_not_called()
            def bad_pixels(*args, **kwargs):
                (root/'decoded.bgr').write_bytes(b'partial frame')
            with patch.object(scaling.demo_events, 'prepare', return_value=(video, timeline, frames, 0, 0)), \
                    patch.object(scaling.demo_events, 'run_tool', side_effect=bad_pixels), \
                    patch.object(scaling.demo_events, 'observation') as observe:
                with self.assertRaises(ValueError):
                    scaling.demo_events.analyze(root/'source', tools, root)
                observe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
