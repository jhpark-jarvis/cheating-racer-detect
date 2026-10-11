"""ML-free owned pixel lifetime/receipt tests; raw command is mocked here."""

from contextvars import copy_context
from fractions import Fraction as F
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from cheating_racer_detect import pixels
from cheating_racer_detect.errors import AppError
from cheating_racer_detect.media import Frame, Timeline
from cheating_racer_detect.source import source_session
from cheating_racer_detect.tools import Toolchain


class Windows(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='crd-pixel-unit-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source.mp4'
        self.source.write_bytes(b'owned mocked media')
        self.video = {'index': 0, 'width': 2, 'height': 2}
        self.time = Timeline(F(1, 10), tuple(Frame(n, 1) for n in range(20)))
        self.tools = Toolchain(Path('ffmpeg'), Path('ffprobe'))
        self.commands, self.paths = [], []
        self.gate = patch('cheating_racer_detect.media.validate_container')
        self.gate.start()
        self.addCleanup(self.gate.stop)
        self.runner = patch.object(pixels, 'run_tool', side_effect=self.decode)
        self.runner.start()
        self.addCleanup(self.runner.stop)

    def decode(self, args, **kwargs):
        self.commands.append(args)
        first, last = [int(n.rstrip(')')) for n in args[args.index('-vf')+1].split('\\,')[1:]]
        path = Path(args[-1])
        self.paths.append(path)
        path.write_bytes(b''.join(bytes([n])*12 for n in range(first, last+1)))

    def window(self, first=2, count=6, complete=False, **kwargs):
        return pixels.pixel_window(self.source, kwargs.pop('tools', self.tools), self.video,
                                   kwargs.pop('timeline', self.time), first, count, complete=complete)

    def read(self, first=2, count=6, **kwargs):
        with self.window(first, count, **kwargs) as stream:
            self.assertFalse(stream.writable())
            return stream.read(count*12)

    def test_contained_range_readonly_reuse_and_outer_cleanup(self):
        with source_session(pixel_parent=self.root) as job:
            self.assertEqual(self.read(2, 6, complete=True), b''.join(bytes([n])*12 for n in range(2, 8)))
            self.assertEqual(self.read(4, 2), bytes([4])*12+bytes([5])*12)
            self.assertEqual(len(self.commands), 1)
            self.assertNotIn('-frames:v', self.commands[0])
            self.assertIsNotNone(job.pixels)
        self.assertIsNone(job.pixels)
        self.assertFalse(self.paths[0].parent.exists())

    def test_review_miss_ordinal_bound_and_eof_request_not_reused(self):
        with source_session(pixel_parent=self.root):
            self.read()
            self.assertEqual(self.commands[0][self.commands[0].index('-frames:v')+1], '6')
            self.read(4, 2, complete=True)
            self.assertEqual(len(self.commands), 2)
            self.assertNotIn('-frames:v', self.commands[1])

    def test_no_reuse_no_scope_and_outside_range_decode_independently(self):
        with source_session(reuse=False, pixel_parent=self.root):
            self.read()
            self.read(4, 2)
            self.assertTrue(all(not p.exists() for p in self.paths))
        self.read()
        with source_session(pixel_parent=self.root):
            self.read()
            self.read(7, 4)
            self.read(4, 2)
        self.assertEqual(len(self.commands), 5)
        self.assertTrue(all(not p.exists() for p in self.paths))

    def test_other_clock_and_tools_never_hit(self):
        with source_session(pixel_parent=self.root):
            self.read()
            self.read(timeline=Timeline(F(1, 20), self.time.frames))
            self.read(tools=Toolchain(Path('other'), Path('ffprobe')))
        self.assertEqual(len(self.commands), 3)

    def test_failed_nested_job_clears_cache_and_retry_decodes(self):
        with source_session(pixel_parent=self.root) as job:
            self.read()
            with self.assertRaises(KeyboardInterrupt):
                with source_session():
                    raise KeyboardInterrupt()
            self.assertIsNone(job.pixels)
            self.assertFalse(self.paths[0].exists())
            self.read()
        self.assertEqual(len(self.commands), 2)

    def test_consumer_cancel_closes_handle_clears_cache_and_retries(self):
        with source_session(pixel_parent=self.root) as job:
            self.read()
            with self.assertRaises(KeyboardInterrupt):
                with self.window(4, 2) as stream:
                    raise KeyboardInterrupt()
            self.assertTrue(stream.closed)
            self.assertIsNone(job.pixels)
            self.read()
        self.assertEqual(len(self.commands), 2)

    def test_decode_failure_short_data_and_cancel_cleanup(self):
        for error in (AppError('EXTRACTION_FAILED', 'mock'), KeyboardInterrupt(), OSError('mock')):
            def fail(args, **kwargs):
                self.paths.append(Path(args[-1]))
                Path(args[-1]).write_bytes(b'partial')
                raise error
            with patch.object(pixels, 'run_tool', side_effect=fail), self.assertRaises(type(error)):
                self.read()
            self.assertFalse(self.paths[-1].parent.exists())
        def short(args, **kwargs):
            self.paths.append(Path(args[-1]))
            Path(args[-1]).write_bytes(b'partial')
        with patch.object(pixels, 'run_tool', side_effect=short), self.assertRaises(AppError) as caught:
            self.read()
        self.assertEqual(caught.exception.code, 'OUTPUT_INVALID')
        self.assertFalse(self.paths[-1].parent.exists())
        self.read()

    def test_source_change_during_decode_consumption_and_hit_rejected(self):
        with source_session(pixel_parent=self.root) as job:
            self.read()
            self.source.write_bytes(b'changed owned source')
            with self.assertRaises(AppError) as caught:
                self.read()
            self.assertEqual(caught.exception.code, 'INPUT_CHANGED')
            self.assertTrue(job.failed)
        with self.assertRaises(AppError) as caught:
            with self.window() as stream:
                self.source.write_bytes(b'new owned source again')
        self.assertEqual(caught.exception.code, 'INPUT_CHANGED')

    def test_owned_raw_change_rejected_and_new_job_retry(self):
        with source_session(pixel_parent=self.root) as job:
            self.read()
            job.pixels.raw.write_bytes(b'changed raw')
            with self.assertRaises(AppError) as caught:
                self.read()
            self.assertEqual(caught.exception.code, 'OUTPUT_INVALID')
            self.assertIsNone(job.pixels)
            self.read()

    def test_decode_source_change_rejected_before_pixels_are_returned(self):
        def mutate(args, **kwargs):
            self.decode(args, **kwargs)
            self.source.write_bytes(b'changed by owned decoder test')
        with patch.object(pixels, 'run_tool', side_effect=mutate), self.assertRaises(AppError) as caught:
            self.read()
        self.assertEqual(caught.exception.code, 'INPUT_CHANGED')
        self.assertFalse(self.paths[-1].parent.exists())

    def test_uncertain_directory_owner_is_preserved_not_deleted(self):
        with source_session(pixel_parent=self.root):
            self.read()
            path = self.paths[-1].parent
            actual = pixels._identity
            def identity(value):
                result = actual(value)
                return (result[0], result[1]+1, *result[2:]) if value == path else result
            with patch.object(pixels, '_identity', side_effect=identity):
                with self.assertRaises(AppError) as caught:
                    self.read()
            self.assertEqual(caught.exception.code, 'CLEANUP_FAILED')
            self.assertTrue(path.exists())
        # Outer test TemporaryDirectory owns and removes the preserved fixture.

    def test_expired_context_and_other_thread_do_not_reuse(self):
        with source_session(pixel_parent=self.root):
            self.read()
            context = copy_context()
            thread = threading.Thread(target=lambda: context.run(self.read))
            thread.start()
            thread.join()
        context.run(self.read)
        self.assertEqual(len(self.commands), 3)

    def test_invalid_range_remote_source_and_axis_before_decode(self):
        for first, count in ((-1, 2), (0, 0), (0, 121), (18, 3), (True, 1)):
            with self.assertRaises(AppError):
                self.read(first, count)
        self.video['width'] = 1025
        with self.assertRaises(AppError):
            self.read()
        self.assertEqual(self.commands, [])
        with self.assertRaises(AppError):
            with pixels.pixel_window('https://example.invalid/x.mp4', self.tools, self.video, self.time, 0, 1):
                self.fail('must reject')


if __name__ == '__main__':
    unittest.main()
