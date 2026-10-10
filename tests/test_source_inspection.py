"""Stdlib inspection lifetime/ownership guards; media calls are mocked."""

from contextlib import ExitStack
from contextvars import copy_context
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

from cheating_racer_detect import media, source
from cheating_racer_detect.errors import AppError
from cheating_racer_detect.tools import Toolchain


class Inspection(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='crd-inspection-unit-')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root/'source.mp4'
        self.path.write_bytes(b'owned mock fixture')
        self.tools = Toolchain(Path('ffmpeg'), Path('ffprobe'))
        self.video = {'index': 0, 'width': 96, 'height': 64, 'side_data_list': [{'rotation': 90}]}
        self.audio = {'index': 1, 'nested': {'sample_rate': 48000}}
        self.timeline = media.Timeline(F(1, 10), (media.Frame(0, 1), media.Frame(1, 1)))
        stack = self.enterContext(ExitStack())
        self.container = stack.enter_context(patch.object(media, 'validate_container'))
        self.probe = stack.enter_context(patch.object(media, '_probe', return_value={}))
        stack.enter_context(patch.object(media, '_profile', return_value=(self.video, self.audio)))
        self.time = stack.enter_context(patch.object(media, '_timeline', return_value=self.timeline))

    def inspect(self, path=None, tools=None, **kwargs):
        return source.inspect_source(path or self.path, tools or self.tools, **kwargs)

    def test_nested_scope_one_inspection_owned_profiles_frozen_timeline(self):
        with source.source_session():
            video, audio, timeline = self.inspect()
            video['side_data_list'][0]['rotation'] = 0
            audio['nested']['sample_rate'] = 1
            with source.source_session():
                v2, a2, t2 = self.inspect()
            self.assertEqual(v2['side_data_list'][0]['rotation'], 90)
            self.assertEqual(a2['nested']['sample_rate'], 48000)
            self.assertIs(timeline, t2)
            with self.assertRaises(FrozenInstanceError):
                timeline.frames[0].pts = 2
        self.assertEqual(self.probe.call_count, 1)
        self.assertEqual(self.time.call_count, 1)
        self.assertEqual(self.container.call_count, 2)

    def test_separate_jobs_and_no_scope_do_not_share(self):
        for _ in range(2):
            with source.source_session():
                self.inspect()
                self.inspect()
        self.inspect()
        self.inspect()
        self.assertEqual(self.probe.call_count, 4)

    def test_disabled_outer_scope_is_inherited(self):
        with source.source_session(reuse=False):
            self.inspect()
            with source.source_session():
                self.inspect()
        self.assertEqual(self.probe.call_count, 2)

    def test_other_source_and_toolchain_not_cached_or_trusted(self):
        other = self.root/'other.mp4'
        other.write_bytes(b'other owned fixture')
        tools = Toolchain(Path('other-ffmpeg'), Path('other-ffprobe'))
        with source.source_session():
            self.inspect()
            self.inspect(other)
            self.inspect(other)
            self.inspect(tools=tools)
            self.inspect(tools=tools)
            self.inspect()
        self.assertEqual(self.probe.call_count, 5)

    def test_change_on_reuse_poisoned_until_new_scope(self):
        with source.source_session():
            self.inspect()
            self.path.write_bytes(b'changed owned fixture with different size')
            for _ in range(2):
                with self.assertRaises(AppError) as error:
                    self.inspect()
                self.assertEqual(error.exception.code, 'INPUT_CHANGED')
        self.assertEqual(self.probe.call_count, 1)
        with source.source_session():
            self.inspect()
        self.assertEqual(self.probe.call_count, 2)

    def test_change_during_first_probe_not_saved(self):
        def change(*args):
            self.path.write_bytes(b'changed during probe; owned')
            return {}
        with source.source_session() as job, patch.object(media, '_probe', side_effect=change):
            with self.assertRaises(AppError) as error:
                self.inspect()
            self.assertEqual(error.exception.code, 'INPUT_CHANGED')
            self.assertIsNone(job.snapshot)

    def test_local_path_gate_runs_on_every_hit(self):
        actual = source.local_path
        with source.source_session():
            self.inspect()
            with patch.object(source, 'local_path', side_effect=AppError('INVALID_ARGUMENT', 'mock reparse')) as gate:
                with self.assertRaises(AppError):
                    self.inspect()
                gate.assert_called_once_with(self.path)
        self.assertEqual(actual(self.path), self.path)

    def test_container_reference_gate_runs_even_with_matching_fingerprint(self):
        with source.source_session():
            self.inspect()
            with patch.object(media, 'validate_container', side_effect=AppError('UNSUPPORTED_MEDIA', 'indirect mock')):
                with self.assertRaises(AppError) as error:
                    self.inspect()
                self.assertEqual(error.exception.code, 'UNSUPPORTED_MEDIA')
        self.assertEqual(self.probe.call_count, 1)

    def test_axis_limit_applies_to_cached_and_initial_profiles(self):
        with source.source_session():
            self.inspect()
            with self.assertRaises(AppError) as error:
                self.inspect(max_axis=64)
            self.assertEqual(error.exception.code, 'REVIEW_LIMIT')
        with self.assertRaises(AppError):
            self.inspect(max_axis=64)
        self.assertEqual(self.time.call_count, 1)

    def test_failure_cancel_and_expired_copied_context_clear_scope(self):
        for failure in (RuntimeError, KeyboardInterrupt, OSError):
            with self.assertRaises(failure):
                with source.source_session() as job:
                    self.inspect()
                    copied = copy_context()
                    raise failure('owned unit injection')
            self.assertFalse(job.active)
            self.assertIsNone(job.snapshot)
            copied.run(self.inspect)
            with source.source_session():
                self.inspect()
        self.assertEqual(self.probe.call_count, 9)

    def test_caught_inner_failure_discards_snapshot_before_outer_retry(self):
        with source.source_session() as job:
            self.inspect()
            with self.assertRaises(KeyboardInterrupt):
                with source.source_session():
                    raise KeyboardInterrupt('owned inner cancellation')
            self.assertIsNone(job.snapshot)
            self.inspect()
        self.assertEqual(self.probe.call_count, 2)

    def test_new_thread_does_not_use_inherited_snapshot(self):
        errors = []
        def work():
            try:
                with source.source_session():
                    self.inspect()
            except BaseException as error:
                errors.append(error)
        with source.source_session():
            self.inspect()
            copied = copy_context()
            thread = Thread(target=lambda: copied.run(work))
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.inspect()
        self.assertEqual(errors, [])
        self.assertEqual(self.probe.call_count, 2)

    def test_decorated_job_arguments_result_and_lifetime(self):
        @source.source_job
        def run(first, *, second):
            self.inspect()
            self.inspect()
            return first+second
        self.assertEqual(run(2, second=3), 5)
        self.assertEqual(run(1, second=7), 8)
        self.assertEqual(run.__name__, 'run')
        self.assertEqual(self.probe.call_count, 2)

    def test_malformed_probe_or_cancel_does_not_cache_partial_result(self):
        for failure in (AppError('INVALID_TIMELINE', 'owned mock'), KeyboardInterrupt()):
            with self.assertRaises(type(failure)):
                with source.source_session() as job, patch.object(media, '_timeline', side_effect=failure):
                    self.inspect()
            self.assertIsNone(job.snapshot)
        with source.source_session():
            self.inspect()


if __name__ == '__main__':
    unittest.main()
