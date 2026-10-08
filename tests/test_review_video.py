"""ML-free transaction guards; mocked media is not an encoding verification."""

from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.review import video
from cheating_racer_detect.tools import Toolchain


class Transaction(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='crd-review-unit-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root/'source.mp4'
        self.source.write_bytes(b'unit fixture; media mocked')
        self.output = self.root/'result'
        self.tools = Toolchain(Path('ffmpeg'), Path('ffprobe'))

    def fixture(self, source, start, end, stage, observe, tools, source_hash):
        (stage/'original').mkdir()
        (stage/'original'/'clip.mp4').write_bytes(b'original fixture')
        (stage/'review.mp4').write_bytes(b'review fixture')
        return {'schema': 'unit-only', 'status': 'complete', 'value': 'owned'}

    def invoke(self, start='0', end='1', observe=lambda f, p: {}):
        return video.export_review(self.source, start, end, self.output, observe, self.tools)

    def clean(self):
        self.assertEqual(list(self.root.glob('.crd-review-*.partial')), [])

    def test_success_atomic_pair_and_source_preserved(self):
        before = self.source.read_bytes()
        with patch.object(video, 'build', side_effect=self.fixture):
            result = self.invoke()
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(result, json.loads((self.output/'review.json').read_text(encoding='utf-8')))
        self.assertEqual((self.output/'original'/'clip.mp4').read_bytes(), b'original fixture')
        self.clean()

    def test_invalid_ranges_callback_and_existing_before_work(self):
        with patch.object(video, 'build') as build:
            for start, end in (('nan', 1), (0, 'inf'), (-1, 2), (1, 1), ('bad', 2)):
                with self.assertRaises(AppError):
                    self.invoke(start, end)
            with self.assertRaises(AppError):
                self.invoke(observe=None)
            self.output.mkdir()
            marker = self.output/'user.txt'
            marker.write_text('keep')
            with self.assertRaises(AppError) as caught:
                self.invoke()
            self.assertEqual(caught.exception.code, 'OUTPUT_EXISTS')
            self.assertEqual(marker.read_text(), 'keep')
            build.assert_not_called()
        self.clean()

    def test_remote_inputs_rejected_before_file_or_tool_access(self):
        for path in ('https://example.invalid/a.mp4', r'\\server\a.mp4', '../a.mp4', 'C:video.mp4'):
            with patch.object(video, 'build') as build, patch.object(video, 'discover_tools') as discover:
                with self.assertRaises(AppError):
                    video.export_review(path, 0, 1, self.output, lambda f, p: {})
                build.assert_not_called()
                discover.assert_not_called()

    def test_error_cancel_and_encode_disk_failure_cleanup_retry(self):
        for error in (RuntimeError, KeyboardInterrupt, OSError, ValueError):
            def fail(*args):
                self.fixture(*args)
                raise error('injected failure')
            with patch.object(video, 'build', side_effect=fail):
                with self.assertRaises((RuntimeError, KeyboardInterrupt, AppError)):
                    self.invoke()
            self.assertFalse(self.output.exists())
            self.clean()
        with patch.object(video, 'build', side_effect=self.fixture):
            self.invoke()

    def test_invalid_json_does_not_publish(self):
        with patch.object(video, 'build', return_value={'score': float('nan')}):
            with self.assertRaises(AppError) as caught:
                self.invoke()
        self.assertEqual(caught.exception.code, 'INVALID_REVIEW')
        self.assertFalse(self.output.exists())
        self.clean()

    def test_source_change_rejected_and_changed_source_not_deleted(self):
        def change(*args):
            result = self.fixture(*args)
            self.source.write_bytes(b'changed by fixture')
            return result
        with patch.object(video, 'build', side_effect=change):
            with self.assertRaises(AppError) as caught:
                self.invoke()
        self.assertEqual(caught.exception.code, 'INPUT_CHANGED')
        self.assertEqual(self.source.read_bytes(), b'changed by fixture')
        self.assertFalse(self.output.exists())
        self.clean()

    def test_publish_race_does_not_replace_other_result(self):
        def race(*args):
            result = self.fixture(*args)
            self.output.mkdir()
            (self.output/'user.txt').write_text('other job')
            return result
        with patch.object(video, 'build', side_effect=race):
            with self.assertRaises(AppError) as caught:
                self.invoke()
        self.assertEqual(caught.exception.code, 'OUTPUT_EXISTS')
        self.assertEqual(list(p.name for p in self.output.iterdir()), ['user.txt'])
        self.clean()

    def test_parent_change_leaves_unowned_stage(self):
        original = video._identity
        seen = 0
        def identity(path):
            nonlocal seen
            value = original(path)
            if path == self.root:
                seen += 1
                if seen > 1:
                    return value[0], value[1]+1, *value[2:]
            return value
        with patch.object(video, 'build', side_effect=self.fixture), patch.object(video, '_identity', side_effect=identity):
            with self.assertRaises(AppError) as caught:
                self.invoke()
        self.assertEqual(caught.exception.code, 'CLEANUP_FAILED')
        self.assertEqual(len(list(self.root.glob('.crd-review-*.partial'))), 1)
        self.assertFalse(self.output.exists())
        # TemporaryDirectory owns the entire unit fixture, not a user workspace.

    def test_timestamp_lookup_uses_exact_ticks(self):
        self.assertEqual(video.timestamp_expression([0]), '0')
        self.assertEqual(video.timestamp_expression([0, 3072, 7168]), 'if(eq(N,0),0,if(eq(N,1),3072,7168))')
        self.assertEqual(video.interval(Decimal('.1'), Decimal('.2')), (Decimal('.1'), Decimal('.2')))


if __name__ == '__main__':
    unittest.main()
