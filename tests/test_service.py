"""Boundary and transaction tests. Media correctness uses separate real FFmpeg tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from decimal import Decimal
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.paths import local_path
from cheating_racer_detect.service import create_clip
from cheating_racer_detect.tools import Toolchain, discover_tools, run_tool


class PathBoundaryTests(unittest.TestCase):
    def test_remote_and_special_inputs_do_not_reach_filesystem(self):
        for value in ('https://example.invalid/video.mp4', r'\\server\share\video.mp4',
                      '//server/share/video.mp4', r'\\?\C:\video.mp4', 'C:video.mp4',
                      'video.mp4:secret', '../video.mp4', ''):
            with self.subTest(value=value), patch.object(Path, 'lstat') as inspect:
                with self.assertRaises(AppError):
                    local_path(value)
                inspect.assert_not_called()

    @unittest.skipUnless(os.name == 'nt', 'Windows drive type gate')
    def test_network_drive_rejected_before_file_open(self):
        with patch('cheating_racer_detect.paths._drive_type', return_value=4), patch.object(Path, 'lstat') as inspect:
            with self.assertRaises(AppError):
                local_path('Z:\\video.mp4')
            inspect.assert_not_called()

    @unittest.skipUnless(os.name == 'nt', 'Windows junction fixture')
    def test_real_junction_rejected(self):
        with tempfile.TemporaryDirectory(prefix='crd-junction-') as temp:
            base = Path(temp)
            target = base / 'local-target'
            target.mkdir()
            junction = base / 'junction'
            # No shell interpolation: fixed PowerShell source with paths via argv.
            completed = subprocess.run(['pwsh', '-NoProfile', '-Command',
                "$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path $env:CRD_TEST_JUNCTION -Target $env:CRD_TEST_TARGET | Out-Null"],
                env={**os.environ, 'CRD_TEST_JUNCTION': str(junction), 'CRD_TEST_TARGET': str(target)}, capture_output=True)
            self.assertEqual(completed.returncode, 0, 'Junction fixture creation failed')
            try:
                with self.assertRaises(AppError):
                    local_path(junction / 'input.mp4')
            finally:
                junction.rmdir()  # remove our junction, not the target
            self.assertTrue(target.is_dir())

    def test_remote_tool_directory_not_probed(self):
        with patch.object(Path, 'is_file') as probe:
            with self.assertRaises(AppError):
                discover_tools(Path(r'\\server\tools'))
            probe.assert_not_called()


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='crd-transaction-')
        self.base = Path(self.directory.name)
        self.source = self.base / 'source.mp4'
        self.source.write_bytes(b'test-only source, media engine mocked')
        self.output = self.base / 'result'
        self.tools = Toolchain(Path('ffmpeg'), Path('ffprobe'))
        self.doctor = patch('cheating_racer_detect.service.doctor', return_value={'ffmpeg': 'fixture', 'ffprobe': 'fixture'})
        self.doctor.start()
        self.addCleanup(self.doctor.stop)
        self.addCleanup(self.directory.cleanup)

    def render(self, source, destination, start, end, toolchain):
        destination.write_bytes(b'fixture clip')
        return {'verification': {'fixture': True}}

    def invoke(self):
        return create_clip(self.source, Decimal('0'), Decimal('1'), self.output, self.tools)

    def assert_clean(self):
        self.assertEqual(list(self.base.glob('.crd-*.partial')), [])

    def test_publish_complete_pair_and_source_hash(self):
        original = self.source.read_bytes()
        with patch('cheating_racer_detect.media.extract', side_effect=self.render):
            result = self.invoke()
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(json.loads((self.output / 'result.json').read_text(encoding='utf-8')), result)
        self.assertEqual((self.output / 'clip.mp4').read_bytes(), b'fixture clip')
        self.assertNotIn(str(self.base), json.dumps(result, ensure_ascii=False))
        self.assert_clean()

    def test_existing_output_is_untouched(self):
        self.output.mkdir()
        marker = self.output / 'user.txt'
        marker.write_text('keep', encoding='utf-8')
        with patch('cheating_racer_detect.media.extract') as render, self.assertRaises(AppError) as caught:
            self.invoke()
        self.assertEqual(caught.exception.code, 'OUTPUT_EXISTS')
        self.assertEqual(marker.read_text(), 'keep')
        render.assert_not_called()
        self.assert_clean()

    def test_output_created_during_render_is_not_replaced(self):
        def compete(*args):
            self.output.mkdir()
            (self.output / 'user.txt').write_text('keep', encoding='utf-8')
            return self.render(*args)
        with patch('cheating_racer_detect.media.extract', side_effect=compete), self.assertRaises(AppError) as caught:
            self.invoke()
        self.assertEqual(caught.exception.code, 'OUTPUT_EXISTS')
        self.assertEqual((self.output / 'user.txt').read_text(), 'keep')
        self.assertFalse((self.output / 'clip.mp4').exists())
        self.assert_clean()

    def test_original_change_prevents_publish(self):
        def mutate(*args):
            self.source.write_bytes(b'intentional fixture mutation')
            return self.render(*args)
        with patch('cheating_racer_detect.media.extract', side_effect=mutate), self.assertRaises(AppError) as caught:
            self.invoke()
        self.assertEqual(caught.exception.code, 'INPUT_CHANGED')
        self.assertFalse(self.output.exists())
        self.assert_clean()

    def test_failure_and_keyboard_interrupt_cleanup_and_retry(self):
        for failure in (AppError('EXTRACTION_FAILED', 'fixture'), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), patch('cheating_racer_detect.media.extract', side_effect=failure):
                with self.assertRaises(type(failure)):
                    self.invoke()
                self.assertFalse(self.output.exists())
                self.assert_clean()
        with patch('cheating_racer_detect.media.extract', side_effect=self.render):
            self.invoke()
        self.assertTrue(self.output.exists())

    def test_manifest_write_failure_cleanup(self):
        original_open = Path.open
        def fail_manifest(path, *args, **kwargs):
            if path.name == 'result.json':
                raise OSError('Injected storage-full error')
            return original_open(path, *args, **kwargs)
        with patch('cheating_racer_detect.media.extract', side_effect=self.render), patch.object(Path, 'open', fail_manifest):
            with self.assertRaises(AppError) as caught:
                self.invoke()
        self.assertEqual(caught.exception.code, 'IO_ERROR')
        self.assertFalse(self.output.exists())
        self.assert_clean()

    def test_rename_failure_cleanup(self):
        with patch('cheating_racer_detect.media.extract', side_effect=self.render), patch.object(Path, 'rename', side_effect=PermissionError):
            with self.assertRaises(AppError):
                self.invoke()
        self.assertFalse(self.output.exists())
        self.assert_clean()

    def test_other_job_partial_is_preserved(self):
        other = self.base / '.crd-user.partial'
        other.mkdir()
        with patch('cheating_racer_detect.media.extract', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke()
        self.assertEqual(list(self.base.glob('.crd-*.partial')), [other])

    def test_parent_replacement_preserves_unowned_directory(self):
        parent = self.base / 'output-parent'
        parent.mkdir()
        self.output = parent / 'result'
        moved = self.base / 'moved-parent'
        sentinel = None
        def replace_parent(source, destination, start, end, tools):
            nonlocal sentinel
            parent.rename(moved)
            parent.mkdir()
            foreign_partial = parent / destination.parent.name
            foreign_partial.mkdir()
            sentinel = foreign_partial / 'other-job.txt'
            sentinel.write_text('do not delete', encoding='utf-8')
            return {}
        with patch('cheating_racer_detect.media.extract', side_effect=replace_parent), self.assertRaises(AppError) as caught:
            self.invoke()
        self.assertEqual(caught.exception.code, 'CLEANUP_FAILED')
        self.assertEqual(sentinel.read_text(), 'do not delete')
        self.assertEqual(len(list(moved.glob('.crd-*.partial'))), 1)
        self.assertFalse(self.output.exists())

    def test_temporary_directory_replacement_is_not_deleted(self):
        sentinel = None
        def replace_temp(source, destination, start, end, tools):
            nonlocal sentinel
            original = destination.parent
            original.rename(original.with_name('moved-owned-partial'))
            original.mkdir()
            sentinel = original / 'other-job.txt'
            sentinel.write_text('preserve', encoding='utf-8')
            return {}
        with patch('cheating_racer_detect.media.extract', side_effect=replace_temp), self.assertRaises(AppError) as caught:
            self.invoke()
        self.assertEqual(caught.exception.code, 'CLEANUP_FAILED')
        self.assertEqual(sentinel.read_text(), 'preserve')
        self.assertFalse(self.output.exists())

    def test_moved_parent_without_replacement_reports_unconfirmed_cleanup(self):
        parent = self.base / 'parent'
        parent.mkdir()
        self.output = parent / 'result'
        moved = self.base / 'moved-parent'
        def move_parent(*args):
            parent.rename(moved)
            return {}
        with patch('cheating_racer_detect.media.extract', side_effect=move_parent), self.assertRaises(AppError) as caught:
            self.invoke()
        self.assertEqual(caught.exception.code, 'CLEANUP_FAILED')
        self.assertEqual(len(list(moved.glob('.crd-*.partial'))), 1)

    def test_invalid_times_do_not_reach_tools(self):
        for start, end in (('NaN', '1'), ('0', 'Infinity'), ('-1', '1'), ('2', '1'), ('1', '1')):
            with self.subTest(start=start, end=end), patch('cheating_racer_detect.service.doctor') as doctor:
                with self.assertRaises(AppError) as caught:
                    create_clip(self.source, Decimal(start), Decimal(end), self.output, self.tools)
                self.assertEqual(caught.exception.code, 'INVALID_RANGE')
                doctor.assert_not_called()

    def test_remote_media_never_reaches_tools(self):
        with patch('cheating_racer_detect.service.doctor') as doctor, self.assertRaises(AppError):
            create_clip(r'\\server\share\video.mp4', Decimal(0), Decimal(1), self.output, self.tools)
        doctor.assert_not_called()


class ProcessTests(unittest.TestCase):
    def test_child_timeout_is_reaped(self):
        with tempfile.TemporaryDirectory(prefix='crd-child-') as temp:
            marker = Path(temp) / 'should-not-exist'
            code = 'import pathlib,sys,time;time.sleep(1.5);pathlib.Path(sys.argv[1]).write_text("alive")'
            with self.assertRaises(AppError):
                run_tool([sys.executable, '-c', code, marker], timeout=0.05)
            time.sleep(1.6)
            self.assertFalse(marker.exists())

    def test_nonzero_tool_does_not_leak_private_stderr(self):
        with self.assertRaises(AppError) as caught:
            run_tool([sys.executable, '-c', 'import sys;sys.stderr.write("PRIVATE_MARKER");sys.exit(1)'])
        self.assertNotIn('PRIVATE_MARKER', str(caught.exception))

    def test_keyboard_interrupt_reaps_owned_process(self):
        class FakeProcess:
            returncode = None
            killed = False
            calls = 0
            def communicate(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise KeyboardInterrupt
                return '', ''
            def poll(self):
                return self.returncode
            def kill(self):
                self.killed = True
                self.returncode = -1
        process = FakeProcess()
        with patch('cheating_racer_detect.tools.subprocess.Popen', return_value=process), self.assertRaises(KeyboardInterrupt):
            run_tool(['fixture'])
        self.assertTrue(process.killed)
        self.assertEqual(process.calls, 2)


if __name__ == '__main__':
    unittest.main()
