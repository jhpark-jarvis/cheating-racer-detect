"""Explicit storage/codec faults and real loopback observation (not a firewall audit)."""

from contextlib import redirect_stderr, redirect_stdout
import errno
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.cli import main
from cheating_racer_detect.errors import AppError
from cheating_racer_detect.tools import discover_tools
from tests.fixtures import generate, sha256
from tests.test_media_unit import container


class RemainingQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tools = discover_tools()
        except AppError as error:
            if error.code == 'DEPENDENCY_MISSING':
                raise unittest.SkipTest('NOT_RUN: real FFmpeg / ffprobe unavailable') from error
            raise
        cls.temporary = tempfile.TemporaryDirectory(prefix='crd-remaining-qa-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.media = generate(cls.tools, cls.root / 'fixtures')

    def invoke(self, output):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(['clip', '--input', str(self.media['audio']), '--start', '0.35',
                         '--end', '2.35', '--output', str(output),
                         '--ffmpeg-dir', str(self.tools.ffmpeg.parent)])
        return code, stdout.getvalue(), stderr.getvalue()

    def assert_failure_clean(self, output, code, result, original):
        exit_code, stdout, stderr = result
        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, '')
        self.assertEqual(json.loads(stderr)['code'], code)
        self.assertNotIn(str(self.root), stderr)
        self.assertEqual(sha256(self.media['audio']), original)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob('.crd-*.partial')), [])

    def test_enospc_at_each_transaction_stage_and_retry(self):
        original = sha256(self.media['audio'])
        real_open = Path.open

        def fail_manifest(path, *args, **kwargs):
            if path.name == 'result.json':
                raise OSError(errno.ENOSPC, 'injected no space')
            return real_open(path, *args, **kwargs)

        def fail_media(source, destination, *args):
            destination.write_bytes(b'incomplete test-owned clip')
            raise OSError(errno.ENOSPC, 'injected no space')

        factories = {
            'mkdir': lambda: patch('cheating_racer_detect.service.tempfile.mkdtemp',
                                    side_effect=OSError(errno.ENOSPC, 'injected no space')),
            'media': lambda: patch('cheating_racer_detect.media.extract', side_effect=fail_media),
            'manifest': lambda: patch.object(Path, 'open', fail_manifest),
            'sync': lambda: patch('cheating_racer_detect.service.os.fsync',
                                   side_effect=OSError(errno.ENOSPC, 'injected no space')),
            'publish': lambda: patch.object(Path, 'rename',
                                            side_effect=OSError(errno.ENOSPC, 'injected no space')),
        }
        for name, factory in factories.items():
            with self.subTest(stage=name):
                output = self.root / ('space-' + name)
                with factory():
                    result = self.invoke(output)
                self.assert_failure_clean(output, 'IO_ERROR', result, original)
                self.assertEqual(self.invoke(output)[0], 0)
                self.assertEqual(json.loads((output / 'result.json').read_text('utf-8'))['status'], 'complete')
                self.assertEqual(sha256(self.media['audio']), original)

    def test_each_missing_codec_returns_dependency_error_and_retry(self):
        original = sha256(self.media['audio'])
        for missing in ('libx264_encoder', 'aac_encoder', 'h264_decoder', 'aac_decoder'):
            with self.subTest(missing=missing):
                output = self.root / missing

                def listing(args, **kwargs):
                    if '-version' in args:
                        return 'fixture version\n'
                    if '-encoders' in args:
                        return ''.join(' V..... ' + name + ' fixture\n' for name in ('libx264', 'aac')
                                       if name + '_encoder' != missing)
                    if '-decoders' in args:
                        return ''.join(' V..... ' + name + ' fixture\n' for name in ('h264', 'aac')
                                       if name + '_decoder' != missing)
                    raise AssertionError('Unexpected tool invocation during codec validation')

                with patch('cheating_racer_detect.tools.run_tool', side_effect=listing), \
                        patch('cheating_racer_detect.media.extract') as extract:
                    result = self.invoke(output)
                extract.assert_not_called()
                self.assert_failure_clean(output, 'CODEC_UNAVAILABLE', result, original)
                self.assertEqual(self.invoke(output)[0], 0)

    def test_rejected_network_inputs_make_no_connection_to_observed_endpoint(self):
        # The listener stays active throughout each real CLI invocation. Positive
        # controls before AND after establish that "zero" is not a dead observer.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen(16)
            listener.settimeout(0.2)
            endpoint = listener.getsockname()
            url = f'http://127.0.0.1:{endpoint[1]}/fixture.mp4'

            def positive_control():
                with socket.create_connection(endpoint, timeout=2):
                    connection, _ = listener.accept()
                    connection.close()

            playlist = self.root / 'playlist.mp4'
            playlist.write_text('#EXTM3U\n' + url + '\n', encoding='utf-8')
            reference = self.root / 'external-reference.mp4'
            reference.write_bytes(container(b'\0' * 4 + url.encode('ascii') + b'\0'))
            for index, source in enumerate((url, str(playlist), str(reference))):
                with self.subTest(case=index):
                    positive_control()
                    output = self.root / f'network-{index}'
                    completed = subprocess.run([
                        sys.executable, '-m', 'cheating_racer_detect', 'clip',
                        '--input', source, '--start', '0', '--end', '1', '--output', str(output),
                        '--ffmpeg-dir', str(self.tools.ffmpeg.parent),
                    ], capture_output=True, timeout=15)
                    self.assertEqual(completed.returncode, 2)
                    self.assertEqual(completed.stdout, b'')
                    self.assertIn(json.loads(completed.stderr)['code'], ('INVALID_ARGUMENT', 'UNSUPPORTED_MEDIA'))
                    self.assertFalse(output.exists())
                    self.assertEqual(list(self.root.glob('.crd-*.partial')), [])
                    try:
                        connection, _ = listener.accept()
                    except socket.timeout:
                        pass
                    else:
                        connection.close()
                        self.fail('Product connected to the observed loopback endpoint')
                    positive_control()


class ManualFixtureSafety(unittest.TestCase):
    def test_existing_review_directory_is_never_modified(self):
        from scripts.prepare_manual_qa import prepare

        with tempfile.TemporaryDirectory(prefix='crd-manual-safety-') as temporary:
            root = Path(temporary)
            marker = root / 'keep.txt'
            marker.write_text('existing review data', encoding='utf-8')
            with patch('scripts.prepare_manual_qa.discover_tools') as discover:
                with self.assertRaises(FileExistsError):
                    prepare(root)
            discover.assert_not_called()
            self.assertEqual(marker.read_text('utf-8'), 'existing review data')
            self.assertEqual(list(root.iterdir()), [marker])


if __name__ == '__main__':
    unittest.main()
