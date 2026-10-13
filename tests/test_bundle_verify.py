from pathlib import Path
from fractions import Fraction as F
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events.verify import Files, verify_batch
from scripts.verify_bundle import main
from cheating_racer_detect.media import _shift_pts
from tests.cancel_driver import require_children_exited


class SavedFiles(unittest.TestCase):
    def test_windows_child_exit_diagnostics_do_not_relax_condition(self):
        from unittest.mock import Mock
        api = Mock()
        api.WaitForSingleObject.return_value = 0
        require_children_exited(api, {1: (10, 'ffmpeg.exe')}, 'retry')
        api.WaitForSingleObject.assert_called_once_with(10, 3000)
        for status in (258, 0xffffffff):
            api.WaitForSingleObject.return_value = status
            with self.assertRaisesRegex(AssertionError, 'wait_status='+str(status)):
                require_children_exited(api, {1: (10, 'ffmpeg.exe')}, 'retry')
    def test_pts_expression_cancels_integral_ticks_before_fraction(self):
        self.assertEqual(_shift_pts(F(5), F(1, 12288)), 'PTS-(61440)')
        self.assertEqual(_shift_pts(F(16, 3), F(1, 10)), 'PTS-(53)-(1/3)')
        self.assertEqual(_shift_pts(F(-1, 3), F(1, 10)), 'PTS-(-4)-(2/3)')

    def test_bounded_duplicate_nonfinite_unicode_and_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for data in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'\xff', b' '*1_048_577,
                         b'['*2000+b']'*2000):
                (root/'batch.json').write_bytes(data)
                with self.assertRaises(AppError):
                    verify_batch(root)

    def test_missing_file_changed_content_and_path_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                Files(root).file('missing.json')
            path = root/'record.json'
            path.write_text('{}')
            files = Files(root)
            files.json('record.json')
            path.write_text('{"x":1}')
            with self.assertRaises(ValueError):
                files.json('record.json')
            with self.assertRaises(ValueError):
                files.finish()
            for name in ('../record.json', 'https://example.org/x.json', 'x:stream'):
                with self.assertRaises(AppError):
                    Files(root).file(name)

    def test_cli_error_and_cancel_are_sanitized(self):
        with patch('scripts.verify_bundle.verify_batch', side_effect=AppError('BUNDLE_INVALID', 'private path')), patch('builtins.print') as output:
            self.assertEqual(main(['--bundle', 'missing']), 1)
            self.assertNotIn('private path', output.call_args.args[0])
        with patch('scripts.verify_bundle.verify_batch', side_effect=KeyboardInterrupt), patch('builtins.print'):
            self.assertEqual(main(['--bundle', 'missing']), 130)


if __name__ == '__main__':
    unittest.main()
