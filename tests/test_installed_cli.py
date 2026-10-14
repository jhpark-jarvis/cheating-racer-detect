"""Package-native command contracts, with no model or network dependency."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from cheating_racer_detect.cli import main
from cheating_racer_detect.errors import AppError
from cheating_racer_detect.tools import Toolchain
from scripts.demo_assessment import definition


def invoke(args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


class Commands(unittest.TestCase):
    def test_help_version_and_four_command_help(self):
        for args in (['--help'], ['--version'], ['events', '--help'],
                     *[['events', action, '--help'] for action in ('list', 'replay', 'verify', 'evaluate')]):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                main(args)
            self.assertEqual(result.exception.code, 0)

    def test_argument_errors_are_sanitized_json_exit2(self):
        for args in ([], ['events'], ['events', 'secret-private-command'], ['events', 'list'],
                     ['events', 'evaluate', '--labels', 'private-file', '--secret'],
                     ['events', 'replay', '--input', 'x', '--inputs', 'x', '--output', 'x', '--candidate-index', 'private']):
            code, out, err = invoke(args)
            self.assertEqual((code, out), (2, ''))
            self.assertEqual(json.loads(err)['code'], 'INVALID_ARGUMENT')
            self.assertNotIn('secret-private-command', err)
            self.assertNotIn('private-file', err)

    def test_list_summary_and_explicit_tools_without_native_imports(self):
        tools = Toolchain(Path('ffmpeg'), Path('ffprobe'))
        entry = dict(candidate_index=0, direction='right', start=['0', '1'], crossing=['1', '1'],
                     completion=['1', '2'], signal_state='unknown', left={}, right={}, candidate_sha256='private')
        result = Mock()
        result.report.return_value = {'status': 'complete', 'candidate_count': 1, 'candidates': [entry]}
        with patch('cheating_racer_detect.events.inputs.load_inputs', return_value='inputs'), \
                patch('cheating_racer_detect.events.catalog.export_catalog', return_value=result) as export, \
                patch('cheating_racer_detect.events.cli.discover_tools', return_value=tools) as discover, \
                patch('cheating_racer_detect.events.cli.review_dependencies') as native:
            code, out, err = invoke(['events', 'list', '--input', 'private.mp4', '--inputs', 'inputs.json',
                                     '--output', 'new', '--ffmpeg-dir', 'tools'])
        self.assertEqual((code, err), (0, ''))
        self.assertNotIn('private', out)
        self.assertEqual(json.loads(out)['candidate_count'], 1)
        discover.assert_called_once_with(Path('tools'))
        export.assert_called_once_with(Path('private.mp4'), 'inputs', Path('new'), toolchain=tools)
        native.assert_not_called()

    def test_single_and_catalog_batch_replay_dispatch_preserve_selection(self):
        common = ['events', 'replay', '--input', 'source.mp4', '--inputs', 'inputs.json', '--output', 'new']
        with patch('cheating_racer_detect.events.cli.review_dependencies'), \
                patch('cheating_racer_detect.events.inputs.load_inputs', return_value='inputs'), \
                patch('cheating_racer_detect.events.replay.export_replay', return_value={
                    'status': 'complete', 'candidate_index': 0, 'signal_state': 'unknown'}) as single:
            self.assertEqual(invoke(common+['--candidate-index', '0'])[0], 0)
            self.assertEqual(single.call_args.kwargs['candidate_index'], 0)
        with patch('cheating_racer_detect.events.cli.review_dependencies'), \
                patch('cheating_racer_detect.events.inputs.load_inputs', return_value='inputs'), \
                patch('cheating_racer_detect.events.catalog.load_catalog', return_value='catalog'), \
                patch('cheating_racer_detect.events.replay.export_replay_batch', return_value={
                    'status': 'complete', 'candidate_indices': [1, 0], 'selected_count': 2}) as batch:
            self.assertEqual(invoke(common+['--candidate-indices', '1', '0', '--catalog', 'catalog.json'])[0], 0)
            self.assertEqual(batch.call_args.kwargs['candidate_indices'], [1, 0])
            self.assertEqual(batch.call_args.kwargs['catalog'], 'catalog')

    def test_catalog_requires_batch_and_replay_native_missing_before_output(self):
        common = ['events', 'replay', '--input', 'x', '--inputs', 'x', '--output', 'x', '--candidate-index', '0']
        with patch('cheating_racer_detect.events.cli.review_dependencies') as native:
            code, out, err = invoke(common+['--catalog', 'catalog.json'])
        self.assertEqual((code, out), (2, ''))
        self.assertEqual(json.loads(err)['code'], 'INVALID_ARGUMENT')
        native.assert_not_called()
        with patch('cheating_racer_detect.events.cli.review_dependencies', side_effect=AppError('DEPENDENCY_MISSING', 'safe')), \
                patch('cheating_racer_detect.events.inputs.load_inputs') as load:
            code, out, err = invoke(common)
        self.assertEqual(json.loads(err)['code'], 'DEPENDENCY_MISSING')
        self.assertEqual((code, out), (2, ''))
        load.assert_not_called()

    def test_native_import_errors_are_sanitized(self):
        from cheating_racer_detect.events.cli import review_dependencies
        import builtins
        original = builtins.__import__
        for error in (ImportError('PRIVATE_DLL_PATH'), OSError('PRIVATE_DLL_PATH')):
            def loading(name, *args, **kwargs):
                if name == 'numpy':
                    raise error
                return original(name, *args, **kwargs)
            with patch('builtins.__import__', side_effect=loading), self.assertRaises(AppError) as caught:
                review_dependencies()
            self.assertEqual(caught.exception.code, 'DEPENDENCY_MISSING')
            self.assertNotIn('PRIVATE_DLL_PATH', str(caught.exception))

    def test_verify_source_and_toolchain_forwarded_without_native(self):
        tools = Toolchain(Path('ffmpeg'), Path('ffprobe'))
        with patch('cheating_racer_detect.events.cli.discover_tools', return_value=tools), \
                patch('cheating_racer_detect.events.verify.verify_batch', return_value={'status': 'verified'}) as verify, \
                patch('cheating_racer_detect.events.cli.review_dependencies') as native:
            code, out, err = invoke(['events', 'verify', '--bundle', 'bundle', '--input', 'source.mp4', '--ffmpeg-dir', 'tools'])
        self.assertEqual((code, err), (0, ''))
        verify.assert_called_once_with(Path('bundle'), source=Path('source.mp4'), toolchain=tools)
        native.assert_not_called()

    def test_assessment_actual_stdlib_and_no_media_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'labels.json'
            path.write_text(json.dumps(definition()))
            with patch('cheating_racer_detect.events.cli.discover_tools') as tools:
                code, out, err = invoke(['events', 'evaluate', '--labels', str(path)])
            self.assertEqual((code, err), (0, ''))
            self.assertEqual(json.loads(out)['end_to_end_known_prediction_coverage'], '1/2')
            tools.assert_not_called()
            for data in ('{"x":1,"x":2}', '{"x":1e999}', ' '*1_048_577):
                path.write_text(data)
                code, out, err = invoke(['events', 'evaluate', '--labels', str(path)])
                self.assertEqual((code, out), (2, ''))
                self.assertEqual(json.loads(err)['code'], 'INVALID_ASSESSMENT')

    def test_errors_and_cancel_retry_keep_stderr_protocol(self):
        args = ['events', 'verify', '--bundle', 'bundle']
        for error, expected in ((AppError('BUNDLE_INVALID', 'safe'), 2), (OSError('PRIVATE_PATH'), 2),
                                (ValueError('PRIVATE_PATH'), 2), (KeyboardInterrupt(), 130)):
            with patch('cheating_racer_detect.events.verify.verify_batch', side_effect=error):
                code, out, err = invoke(args)
            self.assertEqual((code, out), (expected, ''))
            self.assertEqual(json.loads(err)['status'], 'error')
            self.assertNotIn('PRIVATE_PATH', err)
        with patch('cheating_racer_detect.events.verify.verify_batch', return_value={'status': 'verified'}):
            self.assertEqual(invoke(args)[0], 0)

    def test_base_import_and_help_never_import_ml_or_private_modules(self):
        code = '''import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('numpy','cv2','torch','yolox','analysis_adapter','analysis_smoke'):
   raise AssertionError('Unexpected optional import')
sys.meta_path.insert(0,Block())
from cheating_racer_detect.cli import main
try: main(['events','--help'])
except SystemExit as e: assert e.code==0
assert not any(n.split('.')[0] in ('numpy','cv2','torch','yolox') for n in sys.modules)
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
