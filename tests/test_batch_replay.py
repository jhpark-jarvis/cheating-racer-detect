"""Bounded explicit selection and all-or-none transaction/CLI contracts."""

import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import replay
from cheating_racer_detect.events.inputs import load_inputs
from scripts.replay_events import main
from tests.test_event_inputs import fixture


class Batch(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='crd-batch-unit-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'source.mp4'
        self.source.write_bytes(b'owned mock source')
        self.inputs = fixture(replay._sha256(self.source))
        self.output = self.root/'result'
        self.analysis = replay.ReplayAnalysis(tuple(SimpleNamespace(direction=d) for d in ('right', 'left', 'right')), None)
        self.analyze = self.enterContext(patch.object(replay, 'analyze_inputs', return_value=self.analysis))
        self.export = self.enterContext(patch.object(replay, 'export_candidate', side_effect=self.media))

    def media(self, _source, candidate, output, *_args, **_kwargs):
        self.assertFalse(self.output.exists())  # Intermediate bundles aren't final results.
        output.mkdir()
        (output/'candidate.json').write_text(json.dumps({'direction': candidate.direction}), encoding='utf-8')
        return {'signal_state': 'unknown'}

    def run_batch(self, indices=(2, 0), inputs=None):
        return replay.export_replay_batch(self.source, inputs or self.inputs, self.output,
                                          candidate_indices=indices, toolchain=object())

    def clean_failure(self):
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('*.partial')), [])

    def test_order_subset_receipt_paths_hash_and_one_analysis(self):
        result = self.run_batch()
        self.assertEqual(result['candidate_indices'], [2, 0])
        self.assertEqual((result['candidate_count'], result['selected_count']), (3, 2))
        self.assertEqual([e['candidate_index'] for e in result['candidates']], [2, 0])
        self.assertEqual([e['candidate'] for e in result['candidates']],
                         ['candidates/000002/candidate.json', 'candidates/000000/candidate.json'])
        self.analyze.assert_called_once()
        self.assertEqual(self.export.call_count, 2)
        self.assertEqual(load_inputs(self.output/'inputs.json'), self.inputs)
        self.assertEqual(result['inputs_sha256'], replay._sha256(self.output/'inputs.json'))
        self.assertEqual(json.loads((self.output/'batch.json').read_text(encoding='utf-8')), result)
        self.assertEqual(len(list((self.output/'candidates').iterdir())), 2)

    def test_selection_invalid_before_any_analysis_or_export(self):
        for value in ((), [], [0, 0], [-1], [True], [0.], ['0'], '0', {0}, None,
                      range(2), (n for n in range(2)), list(range(121))):
            with self.subTest(value=type(value)), self.assertRaises(AppError) as error:
                self.run_batch(value)
            self.assertEqual(error.exception.code, 'INVALID_SELECTION')
        self.analyze.assert_not_called()
        self.export.assert_not_called()
        self.clean_failure()

    def test_selection_snapshot_cannot_be_mutated_during_analysis(self):
        indices = [1, 0]
        def analyze(*_args):
            indices[:] = [2]
            return self.analysis
        self.analyze.side_effect = analyze
        self.assertEqual(self.run_batch(indices)['candidate_indices'], [1, 0])

    def test_entire_range_checked_before_first_export_or_stage(self):
        with self.assertRaises(AppError) as error:
            self.run_batch((0, 3))
        self.assertEqual(error.exception.code, 'INVALID_CANDIDATE')
        self.export.assert_not_called()
        self.clean_failure()

    def test_zero_candidates_is_not_empty_success(self):
        self.analyze.return_value = replay.ReplayAnalysis((), None)
        with self.assertRaises(AppError) as error:
            self.run_batch((0,))
        self.assertEqual(error.exception.code, 'INVALID_CANDIDATE')
        self.clean_failure()

    def test_input_type_and_existing_output_preflight(self):
        with self.assertRaises(AppError) as error:
            replay.export_replay_batch(self.source, object(), self.output, candidate_indices=(0,), toolchain=object())
        self.assertEqual(error.exception.code, 'INVALID_INPUTS')
        self.output.mkdir()
        keep = self.output/'keep'
        keep.write_bytes(b'existing result')
        with self.assertRaises(AppError) as error:
            self.run_batch()
        self.assertEqual(error.exception.code, 'OUTPUT_EXISTS')
        self.assertEqual(keep.read_bytes(), b'existing result')
        self.analyze.assert_not_called()

    def test_late_failure_cancel_and_retry_remove_preceding_bundle(self):
        keep = self.root/'unrelated'
        keep.write_bytes(b'keep')
        before = self.source.read_bytes()
        for failure in (OSError('private disk path'), RuntimeError('native'), KeyboardInterrupt()):
            calls = 0
            def late(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise failure
                return self.media(*args, **kwargs)
            self.export.side_effect = late
            with self.assertRaises((AppError, RuntimeError, KeyboardInterrupt)) as error:
                self.run_batch()
            if isinstance(error.exception, AppError):
                self.assertEqual(error.exception.code, 'IO_ERROR')
                self.assertNotIn('private disk path', str(error.exception))
            self.assertEqual(calls, 2)
            self.clean_failure()
            self.assertEqual(self.source.read_bytes(), before)
            self.assertEqual(keep.read_bytes(), b'keep')
        self.export.side_effect = self.media
        self.assertEqual(self.run_batch()['status'], 'complete')

    def test_final_manifest_disk_failure_and_retry(self):
        original = replay.write_bytes
        def fail(path, data):
            if path.name == 'batch.json':
                raise OSError('full')
            original(path, data)
        with patch.object(replay, 'write_bytes', side_effect=fail), self.assertRaises(AppError):
            self.run_batch()
        self.clean_failure()
        self.assertEqual(self.run_batch()['status'], 'complete')

    def test_final_source_change_aborts_entire_batch(self):
        def mutate(*args, **kwargs):
            value = self.media(*args, **kwargs)
            self.source.write_bytes(b'changed owned source')
            return value
        self.export.side_effect = mutate
        with self.assertRaises(AppError) as error:
            self.run_batch()
        self.assertEqual(error.exception.code, 'INPUT_CHANGED')
        self.clean_failure()

    def test_cleanup_failure_precedes_outer_publication(self):
        with patch.object(replay, 'release_pixels', side_effect=AppError('CLEANUP_FAILED', 'owned resource')), \
                self.assertRaises(AppError) as error:
            self.run_batch()
        self.assertEqual(error.exception.code, 'CLEANUP_FAILED')
        self.clean_failure()
        self.assertEqual(self.run_batch()['status'], 'complete')

    def test_racing_output_preserved_and_owned_batch_removed(self):
        from cheating_racer_detect.events import inputs
        original = inputs._publish
        def race(stage, output):
            output.mkdir()
            (output/'keep').write_bytes(b'other result')
            return original(stage, output)
        with patch.object(inputs, '_publish', side_effect=race), self.assertRaises(AppError):
            self.run_batch()
        self.assertEqual((self.output/'keep').read_bytes(), b'other result')
        self.assertFalse((self.output/'batch.json').exists())
        self.assertEqual(list(self.root.glob('*.partial')), [])

    def test_uncertain_outer_owner_preserved_not_deleted(self):
        from cheating_racer_detect.events import inputs
        original = inputs._identity
        def changed(path):
            value = original(path)
            if Path(path).name.startswith('.crd-inputs-') and self.export.call_count:
                return ('changed', 'changed', *value[2:])
            return value
        with patch.object(inputs, '_identity', side_effect=changed), self.assertRaises(AppError) as error:
            self.run_batch()
        self.assertEqual(error.exception.code, 'CLEANUP_FAILED')
        self.assertFalse(self.output.exists())
        self.assertEqual(len(list(self.root.glob('*.partial'))), 1)
        # The enclosing TemporaryDirectory owns these non-sensitive fixtures.

    def test_preanalysis_io_is_sanitized(self):
        with patch.object(replay, '_sha256', side_effect=OSError('private source path')), self.assertRaises(AppError) as error:
            self.run_batch()
        self.assertEqual(error.exception.code, 'IO_ERROR')
        self.assertNotIn('private source path', str(error.exception))
        self.clean_failure()

    def test_command_batch_success_failure_cancel_no_paths(self):
        args = ['--input', 'source.mp4', '--inputs', 'inputs.json', '--output', 'out', '--candidate-indices', '1', '0']
        with patch('scripts.replay_events.load_inputs', return_value=self.inputs), \
                patch('scripts.replay_events.export_replay_batch') as export, patch('sys.stdout', new_callable=io.StringIO) as out:
            export.return_value = {'status': 'complete', 'candidate_indices': [1, 0], 'selected_count': 2}
            self.assertEqual(main(args), 0)
            self.assertEqual(export.call_args.kwargs['candidate_indices'], [1, 0])
            for failure, code in ((AppError('INVALID_SELECTION', 'private path'), 1), (KeyboardInterrupt(), 130)):
                export.side_effect = failure
                self.assertEqual(main(args), code)
            self.assertNotIn('private path', out.getvalue())
            self.assertNotIn('source.mp4', out.getvalue())

    def test_command_requires_exactly_one_selection_mode(self):
        base = ['--input', 'source.mp4', '--inputs', 'inputs.json', '--output', 'out']
        with patch('sys.stderr', new_callable=io.StringIO):
            for extra in ([], ['--candidate-index', '0', '--candidate-indices', '1'], ['--candidate-indices']):
                with self.assertRaises(SystemExit) as error:
                    main(base+extra)
                self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
