"""Strict immutable compact receipts, transactional listing and selection guards."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import catalog, replay
from cheating_racer_detect.events.inputs import MAX_BYTES, load_inputs
from scripts.list_events import main as list_main
from scripts.replay_events import main as replay_main
from tests.test_event_inputs import fixture


def document(inputs=None):
    inputs = inputs or fixture()
    side = {'state': 'no_blink_observed', 'blink_evidence': False, 'reasons': [], 'valid_samples': 12}
    return {'schema': catalog.SCHEMA, 'status': 'complete', 'inputs': 'inputs.json', 'inputs_sha256': inputs.sha256,
            'source_sha256': inputs.rows[0].frame.source_id, 'source_frame': inputs.rows[0].frame.report(),
            'selected_identity': list(inputs.selected_identity), 'candidate_count': 1, 'limits': list(catalog.LIMITS),
            'candidates': [{'candidate_index': 0, 'candidate_sha256': 'b'*64, 'direction': 'right',
                'start': ['1/10', '1/5'], 'crossing': ['3/10', '2/5'], 'completion': ['1/2', '3/5'],
                'requested_context': ['0', '1'], 'context': ['0', '11/10'], 'signal_state': 'no_blink_observed',
                'left': dict(side), 'right': dict(side)}]}


def receipt(value=None):
    return catalog.CandidateCatalog.from_bytes(catalog._bytes(document() if value is None else value))


class Contract(unittest.TestCase):
    def test_canonical_roundtrip_digest_owned_reports_and_frozen_data(self):
        value = document()
        item = catalog.CandidateCatalog.from_bytes(json.dumps(value, indent=3).encode())
        self.assertEqual(item, receipt(value))
        self.assertEqual(item.sha256, catalog.hashlib.sha256(item.to_bytes()).hexdigest())
        report = item.report()
        report['candidates'][0]['right']['reasons'].append('occluded')
        report['source_frame']['pts'] += 1
        self.assertEqual(item.report(), value)
        with self.assertRaises(FrozenInstanceError):
            item.data = b'changed'

    def test_size_unicode_depth_duplicate_nonfinite_and_zero_denominator_rejected(self):
        for data in (b'', b'{}', b'\xff', b'['*1500+b']'*1500, b'x'*(MAX_BYTES+1),
                     b'{"schema":1,"schema":2}', b'{"x":NaN}', b'{"x":Infinity}', bytearray(b'{}')):
            with self.subTest(data=str(type(data))), self.assertRaises(AppError) as error:
                catalog.CandidateCatalog.from_bytes(data)
            self.assertEqual(error.exception.code, 'INVALID_CATALOG')
        value = document()
        value['candidates'][0]['start'][0] = '1/0'
        with self.assertRaises(AppError):
            receipt(value)

    def test_strict_nested_schema_count_index_hash_frame_clock_and_limits(self):
        changes = [(('extra',), True), (('schema',), 'product-events-v1'), (('status',), 'partial'),
                   (('inputs',), '../private.json'), (('inputs_sha256',), 'A'*64),
                   (('source_frame', 'pts'), True), (('source_frame', 'time_base'), '0.01'),
                   (('source_frame', 'source_id'), 'c'*64), (('selected_identity',), [True, 2, 1]),
                   (('candidate_count',), True), (('candidate_count',), 121), (('candidate_count',), 2),
                   (('limits',), []), (('candidates', 0, 'extra'), 'x'),
                   (('candidates', 0, 'candidate_index'), 1), (('candidates', 0, 'candidate_index'), False),
                   (('candidates', 0, 'candidate_sha256'), 'invalid'), (('candidates', 0, 'direction'), 'forward')]
        for path, replacement in changes:
            value = document()
            target = value
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = replacement
            with self.subTest(path=path), self.assertRaises(AppError):
                receipt(value)

    def test_time_order_and_context_are_exact_canonical_and_positive(self):
        for name, values in (('start', ['2/10', '1/5']), ('crossing', ['1/2', '2/5']),
                             ('completion', ['1/10', '3/5']), ('context', ['1/10', '11/10']),
                             ('context', ['0', '9/10']), ('requested_context', ['0', '3/5'])):
            value = document()
            value['candidates'][0][name] = values
            with self.subTest(name=name), self.assertRaises(AppError):
                receipt(value)

    def test_unknown_keeps_partial_blink_and_side_reasons_without_claiming_off(self):
        value = document()
        entry = value['candidates'][0]
        entry['signal_state'] = 'unknown'
        entry['right'] = {'state': 'unknown', 'blink_evidence': True, 'reasons': ['occluded'], 'valid_samples': 10}
        self.assertEqual(receipt(value).report()['candidates'][0]['right'], entry['right'])
        for key, replacement in (('state', 'assumed_off'), ('blink_evidence', 1), ('reasons', []),
                                 ('reasons', ['occluded', 'occluded']), ('reasons', ['private/path']),
                                 ('valid_samples', 121), ('extra', 'x')):
            changed = json.loads(json.dumps(value))
            changed['candidates'][0]['right'][key] = replacement
            with self.subTest(key=key), self.assertRaises(AppError):
                receipt(changed)
        entry['signal_state'] = 'right_blink'
        with self.assertRaises(AppError):
            receipt(value)

    def test_zero_candidates_complete_catalog_not_empty_successful_batch(self):
        value = document()
        value['candidate_count'], value['candidates'] = 0, []
        self.assertEqual(receipt(value).report()['candidate_count'], 0)

    def test_maximum120_entries_bounded_and_in_original_completion_order(self):
        value = document()
        first = value['candidates'][0]
        value['candidate_count'] = 120
        value['candidates'] = []
        for n in range(120):
            value['candidates'].append({**first, 'candidate_index': n,
                **{key: [str(F(item)+n) for item in first[key]]
                   for key in ('start', 'crossing', 'completion', 'context', 'requested_context')}})
        item = receipt(value)
        self.assertLess(len(item.to_bytes()), MAX_BYTES)
        value['candidates'][1]['completion'] = first['completion']
        with self.assertRaises(AppError):
            receipt(value)

    def test_load_bounded_local_receipt_and_changed_during_read(self):
        with tempfile.TemporaryDirectory(prefix='crd-catalog-load-') as directory:
            root = Path(directory)
            path = root/'catalog.json'
            path.write_bytes(receipt().to_bytes())
            self.assertEqual(catalog.load_catalog(path), receipt())
            with patch.object(catalog, '_identity', side_effect=[(1, 2, 3), (1, 2, 4)]), self.assertRaises(AppError) as error:
                catalog.load_catalog(path)
            self.assertEqual(error.exception.code, 'INPUT_CHANGED')
            for candidate in ('https://example.test/catalog.json', root/'../catalog.json', root/'missing.json'):
                with self.assertRaises(AppError):
                    catalog.load_catalog(candidate)
            path.write_bytes(b'x'*(MAX_BYTES+1))
            with self.assertRaises(AppError):
                catalog.load_catalog(path)


class Transactions(unittest.TestCase):
    def setUp(self):
        owned = tempfile.TemporaryDirectory(prefix='crd-catalog-unit-')
        self.addCleanup(owned.cleanup)
        self.root = Path(owned.name)
        self.source = self.root/'source.mp4'
        self.source.write_bytes(b'owned mock source')
        self.inputs = fixture(catalog._sha256(self.source))
        self.item = receipt(document(self.inputs))
        self.output = self.root/'result'
        self.analyze = self.enterContext(patch.object(replay, 'analyze_inputs', return_value=replay.ReplayAnalysis((SimpleNamespace(direction='right'),), None)))
        self.inspect = self.enterContext(patch.object(catalog, 'inspect_source', return_value=(None, None, object())))
        self.make = self.enterContext(patch.object(catalog, 'make_catalog', return_value=self.item))

    def invoke(self):
        return catalog.export_catalog(self.source, self.inputs, self.output, toolchain=object())

    def clean(self):
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('*.partial')), [])

    def test_atomic_inputs_catalog_only_no_media_and_one_analysis(self):
        self.assertEqual(self.invoke(), self.item)
        self.analyze.assert_called_once()
        self.assertEqual({path.name for path in self.output.iterdir()}, {'catalog.json', 'inputs.json'})
        self.assertEqual(catalog.load_catalog(self.output/'catalog.json'), self.item)
        self.assertEqual(load_inputs(self.output/'inputs.json'), self.inputs)

    def test_existing_output_and_invalid_input_preflight_no_analysis(self):
        with self.assertRaises(AppError):
            catalog.export_catalog(self.source, object(), self.output, toolchain=object())
        self.output.mkdir()
        (self.output/'keep').write_bytes(b'keep')
        with self.assertRaises(AppError) as error:
            self.invoke()
        self.assertEqual(error.exception.code, 'OUTPUT_EXISTS')
        self.analyze.assert_not_called()
        self.assertEqual((self.output/'keep').read_bytes(), b'keep')

    def test_final_write_failure_cancel_and_retry_remove_owned_partial(self):
        original = catalog.write_bytes
        for failure in (OSError('private disk path'), KeyboardInterrupt()):
            def fail(path, data):
                if path.name == 'catalog.json':
                    raise failure
                original(path, data)
            with patch.object(catalog, 'write_bytes', side_effect=fail), self.assertRaises((AppError, KeyboardInterrupt)):
                self.invoke()
            self.clean()
        self.assertEqual(self.invoke(), self.item)

    def test_source_change_after_analysis_and_cleanup_failure_block_publish(self):
        before = self.source.read_bytes()
        def mutate(*_args):
            self.source.write_bytes(b'changed owned source')
            return self.item
        self.make.side_effect = mutate
        with self.assertRaises(AppError) as error:
            self.invoke()
        self.assertEqual(error.exception.code, 'INPUT_CHANGED')
        self.clean()
        self.source.write_bytes(before)
        self.make.side_effect = None
        with patch.object(catalog, 'release_pixels', side_effect=AppError('CLEANUP_FAILED', 'owned')), self.assertRaises(AppError):
            self.invoke()
        self.clean()
        self.invoke()

    def test_racing_output_and_uncertain_stage_ownership_preserved(self):
        from cheating_racer_detect.events import inputs
        original = inputs._publish
        def race(stage, output):
            output.mkdir()
            (output/'keep').write_bytes(b'other')
            return original(stage, output)
        with patch.object(inputs, '_publish', side_effect=race), self.assertRaises(AppError):
            self.invoke()
        self.assertEqual((self.output/'keep').read_bytes(), b'other')
        self.assertEqual(list(self.root.glob('*.partial')), [])
        old_identity = inputs._identity
        calls = 0
        def changed(path):
            nonlocal calls
            result = old_identity(path)
            if Path(path).name.startswith('.crd-inputs-'):
                calls += 1
                if calls > 1:
                    return ('changed', 'changed', *result[2:])
            return result
        self.output = self.root/'uncertain'
        with patch.object(inputs, '_identity', side_effect=changed), self.assertRaises(AppError) as error:
            self.invoke()
        self.assertEqual(error.exception.code, 'CLEANUP_FAILED')
        self.assertEqual(len(list(self.root.glob('*.partial'))), 1)

    def test_batch_type_source_inputs_mismatch_before_analysis(self):
        for item in (object(), receipt(), receipt(document(replace(self.inputs, pre=F(1, 4))))):
            with self.subTest(type=type(item)), self.assertRaises(AppError) as error:
                replay.export_replay_batch(self.source, self.inputs, self.output, candidate_indices=(0,), toolchain=object(), catalog=item)
            self.assertIn(error.exception.code, ('INVALID_CATALOG', 'CATALOG_CHANGED'))
        self.analyze.assert_not_called()
        self.clean()

    def test_batch_fresh_catalog_drift_before_any_bundle_or_stage(self):
        value = document(self.inputs)
        value['candidates'][0]['candidate_sha256'] = 'c'*64
        with patch('cheating_racer_detect.source.inspect_source', return_value=(None, None, object())), \
                patch.object(catalog, 'make_catalog', return_value=receipt(value)), \
                patch.object(replay, 'export_candidate') as export, self.assertRaises(AppError) as error:
            replay.export_replay_batch(self.source, self.inputs, self.output, candidate_indices=(0,), toolchain=object(), catalog=self.item)
        self.assertEqual(error.exception.code, 'CATALOG_CHANGED')
        export.assert_not_called()
        self.clean()

    def test_batch_copies_matching_catalog_and_its_canonical_hash(self):
        def media(_source, _candidate, output, *_args, **_kwargs):
            output.mkdir()
            (output/'candidate.json').write_text('{}', encoding='utf-8')
            return {'signal_state': 'no_blink_observed'}
        with patch('cheating_racer_detect.source.inspect_source', return_value=(None, None, object())), \
                patch.object(replay, 'export_candidate', side_effect=media):
            result = replay.export_replay_batch(self.source, self.inputs, self.output, candidate_indices=(0,), toolchain=object(), catalog=self.item)
        self.assertEqual(result['catalog'], 'catalog.json')
        self.assertEqual(result['catalog_sha256'], self.item.sha256)
        self.assertEqual(catalog.load_catalog(self.output/'catalog.json'), self.item)

    def test_analysis_io_failure_is_non_sensitive(self):
        self.analyze.side_effect = OSError('private original path')
        with self.assertRaises(AppError) as error:
            self.invoke()
        self.assertEqual(error.exception.code, 'IO_ERROR')
        self.assertNotIn('private original path', str(error.exception))
        self.clean()

    def test_list_command_success_error_cancel_has_no_paths_or_digests(self):
        args = ['--input', 'source.mp4', '--inputs', 'inputs.json', '--output', 'out']
        with patch('scripts.list_events.load_inputs', return_value=self.inputs), \
                patch('scripts.list_events.export_catalog') as export, patch('sys.stdout', new_callable=io.StringIO) as out:
            export.return_value = self.item
            self.assertEqual(list_main(args), 0)
            summary = json.loads(out.getvalue())
            self.assertEqual(summary['candidates'][0]['candidate_index'], 0)
            self.assertNotIn(self.inputs.rows[0].frame.source_id, out.getvalue())
            self.assertNotIn('source.mp4', out.getvalue())
            for failure, code in ((AppError('INVALID_CATALOG', 'private path'), 1), (KeyboardInterrupt(), 130)):
                export.side_effect = failure
                self.assertEqual(list_main(args), code)
            self.assertNotIn('private path', out.getvalue())

    def test_replay_catalog_option_requires_batch_and_passes_frozen_receipt(self):
        base = ['--input', 'source.mp4', '--inputs', 'inputs.json', '--output', 'out', '--catalog', 'catalog.json']
        with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
            replay_main(base+['--candidate-index', '0'])
        self.assertEqual(error.exception.code, 2)
        with patch('scripts.replay_events.load_inputs', return_value=self.inputs), \
                patch('scripts.replay_events.load_catalog', return_value=self.item), \
                patch('scripts.replay_events.export_replay_batch') as export, patch('sys.stdout', new_callable=io.StringIO):
            export.return_value = {'status': 'complete', 'candidate_indices': [0], 'selected_count': 1}
            self.assertEqual(replay_main(base+['--candidate-indices', '0']), 0)
            self.assertIs(export.call_args.kwargs['catalog'], self.item)


if __name__ == '__main__':
    unittest.main()
