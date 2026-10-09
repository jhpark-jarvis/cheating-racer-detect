"""Strict, stdlib-only experimental input receipts and owned transactions."""

from dataclasses import replace
from fractions import Fraction as F
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import LaneConfig, LaneSection, Vehicle
from cheating_racer_detect.events.inputs import EventInputs, InputFrame, MAX_BYTES, load_inputs, save_inputs
from cheating_racer_detect.events.lamp import LampConfig, LampROI
from cheating_racer_detect.events.replay import ReplayAnalysis, analyze_inputs, export_replay, recorded_observer
from cheating_racer_detect.review.records import summarize
from cheating_racer_detect.tracking import Box, Detection, Frame
from scripts.replay_events import main


def fixture(source_id='a'*64):
    lane = LaneConfig('test-lane', .15, F(1, 5), F(3, 10), 10.)
    lamp = LampConfig('test-lamp', 50., 180., 8, F(3, 5), F(3, 10), F(1, 10), F(1), 1)
    rows = []
    for n in range(12):
        frame = Frame(source_id, 0, n, n*10, F(1, 100), 0, 96, 64, 90)
        vehicle = Vehicle(frame, (1, 2, 1), Detection(0, 2, Box(12, 10, 36, 30), .9, 'scripted', frame))
        lane_section = LaneSection(frame, 'road', ('a', 'b', 'c'), (0., 48., 96.), 30., 'stable', 'ordinary', True)
        rows.append(InputFrame(frame, vehicle, lane_section, LampROI(Box(16, 20, 20, 24), 'usable'),
                               LampROI(Box(28, 20, 32, 24), 'usable')))
    return EventInputs(lane, lamp, F(1, 5), F(2, 5), (1, 2, 1), tuple(rows))


class Inputs(unittest.TestCase):
    def test_roundtrip_exact_settings_source_time_rotation_and_digest(self):
        inputs = fixture()
        restored = EventInputs.from_bytes(inputs.to_bytes())
        self.assertEqual(restored, inputs)
        self.assertEqual(restored.sha256, hashlib.sha256(inputs.to_bytes()).hexdigest())
        self.assertEqual(restored.rows[3].frame.time, F(3, 10))
        self.assertEqual(restored.rows[0].frame.display_rotation, 90)
        self.assertEqual(restored.lamp_config.report(), inputs.lamp_config.report())

    def test_reports_are_owned_snapshots_and_config_changes_have_different_hash(self):
        inputs = fixture()
        value = inputs.report()
        value['rows'][0]['vehicle']['detection']['bbox'][0] = 0
        value['rows'][0]['lanes']['x'][0] = 20
        self.assertEqual(inputs.rows[0].vehicle.box.x1, 12)
        self.assertEqual(inputs.rows[0].lanes.x[0], 0)
        changed = replace(inputs, lane_config=replace(inputs.lane_config, margin=.2))
        self.assertNotEqual(inputs.sha256, changed.sha256)
        self.assertEqual(EventInputs.from_bytes(changed.to_bytes()), changed)

    def test_missing_actual_and_unknown_gates_survive_without_interpolation(self):
        inputs = fixture()
        missing = replace(inputs.rows[3], vehicle=None, lanes=None,
                          left=LampROI(None, 'out_of_view'), right=LampROI(None, 'model_uncertain'))
        sparse = replace(inputs, rows=inputs.rows[:3]+(missing,)+inputs.rows[5:])
        restored = EventInputs.from_bytes(sparse.to_bytes())
        self.assertIsNone(restored.rows[3].vehicle)
        self.assertIsNone(restored.rows[3].lanes)
        self.assertEqual(restored.rows[4].frame.ordinal, 5)
        report = recorded_observer(restored)(missing.frame, b'')
        self.assertEqual(summarize(missing.frame, report)['annotations'], [])

    def test_untrusted_schema_duplicate_unknown_keys_and_nonfinite_fail(self):
        base = fixture().report()
        values = [b'{}', b'{"schema":1,"schema":2}', b'{"x":NaN}', b'{"x":Infinity}',
                  b'\xff', b'['*1500+b']'*1500, b'\x00', b'x'*(MAX_BYTES+1)]
        for key, value in (('schema', 'product-events-v1'), ('extra', 'unexpected'), ('rows', []),
                           ('pre', '1/0'), ('post', '0.4'), ('pre', '2/10'), ('pre', True)):
            document = {**base, key: value}
            values.append(json.dumps(document).encode())
        for value in values:
            with self.subTest(value=value[:40]), self.assertRaises(AppError) as error:
                EventInputs.from_bytes(value)
            self.assertEqual(error.exception.code, 'INVALID_INPUTS')

    def test_nested_frame_vehicle_lane_roi_metadata_cannot_be_forged(self):
        paths = [(('rows', 0, 'frame', 'normalized_time'), '1'),
                 (('rows', 0, 'frame', 'pts'), True), (('rows', 0, 'frame', 'coded_size'), [96., 64]),
                 (('rows', 0, 'vehicle', 'identity_status'), 'verified'),
                 (('rows', 0, 'vehicle', 'frame', 'ordinal'), 2),
                 (('rows', 0, 'vehicle', 'detection', 'extra'), 1),
                 (('rows', 0, 'lanes', 'rear_view'), 1), (('rows', 0, 'lanes', 'x'), [48., 0., 96.]),
                 (('rows', 0, 'right', 'box'), [1., 1., 4., 4.]), (('rows', 0, 'left', 'status'), 'assumed_off')]
        for path, value in paths:
            document = fixture().report()
            target = document
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(AppError):
                EventInputs.from_bytes(json.dumps(document).encode())

    def test_limits_order_identity_source_and_clock_are_explicit(self):
        inputs = fixture()
        invalid = [replace(inputs.rows[1], frame=inputs.rows[0].frame, vehicle=None, lanes=None),
                   replace(inputs.rows[1], frame=replace(inputs.rows[1].frame, source_id='b'*64), vehicle=None, lanes=None),
                   replace(inputs.rows[1], frame=replace(inputs.rows[1].frame, time_base=F(1, 200)), vehicle=None, lanes=None),
                   replace(inputs.rows[1], vehicle=replace(inputs.rows[1].vehicle, identity=(2, 2, 1)))]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                replace(inputs, rows=(inputs.rows[0], row))
        for rows in ([], (), inputs.rows*11):
            with self.assertRaises(ValueError):
                replace(inputs, rows=rows)
        with self.assertRaises(ValueError):
            replace(inputs, pre=F(0))
        with self.assertRaises(ValueError):
            replace(inputs, selected_identity=(True, 2, 1))

    def test_observer_only_restores_actual_selected_inputs_and_rejects_wrong_frame(self):
        inputs = fixture()
        observe = recorded_observer(inputs)
        row = inputs.rows[2]
        summary = summarize(row.frame, observe(row.frame, b''))
        self.assertEqual(len(summary['annotations']), 1)
        self.assertIsNone(summary['annotations'][0]['estimated_bbox'])
        self.assertEqual(Vehicle.from_tracking(row.frame, observe(row.frame, b''), inputs.selected_identity), row.vehicle)
        absent = replace(row.frame, ordinal=50, pts=500)
        self.assertEqual(observe(absent, b'')['tracks'], [])
        for frame in (replace(row.frame, pts=21), replace(row.frame, source_id='b'*64)):
            with self.assertRaises(AppError):
                observe(frame, b'')
        report = observe(row.frame, b'')
        report['detections'][0]['bbox'][0] = 0
        self.assertEqual(observe(row.frame, b'')['detections'][0]['bbox'][0], 12)

    def test_local_save_load_is_atomic_and_never_overwrites(self):
        with tempfile.TemporaryDirectory(prefix='crd-input-test-') as directory:
            root = Path(directory)
            inputs = fixture()
            self.assertEqual(save_inputs(inputs, root/'saved'), inputs.sha256)
            before = (root/'saved'/'inputs.json').read_bytes()
            self.assertEqual(load_inputs(root/'saved'/'inputs.json'), inputs)
            with self.assertRaises(AppError) as error:
                save_inputs(inputs, root/'saved')
            self.assertEqual(error.exception.code, 'OUTPUT_EXISTS')
            self.assertEqual((root/'saved'/'inputs.json').read_bytes(), before)
            self.assertEqual(list(root.glob('.crd-inputs-*.partial')), [])

    def test_save_disk_failure_cancel_and_retry_clean_only_owned_stage(self):
        with tempfile.TemporaryDirectory(prefix='crd-input-errors-') as directory:
            root = Path(directory)
            keep = root/'unrelated'
            keep.mkdir()
            for failure in (OSError('disk full'), KeyboardInterrupt()):
                with patch('cheating_racer_detect.events.inputs.write_bytes', side_effect=failure):
                    with self.assertRaises((AppError, KeyboardInterrupt)):
                        save_inputs(fixture(), root/'saved')
                self.assertFalse((root/'saved').exists())
                self.assertTrue(keep.is_dir())
                self.assertEqual(list(root.glob('.crd-inputs-*.partial')), [])
            save_inputs(fixture(), root/'saved')

    def test_load_is_size_bounded_and_rejects_remote_and_missing_paths(self):
        with tempfile.TemporaryDirectory(prefix='crd-input-load-') as directory:
            root = Path(directory)
            path = root/'bad.json'
            path.write_bytes(b'x'*(MAX_BYTES+1))
            for value in (path, root/'missing.json', 'https://example.test/input.json', root/'../other.json'):
                with self.assertRaises(AppError):
                    load_inputs(value)

    def test_reading_changed_receipt_is_not_accepted(self):
        with tempfile.TemporaryDirectory(prefix='crd-input-changed-') as directory:
            root = Path(directory)
            save_inputs(fixture(), root/'saved')
            with patch('cheating_racer_detect.events.inputs._identity', side_effect=[(1, 2, 3), (1, 2, 4)]):
                with self.assertRaises(AppError) as error:
                    load_inputs(root/'saved'/'inputs.json')
            self.assertEqual(error.exception.code, 'INPUT_CHANGED')

    def test_replay_bundles_receipt_and_media_atomically_failure_cancel_and_retry(self):
        with tempfile.TemporaryDirectory(prefix='crd-input-bundle-') as directory:
            root = Path(directory)
            source = root/'source.mp4'
            source.write_bytes(b'owned mock source')
            inputs = fixture(hashlib.sha256(source.read_bytes()).hexdigest())
            analysis = ReplayAnalysis((object(),), None)
            def media(_source, _candidate, output, *_args, **_kwargs):
                output.mkdir()
                (output/'candidate.json').write_text('{}', encoding='utf-8')
                return {'signal_state': 'unknown'}
            with patch('cheating_racer_detect.events.replay.analyze_inputs', return_value=analysis):
                for failure in (OSError('full'), RuntimeError('native'), KeyboardInterrupt()):
                    with patch('cheating_racer_detect.events.replay.export_candidate', side_effect=failure):
                        with self.assertRaises((AppError, RuntimeError, KeyboardInterrupt)):
                            export_replay(source, inputs, root/'result', candidate_index=0, toolchain=object())
                    self.assertFalse((root/'result').exists())
                    self.assertEqual(list(root.glob('.crd-inputs-*.partial')), [])
                with patch('cheating_racer_detect.events.replay.export_candidate', side_effect=media):
                    result = export_replay(source, inputs, root/'result', candidate_index=0, toolchain=object())
                    self.assertEqual(result['inputs_sha256'], inputs.sha256)
                    self.assertEqual(load_inputs(root/'result'/'inputs.json'), inputs)
                    self.assertEqual(json.loads((root/'result'/'replay.json').read_text()), result)
                    with self.assertRaises(AppError):
                        export_replay(source, inputs, root/'result', candidate_index=0, toolchain=object())
                with self.assertRaises(AppError):
                    export_replay(source, inputs, root/'bad-index', candidate_index=1, toolchain=object())
                self.assertFalse((root/'bad-index').exists())

    def test_source_change_after_media_aborts_publication(self):
        with tempfile.TemporaryDirectory(prefix='crd-input-source-') as directory:
            root = Path(directory)
            source = root/'source.mp4'
            source.write_bytes(b'owned mock')
            def change(*_args, **_kwargs):
                source.write_bytes(b'changed owned mock')
                return {'signal_state': 'unknown'}
            with patch('cheating_racer_detect.events.replay.analyze_inputs', return_value=ReplayAnalysis((object(),), None)), \
                    patch('cheating_racer_detect.events.replay.export_candidate', side_effect=change):
                with self.assertRaises(AppError) as error:
                    export_replay(source, fixture(), root/'result', candidate_index=0, toolchain=object())
                self.assertEqual(error.exception.code, 'INPUT_CHANGED')
            self.assertFalse((root/'result').exists())
            self.assertEqual(list(root.glob('.crd-inputs-*.partial')), [])

    def test_source_only_command_success_error_and_cancel(self):
        args = ['--input', 'source.mp4', '--inputs', 'inputs.json', '--output', 'out', '--candidate-index', '0']
        with patch('scripts.replay_events.load_inputs', return_value=fixture()), patch('sys.stdout', new_callable=io.StringIO) as out:
            with patch('scripts.replay_events.export_replay', return_value={'status': 'complete', 'candidate_index': 0, 'signal_state': 'unknown'}):
                self.assertEqual(main(args), 0)
            self.assertNotIn('source.mp4', out.getvalue())
            for failure, code in ((AppError('INVALID_INPUTS', 'private path'), 1), (KeyboardInterrupt(), 130)):
                with patch('scripts.replay_events.export_replay', side_effect=failure):
                    self.assertEqual(main(args), code)
            self.assertNotIn('private path', out.getvalue())

    def test_failed_decode_io_has_non_sensitive_error(self):
        with patch('cheating_racer_detect.events.replay._analyze_inputs', side_effect=OSError('private source path')):
            with self.assertRaises(AppError) as error:
                analyze_inputs('source.mp4', fixture(), object())
        self.assertEqual(error.exception.code, 'IO_ERROR')
        self.assertNotIn('private source path', str(error.exception))

    def test_output_race_preserves_unrelated_result_and_cleans_owned_stage(self):
        with tempfile.TemporaryDirectory(prefix='crd-input-race-') as directory:
            root = Path(directory)
            from cheating_racer_detect.events.inputs import _publish
            def race(stage, output):
                output.mkdir()
                (output/'keep').write_bytes(b'unrelated')
                return _publish(stage, output)
            with patch('cheating_racer_detect.events.inputs._publish', side_effect=race):
                with self.assertRaises(AppError):
                    save_inputs(fixture(), root/'result')
            self.assertEqual((root/'result'/'keep').read_bytes(), b'unrelated')
            self.assertEqual(list(root.glob('.crd-inputs-*.partial')), [])

    def test_uncertain_stage_ownership_preserves_partial_and_reports_cleanup_failure(self):
        with tempfile.TemporaryDirectory(prefix='crd-input-owner-') as directory:
            root = Path(directory)
            import cheating_racer_detect.events.inputs as module
            original_identity = module._identity
            calls = 0
            def identity(path):
                nonlocal calls
                value = original_identity(path)
                if Path(path).name.startswith('.crd-inputs-'):
                    calls += 1
                    if calls > 1:
                        return ('changed-owner', 'changed-owner', *value[2:])
                return value
            with patch.object(module, '_identity', side_effect=identity), \
                    patch.object(module, 'write_bytes', side_effect=OSError('full')):
                with self.assertRaises(AppError) as error:
                    save_inputs(fixture(), root/'result')
                self.assertEqual(error.exception.code, 'CLEANUP_FAILED')
            self.assertFalse((root/'result').exists())
            self.assertEqual(len(list(root.glob('.crd-inputs-*.partial'))), 1)
            # The enclosing test TemporaryDirectory owns and removes this fixture.


if __name__ == '__main__':
    unittest.main()
