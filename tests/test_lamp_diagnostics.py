"""Input-only lamp metadata, strict readonly parsing and optional transactions."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import catalog, replay
from cheating_racer_detect.events import lamp_diagnostics as diagnostic
from cheating_racer_detect.events.diagnostics import diagnostics_bytes
from cheating_racer_detect.events.inputs import MAX_BYTES
from cheating_racer_detect.events.lamp import LampROI
from cheating_racer_detect.events.lamp_observations import LampEvidence
from cheating_racer_detect.tracking import Box
from scripts.demo_lamp_replay import with_lamp_evidence
from scripts.list_events import main as source_main
from tests.test_candidate_catalog import document, receipt
from tests.test_event_inputs import fixture


def altered(inputs, index, **changes):
    row = inputs.rows[index]
    o = row.lamp_evidence.observation
    evidence = LampEvidence(None if changes.pop('missing', False) else replace(o, **changes))
    left, right = evidence.rois(row.frame, row.vehicle)
    return replace(inputs, rows=(*inputs.rows[:index], replace(row, lamp_evidence=evidence,
        left=left, right=right), *inputs.rows[index+1:]))


class Metadata(unittest.TestCase):
    def test_legacy_not_recorded_v3_unknown_and_exact_original_clock(self):
        legacy = fixture()
        report = diagnostic.lamp_diagnostics(legacy)
        self.assertEqual(report['counts']['legacy_rows'], 12)
        self.assertEqual(report['counts']['source_resets'], 0)
        self.assertTrue(all(r['provenance_status']=='not_recorded' for r in report['rows']))
        inputs = altered(with_lamp_evidence(legacy), 5, missing=True)
        report = diagnostic.lamp_diagnostics(inputs)
        row = report['rows'][5]
        self.assertEqual(row['provenance_status'], 'unknown')
        self.assertEqual(row['frame'], inputs.rows[5].frame.report())
        self.assertEqual(row['source_reset_reason'], 'lamps_unknown')
        self.assertEqual(row['left']['geometry_status'], 'model_uncertain')
        self.assertEqual(report['counts']['observation_unknown_rows'], 1)
        self.assertIsNone(report['rows'][6]['source_reset_reason'])

    def test_v3_owns_nested_declarations_and_no_brightness_or_decision(self):
        inputs = with_lamp_evidence(fixture())
        before = inputs.to_bytes()
        report = diagnostic.lamp_diagnostics(inputs)
        self.assertEqual(report['rows'][0]['left']['roi'], [16., 20., 20., 24.])
        self.assertEqual(report['rows'][0]['left']['whole_pixels'], 16)
        self.assertEqual(report['rows'][0]['source']['left']['visibility'], 'visible')
        report['rows'][0]['source']['left']['visibility'] = 'fake'
        self.assertEqual(inputs.to_bytes(), before)
        self.assertEqual(diagnostic.lamp_diagnostics(inputs)['rows'][0]['source']['left']['visibility'], 'visible')
        for field in ('brightness', 'signal_state', 'blink_evidence'):
            self.assertNotIn('"'+field+'"', json.dumps(report))

    def test_semantic_identity_pose_and_single_side_visibility_gates(self):
        base = with_lamp_evidence(fixture())
        for changes, expected in (({'rear_view':False}, 'model_uncertain'),
                                  ({'side_mapping_confirmed':False}, 'model_uncertain'),
                                  ({'identity_confirmed':False}, 'identity_uncertain')):
            row = diagnostic.lamp_diagnostics(altered(base, 4, **changes))['rows'][4]
            self.assertEqual(row['left']['geometry_status'], expected)
            self.assertEqual(row['right']['geometry_status'], expected)
        o = base.rows[4].lamp_evidence.observation
        for status, expected in (('occluded','occluded'), ('unknown','model_uncertain'),
                                 ('too_small','too_small'), ('out_of_view','out_of_view')):
            row = diagnostic.lamp_diagnostics(altered(base, 4,
                left=replace(o.left, visibility=status)))['rows'][4]
            self.assertEqual(row['left']['geometry_status'], expected)
            self.assertEqual(row['right']['geometry_status'], 'usable')
        row = diagnostic.lamp_diagnostics(altered(base, 4,
            left=replace(o.left, turn_signal_confirmed=False)))['rows'][4]
        self.assertEqual(row['left']['geometry_status'], 'model_uncertain')

    def test_whole_pixel_geometry_not_canvas_upscale_or_raw_fractional_area(self):
        inputs = fixture()
        row = inputs.rows[2]
        roi = LampROI(Box(16.1,20.1,18.9,22.9), 'usable')
        inputs = replace(inputs, rows=(*inputs.rows[:2], replace(row,left=roi), *inputs.rows[3:]))
        side = diagnostic.lamp_diagnostics(inputs)['rows'][2]['left']
        self.assertEqual(side['whole_pixels'], 1)
        self.assertEqual(side['declared_status'], 'usable')
        self.assertEqual(side['geometry_status'], 'too_small')

    def test_source_key_changes_but_moving_crop_and_box_are_not_source_changes(self):
        inputs = with_lamp_evidence(fixture())
        for changes in ({'model_id':'changed'}, {'context_id':'changed'}, {'model_width':480},
                        {'rotation_cw':180}, {'left':replace(inputs.rows[4].lamp_evidence.observation.left,part_id='new')}):
            result = diagnostic.lamp_diagnostics(altered(inputs, 4, **changes))
            self.assertEqual(result['rows'][4]['source_reset_reason'], 'lamp_source_change')
            self.assertEqual(result['rows'][5]['source_reset_reason'], 'lamp_source_change')
        o = inputs.rows[4].lamp_evidence.observation
        moved = altered(inputs, 4, crop=Box(o.crop.x1+1,o.crop.y1,o.crop.x2,o.crop.y2))
        self.assertEqual(diagnostic.lamp_diagnostics(moved)['counts']['source_resets'], 0)

    def test_gap_exact_time_detector_change_absent_vehicle_and_resumption(self):
        inputs = fixture()
        sparse = replace(inputs, rows=(*inputs.rows[:4], *inputs.rows[5:]))
        row = diagnostic.lamp_diagnostics(sparse)['rows'][4]
        self.assertEqual((row['missing_input_frames'],row['elapsed_seconds'],row['clock_break_reason']),
                         (1,'1/5','missing_frame'))
        wide = replace(inputs, lamp_config=replace(inputs.lamp_config,max_gap=F(1,20)))
        self.assertEqual(diagnostic.lamp_diagnostics(wide)['counts']['gap_rows'], 11)
        row = inputs.rows[4]
        missing = replace(inputs,rows=(*inputs.rows[:4],replace(row,vehicle=None),*inputs.rows[5:]))
        report = diagnostic.lamp_diagnostics(missing)
        self.assertEqual(report['rows'][4]['left']['geometry_status'], 'identity_uncertain')
        self.assertIn('no_actual_observation',report['rows'][4]['history_break_reasons'])
        self.assertIn('actual_observation_resumed',report['rows'][5]['history_break_reasons'])
        target = replace(row.vehicle,detection=replace(row.vehicle.detection,model_id='changed'))
        changed = replace(inputs,rows=(*inputs.rows[:4],replace(row,vehicle=target),*inputs.rows[5:]))
        self.assertIn('identity_or_model_change',diagnostic.lamp_diagnostics(changed)['rows'][4]['history_break_reasons'])

    def test_summary_redacts_all_identifiers_and_returns_owned_lists(self):
        inputs = with_lamp_evidence(fixture())
        report = diagnostic.lamp_diagnostics(inputs)
        summary = diagnostic.lamp_diagnostics_summary(report)
        text = json.dumps(summary)
        for secret in (inputs.sha256, inputs.rows[0].frame.source_id, 'scripted-lamp-model-v1',
                       'scripted-lamp-context-v1','left-turn','roi','vehicle_identity','coded_size'):
            self.assertNotIn(secret,text)
        self.assertEqual(summary['timeline'][3]['time'],'3/10')
        summary['timeline'][0]['history_break_reasons'].clear()
        self.assertEqual(report['rows'][0]['history_break_reasons'], ['start'])

    def test_readonly_load_nested_tampering_types_counts_limits_and_input_hash(self):
        inputs = with_lamp_evidence(fixture())
        report = diagnostic.lamp_diagnostics(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            raw = json.dumps(report,indent=2).encode()
            path.write_bytes(raw)
            stamp = path.stat().st_mtime_ns
            self.assertEqual(diagnostic.load_lamp_diagnostics(path,inputs),report)
            self.assertEqual((path.read_bytes(),path.stat().st_mtime_ns),(raw,stamp))
            changes = [(('extra',),1),(('counts','input_rows'),True),
                (('rows',0,'left','whole_pixels'),16.),(('rows',0,'source','rear_view'),False),
                (('rows',0,'frame','pts'),True),(('rows',0,'history_break_reasons'),[]),
                (('inputs_sha256',),'b'*64),(('limits',),[])]
            for keys,value in changes:
                data = deepcopy(report)
                target = data
                for key in keys[:-1]: target = target[key]
                target[keys[-1]] = value
                path.write_bytes(json.dumps(data).encode())
                with self.subTest(keys=keys),self.assertRaises(AppError):
                    diagnostic.load_lamp_diagnostics(path,inputs)
            path.write_bytes(raw)
            with self.assertRaises(AppError): diagnostic.load_lamp_diagnostics(path,fixture())

    def test_reader_bounds_nonfinite_duplicate_unicode_changed_io_and_remote(self):
        inputs = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            for raw in (b'',b'x'*(MAX_BYTES+1),b'\xff',b'{"x":NaN}',b'{"x":Infinity}',
                        b'{"x":{"a":1,"a":2}}',b'['*1500+b']'*1500):
                path.write_bytes(raw)
                with self.assertRaises(AppError): diagnostic.load_lamp_diagnostics(path,inputs)
            path.write_bytes(diagnostics_bytes(diagnostic.lamp_diagnostics(inputs)))
            with patch.object(diagnostic,'_identity',side_effect=[(1,),(2,)]),self.assertRaises(AppError) as err:
                diagnostic.load_lamp_diagnostics(path,inputs)
            self.assertEqual(err.exception.code,'INPUT_CHANGED')
            with patch('pathlib.Path.open',side_effect=OSError()),self.assertRaises(AppError) as err:
                diagnostic.load_lamp_diagnostics(path,inputs)
            self.assertEqual(err.exception.code,'IO_ERROR')
        with self.assertRaises(AppError): diagnostic.load_lamp_diagnostics('https://example.test/a.json',inputs)
        with patch.object(diagnostic,'MAX_BYTES',1),self.assertRaises(AppError): diagnostic.lamp_diagnostics(inputs)
        with self.assertRaises(AppError): diagnostic.lamp_diagnostics(object())

    def test_pure_replay_and_reader_have_no_ml_dependency(self):
        result = subprocess.run([sys.executable,'-B','-c',
            'import sys; from cheating_racer_detect.events.inputs import EventInputs; '
            'from cheating_racer_detect.events.lamp_diagnostics import lamp_diagnostics; '
            'i=EventInputs.from_bytes(sys.stdin.buffer.read()); '
            'assert lamp_diagnostics(i)["counts"]["input_rows"]==12; '
            'assert not any(n in sys.modules for n in ("numpy","cv2","torch"))'],
            input=with_lamp_evidence(fixture()).to_bytes(),capture_output=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr.decode())


class Transactions(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root/'source.mp4'
        self.source.write_bytes(b'own mock source')
        self.inputs = with_lamp_evidence(fixture(catalog._sha256(self.source)))
        value = document(self.inputs)
        value.update(candidate_count=0,candidates=[])
        self.catalog = receipt(value)
        self.enterContext(patch.object(replay,'analyze_inputs',return_value=replay.ReplayAnalysis((),None)))
        self.enterContext(patch.object(catalog,'inspect_source',return_value=(None,None,object())))
        self.enterContext(patch.object(catalog,'make_catalog',return_value=self.catalog))

    def invoke(self, both=False):
        return catalog.export_catalog_with_lamp_diagnostics(self.source,self.inputs,self.root/'out',
            toolchain=object(),include_lane_diagnostics=both)

    def test_three_or_four_files_default_and_lane_apis_unchanged(self):
        for both in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                item,lamps,lanes = catalog.export_catalog_with_lamp_diagnostics(self.source,self.inputs,
                    Path(directory)/'list',toolchain=object(),include_lane_diagnostics=both)
                names = {'inputs.json','catalog.json','lamp-diagnostics.json'}
                if both: names.add('lane-diagnostics.json')
                self.assertEqual({p.name for p in (Path(directory)/'list').iterdir()},names)
                self.assertEqual(lanes is not None,both)
                self.assertEqual(diagnostic.load_lamp_diagnostics(Path(directory)/'list/lamp-diagnostics.json',self.inputs),lamps)
                self.assertEqual(item.to_bytes(),self.catalog.to_bytes())
        item = catalog.export_catalog(self.source,self.inputs,self.root/'plain',toolchain=object())
        self.assertNotIn('lamp_diagnostics',catalog.listing_summary(item))
        self.assertEqual({p.name for p in (self.root/'plain').iterdir()},{'inputs.json','catalog.json'})
        with self.assertRaises(AppError):
            catalog.export_catalog_with_lamp_diagnostics(self.source,self.inputs,self.root/'invalid',include_lane_diagnostics=1)

    def test_failure_cancel_both_sidecar_writes_cleanup_retry_and_existing_output(self):
        write = catalog.write_bytes
        for filename in ('lamp-diagnostics.json','lane-diagnostics.json'):
            for failure in (OSError('disk'),KeyboardInterrupt()):
                def fail(path,data):
                    if path.name==filename: raise failure
                    return write(path,data)
                with patch.object(catalog,'write_bytes',side_effect=fail),self.assertRaises((AppError,KeyboardInterrupt)):
                    self.invoke(True)
                self.assertFalse((self.root/'out').exists())
                self.assertEqual(list(self.root.glob('*.partial')),[])
        self.invoke()
        before = (self.root/'out/lamp-diagnostics.json').read_bytes()
        with self.assertRaises(AppError): self.invoke()
        self.assertEqual((self.root/'out/lamp-diagnostics.json').read_bytes(),before)

    def test_source_reset_drift_rejected_before_publication(self):
        value = replay.ReplayAnalysis((),None,lamp_resets=((self.inputs.rows[0].frame,'fake'),))
        with patch.object(replay,'analyze_inputs',return_value=value),self.assertRaises(AppError) as err:
            self.invoke()
        self.assertEqual(err.exception.code,'INVALID_DIAGNOSTICS')
        self.assertFalse((self.root/'out').exists())

    def test_source_change_after_sidecar_and_pixel_cleanup_failure_allow_retry(self):
        write = catalog.write_bytes
        def change(path,data):
            write(path,data)
            if path.name=='lamp-diagnostics.json': self.source.write_bytes(b'changed own source')
        with patch.object(catalog,'write_bytes',side_effect=change),self.assertRaises(AppError) as err:
            self.invoke()
        self.assertEqual(err.exception.code,'INPUT_CHANGED')
        self.assertFalse((self.root/'out').exists())
        self.source.write_bytes(b'own mock source')
        with patch.object(catalog,'release_pixels',side_effect=AppError('CLEANUP_FAILED','test')),self.assertRaises(AppError):
            self.invoke()
        self.assertFalse((self.root/'out').exists())
        self.invoke()

    def test_source_and_installed_cli_both_flags_redacted_and_cancel_protocol(self):
        lamps = diagnostic.lamp_diagnostics(self.inputs)
        args = ['--input','mock.mp4','--inputs','mock.json','--output','out',
                '--lamp-diagnostics','--lane-diagnostics']
        with patch('scripts.list_events.load_inputs',return_value=self.inputs), \
             patch('scripts.list_events.export_catalog_with_lamp_diagnostics',return_value=(self.catalog,lamps,None)) as call, \
             patch('sys.stdout',new_callable=io.StringIO) as out:
            self.assertEqual(source_main(args),0)
            self.assertIn('lamp_diagnostics',json.loads(out.getvalue()))
            self.assertNotIn(self.inputs.sha256,out.getvalue())
            self.assertTrue(call.call_args.kwargs['include_lane_diagnostics'])
        for error,code in ((AppError('INVALID_DIAGNOSTICS','private'),1),(KeyboardInterrupt(),130)):
            with patch('scripts.list_events.load_inputs',return_value=self.inputs), \
                 patch('scripts.list_events.export_catalog_with_lamp_diagnostics',side_effect=error), \
                 patch('sys.stdout',new_callable=io.StringIO) as out:
                self.assertEqual(source_main(args),code)
                self.assertNotIn('private',out.getvalue())


if __name__=='__main__':
    unittest.main()
