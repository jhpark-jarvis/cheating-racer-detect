"""Bounded input-derived diagnostics, legacy compatibility and atomic opt-in."""

from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import catalog, diagnostics, replay
from cheating_racer_detect.events.inputs import EventInputs, MAX_BYTES
from scripts.list_events import main as source_main
from tests.test_candidate_catalog import document, receipt
from tests.test_event_inputs import fixture
from tests.test_lane_evidence import scenario, changed_row
from scripts.demo_lane_replay import with_lane_evidence


class Diagnostics(unittest.TestCase):
    def test_legacy_provenance_not_recorded_is_not_v2_visibility(self):
        report = diagnostics.lane_diagnostics(fixture())
        self.assertEqual(report['counts']['legacy_rows'],12)
        self.assertEqual(report['counts']['section_ready_rows'],0)
        self.assertEqual(report['counts']['section_unknown_rows'],0)
        self.assertTrue(all(row['source'] is None and row['section'] is None for row in report['rows']))
        self.assertTrue(all(row['section_status']=='not_recorded'
                            for row in diagnostics.diagnostics_summary(report)['timeline']))

    def test_v2_candidates_exact_frames_samples_and_owned_reports(self):
        inputs = scenario()
        report = diagnostics.lane_diagnostics(inputs)
        self.assertEqual(report['counts']['section_ready_rows'],12)
        self.assertEqual(report['counts']['candidate_count'],1)
        self.assertEqual(report['counts']['source_resets'],0)
        self.assertEqual(report['rows'][3]['frame'],inputs.rows[3].frame.report())
        self.assertEqual(report['rows'][3]['source']['context_id'],'scripted-context-v1')
        self.assertEqual(len(report['rows'][0]['section']['samples']),3)
        report['rows'][0]['source']['bindings'][0]['visible_segments'][0] = False
        self.assertTrue(diagnostics.lane_diagnostics(inputs)['rows'][0]['source']['bindings'][0]['visible_segments'][0])

    def test_source_change_report_and_candidate_abstention(self):
        inputs = scenario()
        rows = tuple(changed_row(r,context=replace(r.lane_evidence.context,context_id='new-context'))
                     if i>=5 else r for i,r in enumerate(inputs.rows))
        report = diagnostics.lane_diagnostics(replace(inputs,rows=rows))
        self.assertEqual(report['counts']['candidate_count'],0)
        self.assertEqual(report['counts']['source_resets'],1)
        self.assertEqual(report['reason_counts']['source_reset'],{'lane_source_change':1})
        self.assertEqual(report['rows'][5]['source_reset_reason'],'lane_source_change')
        self.assertEqual(report['rows'][5]['source']['context_id'],'new-context')

    def test_unknown_reason_and_partial_geometry_kept_when_zero_candidates(self):
        inputs = scenario()
        target = inputs.rows[5]
        context = target.lane_evidence.context
        binding = replace(context.bindings[1],visible_segments=(False,)*71)
        context = replace(context,bindings=(context.bindings[0],binding,*context.bindings[2:]))
        changed = changed_row(target,context=context)
        report = diagnostics.lane_diagnostics(replace(inputs,rows=(*inputs.rows[:5],changed,*inputs.rows[6:])))
        self.assertEqual(report['counts']['candidate_count'],0)
        self.assertEqual(report['counts']['section_unknown_rows'],1)
        row = report['rows'][5]
        self.assertEqual(row['section']['reason'],'uncertified_intersection')
        self.assertEqual(len(row['section']['samples']),1)
        self.assertIsNone(row['section']['section'])
        self.assertEqual(row['lane']['status'],'unknown')

    def test_missing_rows_not_interpolated_and_clock_is_original(self):
        inputs = scenario()
        report = diagnostics.lane_diagnostics(replace(inputs,rows=(*inputs.rows[:5],*inputs.rows[6:])))
        self.assertEqual(report['counts']['input_rows'],11)
        self.assertEqual(report['rows'][5]['frame']['ordinal'],6)
        self.assertEqual(report['rows'][5]['lane']['reason'],'missing_frame')
        self.assertEqual(report['rows'][5]['frame']['pts'],inputs.rows[6].frame.pts)

    def test_stdout_projection_no_paths_hashes_identity_model_or_context(self):
        inputs = scenario()
        report = diagnostics.lane_diagnostics(inputs)
        summary = diagnostics.diagnostics_summary(report)
        text = json.dumps(summary)
        for private in (inputs.sha256,inputs.rows[0].frame.source_id,'scripted-lanes-v1',
                        'scripted-context-v1','boundary_id','vehicle_identity','road_id','coded_size'):
            self.assertNotIn(private,text)
        self.assertEqual(summary['timeline'][2]['time'],'1/5')
        summary['counts']['candidate_count']=999
        self.assertEqual(report['counts']['candidate_count'],1)

    def test_valid_load_is_readonly_and_accepts_only_input_consistent_values(self):
        inputs = scenario()
        report = diagnostics.lane_diagnostics(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'diagnostics.json'
            data = json.dumps(report,indent=2).encode()
            path.write_bytes(data)
            stamp = path.stat().st_mtime_ns
            self.assertEqual(diagnostics.load_lane_diagnostics(path,inputs),report)
            self.assertEqual(path.read_bytes(),data)
            self.assertEqual(path.stat().st_mtime_ns,stamp)
            with self.assertRaises(AppError): diagnostics.load_lane_diagnostics(path,fixture())

    def test_nested_drift_unknown_fields_false_counts_and_markers_rejected(self):
        report = diagnostics.lane_diagnostics(scenario())
        paths = [(('inputs_sha256',),'0'*64),(('extra',),1),
                 (('counts','source_resets'),1),(('rows',0,'frame','pts'),True),
                 (('rows',0,'source','model_id'),'fake'),
                 (('rows',0,'source_reset_reason'),'hidden'),
                 (('rows',0,'section','samples',0,'weight'),0.)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'diagnostics.json'
            for keys,value in paths:
                current = deepcopy(report)
                target = current
                for key in keys[:-1]: target = target[key]
                target[keys[-1]] = value
                path.write_bytes(json.dumps(current).encode())
                with self.subTest(path=keys),self.assertRaises(AppError):
                    diagnostics.load_lane_diagnostics(path,scenario())

    def test_size_depth_nonfinite_duplicate_unicode_and_remote_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.json'
            for data in (b'',b'x'*(MAX_BYTES+1),b'\xff',b'{"x":NaN}',b'{"x":Infinity}',
                         b'{"a":{"x":1,"x":2}}',b'['*1500+b']'*1500):
                path.write_bytes(data)
                with self.assertRaises(AppError): diagnostics.load_lane_diagnostics(path,scenario())
            for value in ('https://example.test/a.json',path.parent/'../bad.json',path.parent/'missing.json'):
                with self.assertRaises(AppError): diagnostics.load_lane_diagnostics(value,scenario())
        with patch.object(diagnostics,'MAX_BYTES',1),self.assertRaises(AppError):
            diagnostics.lane_diagnostics(scenario())

    def test_changed_receipt_while_reading_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'diagnostics.json'
            path.write_bytes(diagnostics.diagnostics_bytes(diagnostics.lane_diagnostics(scenario())))
            with patch.object(diagnostics,'_identity',side_effect=[(1,2,3),(1,2,4)]),self.assertRaises(AppError) as error:
                diagnostics.load_lane_diagnostics(path,scenario())
            self.assertEqual(error.exception.code,'INPUT_CHANGED')

    def test_pure_replay_and_parse_import_no_numpy_cv_torch(self):
        process = subprocess.run([sys.executable,'-B','-c',
            'import sys; from cheating_racer_detect.events.inputs import EventInputs; '
            'from cheating_racer_detect.events.diagnostics import lane_diagnostics; '
            'p=EventInputs.from_bytes(sys.stdin.buffer.read()); '
            'assert lane_diagnostics(p)["counts"]["candidate_count"]==1; '
            'assert not any(n in sys.modules for n in ("numpy","cv2","torch"))'],
            input=scenario().to_bytes(),capture_output=True,timeout=20)
        self.assertEqual(process.returncode,0,process.stderr.decode())


class Transactions(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root/'source.mp4'
        self.source.write_bytes(b'own mock source')
        self.inputs = with_lane_evidence(fixture(catalog._sha256(self.source)))
        value = document(self.inputs)
        value.update(candidate_count=0,candidates=[])
        self.catalog = receipt(value)
        self.analysis = replay.ReplayAnalysis((),None)
        self.enterContext(patch.object(replay,'analyze_inputs',return_value=self.analysis))
        self.enterContext(patch.object(catalog,'inspect_source',return_value=(None,None,object())))
        self.enterContext(patch.object(catalog,'make_catalog',return_value=self.catalog))

    def invoke(self):
        return catalog.export_catalog_with_diagnostics(self.source,self.inputs,self.root/'out',toolchain=object())

    def test_explicit_atomic_three_files_without_schema_or_default_changes(self):
        item,report = self.invoke()
        self.assertEqual(item,self.catalog)
        self.assertEqual({p.name for p in (self.root/'out').iterdir()},
                         {'inputs.json','catalog.json','lane-diagnostics.json'})
        self.assertEqual(diagnostics.load_lane_diagnostics(self.root/'out/lane-diagnostics.json',self.inputs),report)
        plain = catalog.export_catalog(self.source,self.inputs,self.root/'plain',toolchain=object())
        self.assertEqual(plain.to_bytes(),item.to_bytes())
        self.assertEqual({p.name for p in (self.root/'plain').iterdir()},{'inputs.json','catalog.json'})
        self.assertNotIn('lane_diagnostics',catalog.listing_summary(plain))

    def test_third_write_failure_cancel_cleanup_and_fresh_retry_existing_preserved(self):
        write = catalog.write_bytes
        for failure in (OSError('disk'),KeyboardInterrupt()):
            def fail(path,data):
                if path.name=='lane-diagnostics.json': raise failure
                return write(path,data)
            with patch.object(catalog,'write_bytes',side_effect=fail),self.assertRaises((AppError,KeyboardInterrupt)):
                self.invoke()
            self.assertFalse((self.root/'out').exists())
            self.assertEqual(list(self.root.glob('*.partial')),[])
        self.invoke()
        before = (self.root/'out/lane-diagnostics.json').read_bytes()
        with self.assertRaises(AppError): self.invoke()
        self.assertEqual((self.root/'out/lane-diagnostics.json').read_bytes(),before)

    def test_fresh_source_analysis_candidate_or_reset_drift_before_stage(self):
        for analysis in (replay.ReplayAnalysis((object(),),None),
                         replay.ReplayAnalysis((),None,((self.inputs.rows[0].frame,'fake'),))):
            with patch.object(replay,'analyze_inputs',return_value=analysis),self.assertRaises(AppError) as error:
                self.invoke()
            self.assertEqual(error.exception.code,'INVALID_DIAGNOSTICS')
            self.assertFalse((self.root/'out').exists())

    def test_pixel_cleanup_failure_precedes_publish_and_allows_retry(self):
        with patch.object(catalog,'release_pixels',side_effect=AppError('CLEANUP_FAILED','test')),self.assertRaises(AppError):
            self.invoke()
        self.assertFalse((self.root/'out').exists())
        self.assertEqual(list(self.root.glob('*.partial')),[])
        self.invoke()

    def test_source_change_after_third_write_prevents_publication(self):
        write = catalog.write_bytes
        def change(path,data):
            write(path,data)
            if path.name=='lane-diagnostics.json': self.source.write_bytes(b'changed own source')
        with patch.object(catalog,'write_bytes',side_effect=change),self.assertRaises(AppError) as error:
            self.invoke()
        self.assertEqual(error.exception.code,'INPUT_CHANGED')
        self.assertFalse((self.root/'out').exists())
        self.assertEqual(list(self.root.glob('*.partial')),[])

    def test_source_cli_opt_in_redacts_and_preserves_error_cancel_protocol(self):
        report = diagnostics.lane_diagnostics(self.inputs)
        args = ['--input','mock.mp4','--inputs','mock.json','--output','out','--lane-diagnostics']
        with patch('scripts.list_events.load_inputs',return_value=self.inputs), \
             patch('scripts.list_events.export_catalog_with_diagnostics',return_value=(self.catalog,report)) as export, \
             patch('sys.stdout',new_callable=io.StringIO) as output:
            self.assertEqual(source_main(args),0)
            self.assertIn('lane_diagnostics',json.loads(output.getvalue()))
            self.assertNotIn(self.inputs.sha256,output.getvalue())
            export.assert_called_once()
        for error,code in ((AppError('INVALID_DIAGNOSTICS','private path'),1),(KeyboardInterrupt(),130)):
            with patch('scripts.list_events.load_inputs',return_value=self.inputs), \
                 patch('scripts.list_events.export_catalog_with_diagnostics',side_effect=error), \
                 patch('sys.stdout',new_callable=io.StringIO) as output:
                self.assertEqual(source_main(args),code)
                self.assertNotIn('private path',output.getvalue())


if __name__=='__main__':
    unittest.main()
