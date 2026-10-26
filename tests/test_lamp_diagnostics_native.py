"""Actual encoded pixels/listing/CLI with synthetic declarations, not models."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import LampEvidence, LaneEvidence, catalog, replay
from cheating_racer_detect.events.inputs import save_inputs
from cheating_racer_detect.events.lamp_diagnostics import load_lamp_diagnostics
from cheating_racer_detect.events.verify import verify_batch
from cheating_racer_detect.review.video import prepare
from cheating_racer_detect.service import _sha256
from cheating_racer_detect.tracking import Frame
from scripts.demo_batch_replay import generate_inputs
from scripts.demo_lane_replay import with_lane_evidence
from scripts.demo_lamp_replay import with_lamp_evidence
from tests.test_capture_native import AVAILABLE, TOOLS
from tests import fixtures


@unittest.skipUnless(AVAILABLE and TOOLS, 'NOT_RUN: FFmpeg/NumPy/OpenCV required')
class NativeLampDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='crd-lamp-diagnostics-')
        cls.root = Path(cls.temporary.name)
        try:
            cls.source, cls.legacy = generate_inputs(TOOLS, cls.root)
            cls.inputs = with_lamp_evidence(with_lane_evidence(cls.legacy))
            cls.digest = _sha256(cls.source)
        except BaseException:
            cls.temporary.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        if cls.root.exists(): raise AssertionError('Owned fixture remains')

    def tearDown(self):
        self.assertEqual(_sha256(self.source), self.digest)

    def test_one_analysis_same_catalog_readonly_report_and_selected_batch_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(replay,'analyze_inputs',wraps=replay.analyze_inputs) as analyze:
                item,lamps,lanes = catalog.export_catalog_with_lamp_diagnostics(
                    self.source,self.inputs,root/'list',toolchain=TOOLS,include_lane_diagnostics=True)
                self.assertEqual(analyze.call_count,1)
            plain = catalog.export_catalog(self.source,self.inputs,root/'plain',toolchain=TOOLS)
            self.assertEqual(plain.to_bytes(),item.to_bytes())
            self.assertEqual(lamps['counts']['input_rows'],44)
            self.assertEqual(lamps['counts']['source_resets'],0)
            self.assertEqual(lanes['counts']['candidate_count'],2)
            path = root/'list/lamp-diagnostics.json'
            before,stamp = path.read_bytes(),path.stat().st_mtime_ns
            self.assertEqual(load_lamp_diagnostics(path,self.inputs),lamps)
            self.assertEqual((path.read_bytes(),path.stat().st_mtime_ns),(before,stamp))
            replay.export_replay_batch(self.source,self.inputs,root/'batch',candidate_indices=(1,0),
                                      catalog=item,toolchain=TOOLS)
            for source in (None,self.source):
                self.assertEqual(verify_batch(root/'batch',source=source,toolchain=TOOLS)['status'],'verified')
            self.assertFalse((root/'batch/lamp-diagnostics.json').exists())

    def test_actual_installed_and_source_cli_both_options_identical_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_inputs(self.inputs,root/'recorded')
            args = ['--input',str(self.source),'--inputs',str(root/'recorded/inputs.json'),
                    '--lamp-diagnostics','--lane-diagnostics']
            first = subprocess.run([sys.executable,'-B','-m','cheating_racer_detect','events','list',
                *args,'--output',str(root/'installed'),'--ffmpeg-dir',str(TOOLS.ffmpeg.parent)],
                capture_output=True,timeout=45)
            second = subprocess.run([sys.executable,'-B','-m','scripts.list_events',*args,
                '--output',str(root/'source')],capture_output=True,timeout=45)
            for result in (first,second):
                self.assertEqual(result.returncode,0,result.stderr.decode())
                summary = json.loads(result.stdout)
                self.assertEqual(summary['candidate_count'],2)
                self.assertEqual(summary['lamp_diagnostics']['counts']['input_rows'],44)
                self.assertEqual(summary['lane_diagnostics']['counts']['candidate_count'],2)
                for private in (self.digest,str(self.source),'scripted-lamp-model-v1','scripted-lamp-context-v1'):
                    self.assertNotIn(private,result.stdout.decode())
            self.assertEqual(json.loads(first.stdout),json.loads(second.stdout))
            self.assertEqual((root/'installed/lamp-diagnostics.json').read_bytes(),
                             (root/'source/lamp-diagnostics.json').read_bytes())

    def test_missing_observation_source_change_sparse_gap_zero_candidates_and_legacy(self):
        rows = []
        for r in self.inputs.rows:
            o = r.lamp_evidence.observation
            evidence = LampEvidence(replace(o,context_id='changed-context') if r.frame.ordinal>=35 else o)
            rows.append(replace(r,lamp_evidence=evidence))
        changed = replace(self.inputs,rows=tuple(r for r in rows if r.frame.ordinal!=10))
        unknown = replace(self.inputs,rows=tuple(replace(r,lanes=None,lane_evidence=LaneEvidence(None,None),
            lamp_evidence=LampEvidence(None),left=LampEvidence(None).rois(r.frame,r.vehicle)[0],
            right=LampEvidence(None).rois(r.frame,r.vehicle)[1]) for r in self.inputs.rows))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name,inputs in (('changed',changed),('unknown',unknown),('legacy',self.legacy)):
                item,report,_ = catalog.export_catalog_with_lamp_diagnostics(
                    self.source,inputs,root/name,toolchain=TOOLS)
                self.assertEqual(len(report['rows']),len(inputs.rows))
                self.assertEqual(load_lamp_diagnostics(root/name/'lamp-diagnostics.json',inputs),report)
                if name=='changed':
                    self.assertEqual(report['counts']['source_resets'],1)
                    self.assertEqual(report['rows'][10]['clock_break_reason'],'missing_frame')
                    self.assertEqual(report['rows'][34]['source_reset_reason'],'lamp_source_change')
                elif name=='unknown':
                    self.assertEqual(item.report()['candidate_count'],0)
                    self.assertEqual(report['counts']['observation_unknown_rows'],44)
                    self.assertEqual(report['counts']['source_resets'],44)
                else:
                    self.assertEqual(report['counts']['legacy_rows'],44)
                    self.assertEqual(report['counts']['source_resets'],0)

    def test_failure_cancel_sidecar_and_cleanup_prevent_publish_then_retry(self):
        from cheating_racer_detect.events import lamp_diagnostics
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def run():
                return catalog.export_catalog_with_lamp_diagnostics(self.source,self.inputs,root/'out',toolchain=TOOLS)
            with patch('cheating_racer_detect.pixels.tempfile.gettempdir',return_value=str(root)):
                for error in (RuntimeError('injected'),KeyboardInterrupt()):
                    with patch.object(lamp_diagnostics,'lamp_diagnostics',side_effect=error),self.assertRaises(type(error)):
                        run()
                    self.assertEqual(list(root.iterdir()),[])
                write = catalog.write_bytes
                def fail(path,data):
                    if path.name=='lamp-diagnostics.json': raise OSError('disk')
                    return write(path,data)
                with patch.object(catalog,'write_bytes',side_effect=fail),self.assertRaises(AppError): run()
                self.assertEqual(list(root.iterdir()),[])
                with patch.object(catalog,'release_pixels',side_effect=AppError('CLEANUP_FAILED','test')),self.assertRaises(AppError): run()
                self.assertEqual(list(root.iterdir()),[])
                run()
                before = (root/'out/lamp-diagnostics.json').read_bytes()
                with self.assertRaises(AppError): run()
                self.assertEqual((root/'out/lamp-diagnostics.json').read_bytes(),before)
                self.assertEqual({p.name for p in root.iterdir()},{'out'})

    def test_six_profiles_preserve_coded_roi_exact_original_pts_and_sparse_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = fixtures.generate(TOOLS,root)
            for name in ('cfr','vfr','offset','rotation90','rotation180','rotation270'):
                with self.subTest(profile=name):
                    source = media[name]
                    digest = _sha256(source)
                    video,timeline,selected,first,rotation = prepare(source,'0','1.2',TOOLS)
                    rows = []
                    for n,item in enumerate(selected):
                        f = Frame(digest,video['index'],first+n,item.pts,timeline.time_base,
                            timeline.frames[0].pts,video['width'],video['height'],rotation)
                        r = self.legacy.rows[n]
                        v = replace(r.vehicle,frame=f,detection=replace(r.vehicle.detection,frame=f))
                        rows.append(replace(r,frame=f,vehicle=v,lanes=replace(r.lanes,frame=f)))
                    inputs = with_lamp_evidence(replace(self.legacy,rows=tuple(rows)))
                    removed = len(inputs.rows)>6
                    if removed: inputs = replace(inputs,rows=(*inputs.rows[:5],*inputs.rows[6:]))
                    _item,report,_ = catalog.export_catalog_with_lamp_diagnostics(
                        source,inputs,root/(name+'-list'),toolchain=TOOLS)
                    self.assertEqual([r['frame'] for r in report['rows']],[r.frame.report() for r in inputs.rows])
                    self.assertEqual(report['rows'][0]['left']['roi'],list(inputs.rows[0].left.box.values()))
                    if removed: self.assertEqual(report['rows'][5]['clock_break_reason'],'missing_frame')
                    self.assertEqual(_sha256(source),digest)


if __name__=='__main__':
    unittest.main()
