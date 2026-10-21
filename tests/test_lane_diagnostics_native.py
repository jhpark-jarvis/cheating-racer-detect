"""Real synthetic media -> optional diagnostic listing -> unchanged selection."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import LaneEvidence, catalog, replay
from cheating_racer_detect.events.diagnostics import load_lane_diagnostics
from cheating_racer_detect.events.inputs import save_inputs
from cheating_racer_detect.events.verify import verify_batch
from cheating_racer_detect.review.video import prepare
from cheating_racer_detect.service import _sha256
from cheating_racer_detect.tracking import Frame
from scripts.demo_batch_replay import generate_inputs
from scripts.demo_lane_replay import with_lane_evidence
from tests.test_capture_native import AVAILABLE, TOOLS
from tests import fixtures


@unittest.skipUnless(AVAILABLE and TOOLS,'NOT_RUN: FFmpeg/NumPy/OpenCV required')
class NativeDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='crd-lane-diagnostics-')
        cls.root = Path(cls.temporary.name)
        try:
            cls.source,cls.legacy = generate_inputs(TOOLS,cls.root)
            cls.inputs = with_lane_evidence(cls.legacy)
            cls.digest = _sha256(cls.source)
        except BaseException:
            cls.temporary.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        if cls.root.exists(): raise AssertionError('Owned fixture remains')

    def tearDown(self):
        self.assertEqual(_sha256(self.source),self.digest)

    def test_one_source_analysis_same_catalog_readonly_load_then_selected_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(replay,'analyze_inputs',wraps=replay.analyze_inputs) as analyze:
                item,report = catalog.export_catalog_with_diagnostics(self.source,self.inputs,root/'list',toolchain=TOOLS)
                self.assertEqual(analyze.call_count,1)
            plain = catalog.export_catalog(self.source,self.inputs,root/'plain',toolchain=TOOLS)
            self.assertEqual(item.to_bytes(),plain.to_bytes())
            self.assertEqual(report['counts']['candidate_count'],2)
            self.assertEqual(report['counts']['section_ready_rows'],44)
            path = root/'list/lane-diagnostics.json'
            before,stamp = path.read_bytes(),path.stat().st_mtime_ns
            self.assertEqual(load_lane_diagnostics(path,self.inputs),report)
            self.assertEqual((path.read_bytes(),path.stat().st_mtime_ns),(before,stamp))
            replay.export_replay_batch(self.source,self.inputs,root/'batch',candidate_indices=(1,0),catalog=item,toolchain=TOOLS)
            self.assertFalse((root/'batch/lane-diagnostics.json').exists())
            for source in (None,self.source):
                self.assertEqual(verify_batch(root/'batch',source=source,toolchain=TOOLS)['status'],'verified')

    def test_actual_package_entrypoint_and_source_cli_opt_in_redacted_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_inputs(self.inputs,root/'recorded')
            base = ['--input',str(self.source),'--inputs',str(root/'recorded/inputs.json'),'--lane-diagnostics']
            installed = subprocess.run([sys.executable,'-B','-m','cheating_racer_detect','events','list',
                *base,'--output',str(root/'installed'),'--ffmpeg-dir',str(TOOLS.ffmpeg.parent)],
                capture_output=True,timeout=45)
            source = subprocess.run([sys.executable,'-B','-m','scripts.list_events',*base,'--output',str(root/'source')],
                                    capture_output=True,timeout=45)
            for process in (installed,source):
                self.assertEqual(process.returncode,0,process.stderr.decode())
                summary = json.loads(process.stdout)
                self.assertEqual(summary['lane_diagnostics']['counts']['candidate_count'],2)
                self.assertEqual(len(summary['lane_diagnostics']['timeline']),44)
                self.assertNotIn(self.digest,process.stdout.decode())
                self.assertNotIn(str(self.source),process.stdout.decode())
                self.assertNotIn('scripted-lanes-v1',process.stdout.decode())
            self.assertEqual(json.loads(installed.stdout),json.loads(source.stdout))
            self.assertEqual((root/'installed/lane-diagnostics.json').read_bytes(),(root/'source/lane-diagnostics.json').read_bytes())

    def test_zero_candidates_unknown_and_changed_context_still_have_timeline(self):
        unknown = replace(self.inputs,rows=tuple(replace(r,lanes=None,lane_evidence=LaneEvidence(None,None)) for r in self.inputs.rows))
        changed = replace(self.inputs,rows=tuple(replace(r,lane_evidence=replace(r.lane_evidence,
                    context=replace(r.lane_evidence.context,context_id='changed-context')))
                    if i>=5 else r for i,r in enumerate(self.inputs.rows)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name,inputs,count,resets in (('unknown',unknown,0,44),('changed',changed,1,1)):
                item,report = catalog.export_catalog_with_diagnostics(self.source,inputs,root/name,toolchain=TOOLS)
                self.assertEqual(item.report()['candidate_count'],count)
                self.assertEqual(report['counts']['candidate_count'],count)
                self.assertEqual(report['counts']['source_resets'],resets)
                self.assertEqual(len(report['rows']),44)
                self.assertEqual(load_lane_diagnostics(root/name/'lane-diagnostics.json',inputs),report)
            self.assertEqual(report['rows'][5]['source_reset_reason'],'lane_source_change')

    def test_actual_failure_cancel_third_write_source_pixels_cleanup_and_retry(self):
        from cheating_racer_detect.events import diagnostics
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def run():
                return catalog.export_catalog_with_diagnostics(self.source,self.inputs,root/'out',toolchain=TOOLS)
            with patch('cheating_racer_detect.pixels.tempfile.gettempdir',return_value=str(root)):
                for failure in (RuntimeError('injected'),KeyboardInterrupt()):
                    with patch.object(diagnostics,'_replay_diagnostics',side_effect=failure),self.assertRaises(type(failure)):
                        run()
                    self.assertEqual(list(root.iterdir()),[])
                write = catalog.write_bytes
                def fail(path,data):
                    if path.name=='lane-diagnostics.json': raise OSError('disk full')
                    return write(path,data)
                with patch.object(catalog,'write_bytes',side_effect=fail),self.assertRaises(AppError): run()
                self.assertEqual(list(root.iterdir()),[])
                run()
                before = (root/'out/lane-diagnostics.json').read_bytes()
                with self.assertRaises(AppError): run()
                self.assertEqual((root/'out/lane-diagnostics.json').read_bytes(),before)
                self.assertEqual({p.name for p in root.iterdir()},{'out'})

    def test_six_real_profiles_exact_original_frames_offsets_rotation_and_sparse_gap(self):
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
                    inputs = with_lane_evidence(replace(self.legacy,rows=tuple(rows)))
                    removed = len(inputs.rows)>6
                    if removed:
                        inputs = replace(inputs,rows=(*inputs.rows[:5],*inputs.rows[6:]))
                    _item,report = catalog.export_catalog_with_diagnostics(source,inputs,root/(name+'-list'),toolchain=TOOLS)
                    self.assertEqual([row['frame'] for row in report['rows']],[row.frame.report() for row in inputs.rows])
                    if removed:
                        self.assertEqual(report['rows'][5]['lane']['reason'],'missing_frame')
                    self.assertEqual(_sha256(source),digest)


if __name__=='__main__':
    unittest.main()
