"""Real encoded synthetic sources; no trained model or visibility claims."""

from dataclasses import replace
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.analysis import DetectorTrackerBridge
from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events.capture import capture_inputs
from cheating_racer_detect.events.catalog import export_catalog
from cheating_racer_detect.events.inputs import EventInputs, load_inputs, save_inputs
from cheating_racer_detect.events.replay import analyze_inputs, export_replay_batch
from cheating_racer_detect.events.verify import verify_batch
from cheating_racer_detect.service import _sha256
from cheating_racer_detect.tracking import Tracker
from scripts.demo_batch_replay import generate_inputs
from scripts.demo_lane_replay import with_lane_evidence
from scripts.demo_events import event_configs
from tests.test_capture_native import AVAILABLE, TOOLS, Detector
from tests.test_lane_sections import context, scripted
from tests.test_tracking import policy
from tests import fixtures
from cheating_racer_detect.review.video import prepare
from cheating_racer_detect.tracking import Frame


@unittest.skipUnless(AVAILABLE and TOOLS, 'NOT_RUN: FFmpeg/NumPy/OpenCV required')
class NativeEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='crd-lane-recorded-')
        cls.root = Path(cls.temporary.name)
        cls.source, cls.legacy = generate_inputs(TOOLS, cls.root)
        cls.inputs = with_lane_evidence(cls.legacy)
        cls.original_hash = _sha256(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        if cls.root.exists():
            raise AssertionError('Owned native fixture not removed')

    def tearDown(self):
        self.assertEqual(_sha256(self.source), self.original_hash)

    def test_v2_six_actual_source_profiles_keep_pts_origin_rotation_and_legacy_results(self):
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
                    legacy = replace(self.legacy,rows=tuple(rows))
                    inputs = with_lane_evidence(legacy)
                    save_inputs(inputs,root/(name+'-recorded'))
                    restored = load_inputs(root/(name+'-recorded')/'inputs.json')
                    actual = analyze_inputs(source,restored,TOOLS)
                    self.assertEqual(actual.candidates,analyze_inputs(source,legacy,TOOLS).candidates)
                    self.assertEqual(restored.rows[0].frame.first_pts,timeline.frames[0].pts)
                    self.assertEqual(restored.rows[0].frame.display_rotation,rotation)
                    self.assertEqual(tuple(r.frame.pts for r in restored.rows),tuple(f.pts for f in selected))
                    self.assertEqual(_sha256(source),digest)

    def test_saved_v2_catalog_reverse_batch_readonly_offline_source_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_inputs(self.inputs, root/'saved')
            restored = load_inputs(root/'saved/inputs.json')
            analysis = analyze_inputs(self.source, restored, TOOLS)
            legacy = analyze_inputs(self.source, self.legacy, TOOLS)
            self.assertEqual(len(analysis.candidates), 2)
            self.assertEqual(analysis.candidates, legacy.candidates)
            self.assertEqual(analysis.lane_resets, ())
            catalog = export_catalog(self.source, restored, root/'listed', toolchain=TOOLS)
            result = export_replay_batch(self.source, restored, root/'batch', candidate_indices=(1,0),
                                         catalog=catalog, toolchain=TOOLS)
            self.assertEqual(result['candidate_indices'], [1,0])
            self.assertEqual(load_inputs(root/'batch/inputs.json'), self.inputs)
            before = {p.relative_to(root/'batch'): (p.read_bytes(),p.stat().st_mtime_ns)
                      for p in (root/'batch').rglob('*') if p.is_file()}
            for source in (None,self.source):
                verified = verify_batch(root/'batch', source=source, toolchain=TOOLS)
                self.assertEqual(verified['status'], 'verified')
            args = [sys.executable,'-B','-m','cheating_racer_detect','events','verify',
                    '--bundle',str(root/'batch'),'--input',str(self.source),
                    '--ffmpeg-dir',str(TOOLS.ffmpeg.parent)]
            cli = subprocess.run(args, capture_output=True, timeout=45)
            self.assertEqual(cli.returncode, 0, cli.stderr.decode())
            self.assertEqual(json.loads(cli.stdout)['status'], 'verified')
            after = {p.relative_to(root/'batch'): (p.read_bytes(),p.stat().st_mtime_ns)
                     for p in (root/'batch').rglob('*') if p.is_file()}
            self.assertEqual(before, after)

    def test_replayed_source_reset_changes_catalog_and_offline_candidate_reconstruction(self):
        rows = tuple(replace(row, lane_evidence=replace(row.lane_evidence,
                     context=replace(row.lane_evidence.context, context_id='second-context')))
                     if n >= 5 else row for n,row in enumerate(self.inputs.rows))
        changed = EventInputs.from_bytes(replace(self.inputs, rows=rows).to_bytes())
        analysis = analyze_inputs(self.source, changed, TOOLS)
        self.assertEqual(analysis.lane_resets, ((rows[5].frame,'lane_source_change'),))
        self.assertEqual(len(analysis.candidates), 1)
        self.assertEqual(analysis.candidates[0].direction,'left')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = export_catalog(self.source, changed, root/'list', toolchain=TOOLS)
            export_replay_batch(self.source, changed, root/'batch', candidate_indices=(0,), catalog=catalog, toolchain=TOOLS)
            self.assertEqual(verify_batch(root/'batch', toolchain=TOOLS)['status'],'verified')
            self.assertEqual(verify_batch(root/'batch', source=self.source, toolchain=TOOLS)['status'],'verified')

    def test_provenance_edit_or_strip_invalidates_existing_catalog_and_bundle_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = export_catalog(self.source, self.inputs, root/'list', toolchain=TOOLS)
            rows = tuple(replace(r,lane_evidence=None) for r in self.inputs.rows)
            stripped = replace(self.inputs,rows=rows)
            changed_rows = tuple(replace(r,lane_evidence=replace(r.lane_evidence,
                                 context=replace(r.lane_evidence.context,context_id='new-context')))
                                 for r in self.inputs.rows)
            for inputs in (stripped,replace(self.inputs,rows=changed_rows)):
                with self.assertRaises(AppError) as error:
                    export_replay_batch(self.source, inputs, root/'bad', candidate_indices=(0,), catalog=catalog, toolchain=TOOLS)
                self.assertEqual(error.exception.code,'CATALOG_CHANGED')
                self.assertFalse((root/'bad').exists())
            export_replay_batch(self.source,self.inputs,root/'batch',candidate_indices=(0,),toolchain=TOOLS)
            (root/'batch/inputs.json').write_bytes(stripped.to_bytes())
            with self.assertRaises(AppError): verify_batch(root/'batch',toolchain=TOOLS)

    def capture(self, output, callback, gates=None):
        detector = Detector()
        bridge = DetectorTrackerBridge(detector,Tracker(policy()),model_alias='scripted',expected_detector_id=detector.model_id)
        lane,lamp = event_configs()
        def run():
            return capture_inputs(self.source,'0','1.2',output,bridge,
                selected_identity=(1,2,1),lane_config=lane,lamp_config=lamp,pre=F(1,5),post=F(2,5),
                gates=gates,lane_observations=callback,toolchain=TOOLS)
        return bridge,run

    @staticmethod
    def observation(frame,vehicle):
        if vehicle is None: return None,None
        batch = scripted(frame)
        return batch,context(frame,vehicle,batch)

    def test_capture_persists_full_v2_actual_and_explicit_unknown_with_source_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _bridge,run = self.capture(root/'capture',self.observation)
            inputs = run()
            self.assertEqual(inputs,load_inputs(root/'capture/inputs.json'))
            self.assertEqual(inputs.report()['schema'],'experimental-event-inputs-v2')
            self.assertIsNone(inputs.rows[3].vehicle)
            self.assertIsNone(inputs.rows[3].lane_evidence.batch)
            self.assertIsNotNone(inputs.rows[2].lane_evidence.context)
            receipt = json.loads((root/'capture/capture.json').read_bytes())
            self.assertEqual(receipt['inputs_sha256'],inputs.sha256)
            analysis = analyze_inputs(self.source,inputs,TOOLS)
            self.assertIn((inputs.rows[3].frame,'no_actual_observation'),analysis.lane_resets)

    def test_capture_callback_failure_cancel_late_write_and_fresh_retry_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for failure in (ValueError('lane callback'),KeyboardInterrupt()):
                def bad(frame,vehicle):
                    if frame.ordinal==5: raise failure
                    return self.observation(frame,vehicle)
                bridge,run = self.capture(root/'capture',bad)
                with self.assertRaises(type(failure)): run()
                self.assertIsNone(bridge.tracker._previous)
                self.assertEqual(list(root.iterdir()),[])
            bridge,run = self.capture(root/'capture',self.observation)
            with patch('cheating_racer_detect.events.capture.write_bytes',side_effect=OSError('disk')):
                with self.assertRaises(AppError): run()
            self.assertIsNone(bridge.tracker._previous)
            self.assertEqual(list(root.iterdir()),[])
            _bridge,run = self.capture(root/'capture',self.observation)
            saved = run().to_bytes()
            with self.assertRaises(AppError): run()
            self.assertEqual((root/'capture/inputs.json').read_bytes(),saved)
            self.assertEqual(list(root.glob('*.partial')),[])

    def test_capture_stale_or_duplicate_geometry_source_rejected(self):
        from cheating_racer_detect.events.lamp import LampROI
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def stale(frame,vehicle):
                batch,ctx = self.observation(frame,vehicle)
                if batch is None: return None,None
                foreign = replace(frame,pts=frame.pts+1)
                return replace(batch,frame=foreign),replace(ctx,frame=foreign)
            def duplicate(frame,vehicle):
                batch,ctx = self.observation(frame,vehicle)
                lane = None if batch is None else self.inputs.rows[frame.ordinal].lanes
                return lane,LampROI(None,'model_uncertain'),LampROI(None,'model_uncertain')
            for cb,gates in ((stale,None),(self.observation,duplicate),(lambda *_:iter((None,None)),None)):
                bridge,run = self.capture(root/'capture',cb,gates)
                with self.assertRaises(ValueError): run()
                self.assertIsNone(bridge.tracker._previous)
                self.assertEqual(list(root.iterdir()),[])


if __name__=='__main__':
    unittest.main()
