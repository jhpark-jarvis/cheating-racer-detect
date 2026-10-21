"""Own encoded pixels -> explicit crop facts -> v3 capture/replay/review."""

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
from cheating_racer_detect.events import LampPart,LampObservation,LampEvidence,LampROI
from cheating_racer_detect.events.capture import capture_inputs
from cheating_racer_detect.events.catalog import export_catalog_with_diagnostics
from cheating_racer_detect.events.inputs import load_inputs,save_inputs
from cheating_racer_detect.events.replay import analyze_inputs,export_replay_batch
from cheating_racer_detect.events.verify import verify_batch
from cheating_racer_detect.review.video import prepare
from cheating_racer_detect.service import _sha256
from cheating_racer_detect.tracking import Box,Frame,Tracker
from scripts.demo_batch_replay import generate_inputs
from scripts.demo_events import event_configs,generate
from scripts.demo_lane_replay import with_lane_evidence
from scripts.demo_lamp_replay import with_lamp_evidence
from tests.test_capture_native import AVAILABLE,TOOLS,Detector
from tests.test_tracking import policy
from tests import fixtures


@unittest.skipUnless(AVAILABLE and TOOLS,'NOT_RUN: FFmpeg/NumPy/OpenCV required')
class NativeLampObservations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix='crd-lamp-observations-')
        cls.root=Path(cls.temporary.name)
        try:
            cls.source,cls.legacy=generate_inputs(TOOLS,cls.root)
            cls.digest=_sha256(cls.source)
        except BaseException:
            cls.temporary.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        if cls.root.exists(): raise AssertionError('Owned fixture remains')

    def tearDown(self):
        self.assertEqual(_sha256(self.source),self.digest)

    def test_both_lane_modes_exact_old_pixels_candidates_lamp_report_and_selected_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,legacy in (('injected',self.legacy),('lane-evidence',with_lane_evidence(self.legacy))):
                inputs=with_lamp_evidence(legacy)
                old=analyze_inputs(self.source,legacy,TOOLS)
                new=analyze_inputs(self.source,inputs,TOOLS)
                self.assertEqual(new.candidates,old.candidates)
                self.assertEqual(new.lamp_resets,())
                for event in new.candidates:
                    self.assertEqual(new.lamps.report(event.start[0].time,event.completion[1].time),
                                     old.lamps.report(event.start[0].time,event.completion[1].time))
                save_inputs(inputs,root/(name+'-saved'))
                restored=load_inputs(root/(name+'-saved')/'inputs.json')
                self.assertEqual(restored.to_bytes(),inputs.to_bytes())
                catalog,diagnostic=export_catalog_with_diagnostics(self.source,restored,root/(name+'-list'),toolchain=TOOLS)
                self.assertEqual(diagnostic['input_schema'],'experimental-event-inputs-v3')
                self.assertEqual(catalog.report()['candidate_count'],2)
                out=root/(name+'-batch')
                export_replay_batch(self.source,restored,out,candidate_indices=(1,0),catalog=catalog,toolchain=TOOLS)
                before={p.relative_to(out):p.read_bytes() for p in out.rglob('*') if p.is_file()}
                for source in (None,self.source):
                    self.assertEqual(verify_batch(out,source=source,toolchain=TOOLS)['status'],'verified')
                self.assertEqual(before,{p.relative_to(out):p.read_bytes() for p in out.rglob('*') if p.is_file()})

    def test_source_change_or_missing_observation_discards_old_encoded_blink_history(self):
        inputs=with_lamp_evidence(self.legacy)
        for kind in ('context','model','missing'):
            rows=[]
            for row in inputs.rows:
                o=row.lamp_evidence.observation
                if row.frame.ordinal>=35:
                    o=None if kind=='missing' else replace(o,**{('context_id' if kind=='context' else 'model_id'):'changed'})
                evidence=LampEvidence(o)
                left,right=evidence.rois(row.frame,row.vehicle)
                rows.append(replace(row,lamp_evidence=evidence,left=left,right=right))
            changed=replace(inputs,rows=tuple(rows))
            result=analyze_inputs(self.source,changed,TOOLS)
            self.assertEqual(len(result.candidates),2)
            self.assertEqual(result.lamp_resets[0][0].ordinal,35)
            self.assertEqual(result.lamp_resets[0][1],'lamps_unknown' if kind=='missing' else 'lamp_source_change')
            self.assertTrue(all(f.ordinal>=35 for f in result.lamps.frames()))
            with tempfile.TemporaryDirectory() as directory:
                out=Path(directory)/'batch'
                export_replay_batch(self.source,changed,out,candidate_indices=(0,),toolchain=TOOLS)
                self.assertEqual(verify_batch(out,source=self.source,toolchain=TOOLS)['status'],'verified')

    def test_actual_cli_demo_v3_selected_order_and_offline_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'demo'
            process=subprocess.run([sys.executable,'-B','-m','scripts.demo_lamp_replay','--output',str(out)],
                                   capture_output=True,timeout=90)
            self.assertEqual(process.returncode,0,process.stderr.decode())
            self.assertEqual(json.loads(process.stdout)['selected_count'],2)
            self.assertEqual(load_inputs(out/'inputs.json').schema,'experimental-event-inputs-v3')
            process=subprocess.run([sys.executable,'-B','-m','cheating_racer_detect','events','verify',
                '--bundle',str(out),'--ffmpeg-dir',str(TOOLS.ffmpeg.parent)],capture_output=True,timeout=45)
            self.assertEqual(process.returncode,0,process.stderr.decode())
            self.assertEqual(json.loads(process.stdout)['status'],'verified')

    def test_offline_rejects_cross_source_samples_even_with_recomputed_input_hash(self):
        inputs=with_lamp_evidence(self.legacy)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'batch'
            export_replay_batch(self.source,inputs,out,candidate_indices=(0,),toolchain=TOOLS)
            saved=json.loads((out/'candidates/000000/candidate.json').read_bytes())
            samples=saved['lamp']['samples']
            self.assertGreater(len(samples),2)
            boundary=samples[len(samples)//2]['vehicle']['frame']['ordinal']
            rows=tuple(replace(r,lamp_evidence=LampEvidence(replace(r.lamp_evidence.observation,
                       context_id='forged-new-context'))) if r.frame.ordinal>=boundary else r for r in inputs.rows)
            changed=replace(inputs,rows=rows)
            # Deliberately forge ONLY this owned test bundle, not a user result.
            (out/'inputs.json').write_bytes(changed.to_bytes())
            manifest=json.loads((out/'batch.json').read_bytes())
            manifest['inputs_sha256']=changed.sha256
            (out/'batch.json').write_text(json.dumps(manifest),encoding='utf-8')
            before=(out/'inputs.json').read_bytes()
            with self.assertRaises(AppError): verify_batch(out,toolchain=TOOLS)
            # Control proves the new boundary gate, not an unrelated hash check,
            # rejects an otherwise legacy-consistent saved sample sequence.
            with patch('cheating_racer_detect.events.verify.lamp_source_floor',return_value=None):
                self.assertEqual(verify_batch(out,toolchain=TOOLS)['status'],'verified')
            self.assertEqual((out/'inputs.json').read_bytes(),before)

    def test_capture_callback_unknown_missing_actual_atomic_fail_cancel_cleanup_retry(self):
        def observe(frame,vehicle):
            if vehicle is None: return None
            return LampObservation(frame,vehicle.identity,vehicle.box,'capture-lamps','capture-context',
                vehicle.box,20,20,0,LampPart('left','left',Box(2,2,4,4),'visible',True),
                LampPart('right','right',Box(16,2,18,4),'occluded',True),True,True,True)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=generate(TOOLS,root)
            digest=_sha256(source)
            lane,lamp=event_configs()
            def run(callback=observe,**options):
                bridge=DetectorTrackerBridge(Detector(),Tracker(policy()),model_alias='capture',expected_detector_id=Detector.model_id)
                def capture():
                    return capture_inputs(source,'0','2.4',root/'saved',bridge,selected_identity=(1,2,1),
                        lane_config=lane,lamp_config=lamp,pre=F(1,5),post=F(2,5),toolchain=TOOLS,
                        lamp_observations=callback,**options)
                return bridge,capture
            for failure in (RuntimeError('injected'),KeyboardInterrupt()):
                def bad(f,v):
                    if f.ordinal==5: raise failure
                    return observe(f,v)
                bridge,capture=run(bad)
                with self.assertRaises(type(failure)): capture()
                self.assertIsNone(bridge.tracker._previous)
                self.assertFalse((root/'saved').exists())
            bridge,capture=run(gates=lambda f,v:(None,LampROI(Box(1,1,2,2),'usable'),LampROI(None,'model_uncertain')))
            with self.assertRaises(ValueError): capture()
            self.assertIsNone(bridge.tracker._previous)
            bridge,capture=run()
            with patch('cheating_racer_detect.events.capture.write_bytes',side_effect=OSError('disk full')),self.assertRaises(Exception): capture()
            self.assertFalse((root/'saved').exists())
            self.assertIsNone(bridge.tracker._previous)
            _bridge,capture=run()
            result=capture()
            self.assertEqual(result.schema,'experimental-event-inputs-v3')
            self.assertEqual(result.to_bytes(),load_inputs(root/'saved/inputs.json').to_bytes())
            self.assertIsNone(result.rows[3].lamp_evidence.observation)
            self.assertEqual(result.rows[3].left.status,'identity_uncertain')
            self.assertEqual(result.rows[0].right.status,'occluded')
            self.assertEqual(analyze_inputs(source,result,TOOLS).candidates,())
            self.assertEqual(_sha256(source),digest)
            self.assertFalse(list(root.glob('*.partial')))

    def test_six_encoded_profiles_original_pts_rotation_no_automatic_display_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            media=fixtures.generate(TOOLS,root)
            for name in ('cfr','vfr','offset','rotation90','rotation180','rotation270'):
                with self.subTest(profile=name):
                    source=media[name]
                    digest=_sha256(source)
                    video,timeline,selected,first,rotation=prepare(source,'0','1.2',TOOLS)
                    rows=[]
                    for n,item in enumerate(selected):
                        f=Frame(digest,video['index'],first+n,item.pts,timeline.time_base,timeline.frames[0].pts,
                                video['width'],video['height'],rotation)
                        r=self.legacy.rows[n]
                        v=replace(r.vehicle,frame=f,detection=replace(r.vehicle.detection,frame=f))
                        rows.append(replace(r,frame=f,vehicle=v,lanes=replace(r.lanes,frame=f)))
                    inputs=with_lamp_evidence(replace(self.legacy,rows=tuple(rows)))
                    save_inputs(inputs,root/(name+'-saved'))
                    restored=load_inputs(root/(name+'-saved/inputs.json'))
                    self.assertEqual(restored.to_bytes(),inputs.to_bytes())
                    self.assertEqual(restored.rows[0].left.box,rows[0].left.box)
                    result=analyze_inputs(source,restored,TOOLS)
                    self.assertEqual(result.lamp_resets,())
                    self.assertEqual(result.lamps.frames(),tuple(r.frame for r in rows))
                    self.assertEqual(_sha256(source),digest)


if __name__=='__main__':
    unittest.main()
