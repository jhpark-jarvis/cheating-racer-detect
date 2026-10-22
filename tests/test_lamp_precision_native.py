"""Encoded original pixels, not resampled model pixels or real detections."""

from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import tempfile
import unittest

from cheating_racer_detect.events import LampEvidence,LampObservation,LampPart,LampROI
from cheating_racer_detect.events.catalog import export_catalog_with_diagnostics
from cheating_racer_detect.events.inputs import load_inputs,save_inputs
from cheating_racer_detect.events.replay import analyze_inputs,export_replay_batch
from cheating_racer_detect.events.verify import verify_batch
from cheating_racer_detect.service import _sha256
from cheating_racer_detect.tracking import Box
from scripts.demo_batch_replay import generate_inputs
from scripts.demo_lane_replay import with_lane_evidence
from scripts.demo_lamp_replay import with_lamp_evidence
from tests.test_lamp_invariants import forward_part
from tests.test_capture_native import AVAILABLE,TOOLS


@unittest.skipUnless(AVAILABLE and TOOLS,'NOT_RUN: FFmpeg/NumPy/OpenCV required')
class NativePrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix='crd-lamp-precision-')
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

    def test_twelve_explicit_rotated_resize_declarations_same_encoded_lamps_and_review(self):
        base=with_lamp_evidence(with_lane_evidence(self.legacy))
        reference=analyze_inputs(self.source,self.legacy,TOOLS)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for angle in (0,90,180,270):
                for scale in (1,7,11):
                    rows=[]
                    for row in base.rows:
                        o=row.lamp_evidence.observation
                        mw,mh=(20*scale,24*scale) if angle in (90,270) else (24*scale,20*scale)
                        o=replace(o,rotation_cw=angle,model_width=mw,model_height=mh,
                            left=replace(o.left,box=forward_part(row.left.box,o.crop,angle,mw,mh)),
                            right=replace(o.right,box=forward_part(row.right.box,o.crop,angle,mw,mh)))
                        ev=LampEvidence(o)
                        self.assertEqual(ev.rois(row.frame,row.vehicle),(row.left,row.right))
                        rows.append(replace(row,lamp_evidence=ev))
                    inputs=replace(base,rows=tuple(rows))
                    name=f'{angle}-{scale}'
                    save_inputs(inputs,root/(name+'-saved'))
                    restored=load_inputs(root/(name+'-saved/inputs.json'))
                    actual=analyze_inputs(self.source,restored,TOOLS)
                    with self.subTest(angle=angle,scale=scale):
                        self.assertEqual(actual.candidates,reference.candidates)
                        self.assertEqual(actual.lamp_resets,())
                        self.assertEqual(actual.lamps.report(F(0),F(43,10)),reference.lamps.report(F(0),F(43,10)))
                    if angle in (90,270) and scale==7:
                        catalog,_=export_catalog_with_diagnostics(self.source,restored,root/(name+'-list'),toolchain=TOOLS)
                        out=root/(name+'-batch')
                        export_replay_batch(self.source,restored,out,candidate_indices=(1,0),catalog=catalog,toolchain=TOOLS)
                        before={p.relative_to(out):(p.read_bytes(),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
                        for source in (None,self.source):
                            self.assertEqual(verify_batch(out,source=source,toolchain=TOOLS)['status'],'verified')
                        self.assertEqual(before,{p.relative_to(out):(p.read_bytes(),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()})

    def test_exact_narrow_roi_regression_matches_original_encoded_brightness(self):
        rows=[]
        for r in self.legacy.rows:
            v=replace(r.vehicle,detection=replace(r.vehicle.detection,box=Box(2,2,94,62)))
            rows.append(replace(r,vehicle=v,lanes=None,left=LampROI(Box(37,22,39,24),'usable'),
                                right=LampROI(Box(45,22,47,24),'usable')))
        legacy=replace(self.legacy,rows=tuple(rows),lamp_config=replace(self.legacy.lamp_config,min_pixels=4))
        rows=[]
        for r in legacy.rows:
            crop=Box(2,2,76,56)
            o=LampObservation(r.frame,r.vehicle.identity,r.vehicle.box,'narrow-fixture','narrow-crop',crop,74,54,180,
                LampPart('left','left',forward_part(r.left.box,crop,180,74,54),'visible',True),
                LampPart('right','right',forward_part(r.right.box,crop,180,74,54),'visible',True),True,True,True)
            ev=LampEvidence(o)
            self.assertEqual(ev.rois(r.frame,r.vehicle),(r.left,r.right))
            rows.append(replace(r,lamp_evidence=ev))
        inputs=replace(legacy,rows=tuple(rows))
        actual=analyze_inputs(self.source,inputs,TOOLS)
        reference=analyze_inputs(self.source,legacy,TOOLS)
        self.assertEqual(actual.candidates,())
        report=actual.lamps.report(F(0),F(43,10))
        self.assertTrue(all(s['left']['status']=='usable' for s in report['samples']))
        self.assertEqual(report,reference.lamps.report(F(0),F(43,10)))
        with tempfile.TemporaryDirectory() as directory:
            catalog,_=export_catalog_with_diagnostics(self.source,inputs,Path(directory)/'list',toolchain=TOOLS)
            self.assertEqual(catalog.report()['candidate_count'],0)


if __name__=='__main__': unittest.main()
