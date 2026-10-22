"""Independent forward geometry and conservative lamp receipt invariants.

All boxes, pixel patterns and semantic declarations are synthetic test facts.
Numeric point tolerances never relax the exact original whole-pixel tests.
"""

from dataclasses import replace
from fractions import Fraction as F
import json
import math
import random
import unittest

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import LampEvidence,LampObservation,LampObservationEngine,LampPart
from cheating_racer_detect.events.inputs import EventInputs,lamp_source_floor
from cheating_racer_detect.tracking import Box
from scripts.demo_lamp_replay import with_lamp_evidence
from tests.test_event_inputs import fixture
from tests.test_events_lane import frame,vehicle
from tests.test_events_lamp import config,pixels
from tests.test_lamp_observations import observation


def forward_part(coded,crop,angle,model_width,model_height):
    """Independent coded -> CW view -> model oracle (not production inverse)."""
    w,h=F(crop.x2)-F(crop.x1),F(crop.y2)-F(crop.y1)
    vw,vh=(h,w) if angle in (90,270) else (w,h)
    points=[]
    for px,py in ((coded.x1,coded.y1),(coded.x1,coded.y2),(coded.x2,coded.y1),(coded.x2,coded.y2)):
        x,y=F(px)-F(crop.x1),F(py)-F(crop.y1)
        u,v={0:(x,y),90:(h-y,x),180:(w-x,h-y),270:(y,w-x)}[angle]
        points.append((u*model_width/vw,v*model_height/vh))
    return Box(*(float(v) for v in (min(p[0] for p in points),min(p[1] for p in points),
                                    max(p[0] for p in points),max(p[1] for p in points))))


def wide_observation(crop,coded,angle,scale=1):
    f=frame(0)
    v=vehicle(f)
    v=replace(v,detection=replace(v.detection,box=Box(2,2,94,62)))
    w,h=int(crop.x2-crop.x1),int(crop.y2-crop.y1)
    mw,mh=(h*scale,w*scale) if angle in (90,270) else (w*scale,h*scale)
    b=forward_part(coded,crop,angle,mw,mh)
    o=LampObservation(f,v.identity,v.box,'geometry-fixture','crop-fixture',crop,mw,mh,angle,
        LampPart('left','left',b,'visible',True),LampPart('right','right',b,'visible',True),True,True,True)
    return v,o


class GeometryInvariants(unittest.TestCase):
    def test_integer_edges_remain_exact_for_four_rotations_and_integer_resize(self):
        for w in (7,11,23,29,43,61,89):
            for h in (9,17,31,53):
                crop=Box(2,2,2+w,2+h)
                coded=Box(3,3,1+w,1+h)
                for angle in (0,90,180,270):
                    for scale in (1,3,7,11):
                        _v,o=wide_observation(crop,coded,angle,scale)
                        with self.subTest(w=w,h=h,angle=angle,scale=scale):
                            self.assertEqual(o.coded_box(o.left),coded)

    def test_original_whole_pixel_brightness_not_changed_by_preprocessing(self):
        f=frame(0)
        data=bytes((3*x+7*y+c)%256 for y in range(f.height) for x in range(f.width) for c in range(3))
        for w,h in ((23,17),(43,31),(61,53),(89,9)):
            crop=Box(2,2,2+w,2+h)
            coded=Box(3,3,w+1,h+1)
            expected=F(sum((3*x+7*y+c)%256 for y in range(3,h+1) for x in range(3,w+1) for c in range(3)),
                       (w-2)*(h-2)*3)
            for angle in (0,90,180,270):
                for scale in (1,3,7,11):
                    v,o=wide_observation(crop,coded,angle,scale)
                    e=LampObservationEngine(config())
                    e.update(f,v,data,o)
                    sample=e.report(F(0),F(4,5))['samples'][0]['left']
                    with self.subTest(w=w,h=h,angle=angle,scale=scale):
                        self.assertEqual(sample['brightness'],str(expected))

    def test_seeded_interior_integer_edges_preserve_exact_pixel_membership(self):
        rng=random.Random(20261023)
        for _ in range(1024):
            w,h=rng.randint(7,90),rng.randint(7,59)
            x,y=rng.randint(1,w-3),rng.randint(1,h-3)
            crop=Box(2,2,w+2,h+2)
            coded=Box(x+2,y+2,x+4,y+4)
            angle=rng.choice((0,90,180,270))
            scale=rng.choice((1,3,7,11))
            _v,o=wide_observation(crop,coded,angle,scale)
            self.assertEqual(o.coded_box(o.left),coded,(w,h,x,y,angle,scale))

    def test_real_subpixel_edge_is_never_snapped_to_neighbor_integer(self):
        f=frame(0)
        o=observation(f)
        edge=math.nextafter(20.,math.inf)
        crop=o.crop
        coded=Box(edge,12,22,14)
        raw=forward_part(coded,crop,0,12,20)
        o=replace(o,model_width=12,model_height=20,left=replace(o.left,box=raw),
                  right=replace(o.right,box=raw))
        self.assertEqual(o.coded_box(o.left),coded)
        e=LampObservationEngine(config())
        e.update(f,vehicle(f),pixels(f),o)
        self.assertEqual(e.report(F(0),F(4,5))['samples'][0]['left']['status'],'too_small')

    def test_seeded_fractional_crop_forward_inverse_numeric_only(self):
        rng=random.Random(20261022)
        for _ in range(128):
            f=frame(0)
            base=observation(f)
            crop=Box(18.125,10.25,29.875,29.75)
            x1=18.5+rng.random()*3
            y1=10.5+rng.random()*3
            coded=Box(x1,y1,x1+2+rng.random(),y1+2+rng.random())
            angle=rng.choice((0,90,180,270))
            mw,mh=rng.randint(1,1024),rng.randint(1,1024)
            box=forward_part(coded,crop,angle,mw,mh)
            o=replace(base,crop=crop,rotation_cw=angle,model_width=mw,model_height=mh,
                      left=replace(base.left,box=box),right=replace(base.right,box=box))
            for got,wanted in zip(o.coded_box(o.left).values(),coded.values()):
                self.assertAlmostEqual(got,wanted,delta=1e-11)


class ConservativeInvariants(unittest.TestCase):
    def test_any_unknown_gate_in_full_window_prevents_no_blink_confirmation(self):
        def changes(o):
            return (replace(o,side_mapping_confirmed=False),replace(o,rear_view=False),
                    replace(o,identity_confirmed=False),replace(o,left=replace(o.left,turn_signal_confirmed=False)),
                    replace(o,left=replace(o.left,visibility='occluded')),
                    replace(o,left=replace(o.left,visibility='unknown')),replace(o,left=replace(o.left,box=None)))
        for position in range(9):
            for kind in range(7):
                e=LampObservationEngine(config())
                for n in range(9):
                    f=frame(n)
                    o=observation(f)
                    if n==position: o=changes(o)[kind]
                    e.update(f,vehicle(f),pixels(f,20),o)
                with self.subTest(position=position,kind=kind):
                    self.assertEqual(e.report(F(0),F(4,5))['signal_state'],'unknown')

    def test_missing_actual_or_sparse_clock_does_not_bridge_blink(self):
        for kind in ('actual','ordinal','pts'):
            e=LampObservationEngine(config())
            for n,value in enumerate((20,220,220,20,20,220,220,20,20)):
                if kind=='ordinal' and n==4: continue
                f=frame(n,pts=100+n+(5 if n>=4 and kind=='pts' else 0))
                v=vehicle(f)
                e.update(f,None if n==4 and kind=='actual' else v,pixels(f,value),
                         None if n==4 and kind=='actual' else observation(f))
            self.assertEqual(e.report(F(0),F(4,5))['signal_state'],'unknown')

    def test_source_floor_matches_missing_and_multiple_declared_source_boundaries(self):
        inputs=with_lamp_evidence(fixture())
        rows=[]
        for i,r in enumerate(inputs.rows):
            o=r.lamp_evidence.observation
            if i==2:o=None
            elif i>=4:o=replace(o,context_id='new-context')
            evidence=LampEvidence(o)
            left,right=evidence.rois(r.frame,r.vehicle)
            rows.append(replace(r,lamp_evidence=evidence,left=left,right=right))
        changed=replace(inputs,rows=tuple(rows))
        expected=(None,None,2,2,4,4,4,4,4)
        for r,floor in zip(changed.rows,expected):
            self.assertEqual(lamp_source_floor(changed,r.frame.ordinal),floor)

    def test_returned_sample_mutation_cannot_edit_engine_history(self):
        e=LampObservationEngine(config())
        for n in range(9):
            f=frame(n)
            e.update(f,vehicle(f),pixels(f),observation(f))
        before=e.report(F(0),F(4,5))
        altered=e.report(F(0),F(4,5))
        altered['samples'][0]['left']['roi'][0]=0
        altered['samples'][0]['vehicle']['detection']['bbox'][0]=0
        self.assertEqual(e.report(F(0),F(4,5)),before)


class ReceiptInvariants(unittest.TestCase):
    def test_one_ulp_stored_roi_mismatch_is_rejected_not_clipped(self):
        value=with_lamp_evidence(fixture()).report()
        box=value['rows'][0]['left']['box']
        box[0]=math.nextafter(box[0],math.inf)
        with self.assertRaises(AppError): EventInputs.from_bytes(json.dumps(value).encode())

    def test_nested_duplicate_nonfinite_or_wrong_primitive_rejected(self):
        inputs=with_lamp_evidence(fixture())
        raw=inputs.to_bytes().decode()
        for name in ('context_id','visibility','turn_signal_confirmed'):
            value=json.loads(raw)['rows'][0]['lamp_evidence']['observation']
            item=value[name] if name=='context_id' else value['left'][name]
            token=json.dumps(name)+':'+json.dumps(item,separators=(',',':'))
            duplicate=raw.replace(token,token+','+token,1)
            self.assertNotEqual(duplicate,raw)
            with self.assertRaises(AppError): EventInputs.from_bytes(duplicate.encode())
        for token in ('NaN','Infinity','1e999','true','null','"240"'):
            value=raw.replace('"model_width":240','"model_width":'+token,1)
            self.assertNotEqual(value,raw)
            with self.assertRaises(AppError): EventInputs.from_bytes(value.encode())

    def test_seeded_receipt_roundtrip_owned_source_and_boolean_declarations(self):
        inputs=with_lamp_evidence(fixture())
        rng=random.Random(20261022)
        for _ in range(32):
            rows=[]
            for r in inputs.rows:
                o=r.lamp_evidence.observation
                o=replace(o,side_mapping_confirmed=rng.choice((True,False)),rear_view=rng.choice((True,False)),
                          left=replace(o.left,visibility=rng.choice(('visible','occluded','unknown'))))
                ev=LampEvidence(o)
                left,right=ev.rois(r.frame,r.vehicle)
                rows.append(replace(r,lamp_evidence=ev,left=left,right=right))
            changed=replace(inputs,rows=tuple(rows))
            self.assertEqual(EventInputs.from_bytes(changed.to_bytes()).to_bytes(),changed.to_bytes())
            owned=changed.report()
            owned['rows'][0]['lamp_evidence']['observation']['left']['box'][0]=0
            self.assertNotEqual(owned,changed.report())


if __name__=='__main__': unittest.main()
