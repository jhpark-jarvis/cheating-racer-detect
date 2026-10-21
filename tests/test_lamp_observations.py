"""Synthetic crop/semantic declarations and source-aware history, no models."""

from dataclasses import FrozenInstanceError,replace
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import (LampPart,LampObservation,LampEvidence,LampObservationEngine,
                                        resolve_lamp_rois,LampEngine)
from cheating_racer_detect.events.inputs import (EventInputs,InputLampEngine,MAX_BYTES,
                                                load_inputs,save_inputs,lamp_source_floor)
from cheating_racer_detect.tracking import Box
from scripts.demo_lane_replay import with_lane_evidence
from scripts.demo_lamp_replay import with_lamp_evidence
from tests.test_event_inputs import fixture
from tests.test_events_lane import frame,vehicle
from tests.test_events_lamp import config,pixels,rois


def observation(f=None,target=None,**changes):
    f=f or frame(0)
    target=target or vehicle(f)
    o=LampObservation(f,target.identity,target.box,'lamp-fixture-v1','semantic-fixture-v1',
        target.box,120,200,0,LampPart('left','left-turn',Box(20,20,40,40),'visible',True),
        LampPart('right','right-turn',Box(80,20,100,40),'visible',True),True,True,True)
    return replace(o,**changes)


class Geometry(unittest.TestCase):
    def test_direct_resize_oracle_and_exact_actual_frame(self):
        f=frame(0)
        o=observation(f)
        self.assertEqual(resolve_lamp_rois(f,vehicle(f),o),rois())
        with self.assertRaises(FrozenInstanceError): o.context_id='changed'

    def test_all_quarter_turns_use_literal_edge_coordinate_oracle(self):
        cases=((0,12,20,(2,2,4,4)),(90,20,12,(16,2,18,4)),
               (180,12,20,(8,16,10,18)),(270,20,12,(2,8,4,10)))
        for angle,w,h,box in cases:
            f=frame(0,display_rotation=(angle+90)%360)
            base=observation(f)
            o=replace(base,rotation_cw=angle,model_width=w,model_height=h,
                      left=replace(base.left,box=Box(*box)),right=replace(base.right,box=None,visibility='unknown'))
            with self.subTest(angle=angle): self.assertEqual(o.coded_box(o.left),Box(20,12,22,14))

    def test_display_rotation_never_infers_preprocessing_or_vehicle_side(self):
        for angle in (0,90,180,270):
            f=frame(0,display_rotation=angle)
            self.assertEqual(resolve_lamp_rois(f,vehicle(f),observation(f)),rois())
        base=observation()
        # Image location does not rename an explicitly declared vehicle-side.
        o=replace(base,left=replace(base.left,box=base.right.box),right=replace(base.right,box=base.left.box))
        self.assertEqual(resolve_lamp_rois(o.frame,vehicle(o.frame),o)[0].box,rois()[1].box)

    def test_crop_canvas_raw_box_and_boolean_resource_bounds(self):
        base=observation()
        invalid=({'crop':Box(0,0,12,12)},{'model_width':0},{'model_width':1025},
                 {'rotation_cw':True},{'rotation_cw':45},{'rear_view':1},
                 {'left':replace(base.left,box=Box(0,0,121,20))},
                 {'right':replace(base.right,part_id=base.left.part_id)},
                 {'left':replace(base.left,side='right')})
        for changes in invalid:
            with self.subTest(changes=changes),self.assertRaises(ValueError): replace(base,**changes)
        for changes in ({'visibility':'usable'},{'turn_signal_confirmed':1},{'part_id':'x'*65}):
            with self.assertRaises(ValueError): replace(base.left,**changes)

    def test_foreign_frame_clock_identity_box_or_missing_actual_rejected(self):
        f=frame(0)
        for o in (observation(frame(1)),observation(frame(0,pts=101)),
                  observation(vehicle_identity=(2,2,1)),observation(frame(0),vehicle(frame(0),x=25))):
            with self.assertRaises(ValueError): resolve_lamp_rois(f,vehicle(f),o)
        with self.assertRaises(ValueError): resolve_lamp_rois(f,None,observation(f))
        self.assertEqual(resolve_lamp_rois(f,None,None)[0].status,'identity_uncertain')
        self.assertEqual(resolve_lamp_rois(f,vehicle(f),None)[0].status,'model_uncertain')

    def test_semantics_visibility_and_missing_boxes_are_not_promoted(self):
        base=observation()
        for changes,status in (({'side_mapping_confirmed':False},'model_uncertain'),
                               ({'rear_view':False},'model_uncertain'),
                               ({'identity_confirmed':False},'identity_uncertain')):
            self.assertEqual(resolve_lamp_rois(base.frame,vehicle(base.frame),replace(base,**changes))[0].status,status)
        for visibility,status in (('occluded','occluded'),('out_of_view','out_of_view'),
                                  ('too_small','too_small'),('unknown','model_uncertain')):
            o=replace(base,left=replace(base.left,visibility=visibility))
            self.assertEqual(resolve_lamp_rois(o.frame,vehicle(o.frame),o)[0].status,status)
        for part in (replace(base.left,turn_signal_confirmed=False),replace(base.left,box=None)):
            o=replace(base,left=part)
            self.assertEqual(resolve_lamp_rois(o.frame,vehicle(o.frame),o)[0].status,'model_uncertain')

    def test_large_model_canvas_does_not_create_original_pixels(self):
        base=observation()
        o=replace(base,left=replace(base.left,box=Box(20,20,25,25)))
        engine=LampObservationEngine(config())
        engine.update(o.frame,vehicle(o.frame),pixels(o.frame,220),o)
        report=engine.report(F(0),F(4,5))
        self.assertEqual(report['samples'][0]['left']['status'],'too_small')
        self.assertIsNone(report['samples'][0]['left']['brightness'])


class Histories(unittest.TestCase):
    def run_sequence(self,alter=None):
        engine=LampObservationEngine(config())
        results=[]
        for n,value in enumerate((20,220,220,20,20,220,220,20,20)):
            f=frame(n)
            o=observation(f)
            if alter is not None: o=alter(n,o)
            results.append(engine.update(f,vehicle(f),pixels(f,value),o))
        return engine,results

    def test_same_source_blink_matches_original_engine_exactly(self):
        engine,results=self.run_sequence()
        base=LampEngine(config())
        for n,value in enumerate((20,220,220,20,20,220,220,20,20)):
            f=frame(n)
            base.update(f,vehicle(f),pixels(f,value),*rois())
        self.assertEqual(engine.report(F(0),F(4,5)),base.report(F(0),F(4,5)))
        self.assertTrue(all(r['reset_reason'] is None for r in results))

    def test_model_context_canvas_rotation_part_changes_break_whole_history(self):
        for kind in ('model','context','canvas','rotation','part'):
            def alter(n,o):
                if n<5:return o
                if kind=='model':return replace(o,model_id='new-model')
                if kind=='context':return replace(o,context_id='new-context')
                if kind=='canvas':return replace(o,model_width=240)
                if kind=='rotation':return replace(o,rotation_cw=90)
                return replace(o,left=replace(o.left,part_id='new-left'))
            engine,results=self.run_sequence(alter)
            with self.subTest(kind=kind):
                self.assertEqual(results[5]['reset_reason'],'lamp_source_change')
                self.assertEqual(engine.frames()[0].ordinal,5)
                self.assertEqual(engine.report(F(0),F(4,5))['signal_state'],'unknown')

    def test_missing_observation_cannot_bridge_blink(self):
        engine,results=self.run_sequence(lambda n,o:None if n==5 else o)
        self.assertEqual(results[5]['reset_reason'],'lamps_unknown')
        self.assertEqual(engine.frames()[0].ordinal,5)
        self.assertEqual(engine.report(F(0),F(4,5))['signal_state'],'unknown')

    def test_one_side_occlusion_preserves_partial_blink_facts(self):
        engine,_results=self.run_sequence(lambda n,o:replace(o,left=replace(o.left,visibility='occluded')) if n==8 else o)
        report=engine.report(F(0),F(4,5))
        self.assertTrue(report['left']['blink_evidence'])
        self.assertIn('occluded',report['left']['reasons'])
        self.assertEqual(report['signal_state'],'unknown')
        self.assertEqual(len(engine.frames()),9)

    def test_normal_vehicle_crop_translation_not_a_new_semantic_source(self):
        engine=LampObservationEngine(config())
        for n in range(9):
            f=frame(n)
            target=vehicle(f,x=24+n/4)
            result=engine.update(f,target,bytes(f.width*f.height*3),observation(f,target))
            self.assertIsNone(result['reset_reason'])
        self.assertEqual(len(engine.frames()),9)

    def test_failure_cancel_discard_source_and_samples_then_fresh_retry(self):
        engine,_results=self.run_sequence()
        for failure in (ValueError('native'),KeyboardInterrupt()):
            f=frame(9)
            with patch.object(LampEngine,'_roi',side_effect=failure),self.assertRaises(type(failure)):
                engine.update(f,vehicle(f),pixels(f),observation(f))
            self.assertIsNone(engine._source)
            self.assertEqual(engine.frames(),())
            f=frame(0)
            self.assertEqual(engine.update(f,vehicle(f),pixels(f),observation(f))['lamp']['sample_count'],1)


class Receipts(unittest.TestCase):
    def test_v3_both_lane_modes_roundtrip_and_legacy_bytes_unchanged(self):
        legacy=fixture()
        for base in (legacy,with_lane_evidence(legacy)):
            before=base.to_bytes()
            current=with_lamp_evidence(base)
            self.assertEqual(current.schema,'experimental-event-inputs-v3')
            self.assertEqual(current.report()['lane_provenance'],base.rows[0].lane_evidence is not None)
            self.assertEqual(EventInputs.from_bytes(current.to_bytes()),current)
            self.assertEqual(base.to_bytes(),before)
            self.assertEqual(EventInputs.from_bytes(before).to_bytes(),before)
            self.assertEqual(replace(current,rows=tuple(replace(r,lamp_evidence=None) for r in current.rows)).to_bytes(),before)

    def test_v3_unknown_is_explicit_and_mixed_or_missing_evidence_rejected(self):
        inputs=with_lamp_evidence(fixture())
        rows=tuple(replace(r,lamp_evidence=LampEvidence(None),left=LampEvidence(None).rois(r.frame,r.vehicle)[0],
                           right=LampEvidence(None).rois(r.frame,r.vehicle)[1]) for r in inputs.rows)
        unknown=replace(inputs,rows=rows)
        self.assertEqual(EventInputs.from_bytes(unknown.to_bytes()),unknown)
        with self.assertRaises(ValueError): replace(inputs,rows=(replace(inputs.rows[0],lamp_evidence=None),*inputs.rows[1:]))
        for mode in ('missing','null','flag','extra','v2'):
            value=inputs.report()
            if mode=='missing':del value['rows'][0]['lamp_evidence']
            elif mode=='null':value['rows'][0]['lamp_evidence']=None
            elif mode=='flag':value['lane_provenance']=0
            elif mode=='extra':value['extra']=True
            else:value['schema']='experimental-event-inputs-v2'
            with self.subTest(mode=mode),self.assertRaises(AppError):EventInputs.from_bytes(json.dumps(value).encode())

    def test_nested_source_shape_semantics_and_roi_drift_strict(self):
        inputs=with_lamp_evidence(fixture())
        paths=((('vehicle_identity',),[2,2,1]),(('vehicle_box',),[0,0,96,64]),
               (('crop',),[0,0,1,1]),(('model_width',),True),(('rotation_cw',),45),
               (('left','visibility'),'usable'),(('left','turn_signal_confirmed'),1),
               (('left','box'),[0,0,9999,4]),(('extra',),1),(('frame',),{}))
        for path,new in paths:
            value=inputs.report()
            target=value['rows'][0]['lamp_evidence']['observation']
            for key in path[:-1]:target=target[key]
            target[path[-1]]=new
            with self.subTest(path=path),self.assertRaises(AppError):EventInputs.from_bytes(json.dumps(value).encode())
        value=inputs.report()
        value['rows'][0]['left']['box'][0]+=1
        with self.assertRaises(AppError):EventInputs.from_bytes(json.dumps(value).encode())

    def test_source_edit_changes_hash_but_does_not_assert_authenticity(self):
        inputs=with_lamp_evidence(fixture())
        row=inputs.rows[0]
        changed=replace(row,lamp_evidence=LampEvidence(replace(row.lamp_evidence.observation,context_id='new-context')))
        other=replace(inputs,rows=(changed,*inputs.rows[1:]))
        self.assertNotEqual(other.sha256,inputs.sha256)
        self.assertEqual(EventInputs.from_bytes(other.to_bytes()),other)
        value=inputs.report()
        value['rows'][0]['lamp_evidence']['observation']['left']['box'][0]=0
        self.assertEqual(inputs.rows[0].lamp_evidence.observation.left.box.x1,40.)

    def test_same_source_floor_and_recorded_engine_mixed_mode_cancel_recovery(self):
        inputs=with_lamp_evidence(fixture())
        changed=replace(inputs,rows=tuple(replace(r,lamp_evidence=LampEvidence(replace(r.lamp_evidence.observation,context_id='changed')))
                         if i>=5 else r for i,r in enumerate(inputs.rows)))
        self.assertEqual(lamp_source_floor(changed,8),5)
        self.assertIsNone(lamp_source_floor(fixture(),8))
        engine=InputLampEngine(inputs.lamp_config)
        engine.update_recorded(inputs.rows[0],bytes(96*64*3))
        with self.assertRaises(ValueError):engine.update_recorded(fixture().rows[1],bytes(96*64*3))
        self.assertEqual(engine.frames(),())
        engine.update_recorded(inputs.rows[0],bytes(96*64*3))
        with patch.object(LampEngine,'_roi',side_effect=KeyboardInterrupt()),self.assertRaises(KeyboardInterrupt):
            engine.update_recorded(inputs.rows[1],bytes(96*64*3))
        self.assertEqual(engine.frames(),())
        self.assertIsNone(engine._mode)

    def test_atomic_save_cancel_retry_and_byte_limit(self):
        inputs=with_lamp_evidence(fixture())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('cheating_racer_detect.events.inputs.write_bytes',side_effect=KeyboardInterrupt()),self.assertRaises(KeyboardInterrupt):
                save_inputs(inputs,root/'saved')
            self.assertEqual(list(root.iterdir()),[])
            save_inputs(inputs,root/'saved')
            self.assertEqual(load_inputs(root/'saved/inputs.json'),inputs)
            with self.assertRaises(AppError):save_inputs(inputs,root/'saved')
        with patch('cheating_racer_detect.events.inputs.MAX_BYTES',1),self.assertRaises(ValueError):replace(inputs,pre=inputs.pre)

    def test_base_v3_parser_and_geometry_do_not_import_ml(self):
        inputs=with_lamp_evidence(fixture())
        child=subprocess.run([sys.executable,'-B','-c',
            'import sys; from cheating_racer_detect.events.inputs import EventInputs,InputLampEngine; '
            'p=EventInputs.from_bytes(sys.stdin.buffer.read()); e=InputLampEngine(p.lamp_config); '
            'e.update_recorded(p.rows[0],bytes(96*64*3)); '
            'assert not any(n in sys.modules for n in ("numpy","cv2","torch"))'],
            input=inputs.to_bytes(),capture_output=True,timeout=20)
        self.assertEqual(child.returncode,0,child.stderr.decode())


if __name__=='__main__':unittest.main()
