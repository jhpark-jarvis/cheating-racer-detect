"""Explicit synthetic adjacency/visibility/semantic declarations, not models."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from cheating_racer_detect.analysis import LaneBatch, LaneSlot
from cheating_racer_detect.events import (BoundaryBinding, LaneContext, LaneSectionBridge,
                                        section_for_vehicle)
from tests.test_events_lane import config, frame, vehicle


def scripted(source, *, model_id='lane-fixture-v1', rotation=0):
    slots = tuple(LaneSlot(slot, axis, (True,)*count,
                          tuple((x, 63.*i/(count-1)) for i in range(count)))
                  for slot, axis, count, x in ((1,'row',72,10.),(2,'row',72,48.),
                                               (0,'column',81,86.),(3,'column',81,94.)))
    return LaneBatch(source, model_id, rotation, slots)


def context(source, target=None, batch=None, **changes):
    target = target or vehicle(source)
    batch = batch or scripted(source)
    bindings = tuple(BoundaryBinding(slot, name, (True,)*(71 if slot in (1,2) else 80))
                     for slot, name in ((1,'left'),(2,'middle'),(0,'right')))
    result = LaneContext(source, target.identity, batch.model_id, 'synthetic-context-v1',
                         'road', bindings, True, True, 'stable', 'ordinary', True)
    return replace(result, **changes)


def replace_slot(batch, slot_id, **changes):
    return replace(batch, slots=tuple(replace(item, **changes) if item.slot == slot_id else item
                                     for item in batch.slots))


class Sections(unittest.TestCase):
    def run_section(self, *, source=None, target=None, batch=None, declared=None):
        source = source or frame(0)
        target = target or vehicle(source)
        batch = batch or scripted(source)
        declared = declared or context(source, target, batch)
        return section_for_vehicle(source, target, batch, declared)

    def test_linear_segment_oracle_y2_frame_and_order(self):
        source = frame(2, time=9)
        batch = scripted(source)
        points = tuple((10.+i*.2, 63.*i/71) for i in range(72))
        batch = replace_slot(batch, 1, points=points)
        result = self.run_section(source=source, batch=batch)
        self.assertIs(result.frame, source)
        self.assertIs(result.section.frame, source)
        self.assertEqual(result.section.y, 30.)
        self.assertEqual(result.section.boundary_ids, ('left','middle','right'))
        self.assertAlmostEqual(result.section.x[0], 10.+(30./63*71)*.2)
        sample = result.samples[0]
        self.assertEqual(sample.anchors, (33,34))
        self.assertAlmostEqual(sample.weight, (30.-points[33][1])/(points[34][1]-points[33][1]))
        self.assertEqual(result.frame.time, F(9,10))

    def test_shared_anchor_counts_once_and_uses_visible_adjacent_segment(self):
        batch = scripted(frame(0))
        points = list(batch.slots[0].points)
        points[34] = (10.,30.)
        batch = replace_slot(batch, 1, points=tuple(points))
        declared = context(frame(0), batch=batch)
        visibility = [False]*71
        visibility[34] = True
        binding = replace(declared.bindings[0], visible_segments=tuple(visibility))
        declared = replace(declared, bindings=(binding,*declared.bindings[1:]))
        result = self.run_section(batch=batch, declared=declared)
        self.assertIsNotNone(result.section)
        self.assertEqual(result.samples[0].anchors, (34,35))
        self.assertEqual(result.samples[0].weight, 0.)

    def test_visibility_must_certify_intersecting_whole_segment(self):
        declared = context(frame(0))
        mask = [True]*71
        mask[33] = False
        binding = replace(declared.bindings[0], visible_segments=tuple(mask))
        result = self.run_section(declared=replace(declared, bindings=(binding,*declared.bindings[1:])))
        self.assertIsNone(result.section)
        self.assertEqual(result.reason, 'uncertified_intersection')
        self.assertEqual(result.failure_slot, 1)

    def test_missing_anchor_never_bridged_even_with_true_visibility(self):
        batch = scripted(frame(0))
        points = list(batch.slots[0].points)
        points[33] = None
        batch = replace_slot(batch, 1, points=tuple(points))
        result = self.run_section(batch=batch)
        self.assertIsNone(result.section)
        self.assertEqual(result.reason, 'no_adjacent_support')

    def test_outside_support_no_extrapolation_or_nearest(self):
        batch = scripted(frame(0))
        points = tuple((10.,40.+i/71*20) for i in range(72))
        result = self.run_section(batch=replace_slot(batch,1,points=points))
        self.assertIsNone(result.section)
        self.assertEqual(result.reason,'no_adjacent_support')

    def test_horizontal_at_target_is_ambiguous(self):
        batch = scripted(frame(0))
        points = list(batch.slots[0].points)
        points[33],points[34] = (9.,30.),(11.,30.)
        result = self.run_section(batch=replace_slot(batch,1,points=tuple(points)))
        self.assertEqual(result.reason,'horizontal_intersection')
        self.assertIsNone(result.section)

    def test_fold_multiple_intersections_even_if_only_one_certified(self):
        batch = scripted(frame(0))
        points = list(batch.slots[0].points)
        points[40] = (10.,20.)
        declared = context(frame(0))
        mask = [False]*71
        mask[33] = True
        binding = replace(declared.bindings[0],visible_segments=tuple(mask))
        result = self.run_section(batch=replace_slot(batch,1,points=tuple(points)),
            declared=replace(declared,bindings=(binding,*declared.bindings[1:])))
        self.assertEqual(result.reason,'multiple_intersections')

    def test_distinct_anchors_at_same_coordinate_not_deduplicated(self):
        batch = scripted(frame(0))
        points = list(batch.slots[0].points)
        points[33],points[34],points[35],points[36] = (10.,30.),(10.,29.),(10.,30.),(10.,31.)
        result = self.run_section(batch=replace_slot(batch,1,points=tuple(points)))
        self.assertEqual(result.reason,'multiple_intersections')

    def test_boundary_order_not_sorted_and_failed_boundary_not_dropped(self):
        declared = context(frame(0))
        result = self.run_section(declared=replace(declared,bindings=declared.bindings[::-1]))
        self.assertEqual(result.reason,'boundary_order_uncertain')
        batch = scripted(frame(0))
        batch = replace_slot(batch,2,exists=(False,)*72,points=(None,)*72)
        result = self.run_section(batch=batch)
        self.assertIsNone(result.section)
        self.assertEqual(result.reason,'slot_support_missing')
        self.assertEqual(len(result.samples),1)  # Diagnostic, NOT a partial section.

    def test_duplicate_x_is_unknown(self):
        batch = scripted(frame(0))
        batch = replace_slot(batch,2,points=batch.slots[0].points)
        self.assertEqual(self.run_section(batch=batch).reason,'boundary_order_uncertain')

    def test_semantic_gates_and_minimum_boundary_count(self):
        for changes, reason in (({'adjacency_confirmed':False},'boundary_bindings_unconfirmed'),
            ({'coded_axes_confirmed':False},'coded_axes_uncertain'),
            ({'ego':'unknown'},'ego_scene_or_pose_uncertain'),
            ({'ego':'changing'},'ego_scene_or_pose_uncertain'),
            ({'scene':'merge'},'ego_scene_or_pose_uncertain'),
            ({'scene':'unknown'},'ego_scene_or_pose_uncertain'),
            ({'rear_view':False},'ego_scene_or_pose_uncertain')):
            result = self.run_section(declared=context(frame(0),**changes))
            self.assertIsNone(result.section)
            self.assertEqual(result.reason,reason)
        declared = context(frame(0))
        for count in (0,1,2):
            self.assertEqual(self.run_section(declared=replace(declared,bindings=declared.bindings[:count])).reason,
                             'insufficient_boundaries')

    def test_missing_observation_or_sources_stays_unknown(self):
        source = frame(0)
        self.assertEqual(section_for_vehicle(source,None,None,None).reason,'no_actual_observation')
        self.assertEqual(section_for_vehicle(source,vehicle(source),None,None).reason,'lanes_unknown')
        self.assertEqual(section_for_vehicle(source,vehicle(source),scripted(source),None).reason,'context_unknown')

    def test_foreign_clock_frame_vehicle_and_model_invalid(self):
        source = frame(0)
        for other in (frame(1), frame(0,time=1), replace(source,time_base=F(1,20)),
                      replace(source,first_pts=99),replace(source,width=97),replace(source,display_rotation=90)):
            with self.assertRaises(ValueError):
                section_for_vehicle(source,vehicle(source),scripted(other),context(source))
            with self.assertRaises(ValueError):
                section_for_vehicle(source,vehicle(source),scripted(source),context(other))
        for changes in ({'vehicle_identity':(1,2,9)},{'lane_model_id':'other'}):
            with self.assertRaises(ValueError):
                self.run_section(declared=context(source,**changes))

    def test_explicit_axes_not_inferred_from_rotation_metadata(self):
        source = replace(frame(0),display_rotation=90)
        self.assertIsNotNone(self.run_section(source=source).section)
        self.assertEqual(self.run_section(source=source,
            declared=context(source,coded_axes_confirmed=False)).reason,'coded_axes_uncertain')

    def test_binding_and_context_input_bounds(self):
        for slot,name,mask in ((True,'a',(True,)*71),(4,'a',()),(1,'../a',(True,)*71),
                               (1,'a',[True]*71),(0,'a',(True,)*71),(1,'a',(1,)*71)):
            with self.assertRaises(ValueError):
                BoundaryBinding(slot,name,mask)
        declared = context(frame(0))
        for changes in ({'bindings':list(declared.bindings)}, {'bindings':declared.bindings*2},
                        {'bindings':(declared.bindings[0],)*3}, {'ego':'yes'}, {'rear_view':1},
                        {'adjacency_confirmed':1}, {'coded_axes_confirmed':None},
                        {'context_id':'../ctx'}, {'road_id':''}, {'lane_model_id':'a/path'}):
            with self.assertRaises(ValueError):
                replace(declared,**changes)

    def test_input_frozen_and_report_does_not_alias(self):
        result = self.run_section()
        report = result.report()
        self.assertEqual(json.loads(json.dumps(report)),report)
        report['samples'][0]['anchors'][0] = 999
        report['section']['x'][0] = 999
        self.assertNotEqual(result.samples[0].anchors[0],999)
        self.assertNotEqual(result.section.x[0],999)
        self.assertEqual(report['declaration_status'],'caller_asserted_unverified')
        with self.assertRaises(FrozenInstanceError):
            result.section = None

    def test_no_numpy_cv_torch_import_even_for_ready_section(self):
        code = '''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy','torch','torchvision','cv2'):
            raise AssertionError('Forbidden dependency')
sys.meta_path.insert(0,Block())
from tests.test_lane_sections import Sections
assert Sections().run_section().section is not None
'''
        process = subprocess.run([sys.executable,'-B','-c',code],capture_output=True,timeout=20)
        self.assertEqual(process.returncode,0,process.stderr.decode(errors='replace'))


class Continuity(unittest.TestCase):
    def run_path(self, changes=None, *, missing_at=None):
        bridge = LaneSectionBridge(config())
        results = []
        for n,x in enumerate((24,24,24,43,47,50,60,65,70)):
            source = frame(n)
            target,batch = vehicle(source,x),scripted(source)
            declared = context(source,target,batch)
            if changes is not None and n >= 5:
                batch,declared = changes(batch,declared)
            if n == missing_at:
                target = None
            results.append(bridge.update(source,target,batch,declared))
        return bridge,results

    def test_ready_path_produces_same_adjacent_candidate_and_source_pts(self):
        _bridge,results = self.run_path()
        events = [r['lane']['candidate'] for r in results if r['lane']['candidate']]
        self.assertEqual(len(events),1)
        self.assertEqual(events[0].direction,'right')
        self.assertEqual(events[0].from_lane,('left','middle'))
        self.assertEqual(events[0].to_lane,('middle','right'))
        self.assertEqual(events[0].crossing[1].time,F(1,2))

    def test_lane_model_rotation_context_and_binding_change_reset(self):
        changes = (
            lambda b,c:(replace(b,model_id='lane-v2'),replace(c,lane_model_id='lane-v2')),
            lambda b,c:(replace(b,rotation_cw=180),c),
            lambda b,c:(b,replace(c,context_id='policy-v2')),
            lambda b,c:(b,replace(c,road_id='road-v2')),
            lambda b,c:(b,replace(c,bindings=tuple(replace(v,boundary_id=v.boundary_id+'-v2') for v in c.bindings))))
        for changed in changes:
            _bridge,results = self.run_path(changed)
            self.assertEqual(results[5]['reset_reason'],'lane_source_change')
            self.assertFalse(any(r['lane']['candidate'] for r in results))

    def test_unknown_and_missing_vehicle_cannot_bridge_event(self):
        _bridge,results = self.run_path(lambda b,c:(b,replace(c,ego='unknown')))
        self.assertFalse(any(r['lane']['candidate'] for r in results))
        _bridge,results = self.run_path(missing_at=5)
        self.assertEqual(results[5]['reset_reason'],'no_actual_observation')
        self.assertFalse(any(r['lane']['candidate'] for r in results))

    def test_exception_cancel_reset_and_fresh_retry(self):
        for failure in (RuntimeError('scripted failure'),KeyboardInterrupt()):
            bridge = LaneSectionBridge(config())
            for n in range(3):
                source = frame(n)
                bridge.update(source,vehicle(source),scripted(source),context(source))
            with patch.object(bridge._engine,'update',side_effect=failure):
                source = frame(3)
                with self.assertRaises(type(failure)):
                    bridge.update(source,vehicle(source),scripted(source),context(source))
            source = frame(4)
            resumed = bridge.update(source,vehicle(source,x=70),scripted(source),context(source))
            self.assertEqual(resumed['lane']['status'],'warming')
            self.assertIsNone(resumed['lane']['candidate'])

    def test_malformed_context_resets_before_next_valid_update(self):
        bridge = LaneSectionBridge(config())
        for n in range(3):
            source = frame(n)
            bridge.update(source,vehicle(source),scripted(source),context(source))
        source = frame(3)
        with self.assertRaises(ValueError):
            bridge.update(source,vehicle(source),scripted(source),context(source,lane_model_id='foreign'))
        source = frame(4)
        self.assertEqual(bridge.update(source,vehicle(source,70),scripted(source),context(source))['lane']['status'],'warming')


if __name__ == '__main__':
    unittest.main()
