"""Bounded stdlib provenance persistence, legacy bytes and history parity."""

from dataclasses import FrozenInstanceError, replace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import LaneEvidence, LaneSectionBridge
from cheating_racer_detect.events.inputs import (EventInputs, InputLaneEngine, MAX_BYTES,
                                               load_inputs, save_inputs)
from cheating_racer_detect.tracking import Box
from scripts.demo_lane_replay import with_lane_evidence
from tests.test_event_inputs import fixture


def scenario():
    inputs = with_lane_evidence(fixture())
    rows = []
    for row, x in zip(inputs.rows, (24,24,24,30,36,42,50,58,64,70,70,70)):
        detection = replace(row.vehicle.detection, box=Box(x-12,10,x+12,30))
        rows.append(replace(row, vehicle=replace(row.vehicle, detection=detection),
                            left=replace(row.left, box=None, status='model_uncertain'),
                            right=replace(row.right, box=None, status='model_uncertain')))
    return replace(inputs, rows=tuple(rows))


def replay_rows(inputs):
    engine = InputLaneEngine(inputs.lane_config)
    return tuple(engine.update(row) for row in inputs.rows)


def changed_row(row, **changes):
    evidence = replace(row.lane_evidence, **changes)
    return replace(row, lanes=evidence.section(row.frame, row.vehicle).section, lane_evidence=evidence)


class Evidence(unittest.TestCase):
    def test_legacy_has_exact_old_fields_bytes_and_candidate_policy(self):
        legacy = fixture()
        value = legacy.report()
        self.assertEqual(value['schema'], 'experimental-event-inputs-v1')
        self.assertEqual(set(value['rows'][0]), {'frame','vehicle','lanes','left','right'})
        expected = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':'), allow_nan=False)+'\n').encode()
        self.assertEqual(legacy.to_bytes(), expected)
        self.assertEqual(EventInputs.from_bytes(expected).to_bytes(), expected)
        self.assertTrue(all(r['section'] is None and r['reset_reason'] is None for r in replay_rows(legacy)))

    def test_v2_roundtrip_preserves_full_geometry_masks_context_and_digest(self):
        inputs = scenario()
        self.assertEqual(inputs.report()['schema'], 'experimental-event-inputs-v2')
        restored = EventInputs.from_bytes(inputs.to_bytes())
        self.assertEqual(restored, inputs)
        self.assertEqual(restored.to_bytes(), inputs.to_bytes())
        self.assertEqual(inputs.sha256, hashlib.sha256(inputs.to_bytes()).hexdigest())
        self.assertEqual(restored.rows[3].lane_evidence.batch.frame, restored.rows[3].frame)
        self.assertEqual(len(restored.rows[3].lane_evidence.batch.slots[0].points), 72)
        self.assertEqual(restored.rows[3].lane_evidence.context.bindings[0].visible_segments, (True,)*71)

    def test_frozen_and_owned_report(self):
        inputs = scenario()
        report = inputs.report()
        report['rows'][0]['lane_evidence']['batch']['slots'][0]['points'][0][0] = 80
        report['rows'][0]['lane_evidence']['context']['bindings'][0]['visible_segments'][0] = False
        self.assertEqual(inputs.rows[0].lane_evidence.batch.slots[0].points[0][0], 0.)
        self.assertTrue(inputs.rows[0].lane_evidence.context.bindings[0].visible_segments[0])
        with self.assertRaises(FrozenInstanceError):
            inputs.rows[0].lane_evidence.context = None

    def test_live_replay_exact_section_candidate_and_anchor_parity(self):
        inputs = scenario()
        restored = EventInputs.from_bytes(inputs.to_bytes())
        live = LaneSectionBridge(inputs.lane_config)
        recorded = replay_rows(restored)
        for row, replayed in zip(inputs.rows, recorded):
            e = row.lane_evidence
            actual = live.update(row.frame, row.vehicle, e.batch, e.context)
            self.assertEqual(replayed, actual)
        events = [r['lane']['candidate'] for r in recorded if r['lane']['candidate']]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].direction, 'right')

    def test_model_rotation_context_road_binding_changes_reset_history(self):
        base = scenario()
        for kind in ('model','rotation','context','road','binding'):
            rows = []
            for n, row in enumerate(base.rows):
                e = row.lane_evidence
                b, c = e.batch, e.context
                if n >= 5:
                    if kind == 'model':
                        b, c = replace(b, model_id='new-model'), replace(c, lane_model_id='new-model')
                    elif kind == 'rotation':
                        b = replace(b, rotation_cw=90)
                    elif kind == 'context':
                        c = replace(c, context_id='new-context')
                    elif kind == 'road':
                        c = replace(c, road_id='new-road')
                    else:
                        c = replace(c, bindings=tuple(replace(binding, boundary_id='new-'+binding.boundary_id)
                                                     for binding in c.bindings))
                rows.append(changed_row(row, batch=b, context=c))
            inputs = EventInputs.from_bytes(replace(base, rows=tuple(rows)).to_bytes())
            results = replay_rows(inputs)
            with self.subTest(kind=kind):
                self.assertEqual(results[5]['reset_reason'], 'lane_source_change')
                self.assertFalse(any(r['lane']['candidate'] for r in results))

    def test_unknown_masks_missing_samples_gates_and_missing_actual_do_not_fill(self):
        base = scenario()
        row, index = base.rows[5], 5
        e = row.lane_evidence
        binding = replace(e.context.bindings[0], visible_segments=(False,)*71)
        invisible = replace(e.context, bindings=(binding,*e.context.bindings[1:]))
        b = e.batch
        gap = replace(b.slots[0], points=(None,)*72)
        missing = replace(b, slots=(gap,*b.slots[1:]))
        unknowns = [changed_row(row, context=None), changed_row(row, batch=None),
                    changed_row(row, context=invisible), changed_row(row, batch=missing),
                    changed_row(row, context=replace(e.context, ego='unknown')),
                    replace(row, vehicle=None, lanes=None, lane_evidence=LaneEvidence(b,e.context))]
        for item in unknowns:
            restored = EventInputs.from_bytes(replace(base, rows=base.rows[:index]+(item,)+base.rows[index+1:]).to_bytes())
            result = replay_rows(restored)
            with self.subTest(reason=result[index]['reset_reason']):
                self.assertIsNone(result[index]['section'].section)
                self.assertIsNotNone(result[index]['reset_reason'])
                self.assertFalse(any(r['lane']['candidate'] for r in result))

    def test_explicit_completely_unknown_evidence_is_not_legacy(self):
        base = scenario()
        inputs = replace(base, rows=tuple(replace(r, lane_evidence=LaneEvidence(None,None), lanes=None) for r in base.rows))
        restored = EventInputs.from_bytes(inputs.to_bytes())
        self.assertEqual(restored.report()['schema'], 'experimental-event-inputs-v2')
        self.assertTrue(all(r['reset_reason']=='lanes_unknown' for r in replay_rows(restored)))

    def test_sparse_rows_leave_original_clock_and_break_pending_event(self):
        base = scenario()
        sparse = replace(base, rows=base.rows[:5]+base.rows[6:])
        restored = EventInputs.from_bytes(sparse.to_bytes())
        self.assertEqual(restored.rows[5].frame.ordinal, 6)
        self.assertFalse(any(r['lane']['candidate'] for r in replay_rows(restored)))

    def test_v2_null_missing_mixed_unknown_fields_and_version_downgrade_rejected(self):
        value = scenario().report()
        invalid = []
        for kind in ('null','missing','extra','v1','decoder','marker'):
            report = deepcopy(value)
            row = report['rows'][0]
            if kind == 'null': row['lane_evidence'] = None
            elif kind == 'missing': del row['lane_evidence']
            elif kind == 'extra': row['lane_evidence']['extra'] = 0
            elif kind == 'v1': report['schema'] = 'experimental-event-inputs-v1'
            elif kind == 'decoder': row['lane_evidence']['decoder_id'] = 'future-decoder'
            else: row['lane_evidence']['reset_reason'] = None
            invalid.append(report)
        for report in invalid:
            with self.assertRaises(AppError):
                EventInputs.from_bytes(json.dumps(report).encode())
        base = scenario()
        with self.assertRaises(ValueError):
            replace(base, rows=(replace(base.rows[0], lane_evidence=None), *base.rows[1:]))

    def test_nested_types_counts_bounds_and_semantic_drift_rejected(self):
        paths = [
            (('batch','slots',0,'exists'), [True]*71),
            (('batch','slots',0,'points',0), [0,0,0]),
            (('batch','slots',0,'points',0), [1000,0]),
            (('batch','slots',0,'slot'), True),
            (('batch','slots',0,'exists',0), 1),
            (('batch','rotation_cw'), 45), (('batch','slots'), []),
            (('context','bindings',0,'visible_segments',0), 1),
            (('context','adjacency_confirmed'), 1),
            (('context','vehicle_identity'), [2,2,1]),
            (('context','lane_model_id'), 'wrong-model'),
            (('context','road_id'), 'different-road'),
        ]
        for path, value in paths:
            report = scenario().report()
            target = report['rows'][0]['lane_evidence']
            for key in path[:-1]: target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(AppError):
                EventInputs.from_bytes(json.dumps(report).encode())

    def test_derived_section_coordinates_and_gates_cannot_disagree(self):
        for key, value in (('x',[1.,48.,96.]), ('ego','unknown'), ('road_id','another-road')):
            report = scenario().report()
            report['rows'][0]['lanes'][key] = value
            with self.assertRaises(AppError):
                EventInputs.from_bytes(json.dumps(report).encode())

    def test_foreign_frame_and_vehicle_context_rejected_even_when_unknown(self):
        row = scenario().rows[0]
        e = row.lane_evidence
        with self.assertRaises(ValueError):
            replace(row, lane_evidence=replace(e, batch=replace(e.batch, frame=replace(row.frame, pts=99))))
        with self.assertRaises(ValueError):
            replace(row, lanes=None, lane_evidence=LaneEvidence(None, replace(e.context, frame=replace(row.frame, pts=99))))
        base = scenario()
        wrong = replace(row, vehicle=None, lanes=None,
                        lane_evidence=LaneEvidence(None,replace(e.context, vehicle_identity=(9,2,1))))
        with self.assertRaises(ValueError):
            replace(base, rows=(wrong, *base.rows[1:]))

    def test_any_provenance_change_changes_canonical_input_identity(self):
        base = scenario()
        row = base.rows[0]
        binding = replace(row.lane_evidence.context.bindings[0], visible_segments=(False,)+(True,)*70)
        context = replace(row.lane_evidence.context, bindings=(binding,*row.lane_evidence.context.bindings[1:]))
        changed = replace(base, rows=(changed_row(row, context=context), *base.rows[1:]))
        self.assertNotEqual(base.sha256, changed.sha256)
        # Removing ALL provenance and explicitly changing version is a new,
        # unauthenticated legacy document, never identical to the original hash.
        legacy = replace(base, rows=tuple(replace(r, lane_evidence=None) for r in base.rows))
        self.assertNotEqual(base.sha256, legacy.sha256)

    def test_actual_payload_limit_not_just_row_limit(self):
        inputs = scenario()
        self.assertLess(len(inputs.to_bytes()), MAX_BYTES)
        with patch('cheating_racer_detect.events.inputs.MAX_BYTES', len(inputs.to_bytes())-1):
            with self.assertRaises(ValueError):
                replace(inputs, pre=inputs.pre)
            with self.assertRaises(AppError):
                EventInputs.from_bytes(inputs.to_bytes())

    def test_save_failure_cancel_retry_existing_result_and_temp_cleanup(self):
        inputs = scenario()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for failure in (OSError('full'), KeyboardInterrupt()):
                with patch('cheating_racer_detect.events.inputs.write_bytes', side_effect=failure):
                    with self.assertRaises((AppError,KeyboardInterrupt)):
                        save_inputs(inputs,root/'saved')
                self.assertEqual(list(root.iterdir()), [])
            save_inputs(inputs,root/'saved')
            self.assertEqual(load_inputs(root/'saved/inputs.json'), inputs)
            with self.assertRaises(AppError): save_inputs(inputs,root/'saved')
            self.assertEqual(load_inputs(root/'saved/inputs.json').sha256, inputs.sha256)

    def test_engine_failure_cancel_mixed_mode_resets_and_retries_fresh(self):
        inputs = scenario()
        engine = InputLaneEngine(inputs.lane_config)
        for failure in (ValueError('invalid'), KeyboardInterrupt()):
            engine.update(inputs.rows[0])
            with patch.object(engine._bridge, 'update', side_effect=failure):
                with self.assertRaises(type(failure)): engine.update(inputs.rows[1])
            actual = [engine.update(row) for row in inputs.rows]
            self.assertEqual(actual, list(replay_rows(inputs)))
            engine.reset()
        engine.update(inputs.rows[0])
        with self.assertRaises(ValueError): engine.update(fixture().rows[1])
        self.assertIsNone(engine._mode)
        self.assertEqual(engine.update(inputs.rows[0]), replay_rows(inputs)[0])

    def test_base_import_and_v2_parse_need_no_native_dependencies(self):
        child = subprocess.run([sys.executable,'-c',
            'import sys; from cheating_racer_detect.events import LaneEvidence; '
            'from cheating_racer_detect.events.inputs import EventInputs; '
            'p=EventInputs.from_bytes(sys.stdin.buffer.read()); '
            'assert p.rows[0].lane_evidence is not None; '
            'assert not any(n in sys.modules for n in ("numpy","cv2","torch"))'],
            input=scenario().to_bytes(), capture_output=True, timeout=20)
        self.assertEqual(child.returncode,0,child.stderr.decode())


if __name__ == '__main__':
    unittest.main()
