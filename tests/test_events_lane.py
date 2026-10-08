"""Synthetic injected lanes/boxes, never real-world detection accuracy."""

from dataclasses import replace
from fractions import Fraction as F
import unittest

from cheating_racer_detect.events import Candidate, LaneConfig, LaneEngine, LaneSection, Vehicle
from cheating_racer_detect.tracking import Box, Detection, Frame, Tracker
from tests.test_tracking import FakeMotion, policy


def frame(n, time=None, **changes):
    return replace(Frame('synthetic', 0, n, 100+(n if time is None else time), F(1, 10), 100, 96, 64), **changes)


def vehicle(f, x=24, key=(1, 2, 1)):
    return Vehicle(f, key, Detection(0, key[1], Box(x-6, 10, x+6, 30), .9, 'scripted', f))


def lanes(f, shift=0, **changes):
    return replace(LaneSection(f, 'road', ('a', 'b', 'c'), (float(shift), 48.+shift, 96.), 30.,
                               'stable', 'ordinary', True), **changes)


def config():
    return LaneConfig('synthetic-v1', .15, F(1, 5), F(3, 5), 10.)


class LaneTests(unittest.TestCase):
    def run_path(self, xs, times=None, **lane_changes):
        engine = LaneEngine(config())
        events = []
        for n, x in enumerate(xs):
            f = frame(n, None if times is None else times[n])
            result = engine.update(f, vehicle(f, x), lanes(f, **lane_changes))
            if result['candidate']:
                events.append(result['candidate'])
        return engine, events

    def test_adjacent_right_brackets_and_owned_report(self):
        _, events = self.run_path([24, 24, 24, 43, 47, 50, 60, 65, 70])
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.direction, 'right')
        self.assertEqual([f.time for f in event.crossing], [F(2, 5), F(1, 2)])
        self.assertEqual([f.time for f in event.start], [F(1, 5), F(3, 10)])
        self.assertEqual([f.time for f in event.completion], [F(3, 5), F(4, 5)])
        report = event.report()
        report['identity'][0] = 99
        self.assertEqual(event.identity, (1, 2, 1))

    def test_left_vfr_and_repeated_transitions(self):
        _, events = self.run_path([72, 72, 72, 50, 46, 35, 25], [0, 1, 3, 4, 6, 7, 10])
        self.assertEqual([e.direction for e in events], ['left'])
        self.assertEqual(events[0].crossing[1].time, F(3, 5))
        _, events = self.run_path([24]*3+[45, 55]+[72]*3+[50, 46]+[24]*3)
        self.assertEqual([e.direction for e in events], ['right', 'left'])

    def test_jitter_curve_and_abort_are_not_events(self):
        for xs in ([24]*3+[47, 49]*10, [24]*3+[44, 50, 55, 45]+[24]*3, [24]*12):
            self.assertEqual(self.run_path(xs)[1], [])
        engine = LaneEngine(config())
        events = []
        # Common camera/curve shift, with target and lane boundaries shifted together.
        for n, shift in enumerate(range(12)):
            f = frame(n)
            events.append(engine.update(f, vehicle(f, 24+shift), lanes(f, shift))['candidate'])
        self.assertTrue(all(e is None for e in events))

    def test_ego_merge_pose_and_no_observation_gate(self):
        for changes in ({'ego': 'changing'}, {'ego': 'unknown'}, {'scene': 'merge'}, {'rear_view': False}):
            engine, events = self.run_path([24]*3+[72]*4, **changes)
            self.assertEqual(events, [])
            f = frame(7)
            self.assertEqual(engine.update(f, None, lanes(f))['reason'], 'no_actual_observation')

    def test_identity_road_gap_and_geometry_reset(self):
        for kind in ('id', 'road', 'ordinal', 'time', 'source', 'size'):
            engine = LaneEngine(config())
            for n in range(3):
                f = frame(n)
                engine.update(f, vehicle(f), lanes(f))
            events = []
            for n in range(3, 7):
                f = frame(n+2 if kind == 'ordinal' else n, n+10 if kind == 'time' else None,
                          source_id='other' if kind == 'source' else 'synthetic', width=100 if kind == 'size' else 96)
                result = engine.update(f, vehicle(f, 72, (2, 2, 1) if kind == 'id' else (1, 2, 1)),
                                       lanes(f, road_id='other' if kind == 'road' else 'road'))
                events.append(result['candidate'])
            self.assertTrue(all(e is None for e in events), kind)

    def test_invalid_or_stale_resets_then_retry(self):
        engine = LaneEngine(config())
        for n in range(3):
            f = frame(n)
            engine.update(f, vehicle(f), lanes(f))
        for bad in (frame(2), frame(3, time_base=F(1, 5))):
            with self.assertRaises(ValueError):
                engine.update(bad, vehicle(bad), lanes(bad))
            f = frame(0)
            self.assertEqual(engine.update(f, vehicle(f), lanes(f))['status'], 'warming')
        f = frame(1)
        with self.assertRaises(ValueError):
            engine.update(f, vehicle(frame(0)), lanes(f))
        self.assertEqual(engine.update(frame(0), vehicle(frame(0)), lanes(frame(0)))['status'], 'warming')

    def test_tracker_provenance_predictions_and_ambiguity(self):
        tracker = Tracker(policy(), motion_factory=FakeMotion)
        f = frame(0)
        report = tracker.update(f, [vehicle(f).detection])
        key = tuple(report['tracks'][0]['identity'])
        self.assertEqual(Vehicle.from_tracking(f, report, key).box, vehicle(f).box)
        current = frame(1)
        prediction = tracker.update(current, [])
        self.assertIsNone(Vehicle.from_tracking(current, prediction, key))
        with self.assertRaises(ValueError):
            Vehicle.from_tracking(current, report, key)

    def test_contract_bounds_unknown_lanes_and_nonadjacent(self):
        for changes in ({'margin': 0}, {'dwell': .2}, {'max_gap': F(0)}, {'min_lane_pixels': float('nan')}):
            with self.assertRaises(ValueError):
                replace(config(), **changes)
        f = frame(0)
        for changes in ({'x': (0., 48., 48.)}, {'boundary_ids': ('a', 'a', 'c')}, {'rear_view': 1}):
            with self.assertRaises(ValueError):
                lanes(f, **changes)
        engine = LaneEngine(config())
        self.assertEqual(engine.update(f, vehicle(f), None)['status'], 'unknown')
        self.assertEqual(engine.update(f, vehicle(f, 8), lanes(f, x=(9., 48., 96.)))['status'], 'unknown')
        for n in range(3):
            f = frame(n)
            section = lanes(f, x=(0., 32., 64., 96.), boundary_ids=('a', 'b', 'c', 'd'))
            engine.update(f, vehicle(f, 16), section)
        f = frame(3)
        result = engine.update(f, vehicle(f, 80), lanes(f, x=(0., 32., 64., 96.), boundary_ids=('a', 'b', 'c', 'd')))
        self.assertEqual(result['reason'], 'non_adjacent_or_incomplete')
        with self.assertRaises(ValueError):
            Candidate((1, 2, 1), 'road', ('a', 'b'), ('c', 'd'), 'right', (f, f), (f, f), (f, f), 'policy')


if __name__ == '__main__':
    unittest.main()
