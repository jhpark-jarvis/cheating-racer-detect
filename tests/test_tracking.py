"""ML-free contracts/association tests; fake motion is not native verification."""

from dataclasses import replace
from fractions import Fraction
from itertools import count
import json
import unittest
from unittest.mock import patch

from cheating_racer_detect.tracking import Box, Detection, Frame, MotionConfig, Policy, Tracker
from cheating_racer_detect.tracking.tracker import overlap


def config():
    return MotionConfig((4., 4., .02, .02), (.25, .25, .0025, .0025), (1., 1., .1, .1, 100., 100., 1., 1.))


def policy(**changes):
    result = Policy(Fraction(3, 5), Fraction(3, 10), .2, .7, .2, .95, (2, 5, 7), config())
    return replace(result, **changes)


def frame(ordinal=0, pts=None, **changes):
    value = Frame("synthetic", 0, ordinal, 500 + ordinal if pts is None else pts,
                  Fraction(1, 10), 500, 96, 64)
    return replace(value, **changes)


def detection(source, index=0, *, x=10., score=.9, label=2):
    return Detection(index, label, Box(x, 10, x+20, 30), score, "scripted-v1", source)


class FakeMotion:
    def __init__(self, box, configuration):
        self.box = box
        self.steps = []

    def predict(self, dt):
        self.steps.append(dt)

    def correct(self, box):
        self.box = box


class Contracts(unittest.TestCase):
    def test_exact_offset_clock_and_raster_report(self):
        value = frame(2, 503)
        self.assertEqual(value.time, Fraction(3, 10))
        self.assertEqual(value.report()["pts"], 503)
        self.assertEqual(value.report()["normalized_time"], "3/10")
        self.assertEqual(value.report()["coded_size"], [96, 64])

    def test_bad_source_frame_values(self):
        for changes in ({"source_id": "C:/private.mp4"}, {"source_id": ""}, {"ordinal": True},
                        {"pts": 499}, {"pts": 2**52}, {"time_base": .1}, {"width": 0},
                        {"height": 16385}, {"display_rotation": 45}, {"stream_index": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                frame(**changes)

    def test_bad_box_and_half_open_overlap(self):
        for values in ((0, 0, 0, 1), (0, 0, 1, float("nan")), (True, 0, 2, 2), (-1e308, 0, 1e308, 1)):
            with self.assertRaises(ValueError):
                Box(*values)
        self.assertEqual(overlap(Box(0, 0, 10, 10), Box(10, 0, 20, 10)), 0)
        self.assertAlmostEqual(overlap(Box(0, 0, 10, 10), Box(5, 0, 15, 10)), 1/3)
        self.assertIsNone(Box(-2, 0, -1, 10).clip(96, 64))

    def test_bad_detection_provenance_geometry_and_scores(self):
        value = detection(frame())
        for changes in ({"score": True}, {"score": 1.01}, {"index": -1}, {"class_id": -1},
                        {"model_id": "a/path"}, {"frame": None}, {"box": Box(90, 0, 100, 10)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(value, **changes)

    def test_policy_requires_exact_time_and_valid_gates_bounds(self):
        for changes in ({"max_gap": .1}, {"max_gap": Fraction(7, 10)}, {"max_unobserved": Fraction(0)},
                        {"min_score": .8}, {"duplicate_iou": .1}, {"min_iou": 0}, {"classes": (2, 2)},
                        {"classes": (True,)}, {"motion": None}, {"max_tracks": 129}, {"max_detections": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                policy(**changes)

    def test_noise_config_copy_and_validation(self):
        spectral = [1, 2, 3, 4]
        result = MotionConfig(spectral, [1]*4, [1]*8)
        spectral[0] = -1
        self.assertEqual(result.spectral[0], 1)
        for fields in (([-1]*4, [1]*4, [1]*8), ([0]*4, [0]*4, [1]*8),
                       ([0]*4, [1]*4, [1]*7), ([float("nan")]*4, [1]*4, [1]*8)):
            with self.assertRaises(ValueError):
                MotionConfig(*fields)


class Association(unittest.TestCase):
    def tracker(self, **changes):
        return Tracker(policy(**changes), motion_factory=FakeMotion)

    def test_variable_dt_preserves_id_and_passes_exact_intervals(self):
        tracker = self.tracker()
        results = []
        for ordinal, pts in enumerate((500, 501, 503, 506)):
            source = frame(ordinal, pts)
            results.append(tracker.update(source, [detection(source, ordinal, x=10+ordinal)]))
        self.assertTrue(tracker.supports_variable_dt)
        self.assertEqual([r["segment"] for r in results], [1]*4)
        self.assertEqual([r["delta_seconds"] for r in results], [None, "1/10", "1/5", "3/10"])
        self.assertEqual(len({tuple(r["tracks"][0]["identity"]) for r in results}), 1)
        self.assertEqual(next(iter(tracker._tracks.values())).motion.steps,
                         [Fraction(1, 10), Fraction(1, 5), Fraction(3, 10)])

    def test_low_score_joins_but_cannot_start_new_identity(self):
        tracker = self.tracker()
        first = frame()
        initial = tracker.update(first, [detection(first)])
        second = frame(1)
        result = tracker.update(second, [detection(second, 3, score=.3), detection(second, 4, x=60, score=.3)])
        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(result["tracks"][0]["identity"], initial["tracks"][0]["identity"])
        self.assertEqual(result["tracks"][0]["detection_index"], 3)
        self.assertEqual(result["tracks"][0]["model_score"], .3)
        self.assertEqual(len(result["detections"]), 2)

    def test_subthreshold_detections_preserved_but_not_observed(self):
        tracker = self.tracker()
        first, second = frame(), frame(1)
        tracker.update(first, [detection(first)])
        result = tracker.update(second, [detection(second, score=.1)])
        self.assertEqual(result["tracks"][0]["state"], "prediction_only")
        self.assertEqual(len(result["detections"]), 1)

    def test_prediction_only_has_no_observation_fields(self):
        tracker = self.tracker()
        initial = frame()
        tracker.update(initial, [detection(initial)])
        result = tracker.update(frame(1), [])
        record = result["tracks"][0]
        for key in ("detection_index", "detection_bbox", "model_id", "model_score"):
            self.assertIsNone(record[key])
        self.assertEqual(record["last_observed_time"], "0")
        self.assertEqual(record["unobserved_seconds"], "1/10")
        self.assertEqual(record["identity_status"], "unverified")

    def test_exact_expiry_boundary_then_new_id(self):
        tracker = self.tracker()
        initial = frame()
        original = tracker.update(initial, [detection(initial)])["tracks"][0]["identity"]
        for ordinal in (1, 2, 3):
            self.assertEqual(tracker.update(frame(ordinal), [])["tracks"][0]["identity"], original)
        self.assertEqual(tracker.update(frame(4), [])["tracks"], [])
        source = frame(5)
        self.assertNotEqual(tracker.update(source, [detection(source)])["tracks"][0]["identity"], original)

    def test_class_namespaces_do_not_associate(self):
        tracker = self.tracker()
        first, second = frame(), frame(1)
        original = tracker.update(first, [detection(first, label=2)])["tracks"][0]["identity"]
        result = tracker.update(second, [detection(second, label=5)])
        self.assertEqual(len(result["tracks"]), 2)
        self.assertEqual(result["tracks"][0]["identity"], original)
        self.assertEqual(result["tracks"][0]["state"], "prediction_only")
        self.assertEqual(result["tracks"][1]["identity"][1], 5)

    def test_many_candidates_terminate_history_and_suppress_births(self):
        tracker = self.tracker()
        first, second = frame(), frame(1)
        original = tracker.update(first, [detection(first)])["tracks"][0]["identity"]
        result = tracker.update(second, [detection(second, 3, x=5), detection(second, 4, x=15)])
        self.assertEqual(result["tracks"], [])
        self.assertEqual(result["ambiguities"][0]["identities"], [original])
        self.assertEqual(result["ambiguities"][0]["detection_indices"], [3, 4])
        third = frame(2)
        self.assertNotEqual(tracker.update(third, [detection(third)])["tracks"][0]["identity"], original)

    def test_shared_detection_drops_all_affected_identities(self):
        tracker = self.tracker()
        first, second = frame(), frame(1)
        tracker.update(first, [detection(first, 1, x=5), detection(first, 2, x=25)])
        result = tracker.update(second, [detection(second, x=15)])
        self.assertEqual(result["tracks"], [])
        self.assertEqual(len(result["ambiguities"][0]["identities"]), 2)

    def test_duplicate_boxes_with_distinct_indices_do_not_start_tracks(self):
        tracker = self.tracker()
        source = frame()
        result = tracker.update(source, [detection(source, 1), detection(source, 2)])
        self.assertEqual(result["tracks"], [])
        self.assertEqual(result["ambiguities"][0]["detection_indices"], [1, 2])

    def test_duplicate_component_connects_and_drops_existing_history(self):
        tracker = self.tracker()
        first, second = frame(), frame(1)
        tracker.update(first, [detection(first)])
        result = tracker.update(second, [detection(second, 1), detection(second, 2)])
        self.assertEqual(result["tracks"], [])
        self.assertEqual(len(result["ambiguities"][0]["identities"]), 1)

    def test_disjoint_component_survives_other_ambiguity(self):
        tracker = self.tracker()
        first, second = frame(), frame(1)
        original = tracker.update(first, [detection(first, 1), detection(first, 2, x=60)])["tracks"][1]["identity"]
        result = tracker.update(second, [detection(second, 3, x=5), detection(second, 4, x=15),
                                         detection(second, 5, x=60)])
        self.assertEqual([r["identity"] for r in result["tracks"]], [original])
        self.assertEqual(result["tracks"][0]["detection_index"], 5)

    def test_input_order_does_not_change_results(self):
        outputs = []
        for reverse in (False, True):
            tracker = self.tracker()
            first, second = frame(), frame(1)
            observations = [detection(first, 8), detection(first, 2, x=60)]
            tracker.update(first, reversed(observations) if reverse else observations)
            observations = [detection(second, 7), detection(second, 3, x=60)]
            outputs.append(tracker.update(second, reversed(observations) if reverse else observations))
        self.assertEqual(outputs[0], outputs[1])

    def test_reports_do_not_alias_tracker_or_old_records(self):
        tracker = self.tracker()
        first, second = frame(), frame(1)
        initial = tracker.update(first, [detection(first)])
        before = json.dumps(initial)
        tracker.update(second, [detection(second, x=11)])
        self.assertEqual(json.dumps(initial), before)
        initial["tracks"][0]["identity"][2] = -1
        initial["tracks"][0]["estimated_bbox"][0] = -100
        result = tracker.update(frame(2), [])
        self.assertGreater(result["tracks"][0]["identity"][2], 0)
        self.assertGreater(result["tracks"][0]["estimated_bbox"][0], 0)


class Recovery(unittest.TestCase):
    def tracker(self, **changes):
        return Tracker(policy(**changes), motion_factory=FakeMotion)

    def test_source_geometry_gap_and_missing_frame_split_segments(self):
        for changes, reason in (({"source_id": "other"}, "source_or_geometry_change"),
                                ({"width": 97}, "source_or_geometry_change"),
                                ({"display_rotation": 90}, "source_or_geometry_change"),
                                ({"ordinal": 2}, "missing_decoded_frame"),
                                ({"pts": 507}, "time_gap")):
            tracker = self.tracker()
            first = frame()
            original = tracker.update(first, [detection(first)])["tracks"][0]["identity"]
            second = replace(frame(1), **changes)
            result = tracker.update(second, [detection(second)])
            self.assertEqual(result["segment"], 2)
            self.assertEqual(result["reset_reason"], reason)
            self.assertNotEqual(result["tracks"][0]["identity"], original)

    def test_bad_time_origin_order_and_input_poison_then_recover(self):
        for bad in (frame(1, 500), frame(0, 501), frame(1, first_pts=499),
                    frame(1, time_base=Fraction(1, 20)), None):
            tracker = self.tracker()
            first = frame()
            tracker.update(first, [detection(first)])
            with self.assertRaises(ValueError):
                tracker.update(bad, [])
            restored = frame(2, 502)
            result = tracker.update(restored, [detection(restored)])
            self.assertEqual(result["reset_reason"], "failed_update")
            self.assertEqual(result["segment"], 2)

    def test_backward_time_rejected(self):
        tracker = self.tracker()
        first = frame(0, 502)
        tracker.update(first, [detection(first)])
        with self.assertRaises(ValueError):
            tracker.update(frame(1, 501), [])

    def test_stale_duplicate_or_unsupported_detection_rejected(self):
        first, second = frame(), frame(1)
        for values in ([detection(first)], [detection(second), detection(second)],
                       [detection(second, label=0)], [None]):
            tracker = self.tracker()
            tracker.update(first, [detection(first)])
            with self.assertRaises(ValueError):
                tracker.update(second, values)
            self.assertEqual(tracker._tracks, {})

    def test_generator_is_bounded_and_tracks_limit_fail_closed(self):
        tracker = self.tracker(max_detections=2)
        source = frame()
        yielded = []
        def many():
            for index in count():
                yielded.append(index)
                yield detection(source, index)
        with self.assertRaises(ValueError):
            tracker.update(source, many())
        self.assertEqual(yielded, [0, 1, 2])
        tracker = self.tracker(max_tracks=1)
        with self.assertRaises(ValueError):
            tracker.update(source, [detection(source, 1), detection(source, 2, x=60)])
        self.assertEqual(tracker._tracks, {})
        self.assertEqual(tracker.update(source, [detection(source)])["reset_reason"], "failed_update")

    def test_fault_and_cancel_in_predict_correct_or_factory_recover(self):
        for stage in ("predict", "correct", "factory"):
            for error in (RuntimeError("injected"), KeyboardInterrupt()):
                tracker = self.tracker()
                first, second = frame(), frame(1)
                tracker.update(first, [detection(first)])
                target = tracker if stage == "factory" else next(iter(tracker._tracks.values())).motion
                name = "_factory" if stage == "factory" else stage
                values = [detection(second, x=60 if stage == "factory" else 10)]
                with patch.object(target, name, side_effect=error):
                    with self.assertRaises(type(error)):
                        tracker.update(second, values)
                self.assertEqual(tracker._tracks, {})
                restored = frame(2)
                self.assertEqual(tracker.update(restored, [detection(restored)])["reset_reason"], "failed_update")

    def test_manual_reset_starts_new_segment_without_id_reuse(self):
        tracker = self.tracker()
        first = frame()
        original = tracker.update(first, [detection(first)])["tracks"][0]["identity"]
        tracker.reset()
        second = frame(1)
        result = tracker.update(second, [detection(second)])
        self.assertEqual(result["reset_reason"], "manual_reset")
        self.assertNotEqual(result["tracks"][0]["identity"], original)

    def test_reset_reason_guard_and_dependency_failure_recovery(self):
        tracker = Tracker(policy())
        with self.assertRaises(ValueError):
            tracker.reset("a/private/path")
        source = frame()
        with patch.dict("sys.modules", {"cheating_racer_detect.tracking.motion": None}):
            with self.assertRaises(ModuleNotFoundError):
                tracker.update(source, [detection(source)])
        self.assertEqual(tracker._tracks, {})
        # A non-dependency import failure must not be silently reclassified or
        # cause package installation; explicit trusted factory enables retry.
        tracker._factory = FakeMotion
        self.assertEqual(tracker.update(source, [detection(source)])["reset_reason"], "failed_update")


if __name__ == "__main__":
    unittest.main()
