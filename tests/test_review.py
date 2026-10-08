"""Review summary tests in a standard-library-only runtime."""

import copy
from fractions import Fraction
import unittest

from cheating_racer_detect.review import summarize
from cheating_racer_detect.tracking import Tracker
from tests.test_tracking import FakeMotion, detection, frame, policy


class Records(unittest.TestCase):
    def setUp(self):
        self.frame = frame()
        self.tracker = Tracker(policy(), motion_factory=FakeMotion)
        self.report = self.tracker.update(self.frame, [detection(self.frame)])

    def test_observation_exact_time_and_schema(self):
        result = summarize(self.frame, self.report)
        self.assertEqual(result["schema"], "experimental-review-frame-v1")
        self.assertEqual(result["frame"], self.frame.report())
        self.assertEqual(result["annotations"][0]["state"], "observed")
        self.assertEqual(result["annotations"][0]["detection_index"], 0)
        self.assertEqual(result["annotations"][0]["identity_status"], "unverified")

    def test_prediction_exact_age_and_null_evidence(self):
        current = frame(1, 502)
        result = summarize(current, self.tracker.update(current, []))
        predicted = result["annotations"][0]
        self.assertEqual(result["delta_seconds"], "1/5")
        self.assertEqual(predicted["unobserved_seconds"], "1/5")
        self.assertEqual(predicted["style"], "dashed")
        self.assertTrue(all(predicted[key] is None for key in ("detection_index", "model_score", "model_id")))

    def test_no_guessed_identity_on_ambiguity(self):
        current = frame(1)
        result = summarize(current, self.tracker.update(current, [detection(current, 1), detection(current, 2)]))
        self.assertEqual(len(result["ambiguity_groups"]), 1)
        self.assertEqual([a["detection_index"] for a in result["annotations"]], [1, 2])
        self.assertTrue(all(a["identity"] is None and a["state"] == "ambiguous" for a in result["annotations"]))

    def test_owned_result_cannot_change_source_report(self):
        before = copy.deepcopy(self.report)
        result = summarize(self.frame, self.report)
        result["annotations"][0]["bbox"][0] = 99
        result["frame"]["coded_size"][0] = 1
        self.assertEqual(before, self.report)

    def test_stale_source_and_metadata_numeric_types(self):
        with self.assertRaises(ValueError):
            summarize(frame(1), self.report)
        for field, value in (("ordinal", False), ("coded_size", [96., 64])):
            report = copy.deepcopy(self.report)
            report["frame"][field] = value
            with self.assertRaises(ValueError):
                summarize(self.frame, report)

    def test_missing_malformed_fields_are_value_errors(self):
        for change in ({"detections": [None]}, {"tracks": [None]}, {"ambiguities": [None]}, {"segment": "one"}):
            with self.assertRaises(ValueError):
                summarize(self.frame, {**self.report, **change})
        report = copy.deepcopy(self.report)
        del report["tracks"][0]["last_observed_time"]
        with self.assertRaises(ValueError):
            summarize(self.frame, report)

    def test_forged_observation_score_alias_age_and_box(self):
        for change in ({"model_id": "other"}, {"model_score": True}, {"detection_index": 99},
                       {"identity_status": "verified"}, {"unobserved_seconds": "1/10"},
                       {"estimated_bbox": [-1., 0., 10., 10.]}):
            report = copy.deepcopy(self.report)
            report["tracks"][0].update(change)
            with self.assertRaises(ValueError):
                summarize(self.frame, report)

    def test_prediction_cannot_be_an_observation(self):
        current = frame(1)
        report = self.tracker.update(current, [])
        for field, value in (("model_score", .9), ("detection_index", 0), ("model_id", "scripted")):
            changed = copy.deepcopy(report)
            changed["tracks"][0][field] = value
            with self.assertRaises(ValueError):
                summarize(current, changed)


if __name__ == "__main__":
    unittest.main()
