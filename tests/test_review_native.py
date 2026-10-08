import copy
from fractions import Fraction
import unittest

from importlib.util import find_spec

_NATIVE = find_spec("numpy") is not None and find_spec("cv2") is not None
if _NATIVE:
    import numpy as np

from cheating_racer_detect.tracking import Box, Detection, Frame, Tracker
from tests.test_tracking import policy


def fixture_policy():
    return policy(max_unobserved=Fraction(3, 5))
from cheating_racer_detect.review import render, summarize
from cheating_racer_detect.review.overlay import draw_box, pixel_edges
from cheating_racer_detect.review.records import COLORS


@unittest.skipUnless(_NATIVE, "NOT_RUN: optional review NumPy/OpenCV unavailable")
class RenderTests(unittest.TestCase):
    def setUp(self):
        self.frame = Frame("synthetic", 0, 0, 50, Fraction(1, 10), 50, 100, 80)
        self.pixels = bytes([25])*100*80*3
        self.tracker = Tracker(fixture_policy())
        self.detection = Detection(3, 2, Box(10, 15, 30, 35), .9, "scripted", self.frame)
        self.report = self.tracker.update(self.frame, [self.detection])

    def next(self, ordinal=1, pts=51):
        frame = Frame("synthetic", 0, ordinal, pts, Fraction(1, 10), 50, 100, 80)
        return frame, self.tracker.update(frame, [])

    def test_observed_uses_actual_not_filter_box(self):
        self.report["tracks"][0]["estimated_bbox"] = [50., 40., 70., 60.]
        image, summary = render(self.frame, self.pixels, self.report)
        item = summary["annotations"][0]
        self.assertEqual(item["bbox"], [10., 15., 30., 35.])
        self.assertEqual(item["detection_index"], 3)
        self.assertEqual(item["model_score"], .9)
        self.assertEqual(item["identity_status"], "unverified")
        scale = summary["layout"]["integer_scale"]
        y = summary["layout"]["raster_origin"][1]
        self.assertEqual(image[y+35*scale-1, 10*scale+15].tolist(), list(COLORS["observed"]))
        self.assertEqual(image[y+60*scale-1, 50*scale+15].tolist(), [25]*3)

    def test_prediction_null_provenance_and_dashed(self):
        frame, report = self.next()
        _, summary = render(frame, self.pixels, report)
        predicted = summary["annotations"][0]
        self.assertEqual(predicted["state"], "prediction_only")
        self.assertEqual(predicted["style"], "dashed")
        self.assertEqual(predicted["unobserved_seconds"], "1/10")
        for key in ("detection_index", "model_id", "model_score"):
            self.assertIsNone(predicted[key])
        canvas = np.zeros((40, 40, 3), dtype=np.uint8)
        draw_box(canvas, (1, 1, 30, 30), COLORS["prediction_only"], "dashed")
        self.assertEqual(canvas[1, 1].tolist(), list(COLORS["prediction_only"]))
        self.assertEqual(canvas[1, 8].tolist(), [0]*3)

    def test_prediction_without_visible_box_preserved(self):
        frame, report = self.next()
        report["tracks"][0]["estimated_bbox"] = None
        _, summary = render(frame, self.pixels, report)
        self.assertIsNone(summary["annotations"][0]["bbox"])
        self.assertEqual(summary["annotations"][0]["state"], "prediction_only")

    def test_low_unassociated_detection_is_not_hidden(self):
        tracker = Tracker(fixture_policy())
        low = Detection(7, 5, self.detection.box, .1, "scripted", self.frame)
        summary = summarize(self.frame, tracker.update(self.frame, [low]))
        self.assertEqual(len(summary["annotations"]), 1)
        self.assertEqual(summary["annotations"][0]["state"], "unassociated")
        self.assertIsNone(summary["annotations"][0]["identity"])
        self.assertEqual(summary["annotations"][0]["detection_index"], 7)

    def test_ambiguity_ends_identity_preserves_raw(self):
        frame = Frame("synthetic", 0, 1, 51, Fraction(1, 10), 50, 100, 80)
        values = [Detection(i, 2, self.detection.box, .9, "scripted", frame) for i in (4, 5)]
        report = self.tracker.update(frame, values)
        _, summary = render(frame, self.pixels, report)
        self.assertEqual(len(summary["ambiguity_groups"]), 1)
        self.assertEqual([a["detection_index"] for a in summary["annotations"]], [4, 5])
        self.assertTrue(all(a["state"] == "ambiguous" and a["identity"] is None
                            and a["style"] == "double" for a in summary["annotations"]))
        self.assertEqual(summary["layout"]["ledger_rows"], 2)
        summary["ambiguity_groups"][0]["identities"][0][2] = 99
        self.assertNotEqual(summary["ambiguity_groups"], report["ambiguities"])

    def test_expiry_removes_prediction_not_raw_context(self):
        for ordinal in range(1, 5):
            frame, report = self.next(ordinal, 50+ordinal*2)
        summary = summarize(frame, report)
        self.assertEqual(summary["annotations"], [])
        self.assertEqual(summary["frame"]["normalized_time"], "4/5")

    def test_exact_offset_vfr_time(self):
        frame, report = self.next(1, 54)
        summary = summarize(frame, report)
        self.assertEqual(summary["frame"]["pts"], 54)
        self.assertEqual(summary["frame"]["first_pts"], 50)
        self.assertEqual(summary["frame"]["normalized_time"], "2/5")
        self.assertEqual(summary["delta_seconds"], "2/5")

    def test_gap_reset_delta_can_exceed_motion_bound(self):
        frame, report = self.next(1, 60)
        summary = summarize(frame, report)
        self.assertEqual(summary["reset_reason"], "time_gap")
        self.assertEqual(summary["delta_seconds"], "1")

    def test_coded_rotation_metadata_not_applied(self):
        for rotation in (90, 180, 270):
            frame = Frame("rotation", 0, 0, 0, Fraction(1, 10), 0, 100, 80, rotation)
            pixels = np.zeros((80, 100, 3), dtype=np.uint8)
            pixels[50:, 70:] = [11, 22, 33]
            report = Tracker(fixture_policy()).update(frame, [])
            image, summary = render(frame, pixels.tobytes(), report)
            scale = summary["layout"]["integer_scale"]
            y = summary["layout"]["raster_origin"][1]
            self.assertEqual(image[y+70*scale, 90*scale].tolist(), [11, 22, 33])
            self.assertEqual(summary["frame"]["display_rotation"], rotation)
            self.assertEqual(summary["view"], "coded_raster_no_display_rotation")

    def test_half_open_right_bottom_and_fractional_edges(self):
        self.assertEqual(pixel_edges([0., 0., 10., 10.], 1), (0, 0, 9, 9))
        self.assertEqual(pixel_edges([1.2, 2.1, 4.1, 6.3], 4), (4, 8, 16, 25))
        canvas = np.zeros((11, 11, 3), dtype=np.uint8)
        draw_box(canvas, pixel_edges([0., 0., 10., 10.], 1), COLORS["observed"], "solid")
        self.assertFalse(canvas[10].any())
        self.assertFalse(canvas[:, 10].any())
        self.assertTrue(canvas[9, 9].any())

    def test_original_inputs_and_summary_ownership(self):
        original = copy.deepcopy(self.report)
        image, summary = render(self.frame, self.pixels, self.report)
        summary["frame"]["coded_size"][0] = 9
        summary["annotations"][0]["bbox"][0] = 99
        image[:] = 0
        self.assertEqual(self.report, original)
        self.assertEqual(self.pixels, bytes([25])*100*80*3)

    def test_bad_frame_pixels_report_and_bounds_rejected(self):
        for pixels in (b"short", bytearray(self.pixels), np.zeros((80, 100, 3), dtype=np.uint8)):
            with self.subTest(type=type(pixels)), self.assertRaises(ValueError):
                render(self.frame, pixels, self.report)
        for change in ({"schema": "other"}, {"frame": {}}, {"segment": True},
                       {"detections": self.report["detections"]*65}, {"tracks": []+self.report["tracks"]*129},
                       {"delta_seconds": "nan"}, {"delta_seconds": "0"}, {"reset_reason": "success"}):
            report = {**self.report, **change}
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                summarize(self.frame, report)
        report = copy.deepcopy(self.report)
        report["frame"]["ordinal"] = False
        with self.assertRaises(ValueError):
            summarize(self.frame, report)
        with self.assertRaises(ValueError):
            summarize(Frame("large", 0, 0, 0, Fraction(1), 0, 1025, 80), self.report)

    def test_forged_observation_or_age_rejected(self):
        for change in ({"identity_status": "verified"}, {"detection_index": 4},
                       {"detection_index": True}, {"detection_bbox": [1., 1., 4., 4.]},
                       {"model_score": float("nan")}, {"model_id": "other"},
                       {"identity": [1, 5, 1]}, {"unobserved_seconds": "1/10"},
                       {"last_observed_time": "1/10"}, {"estimated_bbox": [0., 0., 101., 80.]},
                       {"state": "off"}):
            report = copy.deepcopy(self.report)
            report["tracks"][0].update(change)
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                summarize(self.frame, report)

    def test_prediction_cannot_reuse_observation(self):
        frame, report = self.next()
        for key, value in (("detection_index", 3), ("detection_bbox", [10., 15., 30., 35.]),
                           ("model_score", .9), ("model_id", "scripted")):
            changed = copy.deepcopy(report)
            changed["tracks"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize(frame, changed)

    def test_duplicate_nonfinite_outside_raw_detection_rejected(self):
        for change in ({"bbox": [0., 0., float("inf"), 10.]}, {"bbox": [-1., 0., 10., 10.]},
                       {"model_score": True}, {"model_id": "C:/model"}, {"class_id": False}):
            report = copy.deepcopy(self.report)
            report["detections"][0].update(change)
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                summarize(self.frame, report)
        report = copy.deepcopy(self.report)
        report["detections"] *= 2
        with self.assertRaises(ValueError):
            summarize(self.frame, report)

    def test_ambiguous_active_or_foreign_reference_rejected(self):
        for indices, identities in (([3], []), ([999], []), ([3, 3], [])):
            report = copy.deepcopy(self.report)
            report["ambiguities"] = [{"detection_indices": indices, "identities": identities,
                                      "reason": "non_unique_candidate_component"}]
            with self.assertRaises(ValueError):
                summarize(self.frame, report)
        report = copy.deepcopy(self.report)
        report["tracks"] *= 2
        with self.assertRaises(ValueError):
            summarize(self.frame, report)

    def test_dense_ledger_and_invisible_prediction_keep_every_row(self):
        report = copy.deepcopy(self.report)
        report["tracks"] = []
        report["detections"] = [Detection(i, 2, Box(1, 1, 3, 3), .1, "scripted", self.frame).report()
                                for i in range(64)]
        image, summary = render(self.frame, self.pixels, report)
        self.assertEqual(summary["layout"]["ledger_rows"], 64)
        ledger_y = summary["layout"]["ledger_origin"][1]
        self.assertGreater(image.shape[0], ledger_y+36+63*18)
        frame, report = self.next()
        report["tracks"][0]["estimated_bbox"] = None
        _, summary = render(frame, self.pixels, report)
        self.assertEqual(summary["layout"]["ledger_rows"], 1)

    def test_invalid_and_noncanonical_fraction_rejected(self):
        for value in ("1/0", "01/10", "-1", "1.0", "9"*65):
            report = copy.deepcopy(self.report)
            report["tracks"][0]["last_observed_time"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize(self.frame, report)


if __name__ == "__main__":
    unittest.main()
