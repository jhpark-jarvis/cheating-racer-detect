"""Actual optional detector binding/native tracking; no models or private imports."""

from dataclasses import replace
from fractions import Fraction
from importlib.util import find_spec
import unittest
from unittest.mock import patch

from cheating_racer_detect.analysis import DetectorBatch, DetectorObservation, DetectorTrackerBridge, Letterbox
from cheating_racer_detect.tracking import Box, Tracker
from tests.test_tracking import frame, policy

_NATIVE = find_spec("numpy") is not None and find_spec("cv2") is not None


class Scripted:
    model_id = "fixture:"+"a"*64

    def __init__(self):
        self.observations = (DetectorObservation(7, 2, Box(10, 10, 30, 30), .9, self.model_id),)
        self.calls = []
        self.output_frame = None
        self.geometry = None

    def detect(self, current, image):
        self.calls.append((current, image))
        return DetectorBatch(self.output_frame or current, self.observations,
                             self.geometry or Letterbox(current.width, current.height), self.model_id)


@unittest.skipUnless(_NATIVE, "NOT_RUN: optional analysis NumPy/OpenCV unavailable")
class Binding(unittest.TestCase):
    def setUp(self):
        self.detector = Scripted()
        self.tracker = Tracker(policy(max_unobserved=Fraction(3, 5)))
        self.bridge = DetectorTrackerBridge(self.detector, self.tracker, model_alias="fixture-v1",
                                            expected_detector_id=self.detector.model_id)

    def update(self, current=None):
        current = current or frame()
        return self.bridge.update(current, bytes((1, 2, 3))*current.width*current.height)

    def test_current_frame_readonly_bgr_alias_external_and_no_double_transform(self):
        current = frame(display_rotation=90)
        output = self.update(current)
        self.assertIs(self.detector.calls[0][0], current)
        image = self.detector.calls[0][1]
        self.assertEqual(image.shape, (64, 96, 3))
        self.assertFalse(image.flags.writeable)
        self.assertEqual(image[0, 0].tolist(), [1, 2, 3])
        self.assertEqual(output["tracking"]["frame"], current.report())
        self.assertEqual(output["tracking"]["detections"][0], {"index": 7, "class_id": 2,
                         "bbox": [10., 10., 30., 30.], "model_score": .9, "model_id": "fixture-v1"})
        self.assertEqual(output["model"]["external_identifier"], self.detector.model_id)

    def test_explicit_constructor_receipts(self):
        for alias, expected in (("a/path", self.detector.model_id), ("x"*65, self.detector.model_id),
                                ("alias", "wrong"), ("alias", "../private")):
            with self.assertRaises(ValueError):
                DetectorTrackerBridge(self.detector, self.tracker, model_alias=alias, expected_detector_id=expected)
        with self.assertRaises(ValueError):
            DetectorTrackerBridge(self.detector, object(), model_alias="alias", expected_detector_id=self.detector.model_id)

    def test_bad_input_before_detection_then_new_segment(self):
        old = self.update()["tracking"]["tracks"][0]["identity"]
        for value in (b"bad", bytearray(96*64*3), None):
            self.detector.calls.clear()
            with self.assertRaises(ValueError):
                self.bridge.update(frame(1), value)
            self.assertEqual(self.detector.calls, [])
        restored = self.update(frame(2))["tracking"]
        self.assertEqual(restored["reset_reason"], "failed_update")
        self.assertNotEqual(restored["tracks"][0]["identity"], old)
        for value in (object(), frame(width=1025)):
            with self.assertRaises(ValueError):
                self.bridge.update(value, b"")

    def test_stale_foreign_frame_is_rejected(self):
        self.update()
        for current in (frame(), frame(1, source_id="other"), frame(1, display_rotation=90)):
            self.detector.output_frame = current
            with self.assertRaises(ValueError):
                self.update(frame(1))
        self.detector.output_frame = None
        self.assertEqual(self.update(frame(2))["tracking"]["reset_reason"], "failed_update")

    def test_model_changes_before_and_during_detect(self):
        expected = self.detector.model_id
        self.detector.model_id = "changed"
        with self.assertRaises(ValueError):
            self.update()
        self.detector.model_id = expected
        original = self.detector.detect
        def change(current, image):
            batch = original(current, image)
            self.detector.model_id = "changed"
            return batch
        with patch.object(self.detector, "detect", side_effect=change), self.assertRaises(ValueError):
            self.update()
        self.detector.model_id = expected
        self.assertEqual(self.update()["tracking"]["reset_reason"], "failed_update")

    def test_geometry_target_and_invalid_batch_rejected(self):
        for geometry in (Letterbox(97, 64), Letterbox(96, 64, 320, 320)):
            self.detector.geometry = geometry
            with self.assertRaises(ValueError):
                self.update()
        with patch.object(self.detector, "detect", return_value=object()), self.assertRaises(ValueError):
            self.update()

    def test_policy_classes_and_detection_bound(self):
        item = self.detector.observations[0]
        self.detector.observations = (replace(item, class_id=0),)
        with self.assertRaises(ValueError):
            self.update()
        self.tracker = Tracker(policy(max_detections=1))
        self.bridge = DetectorTrackerBridge(self.detector, self.tracker, model_alias="alias", expected_detector_id=self.detector.model_id)
        self.detector.observations = (item, replace(item, index=8))
        with self.assertRaises(ValueError):
            self.update()

    def test_detector_error_cancel_and_native_failure_reset(self):
        old = self.update()["tracking"]["tracks"][0]["identity"]
        for error in (RuntimeError("injected"), KeyboardInterrupt()):
            with patch.object(self.detector, "detect", side_effect=error), self.assertRaises(type(error)):
                self.update(frame(1))
        restored = self.update(frame(2))["tracking"]
        self.assertNotEqual(restored["tracks"][0]["identity"], old)
        motion = next(iter(self.tracker._tracks.values())).motion
        with patch.object(motion, "predict", side_effect=RuntimeError("native error")), self.assertRaises(RuntimeError):
            self.update(frame(3))
        self.assertEqual(self.update(frame(4))["tracking"]["reset_reason"], "failed_update")

    def test_empty_predictions_expire_without_observation_fields(self):
        old = self.update()["tracking"]["tracks"][0]["identity"]
        self.detector.observations = ()
        prediction = self.update(frame(1))["tracking"]["tracks"][0]
        self.assertEqual(prediction["identity"], old)
        self.assertEqual(prediction["state"], "prediction_only")
        self.assertTrue(all(prediction[k] is None for k in ("detection_index", "detection_bbox", "model_score", "model_id")))
        for ordinal in range(2, 8):
            latest = self.update(frame(ordinal))["tracking"]
        self.assertEqual(latest["tracks"], [])

    def test_vfr_identity_clock_failure_and_geometry_boundaries(self):
        old = self.update()["tracking"]["tracks"][0]["identity"]
        output = self.update(frame(1, 504))["tracking"]
        self.assertEqual(output["delta_seconds"], "2/5")
        self.assertIsNone(output["reset_reason"])
        self.assertEqual(output["tracks"][0]["identity"], old)
        with self.assertRaises(ValueError):
            self.update(frame(2, 504))
        restored = self.update(frame(3, 505))["tracking"]
        self.assertEqual(restored["reset_reason"], "failed_update")
        for source, reason in ((frame(4, 506, source_id="other"), "source_or_geometry_change"),
                               (frame(5, 507, display_rotation=90), "source_or_geometry_change"),
                               (frame(7, 509, display_rotation=90), "missing_decoded_frame"),
                               (frame(8, 519, display_rotation=90), "time_gap")):
            self.assertEqual(self.update(source)["tracking"]["reset_reason"], reason)

    def test_duplicate_geometry_keeps_raw_but_ends_identity(self):
        self.update()
        item = self.detector.observations[0]
        self.detector.observations = (item, replace(item, index=8))
        output = self.update(frame(1))["tracking"]
        self.assertEqual(output["tracks"], [])
        self.assertEqual(len(output["ambiguities"]), 1)
        self.assertEqual(len(output["detections"]), 2)


if __name__ == "__main__":
    unittest.main()
