"""Synchronous original-coordinate detector binding to the independent Tracker."""

from ..tracking import Detection, Frame, Tracker
from ..tracking.contracts import identifier
from .contracts import DetectorBatch, model_identifier


class DetectorTrackerBridge:
    """Single-owner detector.detect(frame, readonly_bgr) -> DetectorBatch.

    The caller supplies a stable opaque alias and a full external model receipt.
    No model is loaded here. Same-frame binding cannot certify an arbitrary
    detector's internal caching or accuracy. Any failed/cancelled update drops
    the tracker's old history, including failures before Tracker.update().
    """

    def __init__(self, detector, tracker, *, model_alias, expected_detector_id):
        identifier(model_alias)
        model_identifier(expected_detector_id)
        if (not isinstance(tracker, Tracker) or not callable(getattr(detector, "detect", None))
                or getattr(detector, "model_id", None) != expected_detector_id):
            raise ValueError("Expected explicit detector/tracker/model receipt")
        self.detector, self.tracker = detector, tracker
        self.model_alias, self.detector_id = model_alias, expected_detector_id

    def update(self, frame, pixels):
        try:
            if (not isinstance(frame, Frame) or max(frame.width, frame.height) > 1024
                    or type(pixels) is not bytes or len(pixels) != frame.width*frame.height*3):
                raise ValueError("Expected bounded Frame and immutable same-decode coded BGR bytes")
            if self.detector.model_id != self.detector_id:
                raise ValueError("Detector model receipt changed")
            try:
                import numpy as np
            except ModuleNotFoundError as error:
                if error.name != "numpy":
                    raise
                raise RuntimeError("Optional analysis dependencies missing; install the tracking extra") from error
            image = np.frombuffer(pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            batch = self.detector.detect(frame, image)
            if (self.detector.model_id != self.detector_id or not isinstance(batch, DetectorBatch)
                    or batch.frame != frame or batch.model_id != self.detector_id
                    or (batch.geometry.target_width, batch.geometry.target_height) != (640, 640)
                    or len(batch.observations) > self.tracker.policy.max_detections):
                raise ValueError("Detector output/model/geometry does not describe this invocation")
            detections = [Detection(item.index, item.class_id, item.box, item.score, self.model_alias, frame)
                          for item in batch.observations]
            # Already original-coordinate boxes. Never unletterbox them again.
            transform = batch.geometry.report()
            tracking = self.tracker.update(frame, detections)
            return {"tracking": tracking, "transform": transform,
                    "model": {"alias": self.model_alias, "external_identifier": self.detector_id},
                    "binding": "synchronous_current_decoded_frame"}
        except BaseException:
            self.tracker.reset("failed_update")
            raise
