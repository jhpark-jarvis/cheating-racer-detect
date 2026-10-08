"""Experimental detector/tracker contracts; no model loading or implicit downloads."""

from .bridge import DetectorTrackerBridge
from .contracts import DetectorBatch, DetectorObservation, Letterbox, post_nms_observations

__all__ = ["DetectorTrackerBridge", "DetectorBatch", "DetectorObservation", "Letterbox", "post_nms_observations"]
