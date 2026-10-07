"""Experimental detection-to-track API; importing it needs no ML packages."""

from .contracts import Box, Detection, Frame, MotionConfig, Policy
from .tracker import Tracker

__all__ = ["Box", "Detection", "Frame", "MotionConfig", "Policy", "Tracker"]
