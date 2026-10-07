"""Exact source time and continuous coded-raster boxes, independent of a detector."""

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Real
import re

MAX_DT = Fraction(3, 5)


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid integer contract")
    return value


def finite(value):
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("Expected finite real number")
    return float(value)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value):
        raise ValueError("Expected opaque identifier, not a path")


def exact_positive(value):
    if not isinstance(value, Fraction) or value <= 0:
        raise ValueError("Expected positive Fraction")


@dataclass(frozen=True)
class Box:
    """Continuous half-open xyxy edges; never inclusive-pixel +1 coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        for name in ("x1", "y1", "x2", "y2"):
            object.__setattr__(self, name, finite(getattr(self, name)))
        if (self.x2 <= self.x1 or self.y2 <= self.y1
                or not math.isfinite((self.x2 - self.x1) * (self.y2 - self.y1))):
            raise ValueError("Expected finite positive extent/area")

    def values(self):
        return self.x1, self.y1, self.x2, self.y2

    def clip(self, width, height):
        values = (max(0., min(width, self.x1)), max(0., min(height, self.y1)),
                  max(0., min(width, self.x2)), max(0., min(height, self.y2)))
        return Box(*values) if values[2] > values[0] and values[3] > values[1] else None


@dataclass(frozen=True)
class Frame:
    source_id: str
    stream_index: int
    ordinal: int
    pts: int
    time_base: Fraction
    first_pts: int
    width: int
    height: int
    display_rotation: int = 0

    def __post_init__(self):
        identifier(self.source_id)
        integer(self.stream_index)
        integer(self.ordinal)
        for value in (self.pts, self.first_pts):
            if type(value) is not int or abs(value) >= 2**52:
                raise ValueError("PTS exceeds experiment precision bound")
        exact_positive(self.time_base)
        for size in (self.width, self.height):
            integer(size, 1)
            if size > 16384:
                raise ValueError("Raster exceeds experiment bound")
        integer(self.display_rotation)
        if self.pts < self.first_pts or self.display_rotation not in (0, 90, 180, 270):
            raise ValueError("Invalid source origin/orientation")

    @property
    def time(self):
        return (self.pts - self.first_pts) * self.time_base

    @property
    def key(self):
        return self.source_id, self.stream_index, self.width, self.height, self.display_rotation

    def report(self):
        return {"source_id": self.source_id, "stream_index": self.stream_index, "ordinal": self.ordinal,
                "pts": self.pts, "time_base": str(self.time_base), "first_pts": self.first_pts,
                "normalized_time": str(self.time), "coded_size": [self.width, self.height],
                "display_rotation": self.display_rotation, "coordinate_space": "coded_raster_half_open_xyxy"}


@dataclass(frozen=True)
class Detection:
    index: int
    class_id: int
    box: Box
    score: float
    model_id: str
    frame: Frame

    def __post_init__(self):
        integer(self.index)
        integer(self.class_id)
        identifier(self.model_id)
        object.__setattr__(self, "score", finite(self.score))
        if not 0 <= self.score <= 1 or not isinstance(self.frame, Frame) or not isinstance(self.box, Box):
            raise ValueError("Invalid detection contract")
        if self.box.clip(self.frame.width, self.frame.height) != self.box:
            raise ValueError("Detection must be in the original coded raster")

    def report(self):
        return {"index": self.index, "class_id": self.class_id, "bbox": list(self.box.values()),
                "model_score": self.score, "model_id": self.model_id}


@dataclass(frozen=True)
class MotionConfig:
    spectral: tuple[float, ...]
    measurement_variance: tuple[float, ...]
    initial_variance: tuple[float, ...]

    def __post_init__(self):
        for name, size, strictly_positive in (("spectral", 4, False), ("measurement_variance", 4, True),
                                              ("initial_variance", 8, True)):
            values = tuple(finite(v) for v in getattr(self, name))
            if len(values) != size or any(v < 0 or (strictly_positive and v == 0) for v in values):
                raise ValueError("Invalid diagonal noise/uncertainty configuration")
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class Policy:
    """All numerical gates must be supplied; there are no calibrated vehicle defaults."""

    max_gap: Fraction
    max_unobserved: Fraction
    min_score: float
    new_score: float
    min_iou: float
    duplicate_iou: float
    classes: tuple[int, ...]
    motion: MotionConfig
    max_detections: int = 64
    max_tracks: int = 128

    def __post_init__(self):
        exact_positive(self.max_gap)
        exact_positive(self.max_unobserved)
        if self.max_gap > MAX_DT:
            raise ValueError("Motion gap exceeds validated experiment bound")
        for name in ("min_score", "new_score", "min_iou", "duplicate_iou"):
            object.__setattr__(self, name, finite(getattr(self, name)))
        if not (0 <= self.min_score <= self.new_score <= 1 and 0 < self.min_iou <= self.duplicate_iou <= 1):
            raise ValueError("Invalid score/overlap policy")
        if (not isinstance(self.classes, tuple) or not self.classes or len(set(self.classes)) != len(self.classes)
                or not isinstance(self.motion, MotionConfig)):
            raise ValueError("Invalid class/motion policy")
        for label in self.classes:
            integer(label)
        integer(self.max_detections, 1)
        integer(self.max_tracks, 1)
        if self.max_detections > 64 or self.max_tracks > 128:
            raise ValueError("Policy exceeds bounded prototype resources")
