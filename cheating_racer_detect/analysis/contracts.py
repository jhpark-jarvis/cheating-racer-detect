"""Detector output contracts in original coded coordinates, without ML imports."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
import re

from ..tracking import Box, Frame
from ..tracking.contracts import finite, integer


def model_identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value):
        raise ValueError("Expected bounded external model identifier, not a path")


@dataclass(frozen=True)
class Letterbox:
    """Top-left content with right/bottom padding; use actual integer axis scales.

    This records the detector's resize, not an image preprocessing implementation.
    to_original() is for model-space boxes only, never already-original boxes.
    """

    width: int
    height: int
    target_width: int = 640
    target_height: int = 640

    def __post_init__(self):
        for size in (self.width, self.height, self.target_width, self.target_height):
            integer(size, 1)
            if size > 16384:
                raise ValueError("Letterbox exceeds bounded raster contract")
        if self.resized_width == 0 or self.resized_height == 0:
            raise ValueError("Integer resize collapses a source dimension")

    @property
    def ratio(self):
        return min(Fraction(self.target_width, self.width), Fraction(self.target_height, self.height))

    @property
    def resized_width(self):
        return int(self.width*self.ratio)

    @property
    def resized_height(self):
        return int(self.height*self.ratio)

    def to_original(self, box):
        if not isinstance(box, Box):
            raise ValueError("Expected a validated model-space Box")
        content = box.clip(self.resized_width, self.resized_height)
        if content is None:
            return None
        sx, sy = self.resized_width/self.width, self.resized_height/self.height
        return Box(content.x1/sx, content.y1/sy, content.x2/sx, content.y2/sy).clip(self.width, self.height)

    def report(self):
        return {"source_size": [self.width, self.height], "target_size": [self.target_width, self.target_height],
                "resized_size": [self.resized_width, self.resized_height], "padding_origin": [0, 0],
                "scale_x": str(Fraction(self.resized_width, self.width)),
                "scale_y": str(Fraction(self.resized_height, self.height))}


@dataclass(frozen=True)
class DetectorObservation:
    """A post-NMS observation already mapped to the original coded raster."""

    index: int
    class_id: int
    box: Box
    score: float
    model_id: str

    def __post_init__(self):
        integer(self.index)
        integer(self.class_id)
        model_identifier(self.model_id)
        object.__setattr__(self, "score", finite(self.score))
        if not isinstance(self.box, Box) or not 0 <= self.score <= 1:
            raise ValueError("Invalid detector observation")


@dataclass(frozen=True)
class DetectorBatch:
    """Immutable, explicitly same-frame output of one synchronous detector call."""

    frame: Frame
    observations: tuple[DetectorObservation, ...]
    geometry: Letterbox
    model_id: str

    def __post_init__(self):
        model_identifier(self.model_id)
        if (not isinstance(self.frame, Frame) or not isinstance(self.geometry, Letterbox)
                or (self.geometry.width, self.geometry.height) != (self.frame.width, self.frame.height)
                or type(self.observations) is not tuple or len(self.observations) > 64):
            raise ValueError("Expected bounded current-frame detector batch")
        indices = set()
        for item in self.observations:
            if (not isinstance(item, DetectorObservation) or item.model_id != self.model_id
                    or item.index in indices or item.box.clip(self.frame.width, self.frame.height) != item.box):
                raise ValueError("Foreign/duplicate/non-original observation")
            indices.add(item.index)


def post_nms_observations(rows, geometry, model_id, *, classes=(2, 5, 7)):
    """YOLOX-style Nx7: xyxy, objectness, class score, class ID.

    Preserves the original row index and score product, filters classes/padding,
    and performs exactly one model-space -> source-space conversion. Does NOT
    run inference or NMS. Default class IDs describe this format's COCO mapping.
    """
    model_identifier(model_id)
    if (not isinstance(geometry, Letterbox) or type(classes) is not tuple
            or not classes or len(classes) > 64 or len(set(classes)) != len(classes)):
        raise ValueError("Expected explicit geometry and bounded class mapping")
    for label in classes:
        integer(label)
    values = tuple(islice(iter(rows), 513))
    if len(values) > 512:
        raise ValueError("Post-NMS row bound exceeded")
    result = []
    for index, row in enumerate(values):
        columns = tuple(islice(iter(row), 8))
        if len(columns) != 7:
            raise ValueError("Expected post-NMS Nx7 rows")
        x1, y1, x2, y2, objectness, score, label = (finite(v) for v in columns)
        model_box = Box(x1, y1, x2, y2)
        if not (0 <= objectness <= 1 and 0 <= score <= 1 and 0 <= label <= 2**31-1 and label == int(label)):
            raise ValueError("Invalid post-NMS confidence/class")
        original = geometry.to_original(model_box)
        if int(label) in classes and original is not None:
            result.append(DetectorObservation(index, int(label), original, objectness*score, model_id))
    return tuple(result)
