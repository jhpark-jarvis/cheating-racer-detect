"""Immutable original-frame event inputs, not calibrated product contracts."""

from dataclasses import dataclass
from fractions import Fraction

from ..review.records import summarize
from ..tracking.contracts import Box, Detection, Frame, finite, identifier, integer


def seconds(value, positive=False):
    if (not isinstance(value, Fraction) or value < 0 or (positive and not value)
            or max(value.numerator.bit_length(), value.denominator.bit_length()) > 64):
        raise ValueError('Expected bounded exact Fraction seconds')
    return value


def identity(value):
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError('Expected segment/class/track tuple')
    integer(value[0], minimum=1)
    integer(value[1])
    integer(value[2], minimum=1)
    if max(value) > 2**31-1:
        raise ValueError('Identity bound exceeded')
    return value


def current(frame, vehicle):
    if not isinstance(frame, Frame) or max(frame.width, frame.height) > 1024:
        raise ValueError('Expected bounded original Frame')
    if vehicle is not None and (not isinstance(vehicle, Vehicle) or vehicle.frame != frame):
        raise ValueError('Expected same-frame actual vehicle observation')


def boundary(previous, frame, max_gap):
    """A changed clock is invalid; other discontinuities start fresh histories."""
    if previous is None:
        return 'start'
    if frame.key != previous.key:
        return 'source_or_geometry_change'
    if (frame.time_base, frame.first_pts) != (previous.time_base, previous.first_pts):
        raise ValueError('Source clock changed')
    if frame.ordinal <= previous.ordinal or frame.time <= previous.time:
        raise ValueError('Duplicate/backward source observation')
    if frame.ordinal != previous.ordinal+1:
        return 'missing_frame'
    if frame.time-previous.time > max_gap:
        return 'time_gap'
    return None


@dataclass(frozen=True)
class Vehicle:
    frame: Frame
    identity: tuple[int, int, int]
    detection: Detection

    def __post_init__(self):
        identity(self.identity)
        if (not isinstance(self.detection, Detection) or self.detection.frame != self.frame
                or self.identity[1] != self.detection.class_id):
            raise ValueError('Vehicle needs same-frame actual Detection')
        current(self.frame, None)

    @property
    def box(self):
        return self.detection.box

    @classmethod
    def from_tracking(cls, frame, report, selected_identity):
        """Prediction, ambiguity, unassociated detections and missing IDs -> None."""
        selected_identity = identity(selected_identity)
        summary = summarize(frame, report)
        for item in summary['annotations']:
            if item['state'] == 'observed' and item['identity'] == list(selected_identity):
                return cls(frame, selected_identity, Detection(item['detection_index'], selected_identity[1],
                    Box(*item['bbox']), item['model_score'], item['model_id'], frame))
        return None

    def report(self):
        return {'frame': self.frame.report(), 'identity': list(self.identity),
                'identity_status': 'unverified', 'detection': self.detection.report()}


@dataclass(frozen=True)
class LaneSection:
    """Injected boundaries at the observed box bottom y, in coded coordinates.

    IDs must remain stable in a road segment. Caller supplies ego/scene/rear-view
    gates; this class does NOT discover lanes or infer ego motion/vehicle pose.
    """
    frame: Frame
    road_id: str
    boundary_ids: tuple[str, ...]
    x: tuple[float, ...]
    y: float
    ego: str
    scene: str
    rear_view: bool

    def __post_init__(self):
        current(self.frame, None)
        identifier(self.road_id)
        if (not isinstance(self.boundary_ids, tuple) or not isinstance(self.x, tuple)
                or not 3 <= len(self.x) <= 9 or len(self.x) != len(self.boundary_ids)
                or len(set(self.boundary_ids)) != len(self.boundary_ids)):
            raise ValueError('Expected 2..8 lanes with unique stable boundaries')
        for name in self.boundary_ids:
            identifier(name)
        for x in self.x:
            if not 0 <= finite(x) <= self.frame.width:
                raise ValueError('Lane outside coded frame')
        if any(b <= a for a, b in zip(self.x, self.x[1:])) or not 0 <= finite(self.y) <= self.frame.height:
            raise ValueError('Invalid lane cross-section')
        if self.ego not in ('stable', 'changing', 'unknown') or self.scene not in ('ordinary', 'merge', 'unknown'):
            raise ValueError('Explicit ego/scene gate required')
        if type(self.rear_view) is not bool:
            raise ValueError('Explicit rear-view gate required')


@dataclass(frozen=True)
class Candidate:
    """A completed adjacent transition, with sampled-time uncertainty brackets."""
    identity: tuple[int, int, int]
    road_id: str
    from_lane: tuple[str, str]
    to_lane: tuple[str, str]
    direction: str
    start: tuple[Frame, Frame]
    crossing: tuple[Frame, Frame]
    completion: tuple[Frame, Frame]
    config_id: str

    def __post_init__(self):
        identity(self.identity)
        identifier(self.road_id)
        identifier(self.config_id)
        for lane in (self.from_lane, self.to_lane):
            if not isinstance(lane, tuple) or len(lane) != 2 or lane[0] == lane[1]:
                raise ValueError('Expected boundary pair')
            for name in lane:
                identifier(name)
        if (self.direction not in ('left', 'right') or len(set(self.from_lane+self.to_lane)) != 3
                or (self.direction == 'right' and self.from_lane[1] != self.to_lane[0])
                or (self.direction == 'left' and self.to_lane[1] != self.from_lane[0])):
            raise ValueError('Expected adjacent rear-view lanes')
        for pair in (self.start, self.crossing, self.completion):
            if not isinstance(pair, tuple) or len(pair) != 2 or not all(isinstance(f, Frame) for f in pair):
                raise ValueError('Expected original Frame brackets')
        reference = self.crossing[0]
        for pair in (self.start, self.crossing, self.completion):
            for f in pair:
                current(f, None)
                if (f.key, f.time_base, f.first_pts) != (reference.key, reference.time_base, reference.first_pts):
                    raise ValueError('Candidate crosses source/clock/geometry')
            if pair[0].time > pair[1].time or pair[0].ordinal > pair[1].ordinal:
                raise ValueError('Backward bracket')
        if not self.start[0].time <= self.start[1].time <= self.crossing[1].time <= self.completion[0].time <= self.completion[1].time:
            raise ValueError('Invalid event chronology')

    def report(self):
        return {'schema': 'experimental-lane-candidate-v1', 'identity': list(self.identity),
                'identity_status': 'unverified', 'road_id': self.road_id,
                'from_lane': list(self.from_lane), 'to_lane': list(self.to_lane), 'direction': self.direction,
                'start': [f.report() for f in self.start], 'crossing': [f.report() for f in self.crossing],
                'completion': [f.report() for f in self.completion], 'config_id': self.config_id,
                'limits': ['injected lanes/ego/rear-view gates', 'bottom-center contact proxy',
                           'identity unverified', 'candidate is not a violation']}
