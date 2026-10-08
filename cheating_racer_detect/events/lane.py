"""One-vehicle bounded lane-relative hysteresis; thresholds are caller-owned."""

from dataclasses import dataclass
from fractions import Fraction
import bisect

from ..tracking.contracts import finite, identifier
from .contracts import Candidate, LaneSection, boundary, current, seconds


@dataclass(frozen=True)
class LaneConfig:
    config_id: str
    margin: float
    dwell: Fraction
    max_gap: Fraction
    min_lane_pixels: float

    def __post_init__(self):
        identifier(self.config_id)
        if not 0 < finite(self.margin) < .5 or finite(self.min_lane_pixels) <= 0:
            raise ValueError('Invalid lane policy')
        seconds(self.dwell, True)
        seconds(self.max_gap, True)


class LaneEngine:
    def __init__(self, config):
        if not isinstance(config, LaneConfig):
            raise ValueError('Explicit LaneConfig required')
        self.config = config
        self.reset()

    def reset(self):
        self._previous = self._key = self._base = self._stable = None
        self._pending = self._depart = self._cross = self._last_lane = None

    def update(self, frame, vehicle, lanes):
        try:
            return self._update(frame, vehicle, lanes)
        except BaseException:
            self.reset()
            raise

    def _unknown(self, reason):
        self.reset()
        return {'status': 'unknown', 'reason': reason, 'candidate': None}

    def _update(self, frame, vehicle, lanes):
        current(frame, vehicle)
        reason = boundary(self._previous, frame, self.config.max_gap)
        if vehicle is None:
            return self._unknown('no_actual_observation')
        if lanes is None:
            return self._unknown('lanes_unknown')
        if not isinstance(lanes, LaneSection) or lanes.frame != frame or lanes.y != vehicle.box.y2:
            raise ValueError('Expected same-frame lanes at observed contact y')
        if lanes.ego != 'stable' or lanes.scene != 'ordinary' or not lanes.rear_view:
            return self._unknown('ego_scene_or_pose_uncertain')
        key = vehicle.identity, lanes.road_id, lanes.boundary_ids
        if reason or key != self._key:
            self.reset()
        x = (vehicle.box.x1+vehicle.box.x2)/2
        if not lanes.x[0] < x < lanes.x[-1]:
            return self._unknown('outside_lane_section')
        lane = bisect.bisect_right(lanes.x, x)-1
        width = lanes.x[lane+1]-lanes.x[lane]
        if width < self.config.min_lane_pixels:
            return self._unknown('lane_too_small')
        fraction = (x-lanes.x[lane])/width
        interior = self.config.margin <= fraction <= 1-self.config.margin
        previous = self._previous
        self._previous, self._key = frame, key
        if self._base is not None and abs(lane-self._base) > 1:
            return self._unknown('non_adjacent_or_incomplete')
        if self._base is not None and (lane != self._base or not interior) and self._depart is None:
            self._depart = self._stable, frame
        if lane == self._base:
            self._cross = None
        if (self._base is not None and lane != self._base and self._cross is None
                and self._last_lane == self._base):
            self._cross = previous, frame
        self._last_lane = lane
        if not interior:
            self._pending = None
            return {'status': 'moving' if self._base is not None else 'warming', 'reason': reason, 'candidate': None}
        if lane == self._base:
            # An excursion returning to the established lane is not a change.
            self._stable, self._depart, self._cross, self._pending = frame, None, None, None
            return {'status': 'stable', 'reason': reason, 'candidate': None}
        if self._pending is None or self._pending[0] != lane:
            self._pending = lane, frame
        if frame.time-self._pending[1].time < self.config.dwell:
            return {'status': 'moving' if self._base is not None else 'warming', 'reason': reason, 'candidate': None}
        candidate = None
        if self._base is not None and self._depart is not None and self._cross is not None:
            candidate = Candidate(vehicle.identity, lanes.road_id,
                tuple(lanes.boundary_ids[self._base:self._base+2]), tuple(lanes.boundary_ids[lane:lane+2]),
                'right' if lane > self._base else 'left', self._depart, self._cross,
                (self._pending[1], frame), self.config.config_id)
        self._base, self._stable = lane, frame
        self._pending = self._depart = self._cross = None
        return {'status': 'candidate' if candidate else 'stable', 'reason': reason, 'candidate': candidate}
