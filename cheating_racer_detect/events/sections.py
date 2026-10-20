"""Caller-certified adjacent lane segments -> observed-box cross-section.

No visibility/ID/pose model, interpolation over gaps, extrapolation, implicit
sorting or default semantic gates. Caller declarations are NOT authenticated
evidence. Geometry uses the existing continuous coded coordinate convention.
"""

from dataclasses import dataclass

from ..analysis.contracts import model_identifier
from ..analysis.lanes import LaneBatch
from ..tracking import Frame
from ..tracking.contracts import identifier, integer
from .contracts import LaneSection, Vehicle, current, identity
from .lane import LaneEngine


@dataclass(frozen=True)
class BoundaryBinding:
    """An external physical ID and whole-segment visibility assertions.

    Each flag covers exactly adjacent decoded anchors i,i+1, NOT just their
    endpoints. True does not repair missing samples. There are 71 row or 80
    column segments. These assertions must come from outside the decoder.
    """
    slot: int
    boundary_id: str
    visible_segments: tuple[bool, ...]

    def __post_init__(self):
        integer(self.slot)
        identifier(self.boundary_id)
        if self.slot not in (0, 1, 2, 3):
            raise ValueError('Invalid model slot binding')
        count = 71 if self.slot in (1, 2) else 80
        if (type(self.visible_segments) is not tuple or len(self.visible_segments) != count
                or any(type(value) is not bool for value in self.visible_segments)):
            raise ValueError('Explicit adjacent-segment visibility mask required')


@dataclass(frozen=True)
class LaneContext:
    """Same-frame/vehicle/model external semantic declarations.

    bindings are declared consecutive physical boundaries, LEFT TO RIGHT in
    coded coordinates. No automatic slot->ID association or x sorting. Stable
    context_id identifies the interpretation/policy across frames, not each
    timestamp. coded_axes_confirmed asserts coded left/right and box.y2 are
    appropriate for this scene; display rotation does not prove this.
    """
    frame: Frame
    vehicle_identity: tuple[int, int, int]
    lane_model_id: str
    context_id: str
    road_id: str
    bindings: tuple[BoundaryBinding, ...]
    adjacency_confirmed: bool
    coded_axes_confirmed: bool
    ego: str
    scene: str
    rear_view: bool

    def __post_init__(self):
        current(self.frame, None)
        identity(self.vehicle_identity)
        model_identifier(self.lane_model_id)
        identifier(self.context_id)
        identifier(self.road_id)
        if (type(self.bindings) is not tuple or len(self.bindings) > 4
                or any(not isinstance(item, BoundaryBinding) for item in self.bindings)
                or len({item.slot for item in self.bindings}) != len(self.bindings)
                or len({item.boundary_id for item in self.bindings}) != len(self.bindings)):
            raise ValueError('Expected unique ordered boundary bindings')
        if (any(type(value) is not bool for value in
                (self.adjacency_confirmed, self.coded_axes_confirmed, self.rear_view))
                or self.ego not in ('stable', 'changing', 'unknown')
                or self.scene not in ('ordinary', 'merge', 'unknown')):
            raise ValueError('Explicit semantic gates required')


@dataclass(frozen=True)
class BoundarySample:
    slot: int
    boundary_id: str
    x: float
    y: float
    anchors: tuple[int, int]
    weight: float

    def report(self):
        return {'slot': self.slot, 'boundary_id': self.boundary_id, 'x': self.x,
                'y': self.y, 'anchors': list(self.anchors), 'weight': self.weight}


@dataclass(frozen=True)
class LaneSectionResult:
    frame: Frame
    vehicle: Vehicle | None
    model_id: str | None
    context: LaneContext | None
    samples: tuple[BoundarySample, ...]
    section: LaneSection | None
    reason: str | None
    failure_slot: int | None

    def report(self):
        context, section = self.context, self.section
        return {'frame': self.frame.report(), 'status': 'ready' if section else 'unknown',
                'reason': self.reason, 'failure_slot': self.failure_slot,
                'vehicle_identity': None if self.vehicle is None else list(self.vehicle.identity),
                'lane_model_id': self.model_id,
                'context_id': None if context is None else context.context_id,
                'declaration_status': 'caller_asserted_unverified',
                'gates': None if context is None else {
                    'adjacency_confirmed': context.adjacency_confirmed,
                    'coded_axes_confirmed': context.coded_axes_confirmed,
                    'ego': context.ego, 'scene': context.scene, 'rear_view': context.rear_view},
                'samples': [sample.report() for sample in self.samples],
                'section': None if section is None else {
                    'road_id': section.road_id, 'boundary_ids': list(section.boundary_ids),
                    'x': list(section.x), 'y': section.y, 'ego': section.ego,
                    'scene': section.scene, 'rear_view': section.rear_view}}


def _sample(slot, binding, y):
    if not slot.support_passed:
        return None, 'slot_support_missing'
    crossings = {}
    for index, (left, right) in enumerate(zip(slot.points, slot.points[1:])):
        if left is None or right is None:
            continue  # Never connect across even one missing anchor.
        if not min(left[1], right[1]) <= y <= max(left[1], right[1]):
            continue
        if left[1] == right[1]:
            return None, 'horizontal_intersection'
        if y == left[1]:
            key, x, weight = ('anchor', index), left[0], 0.
        elif y == right[1]:
            key, x, weight = ('anchor', index+1), right[0], 1.
        else:
            weight = (y-left[1])/(right[1]-left[1])
            key, x = ('segment', index), left[0]+weight*(right[0]-left[0])
        crossings.setdefault(key, []).append((x, index, weight, binding.visible_segments[index]))
    if not crossings:
        return None, 'no_adjacent_support'
    if len(crossings) != 1:
        return None, 'multiple_intersections'
    # Shared vertex counts once; different anchors at identical coordinates do NOT.
    alternatives = next(iter(crossings.values()))
    certified = next((value for value in alternatives if value[3]), None)
    if certified is None:
        return None, 'uncertified_intersection'
    x, index, weight, _visible = certified
    return BoundarySample(slot.slot, binding.boundary_id, x, y, (index, index+1), weight), None


def section_for_vehicle(frame, vehicle, batch, context):
    """Stateless geometry/gate check, NOT discovery or a history-aware adapter.

    All selected boundaries are required; no dropping failures to invent an
    adjacent pair. Unknown returns section=None (samples may be diagnostics).
    Malformed/foreign declarations raise ValueError. No NumPy, pixels or IO.
    Independent calls must not be used to hide model/context source changes
    from a long-lived LaneEngine; use LaneSectionBridge for live accumulation.
    Use explicit LaneEvidence/InputFrame v2 for persisted provenance; legacy v1
    injected sections do not carry this evidence.
    """
    current(frame, vehicle)
    if batch is not None and (not isinstance(batch, LaneBatch) or batch.frame != frame):
        raise ValueError('Expected same-frame lane batch')
    if context is not None:
        if not isinstance(context, LaneContext) or context.frame != frame:
            raise ValueError('Expected same-frame external lane context')
        if vehicle is not None and context.vehicle_identity != vehicle.identity:
            raise ValueError('Lane context belongs to another vehicle')
        if batch is not None and context.lane_model_id != batch.model_id:
            raise ValueError('Lane model/context mismatch')

    def result(reason, samples=(), section=None, failure_slot=None):
        return LaneSectionResult(frame, vehicle, None if batch is None else batch.model_id,
                                 context, samples, section, reason, failure_slot)

    if vehicle is None:
        return result('no_actual_observation')
    if batch is None:
        return result('lanes_unknown')
    if context is None:
        return result('context_unknown')
    if not context.adjacency_confirmed:
        return result('boundary_bindings_unconfirmed')
    if not context.coded_axes_confirmed:
        return result('coded_axes_uncertain')
    if context.ego != 'stable' or context.scene != 'ordinary' or not context.rear_view:
        return result('ego_scene_or_pose_uncertain')
    if len(context.bindings) < 3:
        return result('insufficient_boundaries')
    slots, samples = {slot.slot: slot for slot in batch.slots}, []
    for binding in context.bindings:
        sample, reason = _sample(slots[binding.slot], binding, vehicle.box.y2)
        if reason:
            return result(reason, tuple(samples), failure_slot=binding.slot)
        samples.append(sample)
    if any(right.x <= left.x for left, right in zip(samples, samples[1:])):
        return result('boundary_order_uncertain', tuple(samples))
    section = LaneSection(frame, context.road_id,
                          tuple(item.boundary_id for item in samples),
                          tuple(item.x for item in samples), vehicle.box.y2,
                          context.ego, context.scene, context.rear_view)
    return result(None, tuple(samples), section)


class LaneSectionBridge:
    """Owns a fresh LaneEngine, resetting on lane-source uncertainty/change.

    Persistence uses explicit LaneEvidence/InputFrame v2. Numerical lane-event
    policy remains explicit caller-owned LaneConfig, not calibrated.
    """
    def __init__(self, config):
        self._engine = LaneEngine(config)
        self._source = None

    def reset(self):
        self._engine.reset()
        self._source = None

    def update(self, frame, vehicle, batch, context):
        try:
            result = section_for_vehicle(frame, vehicle, batch, context)
            source = None if result.section is None else (
                batch.model_id, batch.rotation_cw, context.context_id, context.road_id,
                tuple((binding.slot, binding.boundary_id) for binding in context.bindings))
            changed = self._source is not None and source != self._source
            reason = result.reason if source is None else ('lane_source_change' if changed else None)
            if source is None or changed:
                self.reset()
            self._source = source
            lane = self._engine.update(frame, vehicle, result.section)
            return {'section': result, 'lane': lane, 'reset_reason': reason}
        except BaseException:
            self.reset()
            raise
