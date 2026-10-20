"""Immutable, caller-declared lane geometry/context for deterministic replay.

No model loading, authenticity assertion or trusted manual reset markers.
The containing input row supplies the exact original Frame to both records.
"""

from dataclasses import dataclass

from ..analysis.lanes import DECODER_ID, LaneBatch, LaneSlot
from .sections import BoundaryBinding, LaneContext, section_for_vehicle


@dataclass(frozen=True)
class LaneEvidence:
    batch: LaneBatch | None
    context: LaneContext | None

    def __post_init__(self):
        if self.batch is not None and not isinstance(self.batch, LaneBatch):
            raise ValueError('Expected immutable lane batch')
        if self.context is not None and not isinstance(self.context, LaneContext):
            raise ValueError('Expected immutable lane context')
        if self.batch is not None and self.context is not None:
            if (self.batch.frame != self.context.frame
                    or self.batch.model_id != self.context.lane_model_id):
                raise ValueError('Lane evidence source mismatch')

    def section(self, frame, vehicle):
        return section_for_vehicle(frame, vehicle, self.batch, self.context)

    def report(self):
        batch, context = self.batch, self.context
        return {'decoder_id': DECODER_ID,
                'batch': None if batch is None else {
                    'model_id': batch.model_id, 'rotation_cw': batch.rotation_cw,
                    'slots': [{'slot': s.slot, 'axis': s.axis, 'exists': list(s.exists),
                               'points': [None if p is None else list(p) for p in s.points]}
                              for s in batch.slots]},
                'context': None if context is None else {
                    **{k: v for k, v in vars(context).items()
                       if k not in ('frame', 'vehicle_identity', 'bindings')},
                    'vehicle_identity': list(context.vehicle_identity),
                    'bindings': [{'slot': b.slot, 'boundary_id': b.boundary_id,
                                  'visible_segments': list(b.visible_segments)}
                                 for b in context.bindings]}}

    @classmethod
    def from_report(cls, frame, value):
        # Local import avoids a module cycle; shared strict JSON helpers stay in
        # inputs. Shapes are bounded BEFORE conversion, not after allocation.
        from .inputs import _array, _fields
        _fields(value, ('decoder_id', 'batch', 'context'))
        if value['decoder_id'] != DECODER_ID:
            raise ValueError('Unsupported lane decoder interpretation')
        batch, context = value['batch'], value['context']
        if batch is not None:
            _fields(batch, ('model_id', 'rotation_cw', 'slots'))
            slots = []
            for s in _array(batch['slots'], 4):
                _fields(s, ('slot', 'axis', 'exists', 'points'))
                count = 72 if type(s['slot']) is int and s['slot'] in (1, 2) else 81
                slots.append(LaneSlot(s['slot'], s['axis'], tuple(_array(s['exists'], count)),
                    tuple(None if p is None else tuple(_array(p, 2))
                          for p in _array(s['points'], count))))
            batch = LaneBatch(frame, batch['model_id'], batch['rotation_cw'], tuple(slots))
        if context is not None:
            _fields(context, tuple(k for k in LaneContext.__dataclass_fields__ if k != 'frame'))
            if type(context['bindings']) is not list or len(context['bindings']) > 4:
                raise ValueError('Expected bounded ordered bindings')
            bindings = []
            for b in context['bindings']:
                _fields(b, ('slot', 'boundary_id', 'visible_segments'))
                count = 71 if type(b['slot']) is int and b['slot'] in (1, 2) else 80
                bindings.append(BoundaryBinding(b['slot'], b['boundary_id'],
                                               tuple(_array(b['visible_segments'], count))))
            context = LaneContext(frame=frame, **{
                **context, 'vehicle_identity': tuple(_array(context['vehicle_identity'], 3)),
                'bindings': tuple(bindings)})
        return cls(batch, context)
