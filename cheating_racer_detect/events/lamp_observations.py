"""External vehicle-crop lamp declarations, NOT a lamp detector or recognizer.

Only direct resize plus one explicit clockwise quarter-turn is supported. No
padding inference, image-side -> vehicle-side guess, clipping or pixel synthesis.
"""

from dataclasses import dataclass
from fractions import Fraction

from ..analysis.contracts import model_identifier
from ..tracking import Box, Frame
from ..tracking.contracts import identifier, integer
from .contracts import current, identity
from .lamp import LampEngine, LampROI


@dataclass(frozen=True)
class LampPart:
    side: str
    part_id: str
    box: Box | None
    visibility: str
    turn_signal_confirmed: bool

    def __post_init__(self):
        identifier(self.part_id)
        if (self.side not in ('left','right') or self.visibility not in
                ('visible','occluded','out_of_view','too_small','unknown')
                or type(self.turn_signal_confirmed) is not bool
                or self.box is not None and not isinstance(self.box,Box)):
            raise ValueError('Explicit vehicle-side lamp declarations required')


@dataclass(frozen=True)
class LampObservation:
    frame: Frame
    vehicle_identity: tuple[int,int,int]
    vehicle_box: Box
    model_id: str
    context_id: str
    crop: Box
    model_width: int
    model_height: int
    rotation_cw: int
    left: LampPart
    right: LampPart
    side_mapping_confirmed: bool
    rear_view: bool
    identity_confirmed: bool

    def __post_init__(self):
        current(self.frame,None)
        identity(self.vehicle_identity)
        model_identifier(self.model_id)
        identifier(self.context_id)
        integer(self.model_width,minimum=1)
        integer(self.model_height,minimum=1)
        integer(self.rotation_cw)
        if max(self.model_width,self.model_height)>1024 or self.rotation_cw not in (0,90,180,270):
            raise ValueError('Unsupported bounded crop preprocessing')
        if not isinstance(self.vehicle_box,Box) or not isinstance(self.crop,Box):
            raise ValueError('Actual box and explicit crop required')
        b,c,f = self.vehicle_box,self.crop,self.frame
        if not (0<=b.x1<b.x2<=f.width and 0<=b.y1<b.y2<=f.height
                and b.x1<=c.x1<c.x2<=b.x2 and b.y1<=c.y1<c.y2<=b.y2):
            raise ValueError('Crop outside declared actual vehicle')
        if any(type(v) is not bool for v in
               (self.side_mapping_confirmed,self.rear_view,self.identity_confirmed)):
            raise ValueError('Explicit semantic/identity gates required')
        if (not isinstance(self.left,LampPart) or not isinstance(self.right,LampPart)
                or self.left.side!='left' or self.right.side!='right'
                or self.left.part_id==self.right.part_id):
            raise ValueError('Distinct vehicle-relative left/right parts required')
        for part in (self.left,self.right):
            p=part.box
            if p is not None and not (0<=p.x1<p.x2<=self.model_width and 0<=p.y1<p.y2<=self.model_height):
                raise ValueError('Part outside explicit model canvas')

    def coded_box(self, part):
        if not isinstance(part,LampPart) or part not in (self.left,self.right):
            raise ValueError('Expected this observation\'s part')
        if part.box is None:
            return None
        b,r=part.box,self.rotation_cw
        # Preserve the exact supplied binary values until the final coordinate.
        # Float divide/multiply can turn an integer edge into n+1ulp, changing
        # ceil/floor whole-pixel membership. No epsilon, snapping or clipping.
        cx1,cy1,cx2,cy2=(Fraction(v) for v in self.crop.values())
        w,h=cx2-cx1,cy2-cy1
        vw,vh=(h,w) if r in (90,270) else (w,h)
        points=[]
        for x,y in ((b.x1,b.y1),(b.x1,b.y2),(b.x2,b.y1),(b.x2,b.y2)):
            u,v=Fraction(x)/self.model_width*vw,Fraction(y)/self.model_height*vh
            px,py = ((v,h-u) if r==90 else (w-u,h-v) if r==180 else
                     (w-v,u) if r==270 else (u,v))
            points.append((cx1+px,cy1+py))
        return Box(float(min(p[0] for p in points)),float(min(p[1] for p in points)),
                   float(max(p[0] for p in points)),float(max(p[1] for p in points)))


def resolve_lamp_rois(frame,vehicle,observation):
    """Exact same actual observation only; every semantic gate is caller-owned."""
    current(frame,vehicle)
    if observation is not None:
        if (not isinstance(observation,LampObservation) or observation.frame!=frame or vehicle is None
                or observation.vehicle_identity!=vehicle.identity or observation.vehicle_box!=vehicle.box):
            raise ValueError('Lamp observation differs from same-frame actual vehicle')
    if vehicle is None or observation is None:
        status='identity_uncertain' if vehicle is None else 'model_uncertain'
        return LampROI(None,status),LampROI(None,status)
    o=observation
    rois=[]
    for part in (o.left,o.right):
        box=o.coded_box(part)
        if not o.identity_confirmed:
            status='identity_uncertain'
        elif not o.side_mapping_confirmed or not o.rear_view or not part.turn_signal_confirmed or box is None:
            status='model_uncertain'
        else:
            status={'visible':'usable','unknown':'model_uncertain'}.get(part.visibility,part.visibility)
        rois.append(LampROI(box,status))
    return tuple(rois)


def lamp_source_key(observation):
    if observation is None:
        return None
    o=observation
    return (o.model_id,o.context_id,o.model_width,o.model_height,o.rotation_cw,
            o.left.part_id,o.right.part_id)


class LampObservationEngine(LampEngine):
    """LampEngine-compatible report with explicit source-aware update signature.

    update(frame, actual_vehicle, original_bgr_bytes, LampObservation|None).
    A side's uncertain visibility remains an uncertain sample, preserving the
    other side's partial facts. Missing observations/source changes reset both.
    """
    def reset(self):
        super().reset()
        self._source=None

    def update(self,frame,vehicle,pixels,observation):
        try:
            left,right=resolve_lamp_rois(frame,vehicle,observation)
            source=lamp_source_key(observation)
            changed=self._source is not None and source!=self._source
            reason='lamps_unknown' if source is None else 'lamp_source_change' if changed else None
            if source is None or changed:
                self.reset()
            result=LampEngine.update(self,frame,vehicle,pixels,left,right)
            # Base clock/identity guards can also reset us. Restore only after
            # success; exceptions/cancellation discard every history field.
            self._source=source
            return {'lamp':result,'reset_reason':reason,'left':left,'right':right}
        except BaseException:
            self.reset()
            raise


@dataclass(frozen=True)
class LampEvidence:
    observation: LampObservation | None

    def __post_init__(self):
        if self.observation is not None and not isinstance(self.observation,LampObservation):
            raise ValueError('Expected immutable lamp observation or explicit unknown')

    def rois(self,frame,vehicle):
        return resolve_lamp_rois(frame,vehicle,self.observation)

    def report(self):
        o=self.observation
        if o is None:
            return {'observation':None}
        def part(p):
            return {'side':p.side,'part_id':p.part_id,'box':None if p.box is None else list(p.box.values()),
                    'visibility':p.visibility,'turn_signal_confirmed':p.turn_signal_confirmed}
        return {'observation':{
            **{k:v for k,v in vars(o).items() if k not in ('frame','vehicle_identity','vehicle_box','crop','left','right')},
            'vehicle_identity':list(o.vehicle_identity),'vehicle_box':list(o.vehicle_box.values()),
            'crop':list(o.crop.values()),'left':part(o.left),'right':part(o.right)}}

    @classmethod
    def from_report(cls,frame,value):
        from .inputs import _fields,_array
        _fields(value,('observation',))
        o=value['observation']
        if o is None:
            return cls(None)
        _fields(o,tuple(k for k in LampObservation.__dataclass_fields__ if k!='frame'))
        def part(p):
            _fields(p,LampPart.__dataclass_fields__)
            return LampPart(**{**p,'box':None if p['box'] is None else Box(*_array(p['box'],4))})
        return cls(LampObservation(frame=frame,**{**o,'vehicle_identity':tuple(_array(o['vehicle_identity'],3)),
            'vehicle_box':Box(*_array(o['vehicle_box'],4)),'crop':Box(*_array(o['crop'],4)),
            'left':part(o['left']),'right':part(o['right'])}))
