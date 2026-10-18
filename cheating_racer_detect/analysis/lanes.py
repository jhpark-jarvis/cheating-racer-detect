"""Bounded UFLDv2 CULane output postprocessing; no inference or ML imports.

Independent implementation of local ordinal expectation/anchor geometry, with
algorithm reference: cfzd/Ultra-Fast-Lane-Detection-v2, revision
c903880678454dfd9b55a63022368db05c00bc6d, demo.py pred2coords/configs/culane_res18.py.
No upstream modules, weights, config execution, training helpers or IO included.
Unlike the rendering demo: continuous coordinates, invalid/outside samples kept
as None, four explicit slots retained, and source Frame/clock preserved.
"""

from dataclasses import dataclass

from .contracts import model_identifier
from ..tracking import Frame
from ..tracking.contracts import finite, integer

DECODER_ID = 'ufld-culane-18-local1-v1'
_LAYOUT = ((1, 'row', 72), (2, 'row', 72), (0, 'column', 81), (3, 'column', 81))
_SHAPES = {'loc_row': (1, 200, 72, 4), 'loc_col': (1, 100, 81, 4),
           'exist_row': (1, 2, 72, 4), 'exist_col': (1, 2, 81, 4)}


def _frame(value):
    if not isinstance(value, Frame) or max(value.width, value.height) > 1024:
        raise ValueError('Expected bounded original Frame')


def _rotation(value):
    integer(value)
    if value not in (0, 90, 180, 270):
        raise ValueError('Expected applied clockwise quarter-turn')


def to_coded_point(frame, rotation_cw, point):
    """Invert ONE actual preprocessing rotation, in continuous edge coordinates.

    point is already decoded to unresized model-view W/H, NOT 1600x320 crop
    pixels. display_rotation metadata is deliberately not consulted. No clipping,
    crop inverse, pixel-index W-1 adjustment or inferred orientation is applied.
    """
    _frame(frame)
    _rotation(rotation_cw)
    if type(point) is not tuple or len(point) != 2:
        raise ValueError('Expected a continuous view point')
    u, v = (finite(value) for value in point)
    width, height = ((frame.height, frame.width) if rotation_cw in (90, 270)
                     else (frame.width, frame.height))
    if not 0 <= u <= width or not 0 <= v <= height:
        raise ValueError('Point outside unresized model view')
    if rotation_cw == 90:
        return v, frame.height-u
    if rotation_cw == 180:
        return frame.width-u, frame.height-v
    if rotation_cw == 270:
        return frame.width-v, u
    return u, v


@dataclass(frozen=True)
class UFLDOutput:
    """Caller-declared synchronous output, not a verified model/clock receipt.

    rotation_cw is the actual coded->model-view preprocessing quarter-turn, NOT
    the source display rotation. logits must be the four CPU NumPy arrays, not
    Torch/GPU tensors. They are copied/validated by decode_ufld; caller must not
    mutate them concurrently. No hidden tensor conversion/device transfer.
    """
    frame: Frame
    model_id: str
    rotation_cw: int
    logits: dict

    def __post_init__(self):
        _frame(self.frame)
        model_identifier(self.model_id)
        _rotation(self.rotation_cw)
        if type(self.logits) is not dict or set(self.logits) != set(_SHAPES):
            raise ValueError('Expected exactly four CULane logits arrays')


@dataclass(frozen=True)
class LaneSlot:
    """Model slot, NOT physical boundary ID or proof of visible road paint.

    exists preserves raw binary argmax. support_passed is only the original
    format's anchor-count gate. Missing/outside samples stay None, never joined
    or extrapolated; neither gate nor point is calibrated semantic observability.
    row/column refers to the input model view even after coded rotation inverse.
    """
    slot: int
    axis: str
    exists: tuple[bool, ...]
    points: tuple[tuple[float, float] | None, ...]

    def __post_init__(self):
        integer(self.slot)
        if self.slot not in (0, 1, 2, 3):
            raise ValueError('Invalid lane slot')
        axis, count = ('row', 72) if self.slot in (1, 2) else ('column', 81)
        if (self.axis != axis or type(self.exists) is not tuple or len(self.exists) != count
                or any(type(v) is not bool for v in self.exists)
                or type(self.points) is not tuple or len(self.points) != count):
            raise ValueError('Invalid slot axis/mask/sample count')
        normalized = []
        for exists, point in zip(self.exists, self.points):
            if point is None:
                normalized.append(None)
            elif (not exists or not self.support_passed or type(point) is not tuple or len(point) != 2):
                raise ValueError('Point lacks slot support')
            else:
                normalized.append(tuple(finite(value) for value in point))
        object.__setattr__(self, 'points', tuple(normalized))

    @property
    def support_passed(self):
        # Strict > half/quarter is a source-format rule, NOT a product threshold.
        return sum(self.exists) > len(self.exists)/(2 if self.axis == 'row' else 4)


@dataclass(frozen=True)
class LaneBatch:
    frame: Frame
    model_id: str
    rotation_cw: int
    slots: tuple[LaneSlot, ...]

    def __post_init__(self):
        _frame(self.frame)
        model_identifier(self.model_id)
        _rotation(self.rotation_cw)
        if (type(self.slots) is not tuple or len(self.slots) != 4
                or any(not isinstance(slot, LaneSlot) for slot in self.slots)
                or tuple(slot.slot for slot in self.slots) != (1, 2, 0, 3)):
            raise ValueError('Expected all four explicit slots in format order')
        for slot in self.slots:
            for point in slot.points:
                if point is not None and not (0 <= point[0] <= self.frame.width
                                              and 0 <= point[1] <= self.frame.height):
                    raise ValueError('Lane point outside coded frame')

    def report(self):
        return {'frame': self.frame.report(), 'coordinate_space': 'coded_raster_continuous_edges',
                'model_id': self.model_id, 'model_id_status': 'caller_declared_unverified',
                'decoder_id': DECODER_ID, 'applied_rotation_cw': self.rotation_cw,
                'slot_id_status': 'model_slot_not_physical_boundary',
                'slots': [{'slot': slot.slot, 'axis': slot.axis,
                           'exists': list(slot.exists), 'support_passed': slot.support_passed,
                           'points': [None if p is None else list(p) for p in slot.points]}
                          for slot in self.slots]}


def decode_ufld(current, output):
    """Fixed CULane18 batch1 float32/64 -> immutable source-coded LaneBatch.

    NumPy is lazy/optional; no Torch/OpenCV import. Exact shape and finite checks
    precede bounded float64 snapshots. Already-original anchor ratios are scaled
    once to the model view, then at most one rotation inverse. No stable IDs,
    vehicle-bottom section, ego/scene/pose inference or LaneSection creation.
    Stateless: failure/cancellation returns no partial output; retry is fresh.
    """
    _frame(current)
    if not isinstance(output, UFLDOutput) or output.frame != current:
        raise ValueError('Expected exactly same source Frame and clock')
    # Revalidate shallow input metadata before numerical access.
    output.__post_init__()
    import numpy as np

    arrays = {}
    for name, shape in _SHAPES.items():
        value = output.logits[name]
        if (type(value) is not np.ndarray or value.shape != shape
                or value.dtype not in (np.dtype('float32'), np.dtype('float64'))):
            raise ValueError('Expected exact float32/64 CULane array layout')
        snapshot = value.astype(np.float64, copy=True)
        if not bool(np.isfinite(snapshot).all()):
            raise ValueError('Nonfinite lane logits')
        arrays[name] = snapshot
    width, height = ((current.height, current.width) if output.rotation_cw in (90, 270)
                     else (current.width, current.height))
    slots = []
    for slot, axis, count in _LAYOUT:
        suffix = 'row' if axis == 'row' else 'col'
        logits = arrays['loc_'+suffix][0, :, :, slot]
        existence = arrays['exist_'+suffix][0, :, :, slot].argmax(axis=0)
        exists = tuple(bool(value) for value in existence)
        supported = sum(exists) > count/(2 if axis == 'row' else 4)
        peaks = logits.argmax(axis=0)
        grid_count = logits.shape[0]
        points = []
        for anchor_index, valid in enumerate(exists):
            point = None
            if supported and valid:
                peak = int(peaks[anchor_index])
                indices = np.arange(max(0, peak-1), min(grid_count, peak+2))
                local = logits[indices, anchor_index]
                with np.errstate(over='ignore', under='ignore'):
                    weights = np.exp(local-local.max())
                ratio = (float(weights @ indices/weights.sum())+.5)/(grid_count-1)
                if axis == 'row':
                    u, v = ratio*width, (.42+.58*anchor_index/(count-1))*height
                else:
                    u, v = anchor_index/(count-1)*width, ratio*height
                # +.5 at the final grid may exceed 1: never invent edge evidence.
                if 0 <= u <= width and 0 <= v <= height:
                    point = to_coded_point(current, output.rotation_cw, (u, v))
            points.append(point)
        slots.append(LaneSlot(slot, axis, exists, tuple(points)))
    return LaneBatch(current, output.model_id, output.rotation_cw, tuple(slots))
