"""Bounded experimental input receipts; caller annotations are not ground truth."""

from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import os

from ..errors import AppError
from ..paths import local_path
from ..service import _identity, _publish
from ..tracking import Box, Detection, Frame
from .contracts import LaneSection, Vehicle, current, identity, seconds
from .lane import LaneConfig
from .lamp import LampConfig, LampROI

MAX_BYTES = 1_048_576
MAX_ROWS = 120
SCHEMA = 'experimental-event-inputs-v1'


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _fields(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError('Unexpected input receipt fields')
    return value


def _exact(value):
    if type(value) is not str or len(value) > 64:
        raise ValueError('Expected bounded canonical Fraction')
    result = Fraction(value)
    seconds(result)
    if str(result) != value:
        raise ValueError('Expected canonical Fraction')
    return result


def _array(value, length):
    if type(value) is not list or len(value) != length:
        raise ValueError('Invalid receipt array')
    return value


def _frame(value):
    _fields(value, Frame('0'*64, 0, 0, 0, Fraction(1), 0, 1, 1).report())
    width, height = _array(value['coded_size'], 2)
    result = Frame(value['source_id'], value['stream_index'], value['ordinal'], value['pts'],
                   _exact(value['time_base']), value['first_pts'], width, height, value['display_rotation'])
    if _json(result.report()) != _json(value):
        raise ValueError('Inconsistent frame receipt')
    return result


def _config(value, cls, durations):
    _fields(value, cls.__dataclass_fields__)
    return cls(**{key: _exact(item) if key in durations else item for key, item in value.items()})


def _config_report(config):
    return {key: str(value) if isinstance(value, Fraction) else value for key, value in vars(config).items()}


def _roi(value):
    _fields(value, ('box', 'status'))
    return LampROI(None if value['box'] is None else Box(*_array(value['box'], 4)), value['status'])


@dataclass(frozen=True)
class InputFrame:
    frame: Frame
    vehicle: Vehicle | None
    lanes: LaneSection | None
    left: LampROI
    right: LampROI

    def __post_init__(self):
        current(self.frame, self.vehicle)
        if self.lanes is not None:
            if (not isinstance(self.lanes, LaneSection) or self.lanes.frame != self.frame
                    or (self.vehicle is not None and self.lanes.y != self.vehicle.box.y2)):
                raise ValueError('Expected same-frame lane section')
        for roi in (self.left, self.right):
            if not isinstance(roi, LampROI):
                raise ValueError('Explicit left/right ROI gates required')
            if roi.box is not None:
                b = roi.box
                v = self.vehicle.box if self.vehicle else Box(0, 0, self.frame.width, self.frame.height)
                if not (v.x1 <= b.x1 < b.x2 <= v.x2 and v.y1 <= b.y1 < b.y2 <= v.y2):
                    raise ValueError('ROI outside actual vehicle or coded frame')

    def report(self):
        lane = None if self.lanes is None else {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in vars(self.lanes).items() if key != 'frame'}
        return {'frame': self.frame.report(), 'vehicle': None if self.vehicle is None else self.vehicle.report(),
                'lanes': lane, **{side: {'box': None if roi.box is None else list(roi.box.values()),
                                       'status': roi.status} for side, roi in (('left', self.left), ('right', self.right))}}

    @classmethod
    def from_report(cls, value):
        _fields(value, ('frame', 'vehicle', 'lanes', 'left', 'right'))
        frame, vehicle = _frame(value['frame']), None
        if value['vehicle'] is not None:
            v = _fields(value['vehicle'], ('frame', 'identity', 'identity_status', 'detection'))
            d = _fields(v['detection'], ('index', 'class_id', 'bbox', 'model_score', 'model_id'))
            vehicle = Vehicle(frame, tuple(_array(v['identity'], 3)), Detection(d['index'], d['class_id'],
                Box(*_array(d['bbox'], 4)), d['model_score'], d['model_id'], frame))
            if _json(vehicle.report()) != _json(v):
                raise ValueError('Vehicle observation receipt mismatch')
        lane = value['lanes']
        if lane is not None:
            _fields(lane, ('road_id', 'boundary_ids', 'x', 'y', 'ego', 'scene', 'rear_view'))
            if type(lane['x']) is not list or type(lane['boundary_ids']) is not list:
                raise ValueError('Expected boundary arrays')
            lane = LaneSection(frame, lane['road_id'], tuple(lane['boundary_ids']), tuple(lane['x']),
                               lane['y'], lane['ego'], lane['scene'], lane['rear_view'])
        return cls(frame, vehicle, lane, _roi(value['left']), _roi(value['right']))


@dataclass(frozen=True)
class EventInputs:
    lane_config: LaneConfig
    lamp_config: LampConfig
    pre: Fraction
    post: Fraction
    selected_identity: tuple[int, int, int]
    rows: tuple[InputFrame, ...]

    def __post_init__(self):
        if not isinstance(self.lane_config, LaneConfig) or not isinstance(self.lamp_config, LampConfig):
            raise ValueError('Explicit lane/lamp configurations required')
        seconds(self.pre, True)
        seconds(self.post, True)
        identity(self.selected_identity)
        if not isinstance(self.rows, tuple) or not 1 <= len(self.rows) <= MAX_ROWS:
            raise ValueError('Expected 1..120 input frames')
        previous = reference = None
        for row in self.rows:
            if not isinstance(row, InputFrame):
                raise ValueError('Expected immutable input frame')
            f = row.frame
            seconds(f.time)
            seconds(f.time_base, True)
            if f.ordinal > 2**31-1 or f.stream_index > 2**31-1 or not re.fullmatch('[0-9a-f]{64}', f.source_id):
                raise ValueError('Expected source SHA256 and bounded frame indices')
            key = f.key, f.time_base, f.first_pts
            if reference is not None and key != reference:
                raise ValueError('Input receipt crosses source/clock/geometry')
            if previous is not None and (f.ordinal <= previous.ordinal or f.time <= previous.time):
                raise ValueError('Duplicate/backward input receipt')
            if row.vehicle is not None and row.vehicle.identity != self.selected_identity:
                raise ValueError('Input receipt mixes selected vehicle identities')
            previous, reference = f, key
        if len(self.to_bytes()) > MAX_BYTES:
            raise ValueError('Input receipt exceeds 1MiB')

    def report(self):
        return {'schema': SCHEMA, 'selected_identity': list(self.selected_identity),
                'lane_config': _config_report(self.lane_config), 'lamp_config': _config_report(self.lamp_config),
                'pre': str(self.pre), 'post': str(self.post), 'rows': [row.report() for row in self.rows]}

    def to_bytes(self):
        return (_json(self.report())+'\n').encode('utf-8')

    @property
    def sha256(self):
        """Content identity of canonical bytes, NOT a signature or authenticity proof."""
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data):
        try:
            if type(data) is not bytes or not 0 < len(data) <= MAX_BYTES:
                raise ValueError('Invalid input receipt size')
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError('Duplicate JSON key')
                    result[key] = value
                return result
            def constant(_value):
                raise ValueError('Non-finite JSON value')
            value = json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
            _fields(value, ('schema', 'selected_identity', 'lane_config', 'lamp_config', 'pre', 'post', 'rows'))
            if value['schema'] != SCHEMA or type(value['rows']) is not list or not 1 <= len(value['rows']) <= MAX_ROWS:
                raise ValueError('Unknown schema or row limit')
            return cls(_config(value['lane_config'], LaneConfig, ('dwell', 'max_gap')),
                       _config(value['lamp_config'], LampConfig, ('min_window', 'max_gap', 'min_phase', 'max_phase')),
                       _exact(value['pre']), _exact(value['post']), tuple(_array(value['selected_identity'], 3)),
                       tuple(InputFrame.from_report(row) for row in value['rows']))
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError, ZeroDivisionError, RecursionError):
            raise AppError('INVALID_INPUTS', '관측 입력의 형식·출처·설정·자원 한계를 확인하세요.') from None


@contextmanager
def owned_output(value):
    """One owned, no-replace directory transaction shared by save and replay."""
    output = local_path(value)
    if output.exists():
        raise AppError('OUTPUT_EXISTS', '결과가 이미 있습니다. 새 이름을 지정하세요.')
    if not output.parent.is_dir():
        raise AppError('IO_ERROR', '출력 상위 폴더를 먼저 준비하세요.')
    parent, stage, owner, published = _identity(output.parent)[:2], None, None, False
    try:
        stage = Path(tempfile.mkdtemp(prefix='.crd-inputs-', suffix='.partial', dir=output.parent))
        owner = _identity(stage)[:2]
        yield stage
        for path in (output, output.parent, stage):
            local_path(path)
        if _identity(output.parent)[:2] != parent or _identity(stage)[:2] != owner:
            raise AppError('OUTPUT_CONFLICT', '출력 작업 소유권이 변경되었습니다.')
        _publish(stage, output)
        published = True
    except OSError:
        raise AppError('IO_ERROR', '저장 공간과 파일 접근 권한을 확인하세요.') from None
    finally:
        if stage is not None and not published:
            try:
                local_path(stage)
                local_path(output.parent)
                if (stage.parent != output.parent or _identity(output.parent)[:2] != parent
                        or _identity(stage)[:2] != owner):
                    raise OSError('Ownership changed')
                shutil.rmtree(stage)
            except (OSError, AppError):
                raise AppError('CLEANUP_FAILED', '작업 소유권을 확인하지 못해 임시 결과를 보존했습니다.') from None


def write_bytes(path, data):
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def save_inputs(inputs, output):
    if not isinstance(inputs, EventInputs):
        raise AppError('INVALID_INPUTS', '검증된 관측 입력이 필요합니다.')
    with owned_output(output) as stage:
        write_bytes(stage/'inputs.json', inputs.to_bytes())
    return inputs.sha256


def load_inputs(path):
    path = local_path(path)
    if not path.is_file() or path.suffix.lower() != '.json':
        raise AppError('INVALID_INPUTS', '로컬 JSON 관측 입력 파일이 필요합니다.')
    try:
        before = _identity(path)
        with path.open('rb') as stream:
            data = stream.read(MAX_BYTES+1)
        local_path(path)
        if before != _identity(path):
            raise AppError('INPUT_CHANGED', '읽는 중 관측 입력이 변경되었습니다.')
        return EventInputs.from_bytes(data)
    except OSError:
        raise AppError('IO_ERROR', '관측 입력 파일의 접근 권한을 확인하세요.') from None
