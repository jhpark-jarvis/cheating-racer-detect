"""Source-verified compact candidate catalogues; never a trusted analysis cache."""

from dataclasses import dataclass
import hashlib
import json
import re

from ..errors import AppError
from ..paths import destination_path, local_path, source_path
from ..service import _identity, _sha256
from ..source import inspect_source, release_pixels, source_job
from ..tools import discover_tools
from .contracts import identity
from .export import candidate_context
from .inputs import EventInputs, MAX_BYTES, MAX_ROWS, _array, _exact, _fields, _frame, _json, owned_output, write_bytes

SCHEMA = 'experimental-candidate-catalog-v1'
LIMITS = ('caller-recorded inputs, not ground truth', 'identity unverified',
          'content hash is not authentication', 'planned context; media not yet validated',
          'brightness pattern, not signal recognition', 'not a violation decision')
SIDE_STATES = ('unknown', 'blink_observed', 'no_blink_observed')
REASONS = ('occluded', 'out_of_view', 'too_small', 'identity_uncertain', 'model_uncertain',
           'insufficient_history', 'insufficient_cycles')


def _bytes(value):
    return (_json(value)+'\n').encode('utf-8')


def _hash(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Invalid content digest')


def _count(value, maximum=MAX_ROWS):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError('Invalid catalogue count')


def _pair(value, positive=False):
    first, last = (_exact(item) for item in _array(value, 2))
    if first > last or (positive and first == last):
        raise ValueError('Invalid time interval')
    return first, last


def _side(value):
    _fields(value, ('state', 'blink_evidence', 'reasons', 'valid_samples'))
    if value['state'] not in SIDE_STATES or type(value['blink_evidence']) is not bool:
        raise ValueError('Invalid signal summary')
    reasons = value['reasons']
    if (type(reasons) is not list or len(reasons) > len(REASONS)
            or any(type(reason) is not str or reason not in REASONS for reason in reasons)
            or reasons != sorted(set(reasons)) or (value['state'] == 'unknown') != bool(reasons)):
        raise ValueError('Invalid unknown reasons')
    if (value['state'] == 'blink_observed' and not value['blink_evidence']
            or value['state'] == 'no_blink_observed' and value['blink_evidence']):
        raise ValueError('Inconsistent blink fact')
    _count(value['valid_samples'])


def _signal(left, right):
    if 'unknown' in (left['state'], right['state']):
        return 'unknown'
    if left['blink_evidence'] and right['blink_evidence']:
        return 'both_blink'
    return 'left_blink' if left['blink_evidence'] else 'right_blink' if right['blink_evidence'] else 'no_blink_observed'


def _validate(value):
    _fields(value, ('schema', 'status', 'inputs', 'inputs_sha256', 'source_sha256', 'source_frame',
                    'selected_identity', 'candidate_count', 'candidates', 'limits'))
    if value['schema'] != SCHEMA or value['status'] != 'complete' or value['inputs'] != 'inputs.json':
        raise ValueError('Unknown catalogue contract')
    _hash(value['inputs_sha256'])
    _hash(value['source_sha256'])
    frame = _frame(value['source_frame'])
    if frame.source_id != value['source_sha256']:
        raise ValueError('Source receipt differs')
    identity(tuple(_array(value['selected_identity'], 3)))
    _count(value['candidate_count'])
    entries = value['candidates']
    if type(entries) is not list or len(entries) != value['candidate_count'] or value['limits'] != list(LIMITS):
        raise ValueError('Invalid candidate collection or limits')
    previous = None
    for index, entry in enumerate(entries):
        _fields(entry, ('candidate_index', 'candidate_sha256', 'direction', 'start', 'crossing', 'completion',
                        'requested_context', 'context', 'signal_state', 'left', 'right'))
        _count(entry['candidate_index'])
        if entry['candidate_index'] != index or entry['direction'] not in ('left', 'right'):
            raise ValueError('Invalid original candidate index/direction')
        _hash(entry['candidate_sha256'])
        start, crossing, completion = (_pair(entry[name]) for name in ('start', 'crossing', 'completion'))
        if not start[0] <= start[1] <= crossing[1] <= completion[0] <= completion[1]:
            raise ValueError('Invalid candidate chronology')
        if previous is not None and completion[1] <= previous:
            raise ValueError('Backward candidate order')
        previous = completion[1]
        requested, context = (_pair(entry[name], True) for name in ('requested_context', 'context'))
        if not context[0] <= requested[0] <= start[0] or not completion[1] < requested[1] <= context[1]:
            raise ValueError('Invalid planned context')
        for name in ('left', 'right'):
            _side(entry[name])
        if entry['signal_state'] != _signal(entry['left'], entry['right']):
            raise ValueError('Signal summary differs from sides')


@dataclass(frozen=True)
class CandidateCatalog:
    """Immutable canonical bytes; report() always returns a fresh owned object."""
    data: bytes

    def __post_init__(self):
        try:
            if type(self.data) is not bytes or not 0 < len(self.data) <= MAX_BYTES:
                raise ValueError('Catalogue size limit')
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError('Duplicate catalogue key')
                    result[key] = value
                return result
            def constant(_value):
                raise ValueError('Non-finite catalogue value')
            value = json.loads(self.data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
            _validate(value)
            canonical = _bytes(value)
            if len(canonical) > MAX_BYTES:
                raise ValueError('Canonical catalogue size limit')
            object.__setattr__(self, 'data', canonical)
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError, ZeroDivisionError, RecursionError):
            raise AppError('INVALID_CATALOG', '후보 목록의 형식·출처·시각·자원 한계를 확인하세요.') from None

    @classmethod
    def from_bytes(cls, data):
        return cls(data)

    def to_bytes(self):
        return self.data

    def report(self):
        return json.loads(self.data)

    @property
    def sha256(self):
        return hashlib.sha256(self.data).hexdigest()


def make_catalog(inputs, analysis, timeline):
    """Internal: a newly source-verified replay only; never untrusted engine state."""
    entries = []
    for index, candidate in enumerate(analysis.candidates):
        start, end, requested, lamp = candidate_context(candidate, inputs.pre, inputs.post, timeline, analysis.lamps)
        entries.append({'candidate_index': index, 'candidate_sha256': hashlib.sha256(_bytes(candidate.report())).hexdigest(),
                        'direction': candidate.direction,
                        **{name: [str(frame.time) for frame in getattr(candidate, name)]
                           for name in ('start', 'crossing', 'completion')},
                        'requested_context': requested, 'context': [str(start), str(end)],
                        'signal_state': lamp['signal_state'],
                        **{side: {key: lamp[side][key] for key in ('state', 'blink_evidence', 'reasons', 'valid_samples')}
                           for side in ('left', 'right')}})
    return CandidateCatalog(_bytes({'schema': SCHEMA, 'status': 'complete', 'inputs': 'inputs.json',
        'inputs_sha256': inputs.sha256, 'source_sha256': inputs.rows[0].frame.source_id,
        'source_frame': inputs.rows[0].frame.report(), 'selected_identity': list(inputs.selected_identity),
        'candidate_count': len(entries), 'candidates': entries, 'limits': list(LIMITS)}))


def load_catalog(path):
    path = local_path(path)
    if not path.is_file() or path.suffix.lower() != '.json':
        raise AppError('INVALID_CATALOG', '로컬 JSON 후보 목록 파일이 필요합니다.')
    try:
        before = _identity(path)
        with path.open('rb') as stream:
            data = stream.read(MAX_BYTES+1)
        local_path(path)
        if before != _identity(path):
            raise AppError('INPUT_CHANGED', '읽는 중 후보 목록이 변경되었습니다.')
        return CandidateCatalog(data)
    except OSError:
        raise AppError('IO_ERROR', '후보 목록 파일의 접근 권한을 확인하세요.') from None


@source_job
def export_catalog(input_path, inputs, output_path, *, toolchain=None):
    """Fresh source analysis -> compact catalogue + inputs, with no media output."""
    return _export_listing(input_path, inputs, output_path, toolchain, False)[0]


@source_job
def export_catalog_with_diagnostics(input_path, inputs, output_path, *, toolchain=None):
    """Explicit opt-in: atomically publish inputs, unchanged catalog-v1 and lanes.

    Returns (CandidateCatalog, owned diagnostic dict). The sidecar is descriptive,
    not a new trusted catalogue/batch schema or an analysis cache.
    """
    return _export_listing(input_path, inputs, output_path, toolchain, True)[:2]


@source_job
def export_catalog_with_lamp_diagnostics(input_path, inputs, output_path, *, toolchain=None,
                                         include_lane_diagnostics=False):
    """Return (catalog, lamp report, optional lane report); publish all or none.

    Neither diagnostic is a cache or a new candidate/batch contract. Default
    listing and the existing lane-only API keep their previous bytes/returns.
    """
    if type(include_lane_diagnostics) is not bool:
        raise AppError('INVALID_DIAGNOSTICS', '차선 진단 선택은 명시적 bool이어야 합니다.')
    catalog, lanes, lamps = _export_listing(input_path, inputs, output_path, toolchain,
                                            include_lane_diagnostics, True)
    return catalog, lamps, lanes


def _export_listing(input_path, inputs, output_path, toolchain, include_diagnostics, include_lamps=False):
    from .replay import analyze_inputs
    if not isinstance(inputs, EventInputs):
        raise AppError('INVALID_INPUTS', '검증된 관측 입력이 필요합니다.')
    source = source_path(input_path)
    destination_path(output_path, source)
    tools = toolchain or discover_tools()
    try:
        before, source_hash = _identity(source), _sha256(source)
        analysis = analyze_inputs(source, inputs, tools)
        _video, _audio, timeline = inspect_source(source, tools, max_axis=1024)
        catalog = make_catalog(inputs, analysis, timeline)
        diagnostics = None
        if include_diagnostics:
            from .diagnostics import _replay_diagnostics
            diagnostics, candidates = _replay_diagnostics(inputs)
            # Compare full candidates, not just a count. The extra pure replay
            # never reads pixels or reuses potentially stale caller engine state.
            resets = tuple((row.frame, item['source_reset_reason'])
                for row,item in zip(inputs.rows, diagnostics['rows']) if item['source_reset_reason'] is not None)
            if candidates != analysis.candidates or resets != analysis.lane_resets:
                raise AppError('INVALID_DIAGNOSTICS', '원본 재생과 차선 진단 결과가 다릅니다.')
        lamps = None
        if include_lamps:
            from .lamp_diagnostics import lamp_diagnostics
            lamps = lamp_diagnostics(inputs)
            resets = tuple((row.frame, item['source_reset_reason'])
                for row, item in zip(inputs.rows, lamps['rows']) if item['source_reset_reason'] is not None)
            if resets != analysis.lamp_resets:
                raise AppError('INVALID_DIAGNOSTICS', '원본 재생과 램프 출처 단절이 다릅니다.')
        with owned_output(output_path) as stage:
            write_bytes(stage/'inputs.json', inputs.to_bytes())
            write_bytes(stage/'catalog.json', catalog.to_bytes())
            if diagnostics is not None:
                from .diagnostics import diagnostics_bytes
                write_bytes(stage/'lane-diagnostics.json', diagnostics_bytes(diagnostics))
            if lamps is not None:
                from .diagnostics import diagnostics_bytes
                write_bytes(stage/'lamp-diagnostics.json', diagnostics_bytes(lamps))
            local_path(source)
            if before != _identity(source) or source_hash != _sha256(source):
                raise AppError('INPUT_CHANGED', '목록 생성 중 원본 영상이 변경되었습니다.')
            release_pixels()
        return catalog, diagnostics, lamps
    except OSError:
        raise AppError('IO_ERROR', '원본 접근 권한과 목록 저장 공간을 확인하세요.') from None


def listing_summary(catalog, diagnostics=None, lamp_diagnostics=None):
    """Common legacy-compatible/redacted installed and source CLI output."""
    result = catalog.report()
    keys = ('candidate_index', 'direction', 'start', 'crossing', 'completion', 'signal_state', 'left', 'right')
    summary = {'status': result['status'], 'candidate_count': result['candidate_count'],
               'candidates': [{key: item[key] for key in keys} for item in result['candidates']],
               'scope': 'recorded inputs; planned context; not automatic recognition'}
    if diagnostics is not None:
        from .diagnostics import diagnostics_summary
        summary['lane_diagnostics'] = diagnostics_summary(diagnostics)
    if lamp_diagnostics is not None:
        from .lamp_diagnostics import lamp_diagnostics_summary
        summary['lamp_diagnostics'] = lamp_diagnostics_summary(lamp_diagnostics)
    return summary
