"""Input-derived ROI/gate and discontinuity facts, NOT pixel or signal evidence.

Source export cross-checks source resets against fresh pixel replay. Read-only
loading only establishes consistency with the supplied immutable input receipt.
"""

from collections import Counter
import json
import math

from ..errors import AppError
from ..paths import local_path
from ..service import _identity
from .contracts import boundary
from .diagnostics import diagnostics_bytes
from .inputs import EventInputs, MAX_BYTES
from .lamp_observations import lamp_source_key

SCHEMA = 'experimental-lamp-diagnostics-v1'
LIMITS = ('caller declarations, not verified visibility or signal semantics',
          'input geometry and discontinuities only; no brightness calculation',
          'row counts are not observed coverage', 'content hash is not authentication',
          'not a signal or violation decision; source replay still required')


def _side(row, roi, config):
    box = roi.box
    count = None if box is None else (
        max(0, math.floor(box.x2)-math.ceil(box.x1)) *
        max(0, math.floor(box.y2)-math.ceil(box.y1)))
    status = 'identity_uncertain' if row.vehicle is None else roi.status
    if status == 'usable':
        if count > 65536:
            raise AppError('INVALID_DIAGNOSTICS', '램프 ROI의 자원 한계를 확인하세요.')
        if count < config.min_pixels:
            status = 'too_small'
    return {'roi': None if box is None else list(box.values()),
            'declared_status': roi.status, 'geometry_status': status,
            'whole_pixels': count}


def lamp_diagnostics(inputs):
    """Fresh owned metadata for ALL supplied rows, even without candidates.

    history_break_reasons describes input discontinuity conditions, not the
    LampEngine's sticky history_start_reason or a pixel-verified sample ledger.
    Legacy provenance is not recorded, unlike explicit unknown v3 evidence.
    """
    if not isinstance(inputs, EventInputs):
        raise AppError('INVALID_INPUTS', '검증된 관측 입력이 필요합니다.')
    counts = dict(input_rows=len(inputs.rows), legacy_rows=0, observation_unknown_rows=0,
                  source_resets=0, gap_rows=0, no_actual_rows=0)
    reasons = {key: Counter() for key in ('history_break', 'source_reset', 'left', 'right')}
    rows, previous, previous_source = [], None, None
    for row in inputs.rows:
        evidence = row.lamp_evidence
        source = None if evidence is None else evidence.report()['observation']
        key = None if evidence is None else lamp_source_key(evidence.observation)
        reset = None
        if evidence is None:
            provenance = 'not_recorded'
            counts['legacy_rows'] += 1
        else:
            provenance = 'unknown' if key is None else 'recorded'
            counts['observation_unknown_rows'] += key is None
            reset = ('lamps_unknown' if key is None else 'lamp_source_change'
                     if previous_source is not None and key != previous_source else None)
        clock = boundary(None if previous is None else previous.frame,
                         row.frame, inputs.lamp_config.max_gap)
        breaks = [] if clock is None else [clock]
        if reset is not None:
            breaks.append(reset)
            reasons['source_reset'][reset] += 1
            counts['source_resets'] += 1
        if row.vehicle is None:
            breaks.append('no_actual_observation')
            counts['no_actual_rows'] += 1
        elif previous is not None:
            if previous.vehicle is None:
                breaks.append('actual_observation_resumed')
            elif (row.vehicle.identity != previous.vehicle.identity or
                  row.vehicle.detection.model_id != previous.vehicle.detection.model_id):
                breaks.append('identity_or_model_change')
        if any(roi.status == 'identity_uncertain' for roi in (row.left, row.right)):
            breaks.append('identity_uncertain')
        missing = 0 if previous is None else row.frame.ordinal-previous.frame.ordinal-1
        elapsed = None if previous is None else str(row.frame.time-previous.frame.time)
        counts['gap_rows'] += clock in ('missing_frame', 'time_gap')
        sides = {side: _side(row, getattr(row, side), inputs.lamp_config) for side in ('left', 'right')}
        for side in sides:
            reasons[side][sides[side]['geometry_status']] += 1
        breaks = sorted(set(breaks))
        reasons['history_break'].update(breaks)
        rows.append({'frame': row.frame.report(), 'provenance_status': provenance,
                     'source': source, **sides, 'missing_input_frames': missing,
                     'elapsed_seconds': elapsed, 'clock_break_reason': clock,
                     'source_reset_reason': reset, 'history_break_reasons': breaks})
        previous, previous_source = row, key
    report = {'schema': SCHEMA, 'inputs_sha256': inputs.sha256,
              'source_sha256': inputs.rows[0].frame.source_id, 'input_schema': inputs.schema,
              'declaration_status': 'caller_asserted_unverified', 'counts': counts,
              'reason_counts': {k: dict(sorted(v.items())) for k, v in reasons.items()},
              'rows': rows, 'limits': list(LIMITS)}
    if len(diagnostics_bytes(report)) > MAX_BYTES:
        raise AppError('INVALID_DIAGNOSTICS', '램프 진단의 자원 한계를 확인하세요.')
    return report


def lamp_diagnostics_summary(report):
    """Redacted CLI projection; IDs, boxes, paths and hashes stay in sidecar."""
    return {'counts': dict(report['counts']),
            'reason_counts': {k: dict(v) for k, v in report['reason_counts'].items()},
            'timeline': [{
                'ordinal': row['frame']['ordinal'], 'pts': row['frame']['pts'],
                'time': row['frame']['normalized_time'],
                'provenance_status': row['provenance_status'],
                'left_status': row['left']['geometry_status'],
                'right_status': row['right']['geometry_status'],
                'missing_input_frames': row['missing_input_frames'],
                'elapsed_seconds': row['elapsed_seconds'],
                'clock_break_reason': row['clock_break_reason'],
                'source_reset_reason': row['source_reset_reason'],
                'history_break_reasons': list(row['history_break_reasons'])} for row in report['rows']],
            'scope': 'input declarations and geometry; not brightness or observed coverage'}


def load_lamp_diagnostics(path, inputs):
    """Strict bounded read-only JSON -> exact fresh input-derived comparison."""
    path = local_path(path)
    if not path.is_file() or path.suffix.lower() != '.json':
        raise AppError('INVALID_DIAGNOSTICS', '로컬 JSON 램프 진단 파일이 필요합니다.')
    try:
        before = _identity(path)
        with path.open('rb') as stream:
            data = stream.read(MAX_BYTES+1)
        local_path(path)
        if before != _identity(path):
            raise AppError('INPUT_CHANGED', '읽는 중 램프 진단이 변경되었습니다.')
        if not 0 < len(data) <= MAX_BYTES:
            raise ValueError('Diagnostic byte limit')
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('Duplicate diagnostic key')
                result[key] = value
            return result
        def constant(_value):
            raise ValueError('Non-finite diagnostic value')
        parsed = json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
        expected = lamp_diagnostics(inputs)
        if diagnostics_bytes(parsed) != diagnostics_bytes(expected):
            raise ValueError('Diagnostic input or derived state differs')
        return expected
    except OSError:
        raise AppError('IO_ERROR', '램프 진단 파일의 접근 권한을 확인하세요.') from None
    except (TypeError, ValueError, KeyError, AttributeError, OverflowError, RecursionError):
        raise AppError('INVALID_DIAGNOSTICS', '램프 진단의 형식·입력·재계산 결과를 확인하세요.') from None
