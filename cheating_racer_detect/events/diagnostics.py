"""Caller-recorded lane diagnostics, not observability or absence-of-event proof.

Pure replay has no media/model imports. Source verification is performed by the
catalog exporter; loading this sidecar only checks consistency with its inputs.
"""

from collections import Counter
import json

from ..errors import AppError
from ..analysis.lanes import DECODER_ID
from ..paths import local_path
from ..service import _identity
from .inputs import EventInputs, InputLaneEngine, MAX_BYTES, _json
from .inputs import SCHEMA as LEGACY_SCHEMA, PROVENANCE_SCHEMA

SCHEMA = 'experimental-lane-diagnostics-v1'


def lane_diagnostics(inputs):
    """Fresh owned report for every supplied row, including zero-candidate runs.

    section=None on legacy rows means provenance NOT RECORDED, not that their
    injected lane section was absent. source_reset_reason covers the v2 bridge,
    while lane.reason separately preserves ordinary engine discontinuities.
    """
    return _replay_diagnostics(inputs)[0]


def _replay_diagnostics(inputs):
    """Also retain full candidate values for the source exporter's cross-check."""
    if not isinstance(inputs, EventInputs):
        raise AppError('INVALID_INPUTS', '검증된 관측 입력이 필요합니다.')
    engine = InputLaneEngine(inputs.lane_config)
    rows, candidates = [], []
    counts = dict(input_rows=len(inputs.rows), legacy_rows=0, section_ready_rows=0,
                  section_unknown_rows=0, lane_unknown_rows=0, source_resets=0)
    reasons = {name: Counter() for name in ('section', 'lane', 'source_reset')}
    for row in inputs.rows:
        result = engine.update(row)
        evidence, section, lane = row.lane_evidence, result['section'], result['lane']
        source = None
        if evidence is None:
            counts['legacy_rows'] += 1
        else:
            batch, context = evidence.batch, evidence.context
            source = {'decoder_id': DECODER_ID,
                      'model_id': None if batch is None else batch.model_id,
                      'context_lane_model_id': None if context is None else context.lane_model_id,
                      'rotation_cw': None if batch is None else batch.rotation_cw,
                      'context_id': None if context is None else context.context_id,
                      'road_id': None if context is None else context.road_id,
                      'bindings': None if context is None else [
                          {'slot': b.slot, 'boundary_id': b.boundary_id,
                           'visible_segments': list(b.visible_segments)} for b in context.bindings]}
            counts['section_ready_rows' if section.section is not None else 'section_unknown_rows'] += 1
        counts['lane_unknown_rows'] += lane['status'] == 'unknown'
        counts['source_resets'] += result['reset_reason'] is not None
        for name, reason in (('section', None if section is None else section.reason),
                             ('lane', lane['reason']), ('source_reset', result['reset_reason'])):
            if reason is not None:
                reasons[name][reason] += 1
        candidate_index = len(candidates) if lane['candidate'] is not None else None
        if lane['candidate'] is not None:
            candidates.append(lane['candidate'])
        rows.append({'frame': row.frame.report(), 'source': source,
                     'section': None if section is None else section.report(),
                     'lane': {'status': lane['status'], 'reason': lane['reason'],
                              'candidate_index': candidate_index},
                     'source_reset_reason': result['reset_reason']})
    report = {'schema': SCHEMA, 'inputs_sha256': inputs.sha256,
              'source_sha256': inputs.rows[0].frame.source_id,
              'input_schema': LEGACY_SCHEMA if inputs.rows[0].lane_evidence is None else PROVENANCE_SCHEMA,
              'declaration_status': 'caller_asserted_unverified',
              'counts': {**counts, 'candidate_count': len(candidates)},
              'reason_counts': {k: dict(sorted(v.items())) for k,v in reasons.items()}, 'rows': rows}
    if len(diagnostics_bytes(report)) > MAX_BYTES:
        raise AppError('INVALID_DIAGNOSTICS', '차선 진단의 자원 한계를 확인하세요.')
    return report, tuple(candidates)


def diagnostics_bytes(report):
    return (_json(report)+'\n').encode('utf-8')


def diagnostics_summary(report):
    """Redacted stdout projection of a freshly generated/validated report."""
    return {'counts': dict(report['counts']),
            'reason_counts': {k: dict(v) for k,v in report['reason_counts'].items()},
            'timeline': [{
                'ordinal': row['frame']['ordinal'], 'pts': row['frame']['pts'],
                'time': row['frame']['normalized_time'],
                'section_status': 'not_recorded' if row['section'] is None else row['section']['status'],
                'section_reason': None if row['section'] is None else row['section']['reason'],
                'lane_status': row['lane']['status'], 'lane_reason': row['lane']['reason'],
                'candidate_index': row['lane']['candidate_index'],
                'source_reset_reason': row['source_reset_reason']} for row in report['rows']]}


def load_lane_diagnostics(path, inputs):
    """Read-only strict JSON -> exact fresh input-derived comparison.

    Never authorizes skipping source replay. A rewritten input AND sidecar are
    unauthenticated declarations; content hashes are not signatures.
    """
    path = local_path(path)
    if not path.is_file() or path.suffix.lower() != '.json':
        raise AppError('INVALID_DIAGNOSTICS', '로컬 JSON 차선 진단 파일이 필요합니다.')
    try:
        before = _identity(path)
        with path.open('rb') as stream:
            data = stream.read(MAX_BYTES+1)
        local_path(path)
        if before != _identity(path):
            raise AppError('INPUT_CHANGED', '읽는 중 차선 진단이 변경되었습니다.')
        if not 0 < len(data) <= MAX_BYTES:
            raise ValueError('Diagnostic byte limit')
        def pairs(items):
            result = {}
            for key,value in items:
                if key in result:
                    raise ValueError('Duplicate diagnostic key')
                result[key] = value
            return result
        def constant(_value):
            raise ValueError('Non-finite diagnostic value')
        parsed = json.loads(data.decode('utf-8'),object_pairs_hook=pairs,parse_constant=constant)
        expected = lane_diagnostics(inputs)
        if diagnostics_bytes(parsed) != diagnostics_bytes(expected):
            raise ValueError('Diagnostic input or derived state differs')
        return expected
    except OSError:
        raise AppError('IO_ERROR', '차선 진단 파일의 접근 권한을 확인하세요.') from None
    except (TypeError, ValueError, KeyError, AttributeError, OverflowError, RecursionError):
        raise AppError('INVALID_DIAGNOSTICS', '차선 진단의 형식·입력·재계산 결과를 확인하세요.') from None
