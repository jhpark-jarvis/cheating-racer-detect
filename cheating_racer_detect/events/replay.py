"""Replay caller-recorded inputs against original pixels, without re-tracking."""

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR
import json
from pathlib import Path
import tempfile

from ..errors import AppError
from ..paths import local_path, source_path
from ..review.video import prepare
from ..service import _identity, _sha256
from ..tools import discover_tools, run_tool
from ..tracking.contracts import integer
from .export import decimal_bound, export_candidate, validate_frame
from .inputs import EventInputs, owned_output, write_bytes
from .lane import LaneEngine
from .lamp import LampEngine


@dataclass(frozen=True)
class ReplayAnalysis:
    candidates: tuple
    lamps: LampEngine


def analyze_inputs(input_path, inputs, toolchain=None):
    """Fresh engines each call, <=120 decoded span frames, no interpolation."""
    try:
        return _analyze_inputs(input_path, inputs, toolchain)
    except OSError:
        raise AppError('IO_ERROR', '원본 디코딩과 임시 저장 공간을 확인하세요.') from None


def _analyze_inputs(input_path, inputs, toolchain):
    if not isinstance(inputs, EventInputs):
        raise AppError('INVALID_INPUTS', '검증된 관측 입력이 필요합니다.')
    source, tools = source_path(input_path), toolchain or discover_tools()
    before, source_hash = _identity(source), _sha256(source)
    first, last = inputs.rows[0].frame, inputs.rows[-1].frame
    if first.source_id != source_hash:
        raise AppError('INVALID_INPUTS', '관측 입력과 원본 영상의 해시가 다릅니다.')
    video, timeline, selected, ordinal, rotation = prepare(source,
        decimal_bound(first.time, ROUND_FLOOR), decimal_bound(last.time+last.time_base, ROUND_CEILING), tools)
    for row in inputs.rows:
        validate_frame(row.frame, source_hash, video, timeline, rotation)
    lane, lamp = LaneEngine(inputs.lane_config), LampEngine(inputs.lamp_config)
    events, rows = [], {row.frame.ordinal: row for row in inputs.rows}
    with tempfile.TemporaryDirectory(prefix='crd-input-replay-') as directory:
        raw = Path(directory)/'source.bgr'
        from ..media import _input_options
        run_tool([tools.ffmpeg, '-nostdin', '-v', 'error', '-xerror', '-err_detect', 'explode',
                  *_input_options(), '-noautorotate', '-i', source, '-map', '0:v:0',
                  '-vf', f'select=between(n\\,{ordinal}\\,{ordinal+len(selected)-1})',
                  '-fps_mode', 'passthrough', '-an', '-pix_fmt', 'bgr24', '-f', 'rawvideo', raw], timeout=120)
        size = first.width*first.height*3
        if raw.stat().st_size != size*len(selected):
            raise AppError('OUTPUT_INVALID', '원본 프레임과 디코딩 결과가 다릅니다.')
        with raw.open('rb') as stream:
            for n in range(len(selected)):
                pixels = stream.read(size)
                row = rows.get(ordinal+n)
                if row is None:
                    continue  # Missing annotated frames are NOT filled in.
                event = lane.update(row.frame, row.vehicle, row.lanes)['candidate']
                lamp.update(row.frame, row.vehicle, pixels, row.left, row.right)
                if event is not None:
                    events.append(event)
    local_path(source)
    if before != _identity(source) or source_hash != _sha256(source):
        raise AppError('INPUT_CHANGED', '재생 중 원본 영상이 변경되었습니다.')
    return ReplayAnalysis(tuple(events), lamp)


def recorded_observer(inputs):
    """Only the selected recorded actual vehicle; no estimated motion or other IDs."""
    if not isinstance(inputs, EventInputs):
        raise AppError('INVALID_INPUTS', '검증된 관측 입력이 필요합니다.')
    rows = {row.frame.ordinal: row for row in inputs.rows}
    reference = inputs.rows[0].frame
    def observe(frame, _pixels):
        if (frame.key, frame.time_base, frame.first_pts) != (reference.key, reference.time_base, reference.first_pts):
            raise AppError('INVALID_INPUTS', '검토 프레임과 저장된 관측의 출처가 다릅니다.')
        row = rows.get(frame.ordinal)
        if row is not None and row.frame != frame:
            raise AppError('INVALID_INPUTS', '저장된 관측과 원본 시각이 다릅니다.')
        vehicle = None if row is None else row.vehicle
        tracks = [] if vehicle is None else [{
            'identity': list(vehicle.identity), 'identity_status': 'unverified', 'state': 'observed',
            'detection_index': vehicle.detection.index, 'detection_bbox': list(vehicle.box.values()),
            'estimated_bbox': None, 'model_score': vehicle.detection.score, 'model_id': vehicle.detection.model_id,
            'last_observed_time': str(frame.time), 'unobserved_seconds': '0'}]
        return {'schema': 'experimental-tracks-v1', 'frame': frame.report(),
                'segment': inputs.selected_identity[0], 'delta_seconds': None, 'reset_reason': None,
                'detections': [] if vehicle is None else [vehicle.detection.report()],
                'tracks': tracks, 'ambiguities': []}
    return observe


def export_replay(input_path, inputs, output_path, *, candidate_index, toolchain=None):
    """Atomically publish the receipt + selected replay candidate/media bundle."""
    integer(candidate_index)
    if not isinstance(inputs, EventInputs):
        raise AppError('INVALID_INPUTS', '검증된 관측 입력이 필요합니다.')
    # Reject existing/invalid destinations before spending work on decoding.
    from ..paths import destination_path
    destination_path(output_path, source_path(input_path))
    source, tools = source_path(input_path), toolchain or discover_tools()
    before, source_hash = _identity(source), _sha256(source)
    analysis = analyze_inputs(source, inputs, tools)
    if candidate_index >= len(analysis.candidates):
        raise AppError('INVALID_CANDIDATE', '재생 결과에 해당 후보가 없습니다. 후보 번호를 확인하세요.')
    candidate = analysis.candidates[candidate_index]
    with owned_output(output_path) as stage:
        write_bytes(stage/'inputs.json', inputs.to_bytes())
        bundle = export_candidate(source, candidate, stage/'candidate', recorded_observer(inputs),
                                  pre=inputs.pre, post=inputs.post, lamps=analysis.lamps, toolchain=tools)
        result = {'schema': 'experimental-event-replay-v1', 'status': 'complete',
                  'inputs': 'inputs.json', 'inputs_sha256': inputs.sha256, 'source_sha256': source_hash,
                  'candidate_index': candidate_index, 'candidate_count': len(analysis.candidates),
                  'candidate': 'candidate/candidate.json', 'signal_state': bundle['signal_state'],
                  'limits': ['caller-recorded inputs, not ground truth', 'selected actual vehicle only; no re-tracking',
                             'identity unverified', 'content hash is not authentication', 'not a violation decision']}
        write_bytes(stage/'replay.json', (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8'))
        local_path(source)
        if before != _identity(source) or source_hash != _sha256(source):
            raise AppError('INPUT_CHANGED', '처리 중 원본 영상이 변경되었습니다.')
    return result
