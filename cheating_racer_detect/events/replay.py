"""Replay caller-recorded inputs against original pixels, without re-tracking."""

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR
import json

from ..errors import AppError
from ..paths import local_path, source_path
from ..pixels import pixel_window, prime_pixel_window
from ..review.video import prepare
from ..service import _identity, _sha256
from ..source import active_job, inspect_source, release_pixels, retain_pixels, source_job
from ..tools import discover_tools
from ..tracking.contracts import integer
from .export import candidate_context, decimal_bound, export_candidate, validate_frame
from .inputs import EventInputs, InputLaneEngine, InputLampEngine, MAX_ROWS, owned_output, write_bytes
from .lamp import LampEngine


@dataclass(frozen=True)
class ReplayAnalysis:
    candidates: tuple
    lamps: LampEngine
    lane_resets: tuple = ()  # (original Frame, recomputed lane-source reason)
    lamp_resets: tuple = ()  # v3 ROI-source discontinuities, not lamp semantics


@source_job
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
    lane = InputLaneEngine(inputs.lane_config)
    lamp = (LampEngine(inputs.lamp_config) if inputs.rows[0].lamp_evidence is None
            else InputLampEngine(inputs.lamp_config))
    events, resets, lamp_resets, rows = [], [], [], {row.frame.ordinal: row for row in inputs.rows}
    with pixel_window(source, tools, video, timeline, ordinal, len(selected), complete=True) as stream:
        size = first.width*first.height*3
        for n in range(len(selected)):
            pixels = stream.read(size)
            row = rows.get(ordinal+n)
            if row is None:
                continue  # Missing annotated frames are NOT filled in.
            result = lane.update(row)
            event = result['lane']['candidate']
            if result['reset_reason'] is not None:
                resets.append((row.frame, result['reset_reason']))
            if isinstance(lamp,InputLampEngine):
                observed=lamp.update_recorded(row,pixels)
                if observed['reset_reason'] is not None:
                    lamp_resets.append((row.frame,observed['reset_reason']))
            else:
                lamp.update(row.frame,row.vehicle,pixels,row.left,row.right)
            if event is not None:
                events.append(event)
    local_path(source)
    if before != _identity(source) or source_hash != _sha256(source):
        raise AppError('INPUT_CHANGED', '재생 중 원본 영상이 변경되었습니다.')
    return ReplayAnalysis(tuple(events), lamp, tuple(resets),tuple(lamp_resets))


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


@source_job
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


def _selection(value):
    # Only bounded materialized sequences: no unbounded iterators or ambiguous
    # ordering. Snapshot mutable lists before analysis or caller callbacks run.
    if (type(value) not in (list, tuple) or not 1 <= len(value) <= MAX_ROWS
            or any(type(index) is not int or index < 0 for index in value)
            or len(set(value)) != len(value)):
        raise AppError('INVALID_SELECTION', '중복 없이 0 이상의 후보 번호를 1~120개 지정하세요.')
    return tuple(value)


@dataclass(frozen=True)
class _WindowGroup:
    first: int
    count: int
    indices: tuple[int, ...]
    prime: bool = True


def _group_windows(windows, retained=None):
    """Internal ordinal plan; output receipts keep the caller's original order."""
    windows = tuple(windows)
    groups = []
    for first, count, index in sorted(windows):
        if type(first) is not int or type(count) is not int or first < 0 or not 1 <= count <= MAX_ROWS:
            raise AppError('REVIEW_LIMIT', '검토 구간은 최대 120프레임이어야 합니다.')
        if groups and first <= groups[-1].first+groups[-1].count:
            previous = groups[-1]
            end = max(first+count, previous.first+previous.count)
            if end-previous.first <= MAX_ROWS:
                groups[-1] = _WindowGroup(previous.first, end-previous.first, previous.indices+(index,))
                continue
        groups.append(_WindowGroup(first, count, (index,)))
    if retained is not None:
        left, length = retained
        for n, group in enumerate(groups):
            # Do not replace a useful analysis window with a larger union when
            # the uncovered individual requests would generate fewer pixels.
            separate = sum(count for first,count,index in windows if index in group.indices
                           and not (left <= first and first+count <= left+length))
            if 0 < separate < group.count:
                groups[n] = _WindowGroup(group.first,group.count,group.indices,False)
    return tuple(groups)


def _batch_windows(source, inputs, analysis, indices, tools):
    video, _audio, timeline = inspect_source(source, tools, max_axis=1024)
    windows = []
    for index in indices:
        start, end, _requested, _lamp = candidate_context(analysis.candidates[index], inputs.pre, inputs.post,
                                                         timeline, analysis.lamps)
        _video, _timeline, selected, first, _rotation = prepare(
            source, decimal_bound(start, ROUND_FLOOR), decimal_bound(end, ROUND_CEILING), tools)
        windows.append((first, len(selected), index))
    job = active_job()
    cached = None if job is None or not job.reuse else job.pixels
    retained = (cached.first,cached.count) if cached is not None and cached.key[:3] == (source,tools,_identity(source)) else None
    return video, timeline, _group_windows(windows,retained)


@source_job
def export_replay_batch(input_path, inputs, output_path, *, candidate_indices, toolchain=None, catalog=None):
    """Analyze once; publish all explicitly selected bundles or none.

    Caller order and zero-based replay indices are preserved. Empty, duplicate
    or invalid selections fail; no implicit all-candidates or partial success.
    Each candidate still runs its original full media/hash/output checks.
    An optional immutable catalogue must match a newly replayed full summary;
    it never substitutes for analysis and is copied with its canonical hash.
    """
    indices = _selection(candidate_indices)
    if not isinstance(inputs, EventInputs):
        raise AppError('INVALID_INPUTS', '검증된 관측 입력이 필요합니다.')
    from ..paths import destination_path
    source = source_path(input_path)
    destination_path(output_path, source)
    tools = toolchain or discover_tools()
    try:
        before, source_hash = _identity(source), _sha256(source)
        if catalog is not None:
            from .catalog import CandidateCatalog, make_catalog
            if type(catalog) is not CandidateCatalog:
                raise AppError('INVALID_CATALOG', '검증된 후보 목록이 필요합니다.')
            reference = catalog.report()
            if reference['source_sha256'] != source_hash or reference['inputs_sha256'] != inputs.sha256:
                raise AppError('CATALOG_CHANGED', '후보 목록과 원본·관측 입력이 다릅니다. 목록을 다시 생성하세요.')
        analysis = analyze_inputs(source, inputs, tools)
        if catalog is not None:
            from ..source import inspect_source
            _video, _audio, timeline = inspect_source(source, tools, max_axis=1024)
            current = make_catalog(inputs, analysis, timeline)
            if catalog.to_bytes() != current.to_bytes():
                raise AppError('CATALOG_CHANGED', '후보 목록과 새 재생 결과가 다릅니다. 목록을 다시 생성하세요.')
        if any(index >= len(analysis.candidates) for index in indices):
            raise AppError('INVALID_CANDIDATE', '재생 결과에 해당 후보가 없습니다. 모든 후보 번호를 확인하세요.')
        video, timeline, groups = _batch_windows(source, inputs, analysis, indices, tools)
        with owned_output(output_path) as stage:
            write_bytes(stage/'inputs.json', inputs.to_bytes())
            (stage/'candidates').mkdir()
            by_index = {}
            with retain_pixels():
                for group in groups:
                    if group.prime:
                        prime_pixel_window(source, tools, video, timeline, group.first, group.count)
                    for index in group.indices:
                        candidate = analysis.candidates[index]
                        relative = f'candidates/{index:06d}'
                        bundle = export_candidate(source, candidate, stage/relative, recorded_observer(inputs),
                                                  pre=inputs.pre, post=inputs.post, lamps=analysis.lamps, toolchain=tools)
                        by_index[index] = {'candidate_index': index, 'candidate': relative+'/candidate.json',
                                           'direction': candidate.direction, 'signal_state': bundle['signal_state']}
            entries = [by_index[index] for index in indices]
            result = {'schema': 'experimental-event-replay-batch-v1', 'status': 'complete',
                      'inputs': 'inputs.json', 'inputs_sha256': inputs.sha256, 'source_sha256': source_hash,
                      'candidate_count': len(analysis.candidates), 'selected_count': len(indices),
                      'candidate_indices': list(indices), 'candidates': entries,
                      'limits': ['caller-recorded inputs, not ground truth',
                                 'selected actual vehicle only; no re-tracking', 'identity unverified',
                                 'content hash is not authentication', 'not a violation decision',
                                 'all selected candidates or none; no partial success']}
            if catalog is not None:
                write_bytes(stage/'catalog.json', catalog.to_bytes())
                result['catalog'] = 'catalog.json'
                result['catalog_sha256'] = catalog.sha256
            write_bytes(stage/'batch.json', (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8'))
            local_path(source)
            if before != _identity(source) or source_hash != _sha256(source):
                raise AppError('INPUT_CHANGED', '처리 중 원본 영상이 변경되었습니다.')
            release_pixels()  # Cleanup errors must precede the outer final publish.
        return result
    except OSError:
        raise AppError('IO_ERROR', '원본 접근 권한과 출력 저장 공간을 확인하세요.') from None
