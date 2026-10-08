"""Candidate, observed lamp facts and verified review media in one transaction."""

from decimal import Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import tempfile

from ..errors import AppError
from ..paths import destination_path, local_path, source_path
from ..review.video import export_review, prepare
from ..service import _identity, _publish, _sha256
from ..tools import discover_tools
from .contracts import Candidate, Vehicle, seconds
from .lamp import LampEngine


def decimal_bound(value, rounding):
    with localcontext() as context:
        context.prec = 40
        context.rounding = rounding
        return Decimal(value.numerator)/Decimal(value.denominator)


def validate_frame(frame, source_hash, video, timeline, rotation):
    if (frame.source_id != source_hash or frame.stream_index != video['index']
            or (frame.width, frame.height, frame.display_rotation) != (video['width'], video['height'], rotation)
            or (frame.time_base, frame.first_pts) != (timeline.time_base, timeline.frames[0].pts)
            or frame.ordinal >= len(timeline.frames) or frame.pts != timeline.frames[frame.ordinal].pts):
        raise AppError('INVALID_CANDIDATE', '후보 관측과 원본 영상의 시간·출처가 일치하지 않습니다.')


def export_candidate(input_path, candidate, output_path, observe, *, pre, post, lamps=None, toolchain=None):
    """Explicit positive pre/post seconds; all outputs publish or none do.

    A caller-owned LampEngine is optional; missing history is unknown, not off.
    No automatic detection, signal interpretation, violation or reporting.
    """
    if not isinstance(candidate, Candidate) or not callable(observe) or (lamps is not None and not isinstance(lamps, LampEngine)):
        raise AppError('INVALID_CANDIDATE', '검증된 후보와 현재 관측 callback이 필요합니다.')
    seconds(pre, True)
    seconds(post, True)
    source = source_path(input_path)
    output = destination_path(output_path, source)
    tools = toolchain or discover_tools()
    stage, owner, published = None, None, False
    parent = _identity(output.parent)[:2]
    try:
        before, source_hash = _identity(source), _sha256(source)
        # Verify source receipts before spending work on a bundle. Existing bounds apply.
        sample_time = candidate.crossing[1].time
        video, timeline, _, _, rotation = prepare(source, decimal_bound(sample_time, ROUND_FLOOR),
                                                   decimal_bound(sample_time+candidate.crossing[1].time_base, ROUND_CEILING), tools)
        anchors = set(candidate.start+candidate.crossing+candidate.completion)
        for frame in anchors:
            validate_frame(frame, source_hash, video, timeline, rotation)
        duration = timeline.frames[-1].pts*timeline.time_base-timeline.frames[0].pts*timeline.time_base+timeline.frames[-1].duration*timeline.time_base
        start = max(Fraction(0), candidate.start[0].time-pre)
        end = min(duration, candidate.completion[1].time+post)
        if end <= candidate.completion[1].time:
            raise AppError('INVALID_CANDIDATE', '후보 완료 프레임의 길이를 확인하지 못했습니다.')
        lamp = None
        if lamps is not None:
            if lamps.identity is not None and lamps.identity != candidate.identity:
                raise AppError('INVALID_CANDIDATE', '램프와 후보의 차량 이력이 다릅니다.')
            for frame in lamps.frames():
                validate_frame(frame, source_hash, video, timeline, rotation)
            lamp = lamps.report(start, end)
        stage = Path(tempfile.mkdtemp(prefix='.crd-candidate-', suffix='.partial', dir=output.parent))
        owner = _identity(stage)[:2]
        seen = set()
        def checked_observer(frame, pixels):
            report = observe(frame, pixels)
            if frame in anchors:
                if Vehicle.from_tracking(frame, report, candidate.identity) is None:
                    raise AppError('INVALID_CANDIDATE', '후보 시각의 실제 차량 관측을 확인하지 못했습니다.')
                seen.add(frame)
            return report
        media = export_review(source, decimal_bound(start, ROUND_FLOOR), decimal_bound(end, ROUND_CEILING),
                              stage/'media', checked_observer, tools)
        if seen != anchors or media['source']['sha256'] != source_hash:
            raise AppError('INVALID_CANDIDATE', '후보와 클립의 원본 관측이 일치하지 않습니다.')
        result = {'schema': 'experimental-candidate-bundle-v1', 'status': 'complete',
                  'candidate': candidate.report(), 'context': [str(start), str(end)],
                  'lamp': lamp, 'signal_state': 'unknown' if lamp is None else lamp['signal_state'],
                  'review': 'media/review.mp4', 'original_clip': 'media/original/clip.mp4',
                  'time_mapping': 'media/review.json', 'source_sha256': source_hash,
                  'limits': ['identity unverified', 'provided lane/ROI gates', 'not a violation decision']}
        local_path(source)
        if before != _identity(source) or source_hash != _sha256(source):
            raise AppError('INPUT_CHANGED', '처리 중 원본이 변경되었습니다.')
        local_path(stage)
        local_path(output)
        if _identity(output.parent)[:2] != parent or _identity(stage)[:2] != owner:
            raise AppError('OUTPUT_CONFLICT', '출력 작업 소유권이 변경되었습니다.')
        with (stage/'candidate.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        _publish(stage, output)
        published = True
        return result
    except OSError:
        raise AppError('IO_ERROR', '저장 공간과 접근 권한을 확인하세요.') from None
    finally:
        if stage is not None and not published:
            try:
                local_path(stage)
                local_path(output.parent)
                if (stage.parent != output.parent or not stage.name.startswith('.crd-candidate-')
                        or _identity(output.parent)[:2] != parent or _identity(stage)[:2] != owner):
                    raise OSError('Ownership changed')
                shutil.rmtree(stage)
            except (OSError, AppError):
                raise AppError('CLEANUP_FAILED', '작업 소유권을 확인하지 못해 임시 결과를 보존했습니다.') from None
