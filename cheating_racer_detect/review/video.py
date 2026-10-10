"""Bounded Windows review export; detector/model loading remains caller-owned."""

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import tempfile

from ..errors import AppError
from ..paths import destination_path, local_path, source_path
from ..service import _identity, _publish, _sha256, create_clip
from ..source import inspect_source, source_job
from ..tools import discover_tools, run_tool
from ..tracking import Frame
from .records import summarize
from .overlay import render

MAX_FRAMES = 120
MAX_AXIS = 1024


def interval(start, end):
    try:
        start, end = Decimal(str(start)), Decimal(str(end))
    except (InvalidOperation, ValueError):
        raise AppError('INVALID_RANGE', '시작과 끝은 유한한 초 단위 숫자여야 합니다.') from None
    if not start.is_finite() or not end.is_finite() or start < 0 or start >= end:
        raise AppError('INVALID_RANGE', '0 <= 시작 < 끝인 시간을 지정하세요.')
    return start, end


def timestamp_expression(ticks):
    """Integer output PTS lookup, not an assumed frame rate or decimal seconds."""
    expression = str(ticks[-1])
    for index in reversed(range(len(ticks)-1)):
        expression = f'if(eq(N,{index}),{ticks[index]},{expression})'
    return expression


def prepare(source, start, end, tools):
    from ..media import _rotation, select_frames

    video, _audio, timeline = inspect_source(source, tools, max_axis=MAX_AXIS)
    selected = select_frames(timeline, Fraction(start), Fraction(end))
    if (len(selected) > MAX_FRAMES or timeline.time_base.denominator > 1_000_000
            or timeline.time_base.numerator > 1_000_000):
        raise AppError('REVIEW_LIMIT', '검토 구간은 최대 120프레임과 제한된 time base를 지원합니다.')
    first = next(i for i, f in enumerate(timeline.frames) if f.pts == selected[0].pts)
    return video, timeline, selected, first, _rotation(video, 'UNSUPPORTED_MEDIA')


def build(source, start, end, stage, observe, tools, source_hash):
    """Same selected decoded pixels -> callback -> validated, owned review records."""
    from ..media import _full_decode, _input_options, _probe, _profile, _timeline

    video, timeline, selected, first, rotation = prepare(source, start, end, tools)
    original = create_clip(source, start, end, stage/'original', tools)
    if original['source']['sha256'] != source_hash:
        raise AppError('INPUT_CHANGED', '처리 중 원본이 변경되었습니다.')
    raw = stage/'source.bgr'
    run_tool([tools.ffmpeg, '-nostdin', '-v', 'error', '-xerror', '-err_detect', 'explode',
              *_input_options(), '-noautorotate', '-i', source, '-map', '0:v:0',
              '-vf', f'select=between(n\\,{first}\\,{first+len(selected)-1})',
              '-fps_mode', 'passthrough', '-an', '-pix_fmt', 'bgr24', '-f', 'rawvideo', raw], timeout=120)
    size = video['width']*video['height']*3
    if raw.stat().st_size != size*len(selected):
        raise AppError('OUTPUT_INVALID', '선택 프레임과 디코딩 결과가 일치하지 않습니다.')
    frames, reports, summaries = [], [], []
    height = 0
    scale = min(4, max(1, 640//video['width']))
    width = max(880, video['width']*scale+480)
    with raw.open('rb') as stream:
        for index, item in enumerate(selected):
            frame = Frame(source_hash, video['index'], first+index, item.pts,
                          timeline.time_base, timeline.frames[0].pts,
                          video['width'], video['height'], rotation)
            pixels = stream.read(size)
            report = json.loads(json.dumps(observe(frame, pixels), allow_nan=False))
            summary = summarize(frame, report)
            height = max(height, 186+max(video['height']*scale, 26+18*len(summary['annotations'])))
            frames.append(frame)
            reports.append(report)
            summaries.append(summary)
    width, height = (width+1)//2*2, (height+1)//2*2
    # Stage only one frame in RAM; fixed canvas prevents geometry reinitialization.
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        if error.name != 'numpy':
            raise
        raise AppError('DEPENDENCY_MISSING', '검토 렌더링에는 tracking 선택 의존성이 필요합니다.') from error
    rendered = stage/'review.bgr'
    with raw.open('rb') as stream, rendered.open('xb') as destination:
        for frame, report in zip(frames, reports):
            image, _summary = render(frame, stream.read(size), report)
            canvas = np.full((height, width, 3), 22, np.uint8)
            canvas[:image.shape[0], :image.shape[1]] = image
            destination.write(canvas.tobytes())
    denominator = timeline.time_base.denominator
    ticks = [(f.pts-selected[0].pts)*timeline.time_base.numerator for f in selected]
    last_ticks = selected[-1].duration*timeline.time_base.numerator
    if last_ticks <= 0 or last_ticks > 2**31-1 or ticks[-1] >= 2**52:
        raise AppError('INVALID_TIMELINE', '마지막 프레임 길이를 확인할 수 없습니다.')
    expression = timestamp_expression(ticks)
    rate = Fraction(denominator, last_ticks)
    run_tool([tools.ffmpeg, '-nostdin', '-v', 'error', '-xerror', '-protocol_whitelist', 'file',
              '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', f'{width}x{height}',
              '-framerate', str(rate), '-i', rendered,
              '-vf', f"settb=1/{denominator},setpts='{expression}'", '-an',
              '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-bf', '0',
              '-fps_mode', 'passthrough', '-enc_time_base', f'1:{denominator}',
              '-bsf:v', f"setts=duration='if(eq(N,{len(ticks)-1}),{last_ticks},DURATION)'",
              '-video_track_timescale', str(denominator), '-movflags', '+faststart',
              '-f', 'mp4', stage/'review.mp4'], timeout=120)
    output_video, audio = _profile(_probe(stage/'review.mp4', tools, 'OUTPUT_INVALID'), 'OUTPUT_INVALID')
    output_time = _timeline(stage/'review.mp4', output_video, tools, 'OUTPUT_INVALID')
    actual = [f.pts*output_time.time_base for f in output_time.frames]
    if (audio is not None or (output_video['width'], output_video['height']) != (width, height)
            or actual != [Fraction(t, denominator) for t in ticks]
            or output_time.frames[-1].duration*output_time.time_base != Fraction(last_ticks, denominator)):
        raise AppError('OUTPUT_INVALID', '검토 영상의 프레임·시각·크기·길이가 일치하지 않습니다.')
    _full_decode(stage/'review.mp4', tools, 'OUTPUT_INVALID')
    mapping = [{'source_frame': f.report(), 'review_pts': t,
                'review_time': str(Fraction(t, denominator)), 'evidence': s}
               for f, t, s in zip(frames, ticks, summaries)]
    return {'schema': 'experimental-review-video-v1', 'status': 'complete',
            'source': original['source'], 'request': original['request'],
            'original_clip': 'original/clip.mp4', 'original_metadata': 'original/result.json',
            'review': {'file': 'review.mp4', 'sha256': _sha256(stage/'review.mp4'),
                       'frame_count': len(frames), 'canvas_size': [width, height],
                       'time_base': f'1/{denominator}', 'source_time_at_review_zero': str(frames[0].time),
                       'last_frame_duration': str(Fraction(last_ticks, denominator)),
                       'view': 'coded_raster_no_display_rotation', 'audio': 'none'},
            'frames': mapping, 'checks': {'source_unchanged': True, 'review_decoded': True,
                'exact_pts': True, 'original_clip_verified': True},
            'limits': ['derived preview, not original evidence', 'identity unverified',
                       'prediction is not observation', 'no lane/lamp/violation decision',
                       'last frame may extend beyond request end']}


@source_job
def export_review(input_path, start, end, output_path, observe, toolchain=None):
    """Observe(Frame, immutable BGR bytes) -> current Tracker report.

    Windows, <=120 selected frames and source axes <=1024. No detector loading.
    Callback owns tracking state and must reset it after a failed/cancelled job.
    Writes review.mp4/review.json + verified original/clip.mp4/result.json together.
    Review time starts at the first selected frame; exact source mapping is kept.
    """
    start, end = interval(start, end)
    source = source_path(input_path)
    output = destination_path(output_path, source)
    if not callable(observe):
        raise AppError('INVALID_REVIEW', '현재 프레임의 추적 기록을 제공하는 callback이 필요합니다.')
    tools = toolchain or discover_tools()
    stage = None
    owner = None
    published = False
    parent = _identity(output.parent)[:2]
    try:
        before, source_hash = _identity(source), _sha256(source)
        stage = Path(tempfile.mkdtemp(prefix='.crd-review-', suffix='.partial', dir=output.parent))
        owner = _identity(stage)[:2]
        result = build(source, start, end, stage, observe, tools, source_hash)
        local_path(source)
        if before != _identity(source) or source_hash != _sha256(source):
            raise AppError('INPUT_CHANGED', '처리 중 원본이 변경되었습니다.')
        local_path(output)
        local_path(stage)
        if _identity(output.parent)[:2] != parent or _identity(stage)[:2] != owner:
            raise AppError('OUTPUT_CONFLICT', '출력 또는 임시 작업 소유권이 변경되었습니다.')
        # Delete intermediates only after checking the actual stage ownership.
        for name in ('source.bgr', 'review.bgr'):
            intermediate = local_path(stage/name)
            if intermediate.exists():
                if _identity(stage)[:2] != owner or _identity(output.parent)[:2] != parent:
                    raise AppError('OUTPUT_CONFLICT', '임시 작업 소유권이 변경되었습니다.')
                intermediate.unlink()
        with (stage/'review.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        _publish(stage, output)
        published = True
        return result
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise AppError('INVALID_REVIEW', '검토 기록·좌표·시각·현재 프레임 계약을 확인하세요.') from None
    except OSError:
        raise AppError('IO_ERROR', '저장 공간과 파일 접근 권한을 확인하세요.') from None
    finally:
        if stage is not None and not published:
            try:
                local_path(stage)
                local_path(output.parent)
                if (_identity(output.parent)[:2] != parent or _identity(stage)[:2] != owner
                        or stage.parent != output.parent or not stage.name.startswith('.crd-review-')):
                    raise OSError('Ownership changed')
                shutil.rmtree(stage)
            except (OSError, AppError):
                raise AppError('CLEANUP_FAILED', '작업 소유권·정리를 확인하지 못했습니다. 이번 임시 폴더를 확인하세요.') from None
