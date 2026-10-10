"""Local MP4 decoding, exact source-frame selection, and output verification.

All user paths have already passed the service's local-path checks. No model,
network client, image reconstruction, or stream-copy path is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
from typing import BinaryIO

from .errors import AppError
from .tools import Toolchain, run_tool


@dataclass(frozen=True)
class Frame:
    pts: int
    duration: int


@dataclass(frozen=True)
class Timeline:
    time_base: Fraction
    frames: tuple[Frame, ...]

    @property
    def origin(self) -> Fraction:
        return self.frames[0].pts * self.time_base

    @property
    def end(self) -> Fraction:
        last = self.frames[-1]
        return (last.pts + last.duration) * self.time_base - self.origin


def _seconds(value: Fraction) -> str:
    """Display decimal only; rational values remain available in metadata."""
    with localcontext() as context:
        context.prec = 30
        result = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def _fraction(value: object, code: str = "INVALID_TIMELINE") -> Fraction:
    try:
        result = Fraction(str(value))
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise AppError(code, "미디어 시간 정보를 확인할 수 없습니다.") from exc
    return result


def _integer(value: object, code: str = "INVALID_TIMELINE") -> int:
    try:
        result = int(str(value))
    except (ValueError, TypeError, OverflowError) as exc:
        raise AppError(code, "미디어 정수 정보를 확인할 수 없습니다.") from exc
    return result


def _boxes(handle: BinaryIO, begin: int, end: int):
    """Iterate bounded ISO BMFF boxes without reading media payloads."""
    position = begin
    while position < end:
        if end - position < 8:
            raise AppError("UNSUPPORTED_MEDIA", "MP4 상자 구조가 불완전합니다.")
        handle.seek(position)
        header = handle.read(8)
        if len(header) != 8:
            raise AppError("UNSUPPORTED_MEDIA", "MP4 헤더를 읽을 수 없습니다.")
        size, kind = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            extended = handle.read(8)
            if len(extended) != 8:
                raise AppError("UNSUPPORTED_MEDIA", "MP4 확장 헤더가 불완전합니다.")
            size = struct.unpack(">Q", extended)[0]
            header_size = 16
        elif size == 0:
            size = end - position
        if size < header_size or position + size > end:
            raise AppError("UNSUPPORTED_MEDIA", "MP4 상자 크기가 잘못되었습니다.")
        yield kind, position + header_size, position + size
        position += size


def validate_container(source: Path) -> None:
    """Reject references before any demuxer has a chance to follow them.

    FFmpeg also receives a protocol whitelist and disabled external drefs.
    This structural check is deliberately not a codec parser.
    """
    containers = {b"moov", b"trak", b"mdia", b"minf", b"dinf"}
    brands = {b"isom", b"iso2", b"iso3", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1"}
    seen: dict[bytes, int] = {}
    dref_count = 0
    box_count = 0
    with source.open("rb") as handle:
        file_size = source.stat().st_size

        def walk(begin: int, end: int, depth: int = 0) -> None:
            nonlocal dref_count, box_count
            if depth > 8:
                raise AppError("UNSUPPORTED_MEDIA", "MP4 중첩 구조를 지원하지 않습니다.")
            for kind, payload, limit in _boxes(handle, begin, end):
                box_count += 1
                if box_count > 100_000:
                    raise AppError("UNSUPPORTED_MEDIA", "MP4 상자가 지나치게 많습니다.")
                if depth == 0:
                    seen[kind] = seen.get(kind, 0) + 1
                if kind in {b"cmov", b"rmra", b"rdrf"}:
                    raise AppError("UNSUPPORTED_MEDIA", "압축 메타데이터 또는 외부 참조 MP4는 지원하지 않습니다.")
                if kind == b"ftyp" and depth == 0:
                    if limit - payload < 8 or limit - payload > 4096:
                        raise AppError("UNSUPPORTED_MEDIA", "MP4 형식 식별자가 잘못되었습니다.")
                    handle.seek(payload)
                    content = handle.read(limit - payload)
                    compatible = {content[0:4]} | {content[i:i + 4] for i in range(8, len(content), 4)}
                    if not compatible & brands:
                        raise AppError("UNSUPPORTED_MEDIA", "지원하는 MP4 형식이 아닙니다.")
                if kind in containers:
                    walk(payload, limit, depth + 1)
                elif kind == b"dref":
                    handle.seek(payload)
                    prefix = handle.read(8)
                    if len(prefix) != 8 or prefix[:4] != b"\x00" * 4:
                        raise AppError("UNSUPPORTED_MEDIA", "MP4 데이터 참조가 잘못되었습니다.")
                    count = struct.unpack(">I", prefix[4:])[0]
                    if not 0 < count <= 100_000:
                        raise AppError("UNSUPPORTED_MEDIA", "MP4 데이터 참조 개수가 잘못되었습니다.")
                    actual_count = 0
                    for entry_type, entry_payload, entry_end in _boxes(handle, payload + 8, limit):
                        actual_count += 1
                        if actual_count > count:
                            raise AppError("UNSUPPORTED_MEDIA", "MP4 데이터 참조 개수가 잘못되었습니다.")
                        handle.seek(entry_payload)
                        flags = handle.read(4)
                        if entry_type != b"url " or flags != b"\x00\x00\x00\x01" or entry_end - entry_payload != 4:
                            raise AppError("UNSUPPORTED_MEDIA", "외부 파일·URL을 참조하는 MP4는 허용되지 않습니다.")
                    if actual_count != count:
                        raise AppError("UNSUPPORTED_MEDIA", "MP4 데이터 참조 개수가 잘못되었습니다.")
                    dref_count += 1

        walk(0, file_size)
    if seen.get(b"ftyp") != 1 or seen.get(b"moov") != 1 or not seen.get(b"mdat") or not dref_count:
        raise AppError("UNSUPPORTED_MEDIA", "자체 미디어를 포함한 단일 MP4 파일이 필요합니다.")


def _input_options() -> list[str]:
    return ["-protocol_whitelist", "file", "-f", "mov", "-enable_drefs", "0", "-use_absolute_path", "0", "-ignore_chapters", "1"]


def _json_tool(args: list[str | Path], code: str) -> dict:
    try:
        result = json.loads(run_tool(args, error_code=code))
    except (ValueError, TypeError) as exc:
        raise AppError(code, "미디어 도구의 결과 형식이 올바르지 않습니다.") from exc
    if not isinstance(result, dict) or "error" in result:
        raise AppError(code, "미디어 도구가 유효한 결과를 반환하지 않았습니다.")
    return result


def _probe(source: Path, toolchain: Toolchain, code: str) -> dict:
    return _json_tool([
        toolchain.ffprobe, "-v", "error", *_input_options(), "-show_streams", "-show_format", "-of", "json", source,
    ], code)


def _rotation(stream: dict, code: str = "UNSUPPORTED_MEDIA") -> int:
    angle = Fraction(0)
    for item in stream.get("side_data_list", []):
        if "rotation" in item:
            angle = _fraction(item["rotation"], code)
        matrix = item.get("displaymatrix")
        if matrix:
            try:
                rows = [line.split(":", 1)[1].split() for line in matrix.strip().splitlines()]
                a, b, c, d = int(rows[0][0]), int(rows[0][1]), int(rows[1][0]), int(rows[1][1])
                if a * d - b * c != 65536**2 or any(value not in {-65536, 0, 65536} for value in (a, b, c, d)):
                    raise ValueError("not a pure cardinal rotation")
            except (ValueError, IndexError) as exc:
                raise AppError(code, "반사·비표준 화면 변환은 지원하지 않습니다.") from exc
    if angle % 90:
        raise AppError(code, "90도 단위 회전만 지원합니다.")
    return int(angle) % 360


def _profile(info: dict, code: str = "UNSUPPORTED_MEDIA") -> tuple[dict, dict | None]:
    streams = info.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) > 1 or len(streams) != len(videos) + len(audios):
        raise AppError(code, "H.264 비디오 하나와 AAC 오디오 최대 하나인 MP4가 필요합니다.")
    video = videos[0]
    if video.get("codec_name") != "h264" or video.get("pix_fmt") not in {"yuv420p", "yuvj420p"}:
        raise AppError(code, "현재 H.264 8-bit 4:2:0 비디오만 지원합니다.")
    if video.get("color_transfer") in {"smpte2084", "arib-std-b67"} or video.get("color_primaries") == "bt2020":
        raise AppError(code, "HDR 또는 광색역 비디오는 현재 지원하지 않습니다.")
    if video.get("disposition", {}).get("attached_pic"):
        raise AppError(code, "정지 표지 이미지는 입력 비디오로 사용할 수 없습니다.")
    if _integer(video.get("width", 0), code) <= 0 or _integer(video.get("height", 0), code) <= 0:
        raise AppError(code, "비디오 크기를 확인할 수 없습니다.")
    _rotation(video, code)
    audio = audios[0] if audios else None
    if audio:
        if audio.get("codec_name") != "aac" or _integer(audio.get("sample_rate", 0), code) <= 0:
            raise AppError(code, "오디오가 있는 경우 유효한 AAC 스트림이어야 합니다.")
        if _integer(audio.get("channels", 0), code) <= 0:
            raise AppError(code, "오디오 채널 정보를 확인할 수 없습니다.")
    return video, audio


def _timeline(source: Path, video: dict, toolchain: Toolchain, code: str) -> Timeline:
    base = _fraction(video.get("time_base"), code)
    if base <= 0 or base.denominator > 2_147_483_647:
        raise AppError(code, "비디오 time base를 지원할 수 없습니다.")
    data = _json_tool([
        toolchain.ffprobe, "-v", "error", "-err_detect", "explode", *_input_options(),
        "-select_streams", "v:0", "-show_frames", "-show_entries",
        "frame=pts,duration,pkt_duration,width,height,pix_fmt,color_transfer,color_primaries", "-of", "json", source,
    ], code)
    frames: list[Frame] = []
    for value in data.get("frames", []):
        pts = _integer(value.get("pts"), code)
        if abs(pts) >= 2**52 or (frames and pts <= frames[-1].pts):
            raise AppError(code, "비디오 표시 시각은 유효한 범위에서 순서대로 증가해야 합니다.")
        if value.get("width") != video["width"] or value.get("height") != video["height"] or value.get("pix_fmt") != video["pix_fmt"]:
            raise AppError("UNSUPPORTED_MEDIA" if code != "OUTPUT_INVALID" else code, "파일 중간의 해상도·픽셀 형식 변경은 지원하지 않습니다.")
        if value.get("color_transfer") in {"smpte2084", "arib-std-b67"} or value.get("color_primaries") == "bt2020":
            raise AppError("UNSUPPORTED_MEDIA" if code != "OUTPUT_INVALID" else code, "HDR 프레임은 현재 지원하지 않습니다.")
        duration = _integer(value.get("duration", value.get("pkt_duration", 0)), code)
        if duration < 0:
            raise AppError(code, "음수 프레임 길이를 처리할 수 없습니다.")
        frames.append(Frame(pts, duration))
    if not frames:
        raise AppError(code, "표시 시각이 있는 비디오 프레임이 없습니다.")
    for index in range(len(frames) - 1):
        if frames[index].duration == 0:
            frames[index] = Frame(frames[index].pts, frames[index + 1].pts - frames[index].pts)
    if frames[-1].duration <= 0:
        stream_end = _integer(video.get("start_pts"), code) + _integer(video.get("duration_ts"), code)
        duration = stream_end - frames[-1].pts
        if duration <= 0:
            raise AppError(code, "마지막 프레임의 표시 끝을 확인할 수 없습니다.")
        frames[-1] = Frame(frames[-1].pts, duration)
    return Timeline(base, tuple(frames))


def select_frames(timeline: Timeline, start: Fraction, end: Fraction) -> tuple[Frame, ...]:
    if start < 0 or end <= start or end > timeline.end:
        raise AppError("INVALID_RANGE", "0 ≤ 시작 < 끝 ≤ 비디오 관측 끝인 구간을 지정하세요.")
    frames = tuple(f for f in timeline.frames if start <= f.pts * timeline.time_base - timeline.origin < end)
    if not frames:
        raise AppError("EMPTY_INTERVAL", "요청한 구간에 표시를 시작하는 비디오 프레임이 없습니다.")
    return frames


def _audio_intervals(source: Path, audio: dict, toolchain: Toolchain, code: str) -> list[tuple[Fraction, Fraction]]:
    """Decoded samples, excluding skip-sample priming reported by the decoder."""
    base = _fraction(audio.get("time_base"), code)
    sample_rate = _integer(audio.get("sample_rate"), code)
    if base <= 0 or sample_rate <= 0:
        raise AppError(code, "오디오 시간 기준을 확인할 수 없습니다.")
    data = _json_tool([
        toolchain.ffprobe, "-v", "error", "-err_detect", "explode", *_input_options(),
        "-select_streams", "a:0", "-show_frames", "-show_entries", "frame=pts,nb_samples", "-of", "json", source,
    ], code)
    intervals = []
    for frame in data.get("frames", []):
        begin = _integer(frame.get("pts"), code) * base
        sample_count = _integer(frame.get("nb_samples"), code)
        if sample_count <= 0 or (intervals and begin < intervals[-1][1] - Fraction(1, sample_rate)):
            raise AppError(code, "오디오 표시 시각이 겹치거나 샘플 수가 잘못되었습니다.")
        intervals.append((begin, begin + Fraction(sample_count, sample_rate)))
    return intervals


def _check_codecs(toolchain: Toolchain, audio: bool) -> None:
    encoders = run_tool([toolchain.ffmpeg, "-hide_banner", "-encoders"], error_code="CODEC_UNAVAILABLE")
    if "libx264" not in encoders or (audio and not any(line.split()[1:2] == ["aac"] for line in encoders.splitlines())):
        raise AppError("CODEC_UNAVAILABLE", "FFmpeg에 libx264와 필요한 AAC 인코더가 있어야 합니다.")


def _full_decode(source: Path, toolchain: Toolchain, code: str) -> None:
    run_tool([
        toolchain.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-xerror", "-err_detect", "explode",
        *_input_options(), "-i", source, "-map", "0:v:0", "-map", "0:a:0?", "-f", "null", "-",
    ], error_code=code)


def extract(source: Path, destination: Path, start: Decimal, end: Decimal, toolchain: Toolchain) -> dict:
    """Re-encode a half-open source interval into an unpublished job folder."""
    if not start.is_finite() or not end.is_finite():
        raise AppError("INVALID_RANGE", "시작·끝은 유한한 초 단위 숫자여야 합니다.")
    begin, finish = Fraction(start), Fraction(end)
    from .source import inspect_source
    video, audio, timeline = inspect_source(source, toolchain)
    selected = select_frames(timeline, begin, finish)
    _check_codecs(toolchain, audio is not None)
    absolute_start = timeline.origin + begin
    absolute_end = timeline.origin + finish
    source_audio = _audio_intervals(source, audio, toolchain, "INVALID_TIMELINE") if audio else []
    # AAC frames can straddle the request boundaries; atrim selects individual
    # samples. Samples before a delayed track begins are not synthesized.
    audio_sample_rate = _integer(audio["sample_rate"]) if audio else 1
    audio_trim_start = Fraction(math.ceil(absolute_start * audio_sample_rate), audio_sample_rate)
    audio_trim_end = Fraction(math.ceil(absolute_end * audio_sample_rate), audio_sample_rate)
    selected_audio = [(max(a, audio_trim_start), min(b, audio_trim_end)) for a, b in source_audio if a < audio_trim_end and b > audio_trim_start]
    start_tick = math.ceil(absolute_start / timeline.time_base)
    end_tick = math.ceil(absolute_end / timeline.time_base)
    # Retain the requested origin, not STARTPTS, so a late audio start and the
    # first selected video's fractional offset remain on the same clock.
    origin_expression = f"({absolute_start.numerator}/{absolute_start.denominator})/TB"
    video_filter = f"trim=start_pts={start_tick}:end_pts={end_tick},setpts=PTS-{origin_expression}"
    args: list[str | Path] = [
        toolchain.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-xerror", "-err_detect", "explode", "-copyts",
        *_input_options(), "-i", source, "-map", f"0:{video['index']}", "-vf", video_filter,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-bf", "0",
        "-fps_mode:v", "passthrough", "-enc_time_base:v", f"{timeline.time_base.numerator}:{timeline.time_base.denominator}",
        "-video_track_timescale", str(timeline.time_base.denominator),
    ]
    if audio:
        sample_rate = _integer(audio["sample_rate"])
        audio_start = math.ceil(absolute_start * sample_rate)
        audio_end = math.ceil(absolute_end * sample_rate)
        args.extend([
            "-map", f"0:{audio['index']}", "-af",
            f"asettb=1/{sample_rate},atrim=start_pts={audio_start}:end_pts={audio_end},asetpts=PTS-{origin_expression}",
            "-c:a", "aac", "-b:a", "192k", "-ar", str(sample_rate),
        ])
    else:
        args.append("-an")
    movie_timescale = math.lcm(timeline.time_base.denominator, _integer(audio["sample_rate"]) if audio else 1)
    if movie_timescale > 2_147_483_647:
        movie_timescale = timeline.time_base.denominator
    args.extend([
        "-map_metadata", "-1", "-map_chapters", "-1", "-metadata:s:v:0", "rotate=0",
        "-avoid_negative_ts", "disabled", "-use_editlist", "1", "-movie_timescale", str(movie_timescale),
        "-movflags", "+faststart", "-f", "mp4", destination,
    ])
    # A complete source decode catches damage even outside the selected range.
    _full_decode(source, toolchain, "EXTRACTION_FAILED")
    run_tool(args, error_code="EXTRACTION_FAILED")
    _full_decode(destination, toolchain, "OUTPUT_INVALID")
    output = _probe(destination, toolchain, "OUTPUT_INVALID")
    out_video, out_audio = _profile(output, "OUTPUT_INVALID")
    out_timeline = _timeline(destination, out_video, toolchain, "OUTPUT_INVALID")
    if len(out_timeline.frames) != len(selected):
        raise AppError("OUTPUT_INVALID", "출력 프레임 수가 선택한 원본 프레임 수와 다릅니다.")
    tolerance = max(timeline.time_base, out_timeline.time_base, Fraction(1, movie_timescale))
    max_error = Fraction(0)
    for original, actual in zip(selected, out_timeline.frames):
        expected_time = original.pts * timeline.time_base - absolute_start
        actual_time = actual.pts * out_timeline.time_base
        error = abs(expected_time - actual_time)
        max_error = max(max_error, error)
        if error > tolerance:
            raise AppError("OUTPUT_INVALID", "출력 프레임의 표시 시각이 원본 시간 매핑과 다릅니다.")
    source_last_duration = selected[-1].duration * timeline.time_base
    output_last_duration = out_timeline.frames[-1].duration * out_timeline.time_base
    last_duration_error = abs(output_last_duration - source_last_duration)
    if last_duration_error > tolerance:
        raise AppError("OUTPUT_INVALID", "마지막 출력 프레임의 표시 길이가 원본과 다릅니다.")
    rotation = _rotation(video)
    expected_dimensions = (video["height"], video["width"]) if rotation in {90, 270} else (video["width"], video["height"])
    if (out_video["width"], out_video["height"]) != expected_dimensions or _rotation(out_video, "OUTPUT_INVALID") != 0:
        raise AppError("OUTPUT_INVALID", "출력의 크기 또는 회전 정보가 예상과 다릅니다.")
    audio_expected = bool(selected_audio)
    if bool(out_audio) != audio_expected:
        raise AppError("OUTPUT_INVALID", "요청 구간의 오디오 유무와 출력이 다릅니다.")
    audio_timing: dict = {"status": "not_applicable", "reason": "no_audio_samples_in_interval"}
    if out_audio:
        output_audio = _audio_intervals(destination, out_audio, toolchain, "OUTPUT_INVALID")
        if not output_audio:
            raise AppError("OUTPUT_INVALID", "출력 오디오를 샘플로 디코딩할 수 없습니다.")
        expected_audio_start = selected_audio[0][0] - absolute_start
        expected_audio_end = selected_audio[-1][1] - absolute_start
        audio_start_error = abs(output_audio[0][0] - expected_audio_start)
        audio_end_error = abs(output_audio[-1][1] - expected_audio_end)
        # AAC LC encodes 1024 samples per block. End padding / edit lists are
        # measured separately from video and never inferred from MP4 duration.
        audio_tolerance = Fraction(1024, audio_sample_rate) + Fraction(1, movie_timescale)
        if max(audio_start_error, audio_end_error) > audio_tolerance:
            raise AppError("OUTPUT_INVALID", "출력 오디오의 시작·끝이 원본 공통 시간 기준과 다릅니다.")
        audio_timing = {
            "status": "verified", "expected_first_sample_seconds": _seconds(expected_audio_start),
            "actual_first_sample_seconds": _seconds(output_audio[0][0]),
            "expected_sample_end_seconds": _seconds(expected_audio_end), "actual_sample_end_seconds": _seconds(output_audio[-1][1]),
            "start_error_seconds": _seconds(audio_start_error), "end_error_seconds": _seconds(audio_end_error),
            "tolerance_seconds": _seconds(audio_tolerance),
        }
    output_first = out_timeline.origin
    output_last = out_timeline.frames[-1].pts * out_timeline.time_base
    first_normalized = selected[0].pts * timeline.time_base - timeline.origin
    last_normalized = selected[-1].pts * timeline.time_base - timeline.origin
    warnings = ["검토용 재인코딩 파생물이며 원본 바이트와 같지 않습니다."]
    if audio:
        warnings.append("AAC 인코더의 priming/padding은 MP4 편집 목록으로 표현되며 재생기별 검증은 별도입니다.")
    if selected[-1].duration * timeline.time_base + last_normalized > finish:
        warnings.append("마지막 포함 프레임은 요청 끝 이전에 시작하며 표시 길이가 요청 끝을 넘을 수 있습니다.")
    return {
        "request": {"start_seconds": str(start), "end_seconds": str(end), "interval": "[start,end)", "time_origin": "first_source_video_pts"},
        "source_media": {
            "video_stream_index": video["index"], "audio_stream_index": audio["index"] if audio else None,
            "time_base": str(timeline.time_base), "first_pts": timeline.frames[0].pts,
            "first_pts_seconds": _seconds(timeline.origin), "first_pts_rational": str(timeline.origin),
            "observed_end_seconds": _seconds(timeline.end), "observed_end_rational": str(timeline.end),
            "width": video["width"], "height": video["height"], "rotation_degrees": rotation, "frame_count": len(timeline.frames),
        },
        "selection": {
            "frame_count": len(selected), "first_source_pts": selected[0].pts, "last_source_pts": selected[-1].pts,
            "first_normalized_seconds": _seconds(first_normalized), "last_normalized_seconds": _seconds(last_normalized),
            "first_normalized_rational": str(first_normalized), "last_normalized_rational": str(last_normalized),
            "last_source_frame_duration_seconds": _seconds(source_last_duration),
            "source_pts": [frame.pts for frame in selected],
        },
        "output_media": {
            "video_time_base": str(out_timeline.time_base), "first_pts_seconds": _seconds(output_first),
            "last_pts_seconds": _seconds(output_last), "duration_seconds": str(output.get("format", {}).get("duration", "unknown")),
            "video_display_end_seconds": _seconds(out_timeline.end + out_timeline.origin),
            "last_frame_duration_seconds": _seconds(out_timeline.frames[-1].duration * out_timeline.time_base),
            "width": out_video["width"], "height": out_video["height"], "rotation_degrees": 0,
            "audio_present": bool(out_audio), "audio_start_seconds": out_audio.get("start_time") if out_audio else None,
            "frame_count": len(out_timeline.frames), "video_pts": [frame.pts for frame in out_timeline.frames],
        },
        "extraction": {"method": "decode-trim-reencode", "video_codec": "libx264", "audio_codec": "aac" if out_audio else None,
                       "output_origin_source_seconds": _seconds(absolute_start), "output_origin_source_rational": str(absolute_start),
                       "rotation_applied_to_pixels": True, "stream_copy": False},
        "verification": {"source_full_decode": True, "output_full_decode": True, "frame_count_matches": True,
                         "frame_timestamps_match": True, "max_video_timestamp_error_seconds": _seconds(max_error),
                         "last_frame_duration_matches": True, "last_frame_duration_error_seconds": _seconds(last_duration_error),
                         "video_timestamp_tolerance_seconds": _seconds(tolerance), "rotation_and_dimensions_match": True,
                         "audio_presence_matches": True, "frame_content_identity": "synthetic_fixture_tests_only",
                         "audio_timing": audio_timing, "av_content_sync": "synthetic_fixture_tests_only"},
        "warnings": warnings,
    }
