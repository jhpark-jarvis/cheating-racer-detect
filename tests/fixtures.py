"""Locally generated, non-sensitive fixtures; media stays in owned temp folders.

No download, driving footage, third-party asset, random seed or optional package is
needed. F01 is 30 distinct 96x64 RGB patterns at 10 fps with one long GOP. F02
retains selected source PTS (VFR) or offsets both streams by five seconds. F03 has
a white marker and a 1 kHz / 100 ms beep at 1.5 s, sampled at 48 kHz. F05 remuxes
the directional pattern with the three non-zero right-angle rotations.
"""

from __future__ import annotations

from array import array
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import wave


WIDTH, HEIGHT, FPS, COUNT = 96, 64, 10, 30
SAMPLE_RATE = 48_000
VFR_IDS = (0, 1, 3, 6, 7, 10, 14, 15, 19, 20, 25, 29)


def run(command: list[str], *, data: bytes | None = None) -> bytes:
    """Bounded fixture command; never expose raw tool errors / local paths."""
    result = subprocess.run(
        [str(value) for value in command], input=data, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=90, check=False,
    )
    if result.returncode:
        raise AssertionError(
            f"Synthetic fixture tool {Path(command[0]).name} exited "
            f"{result.returncode}; stderr withheld ({len(result.stderr)} bytes)"
        )
    return result.stdout


def probe(toolchain, path: Path, *, frames: bool = False) -> dict:
    args = [str(toolchain.ffprobe), "-v", "error", "-of", "json"]
    if frames:
        args.extend([
            "-select_streams", "v:0", "-show_frames", "-show_entries",
            "frame=best_effort_timestamp_time,pts_time,duration_time,pkt_duration_time,key_frame",
        ])
    else:
        args.extend(["-show_streams", "-show_format"])
    args.append(str(path))
    return json.loads(run(args))


def timestamps(toolchain, path: Path) -> list[float]:
    return [
        float(item.get("best_effort_timestamp_time", item.get("pts_time")))
        for item in probe(toolchain, path, frames=True)["frames"]
    ]


def rgb_frames(toolchain, path: Path) -> list[bytes]:
    """Decoder separate from the product, including display rotation.

    Scaling to a fixed raster permits direct display-content comparison for
    portrait rotations as well as landscape clips. It deliberately does not use
    product metadata or product frame enumeration.
    """
    raw = run([
        str(toolchain.ffmpeg), "-v", "error", "-i", str(path),
        "-map", "0:v:0", "-vf", f"scale={WIDTH}:{HEIGHT}",
        "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ])
    size = WIDTH * HEIGHT * 3
    if len(raw) % size:
        raise AssertionError("Independent decode produced an incomplete RGB frame")
    return [raw[index:index + size] for index in range(0, len(raw), size)]


def pcm_samples(toolchain, path: Path) -> array:
    raw = run([
        str(toolchain.ffmpeg), "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "pipe:1",
    ])
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def frame_error(left: bytes, right: bytes) -> float:
    if len(left) != len(right):
        raise AssertionError("Decoded frame sizes differ")
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def frame_pattern(index: int) -> bytes:
    rows = []
    for y in range(HEIGHT):
        row = bytearray()
        for x in range(WIDTH):
            if x < WIDTH // 2:
                color = (24 + index * 6,) * 3
            elif index == 15:
                color = (240, 240, 240)  # Flash at presentation time 1.5 s.
            elif y < HEIGHT // 2:
                color = (210, 30, 30)
            else:
                color = (30, 190, 30)
            row.extend(color)
        rows.append(bytes(row))
    return b"".join(rows)


def generate(toolchain, root: Path) -> dict[str, Path]:
    """Generate only inside the caller's TemporaryDirectory; never overwrite."""
    root.mkdir(parents=True, exist_ok=True)
    media = {name: root / f"{name}.mp4" for name in (
        "cfr", "audio", "vfr", "offset", "rotation90", "rotation180",
        "rotation270", "unsupported", "multiple_video", "audio_delayed", "multiple_audio",
    )}
    ffmpeg = str(toolchain.ffmpeg)
    prefix = [ffmpeg, "-v", "error", "-nostdin", "-n"]
    run(prefix + [
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
        "-framerate", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-crf", "8", "-pix_fmt", "yuv420p", "-g", "100", "-keyint_min", "100",
        "-sc_threshold", "0", "-bf", "2", "-movflags", "+faststart", str(media["cfr"]),
    ], data=b"".join(frame_pattern(index) for index in range(COUNT)))

    audio_path = root / "synthetic-beep.wav"
    waveform = array("h", (
        int(20_000 * math.sin(2 * math.pi * 1000 * index / SAMPLE_RATE))
        if int(1.5 * SAMPLE_RATE) <= index < int(1.6 * SAMPLE_RATE) else 0
        for index in range(3 * SAMPLE_RATE)
    ))
    if sys.byteorder != "little":
        waveform.byteswap()
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(waveform.tobytes())
    run(prefix + [
        "-i", str(media["cfr"]), "-i", str(audio_path), "-map", "0:v:0",
        "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        str(media["audio"]),
    ])
    run(prefix + [
        "-i", str(media["cfr"]), "-itsoffset", "0.5", "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-b:a", "128k", str(media["audio_delayed"]),
    ])
    selection = "+".join(f"eq(n\\,{index})" for index in VFR_IDS)
    run(prefix + [
        "-i", str(media["cfr"]), "-vf", f"select={selection}",
        "-fps_mode", "vfr", "-c:v", "libx264", "-crf", "8", "-g", "100",
        "-sc_threshold", "0", str(media["vfr"]),
    ])
    run(prefix + [
        "-copyts", "-itsoffset", "5", "-i", str(media["audio"]),
        "-map", "0", "-c", "copy", "-avoid_negative_ts", "disabled",
        str(media["offset"]),
    ])
    for angle in (90, 180, 270):
        run(prefix + [
            "-display_rotation:v:0", str(angle), "-i", str(media["audio"]),
            "-map", "0", "-c", "copy", str(media[f"rotation{angle}"]),
        ])
        stream = next(s for s in probe(toolchain, media[f"rotation{angle}"])["streams"]
                      if s["codec_type"] == "video")
        rotations = [int(item["rotation"]) % 360 for item in stream.get("side_data_list", [])
                     if "rotation" in item]
        if rotations != [angle]:
            raise AssertionError(f"Fixture did not preserve requested display rotation {angle}")
    run(prefix + [
        "-i", str(media["cfr"]), "-c:v", "mpeg4", "-q:v", "3", str(media["unsupported"]),
    ])
    run(prefix + [
        "-i", str(media["cfr"]), "-map", "0:v:0", "-map", "0:v:0",
        "-c", "copy", str(media["multiple_video"]),
    ])
    run(prefix + [
        "-i", str(media["audio"]), "-map", "0:v:0", "-map", "0:a:0", "-map", "0:a:0",
        "-c", "copy", str(media["multiple_audio"]),
    ])
    return media
