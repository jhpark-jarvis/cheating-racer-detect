"""Explicit tool discovery and child process ownership."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .errors import AppError
from .paths import local_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Toolchain:
    ffmpeg: Path
    ffprobe: Path


def discover_tools(directory: Path | None = None) -> Toolchain:
    """Explicit directory, environment, project bundle, then PATH. No downloads."""
    selected = directory or (Path(os.environ['CRD_FFMPEG_DIR']) if os.environ.get('CRD_FFMPEG_DIR') else None)
    suffix = '.exe' if os.name == 'nt' else ''
    candidates: list[Path] = []
    if selected is not None:
        candidates.append(selected)
    else:
        candidates.append(PROJECT_ROOT / '.tools' / 'ffmpeg-9.0.2-essentials_build' / 'bin')
    for candidate in candidates:
        candidate = local_path(candidate)
        ffmpeg, ffprobe = candidate / f'ffmpeg{suffix}', candidate / f'ffprobe{suffix}'
        if ffmpeg.is_file() and ffprobe.is_file():
            return Toolchain(ffmpeg.absolute(), ffprobe.absolute())
    if selected is None:
        ffmpeg_path, ffprobe_path = shutil.which('ffmpeg'), shutil.which('ffprobe')
        if ffmpeg_path and ffprobe_path:
            return Toolchain(local_path(ffmpeg_path), local_path(ffprobe_path))
    raise AppError('DEPENDENCY_MISSING', 'FFmpeg와 ffprobe가 필요합니다. doctor 또는 --ffmpeg-dir 설정을 확인하세요.')


def run_tool(args: list[str | Path], *, error_code: str = 'EXTRACTION_FAILED', timeout: float | None = None) -> str:
    """Reap only our child on cancellation. Never put raw stderr in public errors."""
    process: subprocess.Popen[str] | None = None
    try:
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        process = subprocess.Popen(
            [str(arg) for arg in args], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', shell=False,
            creationflags=flags,
        )
        stdout, _stderr = process.communicate(timeout=timeout)
        if process.returncode != 0:
            raise AppError(error_code, '미디어 도구 실행에 실패했습니다. 입력 형식·도구 기능·출력 권한을 확인하세요.')
        return stdout
    except FileNotFoundError:
        raise AppError('DEPENDENCY_MISSING', '미디어 실행 파일을 찾을 수 없습니다.') from None
    except subprocess.TimeoutExpired:
        raise AppError(error_code, '미디어 도구 응답 시간이 초과되었습니다.') from None
    except OSError:
        raise AppError(error_code, '미디어 도구를 실행할 수 없습니다. 실행 권한과 설정을 확인하세요.') from None
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.communicate()


def doctor(toolchain: Toolchain | None = None) -> dict:
    tools = toolchain or discover_tools()
    tools = Toolchain(local_path(tools.ffmpeg), local_path(tools.ffprobe))
    ffmpeg_version = run_tool([tools.ffmpeg, '-version'], timeout=15).splitlines()[0]
    ffprobe_version = run_tool([tools.ffprobe, '-version'], timeout=15).splitlines()[0]
    encoders = run_tool([tools.ffmpeg, '-hide_banner', '-encoders'], timeout=15)
    decoders = run_tool([tools.ffmpeg, '-hide_banner', '-decoders'], timeout=15)
    required = {'libx264_encoder': ' libx264 ' in encoders,
                'aac_encoder': ' aac ' in encoders,
                'h264_decoder': ' h264 ' in decoders,
                'aac_decoder': ' aac ' in decoders}
    if not all(required.values()):
        raise AppError('CODEC_UNAVAILABLE', 'H.264/libx264 및 AAC 지원이 있는 FFmpeg 빌드가 필요합니다.')
    return {'status': 'ready', 'ffmpeg': ffmpeg_version, 'ffprobe': ffprobe_version,
            'capabilities': required, 'model_inference': 'not_configured'}
