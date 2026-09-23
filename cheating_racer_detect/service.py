"""Transactional clip workflow, independent of the CLI and future UI."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import __version__
from .errors import AppError
from .paths import destination_path, local_path, source_path
from .tools import Toolchain, discover_tools, doctor


def _identity(path: Path) -> tuple[int, int, int, int]:
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def _sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _publish(temporary: Path, output: Path) -> None:
    """Windows rename fails when a target exists. Other hosts are not supported yet."""
    if output.exists():
        raise AppError('OUTPUT_EXISTS', '출력 경로가 다른 작업에 의해 생성되었습니다.')
    if os.name != 'nt':
        raise AppError('UNSUPPORTED_PLATFORM', '현재 결과 확정 경로는 Windows에서만 검증되었습니다.')
    try:
        temporary.rename(output)
    except FileExistsError:
        raise AppError('OUTPUT_EXISTS', '기존 결과는 교체하지 않습니다.') from None
    except OSError:
        raise AppError('IO_ERROR', '결과를 확정하지 못했습니다. 출력 권한과 다른 작업을 확인하세요.') from None


def create_clip(input_path: str | Path, start: Decimal, end: Decimal,
                output_path: str | Path, toolchain: Toolchain | None = None) -> dict:
    from .media import extract

    try:
        start, end = Decimal(str(start)), Decimal(str(end))
    except InvalidOperation:
        raise AppError('INVALID_RANGE', '시작과 끝은 유한한 초 단위 숫자여야 합니다.') from None
    if not start.is_finite() or not end.is_finite() or start < 0 or start >= end:
        raise AppError('INVALID_RANGE', '0 <= 시작 < 끝인 유한한 시간을 지정하세요.')
    source = source_path(input_path)
    output = destination_path(output_path, source)
    tools = toolchain or discover_tools()
    capabilities = doctor(tools)
    temporary: Path | None = None
    temporary_identity: tuple[int, int] | None = None
    parent_identity: tuple[int, int] | None = None
    published = False
    try:
        before = _identity(source)
        source_hash = _sha256(source)
        parent_identity = _identity(output.parent)[:2]
        temporary = Path(tempfile.mkdtemp(prefix='.crd-', suffix='.partial', dir=output.parent))
        temporary_identity = _identity(temporary)[:2]
        media_result = extract(source, temporary / 'clip.mp4', start, end, tools)
        local_path(source)
        if before != _identity(source) or _sha256(source) != source_hash:
            raise AppError('INPUT_CHANGED', '처리 중 원본이 변경되었습니다. 변경이 끝난 뒤 다시 시도하세요.')
        local_path(output)
        if _identity(output.parent)[:2] != parent_identity:
            raise AppError('OUTPUT_CONFLICT', '처리 중 출력의 상위 디렉터리가 변경되었습니다.')
        local_path(temporary)
        if _identity(temporary)[:2] != temporary_identity:
            raise AppError('OUTPUT_CONFLICT', '처리 중 작업 임시 디렉터리가 변경되었습니다.')
        result = {
            'schema_version': 1, 'status': 'complete', 'application_version': __version__,
            'source': {'name': source.name, 'size_bytes': before[2], 'sha256': source_hash},
            'request': {'start_seconds': str(start), 'end_seconds': str(end)},
            'media': media_result,
            'toolchain': {'ffmpeg': capabilities['ffmpeg'], 'ffprobe': capabilities['ffprobe']},
            'checks': {'source_unchanged': True, 'output_decoded': True,
                       'network_inputs': 'rejected', 'derived_review_clip': True},
        }
        with (temporary / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        _publish(temporary, output)
        published = True
        return result
    except PermissionError:
        raise AppError('IO_ERROR', '파일 접근 권한을 확인하세요.') from None
    except OSError:
        raise AppError('IO_ERROR', '파일 처리에 실패했습니다. 저장 공간과 권한을 확인하세요.') from None
    finally:
        if temporary is not None and not published:
            # Only this invocation's directory, never the user-selected output or source.
            try:
                if temporary.parent != output.parent or not temporary.name.startswith('.crd-'):
                    raise OSError('Unexpected temporary directory')
                local_path(temporary)
                if _identity(output.parent)[:2] != parent_identity or _identity(temporary)[:2] != temporary_identity:
                    raise OSError('Directory ownership changed; leave all files intact')
                shutil.rmtree(temporary)
            except FileNotFoundError:
                raise AppError('CLEANUP_FAILED', f'임시 작업 경로가 사라지거나 이동되어 정리를 확인하지 못했습니다: {temporary.name}. 작업 소유 경로를 확인하세요.') from None
            except (OSError, AppError):
                raise AppError('CLEANUP_FAILED', f'소유권 또는 접근 상태를 확인할 수 없어 임시 결과를 보존했습니다: {temporary.name}. 이동된 경로도 확인하고 작업 소유 자료만 정리하세요.') from None
