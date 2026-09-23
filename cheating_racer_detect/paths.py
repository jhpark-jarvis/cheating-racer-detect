"""Reject remote and indirect paths before opening any media or output."""

from __future__ import annotations

import ctypes
import ntpath
import os
import stat
from pathlib import Path

from .errors import AppError


def _drive_type(anchor: str) -> int:
    return ctypes.windll.kernel32.GetDriveTypeW(anchor)  # type: ignore[attr-defined]


def local_path(value: str | Path) -> Path:
    raw = os.fspath(value)
    if not raw or '\x00' in raw or raw.startswith(('\\\\', '//')) or '://' in raw:
        raise AppError('INVALID_ARGUMENT', '로컬 일반 파일 경로만 사용할 수 있습니다.')
    # Reject drive-relative paths and NT device/ADS syntax even on a non-Windows test host.
    drive, tail = ntpath.splitdrive(raw)
    if drive and (not tail.startswith(('\\', '/')) or len(drive) != 2 or drive[1] != ':'):
        raise AppError('INVALID_ARGUMENT', '절대 경로 또는 일반 상대 경로를 사용하세요.')
    if ':' in tail or '..' in raw.replace('\\', '/').split('/'):
        raise AppError('INVALID_ARGUMENT', '상위 경로 이동이나 대체 데이터 스트림은 지원하지 않습니다.')
    path = Path(os.path.abspath(raw))  # lexical only; do not resolve symlinks
    if os.name == 'nt':
        if os.path.isreserved(str(path)):
            raise AppError('INVALID_ARGUMENT', 'Windows 예약 경로는 사용할 수 없습니다.')
        if _drive_type(path.anchor) not in {2, 3, 6}:
            raise AppError('INVALID_ARGUMENT', '네트워크 드라이브와 확인되지 않은 드라이브는 지원하지 않습니다.')
    for ancestor in [*reversed(path.parents), path]:
        try:
            metadata = ancestor.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise AppError('IO_ERROR', '경로 정보를 확인할 수 없습니다.') from None
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, 'st_file_attributes', 0) & 0x400:
            raise AppError('INVALID_ARGUMENT', '심볼릭 링크·재분석 지점 경로는 지원하지 않습니다.')
    return path


def source_path(value: str | Path) -> Path:
    path = local_path(value)
    if not path.is_file():
        raise AppError('INPUT_NOT_FOUND', '입력 파일이 없습니다. 경로를 확인하세요.')
    if path.suffix.lower() != '.mp4':
        raise AppError('UNSUPPORTED_MEDIA', '현재 검증 프로파일은 로컬 MP4 파일입니다.')
    return path


def destination_path(value: str | Path, source: Path) -> Path:
    path = local_path(value)
    if path == source:
        raise AppError('OUTPUT_CONFLICT', '원본과 출력 경로가 같습니다.')
    if path.exists():
        raise AppError('OUTPUT_EXISTS', '결과 디렉터리가 이미 있습니다. 새 이름을 지정하세요.')
    if not path.parent.is_dir():
        raise AppError('IO_ERROR', '출력의 상위 디렉터리를 먼저 준비하세요.')
    return path
