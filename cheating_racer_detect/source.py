"""Job-local source inspection reuse; never a persistent or caller-trusted cache.

Metadata, frozen video timelines and one bounded owned pixel window are reused. Existing SHA-256,
complete source decode and independent derived-output checks remain mandatory.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import json
import threading

from .errors import AppError
from .paths import local_path

_ACTIVE = ContextVar('crd_source_inspection_job', default=None)


@dataclass
class _Job:
    reuse: bool
    thread: int
    active: bool = True
    snapshot: tuple | None = None
    failed: bool = False
    pixels: object | None = None
    pixel_parent: object | None = None

    def clear(self):
        self.snapshot = None
        self.clear_pixels()

    def clear_pixels(self):
        pixels, self.pixels = self.pixels, None
        if pixels is not None:
            pixels.close()


def active_job():
    """Internal synchronous owner only; expired/copied/threaded scopes are ignored."""
    job = _ACTIVE.get()
    return job if job is not None and job.active and job.thread == threading.get_ident() else None


def release_pixels():
    """Dispose pixel resources before a workflow publishes its final result."""
    job = active_job()
    if job is not None:
        job.clear_pixels()


@contextmanager
def source_session(*, reuse=True, pixel_parent=None):
    """Internal synchronous scope. Nested jobs inherit the outer reuse policy."""
    job = _ACTIVE.get()
    if job is not None and job.active and job.thread == threading.get_ident():
        try:
            yield job
        except BaseException:
            # A caller may catch an inner failure and retry within its outer
            # scope. Never retain the failed workflow's prior inspection.
            job.clear()
            raise
        return
    job = _Job(reuse, threading.get_ident(), pixel_parent=pixel_parent)
    token = _ACTIVE.set(job)
    try:
        yield job
    finally:
        job.active = False
        try:
            job.clear()
        finally:
            _ACTIVE.reset(token)


def source_job(function):
    """Scope an existing synchronous workflow without changing its arguments."""
    @wraps(function)
    def run(*args, **kwargs):
        with source_session():
            return function(*args, **kwargs)
    return run


def _axis_limit(video, max_axis):
    if max_axis is not None and max(video['width'], video['height']) > max_axis:
        raise AppError('REVIEW_LIMIT', '검토 영상은 원본 축 1024 이하를 지원합니다.')


def inspect_source(source, tools, *, max_axis=None):
    """Return owned profile copies and an immutable Timeline for a local source.

At most one source/tool pair is stored per job. Other paths/tools are inspected
independently. Fingerprints are guards, not authentication; publish-time hashes
in the owning workflows still detect same-size/same-mtime content changes.
"""
    from .media import _probe, _profile, _timeline, validate_container
    from .service import _identity

    source = local_path(source)
    before = _identity(source)
    job = _ACTIVE.get()
    if job is not None and (not job.active or job.thread != threading.get_ident()):
        job = None
    if job is not None and job.failed:
        raise AppError('INPUT_CHANGED', '변경된 원본의 작업은 새로 시작하세요.')
    snapshot = None if job is None else job.snapshot
    if snapshot is not None and snapshot[:2] == (source, tools):
        if before != snapshot[2]:
            job.failed = True
            raise AppError('INPUT_CHANGED', '처리 중 원본 정보가 변경되었습니다.')
        # Keep the cheap container/reference gate even on hits. A fingerprint
        # alone must not authorize a rewritten indirect media reference.
        validate_container(source)
        video, audio = json.loads(snapshot[3])
        _axis_limit(video, max_axis)
        return video, audio, snapshot[4]
    validate_container(source)
    video, audio = _profile(_probe(source, tools, 'UNSUPPORTED_MEDIA'), 'UNSUPPORTED_MEDIA')
    _axis_limit(video, max_axis)
    timeline = _timeline(source, video, tools, 'INVALID_TIMELINE')
    local_path(source)
    if before != _identity(source):
        if job is not None:
            job.failed = True
        raise AppError('INPUT_CHANGED', '원본 검사 중 파일 정보가 변경되었습니다.')
    # Owned serialization prevents the caller mutating the memo's nested profile.
    profile = json.dumps([video, audio], allow_nan=False)
    if job is not None and job.reuse and job.snapshot is None:
        job.snapshot = (source, tools, before, profile, timeline)
    video, audio = json.loads(profile)
    return video, audio, timeline
