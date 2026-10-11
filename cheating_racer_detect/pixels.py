"""Owned, disk-backed, ordinal-exact pixel windows; no caller-supplied cache."""

from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile

from .errors import AppError
from .paths import local_path
from .service import _identity
from .source import active_job
from .tools import run_tool


class _Snapshot:
    def __init__(self, key, first, count, size, complete, parent):
        self.key, self.first, self.count, self.size = key, first, count, size
        self.complete = complete
        parent = local_path(tempfile.gettempdir() if parent is None else parent)
        self.directory = Path(tempfile.mkdtemp(prefix='crd-pixels-', dir=parent))
        self.owner = _identity(self.directory)[:2]
        self.raw = self.directory/'source.bgr'
        self.raw_identity = None

    def check(self):
        local_path(self.directory)
        local_path(self.raw)
        if (_identity(self.directory)[:2] != self.owner
                or _identity(self.raw) != self.raw_identity):
            raise AppError('OUTPUT_INVALID', '작업 소유 픽셀 자료가 변경되었습니다.')

    def close(self):
        try:
            local_path(self.directory)
            if _identity(self.directory)[:2] != self.owner:
                raise OSError('Pixel directory owner changed')
            shutil.rmtree(self.directory)
        except (OSError, AppError):
            raise AppError('CLEANUP_FAILED', '픽셀 임시 자료의 소유권과 정리 상태를 확인하세요.') from None


@contextmanager
def pixel_window(source, tools, video, timeline, first, count, *, complete=False):
    """Yield a readonly stream positioned at a validated contiguous frame range.

    Analysis keeps EOF validation. Review may stop after count only AFTER its
    separate complete source/clip validation. A complete request never reuses a
    bounded-only decode. The caller consumes exactly count frames, one at a time.
    """
    from .media import _input_options, _rotation, validate_container

    source = local_path(source)
    if (type(first) is not int or type(count) is not int or first < 0
            or not 1 <= count <= 120 or first+count > len(timeline.frames)
            or not 0 < video['width'] <= 1024 or not 0 < video['height'] <= 1024):
        raise AppError('REVIEW_LIMIT', '검증된 최대 120프레임 픽셀 구간이 필요합니다.')
    before = _identity(source)
    validate_container(source)
    job = active_job()
    if job is not None and job.failed:
        raise AppError('INPUT_CHANGED', '변경된 원본의 작업은 새로 시작하세요.')
    key = (source, tools, before, video['index'], video['width'], video['height'],
           _rotation(video), timeline.time_base, timeline.frames[0].pts)
    cached = None if job is None or not job.reuse else job.pixels
    if cached is not None and cached.key[:2] == key[:2] and cached.key[2] != before:
        job.failed = True
        job.clear()
        raise AppError('INPUT_CHANGED', '픽셀 처리 중 원본 정보가 변경되었습니다.')
    hit = (cached is not None and cached.key == key and cached.first <= first
           and first+count <= cached.first+cached.count and (not complete or cached.complete))
    snapshot = cached if hit else _Snapshot(key, first, count, video['width']*video['height']*3,
                                           complete, None if job is None else job.pixel_parent)
    retained = hit
    try:
        if not hit:
            args = [tools.ffmpeg, '-nostdin', '-v', 'error', '-n', '-xerror', '-err_detect', 'explode',
                    *_input_options(), '-noautorotate', '-i', source, '-map', '0:v:0',
                    '-vf', f'select=between(n\\,{first}\\,{first+count-1})',
                    '-fps_mode', 'passthrough', '-an', '-pix_fmt', 'bgr24']
            if not complete:
                args.extend(['-frames:v', str(count)])
            run_tool([*args, '-f', 'rawvideo', snapshot.raw], timeout=120)
            local_path(snapshot.raw)
            if snapshot.raw.stat().st_size != snapshot.size*count:
                raise AppError('OUTPUT_INVALID', '원본 프레임과 디코딩 결과가 다릅니다.')
            snapshot.raw_identity = _identity(snapshot.raw)
        local_path(source)
        if _identity(source) != before:
            if job is not None:
                job.failed = True
            raise AppError('INPUT_CHANGED', '픽셀 디코딩 중 원본 정보가 변경되었습니다.')
        snapshot.check()
        with snapshot.raw.open('rb') as stream:
            stream.seek((first-snapshot.first)*snapshot.size)
            yield stream
        local_path(source)
        if _identity(source) != before:
            if job is not None:
                job.failed = True
            raise AppError('INPUT_CHANGED', '픽셀 소비 중 원본 정보가 변경되었습니다.')
        snapshot.check()
        if not hit and job is not None and job.reuse and job.pixels is None:
            job.pixels, retained = snapshot, True
    except BaseException:
        if job is not None:
            job.clear()
        retained = hit  # clear() already removed the cached snapshot.
        raise
    finally:
        if not retained:
            snapshot.close()
