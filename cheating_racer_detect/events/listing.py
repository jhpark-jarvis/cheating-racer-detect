"""Bounded read-only listing audit; offline consistency is not pixel evidence."""

from fractions import Fraction
import hashlib
import json
import os
import stat

from ..errors import AppError
from ..paths import local_path, source_path
from ..service import _identity, _sha256
from ..source import inspect_source, release_pixels, source_job
from ..tools import discover_tools
from .catalog import CandidateCatalog, _bytes, make_catalog
from .diagnostics import diagnostics_bytes, lane_diagnostics
from .inputs import EventInputs, InputLaneEngine, MAX_BYTES, _exact, _json
from .lamp_diagnostics import lamp_diagnostics
from .replay import analyze_inputs

CORE = ('inputs.json', 'catalog.json')
SIDECARS = {'lane': 'lane-diagnostics.json', 'lamp': 'lamp-diagnostics.json'}


def _require(condition):
    if not condition:
        raise ValueError('Inconsistent saved listing')


class ListingFiles:
    """At most four fixed regular files; bounded reads also used at finish.

    The directory is never repaired, written or recursively enumerated. No
    unbounded file_digest read is used for these untrusted JSON documents.
    """
    def __init__(self, root):
        self.root = local_path(root)
        _require(self.root.is_dir())
        self.owner = _identity(self.root)[:2]
        self.members = self.inventory()
        self.seen = {}

    def inventory(self):
        local_path(self.root)
        _require(self.root.is_dir() and _identity(self.root)[:2] == self.owner)
        names = []
        with os.scandir(self.root) as entries:
            for entry in entries:
                _require(len(names) < 4 and entry.name in (*CORE, *SIDECARS.values()))
                path = local_path(self.root/entry.name)
                _require(stat.S_ISREG(path.lstat().st_mode))
                names.append(entry.name)
        _require(set(CORE) <= set(names))
        return tuple(sorted(names))

    def data(self, name):
        _require(name in self.members)
        path = local_path(self.root/name)
        _require(stat.S_ISREG(path.lstat().st_mode))
        before = _identity(path)
        _require(0 < before[2] <= MAX_BYTES)
        with path.open('rb') as stream:
            data = stream.read(MAX_BYTES+1)
        local_path(path)
        if before != _identity(path):
            raise AppError('INPUT_CHANGED', '검사 중 목록 파일이 변경되었습니다.')
        _require(0 < len(data) <= MAX_BYTES and len(data) == before[2])
        receipt = before, hashlib.sha256(data).digest()
        if name in self.seen and self.seen[name] != receipt:
            raise AppError('INPUT_CHANGED', '검사 중 목록 파일의 내용이 변경되었습니다.')
        self.seen[name] = receipt
        return data

    def json(self, name):
        def pairs(items):
            result = {}
            for key, value in items:
                _require(key not in result)
                result[key] = value
            return result
        def constant(_value):
            raise ValueError('Nonfinite JSON')
        value = json.loads(self.data(name).decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
        _json(value)
        return value

    def finish(self):
        if self.inventory() != self.members:
            raise AppError('INPUT_CHANGED', '검사 중 목록의 파일 구성이 변경되었습니다.')
        _require(set(self.seen) == set(self.members))
        for name in self.members:
            self.data(name)
        if self.inventory() != self.members:
            raise AppError('INPUT_CHANGED', '검사 중 목록의 파일 구성이 변경되었습니다.')


def _required(value):
    if (type(value) not in (list, tuple) or len(value) > 2
            or any(type(item) is not str or item not in SIDECARS for item in value)
            or len(set(value)) != len(value)):
        raise AppError('INVALID_ARGUMENT', '필수 진단은 중복 없이 lane/lamp만 지정하세요.')
    return tuple(name for name in SIDECARS if name in value)


def _recorded_candidates(inputs, value):
    engine = InputLaneEngine(inputs.lane_config)
    candidates, resets = [], []
    for row in inputs.rows:
        result = engine.update(row)
        if result['reset_reason'] is not None:
            resets.append((row.frame, result['reset_reason']))
        if result['lane']['candidate'] is not None:
            candidates.append(result['lane']['candidate'])
    _require(len(candidates) == value['candidate_count'])
    for item, listed in zip(candidates, value['candidates']):
        _require(listed['candidate_sha256'] == hashlib.sha256(_bytes(item.report())).hexdigest()
                 and listed['direction'] == item.direction)
        for name in ('start', 'crossing', 'completion'):
            _require(listed[name] == [str(frame.time) for frame in getattr(item, name)])
        begin, end = map(_exact, listed['requested_context'])
        _require(begin == max(Fraction(0), item.start[0].time-inputs.pre)
                 and end <= item.completion[1].time+inputs.post)
    return tuple(candidates), tuple(resets)


@source_job
def verify_listing(folder, *, source=None, toolchain=None, required_diagnostics=()):
    """Check 2-4 listing files without writing; optionally replay source once.

    Optional-file deletion cannot be inferred from the historical format. Use
    required_diagnostics to enforce an independently known lane/lamp requirement.
    Even source replay verifies supplied annotations, NOT annotation authenticity.
    """
    required = _required(required_diagnostics)
    if source is None and toolchain is not None:
        raise AppError('INVALID_ARGUMENT', '원본 재생 없이 도구 설정을 검사할 수 없습니다.')
    try:
        files = ListingFiles(folder)
        present = tuple(name for name, filename in SIDECARS.items() if filename in files.members)
        _require(set(required) <= set(present))
        inputs = EventInputs.from_bytes(files.data('inputs.json'))
        catalog = CandidateCatalog.from_bytes(files.data('catalog.json'))
        value = catalog.report()
        _require(value['inputs_sha256'] == inputs.sha256
                 and value['source_sha256'] == inputs.rows[0].frame.source_id
                 and _json(value['source_frame']) == _json(inputs.rows[0].frame.report())
                 and value['selected_identity'] == list(inputs.selected_identity))
        candidates, lane_resets = _recorded_candidates(inputs, value)
        reports = {}
        for name in present:
            saved = files.json(SIDECARS[name])
            reports[name] = lane_diagnostics(inputs) if name == 'lane' else lamp_diagnostics(inputs)
            _require(diagnostics_bytes(saved) == diagnostics_bytes(reports[name]))
        if source is not None:
            original = source_path(source)
            before = _identity(original)
            _require(_sha256(original) == value['source_sha256'])
            tools = toolchain or discover_tools()
            analysis = analyze_inputs(original, inputs, tools)
            _video, _audio, timeline = inspect_source(original, tools, max_axis=1024)
            _require(analysis.candidates == candidates and analysis.lane_resets == lane_resets)
            _require(make_catalog(inputs, analysis, timeline).to_bytes() == catalog.to_bytes())
            if 'lamp' in reports:
                resets = tuple((row.frame, item['source_reset_reason']) for row, item in
                    zip(inputs.rows, reports['lamp']['rows']) if item['source_reset_reason'] is not None)
                _require(resets == analysis.lamp_resets)
            release_pixels()
            local_path(original)
            if before != _identity(original) or _sha256(original) != value['source_sha256']:
                raise AppError('INPUT_CHANGED', '검사 중 원본 영상이 변경되었습니다.')
        files.finish()
        return {'schema': 'experimental-listing-verification-v1', 'status': 'verified',
                'candidate_count': value['candidate_count'], 'files_checked': len(files.seen),
                'diagnostics_checked': list(present), 'required_diagnostics': list(required),
                'source_replayed': source is not None, 'signal_context_verified': source is not None,
                'verification_level': 'source_replay' if source is not None else 'recorded_inputs',
                'limits': ['consistency not authenticity; caller annotations not ground truth',
                           'offline signal and source-duration context are not pixel verified',
                           'optional sidecar deletion requires an independent required_diagnostics setting',
                           'no media export or real-world accuracy/violation verification']}
    except AppError as error:
        if error.code in ('INVALID_INPUTS', 'INVALID_CATALOG', 'INVALID_DIAGNOSTICS'):
            raise AppError('LISTING_INVALID', '저장 목록의 입력·후보·진단 계약을 확인하세요.') from None
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, ZeroDivisionError,
            RecursionError, UnicodeError):
        raise AppError('LISTING_INVALID', '저장 목록의 파일·입력·후보·진단 계약을 확인하세요.') from None
    except OSError:
        raise AppError('IO_ERROR', '목록 파일과 원본의 접근 권한을 확인하세요.') from None
