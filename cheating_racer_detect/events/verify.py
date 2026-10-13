"""Read-only bounded batch audit; consistency is not authenticity or accuracy."""

from fractions import Fraction as F
import hashlib
import json

from ..errors import AppError
from ..media import _audio_window, _full_decode, _probe, _profile, _rotation, _timeline, select_frames, validate_container
from ..paths import local_path, source_path
from ..review.records import summarize
from ..service import _identity, _sha256
from ..source import inspect_source, source_job
from ..tools import discover_tools
from .catalog import CandidateCatalog, _side, _signal
from .contracts import Candidate
from .export import candidate_context
from .inputs import EventInputs, MAX_BYTES, _exact, _fields, _frame, _json
from .replay import _selection, analyze_inputs, recorded_observer
from .lane import LaneEngine


def require(condition):
    if not condition:
        raise ValueError('Inconsistent saved bundle')


class Files:
    """Only fixed, caller-independent relative names are opened. No output IO."""

    def __init__(self, root):
        self.root = local_path(root)
        require(self.root.is_dir())
        self.owner = _identity(self.root)[:2]
        self.seen = {}

    def file(self, name):
        path = local_path(self.root/name)
        require(path.is_file())
        before = _identity(path)
        digest = _sha256(path)
        require(before == _identity(path))
        if path in self.seen:
            require(self.seen[path] == (before, digest))
        else:
            self.seen[path] = before, digest
        return path

    def data(self, name):
        candidate_path = local_path(self.root/name)
        require(candidate_path.is_file() and 0 < candidate_path.stat().st_size <= MAX_BYTES)
        path = self.file(name)
        with path.open('rb') as stream:
            value = stream.read(MAX_BYTES+1)
        require(0 < len(value) <= MAX_BYTES)
        return value

    def json(self, name):
        def pairs(items):
            result = {}
            for key, value in items:
                require(key not in result)
                result[key] = value
            return result
        def constant(_):
            raise ValueError('Nonfinite JSON')
        result = json.loads(self.data(name).decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
        _json(result)  # Also reject overflowing JSON numbers such as 1e999.
        return result

    def finish(self):
        local_path(self.root)
        require(self.owner == _identity(self.root)[:2])
        for path, (before, digest) in self.seen.items():
            local_path(path)
            require(before == _identity(path) and digest == _sha256(path))


def candidate(value):
    _fields(value, ('schema', 'identity', 'identity_status', 'road_id', 'from_lane', 'to_lane',
                    'direction', 'start', 'crossing', 'completion', 'config_id', 'limits'))
    result = Candidate(tuple(value['identity']), value['road_id'], tuple(value['from_lane']),
                       tuple(value['to_lane']), value['direction'],
                       *[tuple(_frame(f) for f in value[key]) for key in ('start', 'crossing', 'completion')],
                       value['config_id'])
    require(_json(result.report()) == _json(value))
    return result


def media(files, prefix, bundle, inputs, tools):
    require((bundle['review'], bundle['original_clip'], bundle['time_mapping']) ==
            ('media/review.mp4', 'media/original/clip.mp4', 'media/review.json'))
    record = files.json(prefix+'media/review.json')
    original = files.json(prefix+'media/original/result.json')
    require(record['schema'] == 'experimental-review-video-v1' and record['status'] == 'complete')
    require(original['schema_version'] == 1 and type(original['schema_version']) is int and original['status'] == 'complete')
    require(record['original_clip'] == 'original/clip.mp4' and record['original_metadata'] == 'original/result.json')
    require(record['source'] == original['source'] and record['source']['sha256'] == bundle['source_sha256'])
    require(record['request'] == original['request'])
    begin, end = (_exact(x) for x in bundle['context'])
    request = original['request']
    # The exporter rounds context outwards with a 40-digit Decimal conversion.
    epsilon = F(1, 10**35)
    require(0 <= begin-F(request['start_seconds']) <= epsilon and
            0 <= F(request['end_seconds'])-end <= epsilon)
    meta = record['review']
    require(meta['file'] == 'review.mp4' and meta['audio'] == 'none' and
            meta['view'] == 'coded_raster_no_display_rotation')
    rows = record['frames']
    require(type(rows) is list and 1 <= len(rows) <= 120 and type(meta['frame_count']) is int and len(rows) == meta['frame_count'])
    frames = [_frame(row['source_frame']) for row in rows]
    reference = inputs.rows[0].frame
    observer = recorded_observer(inputs)
    for n, (frame, row) in enumerate(zip(frames, rows)):
        require((frame.key, frame.time_base, frame.first_pts) ==
                (reference.key, reference.time_base, reference.first_pts))
        require(F(request['start_seconds']) <= frame.time < F(request['end_seconds']))
        require(_json(row['evidence']) == _json(summarize(frame, observer(frame, b''))))
        require(type(row['review_pts']) is int and row['review_pts'] >= 0)
        require(_exact(row['review_time']) == frame.time-frames[0].time)
        require(F(row['review_pts'])*_exact(meta['time_base']) == _exact(row['review_time']))
        if n:
            require(frame.ordinal == frames[n-1].ordinal+1 and frame.time > frames[n-1].time)
    require(_exact(meta['source_time_at_review_zero']) == frames[0].time)
    review = files.file(prefix+'media/review.mp4')
    require(_sha256(review) == meta['sha256'])
    validate_container(review)
    video, audio = _profile(_probe(review, tools, 'BUNDLE_INVALID'), 'BUNDLE_INVALID')
    timeline = _timeline(review, video, tools, 'BUNDLE_INVALID')
    require(audio is None and [video['width'], video['height']] == meta['canvas_size'] and _rotation(video) == 0)
    require(timeline.time_base == _exact(meta['time_base']) and len(timeline.frames) == len(frames))
    require([f.pts*timeline.time_base for f in timeline.frames] == [_exact(r['review_time']) for r in rows])
    require(timeline.frames[-1].duration*timeline.time_base == _exact(meta['last_frame_duration']))
    _full_decode(review, tools, 'BUNDLE_INVALID')

    clip = files.file(prefix+'media/original/clip.mp4')
    validate_container(clip)
    video, audio = _profile(_probe(clip, tools, 'BUNDLE_INVALID'), 'BUNDLE_INVALID')
    timeline = _timeline(clip, video, tools, 'BUNDLE_INVALID')
    saved = original['media']
    src, selection, out = saved['source_media'], saved['selection'], saved['output_media']
    require(src['video_stream_index'] == reference.stream_index and src['time_base'] == str(reference.time_base)
            and src['first_pts'] == reference.first_pts and (src['width'], src['height'], src['rotation_degrees']) ==
            (reference.width, reference.height, reference.display_rotation))
    require(selection['source_pts'] == [f.pts for f in frames] and selection['frame_count'] == len(frames))
    require(out['video_pts'] == [f.pts for f in timeline.frames] and out['frame_count'] == len(frames))
    require(out['video_time_base'] == str(timeline.time_base) and out['audio_present'] is (audio is not None))
    dimensions = (reference.height, reference.width) if reference.display_rotation in (90, 270) else (reference.width, reference.height)
    require((video['width'], video['height']) == dimensions and _rotation(video) == 0 and
            (out['width'], out['height'], out['rotation_degrees']) == (*dimensions, 0))
    # Match the original exporter tick/movie-clock bound, not an arbitrary receipt tolerance.
    rate = int(audio['sample_rate']) if audio else 1
    import math
    movie = math.lcm(reference.time_base.denominator, rate)
    if movie > 2_147_483_647:
        movie = reference.time_base.denominator
    tolerance = max(reference.time_base, timeline.time_base, F(1, movie))
    for frame, actual in zip(frames, timeline.frames):
        require(abs(frame.time-F(request['start_seconds'])-actual.pts*timeline.time_base) <= tolerance)
    require(abs(timeline.frames[-1].duration*timeline.time_base-_exact(meta['last_frame_duration'])) <= tolerance)
    if audio:
        window = _audio_window(clip, audio, tools, 'BUNDLE_INVALID')
        timing = saved['verification']['audio_timing']
        require(window.count > 0 and timing['status'] == 'verified')
        sample_tolerance = F(1024, rate)+F(1, movie)
        require(abs(window.first-F(timing['expected_first_sample_seconds'])) <= sample_tolerance and
                abs(window.end-F(timing['expected_sample_end_seconds'])) <= sample_tolerance)
    _full_decode(clip, tools, 'BUNDLE_INVALID')
    return frames


@source_job
def verify_batch(folder, *, source=None, toolchain=None):
    """Audit batch-v1 only. Optional source freshly replays supplied observations.

    No files are written to the bundle. Source mode owns temporary decoded pixels
    outside it. Stored hashes are not signatures; original clip has no saved byte
    hash, so an otherwise valid pixel alteration there is not detectable offline.
    """
    try:
        files = Files(folder)
        manifest = files.json('batch.json')
        fields = ('schema', 'status', 'inputs', 'inputs_sha256', 'source_sha256', 'candidate_count',
                  'selected_count', 'candidate_indices', 'candidates', 'limits')
        has_catalog = 'catalog' in manifest
        _fields(manifest, (*fields, 'catalog', 'catalog_sha256') if has_catalog else fields)
        require(manifest['schema'] == 'experimental-event-replay-batch-v1' and manifest['status'] == 'complete')
        require(manifest['inputs'] == 'inputs.json')
        inputs = EventInputs.from_bytes(files.data('inputs.json'))
        require(inputs.sha256 == manifest['inputs_sha256'] and inputs.rows[0].frame.source_id == manifest['source_sha256'])
        indices = _selection(manifest['candidate_indices'])
        require(type(manifest['candidate_count']) is int and 1 <= manifest['candidate_count'] <= 120)
        require(type(manifest['selected_count']) is int and len(indices) == manifest['selected_count'])
        require(type(manifest['candidates']) is list and len(manifest['candidates']) == len(indices))
        require(all(i < manifest['candidate_count'] for i in indices))
        lane_engine = LaneEngine(inputs.lane_config)
        recorded_candidates = []
        for row in inputs.rows:
            event = lane_engine.update(row.frame, row.vehicle, row.lanes)['candidate']
            if event is not None:
                recorded_candidates.append(event)
        require(len(recorded_candidates) == manifest['candidate_count'])
        catalog = None
        if has_catalog:
            require(manifest['catalog'] == 'catalog.json')
            catalog = CandidateCatalog.from_bytes(files.data('catalog.json'))
            value = catalog.report()
            require(catalog.sha256 == manifest['catalog_sha256'] and value['inputs_sha256'] == inputs.sha256
                    and value['source_sha256'] == manifest['source_sha256'] and value['candidate_count'] == manifest['candidate_count']
                    and value['source_frame'] == inputs.rows[0].frame.report() and value['selected_identity'] == list(inputs.selected_identity))
            for item, listed in zip(recorded_candidates, value['candidates']):
                require(listed['candidate_sha256'] == hashlib.sha256((_json(item.report())+'\n').encode()).hexdigest())
        tools = toolchain or discover_tools()
        analysis = timeline = original_source = before_source = None
        if source is not None:
            original_source = source_path(source)
            before_source = _identity(original_source)
            require(_sha256(original_source) == manifest['source_sha256'])
            analysis = analyze_inputs(original_source, inputs, tools)
            _video, _audio, timeline = inspect_source(original_source, tools, max_axis=1024)
            require(len(analysis.candidates) == manifest['candidate_count'])
            if catalog is not None:
                from .catalog import make_catalog
                require(make_catalog(inputs, analysis, timeline).to_bytes() == catalog.to_bytes())
        for index, entry in zip(indices, manifest['candidates']):
            _fields(entry, ('candidate_index', 'candidate', 'direction', 'signal_state'))
            prefix = f'candidates/{index:06d}/'
            require(type(entry['candidate_index']) is int and entry['candidate_index'] == index and entry['candidate'] == prefix+'candidate.json')
            bundle = files.json(prefix+'candidate.json')
            _fields(bundle, ('schema', 'status', 'candidate', 'context', 'requested_context', 'lamp', 'signal_state',
                            'review', 'original_clip', 'time_mapping', 'source_sha256', 'limits'))
            require(bundle['schema'] == 'experimental-candidate-bundle-v1' and bundle['status'] == 'complete'
                    and bundle['source_sha256'] == manifest['source_sha256'])
            item = candidate(bundle['candidate'])
            require(item == recorded_candidates[index] and item.identity == inputs.selected_identity and item.direction == entry['direction']
                    and bundle['signal_state'] == entry['signal_state'])
            require(bundle['signal_state'] in ('left_blink', 'right_blink', 'both_blink', 'no_blink_observed', 'unknown'))
            require(type(bundle['context']) is list and len(bundle['context']) == 2 and _exact(bundle['context'][0]) < _exact(bundle['context'][1]))
            require(type(bundle['requested_context']) is list and len(bundle['requested_context']) == 2
                    and _exact(bundle['context'][0]) <= _exact(bundle['requested_context'][0]) <
                    _exact(bundle['requested_context'][1]) <= _exact(bundle['context'][1]))
            require(bundle['lamp'] is None and bundle['signal_state'] == 'unknown' or
                    type(bundle['lamp']) is dict and bundle['lamp']['signal_state'] == bundle['signal_state'])
            sides = {}
            if bundle['lamp'] is not None:
                lamp = bundle['lamp']
                require(lamp['schema'] == 'experimental-lamp-window-v1' and lamp['config'] == inputs.lamp_config.report()
                        and lamp['identity_status'] == 'unverified' and lamp['identity'] in (None, list(inputs.selected_identity)))
                require(type(lamp['samples']) is list and len(lamp['samples']) <= 120)
                for side in ('left', 'right'):
                    summary = lamp[side]
                    _fields(summary, ('state', 'blink_evidence', 'complete_cycles', 'reasons', 'valid_samples'))
                    sides[side] = {key: summary[key] for key in ('state', 'blink_evidence', 'reasons', 'valid_samples')}
                    _side(sides[side])
                require(_signal(sides['left'], sides['right']) == bundle['signal_state'])
                recorded_rows = {row.frame.ordinal: row for row in inputs.rows}
                last_sample = None
                for sample in lamp['samples']:
                    frame = _frame(sample['vehicle']['frame'])
                    row = recorded_rows.get(frame.ordinal)
                    require(row is not None and row.vehicle is not None and sample['vehicle'] == row.vehicle.report())
                    require(last_sample is None or frame.time > last_sample)
                    last_sample = frame.time
                    for side in ('left', 'right'):
                        value, roi = sample[side], getattr(row, side)
                        _fields(value, ('status', 'brightness', 'roi'))
                        require(value['roi'] == (None if roi.box is None else list(roi.box.values())))
                        require(value['status'] == roi.status or roi.status == 'usable' and value['status'] == 'too_small')
                        require(value['brightness'] is None if value['status'] != 'usable' else
                                value['brightness'] is not None and 0 <= _exact(value['brightness']) <= 255)
            if catalog is not None:
                listed = catalog.report()['candidates'][index]
                require(hashlib.sha256((_json(item.report())+'\n').encode()).hexdigest() == listed['candidate_sha256']
                        and bundle['context'] == listed['context'] and bundle['requested_context'] == listed['requested_context']
                        and bundle['signal_state'] == listed['signal_state'])
                require(all(sides.get(side) == listed[side] for side in ('left', 'right')))
            if analysis is not None:
                require(item == analysis.candidates[index])
                start, end, requested, lamps = candidate_context(item, inputs.pre, inputs.post, timeline, analysis.lamps)
                require(bundle['context'] == [str(start), str(end)] and bundle['requested_context'] == requested
                        and _json(bundle['lamp']) == _json(lamps))
            frames = media(files, prefix, bundle, inputs, tools)
            actual = set(frames)
            require(set(item.start+item.crossing+item.completion) <= actual)
            by_ordinal = {row.frame.ordinal: row for row in inputs.rows}
            for anchor in item.start+item.crossing+item.completion:
                require(anchor.ordinal in by_ordinal and by_ordinal[anchor.ordinal].frame == anchor
                        and by_ordinal[anchor.ordinal].vehicle is not None)
            if timeline is not None:
                require(all(f.ordinal < len(timeline.frames) and timeline.frames[f.ordinal].pts == f.pts for f in frames))
                saved_request = files.json(prefix+'media/original/result.json')['request']
                selected = select_frames(timeline, F(saved_request['start_seconds']), F(saved_request['end_seconds']))
                require([f.pts for f in selected] == [f.pts for f in frames])
        if original_source is not None:
            local_path(original_source)
            require(before_source == _identity(original_source) and _sha256(original_source) == manifest['source_sha256'])
        files.finish()
        return {'schema': 'experimental-bundle-verification-v1', 'status': 'verified',
                'selected_count': len(indices), 'files_checked': len(files.seen),
                'source_replayed': source is not None, 'catalog_checked': has_catalog,
                'limits': ['consistency not authenticity', 'caller annotations not ground truth',
                           'original clip pixel identity not verified', 'not real-world accuracy']}
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, ZeroDivisionError, RecursionError, UnicodeError):
        raise AppError('BUNDLE_INVALID', '저장 묶음의 파일·참조·해시·시각 계약을 확인하세요.') from None
    except OSError:
        raise AppError('IO_ERROR', '저장 묶음의 파일 접근 권한을 확인하세요.') from None
