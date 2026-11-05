"""Fresh-process synthetic batch costs; no user media, calibrated speed claim or saved bundle."""
import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import tracemalloc
from unittest.mock import patch

from cheating_racer_detect import media, pixels, service, tools
from cheating_racer_detect.events import export, replay
from cheating_racer_detect.events.diagnostics import lane_diagnostics
from cheating_racer_detect.events.lamp_diagnostics import lamp_diagnostics
from cheating_racer_detect.review import video
from cheating_racer_detect.source import source_session
from scripts import demo_batch_replay
from scripts.benchmark_events import DiskSampler, own_memory
from scripts.benchmark_scaling import Measurements, allocation_trace

FAMILIES = {
    'lifecycle': {'start', 'manual_reset', 'failed_update'},
    'source': {'source_or_geometry_change', 'lane_source_change', 'lamp_source_change', 'lanes_unknown', 'lamps_unknown'},
    'identity': {'identity_uncertain', 'identity_or_model_change', 'no_actual_observation', 'actual_observation_resumed'},
    'temporal_support': {'missing_frame', 'missing_decoded_frame', 'time_gap', 'insufficient_history', 'insufficient_cycles'},
    'visibility': {'occluded', 'out_of_view', 'too_small', 'uncertified_intersection'},
    'geometry': {'insufficient_boundaries', 'boundary_order_uncertain', 'multiple_intersections', 'outside_support', 'missing_anchor'},
    'context': {'ego_scene_or_pose_uncertain', 'context_unknown', 'coded_axes_uncertain', 'boundary_bindings_unconfirmed'},
    'model_uncertainty': {'model_uncertain'},
}


def reason_summary(inputs):
    """Code families only; never infer lighting/occlusion causes from an unknown."""
    result = {}
    for name, report in (('lane', lane_diagnostics(inputs)), ('lamp', lamp_diagnostics(inputs))):
        rows = {}
        for channel, counts in report['reason_counts'].items():
            families = Counter()
            for code, count in counts.items():
                family = next((group for group, codes in FAMILIES.items() if code in codes),
                              'usable' if code == 'usable' else 'unclassified')
                families[family] += count
            rows[channel] = {'codes': counts, 'families': dict(sorted(families.items()))}
        result[name] = rows
    return {'reason_counts': result, 'scope': 'input declarations and reason codes, not inferred causes or observed coverage'}


def validate(seconds, window, order):
    if type(seconds) is not int or not 5 <= seconds <= 600 or window not in ('head', 'tail') or order not in ('forward', 'reverse'):
        raise ValueError('Synthetic seconds 5..600, head/tail and forward/reverse required')


def fixture(toolchain, root, seconds, window):
    source, inputs = demo_batch_replay.generate_inputs(toolchain, root)
    padded = root/'padded.mp4'
    padding = seconds*10-len(inputs.rows)
    edge = 'start' if window == 'tail' else 'stop'
    tools.run_tool([toolchain.ffmpeg, '-nostdin', '-v', 'error', '-n', '-i', source,
                    '-vf', f'tpad={edge}={padding}:{edge}_mode=clone', '-an',
                    '-c:v', 'libx264', '-crf', '12', '-pix_fmt', 'yuv420p', '-bf', '0',
                    '-fps_mode', 'passthrough', '-f', 'mp4', padded], timeout=120)
    profile, timeline, _frames, _first, _rotation = video.prepare(padded, '0', '.1', toolchain)
    if len(timeline.frames) != seconds*10 or timeline.end != seconds:
        raise ValueError('Padded fixture must retain 10fps source clock')
    digest = service._sha256(padded)
    first = padding if window == 'tail' else 0
    rows = []
    for n, row in enumerate(inputs.rows):
        frame = replace(row.frame, source_id=digest, ordinal=first+n, pts=timeline.frames[first+n].pts,
                        time_base=timeline.time_base, first_pts=timeline.frames[0].pts,
                        stream_index=profile['index'])
        detection = replace(row.vehicle.detection, frame=frame)
        vehicle = replace(row.vehicle, frame=frame, detection=detection)
        rows.append(replace(row, frame=frame, vehicle=vehicle, lanes=replace(row.lanes, frame=frame)))
    return padded, replace(inputs, rows=tuple(rows))


def benchmark(*, seconds=5, window='tail', order='reverse', reuse=True):
    validate(seconds, window, order)
    if tracemalloc.is_tracing():
        raise RuntimeError('Use a fresh process without an existing allocation trace')
    toolchain = tools.discover_tools()
    with tempfile.TemporaryDirectory(prefix='crd-batch-benchmark-') as directory:
        root = Path(directory)
        source, inputs = fixture(toolchain, root, seconds, window)
        output = root/'batch'
        metrics, raw_bytes = Measurements(source, toolchain), 0
        publish = service._publish
        def measured(args, **kwargs):
            nonlocal raw_bytes
            result = metrics.run(args, **kwargs)
            if str(source) in [str(a) for a in args] and str(args[-2]) == 'rawvideo':
                raw_bytes += Path(args[-1]).stat().st_size
            return result
        indices = (1, 0) if order == 'reverse' else (0, 1)
        with DiskSampler(root) as disk, ExitStack() as stack, allocation_trace(), source_session(reuse=reuse, pixel_parent=root):
            for module in (tools, media, pixels, video):
                stack.enter_context(patch.object(module, 'run_tool', side_effect=measured))
            for module in (tools, media):
                stack.enter_context(patch.object(module, 'run_tool_lines', side_effect=metrics.run_lines))
            for module in (service, video, export, replay):
                stack.enter_context(patch.object(module, '_sha256', side_effect=metrics.hash_file))
            for module in (service, video, export):
                stack.enter_context(patch.object(module, '_publish', side_effect=lambda *args: disk.publish(publish, *args)))
            start = time.perf_counter()
            result = replay.export_replay_batch(source, inputs, output, candidate_indices=indices, toolchain=toolchain)
            elapsed = time.perf_counter()-start
            _current, peak = tracemalloc.get_traced_memory()
            memory = own_memory()
        frames = []
        for entry in result['candidates']:
            mapping = json.loads((output/entry['candidate']).parent.joinpath('media/review.json').read_text(encoding='utf-8'))
            frames.append({'candidate_index': entry['candidate_index'], 'signal_state': entry['signal_state'],
                           'ordinals': [r['source_frame']['ordinal'] for r in mapping['frames']],
                           'source_pts': [r['source_frame']['pts'] for r in mapping['frames']],
                           'review_times': [r['review_time'] for r in mapping['frames']]})
        final_bytes = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        return {'schema': 'synthetic-batch-costs-v1', 'input_seconds': seconds, 'window': window,
                'candidate_indices': list(indices), 'source_inspection_reuse': reuse,
                'status': result['status'], 'candidates': frames, 'wall_seconds': elapsed,
                'fixture_generation_in_wall_time': False, 'tool_calls': dict(sorted(metrics.tool_calls.items())),
                'tool_seconds': dict(sorted(metrics.tool_seconds.items())),
                'source_sha256_reads': metrics.hash_reads, 'source_sha256_logical_bytes': metrics.hash_bytes,
                'source_sha256_seconds': metrics.hash_seconds, 'source_raw_payload_bytes': raw_bytes,
                'python_traced_peak_bytes': peak, 'memory': memory,
                'owned_tree_sampled_peak_bytes': disk.peak, 'partial_tree_sampled_peak_bytes': disk.partial_peak,
                'disk_sample_interval_seconds': disk.interval, 'disk_samples': disk.samples,
                'final_output_bytes': final_bytes, **reason_summary(inputs),
                'limits': ['self-generated silent 96x64/10fps, not real-road accuracy or HD performance',
                           'fresh process recommended; OS cache uncontrolled; wall includes tracing overhead',
                           'hash bytes are logical reads, not physical disk IO; raw bytes are generated payload',
                           'own-process RAM excludes children/GPU; sampled disk peaks are lower bounds',
                           'fixture generation/import/discovery excluded; all temporary resources owned and removed']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=5)
    parser.add_argument('--window', choices=('head','tail'), default='tail')
    parser.add_argument('--order', choices=('forward','reverse'), default='reverse')
    parser.add_argument('--no-reuse', action='store_true')
    args = parser.parse_args(argv)
    try:
        validate(args.seconds, args.window, args.order)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(benchmark(seconds=args.seconds, window=args.window, order=args.order, reuse=not args.no_reuse), sort_keys=True))


if __name__ == '__main__': main()
