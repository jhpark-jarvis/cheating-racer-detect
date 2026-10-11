"""Fixed event window in longer self-generated sources; no user video accepted.

Run each sample in a fresh Python process. Wall includes tracemalloc overhead;
own-process RAM excludes FFmpeg children/GPU. No performance acceptance gate.
"""

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from decimal import Decimal
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import time
import tracemalloc
from unittest.mock import patch

from cheating_racer_detect import media, service, tools
from cheating_racer_detect.events import export
from cheating_racer_detect.review import video
from cheating_racer_detect.source import inspect_source, source_session
from scripts import demo_events
from scripts.benchmark_events import DiskSampler, classify, own_memory


def validate(seconds, window):
    if type(seconds) is not int or not 3 <= seconds <= 600:
        raise ValueError('Diagnostic source seconds must be an integer from 3 to 600')
    if window not in ('head', 'tail'):
        raise ValueError('Diagnostic window must be head or tail')


def generate_padded(toolchain, root, seconds, window):
    validate(seconds, window)
    base = demo_events.generate(toolchain, root)
    source = root/'padded.mp4'
    padding = seconds*10-len(demo_events.POSITIONS)
    edge = 'start' if window == 'tail' else 'stop'
    tools.run_tool([toolchain.ffmpeg, '-nostdin', '-v', 'error', '-n', '-i', base,
                    '-vf', f'tpad={edge}={padding}:{edge}_mode=clone', '-an',
                    '-c:v', 'libx264', '-crf', '12', '-pix_fmt', 'yuv420p', '-bf', '0',
                    '-fps_mode', 'passthrough', '-f', 'mp4', source], timeout=120)
    return source


@contextmanager
def allocation_trace():
    if tracemalloc.is_tracing():
        raise RuntimeError('Use a fresh process without an existing allocation trace')
    tracemalloc.start()
    try:
        yield
    finally:
        tracemalloc.stop()


class Measurements:
    def __init__(self, source, toolchain):
        self.source, self.toolchain = source, toolchain
        self.tool_calls, self.tool_seconds = Counter(), Counter()
        self.timeline_stdout_characters = 0
        self.hash_reads = self.hash_bytes = 0
        self.hash_seconds = 0.
        self.actual_tool, self.actual_hash = tools.run_tool, service._sha256
        self.actual_lines = tools.run_tool_lines

    def run(self, args, **kwargs):
        kind = classify(args, self.source, self.toolchain)
        self.tool_calls[kind] += 1
        start = time.perf_counter()
        try:
            result = self.actual_tool(args, **kwargs)
            if kind == 'source_timeline_probe':
                self.timeline_stdout_characters += len(result)
            return result
        finally:
            self.tool_seconds[kind] += time.perf_counter()-start

    def hash_file(self, path):
        if path != self.source:
            return self.actual_hash(path)
        self.hash_reads += 1
        self.hash_bytes += path.stat().st_size  # Logical bytes, not physical disk IO.
        start = time.perf_counter()
        try:
            return self.actual_hash(path)
        finally:
            self.hash_seconds += time.perf_counter()-start

    def run_lines(self, args, consume, **kwargs):
        kind = classify(args, self.source, self.toolchain)
        self.tool_calls[kind] += 1
        def measured(line):
            if kind == 'source_timeline_probe':
                self.timeline_stdout_characters += len(line)
            consume(line)
        start = time.perf_counter()
        try:
            return self.actual_lines(args, measured, **kwargs)
        finally:
            self.tool_seconds[kind] += time.perf_counter()-start


def benchmark(*, seconds=60, window='tail', reuse=True):
    validate(seconds, window)
    if tracemalloc.is_tracing():
        raise RuntimeError('Use a fresh process without an existing allocation trace')
    toolchain = tools.discover_tools()
    with tempfile.TemporaryDirectory(prefix='crd-scaling-benchmark-') as directory:
        root = Path(directory)
        source = generate_padded(toolchain, root, seconds, window)
        output = root/'result'
        metrics = Measurements(source, toolchain)
        actual_publish = service._publish
        start_time = Decimal(seconds)-Decimal('2.4') if window == 'tail' else Decimal(0)
        end_time = start_time+Decimal('2.4')
        from cheating_racer_detect import pixels
        with DiskSampler(root) as disk, ExitStack() as stack, allocation_trace(), source_session(reuse=reuse, pixel_parent=root):
            for module in (tools, media, video, demo_events, pixels):
                stack.enter_context(patch.object(module, 'run_tool', side_effect=metrics.run))
            for module in (tools, media):
                stack.enter_context(patch.object(module, 'run_tool_lines', side_effect=metrics.run_lines))
            for module in (service, video, export, demo_events):
                stack.enter_context(patch.object(module, '_sha256', side_effect=metrics.hash_file))
            for module in (service, video, export):
                stack.enter_context(patch.object(module, '_publish', side_effect=lambda *args: disk.publish(actual_publish, *args)))
            start = time.perf_counter()
            _video, _audio, timeline = inspect_source(source, toolchain)
            inspected = time.perf_counter()
            inspection_current, inspection_peak = tracemalloc.get_traced_memory()
            inspection_memory = own_memory()
            source_frames = len(timeline.frames)
            timeline_storage_bytes = timeline.storage_bytes
            if source_frames != seconds*10 or timeline.end != seconds:
                raise ValueError('Padded fixture duration/frame count mismatch')
            del timeline  # Off mode must not retain an extra diagnostic copy during the workflow.
            first = seconds*10-len(demo_events.POSITIONS) if window == 'tail' else 0
            candidate, lamps = demo_events.analyze(source, toolchain, root, start=start_time, end=end_time)
            analyzed = time.perf_counter()
            report = export.export_candidate(source, candidate, output, demo_events.observation(first=first),
                pre=F(1, 5), post=F(2, 5), lamps=lamps, toolchain=toolchain)
            finished = time.perf_counter()
            mapping = json.loads((output/'media'/'review.json').read_text(encoding='utf-8'))
            allocated_current, allocated_peak = tracemalloc.get_traced_memory()
            memory = own_memory()
        return {'schema': 'synthetic-resource-scaling-v1', 'input_seconds': seconds,
                'window': window, 'analysis_range': [str(start_time), str(end_time)],
                'source_inspection_reuse': reuse, 'source_frames': source_frames,
                'analysis_frames': len(demo_events.POSITIONS), 'analysis_first_ordinal': first,
                'analysis_raw_bytes': len(demo_events.POSITIONS)*96*64*3,
                'analysis_raw_bytes_definition': 'validated selected BGR payload; owned window removed at job exit',
                'pixel_storage_in_sampled_tree': True,
                'review_frames': mapping['review']['frame_count'],
                'review_first_ordinal': mapping['frames'][0]['source_frame']['ordinal'],
                'review_last_ordinal': mapping['frames'][-1]['source_frame']['ordinal'],
                'review_relative_times': [row['review_time'] for row in mapping['frames']],
                'review_source_time_at_zero': mapping['review']['source_time_at_review_zero'],
                'source_unchanged': mapping['checks']['source_unchanged'],
                'status': report['status'], 'signal_state': report['signal_state'],
                'fixture_generation_in_wall_time': False, 'inspection_seconds': inspected-start,
                'analysis_seconds': analyzed-inspected, 'candidate_export_seconds': finished-analyzed,
                'wall_seconds': finished-start, 'wall_over_input_duration': (finished-start)/seconds,
                'python_traced_current_bytes': allocated_current, 'python_traced_peak_bytes': allocated_peak,
                'inspection_python_traced_current_bytes': inspection_current,
                'inspection_python_traced_peak_bytes': inspection_peak, 'inspection_memory': inspection_memory,
                'memory': memory, 'native_child_peak_ram': 'NOT_MEASURED', 'vram': 'NOT_MEASURED',
                'source_file_bytes': source.stat().st_size,
                'source_sha256_reads': metrics.hash_reads, 'source_sha256_logical_bytes': metrics.hash_bytes,
                'source_sha256_seconds': metrics.hash_seconds,
                'source_timeline_stdout_characters': metrics.timeline_stdout_characters,
                'source_timeline_protocol': 'default-frame-sections-v1',
                'source_timeline_packed_payload_bytes': timeline_storage_bytes,
                'tool_calls': dict(sorted(metrics.tool_calls.items())),
                'tool_seconds': dict(sorted(metrics.tool_seconds.items())),
                'owned_tree_sampled_peak_bytes': disk.peak, 'partial_tree_sampled_peak_bytes': disk.partial_peak,
                'disk_sample_interval_seconds': disk.interval, 'disk_samples': disk.samples,
                'disk_sampling_serialized_with_publication': True,
                'final_output_bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
                'limits': ['fresh Python process; OS file cache uncontrolled; wall includes tracing overhead',
                           '96x64 10fps silent CFR static padding, not real-video/HD performance',
                           'extra diagnostic source inspection counted; fixture generation/discovery excluded',
                           'traced allocations exclude native memory; process-start RAM peak includes imports',
                           'child RAM/VRAM not measured; sampled disk peaks are lower bounds',
                           'diagnostic disk sampling serialized with publication; no production retry',
                           'hash bytes are logical reads, timeline characters are not memory bytes',
                           'selected pixels bounded; full source timeline/decode and integrity checks retained',
                           '3..600 seconds is a diagnostic generation bound, not a product support limit']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=60, help='Synthetic source duration, 3..600')
    parser.add_argument('--window', choices=('head', 'tail'), default='tail')
    parser.add_argument('--no-reuse', action='store_true')
    args = parser.parse_args(argv)
    try:
        validate(args.seconds, args.window)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(benchmark(seconds=args.seconds, window=args.window, reuse=not args.no_reuse), sort_keys=True))


if __name__ == '__main__':
    main()
