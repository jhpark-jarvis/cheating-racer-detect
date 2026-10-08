"""Fresh-process synthetic baseline, not a calibrated performance gate.

python -m scripts.benchmark_events prints non-sensitive JSON; no files retained.
Own-process peak working set excludes FFmpeg children/GPU; disk peaks sampled.
"""

from collections import Counter
from contextlib import ExitStack
import ctypes
from ctypes import wintypes
from fractions import Fraction as F
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

from cheating_racer_detect import media, service, tools
from cheating_racer_detect.events import export
from cheating_racer_detect.review import video
from scripts import demo_events


def own_memory():
    if os.name != 'nt':
        return None
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)]+[(name, ctypes.c_size_t) for name in (
            'peak', 'current', 'paged_peak', 'paged', 'nonpaged_peak', 'nonpaged', 'pagefile', 'pagefile_peak')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel, psapi = ctypes.WinDLL('kernel32', use_last_error=True), ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'working_set_bytes': counters.current, 'peak_working_set_bytes_since_process_start': counters.peak}


class DiskSampler:
    """Only a newly created benchmark-owned TemporaryDirectory is walked."""
    def __init__(self, root, interval=.02):
        self.root, self.interval = root, interval
        self.peak = self.partial_peak = self.samples = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name='crd-owned-disk-sampler', daemon=True)

    def sample(self):
        total = partial = 0
        for path in self.root.rglob('*'):
            try:
                if path.is_file():
                    size = path.stat().st_size
                    total += size
                    if any(name.endswith('.partial') for name in path.relative_to(self.root).parts[:-1]):
                        partial += size
            except FileNotFoundError:
                pass  # rename/unlink races lower sampled peaks, never exact maxima
        self.peak, self.partial_peak = max(self.peak, total), max(self.partial_peak, partial)
        self.samples += 1

    def run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()


def classify(args, source, toolchain):
    args = [str(a) for a in args]
    if any(a in args for a in ('-version', '-encoders', '-decoders')):
        return 'capabilities'
    source_read = str(source) in args
    if args[0] == str(toolchain.ffprobe):
        kind = 'timeline_probe' if '-show_frames' in args else 'metadata_probe'
    elif '-f' in args and 'null' in args:
        kind = 'full_decode'
    elif 'rawvideo' in args and '-pixel_format' not in args:
        kind = 'raw_decode'
    else:
        kind = 'encode'
    return ('source_' if source_read else 'derived_')+kind


def benchmark():
    toolchain = tools.discover_tools()
    with tempfile.TemporaryDirectory(prefix='crd-event-benchmark-') as directory:
        root = Path(directory)
        source = demo_events.generate(toolchain, root)
        output = root/'result'
        calls, hash_reads = Counter(), 0
        actual_tool, actual_hash = tools.run_tool, service._sha256
        def run(args, **kwargs):
            calls[classify(args, source, toolchain)] += 1
            return actual_tool(args, **kwargs)
        def hash_file(path):
            nonlocal hash_reads
            hash_reads += path == source
            return actual_hash(path)
        with DiskSampler(root) as disk, ExitStack() as stack:
            for module in (tools, media, video, demo_events):
                stack.enter_context(patch.object(module, 'run_tool', side_effect=run))
            for module in (service, video, export, demo_events):
                stack.enter_context(patch.object(module, '_sha256', side_effect=hash_file))
            start = time.perf_counter()
            candidate, lamps = demo_events.analyze(source, toolchain, root)
            analyzed = time.perf_counter()
            report = export.export_candidate(source, candidate, output, demo_events.observation(),
                pre=F(1, 5), post=F(2, 5), lamps=lamps, toolchain=toolchain)
            finished = time.perf_counter()
            memory = own_memory()
        final_bytes = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        mapping = json.loads((output/'media'/'review.json').read_text(encoding='utf-8'))
        return {'schema': 'synthetic-event-baseline-v1', 'input_seconds': '12/5',
                'fixture_generation_in_wall_time': False, 'analysis_seconds': analyzed-start,
                'candidate_export_seconds': finished-analyzed, 'wall_seconds': finished-start,
                'wall_over_input_duration': (finished-start)/2.4,
                'memory': memory, 'native_child_peak_ram': 'NOT_MEASURED', 'vram': 'NOT_MEASURED',
                'owned_tree_sampled_peak_bytes': disk.peak, 'partial_tree_sampled_peak_bytes': disk.partial_peak,
                'disk_sample_interval_seconds': disk.interval, 'disk_samples': disk.samples,
                'final_output_bytes': final_bytes, 'source_sha256_reads': hash_reads, 'tool_calls': dict(sorted(calls.items())),
                'review_frames': mapping['review']['frame_count'], 'signal_state': report['signal_state'],
                'limits': ['fresh Python process recommended; OS file cache uncontrolled',
                           'own-process memory includes native OpenCV and imports, excludes children',
                           'sampled disk peaks are lower bounds; may miss short-lived writes',
                           'wall includes analysis/export checks, not fixture generation/discovery',
                           '2.4s synthetic input, not long-video performance or product acceptance']}


if __name__ == '__main__':
    print(json.dumps(benchmark(), sort_keys=True))
