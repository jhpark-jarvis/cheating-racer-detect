"""Real FFmpeg/OpenCV equivalence and retained full-read security checks."""

from collections import Counter
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect import media, service
from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import export, replay
from cheating_racer_detect.review import video
from cheating_racer_detect.source import source_session
from cheating_racer_detect.tools import discover_tools
from scripts.demo_events import capture_inputs, generate, observation
from tests.fixtures import sha256

try:
    TOOLS = discover_tools()
except AppError:
    TOOLS = None
AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('numpy', 'cv2'))


@unittest.skipUnless(AVAILABLE and TOOLS, 'NOT_RUN: existing FFmpeg/NumPy/OpenCV required')
class Native(unittest.TestCase):
    def test_replay_equivalent_one_source_probe_and_all_hash_decode_checks(self):
        with tempfile.TemporaryDirectory(prefix='crd-inspection-native-') as directory:
            root = Path(directory)
            source = generate(TOOLS, root)
            before = sha256(source)
            inputs = capture_inputs(source, TOOLS, root)
            original_probe, original_time = media._probe, media._timeline
            original_decode, original_hash = media._full_decode, service._sha256
            results, receipts, mappings, metrics = [], [], [], []
            for reuse in (False, True):
                counts = Counter()
                def count(kind, function):
                    def run(path, *args, **kwargs):
                        counts[('source_' if path == source else 'derived_')+kind] += 1
                        return function(path, *args, **kwargs)
                    return run
                with ExitStack() as stack, source_session(reuse=reuse):
                    for name, kind, function in (('_probe', 'probe', original_probe),
                            ('_timeline', 'timeline', original_time), ('_full_decode', 'decode', original_decode)):
                        stack.enter_context(patch.object(media, name, side_effect=count(kind, function)))
                    for module in (service, video, export, replay):
                        stack.enter_context(patch.object(module, '_sha256', side_effect=count('hash', original_hash)))
                    output = root/('result-'+str(reuse))
                    results.append(replay.export_replay(source, inputs, output, candidate_index=0, toolchain=TOOLS))
                receipts.append(json.loads((output/'candidate'/'candidate.json').read_text(encoding='utf-8')))
                mappings.append(json.loads((output/'candidate'/'media'/'review.json').read_text(encoding='utf-8')))
                metrics.append(counts)
                self.assertEqual(list(output.rglob('*.bgr')), [])
            self.assertEqual(results[0], results[1])
            self.assertEqual(receipts[0], receipts[1])
            self.assertEqual(mappings[0]['frames'], mappings[1]['frames'])
            self.assertEqual(mappings[1]['review']['frame_count'], 14)
            self.assertEqual(results[1]['signal_state'], 'right_blink')
            self.assertEqual(metrics[0]['source_probe'], 4)
            self.assertEqual(metrics[0]['source_timeline'], 4)
            self.assertEqual(metrics[1]['source_probe'], 1)
            self.assertEqual(metrics[1]['source_timeline'], 1)
            for key in ('source_hash', 'source_decode', 'derived_probe', 'derived_timeline', 'derived_decode'):
                self.assertGreater(metrics[0][key], 0)
                self.assertEqual(metrics[0][key], metrics[1][key])
            self.assertEqual(sha256(source), before)
            self.assertEqual(list(root.glob('*.partial')), [])

    def test_same_fingerprint_content_change_rejected_by_final_hash(self):
        with tempfile.TemporaryDirectory(prefix='crd-inspection-change-') as directory:
            root = Path(directory)
            source = generate(TOOLS, root)
            old, st = source.read_bytes(), source.stat()
            callback = observation()
            changed = False
            def mutate(frame, pixels):
                nonlocal changed
                report = callback(frame, pixels)
                if not changed:
                    new = bytearray(old)
                    new[-1] ^= 1
                    source.write_bytes(new)
                    os.utime(source, ns=(st.st_atime_ns, st.st_mtime_ns))
                    self.assertEqual(service._identity(source), (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns))
                    changed = True
                return report
            output = root/'result'
            with self.assertRaises(AppError) as error:
                video.export_review(source, 0, '.2', output, mutate, TOOLS)
            self.assertEqual(error.exception.code, 'INPUT_CHANGED')
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob('*.partial')), [])
            self.assertNotEqual(source.read_bytes(), old)
            source.write_bytes(old)  # Only restore this test-owned fixture.
            result = video.export_review(source, 0, '.2', output, observation(), TOOLS)
            self.assertEqual(result['review']['frame_count'], 2)
            self.assertEqual(source.read_bytes(), old)


if __name__ == '__main__':
    unittest.main()
