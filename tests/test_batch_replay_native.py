"""Real encoded two-change replay, exact media provenance and late aborts."""

from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect import media
from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import replay
from cheating_racer_detect.events.inputs import load_inputs, save_inputs
from cheating_racer_detect.events.lamp import LampROI
from cheating_racer_detect.tools import discover_tools
from scripts.demo_batch_replay import generate_inputs
from tests.fixtures import probe, sha256

try:
    TOOLS = discover_tools()
except AppError:
    TOOLS = None
AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('numpy', 'cv2'))


@unittest.skipUnless(AVAILABLE and TOOLS, 'NOT_RUN: existing FFmpeg/NumPy/OpenCV required')
class Native(unittest.TestCase):
    def test_two_reverse_candidates_actual_media_hash_pts_and_retained_checks(self):
        with tempfile.TemporaryDirectory(prefix='crd-batch-native-') as directory:
            root = Path(directory)
            source, inputs = generate_inputs(TOOLS, root)
            before = sha256(source)
            save_inputs(inputs, root/'saved')
            inputs = load_inputs(root/'saved'/'inputs.json')
            expected = replay.analyze_inputs(source, inputs, TOOLS)
            self.assertEqual([c.direction for c in expected.candidates], ['right', 'left'])
            counts = Counter()
            def counted(kind, function):
                def run(path, *args, **kwargs):
                    counts[('source_' if path == source else 'derived_')+kind] += 1
                    return function(path, *args, **kwargs)
                return run
            with ExitStack() as stack:
                analyze = stack.enter_context(patch.object(replay, 'analyze_inputs', wraps=replay.analyze_inputs))
                for name in ('_probe', '_timeline', '_full_decode'):
                    stack.enter_context(patch.object(media, name, side_effect=counted(name, getattr(media, name))))
                result = replay.export_replay_batch(source, inputs, root/'batch', candidate_indices=(1, 0), toolchain=TOOLS)
                analyze.assert_called_once()
            self.assertEqual(result['candidate_indices'], [1, 0])
            self.assertEqual(result['candidate_count'], 2)
            self.assertEqual(counts['source__probe'], 1)
            self.assertEqual(counts['source__timeline'], 1)
            self.assertEqual(counts['source__full_decode'], 2)
            self.assertEqual(counts['derived__full_decode'], 4)
            self.assertEqual(counts['derived__timeline'], 4)
            self.assertEqual(result['inputs_sha256'], sha256(root/'batch'/'inputs.json'))
            for entry in result['candidates']:
                path = root/'batch'/entry['candidate']
                receipt = json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(receipt['candidate'], expected.candidates[entry['candidate_index']].report())
                self.assertEqual(receipt['source_sha256'], before)
                self.assertEqual(entry['signal_state'], entry['direction']+'_blink')
                mapping = json.loads((path.parent/'media'/'review.json').read_text(encoding='utf-8'))
                source_frames = [f['source_frame'] for f in mapping['frames']]
                for sample in receipt['lamp']['samples']:
                    self.assertIn(sample['vehicle']['frame'], source_frames)
                actual = probe(TOOLS, path.parent/'media'/'review.mp4', frames=True)['frames']
                self.assertEqual(len(actual), mapping['review']['frame_count'])
                for observed, frame in zip(actual, source_frames):
                    self.assertEqual(F(observed['pts_time']), F(frame['normalized_time'])-F(source_frames[0]['normalized_time']))
                self.assertTrue((path.parent/'media'/'original'/'clip.mp4').is_file())
            self.assertEqual(sha256(source), before)
            self.assertEqual(list(root.glob('*.partial')), [])
            self.assertEqual(list((root/'batch').rglob('*.bgr')), [])

    def test_unknown_second_candidate_and_single_api_subset_compatibility(self):
        with tempfile.TemporaryDirectory(prefix='crd-batch-unknown-') as directory:
            root = Path(directory)
            source, inputs = generate_inputs(TOOLS, root)
            rows = list(inputs.rows)
            rows[28] = replace(rows[28], left=LampROI(None, 'occluded'))
            inputs = replace(inputs, rows=tuple(rows))
            batch = replay.export_replay_batch(source, inputs, root/'batch', candidate_indices=(1,), toolchain=TOOLS)
            single = replay.export_replay(source, inputs, root/'single', candidate_index=1, toolchain=TOOLS)
            self.assertEqual(batch['selected_count'], 1)
            self.assertEqual(batch['candidate_count'], 2)
            self.assertEqual(batch['candidates'][0]['signal_state'], 'unknown')
            a = json.loads((root/'batch'/'candidates'/'000001'/'candidate.json').read_text(encoding='utf-8'))
            b = json.loads((root/'single'/'candidate'/'candidate.json').read_text(encoding='utf-8'))
            self.assertEqual(a, b)
            self.assertEqual(single['schema'], 'experimental-event-replay-v1')
            self.assertIn('occluded', a['lamp']['left']['reasons'])

    def test_actual_first_bundle_late_error_cancel_cleanup_and_new_job_retry(self):
        with tempfile.TemporaryDirectory(prefix='crd-batch-abort-') as directory:
            root = Path(directory)
            source, inputs = generate_inputs(TOOLS, root)
            before = sha256(source)
            original = replay.export_candidate
            for failure in (AppError('ENCODE_FAILED', 'injected'), KeyboardInterrupt()):
                calls = 0
                def late(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise failure
                    return original(*args, **kwargs)
                with patch.object(replay, 'export_candidate', side_effect=late), self.assertRaises(type(failure)):
                    replay.export_replay_batch(source, inputs, root/'result', candidate_indices=(0, 1), toolchain=TOOLS)
                self.assertEqual(calls, 2)
                self.assertFalse((root/'result').exists())
                self.assertEqual(list(root.glob('*.partial')), [])
                self.assertEqual(sha256(source), before)
            result = replay.export_replay_batch(source, inputs, root/'result', candidate_indices=(0, 1), toolchain=TOOLS)
            self.assertEqual(result['selected_count'], 2)

    def test_actual_cli_batch_single_invalid_selection_and_demo(self):
        with tempfile.TemporaryDirectory(prefix='crd-batch-cli-') as directory:
            root = Path(directory)
            source, inputs = generate_inputs(TOOLS, root)
            save_inputs(inputs, root/'saved')
            base = [sys.executable, '-m', 'scripts.replay_events', '--input', str(source),
                    '--inputs', str(root/'saved'/'inputs.json')]
            for label, options, code in (('batch', ['--candidate-indices', '1', '0'], 0),
                    ('single', ['--candidate-index', '0'], 0), ('duplicate', ['--candidate-indices', '0', '0'], 1),
                    ('missing', ['--candidate-indices', '0', '2'], 1)):
                process = subprocess.run(base+options+['--output', str(root/label)], capture_output=True, text=True, timeout=90)
                self.assertEqual(process.returncode, code, process.stdout)
                summary = json.loads(process.stdout)
                self.assertNotIn(str(root), process.stdout)
                self.assertEqual((root/label).exists(), code == 0)
                if label == 'batch':
                    self.assertEqual(summary['candidate_indices'], [1, 0])
                    self.assertEqual(summary['selected_count'], 2)
            demo = subprocess.run([sys.executable, '-m', 'scripts.demo_batch_replay', '--output', str(root/'demo')],
                                  capture_output=True, text=True, timeout=90)
            self.assertEqual(demo.returncode, 0, demo.stdout)
            self.assertEqual(json.loads(demo.stdout)['selected_count'], 2)


if __name__ == '__main__':
    unittest.main()
