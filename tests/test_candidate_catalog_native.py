"""Actual listing -> saved receipt -> source re-analysis -> selected media."""

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

from cheating_racer_detect import media, pixels
from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import catalog, replay
from cheating_racer_detect.events.inputs import load_inputs, save_inputs
from cheating_racer_detect.events.lamp import LampROI
from cheating_racer_detect.review.video import prepare
from cheating_racer_detect.source import inspect_source, source_session
from cheating_racer_detect.tools import discover_tools, run_tool
from cheating_racer_detect.tracking import Frame
from scripts.demo_batch_replay import generate_inputs
from tests.fixtures import probe, sha256

try:
    TOOLS = discover_tools()
except AppError:
    TOOLS = None
AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('numpy', 'cv2'))


@unittest.skipUnless(AVAILABLE and TOOLS, 'NOT_RUN: existing FFmpeg/NumPy/OpenCV required')
class Native(unittest.TestCase):
    def test_list_no_encoding_and_saved_selection_matches_media_context_and_pts(self):
        with tempfile.TemporaryDirectory(prefix='crd-catalog-native-') as directory:
            root = Path(directory)
            source, inputs = generate_inputs(TOOLS, root)
            before = sha256(source)
            counts = Counter()
            def counted(kind, function):
                def run(*args, **kwargs):
                    counts[kind] += 1
                    return function(*args, **kwargs)
                return run
            with ExitStack() as stack:
                for module, name, kind in ((media, '_probe', 'probe'), (media, '_timeline', 'timeline'),
                        (media, '_full_decode', 'clip_decode'), (pixels, 'run_tool', 'raw'), (replay, 'analyze_inputs', 'analysis')):
                    stack.enter_context(patch.object(module, name, side_effect=counted(kind, getattr(module, name))))
                item = catalog.export_catalog(source, inputs, root/'listed', toolchain=TOOLS)
            self.assertEqual(counts, Counter(probe=1, timeline=1, raw=1, analysis=1))
            self.assertEqual({p.name for p in (root/'listed').iterdir()}, {'inputs.json', 'catalog.json'})
            item = catalog.load_catalog(root/'listed'/'catalog.json')
            entries = item.report()['candidates']
            self.assertEqual([e['direction'] for e in entries], ['right', 'left'])
            restored = load_inputs(root/'listed'/'inputs.json')
            result = replay.export_replay_batch(source, restored, root/'batch', candidate_indices=(1, 0), toolchain=TOOLS, catalog=item)
            self.assertEqual(result['catalog_sha256'], item.sha256)
            self.assertEqual((root/'batch'/'catalog.json').read_bytes(), item.to_bytes())
            for exported in result['candidates']:
                index = exported['candidate_index']
                path = root/'batch'/exported['candidate']
                bundle = json.loads(path.read_text(encoding='utf-8'))
                entry = entries[index]
                self.assertEqual(entry['context'], bundle['context'])
                self.assertEqual(entry['requested_context'], bundle['requested_context'])
                self.assertEqual(entry['candidate_sha256'], catalog.hashlib.sha256(catalog._bytes(bundle['candidate'])).hexdigest())
                for name in ('start', 'crossing', 'completion'):
                    self.assertEqual(entry[name], [f['normalized_time'] for f in bundle['candidate'][name]])
                self.assertEqual(entry['signal_state'], exported['signal_state'])
                for side in ('left', 'right'):
                    self.assertEqual(entry[side], {key: bundle['lamp'][side][key] for key in entry[side]})
                mapping = json.loads((path.parent/'media'/'review.json').read_text(encoding='utf-8'))
                actual = probe(TOOLS, path.parent/'media'/'review.mp4', frames=True)['frames']
                source_frames = [f['source_frame'] for f in mapping['frames']]
                self.assertEqual(len(actual), len(source_frames))
                for observed, frame in zip(actual, source_frames):
                    self.assertEqual(F(observed['pts_time']), F(frame['normalized_time'])-F(source_frames[0]['normalized_time']))
            self.assertEqual(sha256(source), before)
            self.assertEqual(list(root.glob('*.partial')), [])

    def test_partial_blink_occlusion_unknown_and_zero_candidate_catalog(self):
        with tempfile.TemporaryDirectory(prefix='crd-catalog-unknown-') as directory:
            root = Path(directory)
            source, inputs = generate_inputs(TOOLS, root)
            rows = list(inputs.rows)
            rows[34] = replace(rows[34], left=LampROI(None, 'occluded'))
            inputs = replace(inputs, rows=tuple(rows))
            item = catalog.export_catalog(source, inputs, root/'unknown', toolchain=TOOLS)
            entry = item.report()['candidates'][1]
            self.assertEqual(entry['signal_state'], 'unknown')
            self.assertIn('occluded', entry['left']['reasons'])
            result = replay.export_replay_batch(source, inputs, root/'batch', candidate_indices=(1,), toolchain=TOOLS, catalog=item)
            self.assertEqual(result['candidates'][0]['signal_state'], 'unknown')
            empty = replace(inputs, rows=tuple(replace(row, lanes=None) for row in inputs.rows))
            item = catalog.export_catalog(source, empty, root/'empty', toolchain=TOOLS)
            self.assertEqual(item.report()['candidate_count'], 0)
            self.assertEqual(item.report()['candidates'], [])
            with self.assertRaises(AppError) as error:
                replay.export_replay_batch(source, empty, root/'no-media', candidate_indices=(0,), toolchain=TOOLS, catalog=item)
            self.assertEqual(error.exception.code, 'INVALID_CANDIDATE')
            self.assertFalse((root/'no-media').exists())

    def test_stale_input_or_valid_shaped_catalog_drift_fails_before_media(self):
        with tempfile.TemporaryDirectory(prefix='crd-catalog-stale-') as directory:
            root = Path(directory)
            source, inputs = generate_inputs(TOOLS, root)
            item = catalog.export_catalog(source, inputs, root/'listed', toolchain=TOOLS)
            with patch.object(pixels, 'run_tool') as decode, self.assertRaises(AppError) as error:
                replay.export_replay_batch(source, replace(inputs, pre=F(1, 4)), root/'stale', candidate_indices=(0,), toolchain=TOOLS, catalog=item)
            self.assertEqual(error.exception.code, 'CATALOG_CHANGED')
            decode.assert_not_called()
            changed = item.report()
            changed['candidates'][0]['candidate_sha256'] = 'f'*64
            bad = catalog.CandidateCatalog.from_bytes(catalog._bytes(changed))
            with patch.object(replay, 'export_candidate') as export, self.assertRaises(AppError) as error:
                replay.export_replay_batch(source, inputs, root/'changed', candidate_indices=(0,), toolchain=TOOLS, catalog=bad)
            self.assertEqual(error.exception.code, 'CATALOG_CHANGED')
            export.assert_not_called()
            self.assertFalse((root/'changed').exists())
            self.assertEqual(list(root.glob('*.partial')), [])

    def test_actual_cli_list_then_catalog_batch_errors_and_retry(self):
        with tempfile.TemporaryDirectory(prefix='crd-catalog-cli-') as directory:
            root = Path(directory)
            source, inputs = generate_inputs(TOOLS, root)
            save_inputs(inputs, root/'saved')
            args = ['--input', str(source), '--inputs', str(root/'saved'/'inputs.json')]
            process = subprocess.run([sys.executable, '-m', 'scripts.list_events', *args, '--output', str(root/'listed')],
                                     capture_output=True, text=True, timeout=90)
            self.assertEqual(process.returncode, 0, process.stdout)
            summary = json.loads(process.stdout)
            self.assertEqual(summary['candidate_count'], 2)
            self.assertNotIn(str(root), process.stdout)
            self.assertNotIn(sha256(source), process.stdout)
            options = ['--candidate-indices', '1', '0', '--catalog', str(root/'listed'/'catalog.json')]
            command = [sys.executable, '-m', 'scripts.replay_events', *args, *options]
            good = subprocess.run(command+['--output', str(root/'batch')], capture_output=True, text=True, timeout=90)
            self.assertEqual(good.returncode, 0, good.stdout)
            self.assertEqual(json.loads(good.stdout)['candidate_indices'], [1, 0])
            exists = subprocess.run(command+['--output', str(root/'batch')], capture_output=True, text=True, timeout=90)
            self.assertEqual(exists.returncode, 1)
            self.assertEqual(json.loads(exists.stdout)['code'], 'OUTPUT_EXISTS')
            bad = root/'bad.json'
            bad.write_bytes(b'{"schema":1,"schema":2}')
            rejected = subprocess.run([sys.executable, '-m', 'scripts.replay_events', *args,
                '--candidate-indices', '0', '--catalog', str(bad), '--output', str(root/'bad-result')],
                capture_output=True, text=True, timeout=90)
            self.assertEqual(rejected.returncode, 1)
            self.assertEqual(json.loads(rejected.stdout)['code'], 'INVALID_CATALOG')
            self.assertFalse((root/'bad-result').exists())

    def test_actual_pixel_cleanup_failure_precedes_catalog_publish_and_retry(self):
        with tempfile.TemporaryDirectory(prefix='crd-catalog-cleanup-') as directory:
            root = Path(directory)
            source, inputs = generate_inputs(TOOLS, root)
            with source_session(pixel_parent=root), patch.object(pixels._Snapshot, 'close', side_effect=AppError('CLEANUP_FAILED', 'owned')), \
                    self.assertRaises(AppError) as error:
                catalog.export_catalog(source, inputs, root/'result', toolchain=TOOLS)
            self.assertEqual(error.exception.code, 'CLEANUP_FAILED')
            self.assertFalse((root/'result').exists())
            self.assertEqual(list(root.glob('*.partial')), [])
            self.assertEqual(len(list(root.glob('crd-pixels-*'))), 1)
            self.assertEqual(catalog.export_catalog(source, inputs, root/'result', toolchain=TOOLS).report()['candidate_count'], 2)
            # The outer test temp owns/removes the intentionally preserved cache.

    def test_offset_vfr_rotation_preserve_source_brackets_and_planned_context(self):
        with tempfile.TemporaryDirectory(prefix='crd-catalog-clock-') as directory:
            root = Path(directory)
            source, inputs = generate_inputs(TOOLS, root)
            for name, options in (('offset', ['-copyts', '-itsoffset', '5', '-i', source, '-c', 'copy', '-avoid_negative_ts', 'disabled']),
                    ('rotation', ['-display_rotation:v:0', '270', '-i', source, '-c', 'copy']),
                    ('vfr', ['-i', source, '-vf', r'setpts=if(lt(N\,20)\,N/(10*TB)\,(2+(N-20)/8)/TB)',
                             '-fps_mode', 'passthrough', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-bf', '0'])):
                with self.subTest(profile=name):
                    new = root/(name+'.mp4')
                    run_tool([TOOLS.ffmpeg, '-nostdin', '-v', 'error', '-n', *options, new])
                    _video, _audio, timeline = inspect_source(new, TOOLS)
                    video, timeline, frames, first, rotation = prepare(new, F(0), timeline.end, TOOLS)
                    self.assertEqual(len(frames), len(inputs.rows))
                    rows = []
                    for n, (old, item) in enumerate(zip(inputs.rows, frames)):
                        frame = Frame(sha256(new), video['index'], first+n, item.pts, timeline.time_base,
                                      timeline.frames[0].pts, video['width'], video['height'], rotation)
                        rows.append(replace(old, frame=frame, vehicle=replace(old.vehicle, frame=frame,
                            detection=replace(old.vehicle.detection, frame=frame)), lanes=replace(old.lanes, frame=frame)))
                    current = replace(inputs, rows=tuple(rows))
                    item = catalog.export_catalog(new, current, root/(name+'-listed'), toolchain=TOOLS)
                    reference = item.report()['source_frame']
                    self.assertEqual(reference['first_pts'], timeline.frames[0].pts)
                    self.assertEqual(reference['display_rotation'], rotation)
                    result = replay.export_replay_batch(new, current, root/(name+'-batch'), candidate_indices=(0,), toolchain=TOOLS, catalog=item)
                    bundle = json.loads((root/(name+'-batch')/result['candidates'][0]['candidate']).read_text(encoding='utf-8'))
                    self.assertEqual(bundle['context'], item.report()['candidates'][0]['context'])
                    self.assertEqual([f['normalized_time'] for f in bundle['candidate']['crossing']], item.report()['candidates'][0]['crossing'])


if __name__ == '__main__':
    unittest.main()
