"""Saved synthetic batch audits using actual FFmpeg and independent source replay."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events.catalog import export_catalog
from cheating_racer_detect.events.replay import export_replay_batch
from cheating_racer_detect.events.verify import verify_batch
from cheating_racer_detect.service import _sha256
from cheating_racer_detect.tools import discover_tools, run_tool
from cheating_racer_detect.review.video import prepare
from cheating_racer_detect.source import inspect_source
from cheating_racer_detect.tracking import Frame
from scripts.demo_batch_replay import generate_inputs

try:
    TOOLS = discover_tools()
except AppError:
    TOOLS = None
AVAILABLE = all(importlib.util.find_spec(n) for n in ('numpy', 'cv2'))


@unittest.skipUnless(AVAILABLE and TOOLS, 'NOT_RUN: FFmpeg/NumPy/OpenCV required')
class Bundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='crd-audit-')
        cls.root = Path(cls.temporary.name)
        cls.source, inputs = generate_inputs(TOOLS, cls.root)
        cls.inputs = inputs
        catalog = export_catalog(cls.source, inputs, cls.root/'list', toolchain=TOOLS)
        export_replay_batch(cls.source, inputs, cls.root/'batch', candidate_indices=(1, 0), catalog=catalog, toolchain=TOOLS)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copy(self, root):
        return Path(shutil.copytree(self.root/'batch', root/'moved'))

    def test_actual_batch_offline_source_readonly_and_moved(self):
        with tempfile.TemporaryDirectory() as directory:
            moved = self.copy(Path(directory))
            before = {p.relative_to(moved): (_sha256(p), p.stat().st_mtime_ns) for p in moved.rglob('*') if p.is_file()}
            offline = verify_batch(moved, toolchain=TOOLS)
            source = verify_batch(moved, source=self.source, toolchain=TOOLS)
            self.assertEqual(offline['selected_count'], 2)
            self.assertFalse(offline['source_replayed'])
            self.assertTrue(source['source_replayed'])
            self.assertTrue(source['catalog_checked'])
            self.assertEqual(before, {p.relative_to(moved): (_sha256(p), p.stat().st_mtime_ns) for p in moved.rglob('*') if p.is_file()})
            (moved/'catalog.json').unlink()
            manifest = json.loads((moved/'batch.json').read_text())
            del manifest['catalog'], manifest['catalog_sha256']
            (moved/'batch.json').write_text(json.dumps(manifest))
            self.assertFalse(verify_batch(moved, source=self.source, toolchain=TOOLS)['catalog_checked'])

    def test_missing_corrupt_media_and_review_hash(self):
        for name, data in (('candidates/000001/media/review.mp4', b'broken'),
                           ('candidates/000001/media/original/clip.mp4', b'broken'),
                           ('inputs.json', None)):
            with self.subTest(file=name), tempfile.TemporaryDirectory() as directory:
                moved = self.copy(Path(directory))
                path = moved/name
                if data is None:
                    path.unlink()
                else:
                    path.write_bytes(data)
                with self.assertRaises(AppError):
                    verify_batch(moved, toolchain=TOOLS)

    def test_manifest_index_path_count_hash_and_candidate_drift(self):
        changes = [lambda m: m.update(inputs='../inputs.json'), lambda m: m.update(candidate_indices=[0, 0]),
                   lambda m: m.update(candidate_count=True), lambda m: m.update(inputs_sha256='f'*64),
                   lambda m: m['candidates'][0].update(candidate='https://example.org/secret'),
                   lambda m: m.update(catalog_sha256='f'*64)]
        original = json.loads((self.root/'batch/batch.json').read_text())
        for change in changes:
            with tempfile.TemporaryDirectory() as directory:
                moved = self.copy(Path(directory))
                manifest = deepcopy(original)
                change(manifest)
                (moved/'batch.json').write_text(json.dumps(manifest))
                with self.assertRaises(AppError):
                    verify_batch(moved, toolchain=TOOLS)

    def test_exact_mapping_and_source_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            moved = self.copy(Path(directory))
            path = moved/'candidates/000001/media/review.json'
            record = json.loads(path.read_text())
            record['frames'][1]['review_pts'] += 1
            path.write_text(json.dumps(record))
            with self.assertRaises(AppError):
                verify_batch(moved, toolchain=TOOLS)
            wrong = Path(directory)/'wrong.mp4'
            wrong.write_bytes(self.source.read_bytes()+b'extra')
            with self.assertRaises(AppError):
                verify_batch(self.root/'batch', source=wrong, toolchain=TOOLS)

    def test_actual_cli_success_error_cancel_retry(self):
        args = [sys.executable, '-m', 'scripts.verify_bundle', '--bundle', str(self.root/'batch'), '--input', str(self.source)]
        result = subprocess.run(args, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertNotIn(str(self.root), result.stdout)
        with patch('cheating_racer_detect.events.verify._full_decode', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            verify_batch(self.root/'batch', toolchain=TOOLS)
        self.assertEqual(verify_batch(self.root/'batch', toolchain=TOOLS)['status'], 'verified')
        bad = subprocess.run([sys.executable, '-m', 'scripts.verify_bundle', '--bundle', str(self.root/'missing')],
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(bad.returncode, 1)

    def test_catalog_candidate_hash_and_lamp_side_evidence_changes(self):
        for name, change in (('candidate.json', lambda v: v['candidate'].update(config_id='different')),
                             ('candidate.json', lambda v: v['lamp']['right'].update(valid_samples=0)),
                             ('media/review.json', lambda v: v['frames'][0]['evidence'].update(annotations=[]))):
            with self.subTest(file=name), tempfile.TemporaryDirectory() as directory:
                moved = self.copy(Path(directory))
                path = moved/'candidates/000001'/name
                value = json.loads(path.read_text())
                change(value)
                path.write_text(json.dumps(value))
                with self.assertRaises(AppError):
                    verify_batch(moved, toolchain=TOOLS)

    def test_actual_audio_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'audio-rotation.mp4'
            run_tool([TOOLS.ffmpeg, '-nostdin', '-v', 'error', '-n', '-display_rotation:v:0', '270',
                      '-i', self.source, '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=4.4',
                      '-c:v', 'copy', '-c:a', 'aac', '-t', '4.4', source])
            video, timeline, frames, first, rotation = prepare(source, '0', '4.4', TOOLS)
            rows = []
            for n, (old, item) in enumerate(zip(self.inputs.rows, frames)):
                frame = Frame(_sha256(source), video['index'], first+n, item.pts, timeline.time_base,
                              timeline.frames[0].pts, video['width'], video['height'], rotation)
                rows.append(replace(old, frame=frame, vehicle=replace(old.vehicle, frame=frame,
                            detection=replace(old.vehicle.detection, frame=frame)), lanes=replace(old.lanes, frame=frame)))
            inputs = replace(self.inputs, rows=tuple(rows))
            export_replay_batch(source, inputs, root/'batch', candidate_indices=(0,), toolchain=TOOLS)
            self.assertEqual(verify_batch(root/'batch', source=source, toolchain=TOOLS)['status'], 'verified')

    def test_recurring_rational_context_and_offset_source_pts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'clock.mp4'
            run_tool([TOOLS.ffmpeg, '-nostdin', '-v', 'error', '-n', '-copyts', '-itsoffset', '5',
                      '-i', self.source, '-c:v', 'copy', '-video_track_timescale', '12288',
                      '-avoid_negative_ts', 'disabled', source])
            _video, _audio, timeline = inspect_source(source, TOOLS)
            video, timeline, frames, first, rotation = prepare(source, '0', timeline.end, TOOLS)
            self.assertEqual(timeline.origin, 5)
            self.assertTrue(any((f.pts-timeline.frames[0].pts)*timeline.time_base != old.frame.time
                                for f, old in zip(frames, self.inputs.rows)))
            rows = []
            for n, (old, item) in enumerate(zip(self.inputs.rows, frames)):
                frame = Frame(_sha256(source), video['index'], first+n, item.pts, timeline.time_base,
                              timeline.frames[0].pts, video['width'], video['height'], rotation)
                rows.append(replace(old, frame=frame, vehicle=replace(old.vehicle, frame=frame,
                            detection=replace(old.vehicle.detection, frame=frame)), lanes=replace(old.lanes, frame=frame)))
            inputs = replace(self.inputs, rows=tuple(rows))
            export_replay_batch(source, inputs, root/'batch', candidate_indices=(0,), toolchain=TOOLS)
            self.assertEqual(verify_batch(root/'batch', source=source, toolchain=TOOLS)['status'], 'verified')


if __name__ == '__main__':
    unittest.main()
