"""Candidate transaction guards with mocked media, not encoding proof."""

from dataclasses import replace
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import Candidate
from cheating_racer_detect.events import export
from cheating_racer_detect.tools import Toolchain
from tests.test_events_lane import frame, vehicle
from tests.test_tracking import FakeMotion, policy
from cheating_racer_detect.tracking import Tracker


class Transactions(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='crd-candidate-unit-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root/'source.mp4'
        self.source.write_bytes(b'synthetic unit only; media mocked')
        self.output = self.root/'result'
        source_hash = export._sha256(self.source)
        self.frames = [frame(n, source_id=source_hash) for n in range(10)]
        f = self.frames
        self.candidate = Candidate((1, 2, 1), 'road', ('a', 'b'), ('b', 'c'), 'right',
                                   (f[1], f[2]), (f[3], f[4]), (f[5], f[7]), 'policy')
        self.tools = Toolchain(Path('ffmpeg'), Path('ffprobe'))
        timeline = SimpleNamespace(time_base=F(1, 10), frames=[SimpleNamespace(pts=f.pts, duration=1) for f in self.frames])
        self.preflight = ({'index': 0, 'width': 96, 'height': 64}, timeline, [], 0, 0)
        self.tracker = Tracker(policy(), motion_factory=FakeMotion)

    def observe(self, f, p):
        return self.tracker.update(f, [vehicle(f).detection])

    def media(self, source, start, end, output, observe, tools):
        output.mkdir()
        (output/'review.mp4').write_bytes(b'mocked review')
        for f in self.frames:
            if F(start) <= f.time < F(end):
                observe(f, b'')
        return {'source': {'sha256': export._sha256(source)}}

    def invoke(self, candidate=None, **kwargs):
        return export.export_candidate(self.source, candidate or self.candidate, self.output, self.observe,
                                       pre=F(1, 5), post=F(1, 5), toolchain=self.tools, **kwargs)

    def test_atomic_bundle_and_unknown_without_lamps(self):
        with patch.object(export, 'prepare', return_value=self.preflight), patch.object(export, 'export_review', side_effect=self.media):
            result = self.invoke()
        self.assertEqual(result['signal_state'], 'unknown')
        self.assertEqual(result['context'], ['0', '9/10'])
        self.assertEqual(json.loads((self.output/'candidate.json').read_text(encoding='utf-8')), result)
        self.assertEqual(list(self.root.glob('.crd-candidate-*.partial')), [])

    def test_wrong_source_pts_clock_geometry_and_existing_before_export(self):
        for field, value in (('source_id', 'other'), ('pts', 109), ('time_base', F(1, 20)), ('width', 100), ('display_rotation', 90)):
            pairs = {name: tuple(replace(f, **{field: value}) for f in getattr(self.candidate, name)) for name in ('start', 'crossing', 'completion')}
            candidate = replace(self.candidate, **pairs)
            with patch.object(export, 'prepare', return_value=self.preflight), patch.object(export, 'export_review') as run:
                with self.assertRaises(AppError):
                    self.invoke(candidate)
                run.assert_not_called()
        self.output.mkdir()
        (self.output/'keep').write_text('user')
        with self.assertRaises(AppError) as caught:
            self.invoke()
        self.assertEqual(caught.exception.code, 'OUTPUT_EXISTS')
        self.assertEqual((self.output/'keep').read_text(), 'user')

    def test_errors_cancel_json_io_cleanup_and_retry(self):
        for error in (KeyboardInterrupt, OSError, RuntimeError):
            def fail(*args):
                args[3].mkdir()
                raise error('injected failure')
            with patch.object(export, 'prepare', return_value=self.preflight), patch.object(export, 'export_review', side_effect=fail):
                with self.assertRaises((KeyboardInterrupt, RuntimeError, AppError)):
                    self.invoke()
            self.assertFalse(self.output.exists())
            self.assertEqual(list(self.root.glob('.crd-candidate-*.partial')), [])
        with patch.object(export, 'prepare', return_value=self.preflight), patch.object(export, 'export_review', side_effect=self.media), patch.object(export.json, 'dump', side_effect=OSError('full')):
            with self.assertRaises(AppError):
                self.invoke()
        self.assertFalse(self.output.exists())
        self.tracker = Tracker(policy(), motion_factory=FakeMotion)
        with patch.object(export, 'prepare', return_value=self.preflight), patch.object(export, 'export_review', side_effect=self.media):
            self.invoke()

    def test_missing_or_prediction_anchor_and_incomplete_mapping(self):
        for mode in ('prediction', 'missing'):
            def media(*args):
                args[3].mkdir()
                if mode == 'prediction':
                    for f in self.frames:
                        if F(args[1]) <= f.time < F(args[2]):
                            args[4](f, b'')
                return {'source': {'sha256': export._sha256(self.source)}}
            if mode == 'prediction':
                self.observe = lambda f, p: self.tracker.update(f, [])
            with patch.object(export, 'prepare', return_value=self.preflight), patch.object(export, 'export_review', side_effect=media):
                with self.assertRaises(AppError) as caught:
                    self.invoke()
            self.assertEqual(caught.exception.code, 'INVALID_CANDIDATE')
            self.assertFalse(self.output.exists())

    def test_source_change_publish_race_and_unowned_cleanup(self):
        def changed(*args):
            result = self.media(*args)
            self.source.write_bytes(b'changed by unit fixture')
            return result
        with patch.object(export, 'prepare', return_value=self.preflight), patch.object(export, 'export_review', side_effect=changed):
            with self.assertRaises(AppError) as caught:
                self.invoke()
        self.assertEqual(caught.exception.code, 'INPUT_CHANGED')
        self.assertEqual(self.source.read_bytes(), b'changed by unit fixture')

    def test_publish_race_and_parent_ownership_preserved(self):
        def race(*args):
            result = self.media(*args)
            self.output.mkdir()
            (self.output/'keep').write_text('other job')
            return result
        with patch.object(export, 'prepare', return_value=self.preflight), patch.object(export, 'export_review', side_effect=race):
            with self.assertRaises(AppError) as caught:
                self.invoke()
        self.assertEqual(caught.exception.code, 'OUTPUT_EXISTS')
        self.assertEqual((self.output/'keep').read_text(), 'other job')
        self.output = self.root/'second'
        self.tracker = Tracker(policy(), motion_factory=FakeMotion)
        actual = export._identity
        seen = 0
        def changed(path):
            nonlocal seen
            result = actual(path)
            if path == self.root:
                seen += 1
                if seen > 1:
                    return result[0], result[1]+1, *result[2:]
            return result
        with patch.object(export, 'prepare', return_value=self.preflight), patch.object(export, 'export_review', side_effect=self.media), patch.object(export, '_identity', side_effect=changed):
            with self.assertRaises(AppError) as caught:
                self.invoke()
        self.assertEqual(caught.exception.code, 'CLEANUP_FAILED')
        self.assertEqual(len(list(self.root.glob('.crd-candidate-*.partial'))), 1)
        # Entire unit TemporaryDirectory is test-owned; never remove unowned user paths.


if __name__ == '__main__':
    unittest.main()
