"""Public detector contracts and dependency isolation; no private files/models."""

from dataclasses import FrozenInstanceError, replace
from itertools import count
import os
from pathlib import Path
import subprocess
import sys
import unittest

from cheating_racer_detect.analysis import DetectorBatch, DetectorObservation, Letterbox, post_nms_observations
from cheating_racer_detect.tracking import Box
from tests.test_tracking import frame


class DetectorContracts(unittest.TestCase):
    def test_letterbox_actual_odd_axis_inverse_and_owned_report(self):
        geometry = Letterbox(601, 333)
        mapped = geometry.to_original(Box(320, 100, 600, 200))
        expected = (320*601/640, 100*333/geometry.resized_height, 600*601/640, 200*333/geometry.resized_height)
        for actual, target in zip(mapped.values(), expected):
            self.assertAlmostEqual(actual, target, places=12)
        self.assertNotEqual(geometry.resized_width/601, geometry.resized_height/333)
        report = geometry.report()
        report["source_size"][0] = 9
        self.assertEqual(geometry.report()["source_size"], [601, 333])

    def test_letterbox_padding_and_partial_content(self):
        geometry = Letterbox(96, 64)
        self.assertIsNone(geometry.to_original(Box(0, 500, 10, 510)))
        self.assertEqual(geometry.to_original(Box(-10, 0, 640, 640)), Box(0, 0, 96, 64))
        with self.assertRaises(ValueError):
            geometry.to_original([0, 0, 1, 1])

    def test_invalid_letterbox_dimensions(self):
        for sizes in ((True, 64), (0, 64), (16385, 64), (1, 16384, 1, 1), (96, 64, -1, 640)):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                Letterbox(*sizes)

    def test_post_nms_original_index_score_class_and_padding(self):
        observations = post_nms_observations([[0, 0, 10, 10, 1, 1, 0],
            [0, 500, 10, 510, 1, 1, 7], [10, 10, 30, 30, .6, .5, 2]], Letterbox(96, 64), "external:receipt")
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].index, 2)
        self.assertEqual(observations[0].score, .3)
        self.assertEqual(observations[0].class_id, 2)
        self.assertEqual(observations[0].model_id, "external:receipt")

    def test_post_nms_infinite_rows_and_columns_are_bounded(self):
        consumed = []
        def rows():
            for index in count():
                consumed.append(index)
                yield (0, 0, 10, 10, 1, 1, 2)
        with self.assertRaises(ValueError):
            post_nms_observations(rows(), Letterbox(96, 64), "external")
        self.assertEqual(len(consumed), 513)
        with self.assertRaises(ValueError):
            post_nms_observations([count()], Letterbox(96, 64), "external")

    def test_invalid_post_nms_values_and_mapping(self):
        for row in ((0, 0, 1, 1, 1, 1), (0, 0, 0, 1, 1, 1, 2),
                    (0, 0, float("nan"), 1, 1, 1, 2), (0, 0, 1, 1, True, 1, 2),
                    (0, 0, 1, 1, 1.1, 1, 2), (0, 0, 1, 1, 1, 1, 2.5)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                post_nms_observations([row], Letterbox(96, 64), "external")
        for classes in ([], (), (2, 2), (False,), tuple(range(65))):
            with self.assertRaises(ValueError):
                post_nms_observations([], Letterbox(96, 64), "external", classes=classes)

    def test_batch_immutable_and_explicit_frame_geometry_model(self):
        current = frame()
        item = DetectorObservation(1, 2, Box(10, 10, 30, 30), .9, "fixture:"+"a"*64)
        value = DetectorBatch(current, (item,), Letterbox(96, 64), item.model_id)
        self.assertIs(value.frame, current)
        with self.assertRaises(FrozenInstanceError):
            value.model_id = "other"
        for changes in ({"frame": None}, {"observations": [item]}, {"observations": (item,)*65},
                        {"observations": iter((item,))}, {"observations": (item, item)},
                        {"geometry": Letterbox(97, 64)}, {"model_id": "other"},
                        {"observations": (replace(item, box=Box(0, 0, 97, 64)),)}):
            with self.subTest(change=list(changes)), self.assertRaises(ValueError):
                replace(value, **changes)

    def test_invalid_observations_and_external_identifier(self):
        item = DetectorObservation(1, 2, Box(10, 10, 30, 30), .9, "fixture")
        for changes in ({"index": True}, {"class_id": -1}, {"box": object()}, {"score": float("inf")},
                        {"score": True}, {"model_id": "C:/private"}, {"model_id": "x"*161}):
            with self.assertRaises(ValueError):
                replace(item, **changes)


class ImportIsolation(unittest.TestCase):
    def test_no_optional_or_private_imports_and_missing_dependency_errors(self):
        # Fresh interpreter blocks optional and private imports even when installed.
        root = str(Path(__file__).resolve().parents[1])
        code = '''import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','cv2','torch','analysis_adapter','analysis_smoke','detector_tracking','review_overlay'}:
            raise ModuleNotFoundError('blocked optional/private dependency', name=fullname)
sys.meta_path.insert(0, Block())
from cheating_racer_detect.analysis import DetectorBatch, DetectorTrackerBridge, Letterbox
from cheating_racer_detect.review import render, summarize
from cheating_racer_detect.tracking import Frame, Tracker
from tests.test_tracking import FakeMotion, frame, policy
class Detector:
    model_id='scripted'
    def detect(self, current, image):
        raise AssertionError('missing NumPy must fail before detect')
current=frame()
tracker=Tracker(policy(), motion_factory=FakeMotion)
report=tracker.update(current, [])
assert summarize(current, report)['annotations']==[]
bridge=DetectorTrackerBridge(Detector(), tracker, model_alias='alias', expected_detector_id='scripted')
for operation in (lambda: bridge.update(current, bytes(96*64*3)), lambda: render(current, bytes(96*64*3), report)):
    try: operation()
    except RuntimeError as error: assert 'tracking extra' in str(error)
    else: raise AssertionError('missing dependency silently accepted')
assert tracker.update(frame(1), [])['reset_reason']=='failed_update'
assert not any(n.split('.')[0] in {'numpy','cv2','torch','analysis_adapter','analysis_smoke','detector_tracking','review_overlay'} for n in sys.modules)
print('PASS: public-only stdlib imports/summary; optional errors; failed_update')
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = root
        result = subprocess.run([sys.executable, "-c", code], cwd=root, env=environment,
                                capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn("public-only", result.stdout)


if __name__ == "__main__":
    unittest.main()
