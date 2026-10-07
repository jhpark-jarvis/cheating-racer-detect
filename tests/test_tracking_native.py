"""Actual installed NumPy/OpenCV motion; missing dependencies are NOT_RUN."""

from fractions import Fraction
from importlib.util import find_spec
import unittest
from unittest.mock import patch

from cheating_racer_detect.tracking import Box, Tracker
from tests.test_tracking import config, detection, frame, policy


class NativeTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if find_spec("numpy") is None or find_spec("cv2") is None:
            raise unittest.SkipTest("NOT_RUN: optional tracking NumPy/OpenCV unavailable")
        import numpy as np
        from cheating_racer_detect.tracking.motion import OpenCVMotion, matrices
        cls.np, cls.motion, cls.matrices = np, OpenCVMotion, staticmethod(matrices)

    def test_analytic_transition_and_time_partition_noise(self):
        a, q = self.matrices(Fraction(1, 2), (3, 6, 9, 12))
        self.np.testing.assert_allclose(a[:4, 4:], self.np.eye(4)/2, rtol=0, atol=1e-10)
        self.np.testing.assert_allclose(q[:4, :4], self.np.diag([1/8, 1/4, 3/8, 1/2]), rtol=0, atol=1e-10)
        for first, second in ((Fraction(1, 1000), Fraction(1, 2)), (Fraction(1, 10), Fraction(1, 5))):
            a1, q1 = self.matrices(first, config().spectral)
            a2, q2 = self.matrices(second, config().spectral)
            whole, total = self.matrices(first+second, config().spectral)
            self.np.testing.assert_allclose(whole, a2 @ a1, rtol=0, atol=1e-10)
            self.np.testing.assert_allclose(total, a2 @ q1 @ a2.T + q2, rtol=0, atol=1e-10)

    def test_box_roundtrip_and_owned_snapshots(self):
        for box in (Box(1, 2, 21, 14), Box(-20, -.5, 0, 2.25), Box(0, 0, .01, .02)):
            model = self.motion(box, config())
            self.np.testing.assert_allclose(model.box.values(), box.values(), rtol=0, atol=1e-10)
            state, covariance = model.snapshot()
            state[:] = covariance[:] = -100
            self.np.testing.assert_allclose(model.box.values(), box.values(), rtol=0, atol=1e-10)

    def test_actual_zero_velocity_prediction_then_correction(self):
        model = self.motion(Box(10, 10, 30, 30), config())
        model.predict(Fraction(1, 2))
        self.np.testing.assert_allclose(model.box.values(), [10, 10, 30, 30], atol=1e-10)
        model.correct(Box(15, 10, 35, 30))
        self.assertGreater(model.snapshot()[0][4], 0)
        before = model.box.x1
        model.predict(Fraction(1, 5))
        self.assertGreater(model.box.x1, before)
        self.assertGreaterEqual(self.np.linalg.eigvalsh(model.snapshot()[1]).min(), -1e-10)

    def test_hand_derived_correction(self):
        from cheating_racer_detect.tracking import MotionConfig
        model = self.motion(Box(-.5, -.5, .5, .5), MotionConfig((0,)*4, (1,)*4, (1,)*8))
        model.predict(Fraction(1, 2))
        import math
        extent = math.e
        model.correct(Box(1-extent/2, 1-extent/2, 1+extent/2, 1+extent/2))
        state, covariance = model.snapshot()
        self.np.testing.assert_allclose(state[:4], self.np.ones(4)*5/9, atol=1e-10)
        self.np.testing.assert_allclose(state[4:], self.np.ones(4)*2/9, atol=1e-10)
        self.np.testing.assert_allclose(covariance[4:, 4:], self.np.eye(4)*8/9, atol=1e-10)

    def test_invalid_dt_and_correction_protocol(self):
        model = self.motion(Box(10, 10, 30, 30), config())
        for dt in (.1, Fraction(0), Fraction(-1), Fraction(7, 10)):
            with self.assertRaises(ValueError):
                model.predict(dt)
        with self.assertRaises(ValueError):
            model.correct(model.box)
        model.predict(Fraction(1, 10))
        model.correct(model.box)
        with self.assertRaises(ValueError):
            model.correct(model.box)

    def test_partial_native_failure_poisons_motion_and_tracker(self):
        tracker = Tracker(policy())
        first, second = frame(), frame(1)
        tracker.update(first, [detection(first)])
        model = next(iter(tracker._tracks.values())).motion
        original = model._accept
        def partial(*args):
            original(*args)
            raise KeyboardInterrupt()
        with patch.object(model, "_accept", side_effect=partial):
            with self.assertRaises(KeyboardInterrupt):
                tracker.update(second, [detection(second)])
        with self.assertRaises(RuntimeError):
            model.snapshot()
        restored = frame(2)
        self.assertEqual(tracker.update(restored, [detection(restored)])["reset_reason"], "failed_update")

    def test_actual_variable_cadence_and_short_occlusion_reactivate(self):
        tracker = Tracker(policy(max_unobserved=Fraction(3, 5)))
        results = []
        for ordinal, pts in enumerate((500, 501, 503, 504, 506)):
            source = frame(ordinal, pts)
            values = [] if ordinal == 3 else [detection(source, ordinal, x=10+float(source.time)*5)]
            results.append(tracker.update(source, values))
        self.assertEqual(len({tuple(r["tracks"][0]["identity"]) for r in results}), 1)
        self.assertTrue(all(r["reset_reason"] is None for r in results[1:]))
        self.assertEqual(results[3]["tracks"][0]["state"], "prediction_only")
        self.assertEqual(results[4]["tracks"][0]["state"], "observed")

    def test_actual_ambiguity_terminates_histories(self):
        tracker = Tracker(policy())
        first, second = frame(), frame(1)
        tracker.update(first, [detection(first, 1, x=5), detection(first, 2, x=25)])
        result = tracker.update(second, [detection(second, x=15)])
        self.assertEqual(result["tracks"], [])
        self.assertEqual(len(result["ambiguities"][0]["identities"]), 2)

    def test_actual_expiry_and_low_score_cannot_resurrect_id(self):
        tracker = Tracker(policy())
        source = frame()
        original = tracker.update(source, [detection(source)])["tracks"][0]["identity"]
        for ordinal in (1, 2, 3):
            self.assertEqual(tracker.update(frame(ordinal), [])["tracks"][0]["identity"], original)
        source = frame(4)
        self.assertEqual(tracker.update(source, [detection(source, score=.3)])["tracks"], [])
        source = frame(5)
        self.assertNotEqual(tracker.update(source, [detection(source)])["tracks"][0]["identity"], original)

    def test_actual_correction_failure_and_nonfinite_guards(self):
        from cheating_racer_detect.tracking.motion import checked_covariance, estimate
        model = self.motion(Box(10, 10, 30, 30), config())
        model.predict(Fraction(1, 10))
        with patch.object(model, "_accept", side_effect=RuntimeError("after native correction")):
            with self.assertRaises(RuntimeError):
                model.correct(Box(11, 10, 31, 30))
        with self.assertRaises(RuntimeError):
            model.snapshot()
        for value in (1000, -1000, float("nan")):
            state = self.np.zeros(8)
            state[2] = value
            with self.assertRaises(ValueError):
                estimate(state)
        for covariance in (-self.np.eye(8), self.np.ones((8, 8))*float("nan"), self.np.eye(4)):
            with self.assertRaises(ValueError):
                checked_covariance(covariance)


if __name__ == "__main__":
    unittest.main()
