"""Public OpenCV API wrapper around equation-derived variable-dt motion.

cx/cy/log(width)/log(height) and their per-second velocities. No tracker source
or native OpenCV implementation is embedded here. This module is loaded lazily.
"""

import math
from fractions import Fraction

import cv2
import numpy as np

from .contracts import Box, MAX_DT, MotionConfig


def vector(values, size):
    result = np.array(values, dtype=np.float64, copy=True)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("Invalid finite vector")
    return result


def checked_covariance(values):
    result = np.array(values, dtype=np.float64, copy=True)
    if (result.shape != (8, 8) or not np.isfinite(result).all()
            or not np.allclose(result, result.T, rtol=0, atol=1e-10)):
        raise ValueError("Invalid finite symmetric covariance")
    result = (result + result.T) / 2
    if np.linalg.eigvalsh(result).min() < -1e-10:
        raise ValueError("Covariance is not positive semidefinite")
    return result


def matrices(dt, spectral):
    if not isinstance(dt, Fraction) or not 0 < dt <= MAX_DT:
        raise ValueError("dt must be exact Fraction within the validated bound")
    q = vector(spectral, 4)
    if (q < 0).any():
        raise ValueError("Negative spectral density")
    t = float(dt)
    a, noise = np.eye(8), np.zeros((8, 8))
    a[:4, 4:] = np.eye(4) * t
    # Integral of [t-s,1] [t-s,1]^T over s in [0,t] for each independent channel.
    noise[:4, :4] = np.diag(q * t**3 / 3)
    noise[:4, 4:] = noise[4:, :4] = np.diag(q * t**2 / 2)
    noise[4:, 4:] = np.diag(q * t)
    return a, noise


def measurement(box):
    if not isinstance(box, Box):
        raise ValueError("Expected Box")
    return vector([box.x1/2 + box.x2/2, box.y1/2 + box.y2/2,
                   math.log(box.x2-box.x1), math.log(box.y2-box.y1)], 4)


def estimate(state):
    state = vector(state, 8)
    try:
        w, h = math.exp(state[2]), math.exp(state[3])
    except OverflowError as error:
        raise ValueError("Extent overflow") from error
    if not math.isfinite(w+h) or min(w, h) <= 0:
        raise ValueError("Invalid extent")
    return Box(state[0]-w/2, state[1]-h/2, state[0]+w/2, state[1]+h/2)


class OpenCVMotion:
    def __init__(self, box, config):
        if not isinstance(config, MotionConfig):
            raise ValueError("Expected MotionConfig")
        self._spectral = np.array(config.spectral, dtype=np.float64)
        self._state = np.concatenate((measurement(box), np.zeros(4)))
        self._covariance = np.diag(config.initial_variance)
        self._native = cv2.KalmanFilter(8, 4, 0, cv2.CV_64F)
        self._native.measurementMatrix = np.concatenate((np.eye(4), np.zeros((4, 4))), axis=1)
        self._native.measurementNoiseCov = np.diag(config.measurement_variance)
        self._native.statePost = self._state.reshape(8, 1).copy()
        self._native.errorCovPost = self._covariance.copy()
        self._failed = self._pending = False

    def _alive(self):
        if self._failed:
            raise RuntimeError("Motion failed; construct a new instance")

    def snapshot(self):
        self._alive()
        return self._state.copy(), self._covariance.copy()

    @property
    def box(self):
        self._alive()
        return estimate(self._state)

    def _accept(self, state, covariance):
        values, uncertainty = vector(state.reshape(8), 8), checked_covariance(covariance)
        estimate(values)
        self._state, self._covariance = values, uncertainty

    def predict(self, dt):
        self._alive()
        a, q = matrices(dt, self._spectral)
        try:
            self._native.transitionMatrix, self._native.processNoiseCov = a, q
            self._accept(self._native.predict(), self._native.errorCovPre)
            self._pending = True
        except BaseException:
            self._failed = True
            raise

    def correct(self, box):
        self._alive()
        if not self._pending:
            raise ValueError("Correction requires one preceding prediction")
        values = measurement(box)
        try:
            self._accept(self._native.correct(values.reshape(4, 1)), self._native.errorCovPost)
            self._pending = False
        except BaseException:
            self._failed = True
            raise
