"""Bounded original-ROI brightness histories, not a trained lamp recognizer."""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import math

from ..tracking.contracts import Box, finite, identifier, integer
from .contracts import boundary, current, seconds

REASONS = ('usable', 'occluded', 'out_of_view', 'too_small', 'identity_uncertain', 'model_uncertain')


@dataclass(frozen=True)
class LampROI:
    """Vehicle-relative side; usable also asserts caller-supplied signal semantics."""
    box: Box | None
    status: str

    def __post_init__(self):
        if self.status not in REASONS or (self.box is not None and not isinstance(self.box, Box)):
            raise ValueError('Invalid lamp ROI')
        if self.status == 'usable' and self.box is None:
            raise ValueError('Usable needs an actual ROI')


@dataclass(frozen=True)
class LampConfig:
    config_id: str
    off_threshold: float
    on_threshold: float
    min_pixels: int
    min_window: Fraction
    max_gap: Fraction
    min_phase: Fraction
    max_phase: Fraction
    min_cycles: int

    def __post_init__(self):
        identifier(self.config_id)
        if not 0 <= finite(self.off_threshold) < finite(self.on_threshold) <= 255:
            raise ValueError('Invalid brightness hysteresis')
        integer(self.min_pixels, minimum=1)
        integer(self.min_cycles, minimum=1)
        if self.min_pixels > 65536 or self.min_cycles > 8:
            raise ValueError('Lamp policy bound exceeded')
        for value in (self.min_window, self.max_gap, self.min_phase, self.max_phase):
            seconds(value, True)
        if self.max_phase < self.min_phase:
            raise ValueError('Invalid phase durations')

    def report(self):
        return {key: str(value) if isinstance(value, Fraction) else value
                for key, value in vars(self).items()}


class LampEngine:
    """Each update must be consecutive decoded current-frame actual observation.

    Validity/side/semantic gates are provided, never inferred from brightness.
    Entire inclusive window must be covered by consecutive usable observations.
    """
    def __init__(self, config):
        if not isinstance(config, LampConfig):
            raise ValueError('Explicit LampConfig required')
        self.config = config
        self.reset()

    def reset(self):
        self._previous = self._identity = None
        self._samples = deque(maxlen=120)
        self._reason = 'start'

    def update(self, frame, vehicle, pixels, left, right):
        try:
            return self._update(frame, vehicle, pixels, left, right)
        except BaseException:
            self.reset()
            raise

    def _roi(self, vehicle, pixels, roi):
        if not isinstance(roi, LampROI):
            raise ValueError('Explicit LampROI required')
        if roi.box is not None:
            b, v = roi.box, vehicle.box
            if not (v.x1 <= b.x1 < b.x2 <= v.x2 and v.y1 <= b.y1 < b.y2 <= v.y2):
                raise ValueError('Lamp ROI outside actual observed vehicle')
        if roi.status != 'usable':
            return {'status': roi.status, 'brightness': None, 'roi': None if roi.box is None else list(roi.box.values())}
        b = roi.box
        # Include only whole original pixels strictly within the supplied ROI.
        x1, y1, x2, y2 = math.ceil(b.x1), math.ceil(b.y1), math.floor(b.x2), math.floor(b.y2)
        count = max(0, x2-x1)*max(0, y2-y1)
        if count > 65536:
            raise ValueError('Lamp ROI resource bound exceeded')
        if count < self.config.min_pixels:
            return {'status': 'too_small', 'brightness': None, 'roi': list(b.values())}
        width = vehicle.frame.width
        total = sum(sum(pixels[(y*width+x1)*3:(y*width+x2)*3]) for y in range(y1, y2))
        return {'status': 'usable', 'brightness': Fraction(total, count*3), 'roi': list(b.values())}

    def _update(self, frame, vehicle, pixels, left, right):
        current(frame, vehicle)
        if not isinstance(pixels, bytes) or len(pixels) != frame.width*frame.height*3:
            raise ValueError('Expected immutable same-frame coded BGR bytes')
        reason = boundary(self._previous, frame, self.config.max_gap)
        if vehicle is None:
            self.reset()
            self._reason = 'no_actual_observation'
            return {'status': 'unknown', 'reason': self._reason}
        if reason or self._identity != vehicle.identity:
            self.reset()
            self._reason = reason or 'identity_change'
        if any(isinstance(roi, LampROI) and roi.status == 'identity_uncertain' for roi in (left, right)):
            self.reset()
            self._reason = 'identity_uncertain'
        sample = {'frame': frame, 'left': self._roi(vehicle, pixels, left),
                  'right': self._roi(vehicle, pixels, right), 'vehicle': vehicle.report()}
        self._samples.append(sample)
        self._previous, self._identity = frame, vehicle.identity
        return {'status': 'recorded', 'reason': self._reason, 'sample_count': len(self._samples)}

    def _side(self, samples, side, complete, start, end):
        reasons = sorted({s[side]['status'] for s in samples if s[side]['status'] != 'usable'})
        if not complete:
            reasons.append('insufficient_history')
        phase = None
        transitions, cycles = [], []
        for sample in samples:
            item, time = sample[side], sample['frame'].time
            if item['status'] != 'usable':
                phase = None
                transitions = []
                continue
            value = item['brightness']
            state = 'on' if value >= self.config.on_threshold else 'off' if value <= self.config.off_threshold else phase
            if state is None:
                if 'model_uncertain' not in reasons:
                    reasons.append('model_uncertain')
                continue
            if state != phase:
                if phase is not None:
                    transitions.append((state, time))
                phase = state
            if len(transitions) >= 3:
                a, b, c = transitions[-3:]
                if (a[0], b[0], c[0]) == ('on', 'off', 'on') and (not cycles or cycles[-1][1] != c[1]):
                    if (start <= a[1] and c[1] <= end
                            and all(self.config.min_phase <= d <= self.config.max_phase for d in (b[1]-a[1], c[1]-b[1]))):
                        cycles.append((a[1], c[1]))
        blink = len(cycles) >= self.config.min_cycles
        status = 'unknown' if reasons else 'blink_observed' if blink else 'no_blink_observed'
        return {'state': status, 'blink_evidence': blink, 'complete_cycles': [[str(a), str(b)] for a, b in cycles],
                'reasons': sorted(set(reasons)), 'valid_samples': sum(s[side]['status'] == 'usable' for s in samples)}

    def report(self, start, end):
        seconds(start)
        seconds(end)
        if end <= start:
            raise ValueError('Expected positive inclusive observation window')
        # Bracket window endpoints with nearest actual samples, no interpolation.
        all_samples = list(self._samples)
        before = [i for i, s in enumerate(all_samples) if s['frame'].time <= start]
        after = [i for i, s in enumerate(all_samples) if s['frame'].time >= end]
        complete = bool(before and after and end-start >= self.config.min_window)
        samples = all_samples[before[-1] if before else 0:(after[0]+1 if after else len(all_samples))]
        left = self._side(samples, 'left', complete, start, end)
        right = self._side(samples, 'right', complete, start, end)
        if 'unknown' in (left['state'], right['state']):
            state = 'unknown'
        elif left['blink_evidence'] and right['blink_evidence']:
            state = 'both_blink'
        elif left['blink_evidence']:
            state = 'left_blink'
        elif right['blink_evidence']:
            state = 'right_blink'
        else:
            state = 'no_blink_observed'
        return {'schema': 'experimental-lamp-window-v1', 'window': [str(start), str(end)],
                'identity': None if self._identity is None else list(self._identity), 'identity_status': 'unverified',
                'signal_state': state, 'left': left, 'right': right, 'history_start_reason': self._reason,
                'config': self.config.report(), 'samples': [{'vehicle': s['vehicle'],
                    **{side: {**s[side], 'brightness': None if s[side]['brightness'] is None else str(s[side]['brightness'])}
                       for side in ('left', 'right')}} for s in samples],
                'limits': ['provided ROI/visibility/signal-semantic gates', 'brightness pattern, not signal recognition',
                           'no_blink only within configured observed window', 'not a violation decision']}
