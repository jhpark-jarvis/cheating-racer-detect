"""Experimental signal assessment over explicitly eligible event labels."""

from fractions import Fraction
from itertools import islice

from .evaluate import Prediction, Truth, evaluate
from .inputs import _exact, _fields

STATES = ('left_blink', 'right_blink', 'both_blink', 'no_blink_observed', 'unknown')


def signals(values, events):
    if type(values) is not dict or set(values) != {e.event_id for e in events}:
        raise ValueError('Explicit signal label required for every eligible event')
    if any(type(s) is not str or s not in STATES for s in values.values()):
        raise ValueError('Unknown signal label')


def ratio(a, b):
    return None if not b else str(Fraction(a, b))


def assess(truths, predictions, correspondence, *, truth_signals, prediction_signals, tolerance, max_gt_width):
    """Every supplied truth is an eligible lane change, even when signal unknown.

    No automatic identity mapping, truth exclusion, observability qualification,
    product pass threshold or FP/hour inferred from case-only clips.
    """
    truths, predictions = tuple(islice(iter(truths), 65)), tuple(islice(iter(predictions), 65))
    events = evaluate(truths, predictions, correspondence, tolerance=tolerance, max_gt_width=max_gt_width)
    signals(truth_signals, truths)
    signals(prediction_signals, predictions)
    matrix = {t: {p: 0 for p in STATES} for t in STATES}
    known_matched = correct = dangerous = known_on_observable_truth = 0
    for pair in events['matches']:
        t, p = truth_signals[pair['truth']], prediction_signals[pair['prediction']]
        matrix[t][p] += 1
        known_matched += p != 'unknown'
        known_on_observable_truth += t != 'unknown' and p != 'unknown'
        correct += t != 'unknown' and t == p
        dangerous += t == 'unknown' and p == 'no_blink_observed'
    truth_known = sum(s != 'unknown' for s in truth_signals.values())
    return {'schema': 'experimental-signal-assessment-v1', 'events': events,
            'signal_confusion_on_matches': matrix, 'eligible_truths': len(truths),
            'truth_known_signal': truth_known, 'truth_observability_ratio': ratio(truth_known, len(truths)),
            'matched_known_predictions': known_matched,
            'end_to_end_known_prediction_coverage': ratio(known_matched, len(truths)),
            'correct_known_signal_matches': correct,
            'end_to_end_correct_signal_coverage': ratio(correct, len(truths)),
            'known_predictions_on_observable_truth': known_on_observable_truth,
            'observable_truth_known_prediction_coverage': ratio(known_on_observable_truth, truth_known),
            'observable_truth_correct_signal_coverage': ratio(correct, truth_known),
            'unknown_truth_to_no_blink': dangerous,
            'limits': ['explicit experimental eligible labels; not final ground-truth schema',
                       'missed truths remain in coverage denominator', 'matrix only over event matches',
                       'unknown is not off', 'no product pass threshold or real-world accuracy evidence']}


def from_report(value):
    """Strict bounded experiment definition; paths/footage are not embedded."""
    _fields(value, ('schema', 'truths', 'predictions', 'correspondence', 'tolerance', 'max_gt_width'))
    if value['schema'] != 'experimental-assessment-inputs-v1':
        raise ValueError('Unknown assessment schema')
    truths, predictions, ts, ps = [], [], {}, {}
    for name, cls, result, labels in (('truths', Truth, truths, ts), ('predictions', Prediction, predictions, ps)):
        rows = value[name]
        if type(rows) is not list or len(rows) > 64:
            raise ValueError('Evaluation bound exceeded')
        for row in rows:
            _fields(row, (*cls.__dataclass_fields__, 'signal_state'))
            data = {k: row[k] for k in cls.__dataclass_fields__}
            for key in ('earliest', 'latest') if cls is Truth else ('time',):
                data[key] = _exact(data[key])
            result.append(cls(**data))
            labels[row['event_id']] = row['signal_state']
    rows = value['correspondence']
    if type(rows) is not list or len(rows) > 64:
        raise ValueError('Correspondence bound exceeded')
    mapping = {}
    for row in rows:
        _fields(row, ('source_id', 'segment_id', 'prediction_vehicle_id', 'truth_vehicle_id'))
        key = row['source_id'], row['segment_id'], row['prediction_vehicle_id']
        if key in mapping:
            raise ValueError('Duplicate correspondence')
        mapping[key] = row['truth_vehicle_id']
    return assess(truths, predictions, mapping, truth_signals=ts, prediction_signals=ps,
                  tolerance=_exact(value['tolerance']), max_gt_width=_exact(value['max_gt_width']))
