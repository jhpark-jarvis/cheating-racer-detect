"""Four fictional eligible events: one miss and unknown are never hidden."""

import json

from cheating_racer_detect.events.assessment import from_report


def definition():
    truths = [{'event_id': 'g'+str(n), 'source_id': 'synthetic', 'segment_id': 'session', 'vehicle_id': 'gt-car',
               'earliest': str(n), 'latest': str(n), 'direction': 'right' if n < 4 else 'left', 'signal_state': state}
              for n, state in enumerate(('right_blink', 'unknown', 'right_blink', 'left_blink'), 1)]
    predictions = [{'event_id': 'p'+str(n), 'source_id': 'synthetic', 'segment_id': 'session', 'vehicle_id': 'track',
                    'time': str(n), 'direction': 'right', 'abstained': False, 'signal_state': state}
                   for n, state in enumerate(('right_blink', 'no_blink_observed', 'unknown'), 1)]
    return {'schema': 'experimental-assessment-inputs-v1', 'truths': truths, 'predictions': predictions,
            'correspondence': [{'source_id': 'synthetic', 'segment_id': 'session',
                                'prediction_vehicle_id': 'track', 'truth_vehicle_id': 'gt-car'}],
            'tolerance': '1/10', 'max_gt_width': '1/5'}


if __name__ == '__main__':
    print(json.dumps(from_report(definition())))
