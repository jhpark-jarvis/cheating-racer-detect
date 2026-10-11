"""Source-only replay of a local recorded experiment; NOT automatic recognition."""

import argparse
import json
from pathlib import Path

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events.inputs import load_inputs
from cheating_racer_detect.events.replay import export_replay, export_replay_batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Original local MP4')
    parser.add_argument('--inputs', required=True, type=Path, help='Recorded inputs.json')
    parser.add_argument('--output', required=True, type=Path, help='New output folder')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--candidate-index', type=int, help='Zero-based selected replay candidate')
    selection.add_argument('--candidate-indices', nargs='+', type=int, help='Explicit zero-based batch selection, in output order')
    args = parser.parse_args(argv)
    try:
        inputs = load_inputs(args.inputs)
        if args.candidate_indices is None:
            result = export_replay(args.input, inputs, args.output, candidate_index=args.candidate_index)
            summary = {key: result[key] for key in ('status', 'candidate_index', 'signal_state')}
        else:
            result = export_replay_batch(args.input, inputs, args.output, candidate_indices=args.candidate_indices)
            summary = {key: result[key] for key in ('status', 'candidate_indices', 'selected_count')}
        print(json.dumps({**summary, 'scope': 'recorded inputs; not automatic recognition'}))
        return 0
    except AppError as error:
        print(json.dumps({'status': 'error', 'code': error.code}, ensure_ascii=False))
        return 1
    except (TypeError, ValueError):
        print(json.dumps({'status': 'error', 'code': 'INVALID_INPUTS'}))
        return 1
    except KeyboardInterrupt:
        print(json.dumps({'status': 'cancelled', 'code': 'CANCELLED'}))
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
