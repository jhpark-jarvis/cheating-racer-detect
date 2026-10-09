"""Source-only replay of a local recorded experiment; NOT automatic recognition."""

import argparse
import json
from pathlib import Path

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events.inputs import load_inputs
from cheating_racer_detect.events.replay import export_replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Original local MP4')
    parser.add_argument('--inputs', required=True, type=Path, help='Recorded inputs.json')
    parser.add_argument('--output', required=True, type=Path, help='New output folder')
    parser.add_argument('--candidate-index', required=True, type=int, help='Zero-based selected replay candidate')
    args = parser.parse_args(argv)
    try:
        result = export_replay(args.input, load_inputs(args.inputs), args.output, candidate_index=args.candidate_index)
        print(json.dumps({'status': result['status'], 'candidate_index': result['candidate_index'],
                          'signal_state': result['signal_state'], 'scope': 'recorded inputs; not automatic recognition'}))
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
