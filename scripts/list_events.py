"""List source-verified recorded candidates before explicitly selecting media.

Source-only experimental command, not automatic recognition or ground truth.
"""

import argparse
import json
from pathlib import Path

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events.catalog import export_catalog
from cheating_racer_detect.events.inputs import load_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Original local MP4')
    parser.add_argument('--inputs', required=True, type=Path, help='Recorded inputs.json')
    parser.add_argument('--output', required=True, type=Path, help='New catalogue folder (no media)')
    args = parser.parse_args(argv)
    try:
        result = export_catalog(args.input, load_inputs(args.inputs), args.output).report()
        print(json.dumps({'status': result['status'], 'candidate_count': result['candidate_count'],
                          'candidates': [{key: entry[key] for key in ('candidate_index', 'direction', 'start',
                              'crossing', 'completion', 'signal_state', 'left', 'right')} for entry in result['candidates']],
                          'scope': 'recorded inputs; planned context; not automatic recognition'}, ensure_ascii=False))
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
