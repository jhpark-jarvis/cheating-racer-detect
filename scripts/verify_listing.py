"""Read-only listing/diagnostic audit, optionally with fresh original replay."""

import argparse
import json
from pathlib import Path

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events.listing import verify_listing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listing', required=True, type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--require-diagnostics', nargs='+', choices=('lane', 'lamp'), default=[])
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify_listing(args.listing, source=args.input,
            required_diagnostics=args.require_diagnostics), ensure_ascii=False))
        return 0
    except AppError as error:
        print(json.dumps({'status': 'error', 'code': error.code}))
        return 1
    except KeyboardInterrupt:
        print(json.dumps({'status': 'cancelled', 'code': 'CANCELLED'}))
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
