"""Read-only saved batch audit; optional original source replays annotations."""

import argparse
import json
from pathlib import Path

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events.verify import verify_batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--input', type=Path, help='Optional original MP4 for fresh replay')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify_batch(args.bundle, source=args.input), ensure_ascii=False))
        return 0
    except AppError as error:
        print(json.dumps({'status': 'error', 'code': error.code}))
        return 1
    except KeyboardInterrupt:
        print(json.dumps({'status': 'cancelled', 'code': 'CANCELLED'}))
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
