"""Read an explicit experiment definition; emit exact event/signal metrics."""

import argparse
import json
from pathlib import Path

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events.assessment import from_report
from cheating_racer_detect.events.verify import Files
from cheating_racer_detect.paths import local_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        path = local_path(args.labels)
        files = Files(path.parent)
        result = from_report(files.json(path.name))
        files.finish()
        print(json.dumps(result))
        return 0
    except (AppError, ValueError, TypeError, KeyError, AttributeError, RecursionError, OverflowError, ZeroDivisionError, UnicodeError):
        print(json.dumps({'status': 'error', 'code': 'INVALID_ASSESSMENT'}))
        return 1
    except OSError:
        print(json.dumps({'status': 'error', 'code': 'IO_ERROR'}))
        return 1
    except KeyboardInterrupt:
        print(json.dumps({'status': 'cancelled', 'code': 'CANCELLED'}))
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
