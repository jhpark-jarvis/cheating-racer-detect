"""List source-verified recorded candidates before explicitly selecting media.

Source-only experimental command, not automatic recognition or ground truth.
"""

import argparse
import json
from pathlib import Path

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events.catalog import export_catalog, export_catalog_with_diagnostics, listing_summary
from cheating_racer_detect.events.catalog import export_catalog_with_lamp_diagnostics
from cheating_racer_detect.events.inputs import load_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Original local MP4')
    parser.add_argument('--inputs', required=True, type=Path, help='Recorded inputs.json')
    parser.add_argument('--output', required=True, type=Path, help='New catalogue folder (no media)')
    parser.add_argument('--lane-diagnostics', action='store_true', help='Include lane source/unknown/reset diagnostics')
    parser.add_argument('--lamp-diagnostics', action='store_true', help='Include lamp declaration/ROI/gap/reset diagnostics')
    args = parser.parse_args(argv)
    try:
        inputs = load_inputs(args.inputs)
        if args.lamp_diagnostics:
            catalog, lamps, lanes = export_catalog_with_lamp_diagnostics(
                args.input, inputs, args.output, include_lane_diagnostics=args.lane_diagnostics)
            result = listing_summary(catalog, lanes, lamps)
        elif args.lane_diagnostics:
            catalog, diagnostics = export_catalog_with_diagnostics(args.input, inputs, args.output)
            result = listing_summary(catalog, diagnostics)
        else:
            result = listing_summary(export_catalog(args.input, inputs, args.output))
        print(json.dumps(result, ensure_ascii=False))
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
