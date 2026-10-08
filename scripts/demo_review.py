"""Non-sensitive synthetic demonstration, not automatic vehicle detection.

Run from repository root: python -m scripts.demo_review --output NEW_FOLDER
Requires existing FFmpeg and the optional tracking dependencies. No downloads.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import tempfile

from cheating_racer_detect.paths import local_path
from cheating_racer_detect.review import export_review
from cheating_racer_detect.tools import discover_tools
from cheating_racer_detect.tracking import Box, Detection, MotionConfig, Policy, Tracker
from tests.fixtures import generate


def observer():
    # Synthetic fixture values, NOT calibrated driving thresholds.
    motion = MotionConfig((4., 4., .02, .02), (.25, .25, .0025, .0025),
                          (1., 1., .1, .1, 100., 100., 1., 1.))
    tracker = Tracker(Policy(Fraction(3, 5), Fraction(3, 10), .2, .7, .2, .95, (2, 5, 7), motion))
    def observe(frame, pixels):
        detections = [Detection(2, 5, Box(60, 35, 80, 55), .9, 'scripted-demo', frame)]
        if frame.ordinal != 3 and not 1 <= frame.time < 2:
            detections.append(Detection(1, 2, Box(10, 10, 30, 30), .9, 'scripted-demo', frame))
        return tracker.update(frame, detections)
    return observe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--profile', choices=('cfr', 'vfr', 'offset', 'rotation90', 'rotation180', 'rotation270'), default='vfr')
    args = parser.parse_args()
    output = local_path(args.output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('Choose a new folder in an existing local parent')
    tools = discover_tools()
    with tempfile.TemporaryDirectory(prefix='crd-review-demo-') as directory:
        paths = generate(tools, Path(directory))
        report = export_review(paths[args.profile], '.05', '2.81', output, observer(), tools)
    print(json.dumps({'status': report['status'], 'profile': args.profile,
                      'review': str(output/'review.mp4'), 'frames': report['review']['frame_count'],
                      'scope': 'scripted synthetic demonstration, not detection accuracy'}))


if __name__ == '__main__':
    main()
