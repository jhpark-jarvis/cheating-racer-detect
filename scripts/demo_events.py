"""Self-generated moving-box/lamp fixture -> candidate -> review/original clips.

Source-only demonstration, not automatic vehicle/lane/lamp recognition.
"""

import argparse
from decimal import Decimal
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile

from cheating_racer_detect.events import LaneConfig, LaneEngine, LaneSection, Vehicle
from cheating_racer_detect.events.export import export_candidate
from cheating_racer_detect.events.lamp import LampConfig, LampEngine, LampROI
from cheating_racer_detect.paths import local_path
from cheating_racer_detect.review.video import prepare
from cheating_racer_detect.service import _sha256
from cheating_racer_detect.tools import discover_tools, run_tool
from cheating_racer_detect.tracking import Box, Detection, Frame, MotionConfig, Policy, Tracker

POSITIONS = [24]*4+[30, 36, 42, 46, 50, 54, 58, 62, 66, 70]+[72]*10


def generate(tools, root):
    raw, source = root/'fixture.bgr', root/'synthetic.mp4'
    with raw.open('xb') as stream:
        for n, x in enumerate(POSITIONS):
            data = bytearray(bytes([40])*96*64*3)
            for x1, x2, y1, y2, level in ((x-12, x+12, 10, 30, 80),
                    (x-8, x-4, 20, 24, 24), (x+4, x+8, 20, 24, 230 if n%4 in (1, 2) else 24)):
                for y in range(y1, y2):
                    data[(y*96+x1)*3:(y*96+x2)*3] = bytes([level])*((x2-x1)*3)
            stream.write(data)
    run_tool([tools.ffmpeg, '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'bgr24',
              '-video_size', '96x64', '-framerate', '10', '-i', raw, '-c:v', 'libx264', '-crf', '12',
              '-pix_fmt', 'yuv420p', '-bf', '0', '-f', 'mp4', source])
    return source


def observation():
    motion = MotionConfig((4., 4., .02, .02), (.25, .25, .0025, .0025),
                          (1., 1., .1, .1, 100., 100., 1., 1.))
    tracker = Tracker(Policy(F(3, 5), F(3, 10), .2, .7, .2, .95, (2,), motion))
    def observe(frame, pixels):
        x = POSITIONS[frame.ordinal]
        return tracker.update(frame, [Detection(0, 2, Box(x-12, 10, x+12, 30), .9, 'scripted-event', frame)])
    return observe


def event_configs():
    return (LaneConfig('synthetic-event-v1', .15, F(1, 5), F(3, 10), 10.),
            LampConfig('synthetic-event-v1', 50., 180., 8, F(3, 5), F(3, 10), F(1, 10), F(1), 1))


def analyze(source, tools, root, *, capture=None):
    lane_config, lamp_config = event_configs()
    lane, lamp = LaneEngine(lane_config), LampEngine(lamp_config)
    video, timeline, frames, first, rotation = prepare(source, Decimal(0), Decimal('2.4'), tools)
    raw = root/'decoded.bgr'
    run_tool([tools.ffmpeg, '-nostdin', '-v', 'error', '-noautorotate', '-i', source,
              '-fps_mode', 'passthrough', '-an', '-pix_fmt', 'bgr24', '-f', 'rawvideo', raw])
    observe, events = observation(), []
    source_hash = _sha256(source)
    with raw.open('rb') as stream:
        for n, item in enumerate(frames):
            frame = Frame(source_hash, video['index'], first+n, item.pts, timeline.time_base,
                          timeline.frames[0].pts, video['width'], video['height'], rotation)
            pixels = stream.read(frame.width*frame.height*3)
            report = observe(frame, pixels)
            key = tuple(report['tracks'][0]['identity'])
            vehicle = Vehicle.from_tracking(frame, report, key)
            section = LaneSection(frame, 'synthetic-road', ('a', 'b', 'c'), (0., 48., 96.), 30., 'stable', 'ordinary', True)
            event = lane.update(frame, vehicle, section)['candidate']
            if event is not None:
                events.append(event)
            x = POSITIONS[n]
            left, right = (LampROI(Box(x-8, 20, x-4, 24), 'usable'),
                           LampROI(Box(x+4, 20, x+8, 24), 'usable'))
            lamp.update(frame, vehicle, pixels, left, right)
            if capture is not None:
                from cheating_racer_detect.events.inputs import InputFrame
                capture(InputFrame(frame, vehicle, section, left, right))
    if len(events) != 1:
        raise ValueError('Synthetic fixture expected exactly one adjacent change')
    return events[0], lamp


def capture_inputs(source, tools, root):
    """Explicit synthetic settings + selected actual native-Tracker observations."""
    from cheating_racer_detect.events.inputs import EventInputs
    rows = []
    candidate, _lamps = analyze(source, tools, root, capture=rows.append)
    lane, lamp = event_configs()
    return EventInputs(lane, lamp, F(1, 5), F(2, 5), candidate.identity, tuple(rows))


def demonstrate(output, tools=None):
    output = local_path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('Choose a new local folder in an existing parent')
    tools = tools or discover_tools()
    with tempfile.TemporaryDirectory(prefix='crd-event-demo-') as directory:
        root = Path(directory)
        source = generate(tools, root)
        candidate, lamps = analyze(source, tools, root)
        return export_candidate(source, candidate, output, observation(), pre=F(1, 5), post=F(2, 5), lamps=lamps, toolchain=tools)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = demonstrate(args.output)
    print(json.dumps({'status': result['status'], 'direction': result['candidate']['direction'],
                      'signal_state': result['signal_state'], 'scope': 'synthetic injected inputs, not accuracy'}))


if __name__ == '__main__':
    main()
