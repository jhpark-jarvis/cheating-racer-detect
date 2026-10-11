"""Two self-generated adjacent changes -> recorded inputs -> atomic batch.

Scripted boxes/lane/ROI gates, NOT automatic recognition or accuracy evidence.
"""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile

from cheating_racer_detect.events import LaneSection, Vehicle
from cheating_racer_detect.events.inputs import EventInputs, InputFrame, load_inputs, save_inputs
from cheating_racer_detect.events.lamp import LampROI
from cheating_racer_detect.events.replay import export_replay_batch
from cheating_racer_detect.paths import local_path
from cheating_racer_detect.review.video import prepare
from cheating_racer_detect.service import _sha256
from cheating_racer_detect.tools import discover_tools, run_tool
from cheating_racer_detect.tracking import Box, Detection, Frame
from .demo_events import event_configs

POSITIONS = ([24]*4+[30, 36, 42, 46, 50, 54, 58, 62, 66, 70]+[72]*10
             +[66, 60, 54, 50, 46, 42, 38, 34, 30, 26]+[24]*10)


def generate_inputs(tools, root):
    """Own 44-frame 96x64 CFR MP4 and actual-source bound scripted receipts."""
    raw, source = root/'fixture.bgr', root/'synthetic.mp4'
    with raw.open('xb') as stream:
        for n, x in enumerate(POSITIONS):
            data = bytearray(bytes([40])*96*64*3)
            level = 230 if n % 4 in (1, 2) else 24
            left, right = (24, level) if n < 24 else (level, 24)
            for x1, x2, y1, y2, brightness in ((x-12, x+12, 10, 30, 80),
                    (x-8, x-4, 20, 24, left), (x+4, x+8, 20, 24, right)):
                for y in range(y1, y2):
                    data[(y*96+x1)*3:(y*96+x2)*3] = bytes([brightness])*((x2-x1)*3)
            stream.write(data)
    run_tool([tools.ffmpeg, '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pixel_format', 'bgr24',
              '-video_size', '96x64', '-framerate', '10', '-i', raw, '-c:v', 'libx264', '-crf', '12',
              '-pix_fmt', 'yuv420p', '-bf', '0', '-f', 'mp4', source])
    video, timeline, frames, first, rotation = prepare(source, '0', '4.4', tools)
    if len(frames) != len(POSITIONS):
        raise ValueError('Synthetic scenario requires exactly 44 frames')
    source_hash, rows = _sha256(source), []
    for n, item in enumerate(frames):
        frame = Frame(source_hash, video['index'], first+n, item.pts, timeline.time_base,
                      timeline.frames[0].pts, video['width'], video['height'], rotation)
        x = POSITIONS[n]
        # Actual pixel location is scripted; this is not a model/Tracker claim.
        vehicle = Vehicle(frame, (1, 2, 1), Detection(0, 2, Box(x-12, 10, x+12, 30), .9, 'scripted-batch', frame))
        lanes = LaneSection(frame, 'synthetic-road', ('a', 'b', 'c'), (0., 48., 96.), 30., 'stable', 'ordinary', True)
        rows.append(InputFrame(frame, vehicle, lanes, LampROI(Box(x-8, 20, x-4, 24), 'usable'),
                               LampROI(Box(x+4, 20, x+8, 24), 'usable')))
    lane, lamp = event_configs()
    return source, EventInputs(lane, lamp, F(1, 5), F(2, 5), (1, 2, 1), tuple(rows))


def demonstrate(output, tools=None):
    output = local_path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('Choose a new local folder in an existing parent')
    tools = tools or discover_tools()
    with tempfile.TemporaryDirectory(prefix='crd-batch-demo-') as directory:
        root = Path(directory)
        source, inputs = generate_inputs(tools, root)
        save_inputs(inputs, root/'saved')
        return export_replay_batch(source, load_inputs(root/'saved'/'inputs.json'), output,
                                   candidate_indices=(0, 1), toolchain=tools)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    result = demonstrate(parser.parse_args().output)
    print(json.dumps({'status': result['status'], 'candidate_indices': result['candidate_indices'],
                      'selected_count': result['selected_count'], 'scope': 'synthetic injected inputs, not accuracy'}))


if __name__ == '__main__':
    main()
