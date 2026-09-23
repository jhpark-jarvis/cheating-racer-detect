"""Create non-sensitive, local-only media for human playback and Ctrl+C checks.

Run from the repository root: python -m scripts.prepare_manual_qa --output PATH
The new output directory is retained for review; existing paths are refused.
"""

import argparse
from decimal import Decimal
import json
from pathlib import Path

from cheating_racer_detect.paths import local_path
from cheating_racer_detect.service import create_clip
from cheating_racer_detect.tools import discover_tools, doctor
from tests.fixtures import generate, run, sha256


def prepare(output: Path, *, with_cancel: bool = False):
    output = local_path(output)
    if not output.parent.is_dir():
        raise ValueError('The output parent must already exist')
    if output.exists():
        raise FileExistsError('Refusing an existing QA directory')
    tools = discover_tools()
    doctor(tools)
    output.mkdir()  # exclusive; never clear an old review directory
    media = generate(tools, output / 'fixtures')
    records = []
    for name in ('audio', 'cfr', 'rotation90', 'rotation180', 'rotation270'):
        destination = output / name
        create_clip(media[name], Decimal('0.35'), Decimal('2.35'), destination, tools)
        records.append({'case': name, 'source': str(media[name].relative_to(output)),
                        'clip': str((destination / 'clip.mp4').relative_to(output)),
                        'source_sha256': sha256(media[name]),
                        'clip_sha256': sha256(destination / 'clip.mp4')})
    if with_cancel:
        # 120 seconds gives a person time to press Ctrl+C during full-file decode.
        # Only synthetic input is repeated; no download or user media is involved.
        run([str(tools.ffmpeg), '-v', 'error', '-nostdin', '-n', '-stream_loop', '39',
             '-i', str(media['audio']), '-t', '120', '-vf', 'scale=1280:720',
             '-r', '30', '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
             '-c:a', 'aac', str(output / 'cancel-source.mp4')])
    manifest = {'status': 'fixtures_ready_not_human_verified', 'cases': records,
                'cancel_source': 'cancel-source.mp4' if with_cancel else None}
    with (output / 'qa-fixtures.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': manifest['status'], 'cases': len(records),
                      'cancel_source_ready': with_cancel}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--with-cancel', action='store_true')
    arguments = parser.parse_args()
    prepare(arguments.output, with_cancel=arguments.with_cancel)
