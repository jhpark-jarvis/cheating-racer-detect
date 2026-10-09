"""Self-generated source + saved/reloaded annotations -> verified replay bundle."""

import argparse
import json
from pathlib import Path
import tempfile

from cheating_racer_detect.events.inputs import load_inputs, save_inputs
from cheating_racer_detect.events.replay import export_replay
from cheating_racer_detect.paths import local_path
from cheating_racer_detect.tools import discover_tools
from scripts.demo_events import capture_inputs, generate


def demonstrate(output, tools=None):
    output = local_path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('Choose a new local folder in an existing parent')
    tools = tools or discover_tools()
    with tempfile.TemporaryDirectory(prefix='crd-replay-demo-') as directory:
        root = Path(directory)
        source = generate(tools, root)
        inputs = capture_inputs(source, tools, root)
        save_inputs(inputs, root/'saved')
        return export_replay(source, load_inputs(root/'saved'/'inputs.json'), output,
                             candidate_index=0, toolchain=tools)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = demonstrate(args.output)
    print(json.dumps({'status': result['status'], 'candidate_count': result['candidate_count'],
                      'signal_state': result['signal_state'], 'scope': 'synthetic saved inputs, not accuracy'}))


if __name__ == '__main__':
    main()
