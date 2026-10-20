"""Explicit synthetic lane provenance -> saved v2 -> catalog/replay batch.

All geometry, physical IDs and visibility/semantic gates are SCRIPTED. This
demonstrates persistence, NOT a trained lane detector or actual visibility.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile

from cheating_racer_detect.analysis import LaneBatch, LaneSlot
from cheating_racer_detect.events import BoundaryBinding, LaneContext, LaneEvidence
from cheating_racer_detect.events.catalog import export_catalog
from cheating_racer_detect.events.inputs import load_inputs, save_inputs
from cheating_racer_detect.events.replay import export_replay_batch
from cheating_racer_detect.paths import local_path
from cheating_racer_detect.tools import discover_tools
from .demo_batch_replay import generate_inputs


def with_lane_evidence(inputs):
    """Convert this demo's scripted sections to explicit synthetic evidence.

    Not a way to infer visibility or IDs from arbitrary injected sections.
    Only the demo's 96x64 three-boundary scenario is supported.
    """
    rows = []
    for row in inputs.rows:
        f, lane, vehicle = row.frame, row.lanes, row.vehicle
        if ((f.width, f.height) != (96, 64) or lane is None or vehicle is None
                or len(lane.x) != 3):
            raise ValueError('Expected the explicit synthetic demo scenario')
        slots = tuple(LaneSlot(i, a, (True,)*n,
                      tuple((x, 64.*j/(n-1)) for j in range(n)))
                      for i, a, n, x in ((1,'row',72,lane.x[0]), (2,'row',72,lane.x[1]),
                                         (0,'column',81,lane.x[2]), (3,'column',81,96.)))
        batch = LaneBatch(f, 'scripted-lanes-v1', 0, slots)
        bindings = tuple(BoundaryBinding(i, name, (True,)*(71 if i in (1,2) else 80))
                         for i, name in zip((1,2,0), lane.boundary_ids))
        context = LaneContext(f, vehicle.identity, batch.model_id, 'scripted-context-v1',
                              lane.road_id, bindings, True, True, lane.ego, lane.scene, lane.rear_view)
        rows.append(replace(row, lane_evidence=LaneEvidence(batch, context)))
    return replace(inputs, rows=tuple(rows))


def demonstrate(output, tools=None):
    output = local_path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('Choose a new local folder in an existing parent')
    tools = tools or discover_tools()
    with tempfile.TemporaryDirectory(prefix='crd-lane-demo-') as directory:
        source, legacy = generate_inputs(tools, Path(directory))
        inputs = with_lane_evidence(legacy)
        save_inputs(inputs, Path(directory)/'saved')
        restored = load_inputs(Path(directory)/'saved/inputs.json')
        catalog = export_catalog(source, restored, Path(directory)/'listed', toolchain=tools)
        return export_replay_batch(source, restored, output, candidate_indices=(1, 0),
                                   catalog=catalog, toolchain=tools)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    result = demonstrate(parser.parse_args().output)
    print(json.dumps({'status': result['status'], 'selected_count': result['selected_count'],
                      'scope': 'synthetic v2 lane provenance, not accuracy'}))
