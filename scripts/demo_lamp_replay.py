"""Own synthetic lamp parts -> v3 receipts -> selected review, NOT a model.

All side/turn/visibility/crop declarations below are scripted fixture facts.
Do not use this helper to certify semantics of arbitrary supplied lamp boxes.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile

from cheating_racer_detect.events import LampEvidence,LampObservation,LampPart
from cheating_racer_detect.events.catalog import export_catalog_with_diagnostics
from cheating_racer_detect.events.inputs import save_inputs,load_inputs
from cheating_racer_detect.events.replay import export_replay_batch
from cheating_racer_detect.paths import local_path
from cheating_racer_detect.tools import discover_tools
from cheating_racer_detect.tracking import Box
from .demo_batch_replay import generate_inputs
from .demo_lane_replay import with_lane_evidence


def with_lamp_evidence(inputs):
    """Only this project's own scripted 96x64 demo, not automatic recognition."""
    rows=[]
    for row in inputs.rows:
        if (row.frame.width,row.frame.height)!=(96,64):
            raise ValueError('Expected own 96x64 synthetic scenario')
        if row.vehicle is None:
            evidence=LampEvidence(None)
        else:
            crop=row.vehicle.box
            def part(side,roi):
                b=roi.box
                box=None if b is None else Box((b.x1-crop.x1)/(crop.x2-crop.x1)*240,
                    (b.y1-crop.y1)/(crop.y2-crop.y1)*200,
                    (b.x2-crop.x1)/(crop.x2-crop.x1)*240,
                    (b.y2-crop.y1)/(crop.y2-crop.y1)*200)
                visibility={'usable':'visible','model_uncertain':'unknown',
                            'identity_uncertain':'unknown'}.get(roi.status,roi.status)
                return LampPart(side,side+'-turn',box,visibility,True)
            o=LampObservation(row.frame,row.vehicle.identity,row.vehicle.box,'scripted-lamp-model-v1',
                'scripted-lamp-context-v1',crop,240,200,0,part('left',row.left),part('right',row.right),
                True,True,all(roi.status!='identity_uncertain' for roi in (row.left,row.right)))
            evidence=LampEvidence(o)
        left,right=evidence.rois(row.frame,row.vehicle)
        rows.append(replace(row,left=left,right=right,lamp_evidence=evidence))
    return replace(inputs,rows=tuple(rows))


def demonstrate(output,tools=None):
    output=local_path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('Choose a new local output folder')
    tools=tools or discover_tools()
    with tempfile.TemporaryDirectory(prefix='crd-lamp-demo-') as directory:
        root=Path(directory)
        source,legacy=generate_inputs(tools,root)
        inputs=with_lamp_evidence(with_lane_evidence(legacy))
        save_inputs(inputs,root/'saved')
        restored=load_inputs(root/'saved/inputs.json')
        catalog,_diagnostics=export_catalog_with_diagnostics(source,restored,root/'list',toolchain=tools)
        return export_replay_batch(source,restored,output,candidate_indices=(1,0),catalog=catalog,toolchain=tools)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    result=demonstrate(parser.parse_args().output)
    print(json.dumps({'status':result['status'],'selected_count':result['selected_count'],
                      'scope':'synthetic v3 lamp/lane declarations, not recognition'}))
