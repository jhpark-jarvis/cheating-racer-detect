"""Current detector observations -> source-bound replay inputs, not auto events."""

from ..analysis import DetectorTrackerBridge
from ..errors import AppError
from ..paths import destination_path, local_path, source_path
from ..pixels import pixel_window
from ..review.video import interval, prepare
from ..service import _identity, _sha256
from ..source import release_pixels, source_job
from ..tools import discover_tools
from ..tracking import Frame
from .contracts import Vehicle, identity, seconds
from .inputs import EventInputs, InputFrame, _json, owned_output, write_bytes
from .lamp import LampConfig, LampROI
from .lane import LaneConfig


@source_job
def capture_inputs(input_path, start, end, output_path, bridge, *, selected_identity,
                   lane_config, lamp_config, pre, post, gates=None, toolchain=None):
    """Fresh single-owner bridge required; explicit identity never auto-remapped.

    gates(Frame, Vehicle|None) -> (LaneSection|None, left LampROI, right LampROI).
    Without gates all lanes/ROIs stay unknown. Source <=1024 and <=120 selected
    frames. No loader, network, lane/pose/ROI model, predictions or GT assertion.
    Failed/cancelled jobs reset the supplied tracker before a fresh retry.
    """
    if not isinstance(bridge, DetectorTrackerBridge):
        raise AppError('INVALID_CAPTURE', '명시적인 탐지·추적 연결이 필요합니다.')
    try:
        selected_identity = identity(selected_identity)
        if gates is not None and not callable(gates):
            raise ValueError('Invalid gates')
        # Validate all configurations and scalar bounds before decoding.
        seconds(pre, True)
        seconds(post, True)
        if not isinstance(lane_config, LaneConfig) or not isinstance(lamp_config, LampConfig):
            raise ValueError('Explicit configurations required')
        source = source_path(input_path)
        destination_path(output_path, source)
        start, end = interval(start, end)
        tools = toolchain or discover_tools()
        before, source_hash = _identity(source), _sha256(source)
        video, timeline, frames, first, rotation = prepare(source, start, end, tools)
        rows = []
        with pixel_window(source, tools, video, timeline, first, len(frames), complete=True) as stream:
            for n, item in enumerate(frames):
                frame = Frame(source_hash, video['index'], first+n, item.pts, timeline.time_base,
                              timeline.frames[0].pts, video['width'], video['height'], rotation)
                pixels = stream.read(frame.width*frame.height*3)
                tracking = bridge.update(frame, pixels)['tracking']
                vehicle = Vehicle.from_tracking(frame, tracking, selected_identity)
                lane, left, right = (None, LampROI(None, 'model_uncertain'), LampROI(None, 'model_uncertain')) if gates is None else gates(frame, vehicle)
                rows.append(InputFrame(frame, vehicle, lane, left, right))
        inputs = EventInputs(lane_config, lamp_config, pre, post, selected_identity, tuple(rows))
        with owned_output(output_path) as stage:
            write_bytes(stage/'inputs.json', inputs.to_bytes())
            if bridge.detector.model_id != bridge.detector_id:
                raise ValueError('Detector model receipt changed')
            receipt = {'schema': 'experimental-detector-capture-v1', 'status': 'complete',
                       'inputs': 'inputs.json', 'inputs_sha256': inputs.sha256, 'source_sha256': source_hash,
                       'model': {'alias': bridge.model_alias, 'external_identifier': bridge.detector_id},
                       'selected_identity': list(selected_identity), 'frame_count': len(rows),
                       'actual_selected_observations': sum(r.vehicle is not None for r in rows),
                       'limits': ['caller model receipt; not authenticity', 'identity unverified; never remapped',
                                  'lane/ROI gates caller-owned', 'not automatic event or accuracy evidence']}
            write_bytes(stage/'capture.json', (_json(receipt)+'\n').encode('utf-8'))
            local_path(source)
            if before != _identity(source) or source_hash != _sha256(source):
                raise AppError('INPUT_CHANGED', '처리 중 원본이 변경되었습니다.')
            release_pixels()
        return inputs
    except BaseException:
        bridge.tracker.reset('failed_update')
        raise
