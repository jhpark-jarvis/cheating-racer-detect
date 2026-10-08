"""Validate current-frame evidence before drawing a derived CODED-view preview.

Stateless, bounded, single-owner input. Predictions are never observations.
Display rotation is reported but deliberately not applied to coded coordinates.
"""

from fractions import Fraction

from cheating_racer_detect.tracking import Box, Detection, Frame

COLORS = {"observed": (60, 220, 80), "prediction_only": (0, 180, 255),
          "ambiguous": (230, 80, 230), "unassociated": (230, 210, 70)}
STYLES = {"observed": "solid", "prediction_only": "dashed",
          "ambiguous": "double", "unassociated": "solid"}


def count(value, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**31-1:
        raise ValueError("Invalid bounded review integer")
    return value


def rational(value):
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("Expected bounded exact source-time string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Invalid exact source time") from error
    if str(result) != value or result < 0:
        raise ValueError("Expected canonical nonnegative Fraction")
    return result


def sequence(value, maximum):
    if type(value) is not list or len(value) > maximum:
        raise ValueError("Review report resource bound exceeded")
    return value


def identity(value, segment):
    if (type(value) is not list or len(value) != 3
            or value[0] != segment):
        raise ValueError("Invalid current-segment identity")
    for number, minimum in zip(value, (1, 0, 1)):
        count(number, minimum)
    return tuple(value)


def box(value, frame, *, optional=False):
    if value is None and optional:
        return None
    if type(value) is not list or len(value) != 4:
        raise ValueError("Expected half-open original raster box")
    result = Box(*value)
    if result.clip(frame.width, frame.height) != result:
        raise ValueError("Box outside coded raster")
    return list(result.values())


def summarize(frame, report):
    """Owned annotation list, preserving raw detection provenance and exact time."""
    try:
        return _summarize(frame, report)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("Malformed current-frame tracking report") from error


def _summarize(frame, report):
    if (not isinstance(frame, Frame) or max(frame.width, frame.height) > 1024
            or type(report) is not dict or report.get("schema") != "experimental-tracks-v1"
            or report.get("frame") != frame.report()):
        raise ValueError("Expected bounded same-frame tracking report")
    # Dict equality alone would accept bool as an integer and 1.0 as integer 1.
    for key, expected in frame.report().items():
        if type(report["frame"][key]) is not type(expected):
            raise ValueError("Frame metadata type mismatch")
    if any(type(size) is not int for size in report["frame"]["coded_size"]):
        raise ValueError("Coded dimensions must be exact integers")
    count(frame.ordinal)
    count(frame.stream_index)
    now = rational(str(frame.time))
    rational(str(frame.time_base))
    segment = count(report["segment"], 1)
    delta = report["delta_seconds"]
    if delta is not None and not 0 < rational(delta) <= Fraction(3, 5):
        # Time-gap reports can exceed the motion bound: they reset the segment.
        if report["reset_reason"] != "time_gap" or rational(delta) <= 0:
            raise ValueError("Invalid exact frame delta")
    if report["reset_reason"] not in (None, "start", "manual_reset", "failed_update",
            "source_or_geometry_change", "missing_decoded_frame", "time_gap"):
        raise ValueError("Invalid reset marker")
    detections = {}
    for item in sequence(report["detections"], 64):
        index = count(item["index"])
        if index in detections:
            raise ValueError("Duplicate raw detection index")
        detection = Detection(index, count(item["class_id"]), Box(*box(item["bbox"], frame)),
                              item["model_score"], item["model_id"], frame)
        detections[index] = detection.report()

    annotations, active_ids, associated = [], set(), set()
    for track in sequence(report["tracks"], 128):
        key = identity(track["identity"], segment)
        if key[2] in active_ids or track["identity_status"] != "unverified":
            raise ValueError("Duplicate or falsely verified identity")
        active_ids.add(key[2])
        last, age = rational(track["last_observed_time"]), rational(track["unobserved_seconds"])
        if last > now or now-last != age:
            raise ValueError("Observation age does not match exact source time")
        estimated = box(track["estimated_bbox"], frame, optional=True)
        state = track["state"]
        if state == "observed":
            index = count(track["detection_index"])
            detected = detections.get(index)
            observed_box = box(track["detection_bbox"], frame)
            if (detected is None or index in associated or key[1] != detected["class_id"]
                    or age != 0 or observed_box != detected["bbox"]
                    or type(track["model_score"]) is bool
                    or track["model_score"] != detected["model_score"]
                    or track["model_id"] != detected["model_id"]):
                raise ValueError("Observation provenance mismatch")
            associated.add(index)
            shown = observed_box
        elif state == "prediction_only":
            if (age <= 0 or any(track[k] is not None for k in
                    ("detection_index", "detection_bbox", "model_score", "model_id"))):
                raise ValueError("Prediction carries fabricated observation")
            shown = estimated
        else:
            raise ValueError("Unknown observation state")
        annotations.append({"state": state, "identity": list(key), "identity_status": "unverified",
                            "class_id": key[1], "bbox": shown, "estimated_bbox": estimated,
                            "detection_index": track["detection_index"],
                            "model_score": track["model_score"], "model_id": track["model_id"],
                            "last_observed_time": str(last), "unobserved_seconds": str(age),
                            "style": STYLES[state], "bgr": list(COLORS[state])})
    ambiguous, ended = set(), set()
    ambiguities = sequence(report["ambiguities"], 192)
    for group in ambiguities:
        indices = sequence(group["detection_indices"], 64)
        keys = [identity(k, segment) for k in sequence(group["identities"], 128)]
        if not indices or group["reason"] != "non_unique_candidate_component":
            raise ValueError("Invalid ambiguity marker")
        classes = set()
        for index in indices:
            count(index)
            if index not in detections or index in ambiguous or index in associated:
                raise ValueError("Invalid ambiguous detection reference")
            ambiguous.add(index)
            classes.add(detections[index]["class_id"])
        for key in keys:
            if key[2] in ended or key[2] in active_ids:
                raise ValueError("Ambiguous identity is still active or duplicated")
            ended.add(key[2])
            classes.add(key[1])
        if len(classes) != 1:
            raise ValueError("Ambiguity crossed classes")
    for index, detected in sorted(detections.items()):
        if index not in associated:
            state = "ambiguous" if index in ambiguous else "unassociated"
            annotations.append({**detected, "state": state, "identity": None,
                                "identity_status": "unverified", "detection_index": index,
                                "style": STYLES[state], "bgr": list(COLORS[state])})
    # No reference to caller-owned mutable containers survives this function.
    return {"schema": "experimental-review-frame-v1", "frame": frame.report(), "segment": segment,
            "reset_reason": report["reset_reason"], "delta_seconds": delta,
            "view": "coded_raster_no_display_rotation", "annotations": annotations,
            "ambiguity_groups": [{"identities": [list(k) for k in g["identities"]],
                                  "detection_indices": list(g["detection_indices"]),
                                  "reason": g["reason"]} for g in ambiguities],
            "limits": ["identity unverified", "prediction is not observation", "not a violation decision"]}
