"""Derived coded-view rendering; no source mutation, display rotation or file IO."""

import math

from .records import summarize


def pixel_edges(values, scale):
    """Half-open continuous box -> intersected pixels' enclosing raster edges."""
    x1, y1, x2, y2 = values
    return math.floor(x1*scale), math.floor(y1*scale), math.ceil(x2*scale)-1, math.ceil(y2*scale)-1


def draw_box(image, edges, color, style):
    import cv2

    x1, y1, x2, y2 = edges
    if style == "dashed":
        for x in range(x1, x2+1, 12):
            for y in (y1, y2):
                cv2.line(image, (x, y), (min(x+5, x2), y), color, 1)
        for y in range(y1, y2+1, 12):
            for x in (x1, x2):
                cv2.line(image, (x, y), (x, min(y+5, y2)), color, 1)
    else:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_8)
        if style == "double" and x2-x1 >= 4 and y2-y1 >= 4:
            cv2.rectangle(image, (x1+2, y1+2), (x2-2, y2-2), color, 1, cv2.LINE_8)


def render(frame, pixels, report):
    summary = summarize(frame, report)
    if type(pixels) is not bytes or len(pixels) != frame.width*frame.height*3:
        raise ValueError("Expected immutable current-frame BGR bytes")
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as error:
        if error.name not in ("numpy", "cv2"):
            raise
        raise RuntimeError("Optional review dependencies missing; install the tracking extra") from error
    scale = min(4, max(1, 640//frame.width))
    raster_width, raster_height = frame.width*scale, frame.height*scale
    width, header = max(880, raster_width+480), 138
    content_height = max(raster_height, 26+18*len(summary["annotations"]))
    image = np.full((header+content_height+48, width, 3), 22, dtype=np.uint8)
    source = np.frombuffer(pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    image[header:header+raster_height, :raster_width] = cv2.resize(
        source, (raster_width, raster_height), interpolation=cv2.INTER_NEAREST)
    lines = ["TRACK REVIEW / ID UNVERIFIED",
             f"source={frame.source_id} stream={frame.stream_index} frame={frame.ordinal}",
             f"source time={frame.time}s (exact) raw PTS={frame.pts}",
             f"time base={frame.time_base} first PTS={frame.first_pts} segment={summary['segment']}",
             f"CODED VIEW {frame.width}x{frame.height}; rotation metadata={frame.display_rotation} (NOT APPLIED)",
             f"reset={summary['reset_reason']} ambiguity groups={len(summary['ambiguity_groups'])}"]
    for i, line in enumerate(lines):
        # Fit long exact fractions/opaque IDs rather than silently truncating time.
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .43, 1)[0][0]
        font = .43 * min(1, (width-16)/max(1, size))
        cv2.putText(image, line, (8, 19+i*22), cv2.FONT_HERSHEY_SIMPLEX, font,
                    (235, 235, 235), 1, cv2.LINE_AA)
    codes = {"observed": "OBS", "prediction_only": "PRED", "ambiguous": "AMB", "unassociated": "RAW"}
    ledger_x = raster_width+16
    cv2.putText(image, "EVIDENCE LIST / IDs UNVERIFIED", (ledger_x, header+16),
                cv2.FONT_HERSHEY_SIMPLEX, .43, (235, 235, 235), 1, cv2.LINE_AA)
    for position, item in enumerate(summary["annotations"]):
        key = item["identity"]
        tag = f"S{key[0]} C{key[1]} T{key[2]}" if key else f"C{item['class_id']} D{item['detection_index']}"
        label = codes[item["state"]]+" "+tag
        details = (f" D{item['detection_index']}" if key and item["detection_index"] is not None else "")
        if item["state"] == "prediction_only":
            details += f" age={item['unobserved_seconds']}s"
            if item["bbox"] is None:
                details += " NO VISIBLE BOX"
        ledger = label+details
        length = cv2.getTextSize(ledger, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)[0][0]
        cv2.putText(image, ledger, (ledger_x, header+36+18*position), cv2.FONT_HERSHEY_SIMPLEX,
                    .4*min(1, (width-ledger_x-8)/max(1, length)), tuple(item["bgr"]), 1, cv2.LINE_AA)
        if item["bbox"] is None:
            continue
        x1, y1, x2, y2 = pixel_edges(item["bbox"], scale)
        edges = x1, y1+header, x2, y2+header
        draw_box(image, edges, tuple(item["bgr"]), item["style"])
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .36, 1)[0][0]
        font = .36*min(1, (raster_width-2)/max(1, size))
        fitted = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, 1)[0][0]
        cv2.putText(image, label, (max(0, min(x1, raster_width-fitted-1)), max(header+12, y1+header-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, tuple(item["bgr"]), 1, cv2.LINE_AA)
    footer = ["OBS solid | PRED dashed (not observed) | AMB double | RAW unassociated",
              "No identity/violation certainty. Rotation is metadata only."]
    for i, line in enumerate(footer):
        cv2.putText(image, line, (8, header+content_height+18+i*20), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (235, 235, 235), 1, cv2.LINE_AA)
    summary["layout"] = {"integer_scale": scale, "raster_origin": [0, header],
                         "raster_size": [raster_width, raster_height], "canvas_size": [width, image.shape[0]],
                         "ledger_origin": [ledger_x, header], "ledger_rows": len(summary["annotations"])}
    return image, summary
