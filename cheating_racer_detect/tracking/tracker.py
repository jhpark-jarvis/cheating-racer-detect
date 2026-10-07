"""Bounded conservative candidate graph, not an identity/vehicle accuracy claim.

Only unique one-track/one-detection connected components are associated. Other
components terminate affected histories rather than resolving ties by guessing.
"""

from dataclasses import dataclass
from itertools import islice

from .contracts import Detection, Frame, Policy


def overlap(first, second):
    width = max(0., min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0., min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = width * height
    total = ((first.x2-first.x1)*(first.y2-first.y1)
             + (second.x2-second.x1)*(second.y2-second.y1) - intersection)
    return intersection / total


def native_motion(box, config):
    try:
        from .motion import OpenCVMotion
    except ModuleNotFoundError as error:
        if error.name not in ("numpy", "cv2"):
            raise
        raise RuntimeError("Optional tracking dependencies missing; install the tracking extra") from error
    return OpenCVMotion(box, config)


@dataclass
class _Track:
    class_id: int
    motion: object
    last_observed: object


class Tracker:
    """update(frame, detections) derives exact dt; no nominal-fps argument.

    Policy is mandatory. Stateful instances are single-owner, not thread-safe.
    Reports own their containers; only observed detections provide evidence.
    """

    supports_variable_dt = True

    def __init__(self, policy, *, motion_factory=native_motion):
        if not isinstance(policy, Policy) or not callable(motion_factory):
            raise ValueError("Expected validated policy and motion factory")
        self.policy, self._factory = policy, motion_factory
        self._segment, self._next_id = 0, 1
        self.reset("start")

    def reset(self, reason="manual_reset"):
        if reason not in ("start", "manual_reset", "failed_update"):
            raise ValueError("Invalid reset reason")
        self._previous, self._tracks, self._reason = None, {}, reason

    def _boundary(self, frame):
        previous = self._previous
        if previous is None:
            return self._reason, None
        if previous.key != frame.key:
            return "source_or_geometry_change", None
        if previous.first_pts != frame.first_pts or previous.time_base != frame.time_base:
            raise ValueError("Source clock changed")
        dt = frame.time - previous.time
        if dt <= 0 or frame.ordinal <= previous.ordinal:
            raise ValueError("Source time/order is not strictly increasing")
        if frame.ordinal != previous.ordinal + 1:
            return "missing_decoded_frame", dt
        if dt > self.policy.max_gap:
            return "time_gap", dt
        return None, dt

    def _components(self, edges):
        graph = {}
        for track_id, detection_id in edges:
            track, detection = ("t", track_id), ("d", detection_id)
            graph.setdefault(track, set()).add(detection)
            graph.setdefault(detection, set()).add(track)
        while graph:
            pending, visited = [min(graph)], set()
            while pending:
                node = pending.pop()
                if node not in visited:
                    visited.add(node)
                    pending.extend(graph[node] - visited)
            for node in visited:
                del graph[node]
            yield sorted(value for kind, value in visited if kind == "t"), sorted(
                value for kind, value in visited if kind == "d")

    def update(self, frame, detections):
        try:
            return self._update(frame, detections)
        except BaseException:
            # No partial native/matching update survives failure or cancellation.
            self.reset("failed_update")
            raise

    def _update(self, frame, detections):
        if not isinstance(frame, Frame):
            raise ValueError("Expected Frame")
        values = tuple(islice(iter(detections), self.policy.max_detections + 1))
        if len(values) > self.policy.max_detections:
            raise ValueError("Detection bound exceeded")
        by_index = {}
        for detection in values:
            if (not isinstance(detection, Detection) or detection.frame != frame
                    or detection.class_id not in self.policy.classes or detection.index in by_index):
                raise ValueError("Stale/duplicate/unsupported detection")
            by_index[detection.index] = detection
        reason, dt = self._boundary(frame)
        if reason:
            self._segment += 1
            self._tracks = {}
        expired = [key for key, track in self._tracks.items()
                   if frame.time - track.last_observed > self.policy.max_unobserved]
        for key in expired:
            del self._tracks[key]
        if not reason:
            for track in self._tracks.values():
                track.motion.predict(dt)

        eligible = {index: d for index, d in by_index.items() if d.score >= self.policy.min_score}
        edges = [(key, index) for key, track in sorted(self._tracks.items())
                 for index, d in sorted(eligible.items())
                 if track.class_id == d.class_id and overlap(track.motion.box, d.box) >= self.policy.min_iou]
        # Detection-detection conflicts are represented through temporary negative
        # graph nodes; they cannot be confused with positive persistent track IDs.
        conflicts = set()
        indices = sorted(eligible)
        for position, index in enumerate(indices):
            for other in indices[position+1:]:
                first, second = eligible[index], eligible[other]
                if first.class_id == second.class_id and overlap(first.box, second.box) >= self.policy.duplicate_iou:
                    conflicts.update((index, other))
                    phantom = -1 - len(edges)
                    edges.extend(((phantom, index), (phantom, other)))
        associations, used, ambiguities = {}, set(), []
        for keys, indices in self._components(edges):
            actual_keys = [key for key in keys if key > 0]
            if len(keys) == len(indices) == 1 and not conflicts.intersection(indices):
                associations[keys[0]] = eligible[indices[0]]
                used.add(indices[0])
            else:
                used.update(indices)
                ambiguities.append({"identities": [[self._segment, self._tracks[k].class_id, k] for k in actual_keys],
                                    "detection_indices": indices, "reason": "non_unique_candidate_component"})
                for key in actual_keys:
                    del self._tracks[key]
        for key, detection in associations.items():
            track = self._tracks[key]
            track.motion.correct(detection.box)
            track.last_observed = frame.time
        for index, detection in sorted(eligible.items()):
            if index in used or detection.score < self.policy.new_score:
                continue
            if len(self._tracks) >= self.policy.max_tracks:
                raise ValueError("Track bound exceeded")
            key = self._next_id
            self._next_id += 1
            self._tracks[key] = _Track(detection.class_id, self._factory(detection.box, self.policy.motion), frame.time)
            associations[key] = detection
        records = []
        for key, track in sorted(self._tracks.items()):
            detection = associations.get(key)
            clipped = track.motion.box.clip(frame.width, frame.height)
            records.append({"identity": [self._segment, track.class_id, key], "identity_status": "unverified",
                            "state": "observed" if detection is not None else "prediction_only",
                            "detection_index": detection.index if detection is not None else None,
                            "detection_bbox": list(detection.box.values()) if detection is not None else None,
                            "estimated_bbox": list(clipped.values()) if clipped is not None else None,
                            "model_score": detection.score if detection is not None else None,
                            "model_id": detection.model_id if detection is not None else None,
                            "last_observed_time": str(track.last_observed),
                            "unobserved_seconds": str(frame.time-track.last_observed)})
        self._previous, self._reason = frame, None
        return {"schema": "experimental-tracks-v1", "frame": frame.report(), "segment": self._segment,
                "delta_seconds": str(dt) if dt is not None else None, "reset_reason": reason,
                "detections": [by_index[index].report() for index in sorted(by_index)],
                "tracks": records, "ambiguities": ambiguities}
