"""Experimental event matching oracle; explicit identity correspondence only."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import islice

from ..tracking.contracts import identifier
from .contracts import seconds


def names(event):
    for value in (event.event_id, event.source_id, event.segment_id, event.vehicle_id):
        identifier(value)
    if event.direction not in ('left', 'right', 'unknown'):
        raise ValueError('Invalid direction')


@dataclass(frozen=True)
class Truth:
    """All supplied truths are eligible; exclusion decisions belong to labeling."""
    event_id: str
    source_id: str
    segment_id: str
    vehicle_id: str
    earliest: Fraction
    latest: Fraction
    direction: str

    def __post_init__(self):
        names(self)
        seconds(self.earliest)
        seconds(self.latest)
        if self.latest < self.earliest:
            raise ValueError('Backward GT uncertainty interval')


@dataclass(frozen=True)
class Prediction:
    event_id: str
    source_id: str
    segment_id: str
    vehicle_id: str  # model track alias; never assumed equal to GT vehicle ID
    time: Fraction
    direction: str
    abstained: bool = False

    def __post_init__(self):
        names(self)
        seconds(self.time)
        if type(self.abstained) is not bool:
            raise ValueError('Explicit abstention flag required')


def distance(truth, prediction):
    return max(truth.earliest-prediction.time, prediction.time-truth.latest, Fraction(0))


def matching(costs, n_truth, n_prediction):
    """Unit-capacity residual shortest augmentations: max count, then min cost.

    Fraction costs, reverse edges and deterministic node/edge order. No greedy
    nearest-first approximation. At most 64 truths and 64 predictions.
    """
    source, sink = n_truth+n_prediction, n_truth+n_prediction+1
    graph = [[] for _ in range(sink+1)]
    def edge(a, b, cost):
        forward, backward = [b, len(graph[b]), 1, cost], [a, len(graph[a]), 0, -cost]
        graph[a].append(forward)
        graph[b].append(backward)
        return forward
    for i in range(n_truth):
        edge(source, i, Fraction(0))
    refs = {}
    for (i, j), cost in sorted(costs.items()):
        refs[i, j] = edge(i, n_truth+j, cost)
    for j in range(n_prediction):
        edge(n_truth+j, sink, Fraction(0))
    while True:
        values, parents = [None]*len(graph), [None]*len(graph)
        values[source] = Fraction(0)
        # Reverse costs may be negative; Bellman-Ford, not Dijkstra.
        for _ in range(len(graph)-1):
            changed = False
            for a, edges in enumerate(graph):
                if values[a] is None:
                    continue
                for k, (b, _reverse, capacity, cost) in enumerate(edges):
                    if capacity and (values[b] is None or values[a]+cost < values[b]):
                        values[b], parents[b] = values[a]+cost, (a, k)
                        changed = True
            if not changed:
                break
        if parents[sink] is None:
            break
        node = sink
        while node != source:
            a, k = parents[node]
            item = graph[a][k]
            item[2] = 0
            graph[node][item[1]][2] = 1
            node = a
    return sorted(key for key, item in refs.items() if item[2] == 0)


def evaluate(truths, predictions, correspondence, *, tolerance, max_gt_width):
    """correspondence[(source, segment, model_alias)] = GT vehicle alias.

    Explicit experimental tolerance/GT width, no product accuracy thresholds.
    Unknown predictions do not remove eligible truths from recall's denominator.
    """
    seconds(tolerance)
    seconds(max_gt_width)
    truths, predictions = tuple(islice(iter(truths), 65)), tuple(islice(iter(predictions), 65))
    if len(truths) > 64 or len(predictions) > 64:
        raise ValueError('Evaluation bound exceeded')
    for values, cls in ((truths, Truth), (predictions, Prediction)):
        if any(not isinstance(v, cls) for v in values) or len({v.event_id for v in values}) != len(values):
            raise ValueError('Malformed/duplicate evaluation events')
    if not isinstance(correspondence, dict) or len(correspondence) > 64:
        raise ValueError('Explicit bounded vehicle correspondence required')
    for key, value in correspondence.items():
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError('Invalid correspondence key')
        for name in (*key, value):
            identifier(name)
    if any(t.latest-t.earliest > max_gt_width for t in truths):
        raise ValueError('GT uncertainty exceeds explicit policy')
    truths = tuple(sorted(truths, key=lambda t: t.event_id))
    active = tuple(sorted((p for p in predictions if not p.abstained), key=lambda p: p.event_id))
    costs = {(i, j): distance(t, p) for i, t in enumerate(truths) for j, p in enumerate(active)
             if (t.source_id, t.segment_id) == (p.source_id, p.segment_id)
             and correspondence.get((p.source_id, p.segment_id, p.vehicle_id)) == t.vehicle_id
             and distance(t, p) <= tolerance}
    pairs = matching(costs, len(truths), len(active))
    used_truth, used_prediction = {i for i, _ in pairs}, {j for _, j in pairs}
    tp, fp, fn = len(pairs), len(active)-len(pairs), len(truths)-len(pairs)
    direction_eligible = [(i, j) for i, j in pairs if truths[i].direction != 'unknown']
    direction_correct = sum(truths[i].direction == active[j].direction for i, j in direction_eligible)
    return {'schema': 'experimental-event-evaluation-v1', 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': None if not tp+fp else str(Fraction(tp, tp+fp)),
            'recall': None if not tp+fn else str(Fraction(tp, tp+fn)),
            'abstained_predictions': sum(p.abstained for p in predictions),
            'direction_correct': direction_correct, 'direction_eligible_matches': len(direction_eligible),
            'matches': [{'truth': truths[i].event_id, 'prediction': active[j].event_id,
                         'time_distance': str(costs[i, j]), 'gt_width': str(truths[i].latest-truths[i].earliest)} for i, j in pairs],
            'unmatched_truths': [t.event_id for i, t in enumerate(truths) if i not in used_truth],
            'unmatched_predictions': [p.event_id for j, p in enumerate(active) if j not in used_prediction],
            'policy': {'tolerance': str(tolerance), 'max_gt_width': str(max_gt_width),
                       'matching': 'maximum cardinality then minimum exact interval distance',
                       'ties': 'sorted event IDs and deterministic strict residual relaxation'},
            'limits': ['explicit eligible labels and vehicle correspondence', 'not real-world accuracy evidence',
                       'no lamp confusion matrix or end-to-end signal coverage score']}
