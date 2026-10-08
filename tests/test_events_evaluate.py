from dataclasses import replace
from fractions import Fraction as F
import itertools
import random
import unittest

from cheating_racer_detect.events.evaluate import Prediction, Truth, evaluate, matching


def truth(name='gt', time=F(1), **changes):
    return replace(Truth(name, 'source', 'segment', 'gt-car', time, time, 'right'), **changes)


def prediction(name='pred', time=F(1), **changes):
    return replace(Prediction(name, 'source', 'segment', 'track', time, 'right'), **changes)


def result(gt, pred, mapping=None, **changes):
    return evaluate(gt, pred, {('source', 'segment', 'track'): 'gt-car'} if mapping is None else mapping,
                    tolerance=changes.get('tolerance', F(1, 2)), max_gt_width=changes.get('width', F(1)))


class Evaluation(unittest.TestCase):
    def test_duplicate_other_vehicle_and_zero_denominators(self):
        report = result([truth()], [prediction('a'), prediction('b')])
        self.assertEqual((report['tp'], report['fp'], report['fn']), (1, 1, 0))
        self.assertEqual(report['precision'], '1/2')
        for mapping in ({}, {('source', 'segment', 'track'): 'different'}):
            report = result([truth()], [prediction()], mapping)
            self.assertEqual((report['tp'], report['fp'], report['fn']), (0, 1, 1))
        report = result([truth()], [])
        self.assertIsNone(report['precision'])
        self.assertEqual(report['recall'], '0')
        self.assertIsNone(result([], [])['recall'])

    def test_abstention_is_miss_not_denominator_removal(self):
        report = result([truth()], [prediction(abstained=True)])
        self.assertEqual((report['tp'], report['fp'], report['fn']), (0, 0, 1))
        self.assertEqual(report['abstained_predictions'], 1)

    def test_interval_distance_width_tolerance_source_segment_direction(self):
        report = result([truth(earliest=F(9, 10), latest=F(11, 10))], [prediction(direction='left')])
        self.assertEqual(report['matches'][0]['time_distance'], '0')
        self.assertEqual(report['direction_correct'], 0)
        self.assertEqual(report['direction_eligible_matches'], 1)
        for change in ({'source_id': 'other'}, {'segment_id': 'other'}, {'time': F(2)}):
            self.assertEqual(result([truth()], [prediction(**change)])['tp'], 0)
        with self.assertRaises(ValueError):
            result([truth(latest=F(3))], [])

    def test_maximum_cardinality_before_nearest_cost_and_minimum_cost(self):
        # Greedy g0-p0 prevents g1 from matching; residual reassignment repairs it.
        costs = {(0, 0): F(0), (0, 1): F(1), (1, 0): F(1)}
        self.assertEqual(matching(costs, 2, 2), [(0, 1), (1, 0)])
        costs[1, 1] = F(3)
        self.assertEqual(matching(costs, 2, 2), [(0, 1), (1, 0)])

    def test_small_exhaustive_oracle_random_cost_graphs(self):
        rng = random.Random(314159)
        for n, m in itertools.product(range(1, 5), repeat=2):
            for _ in range(8):
                costs = {(i, j): F(rng.randrange(8), 3) for i in range(n) for j in range(m) if rng.random() < .7}
                best = (0, F(0))
                for assignment in itertools.product(range(-1, m), repeat=n):
                    selected = [(i, j) for i, j in enumerate(assignment) if j != -1]
                    if len({j for _, j in selected}) != len(selected) or any(pair not in costs for pair in selected):
                        continue
                    score = (-len(selected), sum((costs[p] for p in selected), F(0)))
                    best = min(best, score)
                selected = matching(costs, n, m)
                self.assertEqual((-len(selected), sum((costs[p] for p in selected), F(0))), best)

    def test_order_invariant_ties_and_invalid_bounds(self):
        gt = [truth('a'), truth('b')]
        pred = [prediction('x'), prediction('y')]
        self.assertEqual(result(gt, pred), result(list(reversed(gt)), list(reversed(pred))))
        for a, b in (([truth()]*2, []), ([], [prediction()]*2), ([truth(str(i)) for i in range(65)], [])):
            with self.assertRaises(ValueError):
                result(a, b)
        with self.assertRaises(ValueError):
            result([], [], {'bad': 'gt-car'})
        with self.assertRaises(ValueError):
            result([], [], tolerance=.5)


if __name__ == '__main__':
    unittest.main()
