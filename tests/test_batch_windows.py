"""Bounded ordinal grouping and diagnostic cost/causal-scope contracts."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events.replay import _group_windows
from scripts import benchmark_batch as benchmark
from tests.test_event_inputs import fixture


class Planning(unittest.TestCase):
    def test_reverse_overlapping_contained_and_adjacent_windows_merge(self):
        groups=_group_windows([(10,4,2),(4,6,0),(6,2,1),(14,3,3)])
        self.assertEqual([(g.first,g.count,g.indices) for g in groups],[(4,13,(0,1,2,3))])

    def test_disjoint_and_union_over_limit_split_without_unbounded_spool(self):
        groups=_group_windows([(90,80,2),(200,10,3),(0,100,0),(50,20,1)])
        self.assertEqual([(g.first,g.count,g.indices) for g in groups],
                         [(0,100,(0,1)),(90,80,(2,)),(200,10,(3,))])
        self.assertTrue(all(g.count<=120 for g in groups))

    def test_exact_120_inclusive_limit_and_invalid_ranges(self):
        self.assertEqual(_group_windows([(0,60,0),(60,60,1)])[0].count,120)
        for rows in ([(-1,3,0)],[(0,121,0)],[(0,False,0)],[(True,1,0)]):
            with self.subTest(rows=rows),self.assertRaises(AppError):_group_windows(rows)

    def test_partially_retained_analysis_not_replaced_with_more_expensive_union(self):
        groups=_group_windows([(15,14,0),(25,35,1)],retained=(0,44))
        self.assertEqual((groups[0].first,groups[0].count,groups[0].prime),(15,45,False))
        self.assertTrue(_group_windows([(15,14,0),(25,35,1)],retained=(0,70))[0].prime)
        self.assertTrue(_group_windows([(15,14,0),(25,35,1)])[0].prime)


class Costs(unittest.TestCase):
    def test_invalid_generation_options_before_tools_or_fixture(self):
        for kwargs in ({'seconds':4},{'seconds':601},{'seconds':True},{'window':'middle'},{'order':'random'}):
            with (patch.object(benchmark.tools,'discover_tools',side_effect=AssertionError('TOOLS')),
                  self.subTest(kwargs=kwargs),self.assertRaises(ValueError)):
                benchmark.benchmark(**kwargs)

    def test_existing_trace_preserved_and_invalid_cli_does_not_run(self):
        with patch.object(benchmark.tracemalloc,'is_tracing',return_value=True),self.assertRaises(RuntimeError):
            benchmark.benchmark()
        with patch.object(benchmark,'benchmark',side_effect=AssertionError('RUN')),redirect_stdout(io.StringIO()), \
                patch('sys.stderr',new_callable=io.StringIO),self.assertRaises(SystemExit):
            benchmark.main(['--seconds','0'])

    def test_reason_families_preserve_codes_and_never_invent_causes(self):
        report=benchmark.reason_summary(fixture())
        self.assertIn('not inferred causes',report['scope'])
        for channels in report['reason_counts'].values():
            for row in channels.values():self.assertEqual(sum(row['codes'].values()),sum(row['families'].values()))
        with patch.object(benchmark,'lane_diagnostics',return_value={'reason_counts':{'section':{'future_code':2}}}):
            result=benchmark.reason_summary(fixture())['reason_counts']['lane']['section']
            self.assertEqual(result,{'codes':{'future_code':2},'families':{'unclassified':2}})


if __name__=='__main__':unittest.main()
