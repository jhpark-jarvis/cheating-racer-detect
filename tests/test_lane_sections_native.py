"""Actual source pixels/clocks -> scripted UFLD logits -> section/live events."""

from dataclasses import replace
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.analysis import decode_ufld
from cheating_racer_detect.events import BoundaryBinding, LaneContext, LaneSectionBridge, Vehicle
from cheating_racer_detect.events.lane import LaneConfig
from cheating_racer_detect.pixels import pixel_window
from cheating_racer_detect.review.video import prepare
from cheating_racer_detect.tools import discover_tools
from cheating_racer_detect.tracking import Box, Detection, Frame
from tests import fixtures
from tests.test_lane_decoder import logits, output

try:
    TOOLS = discover_tools()
except Exception:
    TOOLS = None
AVAILABLE = importlib.util.find_spec('numpy') is not None


def synthetic_logits():
    values = logits()
    values['loc_row'].fill(-100.)
    values['loc_row'][0,30,:,1] = 100.
    values['loc_row'][0,100,:,2] = 100.
    values['loc_col'][:,:,:,0] = -100.
    for anchor in range(81):
        values['loc_col'][0,10+anchor,anchor,0] = 100.
    values['exist_col'][0,0,:,3] = 200.  # Unselected fourth slot is absent.
    return values


@unittest.skipUnless(TOOLS and AVAILABLE,'NOT_RUN: FFmpeg/NumPy required')
class NativeSections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='crd-section-native-')
        cls.root = Path(cls.temporary.name)
        try:
            cls.media = fixtures.generate(TOOLS,cls.root)
        except BaseException:
            cls.temporary.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        if cls.root.exists():
            raise AssertionError('Owned media directory remains')

    def run_source(self,name,alter=None):
        source = self.media[name]
        original = fixtures.sha256(source)
        video,timeline,selected,first,rotation = prepare(source,'0','1.2',TOOLS)
        bridge = LaneSectionBridge(LaneConfig('fixture-v1',.15,F(1,5),F(3,5),video['width']/10))
        children = set(self.root.iterdir())
        results = []
        path = (24,24,24,43,47,50,60,65,70)
        try:
            with patch('cheating_racer_detect.pixels.tempfile.gettempdir',return_value=str(self.root)):
                with pixel_window(source,TOOLS,video,timeline,first,len(selected),complete=True) as stream:
                    for n,item in enumerate(selected):
                        f = Frame(original,video['index'],first+n,item.pts,timeline.time_base,
                                  timeline.frames[0].pts,video['width'],video['height'],rotation)
                        self.assertEqual(len(stream.read(f.width*f.height*3)),f.width*f.height*3)
                        x = path[min(n,len(path)-1)]/96*f.width
                        box = Box(x-f.width/16,f.height/4,x+f.width/16,f.height*.75)
                        target = Vehicle(f,(1,2,1),Detection(0,2,box,.9,'scripted-vehicle',f))
                        batch = decode_ufld(f,output(f,synthetic_logits()))
                        bindings = tuple(BoundaryBinding(slot,name,(True,)*(71 if slot in (1,2) else 80))
                                         for slot,name in ((1,'left'),(2,'middle'),(0,'right')))
                        declared = LaneContext(f,target.identity,batch.model_id,'fixture-context','road',
                                               bindings,True,True,'stable','ordinary',True)
                        if alter is not None:
                            batch,declared = alter(n,batch,declared)
                        result = bridge.update(f,target,batch,declared)
                        self.assertIs(result['section'].frame,f)
                        results.append(result)
        finally:
            self.assertEqual(set(self.root.iterdir()),children)
            self.assertEqual(fixtures.sha256(source),original)
        return results

    def test_six_source_profiles_decoder_section_coded_contact_and_pts(self):
        for name in ('cfr','vfr','offset','rotation90','rotation180','rotation270'):
            with self.subTest(profile=name):
                results = self.run_source(name)
                self.assertTrue(all(r['section'].section is not None for r in results))
                for result in results:
                    section = result['section'].section
                    self.assertEqual(section.y,section.frame.height*.75)
                    self.assertEqual(section.boundary_ids,('left','middle','right'))
                    self.assertEqual(result['section'].report()['frame']['pts'],section.frame.pts)
                if name in ('cfr','offset'):
                    candidates = [r['lane']['candidate'] for r in results if r['lane']['candidate']]
                    self.assertEqual(len(candidates),1)
                    self.assertEqual(candidates[0].direction,'right')
                    self.assertEqual(candidates[0].crossing[1].time,F(1,2))
                if name == 'offset':
                    self.assertGreater(results[0]['section'].frame.first_pts,0)
                if name.startswith('rotation'):
                    self.assertEqual(results[0]['section'].frame.display_rotation,int(name[8:]))

    def test_real_source_occlusion_unknown_breaks_transition(self):
        def alter(n,batch,declared):
            if n == 5:
                bindings = tuple(replace(b,visible_segments=(False,)*len(b.visible_segments))
                                 for b in declared.bindings)
                declared = replace(declared,bindings=bindings)
            return batch,declared
        results = self.run_source('cfr',alter)
        self.assertIsNone(results[5]['section'].section)
        self.assertEqual(results[5]['section'].reason,'uncertified_intersection')
        self.assertFalse(any(r['lane']['candidate'] for r in results))

    def test_actual_window_context_change_failure_cancel_cleanup_and_retry(self):
        def changed(n,batch,declared):
            return batch,replace(declared,context_id='changed') if n >= 5 else declared
        results = self.run_source('cfr',changed)
        self.assertEqual(results[5]['reset_reason'],'lane_source_change')
        self.assertFalse(any(r['lane']['candidate'] for r in results))
        for failure in (RuntimeError('fixture failure'),KeyboardInterrupt()):
            def fail(n,batch,declared):
                if n == 5:
                    raise failure
                return batch,declared
            with self.assertRaises(type(failure)):
                self.run_source('cfr',fail)
            self.assertTrue(any(r['lane']['candidate'] for r in self.run_source('cfr')))


if __name__ == '__main__':
    unittest.main()
