"""Fresh original replay vs saved listing, without rewriting review artifacts."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import LaneEvidence, listing, catalog
from cheating_racer_detect.events.inputs import save_inputs
from cheating_racer_detect.events.verify import verify_batch
from cheating_racer_detect.events.replay import export_replay_batch
from cheating_racer_detect.review.video import prepare
from cheating_racer_detect.service import _sha256
from cheating_racer_detect.tracking import Frame
from scripts.demo_batch_replay import generate_inputs
from scripts.demo_lane_replay import with_lane_evidence
from scripts.demo_lamp_replay import with_lamp_evidence
from tests.test_capture_native import AVAILABLE, TOOLS
from tests import fixtures


@unittest.skipUnless(AVAILABLE and TOOLS,'NOT_RUN: existing FFmpeg/NumPy/OpenCV required')
class NativeListing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='crd-listing-audit-')
        cls.root = Path(cls.temporary.name)
        try:
            cls.source,cls.legacy = generate_inputs(TOOLS,cls.root)
            cls.inputs = with_lamp_evidence(with_lane_evidence(cls.legacy))
            cls.digest = _sha256(cls.source)
            cls.catalog,_,_ = catalog.export_catalog_with_lamp_diagnostics(
                cls.source,cls.inputs,cls.root/'listing',toolchain=TOOLS,include_lane_diagnostics=True)
        except BaseException:
            cls.temporary.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        if cls.root.exists(): raise AssertionError('Owned fixtures remain')

    def tearDown(self):
        self.assertEqual(_sha256(self.source),self.digest)

    def test_offline_source_levels_single_analysis_readonly_and_selected_batch(self):
        root = self.root/'listing'
        before = {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.iterdir()}
        offline = listing.verify_listing(root,required_diagnostics=('lane','lamp'))
        with patch.object(listing,'analyze_inputs',wraps=listing.analyze_inputs) as analyze:
            result = listing.verify_listing(root,source=self.source,toolchain=TOOLS,
                required_diagnostics=('lane','lamp'))
            analyze.assert_called_once()
        self.assertEqual((offline['candidate_count'],result['candidate_count']),(2,2))
        self.assertFalse(offline['signal_context_verified'])
        self.assertTrue(result['signal_context_verified'])
        self.assertEqual(result['diagnostics_checked'],['lane','lamp'])
        self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.iterdir()})
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'batch'
            export_replay_batch(self.source,self.inputs,out,candidate_indices=(1,0),catalog=self.catalog,toolchain=TOOLS)
            self.assertEqual(verify_batch(out,source=self.source,toolchain=TOOLS)['status'],'verified')

    def test_legacy_v2_v3_zero_candidates_and_optional_sidecars(self):
        unknown = replace(self.inputs,rows=tuple(replace(r,lanes=None,
            lane_evidence=LaneEvidence(None,None)) for r in self.inputs.rows))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name,inputs in (('v1',self.legacy),('v2',with_lane_evidence(self.legacy)),('zero',unknown)):
                folder = root/name
                if name=='v1': catalog.export_catalog(self.source,inputs,folder,toolchain=TOOLS)
                elif name=='v2': catalog.export_catalog_with_diagnostics(self.source,inputs,folder,toolchain=TOOLS)
                else: catalog.export_catalog_with_lamp_diagnostics(self.source,inputs,folder,toolchain=TOOLS)
                offline = listing.verify_listing(folder)
                fresh = listing.verify_listing(folder,source=self.source,toolchain=TOOLS)
                self.assertEqual(offline['candidate_count'],fresh['candidate_count'])
                self.assertEqual(offline['candidate_count'],0 if name=='zero' else 2)
                self.assertEqual(offline['files_checked'],2 if name=='v1' else 3)

    def test_actual_installed_source_commands_redacted_and_equivalent(self):
        root = self.root/'listing'
        for source in (None,self.source):
            args = ['--listing',str(root),'--require-diagnostics','lane','lamp']
            if source is not None: args += ['--input',str(source)]
            installed_args = [sys.executable,'-B','-m','cheating_racer_detect','events','verify-list',*args]
            if source is not None: installed_args += ['--ffmpeg-dir',str(TOOLS.ffmpeg.parent)]
            first = subprocess.run(installed_args,capture_output=True,timeout=45)
            second = subprocess.run([sys.executable,'-B','-m','scripts.verify_listing',*args],
                capture_output=True,timeout=45)
            for result in (first,second):
                self.assertEqual(result.returncode,0,result.stderr.decode())
                self.assertEqual(json.loads(result.stdout)['source_replayed'],source is not None)
                for private in (str(root),self.digest,'scripted-lamp-model-v1','selected_identity'):
                    self.assertNotIn(private,result.stdout.decode())
            self.assertEqual(json.loads(first.stdout),json.loads(second.stdout))

    def test_plausible_saved_signal_edit_is_not_authenticated_offline_but_source_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(shutil.copytree(self.root/'listing',Path(directory)/'listing'))
            path = root/'catalog.json'
            value = json.loads(path.read_bytes())
            for side in ('left','right'):
                value['candidates'][0][side] = dict(state='unknown',blink_evidence=False,
                    reasons=['model_uncertain'],valid_samples=0)
            value['candidates'][0]['signal_state'] = 'unknown'
            path.write_bytes(catalog._bytes(value))
            self.assertFalse(listing.verify_listing(root)['signal_context_verified'])
            before = {p.name:p.read_bytes() for p in root.iterdir()}
            with self.assertRaises(AppError) as err:
                listing.verify_listing(root,source=self.source,toolchain=TOOLS)
            self.assertEqual(err.exception.code,'LISTING_INVALID')
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})

    def test_source_mismatch_failure_cancel_cleanup_no_write_and_fresh_retry(self):
        root = self.root/'listing'
        before = {p.name:p.read_bytes() for p in root.iterdir()}
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            wrong = temp/'wrong.mp4'
            wrong.write_bytes(b'owned mock wrong source')
            with patch.object(listing,'analyze_inputs',side_effect=AssertionError('Should reject before analysis')):
                with self.assertRaises(AppError): listing.verify_listing(root,source=wrong,toolchain=TOOLS)
            with patch('cheating_racer_detect.pixels.tempfile.gettempdir',return_value=str(temp)):
                for error in (RuntimeError('injected'),KeyboardInterrupt()):
                    with patch.object(listing,'make_catalog',side_effect=error),self.assertRaises(type(error)):
                        listing.verify_listing(root,source=self.source,toolchain=TOOLS)
                    self.assertEqual({p.name for p in temp.iterdir()},{'wrong.mp4'})
                with patch.object(listing,'release_pixels',side_effect=AppError('CLEANUP_FAILED','safe')),self.assertRaises(AppError):
                    listing.verify_listing(root,source=self.source,toolchain=TOOLS)
                self.assertEqual({p.name for p in temp.iterdir()},{'wrong.mp4'})
                self.assertEqual(listing.verify_listing(root,source=self.source,toolchain=TOOLS)['status'],'verified')
                self.assertEqual({p.name for p in temp.iterdir()},{'wrong.mp4'})
        self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})

    def test_six_actual_profiles_preserve_source_pts_rotation_and_sparse_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = fixtures.generate(TOOLS,root)
            for name in ('cfr','vfr','offset','rotation90','rotation180','rotation270'):
                with self.subTest(profile=name):
                    source = media[name]
                    digest = _sha256(source)
                    video,timeline,selected,first,rotation = prepare(source,'0','1.2',TOOLS)
                    rows = []
                    for n,item in enumerate(selected):
                        f = Frame(digest,video['index'],first+n,item.pts,timeline.time_base,
                            timeline.frames[0].pts,video['width'],video['height'],rotation)
                        r = self.legacy.rows[n]
                        v = replace(r.vehicle,frame=f,detection=replace(r.vehicle.detection,frame=f))
                        rows.append(replace(r,frame=f,vehicle=v,lanes=replace(r.lanes,frame=f)))
                    inputs = with_lamp_evidence(with_lane_evidence(replace(self.legacy,rows=tuple(rows))))
                    if len(inputs.rows)>6: inputs = replace(inputs,rows=(*inputs.rows[:5],*inputs.rows[6:]))
                    folder = root/(name+'-listing')
                    catalog.export_catalog_with_lamp_diagnostics(source,inputs,folder,toolchain=TOOLS,include_lane_diagnostics=True)
                    before = {p.name:p.read_bytes() for p in folder.iterdir()}
                    self.assertEqual(listing.verify_listing(folder,source=source,toolchain=TOOLS)['status'],'verified')
                    self.assertEqual(before,{p.name:p.read_bytes() for p in folder.iterdir()})
                    self.assertEqual(_sha256(source),digest)


if __name__=='__main__':
    unittest.main()
