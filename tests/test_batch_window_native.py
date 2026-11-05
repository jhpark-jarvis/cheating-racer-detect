"""Actual profile/PTS/raw pixel and cleanup equivalence for retained batch windows."""
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect import pixels
from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events.replay import export_replay_batch
from cheating_racer_detect.events.verify import verify_batch
from cheating_racer_detect.review import video
from cheating_racer_detect.source import inspect_source, source_session
from cheating_racer_detect.tools import run_tool
from cheating_racer_detect.tracking import Frame
from scripts.demo_batch_replay import generate_inputs
from tests.fixtures import run, sha256
from tests.test_capture_native import AVAILABLE, TOOLS


@unittest.skipUnless(AVAILABLE and TOOLS,'NOT_RUN: existing FFmpeg/NumPy/OpenCV required')
class BatchWindows(unittest.TestCase):
    def test_seven_profiles_raw_oracle_exact_pts_reversed_results_and_strict_verifier(self):
        with tempfile.TemporaryDirectory(prefix='crd-batch-windows-') as directory:
            root=Path(directory)
            original,inputs=generate_inputs(TOOLS,root)
            profiles=[('cfr',original)]
            for name,options in (
                ('offset',['-copyts','-itsoffset','5','-i',original,'-c','copy','-avoid_negative_ts','disabled']),
                *((f'rot{angle}',['-display_rotation:v:0',str(angle),'-i',original,'-c','copy']) for angle in (90,180,270)),
                ('vfr',['-i',original,'-vf',r'setpts=if(lt(N\,20)\,N/(10*TB)\,(2+(N-20)/8)/TB)',
                        '-fps_mode','passthrough','-c:v','libx264','-pix_fmt','yuv420p','-bf','0']),
                ('audio',['-i',original,'-f','lavfi','-i','sine=frequency=1000:sample_rate=48000:duration=4.4',
                          '-c:v','copy','-c:a','aac','-shortest'])):
                source=root/(name+'.mp4')
                run_tool([TOOLS.ffmpeg,'-nostdin','-v','error','-n',*options,source])
                profiles.append((name,source))
            for name,source in profiles:
                with self.subTest(profile=name):
                    before=sha256(source)
                    _profile,_audio,timeline=inspect_source(source,TOOLS)
                    profile,timeline,frames,first,rotation=video.prepare(source,Fraction(0),timeline.end,TOOLS)
                    rows=[]
                    for n,(old,item) in enumerate(zip(inputs.rows,frames)):
                        frame=Frame(before,profile['index'],first+n,item.pts,timeline.time_base,timeline.frames[0].pts,
                                    profile['width'],profile['height'],rotation)
                        vehicle=replace(old.vehicle,frame=frame,detection=replace(old.vehicle.detection,frame=frame))
                        rows.append(replace(old,frame=frame,vehicle=vehicle,lanes=replace(old.lanes,frame=frame)))
                    current=replace(inputs,rows=tuple(rows))
                    oracle=run([TOOLS.ffmpeg,'-v','error','-noautorotate','-i',source,'-map','0:v:0',
                                '-fps_mode','passthrough','-pix_fmt','bgr24','-f','rawvideo','pipe:1'])
                    raw_calls=[]
                    actual_render=video._render_window
                    def render(stream,profile,timeline,selected,first,*args):
                        offset=stream.tell();size=profile['width']*profile['height']*3
                        actual=stream.read(len(selected)*size)
                        self.assertEqual(actual,oracle[first*size:(first+len(selected))*size])
                        stream.seek(offset)
                        return actual_render(stream,profile,timeline,selected,first,*args)
                    actual_decode=pixels.run_tool
                    def decode(args,**kwargs):
                        raw_calls.append(True);return actual_decode(args,**kwargs)
                    out=root/(name+'-batch')
                    with source_session(pixel_parent=root),patch.object(video,'_render_window',side_effect=render), \
                            patch.object(pixels,'run_tool',side_effect=decode):
                        result=export_replay_batch(source,current,out,candidate_indices=(1,0),toolchain=TOOLS)
                    self.assertEqual(result['candidate_indices'],[1,0])
                    self.assertEqual([e['candidate_index'] for e in result['candidates']],[1,0])
                    self.assertEqual(len(raw_calls),1)
                    self.assertFalse(list(root.glob('crd-pixels-*')))
                    for entry in result['candidates']:
                        mapping=json.loads((out/entry['candidate']).parent.joinpath('media/review.json').read_text())
                        for row in mapping['frames']:
                            frame=row['source_frame'];self.assertEqual(frame['pts'],timeline.frames[frame['ordinal']].pts)
                    verified=verify_batch(out,source=source,toolchain=TOOLS)
                    self.assertTrue(verified['source_replayed'])
                    self.assertEqual(sha256(source),before)

    def test_retained_window_cleanup_failure_aborts_outer_publish_and_fresh_retry(self):
        with tempfile.TemporaryDirectory(prefix='crd-batch-dispose-') as directory:
            root=Path(directory);source,inputs=generate_inputs(TOOLS,root);out=root/'batch'
            with source_session(pixel_parent=root),patch.object(pixels._Snapshot,'close',side_effect=AppError('CLEANUP_FAILED','injected')), \
                    self.assertRaises(AppError) as error:
                export_replay_batch(source,inputs,out,candidate_indices=(1,0),toolchain=TOOLS)
            self.assertEqual(error.exception.code,'CLEANUP_FAILED')
            self.assertFalse(out.exists());self.assertFalse(list(root.glob('*.partial')))
            # The outer test owner removes the deliberately preserved raw directory.
            result=export_replay_batch(source,inputs,out,candidate_indices=(1,0),toolchain=TOOLS)
            self.assertEqual(result['status'],'complete')


if __name__=='__main__':unittest.main()
