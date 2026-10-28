"""Standalone listing graph/receipt audit without media or ML dependencies."""

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.events import listing
from cheating_racer_detect.events.catalog import CandidateCatalog, _bytes
from cheating_racer_detect.events.diagnostics import _replay_diagnostics, diagnostics_bytes, lane_diagnostics
from cheating_racer_detect.events.inputs import MAX_BYTES
from cheating_racer_detect.events.lamp_diagnostics import lamp_diagnostics
from cheating_racer_detect.events.replay import ReplayAnalysis
from scripts.demo_lamp_replay import with_lamp_evidence
from scripts.verify_listing import main as source_main
from tests.test_candidate_catalog import document
from tests.test_event_inputs import fixture
from tests.test_installed_cli import invoke
from tests.test_lane_evidence import scenario


def stored(root, inputs=None, diagnostics=()):
    """Own metadata fixture, NOT a source-verified exporter output."""
    inputs = inputs or scenario()
    root.mkdir()
    _, events = _replay_diagnostics(inputs)
    value = document(inputs)
    value.update(candidate_count=len(events), candidates=[])
    for index, event in enumerate(events):
        begin = max(0,event.start[0].time-inputs.pre)
        end = event.completion[1].time+inputs.post
        side = dict(state='unknown',blink_evidence=False,reasons=['insufficient_history'],valid_samples=0)
        value['candidates'].append({'candidate_index':index,
            'candidate_sha256':hashlib.sha256(_bytes(event.report())).hexdigest(),
            'direction':event.direction,
            **{name:[str(f.time) for f in getattr(event,name)] for name in ('start','crossing','completion')},
            'requested_context':[str(begin),str(end)],'context':[str(begin),str(end)],
            'signal_state':'unknown','left':dict(side),'right':dict(side)})
    catalog = CandidateCatalog(_bytes(value))
    (root/'inputs.json').write_bytes(inputs.to_bytes())
    (root/'catalog.json').write_bytes(catalog.to_bytes())
    for name in diagnostics:
        report = lane_diagnostics(inputs) if name=='lane' else lamp_diagnostics(inputs)
        (root/listing.SIDECARS[name]).write_bytes(diagnostics_bytes(report))
    return inputs, catalog


class Listing(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_versions_optional_combinations_readonly_and_no_tools_needed(self):
        for version,inputs in (('v1',fixture()),('v2',scenario()),('v3',with_lamp_evidence(scenario()))):
            for names in ((),('lane',),('lamp',),('lane','lamp')):
                root = self.root/(version+''.join(names))
                stored(root,inputs,names)
                before = {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.iterdir()}
                with patch.object(listing,'discover_tools',side_effect=AssertionError('No tools offline')):
                    result = listing.verify_listing(root,required_diagnostics=names)
                self.assertEqual(result['files_checked'],2+len(names))
                self.assertEqual(result['diagnostics_checked'],list(names))
                self.assertEqual(result['verification_level'],'recorded_inputs')
                self.assertFalse(result['source_replayed'])
                self.assertFalse(result['signal_context_verified'])
                self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.iterdir()})

    def test_zero_candidates_are_valid_not_no_blink_or_normal_driving_proof(self):
        stored(self.root/'list',fixture(),('lamp',))
        result = listing.verify_listing(self.root/'list')
        self.assertEqual(result['candidate_count'],0)
        self.assertFalse(result['signal_context_verified'])
        self.assertNotIn('signal_state',result)

    def test_deleted_optional_requires_independent_requirement_to_detect(self):
        stored(self.root/'list',diagnostics=('lane','lamp'))
        (self.root/'list/lamp-diagnostics.json').unlink()
        self.assertEqual(listing.verify_listing(self.root/'list')['diagnostics_checked'],['lane'])
        with self.assertRaises(AppError) as err:
            listing.verify_listing(self.root/'list',required_diagnostics=('lamp',))
        self.assertEqual(err.exception.code,'LISTING_INVALID')

    def test_invalid_requirements_and_offline_tool_setting_rejected(self):
        for value in ('lamp',('lamp','lamp'),('other',),(True,),iter(('lamp',)),None):
            with self.subTest(value=type(value)),self.assertRaises(AppError) as err:
                listing.verify_listing(self.root,required_diagnostics=value)
            self.assertEqual(err.exception.code,'INVALID_ARGUMENT')
        with self.assertRaises(AppError): listing.verify_listing(self.root,toolchain=object())

    def test_unknown_entry_directory_missing_core_and_indirect_path_rejected(self):
        for extra in ('other.json','.hidden','Catalog.json','nested'):
            root = self.root/extra.replace('.','_')
            stored(root)
            if extra=='nested': (root/extra).mkdir()
            else: (root/extra).write_bytes(b'not inspected')
            with self.assertRaises(AppError): listing.verify_listing(root)
        for value in ('https://example.test/list',str(self.root/'../no')):
            with self.assertRaises(AppError): listing.verify_listing(value)
        stored(self.root/'missing')
        (self.root/'missing/inputs.json').unlink()
        with self.assertRaises(AppError): listing.verify_listing(self.root/'missing')

    def test_core_receipts_source_hash_selected_identity_or_brackets_tampered(self):
        root = self.root/'list'
        _,catalog = stored(root)
        value = catalog.report()
        for key,replacement in (('inputs_sha256','b'*64),('selected_identity',[1,2,2])):
            changed = {**value,key:replacement}
            (root/'catalog.json').write_bytes(_bytes(changed))
            with self.assertRaises(AppError): listing.verify_listing(root)
        for changes in ({'direction':'left'},{'candidate_sha256':'b'*64},
                        {'completion':['3/5','7/10']}):
            changed = catalog.report()
            changed['candidates'][0].update(changes)
            (root/'catalog.json').write_bytes(_bytes(changed))
            with self.assertRaises(AppError): listing.verify_listing(root)

    def test_valid_structural_signal_edit_is_explicitly_not_pixel_verified_offline(self):
        root = self.root/'list'
        _,catalog = stored(root)
        value = catalog.report()
        value['candidates'][0]['left'] = dict(state='no_blink_observed',blink_evidence=False,reasons=[],valid_samples=4)
        (root/'catalog.json').write_bytes(_bytes(value))
        result = listing.verify_listing(root)
        self.assertFalse(result['signal_context_verified'])
        self.assertFalse(result['source_replayed'])

    def test_nested_diagnostic_unknown_type_count_reset_and_hash_drift_rejected(self):
        root = self.root/'list'
        inputs,_ = stored(root,with_lamp_evidence(scenario()),('lane','lamp'))
        for name,creator in (('lane',lane_diagnostics),('lamp',lamp_diagnostics)):
            original = creator(inputs)
            for keys,value in ((('extra',),True),(('counts','input_rows'),True),
                               (('rows',0,'source_reset_reason'),'hidden'),(('inputs_sha256',),'b'*64)):
                report = json.loads(json.dumps(original))
                target = report
                for key in keys[:-1]: target = target[key]
                target[keys[-1]] = value
                (root/listing.SIDECARS[name]).write_bytes(_bytes(report))
                with self.subTest(name=name,keys=keys),self.assertRaises(AppError): listing.verify_listing(root)
            (root/listing.SIDECARS[name]).write_bytes(_bytes(original))

    def test_strict_parse_size_depth_unicode_duplicates_nonfinite_all_documents(self):
        root = self.root/'list'
        stored(root,diagnostics=('lamp',))
        for name in ('inputs.json','catalog.json','lamp-diagnostics.json'):
            path = root/name
            before = path.read_bytes()
            for raw in (b'',b' '* (MAX_BYTES+1),b'\xff',b'{"x":NaN}',b'{"x":1e999}',
                        b'{"a":{"b":1,"b":2}}',b'['*1600+b']'*1600):
                path.write_bytes(raw)
                with self.subTest(name=name,kind=raw[:8]),self.assertRaises(AppError): listing.verify_listing(root)
            path.write_bytes(before)

    def test_pretty_json_allowed_but_same_size_mtime_change_during_finish_detected(self):
        root = self.root/'list'
        inputs,_ = stored(root)
        path = root/'inputs.json'
        path.write_bytes(json.dumps(inputs.report(),indent=2).encode())
        self.assertEqual(listing.verify_listing(root)['candidate_count'],1)
        finish = listing.ListingFiles.finish
        def changed(files):
            stamp = path.stat()
            path.write_bytes(path.read_bytes().replace(b'test-lamp',b'edit-lamp'))
            os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
            return finish(files)
        with patch.object(listing.ListingFiles,'finish',changed),self.assertRaises(AppError) as err:
            listing.verify_listing(root)
        self.assertEqual(err.exception.code,'INPUT_CHANGED')

    def test_membership_change_and_file_identity_change_during_read_rejected(self):
        root = self.root/'list'
        stored(root)
        finish = listing.ListingFiles.finish
        def added(files):
            (root/'extra.json').write_bytes(b'{}')
            return finish(files)
        with patch.object(listing.ListingFiles,'finish',added),self.assertRaises(AppError): listing.verify_listing(root)
        (root/'extra.json').unlink()
        identity = listing._identity
        calls = 0
        def changed(path):
            nonlocal calls
            value = identity(path)
            if path.name=='inputs.json':
                calls += 1
                if calls==2: return (*value[:3],value[3]+1)
            return value
        with patch.object(listing,'_identity',side_effect=changed),self.assertRaises(AppError) as err:
            listing.verify_listing(root)
        self.assertEqual(err.exception.code,'INPUT_CHANGED')

    def test_read_error_and_cancellation_do_not_modify_listing_and_allow_retry(self):
        root = self.root/'list'
        stored(root)
        before = {p.name:p.read_bytes() for p in root.iterdir()}
        with patch('pathlib.Path.open',side_effect=OSError('private')),self.assertRaises(AppError) as err:
            listing.verify_listing(root)
        self.assertEqual(err.exception.code,'IO_ERROR')
        with patch.object(listing,'_recorded_candidates',side_effect=KeyboardInterrupt()),self.assertRaises(KeyboardInterrupt):
            listing.verify_listing(root)
        self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})
        self.assertEqual(listing.verify_listing(root)['status'],'verified')

    def test_source_single_analysis_drift_source_change_and_cleanup_failure(self):
        source = self.root/'own.mp4'
        source.write_bytes(b'own source mock')
        root = self.root/'list'
        inputs,catalog = stored(root,fixture(listing._sha256(source)),('lamp',))
        analysis = ReplayAnalysis((),None)
        with patch.object(listing,'analyze_inputs',return_value=analysis) as analyze, \
             patch.object(listing,'inspect_source',return_value=(None,None,object())), \
             patch.object(listing,'make_catalog',return_value=catalog):
            result = listing.verify_listing(root,source=source,toolchain=object())
            self.assertTrue(result['signal_context_verified'])
            analyze.assert_called_once()
            with patch.object(listing,'release_pixels',side_effect=AppError('CLEANUP_FAILED','safe')),self.assertRaises(AppError):
                listing.verify_listing(root,source=source,toolchain=object())
            with patch.object(listing,'analyze_inputs',return_value=replace(analysis,lamp_resets=((inputs.rows[0].frame,'fake'),))),self.assertRaises(AppError):
                listing.verify_listing(root,source=source,toolchain=object())
            def change(*args):
                source.write_bytes(b'edited own source')
                return catalog
            with patch.object(listing,'make_catalog',side_effect=change),self.assertRaises(AppError) as err:
                listing.verify_listing(root,source=source,toolchain=object())
            self.assertEqual(err.exception.code,'INPUT_CHANGED')

    def test_stdout_redaction_source_cli_installed_errors_and_cancel(self):
        root = self.root/'list'
        inputs,_ = stored(root,diagnostics=('lane','lamp'))
        args = ['--listing',str(root),'--require-diagnostics','lane','lamp']
        with patch('sys.stdout',new_callable=io.StringIO) as out:
            self.assertEqual(source_main(args),0)
            for private in (str(root),inputs.sha256,inputs.rows[0].frame.source_id,'selected_identity','model_id'):
                self.assertNotIn(private,out.getvalue())
        code,out,err = invoke(['events','verify-list',*args])
        self.assertEqual((code,err),(0,''))
        self.assertEqual(json.loads(out)['files_checked'],4)
        code,out,err = invoke(['events','verify-list',*args,'--ffmpeg-dir','private'])
        self.assertEqual((code,out),(2,''))
        self.assertEqual(json.loads(err)['code'],'INVALID_ARGUMENT')
        for failure,source_code,installed_code in ((AppError('LISTING_INVALID','private'),1,2),(KeyboardInterrupt(),130,130)):
            with patch('scripts.verify_listing.verify_listing',side_effect=failure),patch('sys.stdout',new_callable=io.StringIO) as out:
                self.assertEqual(source_main(args),source_code)
                self.assertNotIn('private',out.getvalue())
            with patch.object(listing,'verify_listing',side_effect=failure):
                self.assertEqual(invoke(['events','verify-list',*args])[0],installed_code)

    def test_fresh_process_readonly_audit_imports_no_numpy_cv2_torch(self):
        root = self.root/'list'
        stored(root,with_lamp_evidence(scenario()),('lane','lamp'))
        result = subprocess.run([sys.executable,'-B','-c',
            'import sys; from cheating_racer_detect.events.listing import verify_listing; '
            'assert verify_listing(sys.argv[1])["files_checked"]==4; '
            'assert not any(n in sys.modules for n in ("numpy","cv2","torch"))',str(root)],
            capture_output=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr.decode())


if __name__=='__main__':
    unittest.main()
