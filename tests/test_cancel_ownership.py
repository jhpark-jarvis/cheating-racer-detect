"""Fail-closed ownership guards; fake handles never operate on real processes."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from tests.cancel_driver import observe_owned_children


class Ownership(unittest.TestCase):
    def run_observer(self,entries,times,images):
        api=Mock()
        api.OpenProcess.side_effect=lambda access,inherit,pid:pid
        tracked={}
        allowed={Path('/owned/python.exe').resolve(),Path('/owned/ffmpeg.exe').resolve()}
        with patch('tests.cancel_driver.process_entries',return_value=entries),\
             patch('tests.cancel_driver.creation_time',side_effect=lambda api,h:times[h]),\
             patch('tests.cancel_driver.process_image',side_effect=lambda api,h:images[h]):
            observe_owned_children(api,SimpleNamespace(pid=100,_handle=1),tracked,allowed)
        api.TerminateProcess.assert_not_called()
        return api,tracked

    def test_old_orphan_with_reused_root_pid_never_owned(self):
        api,tracked=self.run_observer([(200,100,'python.exe')],{1:100,200:50},
                                     {200:Path('/owned/python.exe').resolve()})
        self.assertEqual(tracked,{})
        api.CloseHandle.assert_called_once_with(200)

    def test_unrelated_service_never_even_opened(self):
        api,tracked=self.run_observer([(200,100,'OneDrive.Sync.Service.exe')],{1:100},{})
        self.assertEqual(tracked,{})
        api.OpenProcess.assert_not_called()

    def test_grandchild_requires_validated_immediate_parent_creation(self):
        api,tracked=self.run_observer([(300,200,'ffmpeg.exe'),(200,100,'python.exe'),(400,200,'ffmpeg.exe')],
            {1:100,200:110,300:120,400:105},{200:Path('/owned/python.exe').resolve(),
             300:Path('/owned/ffmpeg.exe').resolve(),400:Path('/owned/ffmpeg.exe').resolve()})
        self.assertEqual(set(tracked),{200,300})
        self.assertIn(unittest.mock.call(400),api.CloseHandle.call_args_list)

    def test_same_name_foreign_image_rejected(self):
        api,tracked=self.run_observer([(200,100,'python.exe')],{1:100,200:110},
                                     {200:Path('/foreign/python.exe').resolve()})
        self.assertEqual(tracked,{})
        api.CloseHandle.assert_called_once_with(200)

    def test_unavailable_creation_fails_closed_and_closes_handle(self):
        api,tracked=self.run_observer([(200,100,'python.exe')],{1:100,200:110},
                                     {200:Path('/owned/python.exe').resolve()})
        # Re-run a separate observation with a failed child identity query.
        with patch('tests.cancel_driver.process_entries',return_value=[(200,100,'python.exe')]),\
             patch('tests.cancel_driver.creation_time',side_effect=[100,OSError('denied')]):
            tracked={}
            observe_owned_children(api,SimpleNamespace(pid=100,_handle=1),tracked,{Path('/owned/python.exe')})
        self.assertEqual(tracked,{})
        api.CloseHandle.assert_called_with(200)

    def test_unavailable_image_is_closed_without_becoming_cleanup_target(self):
        api=Mock()
        api.OpenProcess.return_value=200
        tracked={}
        with patch('tests.cancel_driver.process_entries',return_value=[(200,100,'python.exe')]),\
             patch('tests.cancel_driver.creation_time',side_effect=[100,110]),\
             patch('tests.cancel_driver.process_image',side_effect=OSError('denied')):
            observe_owned_children(api,SimpleNamespace(pid=100,_handle=1),tracked,{Path('/owned/python.exe')})
        self.assertEqual(tracked,{})
        api.CloseHandle.assert_called_once_with(200)
        api.TerminateProcess.assert_not_called()

    def test_missing_root_identity_never_opens_any_child(self):
        api=Mock()
        with patch('tests.cancel_driver.process_entries',return_value=[(200,100,'python.exe')]),\
             patch('tests.cancel_driver.creation_time',side_effect=OSError('root denied')),self.assertRaises(OSError):
            observe_owned_children(api,SimpleNamespace(pid=100,_handle=1),{},{Path('/owned/python.exe')})
        api.OpenProcess.assert_not_called()
        api.TerminateProcess.assert_not_called()

    def test_broken_or_cyclic_ancestry_never_promoted(self):
        api,tracked=self.run_observer([(200,300,'python.exe'),(300,200,'ffmpeg.exe')],{1:100},{})
        self.assertEqual(tracked,{})
        api.OpenProcess.assert_not_called()

    def test_denied_handle_not_owned_and_never_closed_or_terminated(self):
        api=Mock()
        api.OpenProcess.return_value=0
        tracked={}
        with patch('tests.cancel_driver.process_entries',return_value=[(200,100,'python.exe')]),\
             patch('tests.cancel_driver.creation_time',return_value=100):
            observe_owned_children(api,SimpleNamespace(pid=100,_handle=1),tracked,{Path('/owned/python.exe')})
        self.assertEqual(tracked,{})
        api.CloseHandle.assert_not_called()
        api.TerminateProcess.assert_not_called()

    def test_pinned_handle_reused_without_reopening_pid(self):
        api=Mock()
        tracked={200:(7,'python.exe')}
        allowed={Path('/owned/python.exe').resolve()}
        with patch('tests.cancel_driver.process_entries',return_value=[(200,100,'python.exe')]),\
             patch('tests.cancel_driver.creation_time',side_effect=[100,110]),\
             patch('tests.cancel_driver.process_image',return_value=next(iter(allowed))):
            observe_owned_children(api,SimpleNamespace(pid=100,_handle=1),tracked,allowed)
        self.assertEqual(tracked,{200:(7,'python.exe')})
        api.OpenProcess.assert_not_called()
        api.CloseHandle.assert_not_called()
        api.TerminateProcess.assert_not_called()


if __name__=='__main__': unittest.main()
