"""Archive allowlists/RECORD and owned smoke-resource helpers, stdlib only."""

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from scripts.package_artifacts import check_sdist, check_wheel, safe_name, snapshot, sources
from scripts.check_install import child, environment

ROOT = Path(__file__).resolve().parents[1]
INFO = 'cheating_racer_detect-0.1.0.dist-info/'
EGG = 'cheating_racer_detect.egg-info/'


def wheel_data(root):
    return {**sources(root), INFO+'METADATA': b'Metadata-Version: 2.4\nName: cheating-racer-detect\nVersion: 0.1.0\nRequires-Python: >=3.13,<3.14\nProvides-Extra: tracking\nRequires-Dist: numpy==2.2.6; extra == "tracking"\nRequires-Dist: opencv-python==4.11.0.86; extra == "tracking"\n',
            INFO+'WHEEL': b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
            INFO+'entry_points.txt': b'[console_scripts]\ncrd = cheating_racer_detect.cli:main\n',
            INFO+'top_level.txt': b'cheating_racer_detect\n'}


def write_wheel(path, data, *, tamper_record=False, symlink=False, duplicate=False):
    records = io.StringIO()
    writer = csv.writer(records)
    for name, value in data.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b'=').decode()
        writer.writerow([name, 'sha256='+digest, str(len(value))])
    writer.writerow([INFO+'RECORD', '', ''])
    data = {**data, INFO+'RECORD': records.getvalue().encode()}
    if tamper_record:
        data[INFO+'RECORD'] = data[INFO+'RECORD'].replace(b'sha256=', b'sha512=', 1)
    with zipfile.ZipFile(path, 'w') as archive:
        for n, (name, value) in enumerate(data.items()):
            item = zipfile.ZipInfo(name)
            if symlink and n == 0:
                item.create_system = 3
                item.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(item, value)
            if duplicate and n == 0:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    archive.writestr(item, value)


def write_sdist(path, root, extra=None, symlink=False):
    data = {**sources(root), **{n: (root/n).read_bytes() for n in ('pyproject.toml', 'README.md', 'MANIFEST.in')},
            'PKG-INFO': b'metadata', 'setup.cfg': b'configuration',
            **{EGG+n: b'metadata' for n in ('PKG-INFO', 'SOURCES.txt', 'dependency_links.txt', 'entry_points.txt', 'requires.txt', 'top_level.txt')}}
    if extra:
        data.update(extra)
    with tarfile.open(path, 'w:gz') as archive:
        for n, (name, value) in enumerate(data.items()):
            item = tarfile.TarInfo('cheating_racer_detect-0.1.0/'+name)
            if symlink and n == 0:
                item.type, item.linkname = tarfile.SYMTYPE, '../private'
                archive.addfile(item)
            else:
                item.size = len(value)
                archive.addfile(item, io.BytesIO(value))


class Distribution(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='crd-package-unit-')
        self.root = Path(self.temporary.name)
        self.source = self.root/'snapshot'
        self.original = snapshot(ROOT, self.source)
        self.addCleanup(self.temporary.cleanup)

    def test_snapshot_excludes_runtime_media_docs_and_is_no_replace(self):
        self.assertTrue(all(name.startswith('cheating_racer_detect/') or name in ('pyproject.toml','README.md','MANIFEST.in') for name in self.original))
        self.assertFalse((self.source/'AGENTS.md').exists())
        for name in ('.tools', 'tests', 'docs', '.git'):
            self.assertFalse((self.source/name).exists())
        with self.assertRaises(ValueError):
            snapshot(ROOT, self.source)

    def test_exact_wheel_source_metadata_and_record(self):
        path = self.root/'test.whl'
        write_wheel(path, wheel_data(self.source))
        result = check_wheel(path, self.source)
        self.assertEqual(result['base_dependencies'], 0)
        self.assertEqual(result['python_files'], len(sources(self.source)))

    def test_unknown_file_duplicate_symlink_and_record_hash_rejected(self):
        path = self.root/'test.whl'
        for options, extra in (({}, {'docs/private.md': b'private'}), ({'tamper_record': True}, {}),
                               ({'symlink': True}, {}), ({'duplicate': True}, {}), ({}, {'../escape': b'private'})):
            with self.subTest(options=options):
                write_wheel(path, {**wheel_data(self.source), **extra}, **options)
                with self.assertRaises(ValueError):
                    check_wheel(path, self.source)

    def test_source_drift_entrypoint_base_dependency_and_platform_rejected(self):
        path = self.root/'test.whl'
        values = [(INFO+'entry_points.txt', b'[console_scripts]\ncrd = private.model:main\n'),
                  (INFO+'METADATA', wheel_data(self.source)[INFO+'METADATA']+b'Requires-Dist: torch\n'),
                  (INFO+'WHEEL', b'Root-Is-Purelib: false\nTag: cp313-win_amd64\n'),
                  ('cheating_racer_detect/__init__.py', b'changed')]
        for name, value in values:
            write_wheel(path, {**wheel_data(self.source), name: value})
            with self.assertRaises(ValueError):
                check_wheel(path, self.source)

    def test_sdist_exact_sources_private_media_escape_and_links(self):
        path = self.root/'test.tar.gz'
        write_sdist(path, self.source)
        self.assertEqual(check_sdist(path, self.source)['sdist_python_files'], len(sources(self.source)))
        for extra in ({'AGENTS.md': b'private'}, {'data/input.mp4': b'video'}, {'../escape': b'private'}):
            write_sdist(path, self.source, extra)
            with self.assertRaises(ValueError):
                check_sdist(path, self.source)
        write_sdist(path, self.source, symlink=True)
        with self.assertRaises(ValueError):
            check_sdist(path, self.source)

    def test_names_are_lexical_relative_only(self):
        for name in ('../x', '/x', 'x/../y', 'x\\y', 'C:x', './x', 'x//y', ''):
            with self.assertRaises(ValueError):
                safe_name(name)
        safe_name('cheating_racer_detect/events/cli.py')

    def test_child_error_is_sanitized_and_owned_process_reaped_on_cancel(self):
        with self.assertRaises(RuntimeError) as caught:
            child([sys.executable, '-c', 'import sys;sys.stderr.write("PRIVATE_MARKER");sys.exit(7)'], self.root)
        self.assertNotIn('PRIVATE_MARKER', str(caught.exception))
        process = Mock()
        process.communicate.side_effect = [KeyboardInterrupt(), (b'', b'')]
        process.poll.return_value = None
        with patch('scripts.check_install.subprocess.Popen', return_value=process), self.assertRaises(KeyboardInterrupt):
            child(['owned'], self.root)
        process.kill.assert_called_once()
        self.assertEqual(process.communicate.call_count, 2)

    def test_child_environment_cannot_inherit_pythonpath_or_online_policy(self):
        with patch.dict('os.environ', {'PYTHONPATH': 'PRIVATE', 'PYTHONHOME': 'PRIVATE', 'UV_OFFLINE': '0', 'PIP_NO_INDEX': '0'}):
            result = environment()
        self.assertNotIn('PYTHONPATH', result)
        self.assertNotIn('PYTHONHOME', result)
        self.assertEqual(result['UV_OFFLINE'], '1')
        self.assertEqual(result['PIP_NO_INDEX'], '1')


if __name__ == '__main__':
    unittest.main()
