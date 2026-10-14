"""Exact own-source archive audit, not a legal clearance or secret scanner."""

import base64
import configparser
import csv
from email.parser import BytesParser
import hashlib
import io
from pathlib import PurePosixPath
import stat
import tarfile
import tomllib
import zipfile

from cheating_racer_detect.paths import local_path

LIMIT = 16*1024*1024


def require(condition):
    if not condition:
        raise ValueError('Distribution file or metadata boundary mismatch')


def sources(root):
    root = local_path(root)
    files = {}
    for path in sorted((root/'cheating_racer_detect').rglob('*.py')):
        path = local_path(path)
        require(path.is_file() and path.stat().st_size <= 1_048_576)
        files[path.relative_to(root).as_posix()] = path.read_bytes()
    require(files and sum(map(len, files.values())) <= LIMIT)
    return files


def snapshot(root, destination):
    """Copy explicit sources only; never build inside the original checkout."""
    require(not destination.exists())
    files = {**sources(root), **{name: local_path(root/name).read_bytes()
                               for name in ('pyproject.toml', 'README.md', 'MANIFEST.in')}}
    destination.mkdir()
    for name, data in files.items():
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    return files


def project(root):
    value = tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']
    return value, value['name'].replace('-', '_')


def safe_name(name):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and '\\' not in name and ':' not in name
            and '..' not in path.parts and str(path) == name)


def check_wheel(path, root):
    specification, normalized = project(root)
    prefix = normalized+'-'+specification['version']+'.dist-info/'
    info_names = {'METADATA', 'WHEEL', 'RECORD', 'entry_points.txt', 'top_level.txt'}
    expected = sources(root)
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(0 < len(entries) <= 1024 and len({i.filename for i in entries}) == len(entries))
        require(sum(i.file_size for i in entries) <= LIMIT)
        for item in entries:
            safe_name(item.filename)
            require(not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16) and item.file_size <= 1_048_576)
        data = {i.filename: archive.read(i) for i in entries}
    require(set(data) == set(expected) | {prefix+n for n in info_names})
    require(all(data[name] == content for name, content in expected.items()))
    metadata = BytesParser().parsebytes(data[prefix+'METADATA'])
    require(metadata['Name'] == specification['name'] and metadata['Version'] == specification['version'])
    require(set(metadata['Requires-Python'].split(',')) == set(specification['requires-python'].split(',')))
    require(metadata.get_all('Provides-Extra') == ['tracking'])
    dependencies = metadata.get_all('Requires-Dist', [])
    require(len(dependencies) == 2 and all('; extra == "tracking"' in d for d in dependencies))
    require({d.split(';')[0].replace(' ', '') for d in dependencies} == set(specification['optional-dependencies']['tracking']))
    wheel = BytesParser().parsebytes(data[prefix+'WHEEL'])
    require(wheel['Root-Is-Purelib'] == 'true' and wheel.get_all('Tag') == ['py3-none-any'])
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(data[prefix+'entry_points.txt'].decode('utf-8'))
    require(config.sections() == ['console_scripts'] and dict(config['console_scripts']) == {'crd': 'cheating_racer_detect.cli:main'})
    require(data[prefix+'top_level.txt'].decode().strip() == 'cheating_racer_detect')
    records = list(csv.reader(io.StringIO(data[prefix+'RECORD'].decode())))
    require(len(records) == len(data) and all(len(r) == 3 for r in records) and len({r[0] for r in records}) == len(records))
    require({r[0] for r in records} == set(data))
    for name, digest, size in records:
        if name == prefix+'RECORD':
            require(digest == size == '')
        else:
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data[name]).digest()).rstrip(b'=').decode()
            require(digest == 'sha256='+encoded and size == str(len(data[name])))
    return {'python_files': len(expected), 'wheel_files': len(data), 'base_dependencies': 0}


def check_sdist(path, root):
    specification, normalized = project(root)
    prefix = normalized+'-'+specification['version']+'/'
    egg = normalized+'.egg-info/'
    generated = {'PKG-INFO', 'setup.cfg', *{egg+n for n in
                 ('PKG-INFO', 'SOURCES.txt', 'dependency_links.txt', 'entry_points.txt', 'requires.txt', 'top_level.txt')}}
    expected = {**sources(root), **{name: (root/name).read_bytes() for name in ('pyproject.toml', 'README.md', 'MANIFEST.in')}}
    allowed = set(expected) | generated
    with tarfile.open(path, 'r:gz') as archive:
        entries = archive.getmembers()
        require(0 < len(entries) <= 1024 and len({i.name for i in entries}) == len(entries))
        require(sum(i.size for i in entries) <= LIMIT)
        data = {}
        for item in entries:
            safe_name(item.name)
            require(item.name.startswith(prefix) or item.isdir() and item.name == prefix[:-1])
            relative = item.name[len(prefix):]
            if item.isdir():
                require(not relative or any(n.startswith(relative+'/') for n in allowed))
                continue
            require(item.isfile() and item.size <= 1_048_576 and relative in allowed)
            data[relative] = archive.extractfile(item).read()
    require(set(data) == allowed and all(data[name] == content for name, content in expected.items()))
    return {'sdist_files': len(data), 'sdist_python_files': len(sources(root)), 'prefix': prefix[:-1]}
